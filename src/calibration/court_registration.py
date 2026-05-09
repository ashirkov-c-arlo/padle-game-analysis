from __future__ import annotations

import cv2
import numpy as np
from loguru import logger

from src.calibration.frame_sampler import sample_stable_frames
from src.calibration.line_detection import detect_lines_deeplsd, detect_lines_hough
from src.calibration.line_filtering import cluster_lines, filter_court_lines
from src.calibration.template_fitting import (
    fit_homography,
    get_court_template_lines,
    match_court_line_grid,
    match_lines_to_template,
)
from src.schemas import CourtGeometry2D, CourtRegistration2D


def register_court(video_path: str, config: dict) -> CourtRegistration2D:
    """Perform 2D court registration from video.

    Steps:
    1. Sample stable frames from video at config interval
    2. Run line detection (DeepLSD primary, LSD/Hough fallback)
    3. Filter and cluster lines
    4. Match to known 2D court template
    5. Estimate floor homography via RANSAC
    6. Validate quality
    7. Return CourtRegistration2D with mode="floor_homography" or "pixel_only" on failure
    """
    cal_config = config.get("calibration", {})
    interval_s = cal_config.get("frame_sample_interval_s", 2.0)
    max_error = cal_config.get("max_reprojection_error_px", 10.0)
    min_frames = cal_config.get("min_frames_for_registration", 5)

    geometry = CourtGeometry2D()

    # Step 1: Sample frames
    logger.info("Sampling frames from video: {}", video_path)
    frames = sample_stable_frames(video_path, interval_s)

    if len(frames) < min_frames:
        logger.warning(
            "Only {} frames sampled (need {}), falling back to pixel_only",
            len(frames),
            min_frames,
        )
        return _pixel_only_result()

    # Step 2-6: Process each frame, pick the best registration
    best_result: CourtRegistration2D | None = None
    best_error = float("inf")

    for frame_idx, frame in frames:
        result = _process_single_frame(frame, frame_idx, geometry, max_error, config)
        if result is not None and result.reprojection_error_px is not None:
            if result.reprojection_error_px < best_error:
                best_error = result.reprojection_error_px
                best_result = result

    # Step 7: Return best result or fallback
    if best_result is not None and best_error <= max_error:
        logger.info(
            "Court registration successful: error={:.2f}px, confidence={:.2f}",
            best_error,
            best_result.confidence,
        )
        return best_result

    logger.warning(
        "No frame achieved < {:.1f}px error (best={:.2f}), falling back to pixel_only",
        max_error,
        best_error,
    )
    return _pixel_only_result()


def _process_single_frame(
    frame: np.ndarray,
    frame_idx: int,
    geometry: CourtGeometry2D,
    max_error: float,
    config: dict | None = None,
) -> CourtRegistration2D | None:
    """Process a single frame for court registration."""
    image_shape = frame.shape[:2]

    # Line detection
    method = (config or {}).get("calibration", {}).get("method", "deeplsd")
    if method == "hough":
        lines = detect_lines_hough(frame)
    else:
        lines = detect_lines_deeplsd(frame, config)
    if len(lines) == 0:
        logger.debug("Frame {}: no lines detected", frame_idx)
        return None

    # Filter lines
    filtered = filter_court_lines(lines, image_shape)
    floor_mask = _estimate_floor_mask(frame)
    filtered = _filter_lines_by_floor_color(frame, filtered, floor_mask)
    if len(filtered) < 4:
        logger.debug("Frame {}: too few lines after filtering ({})", frame_idx, len(filtered))
        return None

    # Cluster lines
    clusters = cluster_lines(filtered)
    all_clustered = np.vstack(
        [clusters["horizontal"], clusters["vertical"]]
    ) if len(clusters["horizontal"]) > 0 and len(clusters["vertical"]) > 0 else filtered

    if len(all_clustered) < 4:
        logger.debug("Frame {}: too few lines after clustering ({})", frame_idx, len(all_clustered))
        return None

    # Match to template
    image_points, court_points = match_court_line_grid(
        all_clustered, geometry, image_shape
    )
    if len(image_points) < 4:
        image_points, court_points = match_lines_to_template(
            all_clustered, geometry, image_shape, valid_mask=floor_mask
        )
    if len(image_points) < 4:
        logger.debug(
            "Frame {}: insufficient point correspondences ({})",
            frame_idx,
            len(image_points),
        )
        return None

    # Fit homography
    H_image_to_court, error, num_inliers = fit_homography(
        image_points,
        court_points,
        reprojection_threshold_px=max_error,
        return_inlier_count=True,
    )
    if H_image_to_court is None:
        logger.debug("Frame {}: homography estimation failed", frame_idx)
        return None

    # Compute inverse
    try:
        H_court_to_image = np.linalg.inv(H_image_to_court)
    except np.linalg.LinAlgError:
        logger.debug("Frame {}: homography not invertible", frame_idx)
        return None

    # Validate with measured template-line reprojection. This is a separate
    # geometric check and should never reduce the true point reprojection error.
    line_error = _validate_homography(
        H_court_to_image,
        geometry,
        image_shape,
        filtered,
    )
    if line_error is not None:
        if not np.isfinite(line_error):
            logger.debug("Frame {}: template line validation failed", frame_idx)
            return None
        error = max(error, line_error)

    # Compute confidence (higher is better, capped at 1.0)
    confidence = _compute_confidence(error, max_error, num_inliers)

    logger.debug(
        "Frame {}: error={:.2f}px, line_error={}, inliers={}, confidence={:.2f}",
        frame_idx,
        error,
        f"{line_error:.2f}px" if line_error is not None and np.isfinite(line_error) else line_error,
        num_inliers,
        confidence,
    )

    return CourtRegistration2D(
        mode="floor_homography",
        homography_image_to_court=H_image_to_court.tolist(),
        homography_court_to_image=H_court_to_image.tolist(),
        reprojection_error_px=error,
        num_inliers=num_inliers,
        confidence=confidence,
    )


def _validate_homography(
    H_court_to_image: np.ndarray,
    geometry: CourtGeometry2D,
    image_shape: tuple,
    detected_lines: np.ndarray,
) -> float | None:
    """Validate homography by comparing projected template lines to detected lines."""
    if len(detected_lines) == 0:
        return None

    template_lines = get_court_template_lines(geometry)
    h, w = image_shape[:2]

    errors = []
    visible_template_lines = 0
    for line in template_lines:
        pts_court = np.array([[line[0], line[1]], [line[2], line[3]]], dtype=np.float64)
        ones = np.ones((2, 1))
        pts_h = np.hstack([pts_court, ones])
        projected = (H_court_to_image @ pts_h.T).T
        projected = projected[:, :2] / projected[:, 2:3]

        sample_points = _sample_visible_segment_points(projected[0], projected[1], h, w)
        if len(sample_points) == 0:
            continue

        visible_template_lines += 1
        distances = [
            min(_point_to_segment_distance(pt, detected) for detected in detected_lines)
            for pt in sample_points
        ]
        line_error = float(np.mean(distances))
        if line_error < 50.0:
            errors.append(line_error)

    if not errors:
        return float("inf") if visible_template_lines > 0 else None
    return float(np.mean(errors))


def _filter_lines_by_floor_color(
    frame: np.ndarray,
    lines: np.ndarray,
    floor_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Keep lines whose midpoints fall on the dominant saturated court floor."""
    if len(lines) == 0 or len(frame.shape) != 3:
        return lines

    mask = floor_mask if floor_mask is not None else _estimate_floor_mask(frame)
    if mask is None:
        return lines

    h, w = mask.shape[:2]
    midpoints = np.column_stack([
        (lines[:, 0] + lines[:, 2]) / 2,
        (lines[:, 1] + lines[:, 3]) / 2,
    ])
    midpoints = np.round(midpoints).astype(int)

    keep = []
    for x, y in midpoints:
        keep.append(0 <= x < w and 0 <= y < h and mask[y, x] > 0)
    keep_mask = np.array(keep, dtype=bool)
    floor_lines = lines[keep_mask]

    if len(floor_lines) < 4:
        return lines

    return floor_lines


def _estimate_floor_mask(frame: np.ndarray) -> np.ndarray | None:
    """Estimate the dominant saturated floor component in the central/lower image."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    mask = ((saturation > 80) & (value > 80)).astype(np.uint8) * 255
    h, w = mask.shape[:2]
    mask[: int(h * 0.25), :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    min_area = h * w * 0.08

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        cx, cy = centroids[label]
        if not (w * 0.2 <= cx <= w * 0.8 and h * 0.35 <= cy <= h * 0.9):
            continue

        candidates.append((area, label))

    if not candidates:
        return None

    _, best_label = max(candidates)
    floor_mask = (labels == best_label).astype(np.uint8) * 255
    dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    return cv2.dilate(floor_mask, dilation, iterations=1)


def _sample_visible_segment_points(
    p1: np.ndarray,
    p2: np.ndarray,
    image_h: int,
    image_w: int,
    num_samples: int = 20,
) -> np.ndarray:
    """Sample projected line points that are visible in or near the image."""
    ts = np.linspace(0.0, 1.0, num_samples)
    points = p1[None, :] + ts[:, None] * (p2 - p1)[None, :]

    margin_x = image_w * 0.05
    margin_y = image_h * 0.05
    mask = (
        (points[:, 0] >= -margin_x)
        & (points[:, 0] <= image_w + margin_x)
        & (points[:, 1] >= -margin_y)
        & (points[:, 1] <= image_h + margin_y)
    )
    return points[mask]


def _point_to_segment_distance(point: np.ndarray, segment: np.ndarray) -> float:
    """Distance from a point to a finite line segment in image coordinates."""
    px, py = point
    x1, y1, x2, y2 = segment
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return float(np.hypot(px - x1, py - y1))

    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    t = float(np.clip(t, 0.0, 1.0))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return float(np.hypot(px - closest_x, py - closest_y))


def _compute_confidence(error: float, max_error: float, num_inliers: int) -> float:
    """Compute confidence score from reprojection error and inlier count."""
    # Error component: 1.0 at 0 error, 0.0 at max_error
    error_score = max(0.0, 1.0 - error / max_error)

    # Inlier component: bonus for more inliers (4 minimum, 8+ is excellent)
    inlier_score = min(1.0, (num_inliers - 4) / 6.0) if num_inliers >= 4 else 0.0

    # Combined: weight error more heavily
    confidence = 0.7 * error_score + 0.3 * inlier_score
    return max(0.0, min(1.0, confidence))


def _pixel_only_result() -> CourtRegistration2D:
    """Return a pixel-only fallback result."""
    return CourtRegistration2D(
        mode="pixel_only",
        homography_image_to_court=None,
        homography_court_to_image=None,
        reprojection_error_px=None,
        num_inliers=None,
        confidence=0.0,
    )
