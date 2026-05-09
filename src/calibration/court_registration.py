from __future__ import annotations

import numpy as np
from loguru import logger

from src.calibration.frame_sampler import sample_stable_frames
from src.calibration.line_detection import detect_lines_deeplsd
from src.calibration.line_filtering import cluster_lines, filter_court_lines
from src.calibration.template_fitting import (
    fit_homography,
    get_court_template_lines,
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
        result = _process_single_frame(frame, frame_idx, geometry, max_error)
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
) -> CourtRegistration2D | None:
    """Process a single frame for court registration."""
    image_shape = frame.shape[:2]

    # Line detection
    lines = detect_lines_deeplsd(frame)
    if len(lines) == 0:
        logger.debug("Frame {}: no lines detected", frame_idx)
        return None

    # Filter lines
    filtered = filter_court_lines(lines, image_shape)
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
    image_points, court_points = match_lines_to_template(
        all_clustered, geometry, image_shape
    )
    if len(image_points) < 4:
        logger.debug(
            "Frame {}: insufficient point correspondences ({})",
            frame_idx,
            len(image_points),
        )
        return None

    # Fit homography
    H_image_to_court, error = fit_homography(image_points, court_points)
    if H_image_to_court is None:
        logger.debug("Frame {}: homography estimation failed", frame_idx)
        return None

    # Validate with template line reprojection
    validation_error = _validate_homography(H_image_to_court, geometry, image_shape)
    if validation_error is not None:
        error = (error + validation_error) / 2

    # Compute inverse
    try:
        H_court_to_image = np.linalg.inv(H_image_to_court)
    except np.linalg.LinAlgError:
        logger.debug("Frame {}: homography not invertible", frame_idx)
        return None

    # Compute confidence (higher is better, capped at 1.0)
    confidence = _compute_confidence(error, max_error, len(image_points))

    num_inliers = len(image_points)

    logger.debug(
        "Frame {}: error={:.2f}px, inliers={}, confidence={:.2f}",
        frame_idx,
        error,
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
    H: np.ndarray, geometry: CourtGeometry2D, image_shape: tuple
) -> float | None:
    """Validate homography by reprojecting template lines and measuring consistency."""
    template_lines = get_court_template_lines(geometry)
    h, w = image_shape[:2]

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None

    # Project template line endpoints to image space
    errors = []
    for line in template_lines:
        pts_court = np.array([[line[0], line[1]], [line[2], line[3]]], dtype=np.float64)
        ones = np.ones((2, 1))
        pts_h = np.hstack([pts_court, ones])
        projected = (H_inv @ pts_h.T).T
        projected = projected[:, :2] / projected[:, 2:3]

        # Check if projected points are within reasonable bounds
        for pt in projected:
            if -w * 0.5 <= pt[0] <= w * 1.5 and -h * 0.5 <= pt[1] <= h * 1.5:
                # Point is in reasonable range
                pass
            else:
                errors.append(50.0)  # Penalty for out-of-bounds projection

    # If most template lines project within bounds, the homography is reasonable
    if not errors:
        return 0.0
    return float(np.mean(errors))


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
