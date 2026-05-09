from __future__ import annotations

import itertools

import cv2
import numpy as np
from loguru import logger

from src.schemas import CourtGeometry2D


def get_court_template_points(geometry: CourtGeometry2D) -> np.ndarray:
    """Return Nx2 array of court keypoints in metric coordinates.

    Standard padel court:
    - x: 0..10 (left to right)
    - y: 0..20 (near baseline to far baseline)
    - Net at y=10
    """
    w = geometry.width_m
    length = geometry.length_m
    net_y = geometry.net_y_m
    service_offset = geometry.service_line_offset_from_net_m

    near_service_y = net_y - service_offset
    far_service_y = net_y + service_offset
    center_x = w / 2

    # Key intersection points on the court
    points = np.array(
        [
            # Near baseline
            [0.0, 0.0],
            [center_x, 0.0],
            [w, 0.0],
            # Near service line
            [0.0, near_service_y],
            [center_x, near_service_y],
            [w, near_service_y],
            # Net line
            [0.0, net_y],
            [center_x, net_y],
            [w, net_y],
            # Far service line
            [0.0, far_service_y],
            [center_x, far_service_y],
            [w, far_service_y],
            # Far baseline
            [0.0, length],
            [center_x, length],
            [w, length],
        ],
        dtype=np.float64,
    )
    return points


def get_court_template_lines(geometry: CourtGeometry2D) -> np.ndarray:
    """Return Mx4 array of court line segments in metric coordinates (x1, y1, x2, y2).

    Used for reprojection error validation.
    """
    w = geometry.width_m
    length = geometry.length_m
    net_y = geometry.net_y_m
    service_offset = geometry.service_line_offset_from_net_m

    near_service_y = net_y - service_offset
    far_service_y = net_y + service_offset
    center_x = w / 2

    lines = np.array(
        [
            # Horizontal lines
            [0, 0, w, 0],  # near baseline
            [0, near_service_y, w, near_service_y],  # near service line
            [0, net_y, w, net_y],  # net
            [0, far_service_y, w, far_service_y],  # far service line
            [0, length, w, length],  # far baseline
            # Vertical lines
            [0, 0, 0, length],  # left sideline
            [w, 0, w, length],  # right sideline
            # Center service lines
            [center_x, 0, center_x, near_service_y],  # near center
            [center_x, far_service_y, center_x, length],  # far center
        ],
        dtype=np.float64,
    )
    return lines


def fit_homography(
    image_points: np.ndarray,
    court_points: np.ndarray,
    reprojection_threshold_px: float = 10.0,
    return_inlier_count: bool = False,
) -> tuple[np.ndarray | None, float] | tuple[np.ndarray | None, float, int]:
    """RANSAC homography estimation.

    Args:
        image_points: Nx2 array of points in image coordinates.
        court_points: Nx2 array of corresponding court coordinates.
        reprojection_threshold_px: RANSAC threshold measured in image pixels.
        return_inlier_count: Include the number of RANSAC inliers in the result.

    Returns:
        (H_image_to_court, reprojection_error_px) or (None, inf) on failure.
    """
    if len(image_points) < 4 or len(court_points) < 4:
        if return_inlier_count:
            return None, float("inf"), 0
        return None, float("inf")

    H_court_to_image, mask = cv2.findHomography(
        court_points.astype(np.float64),
        image_points.astype(np.float64),
        cv2.RANSAC,
        ransacReprojThreshold=reprojection_threshold_px,
        maxIters=2000,
        confidence=0.995,
    )

    if H_court_to_image is None or mask is None:
        if return_inlier_count:
            return None, float("inf"), 0
        return None, float("inf")

    # Compute reprojection error on inliers
    inlier_mask = mask.ravel().astype(bool)
    num_inliers = int(inlier_mask.sum())

    if num_inliers < 4:
        if return_inlier_count:
            return None, float("inf"), num_inliers
        return None, float("inf")

    court_inliers = court_points[inlier_mask]
    image_inliers = image_points[inlier_mask]

    # Project court -> image using the same pixel-space model RANSAC estimated.
    ones = np.ones((len(court_inliers), 1))
    court_h = np.hstack([court_inliers, ones])
    projected = (H_court_to_image @ court_h.T).T
    projected = projected[:, :2] / projected[:, 2:3]

    errors = np.sqrt(np.sum((projected - image_inliers) ** 2, axis=1))
    mean_error = float(errors.mean())

    try:
        H_image_to_court = np.linalg.inv(H_court_to_image)
    except np.linalg.LinAlgError:
        if return_inlier_count:
            return None, float("inf"), 0
        return None, float("inf")

    logger.debug(
        "Homography: {} inliers, mean reproj error = {:.2f} px",
        num_inliers,
        mean_error,
    )
    if return_inlier_count:
        return H_image_to_court, mean_error, num_inliers
    return H_image_to_court, mean_error


def match_court_line_grid(
    detected_lines: np.ndarray,
    geometry: CourtGeometry2D,
    image_shape: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    """Match court grid intersections using ordered court line families.

    This path is stricter than generic nearest-template matching: it first
    selects the five horizontal court lines using their expected projective
    spacing, then intersects them with the left/right sidelines and optional
    center service line.
    """
    if len(detected_lines) < 4:
        return np.empty((0, 2)), np.empty((0, 2))

    horizontal, vertical = _split_by_orientation(detected_lines)
    selected_horizontal = _select_horizontal_court_lines(
        horizontal, geometry, image_shape
    )
    if selected_horizontal is None:
        return np.empty((0, 2)), np.empty((0, 2))

    left_line, right_line, center_line = _select_vertical_court_lines(
        vertical, selected_horizontal, image_shape
    )
    if left_line is None or right_line is None:
        return np.empty((0, 2)), np.empty((0, 2))

    w = geometry.width_m
    center_x = w / 2
    court_y_values = _court_y_values_top_to_bottom(geometry)

    image_points = []
    court_points = []
    for court_y, horizontal_line in zip(court_y_values, selected_horizontal):
        for court_x, vertical_line in ((0.0, left_line), (w, right_line)):
            pt = _line_intersection(horizontal_line, vertical_line)
            if pt is not None:
                image_points.append(pt)
                court_points.append([court_x, court_y])

        if center_line is not None:
            pt = _line_intersection(horizontal_line, center_line)
            if pt is not None:
                image_points.append(pt)
                court_points.append([center_x, court_y])

    if len(image_points) < 4:
        return np.empty((0, 2)), np.empty((0, 2))

    return np.array(image_points, dtype=np.float64), np.array(court_points, dtype=np.float64)


def _split_by_orientation(lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    angles = np.abs(np.degrees(np.arctan2(dy, dx)))
    angles = np.where(angles > 90, 180 - angles, angles)
    return lines[angles < 45], lines[angles >= 45]


def _court_y_values_top_to_bottom(geometry: CourtGeometry2D) -> np.ndarray:
    near_service_y = geometry.net_y_m - geometry.service_line_offset_from_net_m
    far_service_y = geometry.net_y_m + geometry.service_line_offset_from_net_m
    return np.array(
        [
            geometry.length_m,
            far_service_y,
            geometry.net_y_m,
            near_service_y,
            0.0,
        ],
        dtype=np.float64,
    )


def _select_horizontal_court_lines(
    horizontal_lines: np.ndarray,
    geometry: CourtGeometry2D,
    image_shape: tuple,
) -> list[np.ndarray] | None:
    if len(horizontal_lines) < 5:
        return None

    h, w = image_shape[:2]
    image_center_x = w / 2
    min_length = 0.03 * np.hypot(h, w)

    candidates = []
    for line in horizontal_lines:
        length = _line_length(line)
        if length < min_length or _normalized_angle_deg(line) > 15:
            continue

        image_y = _line_y_at_x(line, image_center_x)
        if image_y < h * 0.2:
            continue

        candidates.append((image_y, length, line))

    if len(candidates) < 5:
        return None

    court_y_values = _court_y_values_top_to_bottom(geometry)
    best: tuple[float, list[np.ndarray]] | None = None

    for indices in itertools.combinations(range(len(candidates)), 5):
        selected = sorted((candidates[i] for i in indices), key=lambda item: item[0])
        image_y_values = np.array([item[0] for item in selected], dtype=np.float64)

        if np.min(np.diff(image_y_values)) < 25:
            continue

        spacing_error = _projective_1d_fit_error(court_y_values, image_y_values)
        image_span = image_y_values[-1] - image_y_values[0]
        mean_length = float(np.mean([item[1] for item in selected]))
        score = spacing_error - 0.01 * image_span - 0.001 * mean_length

        if best is None or score < best[0]:
            best = (score, [item[2] for item in selected])

    return best[1] if best is not None else None


def _select_vertical_court_lines(
    vertical_lines: np.ndarray,
    horizontal_lines_top_to_bottom: list[np.ndarray],
    image_shape: tuple,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if len(vertical_lines) < 2:
        return None, None, None

    h, w = image_shape[:2]
    top_y = _line_y_at_x(horizontal_lines_top_to_bottom[0], w / 2)
    bottom_y = _line_y_at_x(horizontal_lines_top_to_bottom[-1], w / 2)
    min_length = 0.04 * np.hypot(h, w)

    side_candidates = []
    center_candidates = []
    for line in vertical_lines:
        length = _line_length(line)
        if length < min_length:
            continue

        angle = _normalized_angle_deg(line)
        x_top = _line_x_at_y(line, top_y)
        x_bottom = _line_x_at_y(line, bottom_y)
        if not (-w * 0.1 <= x_top <= w * 1.1 and -w * 0.1 <= x_bottom <= w * 1.1):
            continue

        if 45 <= angle <= 75:
            side_candidates.append((x_top, x_bottom, length, line))
        elif angle >= 80:
            center_x = (x_top + x_bottom) / 2
            if w * 0.35 <= center_x <= w * 0.65:
                score = abs(center_x - w / 2) - 0.001 * length
                center_candidates.append((score, line))

    left_candidates = [
        candidate for candidate in side_candidates if candidate[1] < w * 0.45
    ]
    right_candidates = [
        candidate for candidate in side_candidates if candidate[1] > w * 0.55
    ]

    if not left_candidates or not right_candidates:
        return None, None, None

    left_line = min(left_candidates, key=lambda candidate: candidate[1])[3]
    right_line = max(right_candidates, key=lambda candidate: candidate[1])[3]
    center_line = min(center_candidates, key=lambda candidate: candidate[0])[1] if center_candidates else None

    return left_line, right_line, center_line


def _projective_1d_fit_error(source: np.ndarray, target: np.ndarray) -> float:
    rows = []
    rhs = []
    for src, dst in zip(source, target):
        rows.append([src, 1.0, -dst * src])
        rhs.append(dst)

    params, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    projected = (params[0] * source + params[1]) / (params[2] * source + 1.0)
    return float(np.mean(np.abs(projected - target)))


def _line_y_at_x(line: np.ndarray, x: float) -> float:
    x1, y1, x2, y2 = line
    if abs(x2 - x1) < 1e-8:
        return float((y1 + y2) / 2)
    t = (x - x1) / (x2 - x1)
    return float(y1 + t * (y2 - y1))


def _line_x_at_y(line: np.ndarray, y: float) -> float:
    x1, y1, x2, y2 = line
    if abs(y2 - y1) < 1e-8:
        return float((x1 + x2) / 2)
    t = (y - y1) / (y2 - y1)
    return float(x1 + t * (x2 - x1))


def _line_length(line: np.ndarray) -> float:
    return float(np.hypot(line[2] - line[0], line[3] - line[1]))


def _normalized_angle_deg(line: np.ndarray) -> float:
    dx = line[2] - line[0]
    dy = line[3] - line[1]
    angle = abs(float(np.degrees(np.arctan2(dy, dx))))
    return 180 - angle if angle > 90 else angle


def match_lines_to_template(
    detected_lines: np.ndarray,
    geometry: CourtGeometry2D,
    image_shape: tuple,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Match detected line intersections to court template points.

    Strategy:
    1. Find intersections of detected lines
    2. Use geometric consistency (convex hull, aspect ratio) to match to template
    3. Return (image_points, court_points) correspondences

    Args:
        detected_lines: Nx4 array of detected lines in image coordinates.
        geometry: Court geometry for template.
        image_shape: (height, width) of the image.
        valid_mask: Optional image mask for keeping intersections inside the court ROI.

    Returns:
        (image_points, court_points) both as Mx2 arrays.
    """
    if len(detected_lines) < 4:
        return np.empty((0, 2)), np.empty((0, 2))

    # Find all pairwise intersections
    intersections = _find_line_intersections(detected_lines, image_shape, valid_mask)
    if len(intersections) < 4:
        return np.empty((0, 2)), np.empty((0, 2))

    # Get court template points
    template_points = get_court_template_points(geometry)

    # Try to find the best matching via initial homography guess
    # Use the 4-corner approach: find the approximate court corners
    image_corners = _find_court_corners(intersections, image_shape)
    if image_corners is None or len(image_corners) < 4:
        return np.empty((0, 2)), np.empty((0, 2))

    # Court corners in template coords (4 outer corners)
    w = geometry.width_m
    length = geometry.length_m
    court_corners = np.array(
        [[0, length], [w, length], [w, 0], [0, 0]], dtype=np.float64
    )

    # Estimate initial homography from corners
    H_init, _ = cv2.findHomography(image_corners, court_corners)
    if H_init is None:
        return np.empty((0, 2)), np.empty((0, 2))

    # Project all intersections to court space using initial homography
    ones = np.ones((len(intersections), 1))
    pts_h = np.hstack([intersections, ones])
    projected = (H_init @ pts_h.T).T
    projected = projected[:, :2] / projected[:, 2:3]

    # Match projected points to nearest template points. Keep at most one image
    # point per court keypoint so clutter cannot vote hundreds of times for the
    # same template coordinate.
    candidates_by_template: dict[int, tuple[float, np.ndarray]] = {}

    for i, proj_pt in enumerate(projected):
        # Find nearest template point
        dists = np.sqrt(np.sum((template_points - proj_pt) ** 2, axis=1))
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]

        # Accept if within reasonable distance (2 meters in court space)
        if min_dist < 2.0:
            current = candidates_by_template.get(min_idx)
            if current is None or min_dist < current[0]:
                candidates_by_template[min_idx] = (min_dist, intersections[i])

    if len(candidates_by_template) < 4:
        return np.empty((0, 2)), np.empty((0, 2))

    matched = sorted(candidates_by_template.items())
    matched_image = [candidate[1] for _, candidate in matched]
    matched_court = [template_points[min_idx] for min_idx, _ in matched]

    return np.array(matched_image, dtype=np.float64), np.array(matched_court, dtype=np.float64)


def _find_line_intersections(
    lines: np.ndarray,
    image_shape: tuple,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Find intersections between line segments that fall within the image bounds."""
    h, w = image_shape[:2]
    intersections = []

    n = len(lines)
    for i in range(n):
        for j in range(i + 1, n):
            pt = _line_intersection(lines[i], lines[j])
            if pt is not None:
                x, y = pt
                # Keep only points within image bounds (with small margin)
                if -w * 0.1 <= x <= w * 1.1 and -h * 0.1 <= y <= h * 1.1:
                    if valid_mask is not None and not _point_in_mask(x, y, valid_mask):
                        continue
                    intersections.append(pt)

    if not intersections:
        return np.empty((0, 2))

    pts = np.array(intersections, dtype=np.float64)

    # Remove duplicates (points very close together)
    if len(pts) > 1:
        pts = _remove_duplicate_points(pts, threshold=10.0)

    return pts


def _point_in_mask(x: float, y: float, mask: np.ndarray) -> bool:
    """Return True if a point falls inside a nonzero mask pixel."""
    h, w = mask.shape[:2]
    xi = int(round(x))
    yi = int(round(y))
    return 0 <= xi < w and 0 <= yi < h and mask[yi, xi] > 0


def _line_intersection(line1: np.ndarray, line2: np.ndarray) -> tuple[float, float] | None:
    """Find intersection of two line segments (extended to full lines).

    Returns None if lines are nearly parallel.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-8:
        return None

    # Check that lines are not nearly parallel (angle between them > 15 degrees)
    dx1, dy1 = x2 - x1, y2 - y1
    dx2, dy2 = x4 - x3, y4 - y3
    len1 = np.sqrt(dx1**2 + dy1**2)
    len2 = np.sqrt(dx2**2 + dy2**2)
    if len1 < 1e-8 or len2 < 1e-8:
        return None

    cos_angle = abs(dx1 * dx2 + dy1 * dy2) / (len1 * len2)
    if cos_angle > 0.966:  # ~15 degrees
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)
    return (px, py)


def _remove_duplicate_points(points: np.ndarray, threshold: float) -> np.ndarray:
    """Remove points that are within threshold distance of each other."""
    keep = []
    used = np.zeros(len(points), dtype=bool)

    for i in range(len(points)):
        if used[i]:
            continue
        keep.append(points[i])
        for j in range(i + 1, len(points)):
            if used[j]:
                continue
            dist = np.sqrt(np.sum((points[i] - points[j]) ** 2))
            if dist < threshold:
                used[j] = True

    return np.array(keep, dtype=np.float64) if keep else np.empty((0, 2))


def _find_court_corners(intersections: np.ndarray, image_shape: tuple) -> np.ndarray | None:
    """Find the 4 approximate court corners from intersection points.

    Uses convex hull and selects the 4 points that form the largest quadrilateral
    with reasonable aspect ratio.
    """
    if len(intersections) < 4:
        return None

    h, w = image_shape[:2]

    # Filter to points within image
    mask = (
        (intersections[:, 0] >= 0)
        & (intersections[:, 0] <= w)
        & (intersections[:, 1] >= 0)
        & (intersections[:, 1] <= h)
    )
    pts = intersections[mask]

    if len(pts) < 4:
        return None

    # Find convex hull
    pts_int = pts.astype(np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts_int)

    if len(hull) < 4:
        return None

    # Approximate hull with 4 points
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    # Increase epsilon until we get 4 points or give up
    attempts = 0
    while len(approx) > 4 and attempts < 20:
        epsilon *= 1.2
        approx = cv2.approxPolyDP(hull, epsilon, True)
        attempts += 1

    if len(approx) < 4:
        # Take the 4 extreme points instead
        pts_flat = pts
        top_left = pts_flat[np.argmin(pts_flat[:, 0] + pts_flat[:, 1])]
        top_right = pts_flat[np.argmax(pts_flat[:, 0] - pts_flat[:, 1])]
        bottom_right = pts_flat[np.argmax(pts_flat[:, 0] + pts_flat[:, 1])]
        bottom_left = pts_flat[np.argmin(pts_flat[:, 0] - pts_flat[:, 1])]
        corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float64)
    elif len(approx) == 4:
        corners = approx.reshape(4, 2).astype(np.float64)
    else:
        # Take the 4 with largest area
        hull_pts = hull.reshape(-1, 2)
        if len(hull_pts) >= 4:
            # Use extreme points
            top_left = hull_pts[np.argmin(hull_pts[:, 0] + hull_pts[:, 1])]
            top_right = hull_pts[np.argmax(hull_pts[:, 0] - hull_pts[:, 1])]
            bottom_right = hull_pts[np.argmax(hull_pts[:, 0] + hull_pts[:, 1])]
            bottom_left = hull_pts[np.argmin(hull_pts[:, 0] - hull_pts[:, 1])]
            corners = np.array(
                [top_left, top_right, bottom_right, bottom_left], dtype=np.float64
            )
        else:
            return None

    # Order corners: top-left, top-right, bottom-right, bottom-left
    corners = _order_corners(corners)
    return corners


def _order_corners(corners: np.ndarray) -> np.ndarray:
    """Order 4 corners as: top-left, top-right, bottom-right, bottom-left.

    Corresponds to court corners: (0,0), (w,0), (w,l), (0,l).
    """
    # Sort by y coordinate (top to bottom)
    sorted_by_y = corners[np.argsort(corners[:, 1])]

    # Top two points
    top = sorted_by_y[:2]
    top = top[np.argsort(top[:, 0])]  # left to right

    # Bottom two points
    bottom = sorted_by_y[2:]
    bottom = bottom[np.argsort(bottom[:, 0])]  # left to right

    # Result: TL, TR, BR, BL
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float64)
