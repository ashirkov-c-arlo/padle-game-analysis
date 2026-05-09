from __future__ import annotations

import numpy as np

from src.schemas import CourtRegistration2D, PlayerDetection


def get_footpoint(bbox_xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    """Get bottom-center point of bounding box (foot approximation).

    Args:
        bbox_xyxy: Bounding box as (x1, y1, x2, y2).

    Returns:
        (x, y) of bottom-center point.
    """
    x1, y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2, y2)


def project_to_court(
    pixel_xy: tuple[float, float],
    homography_image_to_court: list[list[float]],
) -> tuple[float, float]:
    """Project image pixel to court coordinates using homography.

    Args:
        pixel_xy: (x, y) in image pixel coordinates.
        homography_image_to_court: 3x3 homography matrix as nested list.

    Returns:
        (x, y) in court coordinates (meters).
    """
    H = np.array(homography_image_to_court, dtype=np.float64)
    pt = np.array([pixel_xy[0], pixel_xy[1], 1.0], dtype=np.float64)
    projected = H @ pt
    # Normalize by w
    w = projected[2]
    if abs(w) < 1e-10:
        return (0.0, 0.0)
    return (projected[0] / w, projected[1] / w)


def filter_detections_by_court_roi(
    detections: list[PlayerDetection],
    registration: CourtRegistration2D | None,
    image_shape: tuple[int, int],
    margin_px: int = 50,
) -> list[PlayerDetection]:
    """Filter out detections whose footpoints fall outside the court area.

    Args:
        detections: List of player detections to filter.
        registration: Court registration result (may be None).
        image_shape: (height, width) of the image.
        margin_px: Margin in court meters to allow detections slightly outside court.

    Returns:
        Filtered list of detections within court bounds.
    """
    # If no registration or pixel_only mode, return all detections
    if registration is None or registration.mode == "pixel_only":
        return detections

    # Need floor_homography mode with a valid homography
    if registration.homography_image_to_court is None:
        return detections

    # Court dimensions: 10m wide x 20m long
    court_width = 10.0
    court_length = 20.0
    margin_m = margin_px  # margin parameter is in court-meters despite the name

    filtered: list[PlayerDetection] = []
    for det in detections:
        foot_px = get_footpoint(det.bbox_xyxy)
        court_xy = project_to_court(foot_px, registration.homography_image_to_court)

        x, y = court_xy
        if (
            -margin_m <= x <= court_width + margin_m
            and -margin_m <= y <= court_length + margin_m
        ):
            filtered.append(det)

    return filtered
