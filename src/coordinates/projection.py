from __future__ import annotations

import numpy as np
from loguru import logger

from src.schemas import CourtRegistration2D, PlayerTrack


def footpoint_to_court(
    bbox_xyxy: tuple[float, float, float, float],
    H_image_to_court: np.ndarray,
) -> tuple[float, float]:
    """Project bbox bottom-center through homography to court coords.

    Args:
        bbox_xyxy: Bounding box as (x1, y1, x2, y2).
        H_image_to_court: 3x3 homography matrix (numpy array).

    Returns:
        (x, y) in court coordinates (meters).
    """
    x1, y1, x2, y2 = bbox_xyxy
    foot_x = (x1 + x2) / 2.0
    foot_y = y2

    pt = np.array([foot_x, foot_y, 1.0], dtype=np.float64)
    projected = H_image_to_court @ pt

    w = projected[2]
    if abs(w) < 1e-10:
        return (0.0, 0.0)

    return (float(projected[0] / w), float(projected[1] / w))


def project_tracks_to_court(
    tracks: list[PlayerTrack],
    registration: CourtRegistration2D,
    fps: float,
) -> dict[str, list[tuple[float, float]]]:
    """Project player tracks from pixel bboxes to court coordinates.

    For each track, project bbox bottom-center (footpoint) to court coords.
    Only works if registration.mode == "floor_homography".

    Args:
        tracks: List of player tracks with bounding boxes.
        registration: Court registration containing homography matrices.
        fps: Video frame rate (unused here but part of pipeline interface).

    Returns:
        Dictionary mapping player_id to list of (x, y) court positions.
        Returns empty dict if registration mode is not floor_homography.
    """
    if registration.mode != "floor_homography":
        logger.warning("Cannot project tracks: registration mode is '{}', need 'floor_homography'", registration.mode)
        return {}

    if registration.homography_image_to_court is None:
        logger.warning("Cannot project tracks: homography_image_to_court is None")
        return {}

    H = np.array(registration.homography_image_to_court, dtype=np.float64)

    result: dict[str, list[tuple[float, float]]] = {}

    for track in tracks:
        positions: list[tuple[float, float]] = []
        for bbox in track.bboxes:
            court_xy = footpoint_to_court(bbox, H)
            positions.append(court_xy)

        result[track.player_id] = positions
        logger.debug("Projected {} positions for player '{}'", len(positions), track.player_id)

    return result
