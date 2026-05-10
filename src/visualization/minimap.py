from __future__ import annotations

import cv2
import numpy as np

from src.schemas import CourtGeometry2D, PlayerMetricFrame


def create_court_base(
    width_px: int = 300,
    length_px: int = 600,
    geometry: CourtGeometry2D | None = None,
) -> np.ndarray:
    """Create a clean top-down court image.

    Draws all court lines on green background.
    Scale: court 10m x 20m mapped to width_px x length_px.
    """
    if geometry is None:
        geometry = CourtGeometry2D()

    # Green background
    court = np.zeros((length_px, width_px, 3), dtype=np.uint8)
    court[:] = (34, 139, 34)  # Forest green (BGR)

    # Scale factor: pixels per meter
    sy = length_px / geometry.length_m

    line_color = (255, 255, 255)  # White
    thickness = 2

    # Baselines (y-axis flipped: near baseline at y=0 drawn at bottom, far at top)
    cv2.line(court, (0, 0), (width_px - 1, 0), line_color, thickness)  # far baseline (top)
    cv2.line(court, (0, length_px - 1), (width_px - 1, length_px - 1), line_color, thickness)  # near baseline (bottom)

    # Sidelines
    cv2.line(court, (0, 0), (0, length_px - 1), line_color, thickness)  # left
    cv2.line(court, (width_px - 1, 0), (width_px - 1, length_px - 1), line_color, thickness)  # right

    # Y-axis is flipped: court y=0 (near) at bottom, y=20 (far) at top
    # Convert court y to pixel y: py = length_px - 1 - int(court_y * sy)
    net_y_px = length_px - 1 - int(geometry.net_y_m * sy)
    _draw_dashed_line_h(court, net_y_px, 0, width_px - 1, (200, 200, 200), thickness)

    svc_offset = geometry.service_line_offset_from_net_m
    near_svc_y_px = length_px - 1 - int((geometry.net_y_m - svc_offset) * sy)
    far_svc_y_px = length_px - 1 - int((geometry.net_y_m + svc_offset) * sy)
    cv2.line(court, (0, far_svc_y_px), (width_px - 1, far_svc_y_px), line_color, 1)
    cv2.line(court, (0, near_svc_y_px), (width_px - 1, near_svc_y_px), line_color, 1)

    center_x_px = width_px // 2
    cv2.line(court, (center_x_px, far_svc_y_px), (center_x_px, net_y_px), line_color, 1)
    cv2.line(court, (center_x_px, net_y_px), (center_x_px, near_svc_y_px), line_color, 1)

    return court


def draw_minimap_frame(
    court_base: np.ndarray,
    players: list[PlayerMetricFrame] | None,
    ball_court_xy: tuple[float, float] | None = None,
    geometry: CourtGeometry2D | None = None,
) -> np.ndarray:
    """Draw players and ball on minimap for one frame.

    - Players as colored circles (blue=near, red=far)
    - Ball as small yellow circle
    - Player ID labels
    """
    if geometry is None:
        geometry = CourtGeometry2D()

    frame = court_base.copy()
    length_px, width_px = frame.shape[:2]

    sx = width_px / geometry.width_m
    sy = length_px / geometry.length_m

    # Draw players (y-axis flipped: court y=0 near at bottom of minimap)
    if players:
        for p in players:
            x_px = int(p.court_xy[0] * sx)
            y_px = length_px - 1 - int(p.court_xy[1] * sy)

            # Clamp to image bounds
            x_px = max(0, min(width_px - 1, x_px))
            y_px = max(0, min(length_px - 1, y_px))

            # Color based on team (infer from player_id)
            is_near = p.player_id.startswith("near")
            color = (255, 0, 0) if is_near else (0, 0, 255)  # blue=near, red=far (BGR)

            cv2.circle(frame, (x_px, y_px), 8, color, -1)
            cv2.circle(frame, (x_px, y_px), 9, (255, 255, 255), 1)

            # Short label
            short_label = p.player_id.split("_")[1][0].upper()  # "L" or "R"
            cv2.putText(
                frame, short_label, (x_px - 4, y_px + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
            )

    # Draw ball (y-axis flipped)
    if ball_court_xy is not None:
        bx_px = int(ball_court_xy[0] * sx)
        by_px = length_px - 1 - int(ball_court_xy[1] * sy)
        bx_px = max(0, min(width_px - 1, bx_px))
        by_px = max(0, min(length_px - 1, by_px))
        cv2.circle(frame, (bx_px, by_px), 5, (0, 255, 255), -1)  # yellow (BGR)

    return frame


def _draw_dashed_line_h(
    img: np.ndarray,
    y: int,
    x_start: int,
    x_end: int,
    color: tuple[int, int, int],
    thickness: int,
    dash_len: int = 12,
) -> None:
    """Draw a horizontal dashed line."""
    x = x_start
    while x < x_end:
        end_x = min(x + dash_len, x_end)
        cv2.line(img, (x, y), (end_x, y), color, thickness)
        x += dash_len * 2
