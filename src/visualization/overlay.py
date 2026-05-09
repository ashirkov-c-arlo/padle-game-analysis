from __future__ import annotations

import cv2
import numpy as np

from src.schemas import (
    BallTrack2D,
    CourtGeometry2D,
    CourtRegistration2D,
    PlayerMetricFrame,
    PlayerTrack,
    ScoreboardState,
)


def draw_court_overlay(
    frame: np.ndarray,
    registration: CourtRegistration2D,
    geometry: CourtGeometry2D,
) -> np.ndarray:
    """Draw projected 2D court lines on frame.

    Uses court_to_image homography to project court line endpoints.
    Draws: baselines, sidelines, net, service lines, center service lines.
    Color: semi-transparent green lines.
    """
    if registration.mode != "floor_homography" or registration.homography_court_to_image is None:
        return frame

    H = np.array(registration.homography_court_to_image, dtype=np.float64)
    overlay = frame.copy()
    color = (0, 200, 0)  # Green
    thickness = 2

    w = geometry.width_m
    court_len = geometry.length_m
    net_y = geometry.net_y_m
    svc_offset = geometry.service_line_offset_from_net_m

    # Define court lines as (x1, y1, x2, y2) in court coordinates
    lines = [
        # Baselines
        (0, 0, w, 0),           # far baseline
        (0, court_len, w, court_len),           # near baseline
        # Sidelines
        (0, 0, 0, court_len),           # left sideline
        (w, 0, w, court_len),           # right sideline
        # Service lines
        (0, net_y - svc_offset, w, net_y - svc_offset),  # far service line
        (0, net_y + svc_offset, w, net_y + svc_offset),  # near service line
        # Center service lines
        (w / 2, net_y - svc_offset, w / 2, net_y),       # far center
        (w / 2, net_y, w / 2, net_y + svc_offset),       # near center
    ]

    for x1, y1, x2, y2 in lines:
        pt1_img = _project_point(H, x1, y1)
        pt2_img = _project_point(H, x2, y2)
        if pt1_img is not None and pt2_img is not None:
            cv2.line(overlay, pt1_img, pt2_img, color, thickness)

    # Draw net as dashed line
    net_pt1 = _project_point(H, 0, net_y)
    net_pt2 = _project_point(H, w, net_y)
    if net_pt1 is not None and net_pt2 is not None:
        _draw_dashed_line(overlay, net_pt1, net_pt2, (0, 255, 255), thickness + 1)

    # Blend overlay with original for semi-transparency
    alpha = 0.4
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return frame


def draw_player_boxes(
    frame: np.ndarray,
    tracks: list[PlayerTrack],
    frame_idx: int,
    metrics: list[PlayerMetricFrame] | None = None,
) -> np.ndarray:
    """Draw player bounding boxes with colored box, player ID label, and speed."""
    # Build a lookup for metrics at this frame
    metric_lookup: dict[str, PlayerMetricFrame] = {}
    if metrics:
        for m in metrics:
            if m.frame == frame_idx:
                metric_lookup[m.player_id] = m

    for track in tracks:
        if frame_idx not in track.frames:
            continue

        idx = track.frames.index(frame_idx)
        bbox = track.bboxes[idx]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Color based on team
        color = (255, 0, 0) if track.team == "near" else (0, 0, 255)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Player ID label
        label = track.player_id
        metric = metric_lookup.get(track.player_id)
        if metric:
            label += f" {metric.speed_mps:.1f}m/s"

        # Draw label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), font, font_scale, (255, 255, 255), 1)

    return frame


def draw_ball_marker(
    frame: np.ndarray,
    ball_track: BallTrack2D | None,
) -> np.ndarray:
    """Draw ball position with color based on state."""
    if ball_track is None or ball_track.state == "missing":
        return frame

    x, y = int(ball_track.image_xy[0]), int(ball_track.image_xy[1])

    # Color based on state
    state_colors = {
        "detected": (0, 255, 0),       # green
        "tracked": (0, 255, 255),       # yellow
        "interpolated": (0, 165, 255),  # orange
    }
    color = state_colors.get(ball_track.state, (0, 255, 255))

    # Draw circle
    cv2.circle(frame, (x, y), 8, color, -1)
    cv2.circle(frame, (x, y), 10, (255, 255, 255), 1)

    return frame


def draw_scoreboard_info(
    frame: np.ndarray,
    score: ScoreboardState | None,
) -> np.ndarray:
    """Draw current score in top-right corner of frame."""
    if score is None:
        return frame

    # Build score text
    parts = []
    if score.parsed_sets:
        sets_str = " ".join(f"{a}-{b}" for a, b in score.parsed_sets)
        parts.append(f"Sets: {sets_str}")
    if score.parsed_game_score:
        parts.append(f"Game: {score.parsed_game_score[0]}-{score.parsed_game_score[1]}")

    if not parts:
        return frame

    text = "  |  ".join(parts)
    h, w = frame.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)

    # Position in top-right
    x = w - tw - 15
    y = 30

    # Background
    cv2.rectangle(frame, (x - 5, y - th - 5), (x + tw + 5, y + 5), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), 1)

    return frame


def draw_formation_info(
    frame: np.ndarray,
    formation_near: str | None,
    formation_far: str | None,
) -> np.ndarray:
    """Draw formation state labels on the left side."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    h = frame.shape[0]

    y_offset = h - 60
    if formation_near:
        text = f"Near: {formation_near}"
        cv2.putText(frame, text, (10, y_offset), font, font_scale, (255, 200, 0), 1)
    if formation_far:
        text = f"Far: {formation_far}"
        cv2.putText(frame, text, (10, y_offset + 20), font, font_scale, (0, 200, 255), 1)

    return frame


def draw_registration_info(
    frame: np.ndarray,
    registration: CourtRegistration2D,
) -> np.ndarray:
    """Draw registration mode and confidence in bottom-right corner."""
    text = f"Reg: {registration.mode} ({registration.confidence:.2f})"
    h, w = frame.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)

    x = w - tw - 10
    y = h - 10

    cv2.putText(frame, text, (x, y), font, font_scale, (200, 200, 200), 1)
    return frame


def annotate_frame(
    frame: np.ndarray,
    frame_idx: int,
    registration: CourtRegistration2D,
    geometry: CourtGeometry2D,
    tracks: list[PlayerTrack],
    metrics: list[PlayerMetricFrame] | None,
    ball_track: BallTrack2D | None,
    score: ScoreboardState | None,
    formation_near: str | None,
    formation_far: str | None,
) -> np.ndarray:
    """Apply all overlays to a frame."""
    frame = draw_court_overlay(frame, registration, geometry)
    frame = draw_player_boxes(frame, tracks, frame_idx, metrics)
    frame = draw_ball_marker(frame, ball_track)
    frame = draw_scoreboard_info(frame, score)
    frame = draw_formation_info(frame, formation_near, formation_far)
    frame = draw_registration_info(frame, registration)
    return frame


def _project_point(
    H: np.ndarray, x: float, y: float
) -> tuple[int, int] | None:
    """Project a court coordinate to image pixel via homography."""
    pt = np.array([x, y, 1.0], dtype=np.float64)
    projected = H @ pt
    w = projected[2]
    if abs(w) < 1e-10:
        return None
    px = int(projected[0] / w)
    py = int(projected[1] / w)
    return (px, py)


def _draw_dashed_line(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    dash_length: int = 15,
) -> None:
    """Draw a dashed line between two points."""
    x1, y1 = pt1
    x2, y2 = pt2
    dx = x2 - x1
    dy = y2 - y1
    dist = int(np.sqrt(dx * dx + dy * dy))
    if dist == 0:
        return

    num_dashes = dist // (dash_length * 2)
    for i in range(num_dashes + 1):
        start_frac = (i * 2 * dash_length) / dist
        end_frac = min(((i * 2 + 1) * dash_length) / dist, 1.0)
        if start_frac > 1.0:
            break
        sx = int(x1 + dx * start_frac)
        sy = int(y1 + dy * start_frac)
        ex = int(x1 + dx * end_frac)
        ey = int(y1 + dy * end_frac)
        cv2.line(img, (sx, sy), (ex, ey), color, thickness)
