from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.schemas import BallState, CourtGeometry2D, PlayerMetricFrame


# --- Trail state management ---

# Trail entry: (court_x, court_y, speed_mps)
TrailEntry = tuple[float, float, float]


@dataclass
class MinimapTrailState:
    """Accumulates position history for trail rendering.

    Create once and pass to draw_minimap_frame on each call.
    """

    player_trail_length: int = 30
    ball_trail_length: int = 20
    # player_id -> deque of TrailEntry
    player_history: dict[str, deque[TrailEntry]] = field(default_factory=dict)
    # deque of TrailEntry (speed unused for ball, stored as 0)
    ball_history: deque[TrailEntry] = field(default_factory=lambda: deque(maxlen=20))

    def __post_init__(self) -> None:
        # Ensure ball_history has the correct maxlen
        if not hasattr(self.ball_history, 'maxlen') or self.ball_history.maxlen != self.ball_trail_length:
            self.ball_history = deque(maxlen=self.ball_trail_length)

    def push_player(self, player_id: str, court_x: float, court_y: float, speed_mps: float) -> None:
        if player_id not in self.player_history:
            self.player_history[player_id] = deque(maxlen=self.player_trail_length)
        self.player_history[player_id].append((court_x, court_y, speed_mps))

    def push_ball(self, court_x: float, court_y: float) -> None:
        self.ball_history.append((court_x, court_y, 0.0))

    def skip_ball(self) -> None:
        """Mark a frame where ball was missing (breaks the trail)."""
        self.ball_history.append(None)  # type: ignore[arg-type]


def _speed_to_color_bgr(base_bgr: tuple[int, int, int], speed_mps: float) -> tuple[int, int, int]:
    """Modulate color brightness based on speed using HSV interpolation.

    Slow (0 m/s) = darker/muted, Fast (>5 m/s) = bright/vivid.
    """
    # Normalize speed: 0 -> 0.4 brightness, >=5 -> 1.0 brightness
    t = min(speed_mps / 5.0, 1.0)
    brightness_factor = 0.4 + 0.6 * t

    # Convert BGR to HSV, scale V channel, convert back
    pixel = np.array([[base_bgr]], dtype=np.uint8)
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
    hsv[0, 0, 1] = np.clip(int(hsv[0, 0, 1] * (0.6 + 0.4 * t)), 0, 255)  # saturation
    hsv[0, 0, 2] = np.clip(int(hsv[0, 0, 2] * brightness_factor), 0, 255)  # value
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2])


def _draw_trail(
    frame: np.ndarray,
    history: deque[TrailEntry | None],
    sx: float,
    sy: float,
    length_px: int,
    color_bgr: tuple[int, int, int],
    radius: int,
) -> np.ndarray:
    """Draw a fading trail from history entries onto frame.

    Uses addWeighted blending for alpha/transparency effect.
    Oldest point = alpha 0.1, newest = alpha 0.8.
    """
    points = []
    for entry in history:
        if entry is None:
            # Break in trail (ball missing) - draw what we have, then reset
            if len(points) >= 2:
                frame = _render_trail_segment(frame, points, color_bgr, radius, len(history))
            points = []
            continue
        court_x, court_y, _ = entry
        x_px = int(court_x * sx)
        y_px = length_px - 1 - int(court_y * sy)
        points.append((x_px, y_px))

    if len(points) >= 2:
        frame = _render_trail_segment(frame, points, color_bgr, radius, len(history))

    return frame


def _render_trail_segment(
    frame: np.ndarray,
    points: list[tuple[int, int]],
    color_bgr: tuple[int, int, int],
    radius: int,
    total_history_len: int,
) -> np.ndarray:
    """Render a continuous trail segment with fading alpha.

    Draws all trail points on an overlay, then blends once. Uses a single
    addWeighted call with a mid-range alpha, and pre-adjusts circle colors
    to approximate per-point fading.
    """
    n = len(points)
    # Overall blend alpha for the overlay pass
    blend_alpha = 0.5
    overlay = frame.copy()

    for i, (px, py) in enumerate(points):
        # Per-point fade: 0.1 (oldest) to 0.8 (newest)
        point_alpha = 0.1 + 0.7 * (i / max(n - 1, 1))
        # Scale color intensity to encode per-point alpha within the single blend
        # effective_alpha = blend_alpha * (intensity / 255), so intensity = point_alpha / blend_alpha * 255
        intensity = min(point_alpha / blend_alpha, 1.0)
        scaled_color = (
            int(color_bgr[0] * intensity),
            int(color_bgr[1] * intensity),
            int(color_bgr[2] * intensity),
        )
        cv2.circle(overlay, (px, py), radius, scaled_color, -1)

    cv2.addWeighted(overlay, blend_alpha, frame, 1.0 - blend_alpha, 0, frame)
    return frame


# --- Court base creation ---


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
    trail_state: MinimapTrailState | None = None,
    ball_state: BallState | None = None,
) -> np.ndarray:
    """Draw players and ball on minimap for one frame.

    - Players as colored circles (blue=near, red=far)
    - Ball as small yellow circle
    - Player ID labels
    - Optional trails (fading position history) when trail_state is provided
    - Speed-based color modulation for player markers
    """
    if geometry is None:
        geometry = CourtGeometry2D()

    frame = court_base.copy()
    length_px, width_px = frame.shape[:2]

    sx = width_px / geometry.width_m
    sy = length_px / geometry.length_m

    # Update trail state with current positions
    if trail_state is not None:
        if players:
            for p in players:
                trail_state.push_player(p.player_id, p.court_xy[0], p.court_xy[1], p.speed_mps)

        if ball_court_xy is not None and ball_state in ("detected", "tracked", "interpolated"):
            trail_state.push_ball(ball_court_xy[0], ball_court_xy[1])
        else:
            trail_state.skip_ball()

    # Draw ball trail (before player trails)
    if trail_state is not None and len(trail_state.ball_history) > 1:
        frame = _draw_trail(
            frame,
            trail_state.ball_history,
            sx, sy, length_px,
            color_bgr=(0, 255, 255),  # yellow
            radius=2,
        )

    # Draw player trails (before current position markers)
    if trail_state is not None and players:
        for p in players:
            history = trail_state.player_history.get(p.player_id)
            if history and len(history) > 1:
                is_near = p.player_id.startswith("near")
                trail_color = (255, 0, 0) if is_near else (0, 0, 255)
                frame = _draw_trail(
                    frame,
                    history,
                    sx, sy, length_px,
                    color_bgr=trail_color,
                    radius=3,
                )

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
            base_color = (255, 0, 0) if is_near else (0, 0, 255)  # blue=near, red=far (BGR)

            # Modulate color by speed
            color = _speed_to_color_bgr(base_color, p.speed_mps) if trail_state is not None else base_color

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
