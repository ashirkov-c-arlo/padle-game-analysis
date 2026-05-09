from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --- Type aliases ---

PlayerID = Literal["near_left", "near_right", "far_left", "far_right"]
RegistrationMode = Literal["floor_homography", "pixel_only"]
FormationState = Literal[
    "both_net", "both_mid", "both_baseline", "one_up_one_back", "split_unknown"
]
BallState = Literal["detected", "tracked", "interpolated", "missing"]
BallVisibility = Literal["visible", "blurred", "occluded", "not_visible"]
BallEventType = Literal[
    "bounce_candidate", "touch_candidate", "serve_candidate", "net_crossing_candidate"
]
MetricQuality = Literal["estimated", "proxy"]
Zone = Literal["net", "mid", "baseline"]


# --- Court Geometry ---


class CourtGeometry2D(BaseModel):
    """Court dimensions, line definitions, and zone boundaries."""

    width_m: float = 10.0
    length_m: float = 20.0
    net_y_m: float = 10.0
    service_line_offset_from_net_m: float = 6.95
    net_height_center_m: float = 0.88
    net_height_posts_m: float = 0.92
    lines: list[str] = Field(default_factory=lambda: [
        "near_baseline",
        "far_baseline",
        "left_sideline",
        "right_sideline",
        "net_line",
        "near_service_line",
        "far_service_line",
        "near_center_service_line",
        "far_center_service_line",
    ])
    zone_net_distance_m: float = 3.5
    zone_mid_distance_m: float = 6.95

    model_config = {"arbitrary_types_allowed": True}


# --- Court Registration ---


class CourtRegistration2D(BaseModel):
    """Result of court-to-image registration."""

    mode: RegistrationMode
    homography_court_to_image: list[list[float]] | None = None
    homography_image_to_court: list[list[float]] | None = None
    reprojection_error_px: float | None = None
    num_inliers: int | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    model_config = {"arbitrary_types_allowed": True}


# --- Player Detection & Tracking ---


class PlayerDetection(BaseModel):
    """Single player detection in a frame."""

    frame: int
    bbox_xyxy: tuple[float, float, float, float]
    cls: int = 0  # person class
    confidence: float = Field(ge=0.0, le=1.0)


class PlayerTrack(BaseModel):
    """A tracked player across frames."""

    player_id: PlayerID
    frames: list[int]
    bboxes: list[tuple[float, float, float, float]]
    confidences: list[float]
    team: Literal["near", "far"]


class PlayerMetricFrame(BaseModel):
    """Per-frame metric data for a player in court coordinates."""

    frame: int
    time_s: float
    player_id: PlayerID
    court_xy: tuple[float, float]
    speed_mps: float = 0.0
    zone: Zone
    confidence: float = Field(ge=0.0, le=1.0)
    metric_quality: MetricQuality = "estimated"


# --- Ball Detection & Tracking ---


class BallDetection2D(BaseModel):
    """Single ball detection in image space."""

    frame: int
    time_s: float
    image_xy: tuple[float, float]
    confidence: float = Field(ge=0.0, le=1.0)
    visibility: BallVisibility = "visible"
    source: str = "wasb_sbdt"


class BallTrack2D(BaseModel):
    """Tracked ball position with velocity in image space."""

    frame: int
    time_s: float
    image_xy: tuple[float, float]
    velocity_px_s: tuple[float, float] | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    state: BallState = "detected"
    interpolated: bool = False
    gap_len: int = 0


class BallEventCandidate(BaseModel):
    """Candidate ball event (bounce, touch, etc.)."""

    frame: int
    time_s: float
    event_type: BallEventType
    image_xy: tuple[float, float]
    court_xy_approx: tuple[float, float] | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    projection_quality: MetricQuality = "estimated"
    evidence: dict | None = None


class BallMetricFrame(BaseModel):
    """Per-frame ball metric in court coordinates."""

    frame: int
    time_s: float
    metric_type: str
    player_id: PlayerID | None = None
    team_id: Literal["near", "far"] | None = None
    court_xy_projected: tuple[float, float] | None = None
    value: float | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    quality: MetricQuality = "proxy"


# --- Analytics Metrics ---


class RallyTempoMetric(BaseModel):
    """Tempo metrics for a rally."""

    rally_id: int
    duration_s: float
    estimated_shots: int
    avg_time_between_touches_s: float | None = None
    median_time_between_touches_s: float | None = None


class ServePlacementMetric(BaseModel):
    """Serve placement analysis."""

    serve_id: int
    frame: int
    time_s: float
    landing_zone: str
    confidence: float = Field(ge=0.0, le=1.0)


class ShotDirectionProxyMetric(BaseModel):
    """Proxy metric for shot direction."""

    frame: int
    time_s: float
    player_id: PlayerID
    direction: Literal["cross_court", "down_the_line", "middle", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)


class ShotDepthProxyMetric(BaseModel):
    """Proxy metric for shot depth."""

    frame: int
    time_s: float
    player_id: PlayerID
    depth: Literal["short", "mid", "deep", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)


class PlayerPressureMetric(BaseModel):
    """Player pressure analysis."""

    frame: int
    time_s: float
    player_id: PlayerID
    pressure_score: float = Field(ge=0.0, le=1.0)
    defender_distance_m: float | None = None
    time_available_s: float | None = None


# --- Scoreboard ---


class ScoreboardState(BaseModel):
    """Parsed scoreboard state from a frame."""

    frame: int
    time_s: float
    raw_text: str | None = None
    parsed_sets: list[tuple[int, int]] | None = None
    parsed_game_score: tuple[int, int] | None = None
    confidence: float = Field(ge=0.0, le=1.0)


# --- Pipeline Output ---


class FrameResult(BaseModel):
    """Complete per-frame analysis result."""

    frame: int
    time_s: float
    players: list[PlayerMetricFrame] = Field(default_factory=list)
    ball: BallTrack2D | None = None
    registration_mode: RegistrationMode = "pixel_only"


class MVPOutput(BaseModel):
    """Top-level output of the MVP pipeline."""

    video_path: str
    total_frames: int
    fps: float
    duration_s: float
    registration_mode: RegistrationMode
    court_geometry: CourtGeometry2D
    court_registration: CourtRegistration2D | None = None
    player_tracks: list[PlayerTrack] = Field(default_factory=list)
    frame_results: list[FrameResult] = Field(default_factory=list)
    ball_tracks: list[BallTrack2D] = Field(default_factory=list)
    ball_events: list[BallEventCandidate] = Field(default_factory=list)
    rally_metrics: list[RallyTempoMetric] = Field(default_factory=list)
    scoreboard_states: list[ScoreboardState] = Field(default_factory=list)
    config: dict = Field(default_factory=dict)
