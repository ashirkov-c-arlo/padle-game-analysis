"""Tests for schema models."""
from __future__ import annotations

from src.schemas import (
    BallDetection2D,
    BallEventCandidate,
    BallMetricFrame,
    BallTrack2D,
    CourtGeometry2D,
    CourtRegistration2D,
    FrameResult,
    MVPOutput,
    PlayerDetection,
    PlayerMetricFrame,
    PlayerPressureMetric,
    PlayerTrack,
    RallyTempoMetric,
    ScoreboardState,
    ServePlacementMetric,
    ShotDepthProxyMetric,
    ShotDirectionProxyMetric,
)


def test_court_geometry_defaults():
    court = CourtGeometry2D()
    assert court.width_m == 10.0
    assert court.length_m == 20.0
    assert court.net_y_m == 10.0
    assert court.service_line_offset_from_net_m == 6.95
    assert len(court.lines) == 9
    assert "near_baseline" in court.lines
    assert "net_line" in court.lines


def test_court_registration_floor_homography():
    reg = CourtRegistration2D(
        mode="floor_homography",
        homography_court_to_image=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        homography_image_to_court=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        reprojection_error_px=2.5,
        num_inliers=12,
        confidence=0.85,
    )
    assert reg.mode == "floor_homography"
    assert reg.reprojection_error_px == 2.5


def test_court_registration_pixel_only():
    reg = CourtRegistration2D(mode="pixel_only", confidence=0.3)
    assert reg.mode == "pixel_only"
    assert reg.homography_court_to_image is None


def test_player_detection():
    det = PlayerDetection(
        frame=100,
        bbox_xyxy=(100.0, 200.0, 150.0, 350.0),
        cls=0,
        confidence=0.92,
    )
    assert det.frame == 100
    assert det.bbox_xyxy == (100.0, 200.0, 150.0, 350.0)


def test_player_track():
    track = PlayerTrack(
        player_id="near_left",
        frames=[1, 2, 3],
        bboxes=[(10, 20, 30, 40), (11, 21, 31, 41), (12, 22, 32, 42)],
        confidences=[0.9, 0.85, 0.88],
        team="near",
    )
    assert track.player_id == "near_left"
    assert len(track.frames) == 3
    assert track.team == "near"


def test_player_metric_frame():
    metric = PlayerMetricFrame(
        frame=50,
        time_s=1.67,
        player_id="far_right",
        court_xy=(5.0, 15.0),
        speed_mps=3.2,
        zone="mid",
        confidence=0.8,
        metric_quality="estimated",
    )
    assert metric.zone == "mid"
    assert metric.speed_mps == 3.2


def test_ball_detection_2d():
    det = BallDetection2D(
        frame=10,
        time_s=0.33,
        image_xy=(500.0, 300.0),
        confidence=0.75,
        visibility="visible",
        source="tracknet",
    )
    assert det.image_xy == (500.0, 300.0)
    assert det.source == "tracknet"


def test_ball_track_2d():
    track = BallTrack2D(
        frame=20,
        time_s=0.67,
        image_xy=(510.0, 295.0),
        velocity_px_s=(15.0, -8.0),
        confidence=0.8,
        state="tracked",
        interpolated=False,
        gap_len=0,
    )
    assert track.state == "tracked"
    assert track.velocity_px_s == (15.0, -8.0)


def test_ball_event_candidate():
    event = BallEventCandidate(
        frame=30,
        time_s=1.0,
        event_type="bounce_candidate",
        image_xy=(400.0, 500.0),
        court_xy_approx=(5.0, 12.0),
        confidence=0.65,
        projection_quality="estimated",
        evidence={"vertical_velocity_change": True},
    )
    assert event.event_type == "bounce_candidate"
    assert event.court_xy_approx == (5.0, 12.0)


def test_ball_metric_frame():
    metric = BallMetricFrame(
        frame=40,
        time_s=1.33,
        metric_type="speed",
        player_id="near_left",
        team_id="near",
        court_xy_projected=(3.0, 8.0),
        value=25.5,
        confidence=0.7,
        quality="proxy",
    )
    assert metric.metric_type == "speed"
    assert metric.value == 25.5


def test_rally_tempo_metric():
    rally = RallyTempoMetric(
        rally_id=1,
        duration_s=12.5,
        estimated_shots=8,
        avg_time_between_touches_s=1.5,
        median_time_between_touches_s=1.3,
    )
    assert rally.estimated_shots == 8


def test_serve_placement_metric():
    serve = ServePlacementMetric(
        serve_id=1,
        frame=100,
        time_s=3.33,
        landing_zone="deuce_wide",
        confidence=0.72,
    )
    assert serve.landing_zone == "deuce_wide"


def test_shot_direction_proxy_metric():
    shot = ShotDirectionProxyMetric(
        frame=200,
        time_s=6.67,
        player_id="near_right",
        direction="cross_court",
        confidence=0.6,
    )
    assert shot.direction == "cross_court"


def test_shot_depth_proxy_metric():
    shot = ShotDepthProxyMetric(
        frame=210,
        time_s=7.0,
        player_id="far_left",
        depth="deep",
        confidence=0.55,
    )
    assert shot.depth == "deep"


def test_player_pressure_metric():
    pressure = PlayerPressureMetric(
        frame=300,
        time_s=10.0,
        player_id="near_left",
        pressure_score=0.75,
        defender_distance_m=2.5,
        time_available_s=0.8,
    )
    assert pressure.pressure_score == 0.75


def test_scoreboard_state():
    state = ScoreboardState(
        frame=500,
        time_s=16.67,
        raw_text="6-4 3-2 40-15",
        parsed_sets=[(6, 4)],
        parsed_game_score=(40, 15),
        confidence=0.85,
    )
    assert state.parsed_game_score == (40, 15)


def test_frame_result():
    result = FrameResult(
        frame=10,
        time_s=0.33,
        players=[],
        ball=None,
        registration_mode="pixel_only",
    )
    assert result.registration_mode == "pixel_only"
    assert result.players == []


def test_mvp_output_minimal():
    output = MVPOutput(
        video_path="/path/to/video.mp4",
        total_frames=1000,
        fps=30.0,
        duration_s=33.33,
        registration_mode="pixel_only",
        court_geometry=CourtGeometry2D(),
    )
    assert output.total_frames == 1000
    assert output.registration_mode == "pixel_only"
    assert output.player_tracks == []
    assert output.ball_tracks == []
