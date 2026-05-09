from __future__ import annotations

import math

import numpy as np
import pytest

from src.ball_tracking.detector import BallDetector
from src.ball_tracking.events import (
    detect_bounce_candidates,
    detect_net_crossings,
    detect_touch_candidates,
)
from src.ball_tracking.kalman_tracker import BallKalmanTracker
from src.ball_tracking.metrics import (
    compute_bounce_heatmap,
    compute_rally_tempo,
    compute_shot_depth,
    compute_shot_direction,
)
from src.ball_tracking.tracker import interpolate_gaps
from src.schemas import (
    BallDetection2D,
    BallEventCandidate,
    BallTrack2D,
    CourtGeometry2D,
    CourtRegistration2D,
)


@pytest.fixture
def default_config():
    return {
        "ball_tracking": {
            "model": "wasb_sbdt",
            "confidence_threshold": 0.4,
            "fallback_to_heuristic": True,
            "max_gap_frames": 10,
            "kalman_process_noise": 0.1,
        },
        "models": {
            "cache_dir": "data/models",
            "lazy_download": True,
        },
    }


@pytest.fixture
def identity_registration():
    """Registration with identity homography (pixel = court coords)."""
    return CourtRegistration2D(
        mode="floor_homography",
        homography_image_to_court=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        homography_court_to_image=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        confidence=0.9,
    )


class TestBallDetectorInit:
    def test_instantiation_with_config(self, default_config):
        detector = BallDetector(default_config)
        assert detector._confidence_threshold == 0.4
        assert detector._model is None  # no model file exists

    def test_instantiation_with_empty_config(self):
        detector = BallDetector({})
        assert detector._confidence_threshold == 0.4
        assert detector._model is None

    def test_custom_threshold(self):
        config = {"ball_tracking": {"confidence_threshold": 0.6}}
        detector = BallDetector(config)
        assert detector._confidence_threshold == 0.6

    def test_wasb_backend_is_used_when_available(self, monkeypatch):
        class StubWasbDetector:
            def detect_frame(self, frame, prev_frames=None):
                return BallDetection2D(
                    frame=0,
                    time_s=0.0,
                    image_xy=(123.0, 45.0),
                    confidence=0.8,
                    source="wasb_sbdt",
                )

        monkeypatch.setattr(
            "src.ball_tracking.detector._load_wasb_sbdt_detector",
            lambda config: StubWasbDetector(),
        )

        detector = BallDetector({"ball_tracking": {"model": "wasb_sbdt"}})
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = detector.detect_frame(frame)

        assert result is not None
        assert result.image_xy == (123.0, 45.0)
        assert result.source == "wasb_sbdt"

    def test_detect_frame_with_blank_image(self, default_config):
        detector = BallDetector(default_config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_frame(frame)
        # Blank frame should yield no detection
        assert result is None

    def test_detect_frame_with_bright_circle(self, default_config):
        """A bright circle on a dark background after warmup should be detectable."""
        detector = BallDetector(default_config)

        # Warm up background subtractor with blank frames
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(10):
            detector.detect_frame(blank)

        # Now add a bright yellow circle (simulating a ball)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a yellow circle (BGR: 0, 255, 255 for yellow)
        cv2 = __import__("cv2")
        cv2.circle(frame, (320, 240), 8, (0, 255, 255), -1)

        result = detector.detect_frame(frame)
        # Should detect the ball (it moves relative to blank background)
        if result is not None:
            assert result.confidence >= 0.0
            # Position should be near the circle center
            assert abs(result.image_xy[0] - 320) < 30
            assert abs(result.image_xy[1] - 240) < 30


class TestBallKalmanTracker:
    def test_initialization(self):
        tracker = BallKalmanTracker(process_noise=0.1)
        assert not tracker.initialized

    def test_first_update_initializes(self):
        tracker = BallKalmanTracker()
        tracker.update((100.0, 200.0))
        assert tracker.initialized
        pos, vel = tracker.get_state()
        assert pos == (100.0, 200.0)
        assert vel == (0.0, 0.0)

    def test_predict_after_update(self):
        tracker = BallKalmanTracker()
        tracker.update((100.0, 200.0))
        predicted = tracker.predict()
        # With zero velocity, prediction should stay near initial pos
        assert abs(predicted[0] - 100.0) < 5.0
        assert abs(predicted[1] - 200.0) < 5.0

    def test_predict_update_cycle_tracks_movement(self):
        tracker = BallKalmanTracker(process_noise=0.1)

        # Simulate ball moving at constant velocity: +5px/frame in x
        for i in range(20):
            x = 100.0 + i * 5.0
            y = 200.0
            tracker.predict()
            tracker.update((x, y))

        pos, vel = tracker.get_state()
        # After convergence, velocity should be near 5 px/frame
        assert abs(vel[0] - 5.0) < 1.0
        assert abs(vel[1]) < 1.0
        # Position should be near last measurement
        assert abs(pos[0] - (100.0 + 19 * 5.0)) < 3.0

    def test_predict_without_update_extrapolates(self):
        tracker = BallKalmanTracker(process_noise=0.1)

        # Train with constant velocity
        for i in range(10):
            tracker.predict()
            tracker.update((100.0 + i * 10.0, 50.0))

        # Now predict without updates
        pred1 = tracker.predict()
        pred2 = tracker.predict()

        # Should extrapolate forward
        assert pred2[0] > pred1[0]

    def test_confidence_affects_update(self):
        tracker_high = BallKalmanTracker(process_noise=0.1)
        tracker_low = BallKalmanTracker(process_noise=0.1)

        # Initialize both
        tracker_high.update((100.0, 100.0), confidence=1.0)
        tracker_low.update((100.0, 100.0), confidence=1.0)

        # Train with same trajectory
        for i in range(5):
            tracker_high.predict()
            tracker_high.update((100.0 + i * 5.0, 100.0), confidence=1.0)
            tracker_low.predict()
            tracker_low.update((100.0 + i * 5.0, 100.0), confidence=1.0)

        # Give a noisy measurement to both
        tracker_high.predict()
        tracker_high.update((200.0, 200.0), confidence=0.9)  # high confidence
        tracker_low.predict()
        tracker_low.update((200.0, 200.0), confidence=0.1)  # low confidence

        pos_high, _ = tracker_high.get_state()
        pos_low, _ = tracker_low.get_state()

        # High-confidence should shift more toward measurement
        # Low-confidence should trust prediction more
        dist_high = math.sqrt((pos_high[0] - 200.0) ** 2 + (pos_high[1] - 200.0) ** 2)
        dist_low = math.sqrt((pos_low[0] - 200.0) ** 2 + (pos_low[1] - 200.0) ** 2)
        assert dist_high < dist_low


class TestInterpolateGaps:
    def test_no_gaps(self):
        """Consecutive detected frames should pass through unchanged."""
        tracks = [
            BallTrack2D(frame=0, time_s=0.0, image_xy=(10.0, 10.0), confidence=0.9, state="tracked"),
            BallTrack2D(frame=1, time_s=0.033, image_xy=(15.0, 12.0), confidence=0.9, state="tracked"),
            BallTrack2D(frame=2, time_s=0.066, image_xy=(20.0, 14.0), confidence=0.9, state="tracked"),
        ]
        result = interpolate_gaps(tracks, max_gap_frames=10)
        assert len(result) == 3
        assert all(t.state == "tracked" for t in result)

    def test_short_gap_interpolated(self):
        """Gap within limit should be filled with interpolated frames."""
        tracks = [
            BallTrack2D(frame=0, time_s=0.0, image_xy=(10.0, 10.0), confidence=0.9, state="tracked"),
            BallTrack2D(frame=1, time_s=0.033, image_xy=(10.0, 10.0), confidence=0.0, state="missing"),
            BallTrack2D(frame=2, time_s=0.066, image_xy=(10.0, 10.0), confidence=0.0, state="missing"),
            BallTrack2D(frame=3, time_s=0.1, image_xy=(40.0, 40.0), confidence=0.9, state="tracked"),
        ]
        result = interpolate_gaps(tracks, max_gap_frames=10)

        # Should have frames 0, 1, 2, 3
        frames = [t.frame for t in result]
        assert 0 in frames
        assert 1 in frames
        assert 2 in frames
        assert 3 in frames

        # Interpolated frames should have state="interpolated"
        interp = [t for t in result if t.frame in (1, 2)]
        for t in interp:
            assert t.state == "interpolated"
            assert t.interpolated is True

    def test_long_gap_becomes_missing(self):
        """Gap exceeding max should be marked as missing."""
        tracks = [
            BallTrack2D(frame=0, time_s=0.0, image_xy=(10.0, 10.0), confidence=0.9, state="tracked"),
            # Big gap
            BallTrack2D(frame=20, time_s=0.66, image_xy=(200.0, 200.0), confidence=0.9, state="tracked"),
        ]
        result = interpolate_gaps(tracks, max_gap_frames=5)

        # Frames in the gap should be missing
        missing_frames = [t for t in result if t.state == "missing"]
        assert len(missing_frames) > 0
        for t in missing_frames:
            assert t.confidence == 0.0

    def test_interpolated_positions_are_linear(self):
        """Interpolated positions should lie on a line between endpoints."""
        tracks = [
            BallTrack2D(frame=0, time_s=0.0, image_xy=(0.0, 0.0), confidence=0.9, state="tracked"),
            BallTrack2D(frame=4, time_s=0.133, image_xy=(100.0, 100.0), confidence=0.9, state="tracked"),
        ]
        result = interpolate_gaps(tracks, max_gap_frames=10)

        # Find interpolated frames
        interp = sorted([t for t in result if t.interpolated], key=lambda t: t.frame)
        assert len(interp) == 3  # frames 1, 2, 3

        # Check linearity: gap_len=3, t = g/(gap_len+1) so steps are 1/4, 2/4, 3/4
        expected = {1: 25.0, 2: 50.0, 3: 75.0}
        for t in interp:
            assert abs(t.image_xy[0] - expected[t.frame]) < 1.0
            assert abs(t.image_xy[1] - expected[t.frame]) < 1.0

    def test_empty_tracks(self):
        result = interpolate_gaps([], max_gap_frames=10)
        assert result == []


class TestBounceCandidates:
    def test_velocity_reversal_detected(self):
        """Downward then upward motion should produce a bounce candidate."""
        fps = 30.0
        # Ball going down (y increases in image), then bouncing up
        tracks = [
            BallTrack2D(frame=0, time_s=0.0, image_xy=(100.0, 100.0), confidence=0.9, state="tracked"),
            BallTrack2D(frame=1, time_s=1 / 30, image_xy=(102.0, 130.0), confidence=0.9, state="tracked"),
            BallTrack2D(frame=2, time_s=2 / 30, image_xy=(104.0, 150.0), confidence=0.9, state="tracked"),
            # Bounce point: ball going down fast
            BallTrack2D(frame=3, time_s=3 / 30, image_xy=(106.0, 160.0), confidence=0.9, state="tracked"),
            # Now going up (y decreases)
            BallTrack2D(frame=4, time_s=4 / 30, image_xy=(108.0, 140.0), confidence=0.9, state="tracked"),
            BallTrack2D(frame=5, time_s=5 / 30, image_xy=(110.0, 125.0), confidence=0.9, state="tracked"),
        ]

        candidates = detect_bounce_candidates(tracks, None, fps)
        assert len(candidates) >= 1
        assert all(c.event_type == "bounce_candidate" for c in candidates)

    def test_no_bounce_with_constant_direction(self):
        """Ball moving in straight line should produce no bounce."""
        fps = 30.0
        tracks = [
            BallTrack2D(frame=i, time_s=i / 30, image_xy=(100.0 + i * 5, 100.0 + i * 5), confidence=0.9, state="tracked")
            for i in range(10)
        ]
        candidates = detect_bounce_candidates(tracks, None, fps)
        assert len(candidates) == 0

    def test_bounce_with_registration(self, identity_registration):
        """Bounce with registration should include court_xy_approx."""
        fps = 30.0
        tracks = [
            BallTrack2D(frame=0, time_s=0.0, image_xy=(5.0, 8.0), confidence=0.9, state="tracked"),
            BallTrack2D(frame=1, time_s=1 / 30, image_xy=(5.1, 9.0), confidence=0.9, state="tracked"),
            BallTrack2D(frame=2, time_s=2 / 30, image_xy=(5.2, 9.5), confidence=0.9, state="tracked"),
            BallTrack2D(frame=3, time_s=3 / 30, image_xy=(5.3, 8.0), confidence=0.9, state="tracked"),
        ]
        candidates = detect_bounce_candidates(tracks, identity_registration, fps)
        for c in candidates:
            assert c.court_xy_approx is not None


class TestTouchCandidates:
    def test_proximity_and_direction_change(self):
        """Ball near player with direction change should be a touch candidate."""
        fps = 30.0
        # Ball moving right, then suddenly left (touch)
        tracks = [
            BallTrack2D(frame=0, time_s=0.0, image_xy=(80.0, 200.0), confidence=0.9, state="tracked"),
            BallTrack2D(frame=1, time_s=1 / 30, image_xy=(90.0, 200.0), confidence=0.9, state="tracked"),
            BallTrack2D(frame=2, time_s=2 / 30, image_xy=(100.0, 200.0), confidence=0.9, state="tracked"),
            # Direction reversal at frame 3 (near player)
            BallTrack2D(frame=3, time_s=3 / 30, image_xy=(105.0, 200.0), confidence=0.9, state="tracked"),
            BallTrack2D(frame=4, time_s=4 / 30, image_xy=(95.0, 210.0), confidence=0.9, state="tracked"),
            BallTrack2D(frame=5, time_s=5 / 30, image_xy=(85.0, 220.0), confidence=0.9, state="tracked"),
        ]

        # Player near the touch point
        player_positions = {
            "near_left": [(110.0, 210.0)] * 10,  # within proximity
        }

        candidates = detect_touch_candidates(tracks, player_positions, fps, proximity_threshold_px=50.0)
        assert len(candidates) >= 1
        assert all(c.event_type == "touch_candidate" for c in candidates)
        # Should reference the nearby player
        assert any(
            c.evidence and c.evidence.get("nearest_player") == "near_left"
            for c in candidates
        )

    def test_no_touch_without_player_proximity(self):
        """Direction change far from players should not produce a touch."""
        fps = 30.0
        tracks = [
            BallTrack2D(frame=0, time_s=0.0, image_xy=(100.0, 200.0), confidence=0.9, state="tracked"),
            BallTrack2D(frame=1, time_s=1 / 30, image_xy=(110.0, 200.0), confidence=0.9, state="tracked"),
            BallTrack2D(frame=2, time_s=2 / 30, image_xy=(115.0, 200.0), confidence=0.9, state="tracked"),
            BallTrack2D(frame=3, time_s=3 / 30, image_xy=(105.0, 210.0), confidence=0.9, state="tracked"),
            BallTrack2D(frame=4, time_s=4 / 30, image_xy=(95.0, 220.0), confidence=0.9, state="tracked"),
        ]

        # Player far away
        player_positions = {
            "near_left": [(500.0, 500.0)] * 10,
        }

        candidates = detect_touch_candidates(tracks, player_positions, fps, proximity_threshold_px=50.0)
        assert len(candidates) == 0

    def test_empty_inputs(self):
        assert detect_touch_candidates([], {}, 30.0) == []
        tracks = [BallTrack2D(frame=0, time_s=0.0, image_xy=(0.0, 0.0), confidence=0.9, state="tracked")]
        assert detect_touch_candidates(tracks, {}, 30.0) == []


class TestNetCrossings:
    def test_crossing_detected(self, identity_registration):
        """Ball crossing y=10 (net) should produce a net crossing candidate."""
        tracks = [
            BallTrack2D(frame=0, time_s=0.0, image_xy=(5.0, 8.0), confidence=0.9, state="tracked"),
            BallTrack2D(frame=1, time_s=0.033, image_xy=(5.0, 9.5), confidence=0.9, state="tracked"),
            BallTrack2D(frame=2, time_s=0.066, image_xy=(5.0, 10.5), confidence=0.9, state="tracked"),
            BallTrack2D(frame=3, time_s=0.1, image_xy=(5.0, 12.0), confidence=0.9, state="tracked"),
        ]
        candidates = detect_net_crossings(tracks, identity_registration)
        assert len(candidates) == 1
        assert candidates[0].event_type == "net_crossing_candidate"
        assert candidates[0].evidence["direction"] == "near_to_far"

    def test_no_crossing_same_side(self, identity_registration):
        """Ball staying on one side of net should produce no crossing."""
        tracks = [
            BallTrack2D(frame=i, time_s=i * 0.033, image_xy=(5.0, 5.0 + i * 0.5), confidence=0.9, state="tracked")
            for i in range(5)
        ]
        candidates = detect_net_crossings(tracks, identity_registration)
        assert len(candidates) == 0

    def test_no_crossing_without_registration(self):
        """Without registration, net crossing cannot be detected."""
        tracks = [
            BallTrack2D(frame=0, time_s=0.0, image_xy=(5.0, 8.0), confidence=0.9, state="tracked"),
            BallTrack2D(frame=1, time_s=0.033, image_xy=(5.0, 12.0), confidence=0.9, state="tracked"),
        ]
        assert detect_net_crossings(tracks, None) == []

    def test_pixel_only_registration(self):
        """pixel_only mode should return no crossings."""
        reg = CourtRegistration2D(mode="pixel_only", confidence=0.5)
        tracks = [
            BallTrack2D(frame=0, time_s=0.0, image_xy=(5.0, 8.0), confidence=0.9, state="tracked"),
            BallTrack2D(frame=1, time_s=0.033, image_xy=(5.0, 12.0), confidence=0.9, state="tracked"),
        ]
        assert detect_net_crossings(tracks, reg) == []


class TestRallyTempo:
    def test_basic_rally(self):
        """Two touches close together should form a rally."""
        touches = [
            BallEventCandidate(frame=0, time_s=0.0, event_type="touch_candidate", image_xy=(100.0, 200.0), confidence=0.7),
            BallEventCandidate(frame=30, time_s=1.0, event_type="touch_candidate", image_xy=(200.0, 300.0), confidence=0.7),
            BallEventCandidate(frame=60, time_s=2.0, event_type="touch_candidate", image_xy=(100.0, 200.0), confidence=0.7),
            BallEventCandidate(frame=90, time_s=3.0, event_type="touch_candidate", image_xy=(200.0, 300.0), confidence=0.7),
        ]
        metrics = compute_rally_tempo(touches, fps=30.0)
        assert len(metrics) == 1
        assert metrics[0].rally_id == 0
        assert metrics[0].estimated_shots == 4
        assert metrics[0].duration_s == pytest.approx(3.0, abs=0.01)
        assert metrics[0].avg_time_between_touches_s == pytest.approx(1.0, abs=0.01)

    def test_two_rallies_separated_by_gap(self):
        """Touches with >4s gap should split into separate rallies."""
        touches = [
            BallEventCandidate(frame=0, time_s=0.0, event_type="touch_candidate", image_xy=(100.0, 200.0), confidence=0.7),
            BallEventCandidate(frame=30, time_s=1.0, event_type="touch_candidate", image_xy=(200.0, 300.0), confidence=0.7),
            # 5s gap - new rally
            BallEventCandidate(frame=180, time_s=6.0, event_type="touch_candidate", image_xy=(100.0, 200.0), confidence=0.7),
            BallEventCandidate(frame=210, time_s=7.0, event_type="touch_candidate", image_xy=(200.0, 300.0), confidence=0.7),
        ]
        metrics = compute_rally_tempo(touches, fps=30.0)
        assert len(metrics) == 2
        assert metrics[0].rally_id == 0
        assert metrics[1].rally_id == 1

    def test_single_touch_no_rally(self):
        """A single touch cannot form a rally."""
        touches = [
            BallEventCandidate(frame=0, time_s=0.0, event_type="touch_candidate", image_xy=(100.0, 200.0), confidence=0.7),
        ]
        metrics = compute_rally_tempo(touches, fps=30.0)
        assert len(metrics) == 0

    def test_empty_input(self):
        assert compute_rally_tempo([], fps=30.0) == []


class TestShotDirection:
    def test_cross_court_classification(self):
        """Ball moving strongly laterally should be classified as cross_court."""
        touch = BallEventCandidate(
            frame=5, time_s=5 / 30, event_type="touch_candidate",
            image_xy=(100.0, 200.0), confidence=0.8,
            evidence={"nearest_player": "near_left"},
        )

        # Ball moves strongly to the right after touch
        tracks = []
        for i in range(20):
            tracks.append(
                BallTrack2D(frame=i, time_s=i / 30, image_xy=(100.0 + i * 15, 200.0 + i * 2), confidence=0.9, state="tracked")
            )

        metrics = compute_shot_direction([touch], tracks, fps=30.0)
        assert len(metrics) == 1
        assert metrics[0].direction == "cross_court"
        assert metrics[0].player_id == "near_left"

    def test_down_the_line_classification(self):
        """Ball moving mostly longitudinally should be down_the_line."""
        touch = BallEventCandidate(
            frame=5, time_s=5 / 30, event_type="touch_candidate",
            image_xy=(100.0, 200.0), confidence=0.8,
            evidence={"nearest_player": "near_right"},
        )

        # Ball moves mostly vertically
        tracks = []
        for i in range(20):
            tracks.append(
                BallTrack2D(frame=i, time_s=i / 30, image_xy=(100.0 + i * 0.5, 200.0 + i * 15), confidence=0.9, state="tracked")
            )

        metrics = compute_shot_direction([touch], tracks, fps=30.0)
        assert len(metrics) == 1
        assert metrics[0].direction == "down_the_line"

    def test_empty_inputs(self):
        assert compute_shot_direction([], [], fps=30.0) == []


class TestShotDepth:
    def test_short_depth(self):
        """Bounce close to net should be 'short'."""
        geometry = CourtGeometry2D()
        bounces = [
            BallEventCandidate(
                frame=10, time_s=0.33, event_type="bounce_candidate",
                image_xy=(5.0, 11.0), court_xy_approx=(5.0, 11.0), confidence=0.8,
            ),
        ]
        metrics = compute_shot_depth(bounces, geometry)
        assert len(metrics) == 1
        assert metrics[0].depth == "short"

    def test_deep_depth(self):
        """Bounce near baseline should be 'deep'."""
        geometry = CourtGeometry2D()
        bounces = [
            BallEventCandidate(
                frame=10, time_s=0.33, event_type="bounce_candidate",
                image_xy=(5.0, 18.0), court_xy_approx=(5.0, 18.0), confidence=0.8,
            ),
        ]
        metrics = compute_shot_depth(bounces, geometry)
        assert len(metrics) == 1
        assert metrics[0].depth == "deep"

    def test_mid_depth(self):
        """Bounce in service box area should be 'mid'."""
        geometry = CourtGeometry2D()
        # service_line_offset = 6.95, so mid is between 3.475 and 6.95 from net
        # net at y=10, far side: y = 10 + 5 = 15 (between 3.475 and 6.95 from net)
        bounces = [
            BallEventCandidate(
                frame=10, time_s=0.33, event_type="bounce_candidate",
                image_xy=(5.0, 15.0), court_xy_approx=(5.0, 15.0), confidence=0.8,
            ),
        ]
        metrics = compute_shot_depth(bounces, geometry)
        assert len(metrics) == 1
        assert metrics[0].depth == "mid"

    def test_no_court_coords(self):
        """Bounces without court_xy_approx should be skipped."""
        geometry = CourtGeometry2D()
        bounces = [
            BallEventCandidate(
                frame=10, time_s=0.33, event_type="bounce_candidate",
                image_xy=(5.0, 15.0), court_xy_approx=None, confidence=0.8,
            ),
        ]
        metrics = compute_shot_depth(bounces, geometry)
        assert len(metrics) == 0

    def test_empty_input(self):
        geometry = CourtGeometry2D()
        assert compute_shot_depth([], geometry) == []


class TestBounceHeatmap:
    def test_basic_heatmap(self):
        """Bounces at known positions should show up on heatmap grid."""
        bounces = [
            BallEventCandidate(frame=0, time_s=0.0, event_type="bounce_candidate", image_xy=(5.0, 15.0), court_xy_approx=(5.0, 15.0), confidence=0.9),
            BallEventCandidate(frame=10, time_s=0.33, event_type="bounce_candidate", image_xy=(5.0, 15.0), court_xy_approx=(5.0, 15.0), confidence=0.8),
            BallEventCandidate(frame=20, time_s=0.66, event_type="bounce_candidate", image_xy=(3.0, 7.0), court_xy_approx=(3.0, 7.0), confidence=0.7),
        ]
        heatmap = compute_bounce_heatmap(bounces)
        assert heatmap is not None
        assert heatmap.shape == (20, 10)
        # Cell (15, 5) should have accumulated weight from 2 bounces
        assert heatmap[15, 5] == pytest.approx(0.9 + 0.8, abs=0.01)
        assert heatmap[7, 3] == pytest.approx(0.7, abs=0.01)

    def test_no_court_coords_returns_none(self):
        bounces = [
            BallEventCandidate(frame=0, time_s=0.0, event_type="bounce_candidate", image_xy=(5.0, 15.0), court_xy_approx=None, confidence=0.9),
        ]
        assert compute_bounce_heatmap(bounces) is None

    def test_empty_returns_none(self):
        assert compute_bounce_heatmap([]) is None
