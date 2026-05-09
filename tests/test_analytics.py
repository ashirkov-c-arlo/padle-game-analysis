from __future__ import annotations

import numpy as np
import pytest

from src.analytics.heatmap import generate_heatmap
from src.analytics.kinematics import compute_kinematics
from src.analytics.metrics import compute_player_metrics
from src.analytics.zones import (
    classify_formation,
    classify_zone,
    compute_partner_spacing,
    compute_zone_time,
)
from src.coordinates.projection import footpoint_to_court
from src.coordinates.smoothing import clip_impossible_jumps, smooth_trajectory
from src.schemas import (
    CourtGeometry2D,
    CourtRegistration2D,
    PlayerTrack,
)

# --- Fixtures ---


@pytest.fixture
def identity_homography():
    """Identity homography (pixel coords == court coords)."""
    return np.eye(3, dtype=np.float64)


@pytest.fixture
def simple_homography():
    """Simple scale homography: pixel (0,0)-(1000,2000) -> court (0,0)-(10,20)."""
    # Scale: x_court = x_px / 100, y_court = y_px / 100
    H = np.array([
        [0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return H


@pytest.fixture
def geometry():
    return CourtGeometry2D()


@pytest.fixture
def floor_registration(simple_homography):
    H_inv = np.linalg.inv(simple_homography)
    return CourtRegistration2D(
        mode="floor_homography",
        homography_image_to_court=simple_homography.tolist(),
        homography_court_to_image=H_inv.tolist(),
        reprojection_error_px=1.0,
        num_inliers=10,
        confidence=0.9,
    )


@pytest.fixture
def pixel_only_registration():
    return CourtRegistration2D(
        mode="pixel_only",
        confidence=0.5,
    )


# --- footpoint_to_court tests ---


class TestFootpointToCourt:
    def test_with_identity(self, identity_homography):
        # bbox: x1=100, y1=50, x2=200, y2=300
        # footpoint = (150, 300)
        result = footpoint_to_court((100, 50, 200, 300), identity_homography)
        assert result == pytest.approx((150.0, 300.0))

    def test_with_scale_homography(self, simple_homography):
        # bbox: x1=400, y1=800, x2=600, y2=1000
        # footpoint pixel = (500, 1000)
        # court = (500*0.01, 1000*0.01) = (5.0, 10.0)
        result = footpoint_to_court((400, 800, 600, 1000), simple_homography)
        assert result[0] == pytest.approx(5.0, abs=1e-6)
        assert result[1] == pytest.approx(10.0, abs=1e-6)

    def test_zero_bbox(self, identity_homography):
        result = footpoint_to_court((0, 0, 0, 0), identity_homography)
        assert result == pytest.approx((0.0, 0.0))


# --- smooth_trajectory tests ---


class TestSmoothTrajectory:
    def test_removes_jitter(self):
        # Trajectory with small noise around (5, 10)
        rng = np.random.default_rng(42)
        positions = [(5.0 + rng.normal(0, 0.1), 10.0 + rng.normal(0, 0.1)) for _ in range(20)]
        smoothed = smooth_trajectory(positions, fps=30.0)

        # Smoothed should have lower variance than original
        orig_var_x = np.var([p[0] for p in positions])
        smooth_var_x = np.var([p[0] for p in smoothed])
        assert smooth_var_x < orig_var_x

    def test_short_trajectory_passthrough(self):
        positions = [(1.0, 2.0), (1.5, 2.5)]
        smoothed = smooth_trajectory(positions, window_frames=7, fps=30.0)
        # Too short for savgol, should still return valid positions
        assert len(smoothed) == 2

    def test_stays_within_bounds(self):
        # Positions near boundary
        positions = [(9.9, 19.9)] * 10
        smoothed = smooth_trajectory(positions, fps=30.0)
        for x, y in smoothed:
            assert 0.0 <= x <= 10.0
            assert 0.0 <= y <= 20.0


# --- clip_impossible_jumps tests ---


class TestClipImpossibleJumps:
    def test_teleport_removed(self):
        # Normal positions, then a teleport, then back to normal
        positions = [(5.0, 10.0)] * 5 + [(50.0, 50.0)] + [(5.0, 10.0)] * 5
        clipped = clip_impossible_jumps(positions, max_speed_mps=8.0, fps=30.0)

        # The teleported point should be interpolated back
        # At index 5, it should be close to (5.0, 10.0)
        assert clipped[5][0] == pytest.approx(5.0, abs=0.1)
        assert clipped[5][1] == pytest.approx(10.0, abs=0.1)

    def test_normal_movement_preserved(self):
        # Movement at 2 m/s (well within 8 m/s limit)
        fps = 30.0
        speed = 2.0  # m/s
        dist_per_frame = speed / fps
        positions = [(i * dist_per_frame, 10.0) for i in range(20)]
        clipped = clip_impossible_jumps(positions, max_speed_mps=8.0, fps=fps)

        # Should be unchanged
        for orig, clip in zip(positions, clipped):
            assert clip[0] == pytest.approx(orig[0], abs=1e-6)
            assert clip[1] == pytest.approx(orig[1], abs=1e-6)


# --- compute_kinematics tests ---


class TestComputeKinematics:
    def test_stationary(self):
        positions = [(5.0, 10.0)] * 30
        result = compute_kinematics(positions, fps=30.0)
        assert result["distance_m"] == pytest.approx(0.0)
        assert result["avg_speed_mps"] == pytest.approx(0.0)
        assert result["max_speed_mps"] == pytest.approx(0.0)

    def test_constant_speed(self):
        # Moving at exactly 3 m/s along x for 1 second at 30 fps
        fps = 30.0
        speed = 3.0
        dist_per_frame = speed / fps
        positions = [(i * dist_per_frame, 10.0) for i in range(31)]  # 31 points = 30 intervals
        result = compute_kinematics(positions, fps=fps)

        # Total distance = 30 frames * (3/30) m = 3.0 m
        assert result["distance_m"] == pytest.approx(3.0, abs=0.01)
        assert result["avg_speed_mps"] == pytest.approx(3.0, abs=0.01)
        assert result["max_speed_mps"] == pytest.approx(3.0, abs=0.01)

    def test_single_point(self):
        result = compute_kinematics([(5.0, 10.0)], fps=30.0)
        assert result["distance_m"] == 0.0
        assert result["speeds"] == [0.0]


# --- classify_zone tests ---


class TestClassifyZone:
    def test_net_zone_near_team(self, geometry):
        # Near player close to net: y = 11.0, dist from net = 1.0
        assert classify_zone((5.0, 11.0), geometry) == "net"

    def test_net_zone_far_team(self, geometry):
        # Far player close to net: y = 9.0, dist from net = 1.0
        assert classify_zone((5.0, 9.0), geometry) == "net"

    def test_mid_zone(self, geometry):
        # dist from net = 5.0 (between 3.5 and 6.95)
        assert classify_zone((5.0, 15.0), geometry) == "mid"
        assert classify_zone((5.0, 5.0), geometry) == "mid"

    def test_baseline_zone(self, geometry):
        # dist from net = 9.0 (> 6.95)
        assert classify_zone((5.0, 19.0), geometry) == "baseline"
        assert classify_zone((5.0, 1.0), geometry) == "baseline"

    def test_exactly_at_net(self, geometry):
        assert classify_zone((5.0, 10.0), geometry) == "net"

    def test_boundary_net_mid(self, geometry):
        # Exactly at 3.5m from net
        assert classify_zone((5.0, 13.5), geometry) == "net"
        # Just past 3.5m
        assert classify_zone((5.0, 13.6), geometry) == "mid"

    def test_boundary_mid_baseline(self, geometry):
        # Exactly at 6.95m from net
        assert classify_zone((5.0, 16.95), geometry) == "mid"
        # Just past 6.95m
        assert classify_zone((5.0, 17.0), geometry) == "baseline"


# --- classify_formation tests ---


class TestClassifyFormation:
    def test_both_net(self):
        assert classify_formation("net", "net") == "both_net"

    def test_both_mid(self):
        assert classify_formation("mid", "mid") == "both_mid"

    def test_both_baseline(self):
        assert classify_formation("baseline", "baseline") == "both_baseline"

    def test_one_up_one_back_net_baseline(self):
        assert classify_formation("net", "baseline") == "one_up_one_back"
        assert classify_formation("baseline", "net") == "one_up_one_back"

    def test_one_up_one_back_net_mid(self):
        assert classify_formation("net", "mid") == "one_up_one_back"

    def test_one_up_one_back_mid_baseline(self):
        assert classify_formation("mid", "baseline") == "one_up_one_back"


# --- compute_partner_spacing tests ---


class TestComputePartnerSpacing:
    def test_same_position(self):
        assert compute_partner_spacing((5.0, 10.0), (5.0, 10.0)) == pytest.approx(0.0)

    def test_horizontal_distance(self):
        assert compute_partner_spacing((2.0, 10.0), (8.0, 10.0)) == pytest.approx(6.0)

    def test_diagonal_distance(self):
        assert compute_partner_spacing((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)


# --- compute_zone_time tests ---


class TestComputeZoneTime:
    def test_fractions_sum_to_one(self, geometry):
        # Mix of positions across zones
        positions = [
            (5.0, 10.5),  # net (0.5m from net)
            (5.0, 11.0),  # net (1.0m from net)
            (5.0, 14.0),  # mid (4.0m from net)
            (5.0, 15.0),  # mid (5.0m from net)
            (5.0, 18.0),  # baseline (8.0m from net)
        ]
        result = compute_zone_time(positions, "near_left", geometry, fps=30.0)
        total = result["net"] + result["mid"] + result["baseline"]
        assert total == pytest.approx(1.0)

    def test_all_in_one_zone(self, geometry):
        positions = [(5.0, 10.5)] * 10  # all net zone
        result = compute_zone_time(positions, "near_left", geometry, fps=30.0)
        assert result["net"] == pytest.approx(1.0)
        assert result["mid"] == pytest.approx(0.0)
        assert result["baseline"] == pytest.approx(0.0)

    def test_empty_positions(self, geometry):
        result = compute_zone_time([], "near_left", geometry, fps=30.0)
        assert result == {"net": 0.0, "mid": 0.0, "baseline": 0.0}


# --- compute_player_metrics tests ---


class TestComputePlayerMetrics:
    def test_pixel_only_returns_empty(self, pixel_only_registration, geometry):
        tracks = [
            PlayerTrack(
                player_id="near_left",
                frames=[0, 1, 2],
                bboxes=[(100, 100, 200, 200)] * 3,
                confidences=[0.9] * 3,
                team="near",
            )
        ]
        result = compute_player_metrics(
            tracks, pixel_only_registration, geometry, {}, fps=30.0
        )
        assert result == {}

    def test_floor_homography_returns_metrics(self, floor_registration, geometry):
        # Bboxes at pixel center (500, 1500) -> court (5, 15) which is mid zone
        tracks = [
            PlayerTrack(
                player_id="near_left",
                frames=list(range(20)),
                bboxes=[(400, 1400, 600, 1500)] * 20,
                confidences=[0.9] * 20,
                team="near",
            ),
            PlayerTrack(
                player_id="near_right",
                frames=list(range(20)),
                bboxes=[(700, 1400, 900, 1500)] * 20,
                confidences=[0.9] * 20,
                team="near",
            ),
        ]
        config = {"smoothing": {"method": "savgol", "window_frames": 7, "polyorder": 3, "max_speed_mps": 8.0}}
        result = compute_player_metrics(
            tracks, floor_registration, geometry, config, fps=30.0
        )
        assert "kinematics" in result
        assert "zone_times" in result
        assert "near_left" in result["kinematics"]
        assert "near_right" in result["kinematics"]


# --- generate_heatmap tests ---


class TestGenerateHeatmap:
    def test_shape_correct(self):
        positions = [(5.0, 10.0)] * 100
        heatmap = generate_heatmap(positions, resolution=0.5)
        # 10m / 0.5 = 20 cols, 20m / 0.5 = 40 rows
        assert heatmap.shape == (40, 20)

    def test_normalized_range(self):
        positions = [(5.0, 10.0), (3.0, 5.0), (7.0, 15.0)]
        heatmap = generate_heatmap(positions, resolution=1.0)
        assert heatmap.max() <= 1.0
        assert heatmap.min() >= 0.0

    def test_empty_positions(self):
        heatmap = generate_heatmap([], resolution=0.5)
        assert heatmap.shape == (40, 20)
        assert heatmap.sum() == 0.0

    def test_different_resolution(self):
        positions = [(5.0, 10.0)]
        heatmap = generate_heatmap(positions, resolution=1.0)
        # 10m / 1.0 = 10 cols, 20m / 1.0 = 20 rows
        assert heatmap.shape == (20, 10)
