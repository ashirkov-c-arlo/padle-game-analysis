from __future__ import annotations

import numpy as np
import pytest

from src.schemas import CourtRegistration2D, PlayerDetection, PlayerTrack
from src.tracking.bytetrack import ByteTracker, KalmanBoxTracker, _iou_batch
from src.tracking.identity import assign_player_identities, stabilize_identities
from src.tracking.tracker import track_players


@pytest.fixture
def bytetrack_config():
    return {
        "tracking": {
            "method": "bytetrack",
            "min_track_duration_s": 1.0,
            "max_active_players": 4,
        },
        "bytetrack": {
            "track_thresh": 0.5,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.6,
            "track_buffer": 30,
            "match_thresh": 0.8,
            "min_box_area": 100,
            "frame_rate": 30,
        },
    }


@pytest.fixture
def sample_detections():
    """Four players in a padel court scenario."""
    return [
        # Near-left player
        PlayerDetection(frame=0, bbox_xyxy=(100.0, 600.0, 180.0, 800.0), confidence=0.9),
        # Near-right player
        PlayerDetection(frame=0, bbox_xyxy=(500.0, 620.0, 580.0, 820.0), confidence=0.85),
        # Far-left player
        PlayerDetection(frame=0, bbox_xyxy=(120.0, 100.0, 200.0, 280.0), confidence=0.8),
        # Far-right player
        PlayerDetection(frame=0, bbox_xyxy=(480.0, 110.0, 560.0, 290.0), confidence=0.75),
    ]


class TestIoUComputation:
    def test_identical_boxes(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        b = np.array([[0, 0, 10, 10]], dtype=np.float64)
        iou = _iou_batch(a, b)
        assert iou[0, 0] == pytest.approx(1.0, abs=1e-6)

    def test_no_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        b = np.array([[20, 20, 30, 30]], dtype=np.float64)
        iou = _iou_batch(a, b)
        assert iou[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_partial_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        b = np.array([[5, 5, 15, 15]], dtype=np.float64)
        # Intersection: 5x5=25, union: 100+100-25=175
        iou = _iou_batch(a, b)
        assert iou[0, 0] == pytest.approx(25.0 / 175.0, abs=1e-6)

    def test_empty_input(self):
        a = np.zeros((0, 4), dtype=np.float64)
        b = np.array([[0, 0, 10, 10]], dtype=np.float64)
        iou = _iou_batch(a, b)
        assert iou.shape == (0, 1)

    def test_multiple_boxes(self):
        a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float64)
        b = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float64)
        iou = _iou_batch(a, b)
        assert iou.shape == (2, 2)
        assert iou[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert iou[1, 1] == pytest.approx(1.0, abs=1e-6)
        assert iou[0, 1] == pytest.approx(0.0, abs=1e-6)


class TestKalmanBoxTracker:
    def test_initialization(self):
        bbox = (10.0, 20.0, 50.0, 100.0)
        tracker = KalmanBoxTracker(bbox)
        assert tracker.track_id > 0
        assert tracker.hits == 1
        assert tracker.time_since_update == 0

    def test_predict_returns_bbox(self):
        bbox = (10.0, 20.0, 50.0, 100.0)
        tracker = KalmanBoxTracker(bbox)
        predicted = tracker.predict()
        assert len(predicted) == 4
        # Predicted should be close to original (no velocity yet)
        assert abs(predicted[0] - 10.0) < 5.0
        assert abs(predicted[2] - 50.0) < 5.0

    def test_update_resets_time_since_update(self):
        bbox = (10.0, 20.0, 50.0, 100.0)
        tracker = KalmanBoxTracker(bbox)
        tracker.predict()
        assert tracker.time_since_update == 1
        tracker.update((12.0, 22.0, 52.0, 102.0))
        assert tracker.time_since_update == 0
        assert tracker.hits == 2


class TestByteTrackerInit:
    def test_default_initialization(self, bytetrack_config):
        tracker = ByteTracker(bytetrack_config)
        assert tracker._track_high_thresh == 0.5
        assert tracker._match_thresh == 0.8
        assert tracker._track_buffer == 30

    def test_empty_config_uses_defaults(self):
        tracker = ByteTracker({})
        assert tracker._track_high_thresh == 0.5
        assert tracker._match_thresh == 0.8
        assert tracker._track_buffer == 30

    def test_custom_config(self):
        config = {
            "bytetrack": {
                "track_thresh": 0.6,
                "match_thresh": 0.7,
                "track_buffer": 50,
            }
        }
        tracker = ByteTracker(config)
        assert tracker._track_high_thresh == 0.6
        assert tracker._match_thresh == 0.7
        assert tracker._track_buffer == 50


class TestByteTrackerUpdate:
    def test_single_frame_creates_tracks(self, bytetrack_config, sample_detections):
        tracker = ByteTracker(bytetrack_config)
        result = tracker.update(sample_detections, frame_idx=0)
        # Should create tracks for high-confidence detections
        assert len(result) > 0
        for r in result:
            assert "track_id" in r
            assert "bbox_xyxy" in r
            assert "confidence" in r
            assert "frame" in r

    def test_empty_detections(self, bytetrack_config):
        tracker = ByteTracker(bytetrack_config)
        result = tracker.update([], frame_idx=0)
        assert result == []

    def test_low_confidence_filtered(self, bytetrack_config):
        tracker = ByteTracker(bytetrack_config)
        dets = [
            PlayerDetection(frame=0, bbox_xyxy=(100.0, 100.0, 200.0, 300.0), confidence=0.3),
        ]
        result = tracker.update(dets, frame_idx=0)
        # Confidence 0.3 is below new_track_thresh (0.6), no new track
        assert len(result) == 0

    def test_small_box_filtered(self, bytetrack_config):
        tracker = ByteTracker(bytetrack_config)
        dets = [
            # Area = 5*5 = 25, below min_box_area (100)
            PlayerDetection(frame=0, bbox_xyxy=(100.0, 100.0, 105.0, 105.0), confidence=0.9),
        ]
        result = tracker.update(dets, frame_idx=0)
        assert len(result) == 0


class TestMultiFrameTracking:
    def test_consistent_ids_across_frames(self, bytetrack_config):
        tracker = ByteTracker(bytetrack_config)

        # Frame 0: player at (100, 100, 200, 300)
        dets_0 = [PlayerDetection(frame=0, bbox_xyxy=(100.0, 100.0, 200.0, 300.0), confidence=0.9)]
        result_0 = tracker.update(dets_0, frame_idx=0)
        assert len(result_0) == 1
        track_id_0 = result_0[0]["track_id"]

        # Frame 1: same player moved slightly
        dets_1 = [PlayerDetection(frame=1, bbox_xyxy=(105.0, 102.0, 205.0, 302.0), confidence=0.88)]
        result_1 = tracker.update(dets_1, frame_idx=1)
        assert len(result_1) == 1
        track_id_1 = result_1[0]["track_id"]

        # Should maintain same track ID
        assert track_id_0 == track_id_1

    def test_multiple_players_tracked(self, bytetrack_config):
        tracker = ByteTracker(bytetrack_config)

        # Two distinct players
        dets = [
            PlayerDetection(frame=0, bbox_xyxy=(100.0, 100.0, 200.0, 300.0), confidence=0.9),
            PlayerDetection(frame=0, bbox_xyxy=(500.0, 100.0, 600.0, 300.0), confidence=0.85),
        ]
        result = tracker.update(dets, frame_idx=0)
        assert len(result) == 2
        ids = {r["track_id"] for r in result}
        assert len(ids) == 2  # Different IDs

    def test_get_tracks_accumulates_history(self, bytetrack_config):
        tracker = ByteTracker(bytetrack_config)

        for i in range(5):
            dets = [
                PlayerDetection(
                    frame=i,
                    bbox_xyxy=(100.0 + i * 2, 100.0, 200.0 + i * 2, 300.0),
                    confidence=0.9,
                )
            ]
            tracker.update(dets, frame_idx=i)

        tracks = tracker.get_tracks()
        assert len(tracks) >= 1
        # The main track should have 5 observations
        longest_track = max(tracks.values(), key=len)
        assert len(longest_track) == 5

    def test_lost_track_recovery(self, bytetrack_config):
        tracker = ByteTracker(bytetrack_config)

        # Player appears
        dets_0 = [PlayerDetection(frame=0, bbox_xyxy=(100.0, 100.0, 200.0, 300.0), confidence=0.9)]
        result_0 = tracker.update(dets_0, frame_idx=0)
        track_id = result_0[0]["track_id"]

        # Player disappears for a few frames
        for i in range(1, 5):
            tracker.update([], frame_idx=i)

        # Player reappears at similar position
        dets_5 = [PlayerDetection(frame=5, bbox_xyxy=(102.0, 101.0, 202.0, 301.0), confidence=0.88)]
        result_5 = tracker.update(dets_5, frame_idx=5)

        # Should recover the same track
        assert len(result_5) == 1
        assert result_5[0]["track_id"] == track_id


class TestIdentityAssignment:
    def test_pixel_only_assignment(self):
        """Test identity assignment in pixel_only mode."""
        # Tracks with known positions (normalized: y>0.5 = near, x<0.5 = left)
        tracks = {
            1: [{"frame": 0, "bbox_xyxy": (100.0, 700.0, 200.0, 900.0), "confidence": 0.9}],  # near-left
            2: [{"frame": 0, "bbox_xyxy": (600.0, 720.0, 700.0, 920.0), "confidence": 0.85}],  # near-right
            3: [{"frame": 0, "bbox_xyxy": (100.0, 50.0, 200.0, 200.0), "confidence": 0.8}],   # far-left
            4: [{"frame": 0, "bbox_xyxy": (600.0, 60.0, 700.0, 210.0), "confidence": 0.75}],  # far-right
        }

        result = assign_player_identities(tracks, None, (1000, 1000))
        assert len(result) == 4

        ids = {t.player_id for t in result}
        assert ids == {"near_left", "near_right", "far_left", "far_right"}

    def test_homography_assignment(self):
        """Test identity assignment with floor homography."""
        # Identity homography (pixel coords = court coords for simplicity)
        reg = CourtRegistration2D(
            mode="floor_homography",
            homography_image_to_court=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            confidence=0.95,
        )

        # Footpoints in court coordinates:
        # near_left: foot at (2, 15) -> y>10 = near, x<5 = left
        # near_right: foot at (7, 16) -> y>10 = near, x>=5 = right
        # far_left: foot at (3, 5) -> y<10 = far, x<5 = left
        # far_right: foot at (8, 4) -> y<10 = far, x>=5 = right
        tracks = {
            1: [{"frame": 0, "bbox_xyxy": (1.0, 13.0, 3.0, 15.0), "confidence": 0.9}],
            2: [{"frame": 0, "bbox_xyxy": (6.0, 14.0, 8.0, 16.0), "confidence": 0.85}],
            3: [{"frame": 0, "bbox_xyxy": (2.0, 3.0, 4.0, 5.0), "confidence": 0.8}],
            4: [{"frame": 0, "bbox_xyxy": (7.0, 2.0, 9.0, 4.0), "confidence": 0.75}],
        }

        result = assign_player_identities(tracks, reg, (1080, 1920))
        assert len(result) == 4

        ids = {t.player_id for t in result}
        assert ids == {"near_left", "near_right", "far_left", "far_right"}

        # Check team assignment
        for track in result:
            if "near" in track.player_id:
                assert track.team == "near"
            else:
                assert track.team == "far"

    def test_empty_tracks(self):
        result = assign_player_identities({}, None, (1080, 1920))
        assert result == []

    def test_single_track(self):
        tracks = {
            1: [{"frame": 0, "bbox_xyxy": (100.0, 700.0, 200.0, 900.0), "confidence": 0.9}],
        }
        result = assign_player_identities(tracks, None, (1000, 1000))
        assert len(result) == 1
        assert result[0].team == "near"

    def test_median_position_stability(self):
        """Median position should be robust to outliers."""
        # Track with mostly left-side positions and one outlier on right
        observations = [
            {"frame": i, "bbox_xyxy": (100.0, 700.0, 200.0, 900.0), "confidence": 0.9}
            for i in range(9)
        ]
        # Add one outlier on the right side
        observations.append({"frame": 9, "bbox_xyxy": (800.0, 700.0, 900.0, 900.0), "confidence": 0.9})

        tracks = {1: observations}
        result = assign_player_identities(tracks, None, (1000, 1000))
        assert len(result) == 1
        # Should still be assigned "left" due to median
        assert "left" in result[0].player_id


class TestStabilizeIdentities:
    def test_removes_short_tracks(self):
        """Tracks shorter than min_duration_s should be removed."""
        short_track = PlayerTrack(
            player_id="near_left",
            frames=[0, 1, 2],  # 3 frames at 30fps = 0.1s
            bboxes=[(100.0, 100.0, 200.0, 300.0)] * 3,
            confidences=[0.9] * 3,
            team="near",
        )
        long_track = PlayerTrack(
            player_id="near_right",
            frames=list(range(60)),  # 60 frames at 30fps = 2s
            bboxes=[(500.0, 100.0, 600.0, 300.0)] * 60,
            confidences=[0.85] * 60,
            team="near",
        )

        result = stabilize_identities([short_track, long_track], min_duration_s=1.0, fps=30.0)
        assert len(result) == 1
        assert result[0].player_id == "near_right"

    def test_resolves_duplicate_ids_keeps_longest(self):
        """When multiple tracks claim same ID, keep the longest."""
        track_a = PlayerTrack(
            player_id="near_left",
            frames=list(range(100)),
            bboxes=[(100.0, 100.0, 200.0, 300.0)] * 100,
            confidences=[0.9] * 100,
            team="near",
        )
        track_b = PlayerTrack(
            player_id="near_left",
            frames=list(range(200, 250)),
            bboxes=[(120.0, 110.0, 220.0, 310.0)] * 50,
            confidences=[0.85] * 50,
            team="near",
        )

        result = stabilize_identities([track_a, track_b], min_duration_s=1.0, fps=30.0)
        assert len(result) == 1
        assert result[0].player_id == "near_left"

    def test_merges_fragments(self):
        """Adjacent fragments with small gap should be merged."""
        track_a = PlayerTrack(
            player_id="near_left",
            frames=list(range(0, 50)),
            bboxes=[(100.0, 100.0, 200.0, 300.0)] * 50,
            confidences=[0.9] * 50,
            team="near",
        )
        # Gap of 10 frames (0.33s at 30fps) - should merge
        track_b = PlayerTrack(
            player_id="near_left",
            frames=list(range(60, 110)),
            bboxes=[(105.0, 102.0, 205.0, 302.0)] * 50,
            confidences=[0.88] * 50,
            team="near",
        )

        result = stabilize_identities([track_a, track_b], min_duration_s=1.0, fps=30.0)
        assert len(result) == 1
        assert len(result[0].frames) == 100  # merged

    def test_no_merge_large_gap(self):
        """Fragments with large gap should not merge - keep longest."""
        track_a = PlayerTrack(
            player_id="near_left",
            frames=list(range(0, 50)),
            bboxes=[(100.0, 100.0, 200.0, 300.0)] * 50,
            confidences=[0.9] * 50,
            team="near",
        )
        # Gap of 100 frames (3.3s at 30fps) - too large
        track_b = PlayerTrack(
            player_id="near_left",
            frames=list(range(150, 200)),
            bboxes=[(105.0, 102.0, 205.0, 302.0)] * 50,
            confidences=[0.88] * 50,
            team="near",
        )

        result = stabilize_identities([track_a, track_b], min_duration_s=1.0, fps=30.0)
        assert len(result) == 1
        # Should keep the one with better score (both 50 frames, track_a has higher conf)
        assert len(result[0].frames) == 50

    def test_empty_input(self):
        result = stabilize_identities([], min_duration_s=1.0, fps=30.0)
        assert result == []


class TestTrackPlayers:
    def test_full_pipeline(self, bytetrack_config):
        """End-to-end tracking pipeline test."""
        # Simulate 60 frames (2 seconds at 30fps) with 4 players
        detections: dict[int, list[PlayerDetection]] = {}
        for i in range(60):
            detections[i] = [
                # Near-left: bottom-left of image
                PlayerDetection(
                    frame=i,
                    bbox_xyxy=(100.0 + i * 0.5, 600.0, 200.0 + i * 0.5, 800.0),
                    confidence=0.9,
                ),
                # Near-right: bottom-right of image
                PlayerDetection(
                    frame=i,
                    bbox_xyxy=(600.0 - i * 0.3, 620.0, 700.0 - i * 0.3, 820.0),
                    confidence=0.85,
                ),
                # Far-left: top-left of image
                PlayerDetection(
                    frame=i,
                    bbox_xyxy=(120.0 + i * 0.2, 50.0, 200.0 + i * 0.2, 200.0),
                    confidence=0.8,
                ),
                # Far-right: top-right of image
                PlayerDetection(
                    frame=i,
                    bbox_xyxy=(580.0 - i * 0.1, 60.0, 660.0 - i * 0.1, 210.0),
                    confidence=0.75,
                ),
            ]

        result = track_players(
            video_path="test_video.mp4",
            detections=detections,
            config=bytetrack_config,
            registration=None,
            fps=30.0,
            image_shape=(1000, 1000),
        )

        assert len(result) <= 4
        assert all(isinstance(t, PlayerTrack) for t in result)
        # All tracks should have semantic IDs
        for track in result:
            assert track.player_id in ("near_left", "near_right", "far_left", "far_right")
            assert track.team in ("near", "far")

    def test_sparse_detections(self, bytetrack_config):
        """Test with sparse detections (not every frame)."""
        detections: dict[int, list[PlayerDetection]] = {}
        for i in range(0, 90, 3):  # every 3rd frame
            detections[i] = [
                PlayerDetection(
                    frame=i,
                    bbox_xyxy=(100.0, 600.0, 200.0, 800.0),
                    confidence=0.9,
                ),
            ]

        result = track_players(
            video_path="test_video.mp4",
            detections=detections,
            config=bytetrack_config,
            registration=None,
            fps=30.0,
            image_shape=(1000, 1000),
        )

        assert len(result) >= 1

    def test_empty_detections(self, bytetrack_config):
        """Pipeline handles no detections gracefully."""
        result = track_players(
            video_path="test_video.mp4",
            detections={},
            config=bytetrack_config,
            registration=None,
            fps=30.0,
        )
        assert result == []
