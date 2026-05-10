from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.schemas import (
    BallDetection2D,
    BallEventCandidate,
    BallTrack2D,
    CourtGeometry2D,
    CourtRegistration2D,
    PlayerMetricFrame,
    PlayerTrack,
    RallyTempoMetric,
    ScoreboardState,
)


@pytest.fixture
def geometry():
    return CourtGeometry2D()


@pytest.fixture
def registration():
    return CourtRegistration2D(mode="pixel_only", confidence=0.0)


@pytest.fixture
def sample_tracks():
    return [
        PlayerTrack(
            player_id="near_left",
            frames=[0, 1, 2],
            bboxes=[
                (100.0, 200.0, 150.0, 300.0),
                (102.0, 201.0, 152.0, 301.0),
                (104.0, 202.0, 154.0, 302.0),
            ],
            confidences=[0.9, 0.85, 0.88],
            team="near",
        ),
        PlayerTrack(
            player_id="far_right",
            frames=[0, 1, 2],
            bboxes=[
                (400.0, 50.0, 450.0, 150.0),
                (401.0, 51.0, 451.0, 151.0),
                (402.0, 52.0, 452.0, 152.0),
            ],
            confidences=[0.92, 0.90, 0.91],
            team="far",
        ),
    ]


@pytest.fixture
def sample_metrics():
    return [
        PlayerMetricFrame(
            frame=0,
            time_s=0.0,
            player_id="near_left",
            court_xy=(3.0, 15.0),
            speed_mps=2.5,
            zone="mid",
            confidence=0.9,
            metric_quality="estimated",
        ),
        PlayerMetricFrame(
            frame=1,
            time_s=0.033,
            player_id="near_left",
            court_xy=(3.1, 15.1),
            speed_mps=2.6,
            zone="mid",
            confidence=0.88,
            metric_quality="estimated",
        ),
    ]


@pytest.fixture
def sample_ball_detections():
    return [
        BallDetection2D(
            frame=0, time_s=0.0, image_xy=(500.0, 300.0), confidence=0.8
        ),
        BallDetection2D(
            frame=1, time_s=0.033, image_xy=(510.0, 290.0), confidence=0.75
        ),
    ]


@pytest.fixture
def sample_ball_tracks():
    return [
        BallTrack2D(
            frame=0, time_s=0.0, image_xy=(500.0, 300.0),
            velocity_px_s=(150.0, -80.0), confidence=0.8, state="tracked",
        ),
        BallTrack2D(
            frame=1, time_s=0.033, image_xy=(510.0, 290.0),
            velocity_px_s=(140.0, -75.0), confidence=0.75, state="tracked",
        ),
        BallTrack2D(
            frame=2, time_s=0.066, image_xy=(520.0, 285.0),
            velocity_px_s=None, confidence=0.5, state="interpolated",
            interpolated=True, gap_len=1,
        ),
    ]


@pytest.fixture
def sample_ball_events():
    return [
        BallEventCandidate(
            frame=5, time_s=0.166, event_type="bounce_candidate",
            image_xy=(550.0, 400.0), confidence=0.6,
        ),
    ]


@pytest.fixture
def sample_scoreboard_states():
    return [
        ScoreboardState(
            frame=0, time_s=0.0, raw_text="6-4 3-2",
            roi_bbox_xyxy=(100, 10, 500, 80),
            parsed_sets=[(6, 4)], parsed_game_score=(3, 2), confidence=0.7,
        ),
        ScoreboardState(
            frame=30, time_s=1.0, raw_text="6-4 3-2",
            roi_bbox_xyxy=(100, 10, 500, 80),
            parsed_sets=[(6, 4)], parsed_game_score=(3, 2), confidence=0.75,
        ),
    ]


@pytest.fixture
def sample_rally_metrics():
    return [
        RallyTempoMetric(
            rally_id=0, duration_s=5.0, estimated_shots=4,
            avg_time_between_touches_s=1.25, median_time_between_touches_s=1.2,
        ),
    ]


def _write_tiny_input_video(path: Path, size: tuple[int, int] = (64, 48), fps: float = 10.0) -> None:
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened()
    for idx in range(3):
        frame = np.full((height, width, 3), idx * 60, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _read_video_fourcc(path: Path) -> str:
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened()
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    ok, _ = cap.read()
    cap.release()
    assert ok
    return "".join(chr((fourcc_int >> 8 * idx) & 0xFF) for idx in range(4))


class TestExportAll:
    """Test export_all creates all expected files."""

    def test_creates_all_files(
        self, geometry, registration, sample_tracks, sample_metrics,
        sample_ball_detections, sample_ball_tracks, sample_ball_events,
        sample_scoreboard_states, sample_rally_metrics,
    ):
        from src.export.writer import export_all

        with tempfile.TemporaryDirectory() as tmpdir:
            summary = {"video_path": "test.mp4", "total_frames": 100}

            export_all(
                output_dir=tmpdir,
                registration=registration,
                geometry=geometry,
                tracks=sample_tracks,
                metric_frames=sample_metrics,
                ball_detections=sample_ball_detections,
                ball_tracks=sample_ball_tracks,
                ball_events=sample_ball_events,
                scoreboard_states=sample_scoreboard_states,
                rally_metrics=sample_rally_metrics,
                summary=summary,
                config={"test": True},
            )

            expected_files = [
                "court_geometry.json",
                "court_registration.json",
                "tracks.csv",
                "metrics.csv",
                "ball_tracks.csv",
                "ball_detections.jsonl",
                "ball_event_candidates.jsonl",
                "scoreboard.csv",
                "rally_metrics.csv",
                "summary.json",
            ]

            for filename in expected_files:
                filepath = Path(tmpdir) / filename
                assert filepath.exists(), f"Missing file: {filename}"
                assert filepath.stat().st_size > 0, f"Empty file: {filename}"


class TestCSVFiles:
    """Test CSV files have correct columns."""

    def test_tracks_csv_columns(self, sample_tracks):
        from src.export.writer import write_tracks_csv

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = Path(f.name)

        try:
            write_tracks_csv(path, sample_tracks)
            with open(path) as f:
                header = f.readline().strip()
            assert header == "frame,player_id,team,x1,y1,x2,y2,confidence"

            # Check data rows exist
            with open(path) as f:
                lines = f.readlines()
            # Header + 3 frames * 2 tracks = 7 lines
            assert len(lines) == 7
        finally:
            os.unlink(path)

    def test_metrics_csv_columns(self, sample_metrics):
        from src.export.writer import write_metrics_csv

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = Path(f.name)

        try:
            write_metrics_csv(path, sample_metrics)
            with open(path) as f:
                header = f.readline().strip()
            assert header == "frame,time_s,player_id,court_x,court_y,speed_mps,zone,confidence,metric_quality"

            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 3  # header + 2 rows
        finally:
            os.unlink(path)

    def test_ball_tracks_csv_columns(self, sample_ball_tracks):
        from src.export.writer import write_ball_tracks_csv

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = Path(f.name)

        try:
            write_ball_tracks_csv(path, sample_ball_tracks)
            with open(path) as f:
                header = f.readline().strip()
            assert header == "frame,time_s,x_px,y_px,vx_px_s,vy_px_s,confidence,state,interpolated,gap_len"

            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 4  # header + 3 rows
        finally:
            os.unlink(path)

    def test_scoreboard_csv_columns(self, sample_scoreboard_states):
        from src.export.writer import write_scoreboard_csv

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = Path(f.name)

        try:
            write_scoreboard_csv(path, sample_scoreboard_states)
            with open(path) as f:
                header = f.readline().strip()
            assert header == "frame,time_s,roi_x1,roi_y1,roi_x2,roi_y2,raw_text,set1_a,set1_b,game_a,game_b,confidence"

            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 3  # header + 2 rows
            assert lines[1].startswith("0,0.0000,100,10,500,80,")
        finally:
            os.unlink(path)

    def test_rally_metrics_csv_columns(self, sample_rally_metrics):
        from src.export.writer import write_rally_metrics_csv

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = Path(f.name)

        try:
            write_rally_metrics_csv(path, sample_rally_metrics)
            with open(path) as f:
                header = f.readline().strip()
            assert header == (
                "rally_id,duration_s,estimated_shots,"
                "avg_time_between_touches_s,median_time_between_touches_s"
            )
        finally:
            os.unlink(path)


class TestJSONFiles:
    """Test JSON files are valid."""

    def test_court_geometry_json(self, geometry):
        from src.export.writer import write_court_geometry

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            write_court_geometry(path, geometry)
            with open(path) as f:
                data = json.load(f)
            assert "width_m" in data
            assert "length_m" in data
            assert "net_y_m" in data
            assert data["width_m"] == 10.0
        finally:
            os.unlink(path)

    def test_court_registration_json(self, registration):
        from src.export.writer import write_court_registration

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            write_court_registration(path, registration)
            with open(path) as f:
                data = json.load(f)
            assert "mode" in data
            assert data["mode"] == "pixel_only"
            assert "confidence" in data
        finally:
            os.unlink(path)

    def test_summary_json(self):
        from src.export.writer import write_summary_json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            summary = {
                "video_path": "test.mp4",
                "total_frames": 900,
                "fps": 30.0,
                "duration_s": 30.0,
                "registration_mode": "pixel_only",
                "pipeline_elapsed_s": 12.5,
            }
            write_summary_json(path, summary)
            with open(path) as f:
                data = json.load(f)
            assert data["video_path"] == "test.mp4"
            assert data["total_frames"] == 900
            assert data["fps"] == 30.0
            assert "duration_s" in data
            assert "registration_mode" in data
            assert "pipeline_elapsed_s" in data
        finally:
            os.unlink(path)

    def test_ball_detections_jsonl(self, sample_ball_detections):
        from src.export.writer import write_ball_detections_jsonl

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        try:
            write_ball_detections_jsonl(path, sample_ball_detections)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2
            for line in lines:
                data = json.loads(line)
                assert "frame" in data
                assert "image_xy" in data
                assert "confidence" in data
        finally:
            os.unlink(path)

    def test_ball_events_jsonl(self, sample_ball_events):
        from src.export.writer import write_ball_events_jsonl

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        try:
            write_ball_events_jsonl(path, sample_ball_events)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["event_type"] == "bounce_candidate"
            assert "frame" in data
            assert "confidence" in data
        finally:
            os.unlink(path)


class TestSummaryFields:
    """Test summary has required fields."""

    def test_summary_required_fields(self):
        """Verify the summary builder produces all required fields."""
        from scripts.run_mvp import _build_summary

        registration = CourtRegistration2D(mode="floor_homography", confidence=0.85)
        tracks = [
            PlayerTrack(
                player_id="near_left",
                frames=[0, 1],
                bboxes=[(100, 200, 150, 300), (101, 201, 151, 301)],
                confidences=[0.9, 0.85],
                team="near",
            )
        ]
        metric_frames = [
            PlayerMetricFrame(
                frame=0, time_s=0.0, player_id="near_left",
                court_xy=(5.0, 15.0), speed_mps=2.0, zone="mid", confidence=0.9,
            )
        ]

        summary = _build_summary(
            video_path="test.mp4",
            total_frames=100,
            fps=30.0,
            duration_s=3.33,
            registration=registration,
            tracks=tracks,
            metric_frames=metric_frames,
            ball_detections=[],
            ball_tracks=[],
            ball_events=[],
            rally_metrics=[],
            scoreboard_states=[
                ScoreboardState(frame=0, time_s=0.0, roi_bbox_xyxy=(100, 10, 500, 80), confidence=0.5)
            ],
            elapsed_s=5.0,
            config={},
        )

        assert summary["video_path"] == "test.mp4"
        assert summary["total_frames"] == 100
        assert summary["fps"] == 30.0
        assert summary["duration_s"] == 3.33
        assert summary["registration_mode"] == "floor_homography"
        assert summary["registration_confidence"] == 0.85
        assert summary["pipeline_elapsed_s"] == 5.0
        assert "player_stats" in summary
        assert "team_stats" in summary
        assert "ball_tracking" in summary
        assert "scoreboard" in summary
        assert summary["scoreboard"]["roi_bbox_xyxy"] == (100, 10, 500, 80)
        assert "near_left" in summary["player_stats"]


class TestVideoWriter:
    """Test exported videos use a VS Code/browser-compatible MP4 codec."""

    def test_write_annotated_video_uses_h264_mp4(self, tmp_path, geometry, registration):
        from src.export.video_writer import write_annotated_video

        input_path = tmp_path / "input.mp4"
        output_path = tmp_path / "annotated.mp4"
        _write_tiny_input_video(input_path)

        write_annotated_video(
            video_path=str(input_path),
            output_path=str(output_path),
            registration=registration,
            geometry=geometry,
            tracks=[],
            metric_frames=[],
            ball_tracks=[],
            scoreboard_states=[],
            fps=10.0,
        )

        assert output_path.exists()
        assert _read_video_fourcc(output_path) in {"h264", "avc1"}

    def test_write_minimap_video_uses_h264_mp4(self, tmp_path, geometry):
        from src.export.video_writer import write_minimap_video

        output_path = tmp_path / "minimap.mp4"

        write_minimap_video(
            output_path=str(output_path),
            geometry=geometry,
            metric_frames=[],
            ball_tracks=[],
            total_frames=3,
            fps=10.0,
        )

        assert output_path.exists()
        assert _read_video_fourcc(output_path) in {"h264", "avc1"}

    def test_write_minimap_video_draws_metric_players(self, tmp_path, geometry, sample_metrics):
        from src.export.video_writer import write_minimap_video

        output_path = tmp_path / "minimap.mp4"
        base_path = tmp_path / "minimap_base.mp4"

        write_minimap_video(
            output_path=str(output_path),
            geometry=geometry,
            metric_frames=sample_metrics[:1],
            ball_tracks=[],
            total_frames=1,
            fps=10.0,
            max_player_gap_fill_frames=15,
        )
        write_minimap_video(
            output_path=str(base_path),
            geometry=geometry,
            metric_frames=[],
            ball_tracks=[],
            total_frames=1,
            fps=10.0,
            max_player_gap_fill_frames=15,
        )

        cap = cv2.VideoCapture(str(output_path))
        ok, frame = cap.read()
        cap.release()
        base_cap = cv2.VideoCapture(str(base_path))
        base_ok, base_frame = base_cap.read()
        base_cap.release()

        assert ok
        assert base_ok
        assert not np.array_equal(frame, base_frame)


class TestAnnotateFrame:
    """Test annotate_frame produces valid image."""

    def test_draw_player_boxes_fills_short_visual_gap(self):
        from src.visualization.overlay import draw_player_boxes

        track = PlayerTrack(
            player_id="near_left",
            frames=[0, 2],
            bboxes=[(20.0, 20.0, 40.0, 60.0), (40.0, 20.0, 60.0, 60.0)],
            confidences=[0.9, 0.7],
            team="near",
        )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        result = draw_player_boxes(frame, [track], frame_idx=1, max_gap_fill_frames=1)

        assert result.sum() > 0

    def test_draw_player_boxes_keeps_gap_when_disabled(self):
        from src.visualization.overlay import draw_player_boxes

        track = PlayerTrack(
            player_id="near_left",
            frames=[0, 2],
            bboxes=[(20.0, 20.0, 40.0, 60.0), (40.0, 20.0, 60.0, 60.0)],
            confidences=[0.9, 0.7],
            team="near",
        )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        result = draw_player_boxes(frame, [track], frame_idx=1)

        assert result.sum() == 0

    def test_draw_player_boxes_ignores_long_visual_gap(self):
        from src.visualization.overlay import draw_player_boxes

        track = PlayerTrack(
            player_id="near_left",
            frames=[0, 20],
            bboxes=[(20.0, 20.0, 40.0, 60.0), (40.0, 20.0, 60.0, 60.0)],
            confidences=[0.9, 0.7],
            team="near",
        )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        result = draw_player_boxes(frame, [track], frame_idx=10, max_gap_fill_frames=5)

        assert result.sum() == 0

    def test_annotate_frame_produces_valid_image(self, geometry, registration, sample_tracks):
        from src.visualization.overlay import annotate_frame

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        result = annotate_frame(
            frame=frame,
            frame_idx=0,
            registration=registration,
            geometry=geometry,
            tracks=sample_tracks,
            metrics=None,
            ball_track=None,
            score=None,
            formation_near=None,
            formation_far=None,
        )

        assert result.shape == (720, 1280, 3)
        assert result.dtype == np.uint8

    def test_annotate_frame_with_all_data(
        self, geometry, sample_tracks, sample_metrics,
        sample_ball_tracks, sample_scoreboard_states,
    ):
        from src.visualization.overlay import annotate_frame

        # Use a registration with homography for full overlay
        registration = CourtRegistration2D(
            mode="floor_homography",
            homography_court_to_image=[
                [100.0, 0.0, 100.0],
                [0.0, 50.0, 50.0],
                [0.0, 0.0, 1.0],
            ],
            homography_image_to_court=[
                [0.01, 0.0, -1.0],
                [0.0, 0.02, -1.0],
                [0.0, 0.0, 1.0],
            ],
            confidence=0.85,
        )

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        result = annotate_frame(
            frame=frame,
            frame_idx=0,
            registration=registration,
            geometry=geometry,
            tracks=sample_tracks,
            metrics=sample_metrics,
            ball_track=sample_ball_tracks[0],
            score=sample_scoreboard_states[0],
            formation_near="both_net",
            formation_far="one_up_one_back",
        )

        assert result.shape == (720, 1280, 3)
        assert result.dtype == np.uint8
        # Frame should not be all black (overlays were drawn)
        assert result.sum() > 0

    def test_draw_scoreboard_info_draws_roi_bbox(self):
        from src.visualization.overlay import draw_scoreboard_info

        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        score = ScoreboardState(
            frame=0,
            time_s=0.0,
            roi_bbox_xyxy=(10, 20, 60, 40),
            confidence=0.5,
        )

        result = draw_scoreboard_info(frame, score)

        assert result.shape == frame.shape
        assert result[20, 10].sum() > 0


class TestMinimap:
    """Test minimap creation works."""

    def test_create_court_base(self, geometry):
        from src.visualization.minimap import create_court_base

        court = create_court_base(width_px=300, length_px=600, geometry=geometry)

        assert court.shape == (600, 300, 3)
        assert court.dtype == np.uint8
        # Should not be all black (green background + white lines)
        assert court.sum() > 0

    def test_create_court_base_default_geometry(self):
        from src.visualization.minimap import create_court_base

        court = create_court_base()
        assert court.shape == (600, 300, 3)

    def test_draw_minimap_frame_with_players(self, geometry, sample_metrics):
        from src.visualization.minimap import create_court_base, draw_minimap_frame

        court_base = create_court_base(geometry=geometry)
        minimap = draw_minimap_frame(
            court_base=court_base,
            players=sample_metrics,
            ball_court_xy=(5.0, 10.0),
            geometry=geometry,
        )

        assert minimap.shape == court_base.shape
        assert minimap.dtype == np.uint8

    def test_draw_minimap_frame_no_players(self, geometry):
        from src.visualization.minimap import create_court_base, draw_minimap_frame

        court_base = create_court_base(geometry=geometry)
        minimap = draw_minimap_frame(
            court_base=court_base,
            players=None,
            ball_court_xy=None,
            geometry=geometry,
        )

        # Should be identical to base when no players/ball
        assert np.array_equal(minimap, court_base)

    def test_draw_minimap_frame_ball_only(self, geometry):
        from src.visualization.minimap import create_court_base, draw_minimap_frame

        court_base = create_court_base(geometry=geometry)
        minimap = draw_minimap_frame(
            court_base=court_base,
            players=None,
            ball_court_xy=(5.0, 10.0),
            geometry=geometry,
        )

        # Ball was drawn, so minimap should differ from base
        assert not np.array_equal(minimap, court_base)

    def test_minimap_players_fill_short_gap(self):
        from src.export.video_writer import _minimap_players_for_frame

        metrics = [
            PlayerMetricFrame(
                frame=0,
                time_s=0.0,
                player_id="near_left",
                court_xy=(2.0, 4.0),
                zone="mid",
                confidence=0.8,
            ),
            PlayerMetricFrame(
                frame=2,
                time_s=0.2,
                player_id="near_left",
                court_xy=(4.0, 8.0),
                zone="mid",
                confidence=0.6,
            ),
        ]

        players = _minimap_players_for_frame({}, {"near_left": metrics}, 1, 1, 10.0)

        assert players is not None
        assert players[0].court_xy == pytest.approx((3.0, 6.0))
        assert players[0].confidence == pytest.approx(0.3)

    def test_minimap_players_keep_gap_when_disabled(self):
        from src.export.video_writer import _minimap_players_for_frame

        metrics = [
            PlayerMetricFrame(
                frame=0,
                time_s=0.0,
                player_id="near_left",
                court_xy=(2.0, 4.0),
                zone="mid",
                confidence=0.8,
            ),
            PlayerMetricFrame(
                frame=2,
                time_s=0.2,
                player_id="near_left",
                court_xy=(4.0, 8.0),
                zone="mid",
                confidence=0.6,
            ),
        ]

        players = _minimap_players_for_frame({}, {"near_left": metrics}, 1, 0, 10.0)

        assert players is None

    def test_minimap_players_ignore_long_gap(self):
        from src.export.video_writer import _minimap_players_for_frame

        metrics = [
            PlayerMetricFrame(
                frame=0,
                time_s=0.0,
                player_id="near_left",
                court_xy=(2.0, 4.0),
                zone="mid",
                confidence=0.8,
            ),
            PlayerMetricFrame(
                frame=20,
                time_s=2.0,
                player_id="near_left",
                court_xy=(4.0, 8.0),
                zone="mid",
                confidence=0.6,
            ),
        ]

        players = _minimap_players_for_frame({}, {"near_left": metrics}, 10, 5, 10.0)

        assert players is None


class TestEmptyInputs:
    """Test export handles empty inputs gracefully."""

    def test_export_all_empty_data(self, geometry, registration):
        from src.export.writer import export_all

        with tempfile.TemporaryDirectory() as tmpdir:
            export_all(
                output_dir=tmpdir,
                registration=registration,
                geometry=geometry,
                tracks=[],
                metric_frames=[],
                ball_detections=[],
                ball_tracks=[],
                ball_events=[],
                scoreboard_states=[],
                rally_metrics=[],
                summary={"empty": True},
                config={},
            )

            # All files should still be created
            assert (Path(tmpdir) / "tracks.csv").exists()
            assert (Path(tmpdir) / "metrics.csv").exists()
            assert (Path(tmpdir) / "ball_tracks.csv").exists()
            assert (Path(tmpdir) / "summary.json").exists()

    def test_tracks_csv_empty(self):
        from src.export.writer import write_tracks_csv

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = Path(f.name)

        try:
            write_tracks_csv(path, [])
            with open(path) as f:
                lines = f.readlines()
            # Should have header only
            assert len(lines) == 1
            assert "frame,player_id" in lines[0]
        finally:
            os.unlink(path)
