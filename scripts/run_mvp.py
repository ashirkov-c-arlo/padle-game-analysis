"""MVP pipeline entry point.

Usage: python scripts/run_mvp.py --video input.mp4 --config configs/default.yaml --out outputs/
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import click
from loguru import logger

from src.config.loader import load_config
from src.logging_config import LOG_LEVELS, configure_logging
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


@click.command()
@click.option("--video", required=True, type=click.Path(exists=True), help="Input video path")
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Config YAML override path",
)
@click.option(
    "--out",
    "output_dir",
    default="data/outputs",
    type=click.Path(),
    help="Output directory",
)
@click.option(
    "--log-level",
    default=None,
    type=click.Choice(LOG_LEVELS, case_sensitive=False),
    help="Log level. Defaults to PADEL_CV_LOG_LEVEL or INFO.",
)
def main(video: str, config_path: str | None, output_dir: str, log_level: str | None) -> None:
    """Run the full MVP padel analysis pipeline."""
    active_log_level = configure_logging(log_level)
    start_time = time.time()
    logger.info("Starting pipeline: video={}, config={}, output={}", video, config_path or "default", output_dir)
    logger.debug("Runtime options: log_level={}", active_log_level)

    # 1. Load config
    cfg = load_config(config_path)
    logger.debug("Config loaded: top_level_keys={}", sorted(cfg.keys()))

    # 2. Get video info
    from src.video_io.reader import get_video_info

    video_info = get_video_info(video)
    total_frames = video_info["total_frames"]
    fps = video_info["fps"]
    duration_s = video_info["duration_s"]
    image_shape = (video_info["height"], video_info["width"])

    logger.info(
        "Video info: {}x{}, {:.1f} fps, {} frames, {:.1f}s",
        video_info["width"], video_info["height"], fps, total_frames, duration_s,
    )

    # 3. Create output directory named after input video
    out_path = Path(output_dir) / Path(video).stem
    out_path.mkdir(parents=True, exist_ok=True)

    # Initialize court geometry
    geometry = CourtGeometry2D(
        width_m=cfg["court"]["width_m"],
        length_m=cfg["court"]["length_m"],
        net_y_m=cfg["court"]["net_y_m"],
    )

    # Initialize pipeline results (will be populated by each stage)
    registration: CourtRegistration2D = CourtRegistration2D(mode="pixel_only", confidence=0.0)
    tracks: list[PlayerTrack] = []
    metric_frames: list[PlayerMetricFrame] = []
    ball_detections: list[BallDetection2D] = []
    ball_tracks: list[BallTrack2D] = []
    ball_events: list[BallEventCandidate] = []
    rally_metrics: list[RallyTempoMetric] = []
    scoreboard_states: list[ScoreboardState] = []

    # 4. Court registration
    try:
        from src.calibration.court_registration import register_court

        logger.debug("Stage court_registration started")
        registration = register_court(video, cfg)
        reprojection_error = (
            f"{registration.reprojection_error_px:.2f}px"
            if registration.reprojection_error_px is not None
            else "n/a"
        )
        logger.info(
            "Court: mode={}, confidence={:.2f}, reprojection_error={}",
            registration.mode,
            registration.confidence,
            reprojection_error,
        )
    except Exception as e:
        logger.warning("Court registration failed; continuing in pixel_only mode")
        logger.opt(exception=e).debug("Court registration failure details")
        registration = CourtRegistration2D(mode="pixel_only", confidence=0.0)

    # 5. Single-pass: player detection + ball detection + scoreboard OCR
    detections: dict[int, list] = {}
    try:
        from src.ball_tracking.ball_processor import BallFrameProcessor
        from src.ball_tracking.tracker import _build_court_roi
        from src.detection.player_processor import PlayerFrameProcessor
        from src.detection.roi_filter import filter_detections_by_court_roi
        from src.scoreboard.scoreboard_processor import ScoreboardFrameProcessor
        from src.video_io.single_pass import run_single_pass

        logger.debug("Stage single_pass started")

        player_proc = PlayerFrameProcessor(cfg)

        court_roi = _build_court_roi(registration, video_info) if registration.mode != "pixel_only" else None
        ball_proc = BallFrameProcessor(cfg, fps, court_roi)

        scoreboard_proc = ScoreboardFrameProcessor(cfg, fps, image_shape)

        processors = [player_proc, ball_proc]
        if scoreboard_proc.is_enabled:
            processors.append(scoreboard_proc)
        logger.debug("Single-pass processors: {}", [type(proc).__name__ for proc in processors])

        run_single_pass(video, processors)

        # Collect player detections and filter by court ROI
        detections = player_proc.detections
        for frame_idx in detections:
            detections[frame_idx] = filter_detections_by_court_roi(
                detections[frame_idx], registration, image_shape
            )
        total_dets = sum(len(d) for d in detections.values())
        logger.debug("Player detection: detections={}, frames={}", total_dets, len(detections))

        # Collect ball detections
        ball_detections = ball_proc.detections
        logger.debug("Ball detection: detections={}", len(ball_detections))

        # Collect scoreboard states
        if scoreboard_proc.is_enabled:
            scoreboard_states = scoreboard_proc.get_states()
        logger.info(
            "Detection: player_detections={}, player_frames={}, ball_detections={}, scoreboard_states={}, ocr={}",
            total_dets,
            len(detections),
            len(ball_detections),
            len(scoreboard_states),
            scoreboard_proc.is_available if scoreboard_proc.is_enabled else False,
        )

    except Exception as e:
        logger.warning("Single-pass detection failed")
        logger.opt(exception=e).debug("Single-pass detection failure details")

    # 6. Player tracking
    try:
        from src.tracking.tracker import track_players

        logger.debug("Stage player_tracking started")
        tracks = track_players(
            video_path=video,
            detections=detections,
            config=cfg,
            registration=registration,
            fps=fps,
            image_shape=image_shape,
        )
        logger.info("Tracking: {} player tracks", len(tracks))
    except Exception as e:
        logger.warning("Player tracking failed")
        logger.opt(exception=e).debug("Player tracking failure details")

    # 7. Court coordinate projection + analytics
    try:
        from src.analytics.metrics import build_player_metric_frames, compute_player_metrics

        logger.debug("Stage analytics started")
        metric_frames = build_player_metric_frames(tracks, registration, geometry, cfg, fps)
        compute_player_metrics(tracks, registration, geometry, cfg, fps)
        logger.info("Analytics: {} metric frames", len(metric_frames))
    except Exception as e:
        logger.warning("Analytics computation failed")
        logger.opt(exception=e).debug("Analytics computation failure details")

    # 8. Ball tracking (Kalman post-processing on detections from single-pass)
    try:
        from src.ball_tracking.tracker import build_ball_tracks

        logger.debug("Stage ball_tracking started")
        ball_tracks = build_ball_tracks(ball_detections, cfg, total_frames, fps)
        logger.info("Ball tracking: {} frames", len(ball_tracks))
    except Exception as e:
        logger.warning("Ball tracking failed; continuing without ball data")
        logger.opt(exception=e).debug("Ball tracking failure details")

    # 9. Ball events + metrics
    try:
        from src.ball_tracking.events import (
            detect_bounce_candidates,
            detect_net_crossings,
            detect_touch_candidates,
        )
        from src.ball_tracking.metrics import compute_rally_tempo

        logger.debug("Stage ball_events started")

        bounce_candidates = detect_bounce_candidates(ball_tracks, registration, fps)
        ball_events.extend(bounce_candidates)

        # Build player pixel positions for touch detection
        player_positions: dict[str, list[tuple[float, float]]] = {}
        for track in tracks:
            positions: list[tuple[float, float]] = []
            frame_to_idx = {f: i for i, f in enumerate(track.frames)}
            for f_idx in range(total_frames):
                if f_idx in frame_to_idx:
                    bbox = track.bboxes[frame_to_idx[f_idx]]
                    # Bottom-center of bbox
                    positions.append(((bbox[0] + bbox[2]) / 2, bbox[3]))
                elif positions:
                    positions.append(positions[-1])
                else:
                    positions.append((0.0, 0.0))
            player_positions[track.player_id] = positions

        touch_candidates = detect_touch_candidates(ball_tracks, player_positions, fps)
        ball_events.extend(touch_candidates)

        net_crossings = detect_net_crossings(ball_tracks, registration)
        ball_events.extend(net_crossings)

        rally_metrics = compute_rally_tempo(touch_candidates, fps)

        logger.info(
            "Ball events: {} bounces, {} touches, {} net crossings, {} rallies",
            len(bounce_candidates), len(touch_candidates), len(net_crossings), len(rally_metrics),
        )
    except Exception as e:
        logger.warning("Ball event detection failed")
        logger.opt(exception=e).debug("Ball event detection failure details")

    # 10. Scoreboard OCR (already processed in single-pass, this is a no-op)
    logger.debug("Scoreboard already processed in single-pass: states={}", len(scoreboard_states))

    # 11. Build summary
    elapsed = time.time() - start_time
    summary = _build_summary(
        video_path=video,
        total_frames=total_frames,
        fps=fps,
        duration_s=duration_s,
        registration=registration,
        tracks=tracks,
        metric_frames=metric_frames,
        ball_detections=ball_detections,
        ball_tracks=ball_tracks,
        ball_events=ball_events,
        rally_metrics=rally_metrics,
        scoreboard_states=scoreboard_states,
        elapsed_s=elapsed,
        config=cfg,
    )

    # 12. Export all files (JSON, CSV, JSONL)
    try:
        from src.export.writer import export_all

        logger.debug("Stage export_data started")
        export_all(
            output_dir=str(out_path),
            registration=registration,
            geometry=geometry,
            tracks=tracks,
            metric_frames=metric_frames,
            ball_detections=ball_detections,
            ball_tracks=ball_tracks,
            ball_events=ball_events,
            scoreboard_states=scoreboard_states,
            rally_metrics=rally_metrics,
            summary=summary,
            config=cfg,
        )
    except Exception as e:
        logger.error("Export failed")
        logger.opt(exception=e).debug("Export failure details")
        # Write at least the summary
        summary_path = out_path / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Wrote fallback summary: {}", summary_path)

    # 13. Write annotated video
    export_cfg = cfg.get("export", {})
    video_cfg = export_cfg.get("video", {})

    if video_cfg.get("annotated", True):
        try:
            from src.export.video_writer import write_annotated_video

            annotated_path = str(out_path / "annotated.mp4")
            logger.debug("Stage annotated_video started")
            write_annotated_video(
                video_path=video,
                output_path=annotated_path,
                registration=registration,
                geometry=geometry,
                tracks=tracks,
                metric_frames=metric_frames,
                ball_tracks=ball_tracks,
                scoreboard_states=scoreboard_states,
                fps=video_cfg.get("fps"),
            )
        except Exception as e:
            logger.warning("Annotated video writing failed")
            logger.opt(exception=e).debug("Annotated video failure details")

    # 14. Write minimap video
    if video_cfg.get("minimap", True):
        try:
            from src.export.video_writer import write_minimap_video

            minimap_path = str(out_path / "minimap.mp4")
            logger.debug("Stage minimap_video started")
            write_minimap_video(
                output_path=minimap_path,
                geometry=geometry,
                metric_frames=metric_frames,
                ball_tracks=ball_tracks,
                total_frames=total_frames,
                fps=fps,
            )
        except Exception as e:
            logger.warning("Minimap video writing failed")
            logger.opt(exception=e).debug("Minimap video failure details")

    # 15. Log completion
    total_elapsed = time.time() - start_time
    logger.info("Pipeline complete: {:.2f}s, output={}", total_elapsed, out_path)


def _build_summary(
    video_path: str,
    total_frames: int,
    fps: float,
    duration_s: float,
    registration: CourtRegistration2D,
    tracks: list[PlayerTrack],
    metric_frames: list[PlayerMetricFrame],
    ball_detections: list[BallDetection2D],
    ball_tracks: list[BallTrack2D],
    ball_events: list[BallEventCandidate],
    rally_metrics: list[RallyTempoMetric],
    scoreboard_states: list[ScoreboardState],
    elapsed_s: float,
    config: dict,
) -> dict:
    """Build the pipeline summary dictionary."""
    # Player stats
    player_stats = {}
    for track in tracks:
        player_metrics = [m for m in metric_frames if m.player_id == track.player_id]
        speeds = [m.speed_mps for m in player_metrics if m.speed_mps > 0]
        player_stats[track.player_id] = {
            "team": track.team,
            "frames_tracked": len(track.frames),
            "avg_speed_mps": sum(speeds) / len(speeds) if speeds else 0.0,
            "max_speed_mps": max(speeds) if speeds else 0.0,
            "avg_confidence": sum(track.confidences) / len(track.confidences) if track.confidences else 0.0,
        }

    # Team stats
    team_stats = {"near": {}, "far": {}}
    for team in ("near", "far"):
        team_players = [t for t in tracks if t.team == team]
        team_metrics = [m for m in metric_frames if m.player_id.startswith(team)]
        team_speeds = [m.speed_mps for m in team_metrics if m.speed_mps > 0]
        team_stats[team] = {
            "players": [t.player_id for t in team_players],
            "avg_speed_mps": sum(team_speeds) / len(team_speeds) if team_speeds else 0.0,
        }

    # Ball tracking stats
    tracked_frames = [t for t in ball_tracks if t.state in ("detected", "tracked")]
    interpolated_frames = [t for t in ball_tracks if t.state == "interpolated"]
    missing_frames = [t for t in ball_tracks if t.state == "missing"]

    ball_stats = {
        "total_detections": len(ball_detections),
        "tracked_frames": len(tracked_frames),
        "interpolated_frames": len(interpolated_frames),
        "missing_frames": len(missing_frames),
        "detection_rate": len(ball_detections) / total_frames if total_frames > 0 else 0.0,
        "event_candidates": len(ball_events),
        "rallies_detected": len(rally_metrics),
    }

    # Scoreboard stats
    valid_scores = [s for s in scoreboard_states if s.parsed_sets or s.parsed_game_score]
    roi_bbox = next((s.roi_bbox_xyxy for s in scoreboard_states if s.roi_bbox_xyxy is not None), None)
    scoreboard_stats = {
        "total_samples": len(scoreboard_states),
        "valid_scores": len(valid_scores),
        "roi_bbox_xyxy": roi_bbox,
        "avg_confidence": (
            sum(s.confidence for s in scoreboard_states) / len(scoreboard_states)
            if scoreboard_states else 0.0
        ),
    }

    return {
        "video_path": video_path,
        "total_frames": total_frames,
        "fps": fps,
        "duration_s": duration_s,
        "registration_mode": registration.mode,
        "registration_confidence": registration.confidence,
        "pipeline_elapsed_s": round(elapsed_s, 2),
        "player_stats": player_stats,
        "team_stats": team_stats,
        "ball_tracking": ball_stats,
        "scoreboard": scoreboard_stats,
        "config": config,
    }


if __name__ == "__main__":
    main()
