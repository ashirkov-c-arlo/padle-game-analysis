from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from src.schemas import (
    BallDetection2D,
    BallEventCandidate,
    BallTrack2D,
    CourtGeometry2D,
    CourtRegistration2D,
    FrameResult,
    PlayerMetricFrame,
    PlayerTrack,
    RallyTempoMetric,
    ScoreboardState,
)


def export_all(
    output_dir: str,
    registration: CourtRegistration2D,
    geometry: CourtGeometry2D,
    tracks: list[PlayerTrack],
    metric_frames: list[PlayerMetricFrame],
    ball_detections: list[BallDetection2D],
    ball_tracks: list[BallTrack2D],
    ball_events: list[BallEventCandidate],
    scoreboard_states: list[ScoreboardState],
    rally_metrics: list[RallyTempoMetric],
    summary: dict,
    config: dict,
) -> None:
    """Write all output files to the output directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    write_court_geometry(out / "court_geometry.json", geometry)
    write_court_registration(out / "court_registration.json", registration)
    write_tracks_csv(out / "tracks.csv", tracks)
    write_metrics_csv(out / "metrics.csv", metric_frames)
    write_ball_tracks_csv(out / "ball_tracks.csv", ball_tracks)
    write_ball_detections_jsonl(out / "ball_detections.jsonl", ball_detections)
    write_ball_events_jsonl(out / "ball_event_candidates.jsonl", ball_events)
    write_scoreboard_csv(out / "scoreboard.csv", scoreboard_states)
    write_rally_metrics_csv(out / "rally_metrics.csv", rally_metrics)
    write_summary_json(out / "summary.json", summary)

    logger.info("Exported all data files to {}", output_dir)


def write_court_geometry(path: Path, geometry: CourtGeometry2D) -> None:
    """Write court geometry to JSON."""
    with open(path, "w") as f:
        json.dump(geometry.model_dump(), f, indent=2)
    logger.debug("Wrote {}", path)


def write_court_registration(path: Path, registration: CourtRegistration2D) -> None:
    """Write court registration to JSON."""
    with open(path, "w") as f:
        json.dump(registration.model_dump(), f, indent=2)
    logger.debug("Wrote {}", path)


def write_frames_jsonl(path: Path, frames: list[FrameResult]) -> None:
    """Write per-frame results to JSONL."""
    with open(path, "w") as f:
        for fr in frames:
            f.write(json.dumps(fr.model_dump(), default=str) + "\n")
    logger.debug("Wrote {} frame results to {}", len(frames), path)


def write_tracks_csv(path: Path, tracks: list[PlayerTrack]) -> None:
    """Write player tracks to CSV.

    Columns: frame,player_id,team,x1,y1,x2,y2,confidence
    """
    with open(path, "w") as f:
        f.write("frame,player_id,team,x1,y1,x2,y2,confidence\n")
        for track in tracks:
            for i, frame in enumerate(track.frames):
                bbox = track.bboxes[i]
                conf = track.confidences[i] if i < len(track.confidences) else 0.0
                f.write(
                    f"{frame},{track.player_id},{track.team},"
                    f"{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f},"
                    f"{conf:.4f}\n"
                )
    logger.debug("Wrote {}", path)


def write_metrics_csv(path: Path, metrics: list[PlayerMetricFrame]) -> None:
    """Write player metrics to CSV.

    Columns: frame,time_s,player_id,court_x,court_y,speed_mps,zone,confidence,metric_quality
    """
    with open(path, "w") as f:
        f.write("frame,time_s,player_id,court_x,court_y,speed_mps,zone,confidence,metric_quality\n")
        for m in metrics:
            f.write(
                f"{m.frame},{m.time_s:.4f},{m.player_id},"
                f"{m.court_xy[0]:.4f},{m.court_xy[1]:.4f},"
                f"{m.speed_mps:.4f},{m.zone},{m.confidence:.4f},{m.metric_quality}\n"
            )
    logger.debug("Wrote {}", path)


def write_ball_tracks_csv(path: Path, tracks: list[BallTrack2D]) -> None:
    """Write ball tracks to CSV.

    Columns: frame,time_s,x_px,y_px,vx_px_s,vy_px_s,confidence,state,interpolated,gap_len
    """
    with open(path, "w") as f:
        f.write("frame,time_s,x_px,y_px,vx_px_s,vy_px_s,confidence,state,interpolated,gap_len\n")
        for t in tracks:
            vx = t.velocity_px_s[0] if t.velocity_px_s else ""
            vy = t.velocity_px_s[1] if t.velocity_px_s else ""
            f.write(
                f"{t.frame},{t.time_s:.4f},"
                f"{t.image_xy[0]:.2f},{t.image_xy[1]:.2f},"
                f"{vx},{vy},"
                f"{t.confidence:.4f},{t.state},"
                f"{str(t.interpolated).lower()},{t.gap_len}\n"
            )
    logger.debug("Wrote {}", path)


def write_ball_detections_jsonl(path: Path, detections: list[BallDetection2D]) -> None:
    """Write ball detections to JSONL."""
    with open(path, "w") as f:
        for d in detections:
            f.write(json.dumps(d.model_dump(), default=str) + "\n")
    logger.debug("Wrote {} ball detections to {}", len(detections), path)


def write_ball_events_jsonl(path: Path, events: list[BallEventCandidate]) -> None:
    """Write ball event candidates to JSONL."""
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e.model_dump(), default=str) + "\n")
    logger.debug("Wrote {} ball events to {}", len(events), path)


def write_scoreboard_csv(path: Path, states: list[ScoreboardState]) -> None:
    """Write scoreboard states to CSV.

    Columns: frame,time_s,roi_x1,roi_y1,roi_x2,roi_y2,raw_text,set1_a,set1_b,game_a,game_b,confidence
    """
    with open(path, "w") as f:
        f.write("frame,time_s,roi_x1,roi_y1,roi_x2,roi_y2,raw_text,set1_a,set1_b,game_a,game_b,confidence\n")
        for s in states:
            raw = (s.raw_text or "").replace(",", ";").replace("\n", " ")
            roi_x1, roi_y1, roi_x2, roi_y2 = s.roi_bbox_xyxy or ("", "", "", "")
            set1_a = s.parsed_sets[0][0] if s.parsed_sets and len(s.parsed_sets) > 0 else ""
            set1_b = s.parsed_sets[0][1] if s.parsed_sets and len(s.parsed_sets) > 0 else ""
            game_a = s.parsed_game_score[0] if s.parsed_game_score else ""
            game_b = s.parsed_game_score[1] if s.parsed_game_score else ""
            f.write(
                f"{s.frame},{s.time_s:.4f},{roi_x1},{roi_y1},{roi_x2},{roi_y2},"
                f"{raw},{set1_a},{set1_b},{game_a},{game_b},{s.confidence:.4f}\n"
            )
    logger.debug("Wrote {}", path)


def write_rally_metrics_csv(path: Path, metrics: list[RallyTempoMetric]) -> None:
    """Write rally tempo metrics to CSV."""
    with open(path, "w") as f:
        f.write("rally_id,duration_s,estimated_shots,avg_time_between_touches_s,median_time_between_touches_s\n")
        for m in metrics:
            avg = f"{m.avg_time_between_touches_s:.3f}" if m.avg_time_between_touches_s is not None else ""
            med = f"{m.median_time_between_touches_s:.3f}" if m.median_time_between_touches_s is not None else ""
            f.write(f"{m.rally_id},{m.duration_s:.3f},{m.estimated_shots},{avg},{med}\n")
    logger.debug("Wrote {}", path)


def write_summary_json(path: Path, summary: dict) -> None:
    """Write pipeline summary to JSON."""
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.debug("Wrote {}", path)
