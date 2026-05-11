from __future__ import annotations

import cv2
import imageio_ffmpeg
import numpy as np
from loguru import logger

from src.schemas import (
    BallTrack2D,
    CourtGeometry2D,
    CourtRegistration2D,
    PlayerMetricFrame,
    PlayerTrack,
    ScoreboardState,
)
from src.visualization.minimap import MinimapTrailState, create_court_base, draw_minimap_frame
from src.visualization.overlay import annotate_frame


def _open_h264_mp4_writer(output_path: str, width: int, height: int, fps: float):
    writer = imageio_ffmpeg.write_frames(
        output_path,
        (width, height),
        pix_fmt_in="bgr24",
        pix_fmt_out="yuv420p",
        fps=fps,
        codec="libx264",
        macro_block_size=2,
        output_params=["-movflags", "+faststart"],
        ffmpeg_log_level="error",
    )
    writer.send(None)
    return writer


def write_annotated_video(
    video_path: str,
    output_path: str,
    registration: CourtRegistration2D,
    geometry: CourtGeometry2D,
    tracks: list[PlayerTrack],
    metric_frames: list[PlayerMetricFrame],
    ball_tracks: list[BallTrack2D],
    scoreboard_states: list[ScoreboardState],
    fps: float | None = None,
    max_player_gap_fill_frames: int = 0,
) -> None:
    """Write annotated video with all overlays.

    Reads input frame by frame, annotates, writes to output.
    Writes H.264 MP4 output for compatibility with browser-based video viewers.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot open video for annotation: {}", video_path)
        return

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = fps if fps is not None else input_fps

    try:
        writer = _open_h264_mp4_writer(output_path, width, height, out_fps)
    except OSError as exc:
        logger.error("Cannot open H.264 MP4 writer for {}: {}", output_path, exc)
        cap.release()
        return

    # Build lookup structures for O(1) access per frame
    ball_by_frame: dict[int, BallTrack2D] = {t.frame: t for t in ball_tracks}
    # For scoreboard, use most recent state at or before each frame
    sorted_scores = sorted(scoreboard_states, key=lambda s: s.frame)

    # Build metrics lookup by frame
    metrics_by_frame: dict[int, list[PlayerMetricFrame]] = {}
    for m in metric_frames:
        metrics_by_frame.setdefault(m.frame, []).append(m)

    logger.debug(
        "Writing annotated video: output={}, frames={}, fps={}, tracks={}, ball_tracks={}, scoreboard_states={}",
        output_path,
        total_frames,
        out_fps,
        len(tracks),
        len(ball_tracks),
        len(scoreboard_states),
    )

    frame_idx = 0
    current_score_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Find current score state
        score: ScoreboardState | None = None
        while current_score_idx < len(sorted_scores) - 1 and sorted_scores[current_score_idx + 1].frame <= frame_idx:
            current_score_idx += 1
        if sorted_scores and sorted_scores[current_score_idx].frame <= frame_idx:
            score = sorted_scores[current_score_idx]

        ball_track = ball_by_frame.get(frame_idx)
        frame_metrics = metrics_by_frame.get(frame_idx)

        annotated = annotate_frame(
            frame=frame,
            frame_idx=frame_idx,
            registration=registration,
            geometry=geometry,
            tracks=tracks,
            metrics=frame_metrics,
            ball_track=ball_track,
            score=score,
            formation_near=None,
            formation_far=None,
            max_player_gap_fill_frames=max_player_gap_fill_frames,
        )

        writer.send(np.ascontiguousarray(annotated))
        frame_idx += 1

    cap.release()
    writer.close()
    logger.info("Annotated video written: {} ({} frames)", output_path, frame_idx)


def write_minimap_video(
    output_path: str,
    geometry: CourtGeometry2D,
    metric_frames: list[PlayerMetricFrame],
    ball_tracks: list[BallTrack2D],
    total_frames: int,
    fps: float,
    max_player_gap_fill_frames: int = 0,
) -> None:
    """Write top-down minimap video."""
    width_px = 300
    length_px = 600

    court_base = create_court_base(width_px=width_px, length_px=length_px, geometry=geometry)

    try:
        writer = _open_h264_mp4_writer(output_path, width_px, length_px, fps)
    except OSError as exc:
        logger.error("Cannot open H.264 minimap video writer for {}: {}", output_path, exc)
        return

    # Build lookups
    metrics_by_frame: dict[int, list[PlayerMetricFrame]] = {}
    metrics_by_player: dict[str, list[PlayerMetricFrame]] = {}
    for m in metric_frames:
        metrics_by_frame.setdefault(m.frame, []).append(m)
        metrics_by_player.setdefault(m.player_id, []).append(m)
    for observations in metrics_by_player.values():
        observations.sort(key=lambda item: item.frame)

    # Ball track lookup by frame
    ball_by_frame: dict[int, BallTrack2D] = {t.frame: t for t in ball_tracks}

    # Trail state for persistent history across frames
    trail_state = MinimapTrailState(
        player_trail_length=int(fps),  # ~1 second
        ball_trail_length=max(int(fps * 0.67), 20),  # ~0.67 seconds
    )

    logger.debug(
        "Writing minimap video: output={}, frames={}, fps={}, metric_frames={}, ball_tracks={}",
        output_path,
        total_frames,
        fps,
        len(metric_frames),
        len(ball_tracks),
    )

    for frame_idx in range(total_frames):
        players = _minimap_players_for_frame(
            metrics_by_frame,
            metrics_by_player,
            frame_idx,
            max_player_gap_fill_frames,
            fps,
        )

        # Ball on minimap: only if we have court coordinates
        # (ball_tracks are in image space, not court space, so we cannot show them without
        # a homography. The minimap will show players only unless ball court coords are available.)
        ball_court_xy = None
        ball_state = None
        ball_track = ball_by_frame.get(frame_idx)
        if ball_track is not None:
            ball_state = ball_track.state

        minimap = draw_minimap_frame(
            court_base=court_base,
            players=players,
            ball_court_xy=ball_court_xy,
            geometry=geometry,
            trail_state=trail_state,
            ball_state=ball_state,
        )

        writer.send(np.ascontiguousarray(minimap))

    writer.close()
    logger.info("Minimap video written: {} ({} frames)", output_path, total_frames)


def _minimap_players_for_frame(
    metrics_by_frame: dict[int, list[PlayerMetricFrame]],
    metrics_by_player: dict[str, list[PlayerMetricFrame]],
    frame_idx: int,
    max_gap_fill_frames: int,
    fps: float,
) -> list[PlayerMetricFrame] | None:
    players = list(metrics_by_frame.get(frame_idx, []))
    if max_gap_fill_frames <= 0:
        return players or None

    present_ids = {player.player_id for player in players}
    for player_id, observations in metrics_by_player.items():
        if player_id in present_ids:
            continue
        interpolated = _interpolate_minimap_player(observations, frame_idx, max_gap_fill_frames, fps)
        if interpolated is not None:
            players.append(interpolated)

    return players or None


def _interpolate_minimap_player(
    observations: list[PlayerMetricFrame],
    frame_idx: int,
    max_gap_fill_frames: int,
    fps: float,
) -> PlayerMetricFrame | None:
    next_idx = next((idx for idx, metric in enumerate(observations) if metric.frame > frame_idx), None)
    if next_idx is None or next_idx == 0:
        return None

    prev = observations[next_idx - 1]
    nxt = observations[next_idx]
    gap_frames = nxt.frame - prev.frame - 1
    if gap_frames <= 0 or gap_frames > max_gap_fill_frames:
        return None

    alpha = (frame_idx - prev.frame) / (nxt.frame - prev.frame)
    court_xy = (
        prev.court_xy[0] + (nxt.court_xy[0] - prev.court_xy[0]) * alpha,
        prev.court_xy[1] + (nxt.court_xy[1] - prev.court_xy[1]) * alpha,
    )
    return PlayerMetricFrame(
        frame=frame_idx,
        time_s=frame_idx / fps,
        player_id=prev.player_id,
        court_xy=court_xy,
        speed_mps=0.0,
        zone=prev.zone,
        confidence=min(prev.confidence, nxt.confidence) * 0.5,
        metric_quality=prev.metric_quality,
    )
