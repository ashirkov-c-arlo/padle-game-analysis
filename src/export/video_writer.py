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
from src.visualization.minimap import create_court_base, draw_minimap_frame
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
    for m in metric_frames:
        metrics_by_frame.setdefault(m.frame, []).append(m)

    logger.debug(
        "Writing minimap video: output={}, frames={}, fps={}, metric_frames={}, ball_tracks={}",
        output_path,
        total_frames,
        fps,
        len(metric_frames),
        len(ball_tracks),
    )

    for frame_idx in range(total_frames):
        players = metrics_by_frame.get(frame_idx)

        # Ball on minimap: only if we have court coordinates
        # (ball_tracks are in image space, not court space, so we cannot show them without
        # a homography. The minimap will show players only unless ball court coords are available.)
        ball_court_xy = None

        minimap = draw_minimap_frame(
            court_base=court_base,
            players=players,
            ball_court_xy=ball_court_xy,
            geometry=geometry,
        )

        writer.send(np.ascontiguousarray(minimap))

    writer.close()
    logger.info("Minimap video written: {} ({} frames)", output_path, total_frames)
