from __future__ import annotations

import cv2
import numpy as np
from loguru import logger

from src.video_io.reader import get_video_info, read_frame


def sample_stable_frames(
    video_path: str, interval_s: float, max_frames: int = 20
) -> list[tuple[int, np.ndarray]]:
    """Sample frames at regular intervals, skip very dark/bright/blurry ones.

    Skips the first and last 5% of the video to avoid intro/outro frames.
    Returns list of (frame_index, frame_bgr) tuples.
    """
    info = get_video_info(video_path)
    fps = info["fps"]
    total_frames = info["total_frames"]

    if total_frames == 0 or fps == 0:
        logger.warning("Video has no frames or zero fps")
        return []

    # Skip first/last 5%
    start_frame = int(total_frames * 0.05)
    end_frame = int(total_frames * 0.95)

    interval_frames = int(interval_s * fps)
    if interval_frames < 1:
        interval_frames = 1

    candidate_indices = list(range(start_frame, end_frame, interval_frames))
    logger.debug(
        "Sampling {} candidate frames from {} to {} (interval={})",
        len(candidate_indices),
        start_frame,
        end_frame,
        interval_frames,
    )

    results: list[tuple[int, np.ndarray]] = []
    for idx in candidate_indices:
        if len(results) >= max_frames:
            break
        try:
            frame = read_frame(video_path, idx)
        except (ValueError, FileNotFoundError):
            continue

        if not _is_stable_frame(frame):
            continue

        results.append((idx, frame))

    logger.info("Sampled {} stable frames from video", len(results))
    return results


def _is_stable_frame(frame: np.ndarray) -> bool:
    """Check if a frame is suitable for line detection.

    Rejects frames that are too dark, too bright, or too blurry.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

    mean_brightness = gray.mean()
    if mean_brightness < 30:
        return False
    if mean_brightness > 240:
        return False

    # Laplacian variance as sharpness measure
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 50:
        return False

    return True
