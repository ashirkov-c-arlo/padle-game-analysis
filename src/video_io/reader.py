from __future__ import annotations

import cv2
import numpy as np
from loguru import logger


def get_video_info(video_path: str) -> dict:
    """Return fps, total_frames, width, height, duration_s."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration_s"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0.0
    cap.release()

    logger.debug(
        "Video info: {}x{}, {:.1f} fps, {} frames, {:.1f}s",
        info["width"],
        info["height"],
        info["fps"],
        info["total_frames"],
        info["duration_s"],
    )
    return info


def read_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """Read a single frame by index. Returns BGR image."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Cannot read frame {frame_idx} from {video_path}")
    return frame
