from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np
from loguru import logger


class FrameProcessor(Protocol):
    """Interface for modules that consume frames in a single-pass pipeline."""

    def should_process(self, frame_idx: int) -> bool:
        """Return True if this processor wants the given frame."""
        ...

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> None:
        """Process a single frame. Results stored internally."""
        ...

    def finalize(self) -> None:
        """Called after the last frame. Use for post-processing."""
        ...


def run_single_pass(
    video_path: str,
    processors: list[FrameProcessor],
) -> int:
    """Decode video once, dispatching each frame to all interested processors.

    Returns total number of frames decoded.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Single-pass: decoding {} frames for {} processors", total_frames, len(processors))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for proc in processors:
            if proc.should_process(frame_idx):
                proc.process_frame(frame, frame_idx)

        frame_idx += 1
        if frame_idx % 500 == 0:
            logger.debug("Single-pass progress: {}/{}", frame_idx, total_frames)

    cap.release()

    for proc in processors:
        proc.finalize()

    logger.info("Single-pass complete: {} frames decoded", frame_idx)
    return frame_idx
