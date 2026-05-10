from __future__ import annotations

import cv2
import numpy as np

from src.ball_tracking.detector import BallDetector
from src.schemas import BallDetection2D


class BallFrameProcessor:
    """Wraps BallDetector for single-pass pipeline."""

    def __init__(self, config: dict, fps: float, court_roi: np.ndarray | None = None):
        self._detector = BallDetector(config)
        self._fps = fps
        self._court_roi = court_roi
        self.detections: list[BallDetection2D] = []

    def should_process(self, frame_idx: int) -> bool:
        return True

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> None:
        if self._court_roi is not None:
            masked = cv2.bitwise_and(frame, frame, mask=self._court_roi)
        else:
            masked = frame

        detection = self._detector.detect_frame(masked)
        if detection is not None:
            time_s = frame_idx / self._fps if self._fps > 0 else 0.0
            self.detections.append(
                BallDetection2D(
                    frame=frame_idx,
                    time_s=time_s,
                    image_xy=detection.image_xy,
                    confidence=detection.confidence,
                    visibility=detection.visibility,
                    source=detection.source,
                )
            )

    def finalize(self) -> None:
        pass
