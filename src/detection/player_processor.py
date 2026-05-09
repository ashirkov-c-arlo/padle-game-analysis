from __future__ import annotations

import numpy as np

from src.detection.player_detector import PlayerDetector
from src.schemas import PlayerDetection


class PlayerFrameProcessor:
    """Wraps PlayerDetector for single-pass pipeline."""

    def __init__(self, config: dict):
        self._detector = PlayerDetector(config)
        self.detections: dict[int, list[PlayerDetection]] = {}

    def should_process(self, frame_idx: int) -> bool:
        return True

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> None:
        dets = self._detector.detect_frame(frame)
        for det in dets:
            det.frame = frame_idx
        self.detections[frame_idx] = dets

    def finalize(self) -> None:
        pass
