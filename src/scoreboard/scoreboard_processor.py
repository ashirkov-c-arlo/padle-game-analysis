from __future__ import annotations

import numpy as np
from loguru import logger

from src.schemas import ScoreboardState
from src.scoreboard.fsm import ScoreFSM
from src.scoreboard.ocr_engine import ScoreboardOCR
from src.scoreboard.parser import parse_score_text
from src.scoreboard.roi_detector import detect_scoreboard_roi
from src.scoreboard.scoreboard import stabilize_scores


class ScoreboardFrameProcessor:
    """Wraps scoreboard OCR for single-pass pipeline. Samples every N frames."""

    def __init__(self, config: dict, fps: float, image_shape: tuple[int, int]):
        scoreboard_config = config.get("scoreboard", {})
        self._enabled = scoreboard_config.get("enabled", True)
        self._sample_interval = max(1, int(fps * scoreboard_config.get("sample_interval_s", 1.0)))
        self._min_consistency = scoreboard_config.get("min_consistency_frames", 3)
        self._image_shape = image_shape

        self._ocr = ScoreboardOCR(scoreboard_config) if self._enabled else None
        self._fsm = ScoreFSM()
        self._roi: tuple[int, int, int, int] | None = None
        self._roi_frames: list[tuple[int, np.ndarray]] = []
        self._roi_detected = False
        self._states: list[ScoreboardState] = []
        self._fps = fps

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def is_available(self) -> bool:
        return self._enabled and self._ocr is not None and self._ocr.is_available

    def should_process(self, frame_idx: int) -> bool:
        if not self.is_enabled:
            return False
        # Collect first few sample frames for ROI detection
        if not self._roi_detected and len(self._roi_frames) < 5:
            if frame_idx % self._sample_interval == 0:
                return True
        return frame_idx % self._sample_interval == 0

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> None:
        # Phase 1: collect frames for ROI detection
        if not self._roi_detected:
            self._roi_frames.append((frame_idx, frame.copy()))
            if len(self._roi_frames) >= 5:
                self._detect_roi()
            return

        # Phase 2: OCR on sampled frames
        self._ocr_frame(frame, frame_idx)

    def finalize(self) -> None:
        # If we collected ROI frames but never hit 5, detect with what we have
        if not self._roi_detected and self._roi_frames:
            self._detect_roi()
            # Process the collected ROI frames through OCR
            for idx, f in self._roi_frames:
                self._ocr_frame(f, idx)

        self._states = stabilize_scores(self._states, self._min_consistency)

    def get_states(self) -> list[ScoreboardState]:
        return self._states

    def _detect_roi(self) -> None:
        self._roi = detect_scoreboard_roi(self._roi_frames, self._image_shape)
        if self._roi is None:
            h, w = self._image_shape
            self._roi = (w // 4, 0, w * 3 // 4, int(h * 0.12))
            logger.debug("Scoreboard ROI fallback: {}", self._roi)
        self._roi_detected = True

        # Process the collected frames through OCR
        for idx, f in self._roi_frames:
            self._ocr_frame(f, idx)

    def _ocr_frame(self, frame: np.ndarray, frame_idx: int) -> None:
        time_s = frame_idx / self._fps if self._fps > 0 else 0.0

        if self._roi is None:
            self._states.append(ScoreboardState(frame=frame_idx, time_s=time_s, confidence=0.0))
            return

        x1, y1, x2, y2 = self._roi
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            self._states.append(
                ScoreboardState(
                    frame=frame_idx,
                    time_s=time_s,
                    roi_bbox_xyxy=self._roi,
                    confidence=0.0,
                )
            )
            return

        if self._ocr is None or not self._ocr.is_available:
            self._states.append(
                ScoreboardState(
                    frame=frame_idx,
                    time_s=time_s,
                    roi_bbox_xyxy=self._roi,
                    confidence=0.0,
                )
            )
            return

        raw_text, ocr_confidence = self._ocr.read_text(crop)
        parsed = parse_score_text(raw_text)
        parse_confidence = parsed.get("parse_confidence", 0.0)

        fsm_state = self._build_fsm_input(parsed)
        fsm_accepted = self._fsm.update(fsm_state) if fsm_state else False

        combined_confidence = ocr_confidence * parse_confidence
        if fsm_accepted:
            combined_confidence = min(combined_confidence * 1.2, 1.0)
        else:
            combined_confidence *= 0.5

        parsed_sets = parsed.get("sets") or None
        game_score = parsed.get("game_score")
        parsed_game_score = None
        if game_score:
            try:
                parsed_game_score = (int(game_score[0]), int(game_score[1]))
            except (ValueError, TypeError):
                pass

        self._states.append(
            ScoreboardState(
                frame=frame_idx,
                time_s=time_s,
                roi_bbox_xyxy=self._roi,
                raw_text=raw_text if raw_text else None,
                parsed_sets=parsed_sets if parsed_sets else None,
                parsed_game_score=parsed_game_score,
                confidence=combined_confidence,
            )
        )

    @staticmethod
    def _build_fsm_input(parsed: dict) -> dict | None:
        game_score = parsed.get("game_score")
        sets = parsed.get("sets")
        if not game_score and not sets:
            return None
        result = {}
        if game_score:
            result["game_score"] = (str(game_score[0]), str(game_score[1]))
        if sets:
            result["sets"] = sets
        return result
