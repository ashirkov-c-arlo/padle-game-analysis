from __future__ import annotations

import numpy as np
from loguru import logger

from src.schemas import ScoreboardState
from src.scoreboard.fsm import ScoreFSM
from src.scoreboard.ocr_engine import ScoreboardOCR
from src.scoreboard.parser import parse_score_text
from src.scoreboard.roi_detector import detect_scoreboard_roi
from src.scoreboard.scoreboard import stabilize_scores
from src.scoreboard.vlm_detector import detect_scoreboard_vlm, is_vlm_available


class ScoreboardFrameProcessor:
    """Wraps scoreboard OCR for single-pass pipeline. Samples every N frames."""

    def __init__(self, config: dict, fps: float, image_shape: tuple[int, int]):
        scoreboard_config = config.get("scoreboard", {})
        self._enabled = scoreboard_config.get("enabled", True)
        self._sample_interval = max(1, int(fps * scoreboard_config.get("sample_interval_s", 1.0)))
        self._min_consistency = scoreboard_config.get("min_consistency_frames", 3)
        self._image_shape = image_shape

        self._scoreboard_config = scoreboard_config
        self._fsm = ScoreFSM()
        self._roi: tuple[int, int, int, int] | None = None
        self._roi_frames: list[tuple[int, np.ndarray]] = []
        self._roi_detected = False
        self._states: list[ScoreboardState] = []
        self._fps = fps

        vlm_config = scoreboard_config.get("vlm", {})
        self._vlm_enabled = self._enabled and is_vlm_available(vlm_config)
        self._vlm_config = vlm_config
        self._vlm_failures = 0
        self._vlm_max_failures = vlm_config.get("max_failures", 3)

        self._ocr: ScoreboardOCR | None = None
        if self._enabled and not self._vlm_enabled:
            self._ocr = ScoreboardOCR(scoreboard_config)

        logger.debug(
            (
                "Scoreboard processor initialized: enabled={}, sample_interval_frames={}, "
                "min_consistency={}, image_shape={}, vlm_enabled={}, ocr_preloaded={}"
            ),
            self._enabled,
            self._sample_interval,
            self._min_consistency,
            self._image_shape,
            self._vlm_enabled,
            self._ocr is not None,
        )

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def is_available(self) -> bool:
        if not self._enabled:
            return False
        if self._vlm_enabled:
            return True
        return self._ocr is not None and self._ocr.is_available

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

        # Phase 2: VLM (primary) or OCR (fallback) on sampled frames
        self._process_detected_frame(frame, frame_idx)

    def finalize(self) -> None:
        # If we collected ROI frames but never hit 5, detect with what we have
        if not self._roi_detected and self._roi_frames:
            self._detect_roi()

        self._states = stabilize_scores(self._states, self._min_consistency)

    def get_states(self) -> list[ScoreboardState]:
        return self._states

    def _detect_roi(self) -> None:
        self._roi_detected = True

        if self._vlm_enabled:
            any_vlm_success = False
            for idx, f in self._roi_frames:
                vlm_result = detect_scoreboard_vlm(f, self._vlm_config)
                if vlm_result is not None:
                    self._vlm_failures = 0
                    any_vlm_success = True
                    time_s = idx / self._fps if self._fps > 0 else 0.0
                    if vlm_result.get("roi_bbox_xyxy") and self._roi is None:
                        self._roi = vlm_result["roi_bbox_xyxy"]
                    self._append_vlm_state(vlm_result, idx, time_s)
                else:
                    self._vlm_failures += 1

            if any_vlm_success:
                if self._roi is None:
                    self._roi = detect_scoreboard_roi(self._roi_frames, self._image_shape)
                    if self._roi is None:
                        h, w = self._image_shape
                        self._roi = (w // 4, 0, w * 3 // 4, int(h * 0.12))
                logger.debug("Scoreboard ROI from VLM: {}", self._roi)
                return

            logger.warning("VLM failed on all ROI frames, falling back to CV")

        self._roi = detect_scoreboard_roi(self._roi_frames, self._image_shape)
        if self._roi is None:
            h, w = self._image_shape
            self._roi = (w // 4, 0, w * 3 // 4, int(h * 0.12))
            logger.debug("Scoreboard ROI fallback: {}", self._roi)
        else:
            logger.debug("Scoreboard ROI selected: {}", self._roi)

        for idx, f in self._roi_frames:
            self._ocr_frame(f, idx)

    def _process_detected_frame(self, frame: np.ndarray, frame_idx: int) -> None:
        """Try VLM first, fall back to OCR."""
        if self._vlm_enabled and self._vlm_failures < self._vlm_max_failures:
            time_s = frame_idx / self._fps if self._fps > 0 else 0.0
            vlm_result = detect_scoreboard_vlm(frame, self._vlm_config)
            if vlm_result is not None:
                self._vlm_failures = 0
                if vlm_result.get("roi_bbox_xyxy") and self._roi is None:
                    self._roi = vlm_result["roi_bbox_xyxy"]
                self._append_vlm_state(vlm_result, frame_idx, time_s)
                return
            self._vlm_failures += 1
            if self._vlm_failures >= self._vlm_max_failures:
                logger.warning(
                    "VLM failed {} times consecutively, disabling for this video",
                    self._vlm_failures,
                )
        logger.warning("Using OCR fallback for frame {}", frame_idx)
        self._ocr_frame(frame, frame_idx)

    def _append_vlm_state(self, vlm_result: dict, frame_idx: int, time_s: float) -> None:
        roi = vlm_result.get("roi_bbox_xyxy") or self._roi
        parsed_sets = vlm_result.get("sets")
        game_score = vlm_result.get("game_score")

        parsed_game_score = None
        if game_score:
            try:
                parsed_game_score = (int(game_score[0]), int(game_score[1]))
            except (ValueError, TypeError):
                pass

        fsm_input = self._build_fsm_input({
            "sets": parsed_sets,
            "game_score": game_score,
        })
        fsm_accepted = self._fsm.update(fsm_input) if fsm_input else False

        confidence = vlm_result.get("confidence", 0.8)
        if fsm_accepted:
            confidence = min(confidence * 1.1, 1.0)
        else:
            confidence *= 0.6

        logger.debug(
            "Scoreboard VLM frame {}: roi={}, raw={!r}, conf={:.3f}, fsm={}",
            frame_idx, roi, vlm_result.get("raw_text"), confidence, fsm_accepted,
        )

        self._states.append(
            ScoreboardState(
                frame=frame_idx,
                time_s=time_s,
                roi_bbox_xyxy=roi,
                raw_text=vlm_result.get("raw_text"),
                parsed_sets=parsed_sets if parsed_sets else None,
                parsed_game_score=parsed_game_score,
                confidence=confidence,
            )
        )

    def _ensure_ocr(self) -> None:
        if self._ocr is None:
            logger.info("Loading PaddleOCR as fallback (VLM unavailable or failed)")
            self._ocr = ScoreboardOCR(self._scoreboard_config)

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

        self._ensure_ocr()
        if not self._ocr.is_available:
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
        logger.debug(
            (
                "Scoreboard frame {}: roi={}, raw_text={!r}, ocr_conf={:.3f}, "
                "parse_conf={:.3f}, fsm_accepted={}, combined_conf={:.3f}"
            ),
            frame_idx,
            self._roi,
            raw_text,
            ocr_confidence,
            parse_confidence,
            fsm_accepted,
            combined_confidence,
        )

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
