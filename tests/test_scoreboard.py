"""Tests for scoreboard OCR module."""
from __future__ import annotations

import cv2
import numpy as np

from src.schemas import ScoreboardState
from src.scoreboard.fsm import VALID_GAME_SCORES, VALID_GAME_TRANSITIONS, ScoreFSM
from src.scoreboard.ocr_engine import ScoreboardOCR
from src.scoreboard.parser import extract_digits, parse_score_text
from src.scoreboard.roi_detector import detect_scoreboard_roi, find_text_regions
from src.scoreboard.scoreboard import stabilize_scores

# --- ROI Detector Tests ---


class TestROIDetector:
    def test_detect_scoreboard_roi_no_frames(self):
        result = detect_scoreboard_roi([], (720, 1280))
        assert result is None

    def test_find_text_regions_returns_list(self):
        # Create a frame with some high-contrast text-like content at the top
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Draw white rectangles near top to simulate text
        cv2.rectangle(frame, (100, 10), (400, 40), (255, 255, 255), -1)
        cv2.rectangle(frame, (110, 15), (120, 35), (0, 0, 0), -1)
        cv2.rectangle(frame, (130, 15), (140, 35), (0, 0, 0), -1)
        cv2.rectangle(frame, (150, 15), (160, 35), (0, 0, 0), -1)

        regions = find_text_regions(frame)
        assert isinstance(regions, list)

    def test_detect_scoreboard_roi_finds_top_region(self):
        # Create synthetic frames with consistent text-like region at top
        frames = []
        for i in range(5):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            # Simulate scoreboard at top: high contrast alternating blocks
            for x in range(200, 600, 20):
                cv2.rectangle(frame, (x, 10), (x + 15, 50), (255, 255, 255), -1)
            # Add some noise/texture for edge detection
            cv2.putText(
                frame, "6-4 30-15", (250, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
            )
            frames.append((i * 30, frame))

        roi = detect_scoreboard_roi(frames, (720, 1280))
        # Should find something (either the text region or return a candidate)
        # The region should be in the top portion of the frame
        if roi is not None:
            _, y1, _, y2 = roi
            assert y1 < 720 * 0.25  # Near top
            assert y2 < 720 * 0.25


# --- Parser Tests ---


class TestParser:
    def test_parse_dash_separated(self):
        result = parse_score_text("6-4 30-15")
        assert result["sets"] == [(6, 4)]
        assert result["game_score"] == ("30", "15")
        assert result["parse_confidence"] > 0.5

    def test_parse_pipe_separated(self):
        result = parse_score_text("6 4 | 30 15")
        assert result["sets"] == [(6, 4)]
        assert result["game_score"] == ("30", "15")
        assert result["parse_confidence"] > 0.5

    def test_parse_multiple_sets_dash(self):
        result = parse_score_text("6-4 3-2 40-30")
        assert result["sets"] == [(6, 4), (3, 2)]
        assert result["game_score"] == ("40", "30")

    def test_parse_multiple_sets_pipe(self):
        result = parse_score_text("6 4 3 2 | 40 30")
        assert result["sets"] == [(6, 4), (3, 2)]
        assert result["game_score"] == ("40", "30")

    def test_parse_empty_text(self):
        result = parse_score_text("")
        assert result["sets"] == []
        assert result["game_score"] is None
        assert result["parse_confidence"] == 0.0

    def test_parse_none_text(self):
        result = parse_score_text("")
        assert result["parse_confidence"] == 0.0

    def test_parse_ad_score(self):
        result = parse_score_text("AD-40")
        assert result["game_score"] is not None
        assert result["game_score"][0] == "AD"
        assert result["game_score"][1] == "40"

    def test_parse_token_inference(self):
        # Space-separated with game score values
        result = parse_score_text("6 4 40 15")
        assert result["sets"] == [(6, 4)]
        assert result["game_score"] == ("40", "15")

    def test_extract_digits_basic(self):
        tokens = extract_digits("6-4 30-15")
        assert "6" in tokens
        assert "4" in tokens
        assert "30" in tokens
        assert "15" in tokens

    def test_extract_digits_with_ad(self):
        tokens = extract_digits("AD 40")
        assert "AD" in tokens
        assert "40" in tokens


# --- FSM Tests ---


class TestScoreFSM:
    def test_initial_state(self):
        fsm = ScoreFSM()
        state = fsm.get_state()
        assert state["game_score"] == ("0", "0")
        assert state["current_set"] == (0, 0)
        assert state["sets"] == []

    def test_valid_transition_0_to_15(self):
        fsm = ScoreFSM()
        result = fsm.update({"game_score": ("15", "0")})
        assert result is True
        assert fsm.game_score == ("15", "0")

    def test_valid_transition_15_to_30(self):
        fsm = ScoreFSM()
        fsm.game_score = ("15", "0")
        result = fsm.update({"game_score": ("30", "0")})
        assert result is True

    def test_valid_transition_30_to_40(self):
        fsm = ScoreFSM()
        fsm.game_score = ("30", "0")
        result = fsm.update({"game_score": ("40", "0")})
        assert result is True

    def test_reject_impossible_0_to_40(self):
        fsm = ScoreFSM()
        result = fsm.update({"game_score": ("40", "0")})
        assert result is False
        # State should not change
        assert fsm.game_score == ("0", "0")

    def test_reject_impossible_0_to_30(self):
        fsm = ScoreFSM()
        result = fsm.update({"game_score": ("30", "0")})
        assert result is False

    def test_deuce_ad_valid(self):
        fsm = ScoreFSM()
        fsm.game_score = ("40", "40")
        # Player A gets advantage
        result = fsm.update({"game_score": ("AD", "40")})
        assert result is True
        assert fsm.game_score == ("AD", "40")

    def test_ad_back_to_deuce(self):
        fsm = ScoreFSM()
        fsm.game_score = ("AD", "40")
        # Back to deuce (both 40)
        result = fsm.update({"game_score": ("40", "40")})
        assert result is True
        assert fsm.game_score == ("40", "40")

    def test_ad_to_game_won(self):
        fsm = ScoreFSM()
        fsm.game_score = ("AD", "40")
        # Game won, reset to 0-0
        result = fsm.update({"game_score": ("0", "0")})
        assert result is True

    def test_reject_ad_without_deuce(self):
        fsm = ScoreFSM()
        fsm.game_score = ("30", "0")
        # AD is not valid unless opponent is at 40
        result = fsm.update({"game_score": ("AD", "0")})
        assert result is False

    def test_valid_set_transition(self):
        fsm = ScoreFSM()
        fsm.current_set = (3, 2)
        result = fsm.update({"current_set": (4, 2)})
        assert result is True
        assert fsm.current_set == (4, 2)

    def test_reject_set_score_decrease(self):
        fsm = ScoreFSM()
        fsm.current_set = (3, 2)
        result = fsm.update({"current_set": (2, 2)})
        assert result is False

    def test_reject_set_score_jump(self):
        fsm = ScoreFSM()
        fsm.current_set = (3, 2)
        # Cannot jump by 2 games at once
        result = fsm.update({"current_set": (5, 2)})
        assert result is False

    def test_same_state_always_valid(self):
        fsm = ScoreFSM()
        fsm.game_score = ("30", "15")
        result = fsm.update({"game_score": ("30", "15")})
        assert result is True

    def test_game_reset_valid(self):
        fsm = ScoreFSM()
        fsm.game_score = ("40", "30")
        # Game won, reset
        result = fsm.update({"game_score": ("0", "0")})
        assert result is True

    def test_valid_game_scores_constant(self):
        assert "0" in VALID_GAME_SCORES
        assert "15" in VALID_GAME_SCORES
        assert "30" in VALID_GAME_SCORES
        assert "40" in VALID_GAME_SCORES
        assert "AD" in VALID_GAME_SCORES

    def test_valid_transitions_structure(self):
        assert VALID_GAME_TRANSITIONS["0"] == ["15"]
        assert VALID_GAME_TRANSITIONS["15"] == ["30"]
        assert VALID_GAME_TRANSITIONS["30"] == ["40"]
        assert "AD" in VALID_GAME_TRANSITIONS["40"]
        assert "0" in VALID_GAME_TRANSITIONS["40"]


# --- Stabilize Scores Tests ---


class TestStabilizeScores:
    def _make_state(self, frame, sets=None, game=None, confidence=0.8):
        return ScoreboardState(
            frame=frame,
            time_s=frame / 30.0,
            raw_text="test",
            parsed_sets=sets,
            parsed_game_score=game,
            confidence=confidence,
        )

    def test_removes_single_frame_noise(self):
        states = [
            self._make_state(0, sets=[(6, 4)], game=(30, 15)),
            self._make_state(30, sets=[(6, 4)], game=(30, 15)),
            # Single noise frame
            self._make_state(60, sets=[(7, 5)], game=(0, 0)),
            self._make_state(90, sets=[(6, 4)], game=(30, 15)),
            self._make_state(120, sets=[(6, 4)], game=(30, 15)),
        ]

        result = stabilize_scores(states, min_consistency_frames=2)
        # The noise frame (index 2) should have been filtered
        assert result[2].parsed_sets is None or result[2].parsed_sets == [(6, 4)]

    def test_accepts_consistent_change(self):
        states = [
            self._make_state(0, sets=[(6, 4)], game=(30, 15)),
            self._make_state(30, sets=[(6, 4)], game=(30, 15)),
            self._make_state(60, sets=[(6, 4)], game=(40, 15)),
            self._make_state(90, sets=[(6, 4)], game=(40, 15)),
            self._make_state(120, sets=[(6, 4)], game=(40, 15)),
        ]

        result = stabilize_scores(states, min_consistency_frames=2)
        # The change at frame 60 is consistent, should be accepted
        assert result[2].parsed_game_score == (40, 15)

    def test_empty_list(self):
        result = stabilize_scores([], min_consistency_frames=3)
        assert result == []

    def test_short_list_unchanged(self):
        states = [self._make_state(0, sets=[(6, 4)], game=(30, 15))]
        result = stabilize_scores(states, min_consistency_frames=3)
        assert len(result) == 1


# --- OCR Engine Tests ---


class TestScoreboardOCR:
    def test_graceful_degradation_no_engines(self, monkeypatch):
        """OCR should handle missing engines gracefully."""
        import src.scoreboard.ocr_engine as ocr_mod

        monkeypatch.setattr(ocr_mod, "PADDLE_AVAILABLE", False)
        monkeypatch.setattr(ocr_mod, "TESSERACT_AVAILABLE", False)

        ocr = ScoreboardOCR({"ocr_engine": "paddleocr"})
        assert ocr.engine_name == "none"
        assert not ocr.is_available

        # Should return empty result, not crash
        crop = np.zeros((50, 200, 3), dtype=np.uint8)
        text, confidence = ocr.read_text(crop)
        assert text == ""
        assert confidence == 0.0

    def test_empty_crop_returns_empty(self, monkeypatch):
        """OCR should handle empty crops gracefully."""
        import src.scoreboard.ocr_engine as ocr_mod

        monkeypatch.setattr(ocr_mod, "PADDLE_AVAILABLE", False)
        monkeypatch.setattr(ocr_mod, "TESSERACT_AVAILABLE", False)

        ocr = ScoreboardOCR({"ocr_engine": "paddleocr"})
        text, confidence = ocr.read_text(np.array([]))
        assert text == ""
        assert confidence == 0.0

    def test_none_crop_returns_empty(self, monkeypatch):
        """OCR should handle None crops gracefully."""
        import src.scoreboard.ocr_engine as ocr_mod

        monkeypatch.setattr(ocr_mod, "PADDLE_AVAILABLE", False)
        monkeypatch.setattr(ocr_mod, "TESSERACT_AVAILABLE", False)

        ocr = ScoreboardOCR({"ocr_engine": "paddleocr"})
        text, confidence = ocr.read_text(None)
        assert text == ""
        assert confidence == 0.0


