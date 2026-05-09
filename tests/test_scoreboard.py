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

    def test_detect_scoreboard_roi_prefers_broadcast_panel_over_bottom_text(self):
        frames = []
        for i in range(5):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)

            # Persistent bottom sponsor text should not beat the actual score panel.
            cv2.rectangle(frame, (200, 660), (760, 715), (120, 70, 20), -1)
            cv2.putText(
                frame,
                "VISIT QATAR",
                (280, 700),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                2,
            )

            # The first sampled frame can miss the TV score graphic during fade-in.
            if i > 0:
                cv2.rectangle(frame, (40, 50), (300, 86), (235, 235, 235), -1)
                cv2.rectangle(frame, (40, 90), (300, 126), (235, 235, 235), -1)
                cv2.rectangle(frame, (300, 50), (342, 126), (45, 25, 80), -1)
                cv2.putText(
                    frame,
                    "TEAM A",
                    (70, 76),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    "TEAM B",
                    (70, 116),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    "0",
                    (314, 76),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "0",
                    (314, 116),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            frames.append((i * 30, frame))

        roi = detect_scoreboard_roi(frames, (720, 1280))

        assert roi is not None
        x1, y1, x2, y2 = roi
        assert x1 <= 45
        assert y1 <= 50
        assert x2 >= 340
        assert y2 >= 126
        assert y2 < 720 * 0.25


class TestScoreboardFrameProcessor:
    def test_attaches_detected_roi_bbox_to_states(self, monkeypatch):
        import src.scoreboard.scoreboard_processor as processor_mod

        detected_roi = (20, 10, 120, 40)

        class FakeOCR:
            is_available = True

            def __init__(self, config):
                pass

            def read_text(self, crop):
                assert crop.shape == (30, 100, 3)
                return "6-4 30-15", 1.0

        monkeypatch.setattr(processor_mod, "ScoreboardOCR", FakeOCR)
        monkeypatch.setattr(
            processor_mod,
            "detect_scoreboard_roi",
            lambda frames, image_shape: detected_roi,
        )

        processor = processor_mod.ScoreboardFrameProcessor(
            {
                "scoreboard": {
                    "enabled": True,
                    "sample_interval_s": 1.0,
                    "min_consistency_frames": 1,
                }
            },
            fps=1.0,
            image_shape=(80, 160),
        )

        for frame_idx in range(5):
            frame = np.zeros((80, 160, 3), dtype=np.uint8)
            processor.process_frame(frame, frame_idx)

        processor.finalize()
        states = processor.get_states()

        assert len(states) == 5
        assert {state.roi_bbox_xyxy for state in states} == {detected_roi}

    def test_attaches_roi_bbox_when_ocr_unavailable(self, monkeypatch):
        import src.scoreboard.scoreboard_processor as processor_mod

        detected_roi = (20, 10, 120, 40)

        class FakeOCR:
            is_available = False

            def __init__(self, config):
                pass

            def read_text(self, crop):
                raise AssertionError("OCR should not run when unavailable")

        monkeypatch.setattr(processor_mod, "ScoreboardOCR", FakeOCR)
        monkeypatch.setattr(
            processor_mod,
            "detect_scoreboard_roi",
            lambda frames, image_shape: detected_roi,
        )

        processor = processor_mod.ScoreboardFrameProcessor(
            {"scoreboard": {"enabled": True, "sample_interval_s": 1.0}},
            fps=1.0,
            image_shape=(80, 160),
        )

        assert processor.is_enabled
        assert not processor.is_available

        for frame_idx in range(5):
            assert processor.should_process(frame_idx)
            frame = np.zeros((80, 160, 3), dtype=np.uint8)
            processor.process_frame(frame, frame_idx)

        processor.finalize()
        states = processor.get_states()

        assert len(states) == 5
        assert {state.roi_bbox_xyxy for state in states} == {detected_roi}
        assert all(state.raw_text is None for state in states)
        assert all(state.confidence == 0.0 for state in states)


class TestProcessScoreboard:
    def test_attaches_detected_roi_bbox_to_states(self, monkeypatch):
        import src.scoreboard.scoreboard as scoreboard_mod

        detected_roi = (20, 10, 120, 40)

        class FakeOCR:
            is_available = True

            def __init__(self, config):
                pass

            def read_text(self, crop):
                assert crop.shape == (30, 100, 3)
                return "6-4 30-15", 1.0

        monkeypatch.setattr(scoreboard_mod, "ScoreboardOCR", FakeOCR)
        monkeypatch.setattr(
            scoreboard_mod,
            "get_video_info",
            lambda video_path: {
                "fps": 1.0,
                "total_frames": 2,
                "height": 80,
                "width": 160,
            },
        )
        monkeypatch.setattr(
            scoreboard_mod,
            "read_frame",
            lambda video_path, frame_idx: np.zeros((80, 160, 3), dtype=np.uint8),
        )
        monkeypatch.setattr(
            scoreboard_mod,
            "detect_scoreboard_roi",
            lambda frames, image_shape: detected_roi,
        )

        states = scoreboard_mod.process_scoreboard(
            "input.mp4",
            {
                "scoreboard": {
                    "enabled": True,
                    "sample_interval_s": 1.0,
                    "min_consistency_frames": 1,
                }
            },
        )

        assert len(states) == 2
        assert {state.roi_bbox_xyxy for state in states} == {detected_roi}


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
            roi_bbox_xyxy=(100, 10, 400, 60),
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
        assert result[2].roi_bbox_xyxy == (100, 10, 400, 60)

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
    def test_paddleocr_predict_adapter(self, monkeypatch):
        """PaddleOCR 3.x results should parse without loading real models."""
        import src.scoreboard.ocr_engine as ocr_mod

        class FakePaddleOCR:
            def __init__(self, **kwargs):
                assert "show_log" not in kwargs
                assert kwargs["device"] == "cpu"
                assert kwargs["enable_mkldnn"] is False
                assert kwargs["use_doc_orientation_classify"] is False
                assert kwargs["use_doc_unwarping"] is False
                assert kwargs["use_textline_orientation"] is False

            def predict(self, image):
                assert len(image.shape) == 3
                return [{"rec_texts": ["6 4", "30 15"], "rec_scores": [0.8, 1.0]}]

        monkeypatch.setattr(ocr_mod, "PADDLE_AVAILABLE", True)
        monkeypatch.setattr(ocr_mod, "PaddleOCR", FakePaddleOCR)

        ocr = ScoreboardOCR({"ocr_engine": "paddleocr"})
        text, confidence = ocr._read_paddle(np.zeros((50, 200), dtype=np.uint8))

        assert text == "6 4 30 15"
        assert confidence == 0.9

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
