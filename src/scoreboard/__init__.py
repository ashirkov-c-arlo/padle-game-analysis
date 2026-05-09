from __future__ import annotations

from src.scoreboard.fsm import ScoreFSM
from src.scoreboard.ocr_engine import ScoreboardOCR
from src.scoreboard.parser import parse_score_text
from src.scoreboard.roi_detector import detect_scoreboard_roi
from src.scoreboard.scoreboard import process_scoreboard

__all__ = [
    "process_scoreboard",
    "ScoreFSM",
    "ScoreboardOCR",
    "detect_scoreboard_roi",
    "parse_score_text",
]
