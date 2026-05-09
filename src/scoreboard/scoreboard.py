from __future__ import annotations

from loguru import logger

from src.schemas import ScoreboardState
from src.scoreboard.fsm import ScoreFSM
from src.scoreboard.ocr_engine import ScoreboardOCR
from src.scoreboard.parser import parse_score_text
from src.scoreboard.roi_detector import detect_scoreboard_roi
from src.video_io.reader import get_video_info, read_frame


def process_scoreboard(
    video_path: str,
    config: dict,
) -> list[ScoreboardState]:
    """
    Full scoreboard OCR pipeline:
    1. Sample frames at config interval
    2. Detect scoreboard ROI (or use fixed position if stable)
    3. For each sample frame:
       - Crop scoreboard region
       - Run OCR
       - Parse score text
       - Validate with FSM
    4. Interpolate score between samples (score doesn't change between points)
    5. Return timeline of ScoreboardState
    """
    scoreboard_config = config.get("scoreboard", {})

    if not scoreboard_config.get("enabled", True):
        logger.info("Scoreboard OCR disabled in config")
        return []

    # Initialize OCR engine
    ocr = ScoreboardOCR(scoreboard_config)
    if not ocr.is_available:
        logger.warning("No OCR engine available, returning empty scoreboard states")
        return []

    # Get video info
    video_info = get_video_info(video_path)
    fps = video_info["fps"]
    total_frames = video_info["total_frames"]
    sample_interval_s = scoreboard_config.get("sample_interval_s", 1.0)
    sample_interval_frames = max(1, int(fps * sample_interval_s))

    # Determine sample frame indices
    sample_indices = list(range(0, total_frames, sample_interval_frames))
    logger.info(
        "Scoreboard OCR: {} sample frames at {:.1f}s interval",
        len(sample_indices),
        sample_interval_s,
    )

    # Detect ROI from first few frames
    roi_sample_count = min(5, len(sample_indices))
    roi_frames = []
    for idx in sample_indices[:roi_sample_count]:
        frame = read_frame(video_path, idx)
        roi_frames.append((idx, frame))

    image_shape = (video_info["height"], video_info["width"])
    roi = detect_scoreboard_roi(roi_frames, image_shape)

    if roi is None:
        logger.warning("Could not detect scoreboard ROI, using top strip as fallback")
        # Fallback: top 12% of frame, center 50% width
        h, w = image_shape
        roi = (w // 4, 0, w * 3 // 4, int(h * 0.12))

    # Process each sample frame
    fsm = ScoreFSM()
    states: list[ScoreboardState] = []

    for frame_idx in sample_indices:
        time_s = frame_idx / fps if fps > 0 else 0.0

        try:
            frame = read_frame(video_path, frame_idx)
        except (ValueError, FileNotFoundError):
            states.append(
                ScoreboardState(frame=frame_idx, time_s=time_s, confidence=0.0)
            )
            continue

        # Crop ROI
        x1, y1, x2, y2 = roi
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            states.append(
                ScoreboardState(frame=frame_idx, time_s=time_s, confidence=0.0)
            )
            continue

        # OCR
        raw_text, ocr_confidence = ocr.read_text(crop)

        # Parse
        parsed = parse_score_text(raw_text)
        parse_confidence = parsed.get("parse_confidence", 0.0)

        # Validate with FSM
        fsm_state = _build_fsm_input(parsed)
        fsm_accepted = fsm.update(fsm_state) if fsm_state else False

        # Combine confidences
        combined_confidence = ocr_confidence * parse_confidence
        if fsm_accepted:
            combined_confidence = min(combined_confidence * 1.2, 1.0)
        else:
            combined_confidence *= 0.5

        # Build state
        parsed_sets = parsed.get("sets") or None
        game_score = parsed.get("game_score")
        parsed_game_score = None
        if game_score:
            try:
                parsed_game_score = (int(game_score[0]), int(game_score[1]))
            except (ValueError, TypeError):
                # AD or other non-numeric scores
                pass

        state = ScoreboardState(
            frame=frame_idx,
            time_s=time_s,
            raw_text=raw_text if raw_text else None,
            parsed_sets=parsed_sets if parsed_sets else None,
            parsed_game_score=parsed_game_score,
            confidence=combined_confidence,
        )
        states.append(state)

    # Stabilize results
    min_consistency = scoreboard_config.get("min_consistency_frames", 3)
    states = stabilize_scores(states, min_consistency_frames=min_consistency)

    logger.info(
        "Scoreboard OCR complete: {} states, {} with valid scores",
        len(states),
        sum(1 for s in states if s.parsed_sets or s.parsed_game_score),
    )
    return states


def stabilize_scores(
    states: list[ScoreboardState],
    min_consistency_frames: int = 3,
) -> list[ScoreboardState]:
    """
    Post-process to remove OCR noise:
    - A score change must be seen in N consecutive samples to be accepted
    - Smooth out single-frame OCR errors
    """
    if len(states) < min_consistency_frames:
        return states

    stabilized: list[ScoreboardState] = []
    last_stable_score: tuple | None = None

    i = 0
    while i < len(states):
        current = states[i]
        current_score = _extract_score_key(current)

        if current_score is None:
            # No score detected, keep last stable
            stabilized.append(
                ScoreboardState(
                    frame=current.frame,
                    time_s=current.time_s,
                    raw_text=current.raw_text,
                    parsed_sets=None,
                    parsed_game_score=None,
                    confidence=current.confidence * 0.5,
                )
            )
            i += 1
            continue

        if current_score == last_stable_score:
            # Same as last stable — accept
            stabilized.append(current)
            i += 1
            continue

        # New score detected — check consistency
        consistent_count = 1
        for j in range(i + 1, min(i + min_consistency_frames, len(states))):
            if _extract_score_key(states[j]) == current_score:
                consistent_count += 1
            else:
                break

        if consistent_count >= min_consistency_frames:
            # Accept the new score
            last_stable_score = current_score
            stabilized.append(current)
        else:
            # Noise — revert to last stable
            stabilized.append(
                ScoreboardState(
                    frame=current.frame,
                    time_s=current.time_s,
                    raw_text=current.raw_text,
                    parsed_sets=None,
                    parsed_game_score=None,
                    confidence=current.confidence * 0.3,
                )
            )
        i += 1

    return stabilized


def _extract_score_key(state: ScoreboardState) -> tuple | None:
    """Extract a comparable score key from a state."""
    if state.parsed_sets is None and state.parsed_game_score is None:
        return None
    sets_key = tuple(state.parsed_sets) if state.parsed_sets else ()
    game_key = state.parsed_game_score if state.parsed_game_score else ()
    return (sets_key, game_key)


def _build_fsm_input(parsed: dict) -> dict | None:
    """Convert parser output to FSM input format."""
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
