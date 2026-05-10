from __future__ import annotations

from loguru import logger

from src.schemas import ScoreboardState
from src.video_io.reader import get_video_info


def process_scoreboard(
    video_path: str,
    config: dict,
) -> list[ScoreboardState]:
    """Compatibility wrapper for the single-pass scoreboard processor."""
    scoreboard_config = config.get("scoreboard", {})
    if not scoreboard_config.get("enabled", True):
        logger.info("Scoreboard OCR disabled in config")
        return []

    from src.scoreboard.scoreboard_processor import ScoreboardFrameProcessor
    from src.video_io.single_pass import run_single_pass

    video_info = get_video_info(video_path)
    fps = video_info["fps"]
    image_shape = (video_info["height"], video_info["width"])

    processor = ScoreboardFrameProcessor(config, fps, image_shape)
    run_single_pass(video_path, [processor])
    states = processor.get_states()

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
                    roi_bbox_xyxy=current.roi_bbox_xyxy,
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
                    roi_bbox_xyxy=current.roi_bbox_xyxy,
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

