from __future__ import annotations

import re


def parse_score_text(raw_text: str) -> dict:
    """
    Parse OCR text into structured score.

    Expected formats:
    - "6 4 | 30 15" -> sets: [(6,4)], game: ("30", "15")
    - "6-4 30-15" -> same
    - "6 4 3 2 | 40 30" -> sets: [(6,4), (3,2)], game: ("40", "30")
    - "6-4 3-2 40-30" -> sets: [(6,4), (3,2)], game: ("40", "30")
    - "AD 40" -> game: ("AD", "40")

    Returns {
        "sets": [(int, int), ...],
        "game_score": (str, str),
        "parse_confidence": float
    }
    """
    if not raw_text or not raw_text.strip():
        return {"sets": [], "game_score": None, "parse_confidence": 0.0}

    text = raw_text.strip().upper()
    tokens = extract_digits(text)

    if not tokens:
        return {"sets": [], "game_score": None, "parse_confidence": 0.0}

    # Try pattern: "X-Y X-Y ... G-G" (dash-separated pairs)
    result = _try_dash_pattern(text)
    if result and result["parse_confidence"] > 0.5:
        return result

    # Try pattern: "X Y | G G" (pipe separator between sets and game)
    result = _try_pipe_pattern(text)
    if result and result["parse_confidence"] > 0.5:
        return result

    # Try pattern: space-separated tokens, infer structure from values
    result = _try_token_inference(tokens)
    if result and result["parse_confidence"] > 0.3:
        return result

    return {"sets": [], "game_score": None, "parse_confidence": 0.0}


def extract_digits(text: str) -> list[str]:
    """Extract digit groups and score-related tokens (AD, etc.)."""
    # Match numbers and score tokens like AD
    pattern = r"\b(\d+|AD|DEUCE)\b"
    return re.findall(pattern, text.upper())


def _try_dash_pattern(text: str) -> dict | None:
    """Try to parse dash-separated score pairs: '6-4 3-2 40-30'."""
    pairs = re.findall(r"(\d+|AD)\s*[-:]\s*(\d+|AD)", text)
    if not pairs:
        return None

    sets = []
    game_score = None

    for a, b in pairs:
        if _is_game_score_token(a) or _is_game_score_token(b):
            game_score = (a, b)
        elif _is_set_score(a, b):
            sets.append((int(a), int(b)))
        else:
            # Ambiguous — could be game score
            game_score = (a, b)

    confidence = 0.8 if sets or game_score else 0.3
    return {"sets": sets, "game_score": game_score, "parse_confidence": confidence}


def _try_pipe_pattern(text: str) -> dict | None:
    """Try to parse pipe-separated format: '6 4 | 30 15'."""
    if "|" not in text:
        return None

    parts = text.split("|")
    if len(parts) != 2:
        return None

    set_tokens = extract_digits(parts[0])
    game_tokens = extract_digits(parts[1])

    sets = []
    game_score = None

    # Parse set scores (pairs of numbers)
    if len(set_tokens) >= 2 and len(set_tokens) % 2 == 0:
        for i in range(0, len(set_tokens), 2):
            a, b = set_tokens[i], set_tokens[i + 1]
            if _is_set_score(a, b):
                sets.append((int(a), int(b)))

    # Parse game score
    if len(game_tokens) >= 2:
        game_score = (game_tokens[0], game_tokens[1])

    confidence = 0.85 if sets and game_score else 0.5
    return {"sets": sets, "game_score": game_score, "parse_confidence": confidence}


def _try_token_inference(tokens: list[str]) -> dict | None:
    """
    Infer score structure from space-separated tokens.
    Strategy: game scores use values in {0,15,30,40,AD}, set scores use 0-7.
    """
    if len(tokens) < 2:
        return None

    GAME_VALUES = {"0", "15", "30", "40", "AD"}
    sets = []
    game_score = None

    # Check if last two tokens look like a game score
    if tokens[-2] in GAME_VALUES and tokens[-1] in GAME_VALUES:
        game_score = (tokens[-2], tokens[-1])
        remaining = tokens[:-2]
    else:
        remaining = tokens

    # Parse remaining as set scores (pairs)
    if len(remaining) >= 2 and len(remaining) % 2 == 0:
        for i in range(0, len(remaining), 2):
            a, b = remaining[i], remaining[i + 1]
            if a.isdigit() and b.isdigit() and _is_set_score(a, b):
                sets.append((int(a), int(b)))

    if not sets and not game_score:
        return None

    confidence = 0.6
    if sets and game_score:
        confidence = 0.75
    elif game_score:
        confidence = 0.5

    return {"sets": sets, "game_score": game_score, "parse_confidence": confidence}


def _is_game_score_token(token: str) -> bool:
    """Check if token is a valid game score value."""
    return token in {"0", "15", "30", "40", "AD", "DEUCE"}


def _is_set_score(a: str, b: str) -> bool:
    """Check if a pair of values could be a set score."""
    try:
        ia, ib = int(a), int(b)
    except ValueError:
        return False
    # Set scores are 0-7 for each side
    return 0 <= ia <= 7 and 0 <= ib <= 7
