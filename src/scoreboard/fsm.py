from __future__ import annotations

VALID_GAME_SCORES = ["0", "15", "30", "40", "AD"]

VALID_GAME_TRANSITIONS = {
    "0": ["15"],
    "15": ["30"],
    "30": ["40"],
    "40": ["AD", "0"],  # 0 means game won, reset
    "AD": ["40", "0"],  # back to deuce or game won
}


class ScoreFSM:
    """
    Padel/tennis scoring FSM to validate and filter OCR results.

    Game scoring: 0 -> 15 -> 30 -> 40 -> game (or deuce/AD)
    Set scoring: games count up, set won at 6 (or 7 in tiebreak)
    """

    def __init__(self) -> None:
        """Initialize with 0-0 state."""
        self.sets: list[tuple[int, int]] = []
        self.current_set: tuple[int, int] = (0, 0)
        self.game_score: tuple[str, str] = ("0", "0")

    def is_valid_transition(self, new_state: dict) -> bool:
        """
        Check if transitioning to new_state is valid from current state.
        Reject impossible jumps (e.g., 0->40, set score going backward).
        """
        new_game = new_state.get("game_score")
        new_sets = new_state.get("sets")
        new_current_set = new_state.get("current_set")

        # Validate game score transition
        if new_game is not None:
            if not self._is_valid_game_transition(new_game):
                return False

        # Validate set score transition
        if new_current_set is not None:
            if not self._is_valid_set_transition(new_current_set):
                return False

        # Validate completed sets (should not change)
        if new_sets is not None:
            if not self._is_valid_sets_history(new_sets):
                return False

        return True

    def update(self, new_state: dict) -> bool:
        """
        Try to update to new state. Returns True if accepted.
        If invalid transition, reject and keep current state.
        """
        if not self.is_valid_transition(new_state):
            return False

        new_game = new_state.get("game_score")
        new_sets = new_state.get("sets")
        new_current_set = new_state.get("current_set")

        if new_game is not None:
            self.game_score = (str(new_game[0]), str(new_game[1]))

        if new_sets is not None:
            self.sets = list(new_sets)

        if new_current_set is not None:
            self.current_set = tuple(new_current_set)

        # Detect game won (score reset to 0-0 with set score change)
        if new_game == ("0", "0") and new_current_set is not None:
            # A game was just completed
            pass

        return True

    def get_state(self) -> dict:
        """Return current validated score state."""
        return {
            "sets": list(self.sets),
            "current_set": self.current_set,
            "game_score": self.game_score,
        }

    def _is_valid_game_transition(self, new_game: tuple[str, str]) -> bool:
        """Check if game score transition is valid."""
        new_a, new_b = str(new_game[0]), str(new_game[1])

        # Both must be valid game scores
        if new_a not in VALID_GAME_SCORES or new_b not in VALID_GAME_SCORES:
            return False

        cur_a, cur_b = self.game_score

        # Same score is always valid (no change)
        if new_a == cur_a and new_b == cur_b:
            return True

        # Reset to 0-0 is valid (game completed)
        if new_a == "0" and new_b == "0":
            return True

        # Only one side should change per point
        a_changed = new_a != cur_a
        b_changed = new_b != cur_b

        if a_changed and b_changed:
            # Both changed — only valid for deuce reset (AD->40 on other side)
            # or game reset to 0-0 (handled above)
            if cur_a == "AD" and new_a == "40" and new_b == "40":
                return True
            if cur_b == "AD" and new_b == "40" and new_a == "40":
                return True
            return False

        # Check the side that changed
        if a_changed:
            return self._is_valid_score_step(cur_a, new_a, cur_b)
        if b_changed:
            return self._is_valid_score_step(cur_b, new_b, cur_a)

        return True

    def _is_valid_score_step(
        self, old_score: str, new_score: str, opponent_score: str
    ) -> bool:
        """Check if a single player's score change is valid."""
        valid_nexts = VALID_GAME_TRANSITIONS.get(old_score, [])
        if new_score in valid_nexts:
            # AD is only valid when opponent is at 40 (deuce situation)
            if new_score == "AD" and opponent_score != "40":
                return False
            return True
        return False

    def _is_valid_set_transition(self, new_set: tuple[int, int]) -> bool:
        """Check if set score transition is valid."""
        cur_a, cur_b = self.current_set
        new_a, new_b = new_set

        # Set scores should not decrease
        if new_a < cur_a or new_b < cur_b:
            return False

        # Only one side should gain a game at a time
        delta_a = new_a - cur_a
        delta_b = new_b - cur_b

        if delta_a > 1 or delta_b > 1:
            return False

        if delta_a > 0 and delta_b > 0:
            return False

        # Max valid game count is 7 (tiebreak)
        if new_a > 7 or new_b > 7:
            return False

        # Check if set is being won
        if new_a >= 6 or new_b >= 6:
            # Valid set-winning scores
            if new_a == 7 or new_b == 7:
                # Tiebreak: must be 7-5 or 7-6
                pass
            elif new_a == 6 and new_b <= 4:
                pass  # Won 6-0 to 6-4
            elif new_b == 6 and new_a <= 4:
                pass  # Won 6-0 to 6-4

        return True

    def _is_valid_sets_history(self, new_sets: list[tuple[int, int]]) -> bool:
        """Validate that completed sets history is consistent."""
        # New history should be same as or extend current history
        if len(new_sets) < len(self.sets):
            return False

        # Existing sets should not change
        for i, existing_set in enumerate(self.sets):
            if i < len(new_sets) and new_sets[i] != existing_set:
                return False

        # New completed sets should have valid final scores
        for a, b in new_sets[len(self.sets) :]:
            if not _is_valid_final_set_score(a, b):
                return False

        return True


def _is_valid_final_set_score(a: int, b: int) -> bool:
    """Check if a completed set score is valid."""
    # Standard set: 6-0 through 6-4, or 7-5, 7-6
    if a == 6 and b <= 4:
        return True
    if b == 6 and a <= 4:
        return True
    if a == 7 and b in (5, 6):
        return True
    if b == 7 and a in (5, 6):
        return True
    return False
