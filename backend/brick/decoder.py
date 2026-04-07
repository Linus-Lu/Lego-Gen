"""State machine for constrained brick token decoding.

Tracks which token type comes next in the brick sequence format:
  dim -> " (" -> x -> "," -> y -> "," -> z -> ") #" -> color -> "\\n" -> (repeat or EOS)

States:
  0 = dim        (e.g. "2x4")
  1 = open       (" (")
  2 = x coord    (0..19)
  3 = comma1     (",")
  4 = y coord    (0..19)
  5 = comma2     (",")
  6 = z coord    (0..19)
  7 = close      (") #")
  8 = color      (6-char hex)
  9 = newline    ("\\n")
"""

from backend.brick.constants import ALLOWED_DIMS, ALLOWED_COLORS, WORLD_DIM

_POSITIONS = [str(i) for i in range(WORLD_DIM)]


class BrickTokenConstraint:
    """Constrained decoding helper that exposes which tokens are valid next."""

    def __init__(self) -> None:
        self.state = 0

    def get_allowed_strings(self) -> list[str]:
        """Return the list of strings that are valid at the current state."""
        if self.state == 0:
            return list(ALLOWED_DIMS)
        elif self.state == 1:
            return [" ("]
        elif self.state in (2, 4, 6):
            return _POSITIONS
        elif self.state in (3, 5):
            return [","]
        elif self.state == 7:
            return [") #"]
        elif self.state == 8:
            return list(ALLOWED_COLORS)
        elif self.state == 9:
            return ["\n"]
        return []

    def feed(self, token_str: str) -> None:
        """Advance the state machine by one token."""
        self.state = (self.state + 1) % 10

    def reset(self) -> None:
        """Reset to the initial state (expecting a dim token)."""
        self.state = 0
