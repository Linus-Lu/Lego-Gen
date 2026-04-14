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

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.brick.constants import ALLOWED_DIMS, ALLOWED_COLORS, WORLD_DIM

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedTokenizerBase

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


# ---------------------------------------------------------------------------
# LogitsProcessor wrapper for use with model.generate()
# ---------------------------------------------------------------------------


def build_brick_logits_processor(
    tokenizer: PreTrainedTokenizerBase,
    eos_token_id: int | None = None,
) -> "BrickLogitsProcessor | None":
    """Build a :class:`BrickLogitsProcessor` if every allowed string tokenises
    to exactly one token.  Returns ``None`` when the tokeniser is incompatible.
    """
    import torch

    state_machine = BrickTokenConstraint()
    token_map: dict[int, list[int]] = {}  # state -> list of allowed token IDs

    for state in range(10):
        state_machine.state = state
        allowed = state_machine.get_allowed_strings()
        ids: list[int] = []
        for s in allowed:
            encoded = tokenizer.encode(s, add_special_tokens=False)
            if len(encoded) != 1:
                # Incompatible tokeniser — fall back to regex validation.
                return None
            ids.append(encoded[0])
        token_map[state] = ids

    # State 0 also allows EOS to signal end-of-generation.
    if eos_token_id is not None:
        token_map[0] = token_map[0] + [eos_token_id]

    return BrickLogitsProcessor(token_map)


class BrickLogitsProcessor:
    """Constrains token generation to follow the brick format state machine.

    At each generation step, all tokens except those valid for the current
    state are masked to ``-inf``.  The processor advances its internal state
    after each call.  Instantiate via :func:`build_brick_logits_processor`.
    """

    def __init__(self, token_map: dict[int, list[int]]) -> None:
        self._token_map = token_map
        self._state = 0

    def reset(self) -> None:
        self._state = 0

    def __call__(
        self, input_ids: "torch.LongTensor", scores: "torch.FloatTensor"
    ) -> "torch.FloatTensor":
        import torch

        allowed = self._token_map.get(self._state, [])
        if allowed:
            mask = torch.full_like(scores, float("-inf"))
            mask[:, allowed] = 0.0
            scores = scores + mask
        self._state = (self._state + 1) % 10
        return scores
