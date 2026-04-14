"""Tests for the constrained decoding LogitsProcessor."""

import pytest
from backend.brick.decoder import BrickTokenConstraint, BrickLogitsProcessor


def test_constraint_state_machine_cycle():
    """Walking through all 10 states should cycle back to 0."""
    c = BrickTokenConstraint()
    for state in range(10):
        assert c.state == state
        allowed = c.get_allowed_strings()
        assert len(allowed) > 0, f"State {state} has no allowed strings"
        c.feed(allowed[0])
    assert c.state == 0  # cycled back


def test_constraint_reset():
    c = BrickTokenConstraint()
    c.feed("2x4")
    c.feed(" (")
    assert c.state == 2
    c.reset()
    assert c.state == 0


def test_constraint_state0_returns_dims():
    c = BrickTokenConstraint()
    allowed = c.get_allowed_strings()
    assert "2x4" in allowed
    assert "1x1" in allowed


def test_constraint_state2_returns_positions():
    c = BrickTokenConstraint()
    c.state = 2
    allowed = c.get_allowed_strings()
    assert "0" in allowed
    assert "19" in allowed
    assert len(allowed) == 20


def test_logits_processor_masks_tokens():
    """BrickLogitsProcessor should mask all but allowed tokens."""
    import torch

    token_map = {0: [1, 5, 10]}
    proc = BrickLogitsProcessor(token_map)

    scores = torch.zeros(1, 20)
    result = proc(torch.tensor([[0]]), scores)

    # Allowed tokens should remain 0, others should be -inf
    assert result[0, 1].item() == 0.0
    assert result[0, 5].item() == 0.0
    assert result[0, 10].item() == 0.0
    assert result[0, 0].item() == float("-inf")
    assert result[0, 3].item() == float("-inf")


def test_logits_processor_state_advances():
    """Each call should advance to the next state."""
    token_map = {
        0: [1],
        1: [2],
        2: [3],
    }
    proc = BrickLogitsProcessor(token_map)
    assert proc._state == 0
    proc(torch.tensor([[0]]), torch.zeros(1, 10))
    assert proc._state == 1
    proc(torch.tensor([[0]]), torch.zeros(1, 10))
    assert proc._state == 2


def test_logits_processor_reset():
    token_map = {0: [1], 1: [2]}
    proc = BrickLogitsProcessor(token_map)
    proc(torch.tensor([[0]]), torch.zeros(1, 10))
    assert proc._state == 1
    proc.reset()
    assert proc._state == 0


def test_build_returns_none_for_multitoken():
    """build_brick_logits_processor should return None when strings need >1 token."""
    from unittest.mock import MagicMock

    tokenizer = MagicMock()
    # Make encode return 2 tokens for multi-character strings
    def mock_encode(s, add_special_tokens=False):
        if len(s) > 2:
            return [1, 2]  # 2 tokens -> incompatible
        return [1]
    tokenizer.encode = mock_encode

    from backend.brick.decoder import build_brick_logits_processor
    result = build_brick_logits_processor(tokenizer, eos_token_id=0)
    assert result is None


# Need torch for the tests above
try:
    import torch
except ImportError:
    pytest.skip("torch not available", allow_module_level=True)
