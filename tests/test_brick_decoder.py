"""Tests for backend.brick.decoder — token-level constraint state machine."""

import pytest

from backend.brick import constants as const
from backend.brick.decoder import BrickTokenConstraint
from backend.brick.constants import ALLOWED_DIMS


@pytest.fixture
def seeded_colors(monkeypatch):
    """Seed the lazy color-palette cache so ALLOWED_COLORS works without colors.json."""
    fake_palette = {
        "C91A09": "Red",
        "FFFFFF": "White",
        "000000": "Black",
        "0055BF": "Blue",
    }
    # Pre-populate the memoized cache used by _lazy_palette.
    const._lazy_palette._cache = fake_palette  # type: ignore[attr-defined]
    yield fake_palette
    # Reset so other tests don't inherit the fake palette.
    if hasattr(const._lazy_palette, "_cache"):
        del const._lazy_palette._cache


def test_allowed_dims_populated():
    assert len(ALLOWED_DIMS) == 14


def test_allowed_colors_populated(seeded_colors):
    from backend.brick.constants import ALLOWED_COLORS
    assert len(ALLOWED_COLORS) == 4
    assert "C91A09" in ALLOWED_COLORS


def test_constraint_initial_state():
    c = BrickTokenConstraint()
    allowed = c.get_allowed_strings()
    for dim in ALLOWED_DIMS:
        assert dim in allowed


def test_constraint_after_dim():
    c = BrickTokenConstraint()
    c.feed("2x4")
    assert c.get_allowed_strings() == [" ("]


def test_constraint_coord_state_exposes_positions():
    c = BrickTokenConstraint()
    for tok in ["2x4", " ("]:
        c.feed(tok)
    # State 2 expects x coord — positions 0..19.
    allowed = c.get_allowed_strings()
    assert "0" in allowed
    assert "19" in allowed
    assert str(const.WORLD_DIM) not in allowed


def test_constraint_comma_states():
    c = BrickTokenConstraint()
    for tok in ["2x4", " (", "5"]:
        c.feed(tok)
    assert c.get_allowed_strings() == [","]
    c.feed(",")
    c.feed("3")
    assert c.get_allowed_strings() == [","]


def test_constraint_close_state():
    c = BrickTokenConstraint()
    for tok in ["2x4", " (", "5", ",", "3", ",", "0"]:
        c.feed(tok)
    assert c.get_allowed_strings() == [") #"]


def test_constraint_color_state(seeded_colors):
    from backend.brick.constants import ALLOWED_COLORS
    c = BrickTokenConstraint()
    for tok in ["2x4", " (", "5", ",", "3", ",", "0", ") #"]:
        c.feed(tok)
    assert c.get_allowed_strings() == list(ALLOWED_COLORS)


def test_constraint_newline_state(seeded_colors):
    c = BrickTokenConstraint()
    for tok in ["2x4", " (", "5", ",", "3", ",", "0", ") #", "C91A09"]:
        c.feed(tok)
    assert c.get_allowed_strings() == ["\n"]


def test_constraint_full_sequence_loops(seeded_colors):
    c = BrickTokenConstraint()
    for tok in ["2x4", " (", "5", ",", "3", ",", "0", ") #", "C91A09", "\n"]:
        c.feed(tok)
    # After newline we're back at state 0 -> dims allowed again.
    for dim in ALLOWED_DIMS:
        assert dim in c.get_allowed_strings()


def test_constraint_reset():
    c = BrickTokenConstraint()
    c.feed("2x4")
    c.feed(" (")
    c.reset()
    for dim in ALLOWED_DIMS:
        assert dim in c.get_allowed_strings()


def test_constraint_out_of_range_state_returns_empty():
    """Defensive: manual state poke should yield empty allowed list."""
    c = BrickTokenConstraint()
    c.state = 99
    assert c.get_allowed_strings() == []
