import pytest
from backend.brick.decoder import BrickTokenConstraint
from backend.brick.constants import ALLOWED_DIMS, ALLOWED_COLORS

_COLORS_LOADED = len(ALLOWED_COLORS) > 0

def test_allowed_dims_populated():
    assert len(ALLOWED_DIMS) == 14

def test_allowed_colors_populated():
    if not _COLORS_LOADED:
        pytest.skip("colors.json not available")
    assert len(ALLOWED_COLORS) > 40

def test_constraint_initial_state():
    c = BrickTokenConstraint()
    allowed = c.get_allowed_strings()
    for dim in ALLOWED_DIMS:
        assert dim in allowed

def test_constraint_after_dim():
    c = BrickTokenConstraint()
    c.feed("2x4")
    assert c.get_allowed_strings() == [" ("]

def test_constraint_full_sequence():
    c = BrickTokenConstraint()
    for tok in ["2x4", " (", "5", ",", "3", ",", "0", ") #", "C91A09", "\n"]:
        c.feed(tok)
    for dim in ALLOWED_DIMS:
        assert dim in c.get_allowed_strings()
