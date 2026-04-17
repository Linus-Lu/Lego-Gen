import pytest
from backend.data_pipeline.prepare_brick_dataset import (
    parse_st2b_bricks,
    pick_color_for_brick,
    colorize_structure,
    format_training_example,
    _extract_caption_color,
    CATEGORY_PALETTES,
    DEFAULT_PALETTE,
)
from backend.brick.parser import Brick


def test_parse_st2b_bricks():
    raw = "1x1 (15,17,0)\n1x4 (15,13,0)\n"
    bricks = parse_st2b_bricks(raw)
    assert len(bricks) == 2
    assert bricks[0] == (1, 1, 15, 17, 0)


def test_parse_st2b_bricks_empty():
    assert parse_st2b_bricks("") == []


def test_pick_color_returns_valid_hex():
    color = pick_color_for_brick(caption="A red car", category="car", z=0, seed=42)
    assert len(color) == 6


def test_colorize_structure():
    raw = [(2, 4, 5, 3, 0), (1, 2, 3, 7, 1)]
    colored = colorize_structure(raw, caption="A blue table", category="table", seed=0)
    assert len(colored) == 2
    assert all(isinstance(b, Brick) for b in colored)


def test_format_training_example():
    bricks = [
        Brick(h=2, w=4, x=5, y=3, z=0, color="C91A09"),
        Brick(h=1, w=2, x=3, y=7, z=1, color="05131D"),
    ]
    example = format_training_example("A red chair", bricks)
    assert example["messages"][0]["role"] == "system"
    assert "A red chair" in example["messages"][1]["content"]
    assert "2x4 (5,3,0) #C91A09" in example["messages"][2]["content"]


def test_parse_st2b_bricks_skips_blank_lines():
    """Blank/whitespace-only lines must be silently dropped."""
    raw = "\n\n   \n1x4 (15,13,0)\n\n"
    bricks = parse_st2b_bricks(raw)
    assert bricks == [(1, 4, 15, 13, 0)]


def test_parse_st2b_bricks_skips_unparseable_lines():
    """Lines that don't match the ST2B regex are dropped, valid ones kept."""
    raw = "notabrick line\n2x4 (0,0,0)\ngarbage\n"
    bricks = parse_st2b_bricks(raw)
    assert bricks == [(2, 4, 0, 0, 0)]


def test_extract_caption_color_returns_none_when_no_color_word():
    """Fall through the COLOR_WORDS loop returns None."""
    assert _extract_caption_color("a mysterious shape") is None
    assert _extract_caption_color("") is None


def test_pick_color_falls_back_to_category_palette_at_z0():
    """No caption color + z=0 → use category palette with dark-weighting."""
    caption = "a mysterious shape"  # no color words
    color = pick_color_for_brick(caption, category="car", z=0, seed=0)
    assert color in CATEGORY_PALETTES["car"]


def test_pick_color_uses_flat_weights_above_ground():
    """No caption color + z>0 → flat weights (no dark boost)."""
    caption = "a mysterious shape"
    color = pick_color_for_brick(caption, category="car", z=1, seed=0)
    assert color in CATEGORY_PALETTES["car"]


def test_pick_color_falls_back_to_default_palette_for_unknown_category():
    """Unknown category triggers the DEFAULT_PALETTE fallback."""
    caption = "a mysterious shape"
    color = pick_color_for_brick(caption, category="not_a_category", z=0, seed=0)
    assert color in DEFAULT_PALETTE
