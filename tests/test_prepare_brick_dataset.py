import pytest
from backend.data_pipeline.prepare_brick_dataset import (
    parse_st2b_bricks, pick_color_for_brick, colorize_structure, format_training_example,
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
