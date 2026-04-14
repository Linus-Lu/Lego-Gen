"""Tests for few-shot example loading and prompt construction."""

import json
from pathlib import Path

import pytest


def test_few_shot_file_exists():
    path = Path(__file__).resolve().parent.parent / "backend" / "brick" / "few_shot_examples.json"
    assert path.exists(), "few_shot_examples.json must exist"


def test_few_shot_file_valid_json():
    path = Path(__file__).resolve().parent.parent / "backend" / "brick" / "few_shot_examples.json"
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert isinstance(data, list)
    assert len(data) >= 3, "Should have at least 3 examples"


def test_few_shot_examples_have_required_fields():
    path = Path(__file__).resolve().parent.parent / "backend" / "brick" / "few_shot_examples.json"
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    for i, ex in enumerate(data):
        assert "caption" in ex, f"Example {i} missing 'caption'"
        assert "bricks" in ex, f"Example {i} missing 'bricks'"
        assert len(ex["caption"]) > 5, f"Example {i} caption too short"
        assert len(ex["bricks"]) > 10, f"Example {i} bricks too short"


def test_few_shot_examples_parse_as_valid_bricks():
    from backend.brick.parser import parse_brick_sequence

    path = Path(__file__).resolve().parent.parent / "backend" / "brick" / "few_shot_examples.json"
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    for i, ex in enumerate(data):
        bricks = parse_brick_sequence(ex["bricks"])
        assert len(bricks) >= 3, f"Example {i} should have >= 3 bricks, got {len(bricks)}"


def test_few_shot_block_can_be_built():
    """The few-shot examples should produce a well-formed prompt block."""
    path = Path(__file__).resolve().parent.parent / "backend" / "brick" / "few_shot_examples.json"
    with open(path, encoding="utf-8") as fh:
        examples = json.load(fh)

    parts = ["Here are some example LEGO models:\n"]
    for ex in examples:
        parts.append(f"### Input:\n{ex['caption']}\n### Output:\n{ex['bricks']}\n")
    parts.append("Do NOT copy the examples. Create your own LEGO model for the following input.\n")
    block = "\n".join(parts)

    assert len(block) > 100
    assert "example LEGO models" in block
    assert "Do NOT copy" in block
    assert "### Input:" in block
    assert "### Output:" in block
