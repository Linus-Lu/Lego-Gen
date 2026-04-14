"""Tests for prompt templates and JSON extraction utilities."""

import pytest

from backend.models.tokenizer import (
    build_planner_chat_messages,
    sample_prompt_template,
    build_chat_messages,
    get_json_prompt,
    extract_json_from_text,
    strip_thinking_blocks,
    SYSTEM_PROMPT,
    USER_PROMPT,
    PLANNER_SYSTEM_PROMPT,
)


# ── build_planner_chat_messages ──────────────────────────────────────

def test_planner_messages_structure():
    msgs = build_planner_chat_messages("build a car")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_planner_messages_content():
    msgs = build_planner_chat_messages("build a spaceship")
    assert PLANNER_SYSTEM_PROMPT in msgs[0]["content"]
    assert msgs[1]["content"] == "build a spaceship"


# ── build_chat_messages ──────────────────────────────────────────────

def test_chat_messages_with_image_url():
    msgs = build_chat_messages(image_url="http://example.com/img.png")
    user_content = msgs[1]["content"]
    image_part = user_content[0]
    assert image_part["type"] == "image"
    assert image_part["image"] == "http://example.com/img.png"


def test_chat_messages_without_image():
    msgs = build_chat_messages()
    user_content = msgs[1]["content"]
    image_part = user_content[0]
    assert image_part["type"] == "image"
    assert "image" not in image_part


# ── get_json_prompt ──────────────────────────────────────────────────

def test_get_json_prompt():
    prompt = get_json_prompt()
    assert SYSTEM_PROMPT in prompt
    assert USER_PROMPT in prompt
    assert "JSON:" in prompt


# ── sample_prompt_template ───────────────────────────────────────────

def test_sample_prompt_contains_object():
    label = {"object": "race car", "dominant_colors": ["Red"]}
    result = sample_prompt_template(label, epoch=0)
    assert isinstance(result, str)
    assert len(result) > 0


def test_sample_prompt_empty_label():
    """Empty dict falls back to defaults."""
    result = sample_prompt_template({}, epoch=0)
    assert "LEGO model" in result


def test_sample_prompt_no_colors():
    """Empty dominant_colors falls back to 'Red'."""
    label = {"object": "table", "dominant_colors": []}
    result = sample_prompt_template(label, epoch=0)
    assert isinstance(result, str)
    assert len(result) > 0


def test_sample_prompt_deterministic_with_rng():
    """Same rng seed produces same output."""
    import random
    label = {"object": "chair", "dominant_colors": ["Blue"]}
    r1 = sample_prompt_template(label, epoch=0, rng=random.Random(42))
    r2 = sample_prompt_template(label, epoch=0, rng=random.Random(42))
    assert r1 == r2


# ── strip_thinking_blocks ───────────────────────────────────────────

def test_strip_thinking_blocks_removes_block():
    text = "<think>internal reasoning</think>the answer"
    assert strip_thinking_blocks(text) == "the answer"


def test_strip_thinking_blocks_no_blocks():
    text = "no thinking here"
    assert strip_thinking_blocks(text) == "no thinking here"


def test_strip_thinking_blocks_multiline():
    text = "<think>\nline1\nline2\n</think>\nresult"
    result = strip_thinking_blocks(text)
    assert "<think>" not in result
    assert "result" in result


def test_strip_thinking_blocks_multiple():
    text = "<think>a</think>middle<think>b</think>end"
    result = strip_thinking_blocks(text)
    assert result == "middleend"


# ── extract_json_from_text ───────────────────────────────────────────

def test_extract_json_valid():
    text = '{"key": "value"}'
    result = extract_json_from_text(text)
    assert result == {"key": "value"}


def test_extract_json_with_preamble():
    text = 'Here is the JSON: {"name": "car"} and trailing text'
    result = extract_json_from_text(text)
    assert result == {"name": "car"}


def test_extract_json_nested():
    text = '{"a": {"b": 1}, "c": 2}'
    result = extract_json_from_text(text)
    assert result == {"a": {"b": 1}, "c": 2}


def test_extract_json_no_json():
    assert extract_json_from_text("no json here") is None


def test_extract_json_unclosed():
    assert extract_json_from_text('{"key": "value"') is None


def test_extract_json_empty_object():
    result = extract_json_from_text("{}")
    assert result == {}
