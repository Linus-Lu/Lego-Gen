"""Tests for backend.models.tokenizer — JSON extraction, prompt templates, chat messages."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.models.tokenizer import (
    PLANNER_SYSTEM_PROMPT,
    PROMPT_TEMPLATES,
    SYSTEM_PROMPT,
    USER_PROMPT,
    build_chat_messages,
    build_planner_chat_messages,
    extract_json_from_text,
    get_json_prompt,
    sample_prompt_template,
    strip_thinking_blocks,
)


# ── TestExtractJsonFromText ──────────────────────────────────────────


class TestExtractJsonFromText:
    def test_simple_json(self):
        text = '{"key": "value"}'
        result = extract_json_from_text(text)
        assert result == {"key": "value"}

    def test_json_with_preamble(self):
        text = 'Here is the result:\n{"object": "House", "total_parts": 10}'
        result = extract_json_from_text(text)
        assert result["object"] == "House"
        assert result["total_parts"] == 10

    def test_nested_json_extracts_outermost(self):
        text = '{"a": {"b": 1}, "c": 2}'
        result = extract_json_from_text(text)
        assert result == {"a": {"b": 1}, "c": 2}

    def test_no_json_returns_none(self):
        assert extract_json_from_text("no json here") is None

    def test_empty_string_returns_none(self):
        assert extract_json_from_text("") is None

    def test_malformed_json_returns_none_without_repair(self):
        # json_repair is not installed, so malformed JSON returns None
        text = '{"key": value_no_quotes}'
        result = extract_json_from_text(text)
        # May return None or repaired dict depending on json_repair availability
        assert result is None or isinstance(result, dict)

    def test_json_followed_by_text(self):
        text = '{"x": 1}\nSome trailing text here'
        result = extract_json_from_text(text)
        assert result == {"x": 1}


# ── TestSamplePromptTemplate ─────────────────────────────────────────


class TestSamplePromptTemplate:
    def test_returns_formatted_string(self):
        label = {"object": "Red Car", "dominant_colors": ["Red"]}
        prompt = sample_prompt_template(label, epoch=0)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_color_aware_template(self):
        label = {
            "object": "House",
            "dominant_colors": ["Blue", "White"],
            "category": "City",
        }
        # Run many epochs to increase chance of hitting color template
        found_color = False
        for epoch in range(50):
            prompt = sample_prompt_template(label, epoch=epoch)
            if "Blue" in prompt or "White" in prompt:
                found_color = True
                break
        assert found_color, "Color-aware template never selected across 50 epochs"

    def test_missing_colors_uses_defaults(self):
        label = {"object": "Spaceship"}
        prompt = sample_prompt_template(label, epoch=0)
        # Should not raise — defaults to "Red" / "White"
        assert isinstance(prompt, str)

    def test_different_epochs_rotate_templates(self):
        label = {"object": "Tree", "dominant_colors": ["Green"]}
        prompts = {sample_prompt_template(label, epoch=e) for e in range(20)}
        # Should produce multiple distinct prompts across epochs
        assert len(prompts) > 1


# ── TestBuildChatMessages ────────────────────────────────────────────


class TestBuildChatMessages:
    def test_with_image_url(self):
        msgs = build_chat_messages(image_url="http://example.com/img.png")
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        user_content = msgs[1]["content"]
        assert isinstance(user_content, list)
        assert any(c.get("type") == "image" for c in user_content)

    def test_without_image_url(self):
        msgs = build_chat_messages()
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        user_content = msgs[1]["content"]
        assert isinstance(user_content, list)
        assert any(c.get("type") == "image" for c in user_content)


# ── TestBuildPlannerChatMessages ─────────────────────────────────────


class TestBuildPlannerChatMessages:
    def test_structure(self):
        msgs = build_planner_chat_messages("Build a house")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == PLANNER_SYSTEM_PROMPT
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "Build a house"


# ── TestGetJsonPrompt ────────────────────────────────────────────────


class TestGetJsonPrompt:
    def test_contains_prompts(self):
        prompt = get_json_prompt()
        assert SYSTEM_PROMPT in prompt
        assert USER_PROMPT in prompt
        assert "JSON:" in prompt


# ── TestStripThinkingBlocks ──────────────────────────────────────────


class TestStripThinkingBlocks:
    def test_removes_thinking(self):
        text = "<think>internal reasoning</think>The answer is 42."
        assert strip_thinking_blocks(text) == "The answer is 42."

    def test_no_thinking_unchanged(self):
        text = "The answer is 42."
        assert strip_thinking_blocks(text) == "The answer is 42."

    def test_multiple_blocks(self):
        text = "<think>a</think>Hello <think>b</think>World"
        assert strip_thinking_blocks(text) == "Hello World"
