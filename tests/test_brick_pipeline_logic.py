"""Tests for backend.inference.brick_pipeline — regex, constants, and generate loop logic.

The BrickPipeline module requires torch and data/cache/colors.json at import time.
Tests are split into:
  - Regex/constant tests that work without the module (use inline definitions)
  - Generate-loop tests that require torch (skipped when unavailable)
"""

import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The regex from brick_pipeline.py line 18 — duplicated here so we can test
# even when torch is not installed.
_BRICK_RE = re.compile(r"(\d+)x(\d+) \((\d+),(\d+),(\d+)\) #([0-9A-Fa-f]{6})")

# Try importing the full module for generate-loop and constants tests
try:
    import torch

    from backend.inference.brick_pipeline import (
        BASE_TEMPERATURE,
        MAX_BRICKS,
        MAX_REJECTIONS,
        MAX_ROLLBACKS,
        MAX_TEMPERATURE,
        TEMP_INCREMENT,
        BrickPipeline,
    )

    _TORCH_AVAILABLE = True
except (ImportError, FileNotFoundError, OSError):
    _TORCH_AVAILABLE = False


# ── TestBrickRegex (no torch needed) ─────────────────────────────────


class TestBrickRegex:
    def test_matches_valid_format(self):
        assert _BRICK_RE.fullmatch("2x4 (5,3,0) #C91A09") is not None

    def test_rejects_invalid_formats(self):
        assert _BRICK_RE.fullmatch("2x4 (5, 3, 0) #C91A09") is None  # spaces
        assert _BRICK_RE.fullmatch("bricks here") is None
        assert _BRICK_RE.fullmatch("") is None

    def test_captures_all_groups(self):
        m = _BRICK_RE.fullmatch("1x2 (10,15,3) #FFFFFF")
        assert m is not None
        h, w, x, y, z, color = m.groups()
        assert (h, w) == ("1", "2")
        assert (x, y, z) == ("10", "15", "3")
        assert color == "FFFFFF"

    def test_rejects_negative_coordinates(self):
        # brick_pipeline uses \d+ (not -?\d+), so negative coords don't match
        assert _BRICK_RE.fullmatch("2x4 (-1,0,0) #C91A09") is None

    def test_matches_lowercase_hex(self):
        assert _BRICK_RE.fullmatch("1x1 (0,0,0) #abcdef") is not None


# ── TestConstants (requires module import) ───────────────────────────


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch or colors.json unavailable")
class TestConstants:
    """Verify brick_pipeline constants haven't drifted from expected values."""

    def test_max_bricks(self):
        assert MAX_BRICKS == 500

    def test_max_rejections(self):
        assert MAX_REJECTIONS == 500

    def test_max_rollbacks(self):
        assert MAX_ROLLBACKS == 100

    def test_base_temperature(self):
        assert BASE_TEMPERATURE == pytest.approx(0.6)

    def test_temp_increment(self):
        assert TEMP_INCREMENT == pytest.approx(0.01)

    def test_max_temperature(self):
        assert MAX_TEMPERATURE == pytest.approx(2.0)


# ── TestGenerateLoop (requires torch) ────────────────────────────────


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch or colors.json unavailable")
class TestGenerateLoop:
    """Test the generate() method by mocking model internals."""

    def _make_pipeline(self, brick_results):
        """Create a BrickPipeline without loading any model.

        Args:
            brick_results: list of (Brick|None, rejection_count) tuples.
        """
        from backend.brick.parser import Brick

        pipeline = object.__new__(BrickPipeline)
        pipeline.device = "cpu"

        call_iter = iter(brick_results)

        def mock_generate_one(input_ids, grid):
            try:
                return next(call_iter)
            except StopIteration:
                return None, 0

        pipeline._generate_one_brick = mock_generate_one

        # Mock tokenizer with minimal interface
        mock_tok = MagicMock()
        mock_tok.apply_chat_template.return_value = "mock prompt"
        mock_tok.encode.return_value = [1, 2, 3]
        mock_tok.pad_token_id = 0
        mock_tok.eos_token_id = 2
        pipeline.tokenizer = mock_tok

        return pipeline

    def test_stable_sequence_no_rollback(self):
        from backend.brick.parser import Brick

        pipeline = self._make_pipeline([
            (Brick(2, 4, 0, 0, 0, "C91A09"), 0),
            (Brick(2, 4, 0, 4, 0, "FFFFFF"), 0),
            (None, 0),
        ])
        result = pipeline.generate("a red house")
        assert result["brick_count"] == 2
        assert result["stable"] is True
        assert result["metadata"]["rollbacks"] == 0

    def test_none_ends_generation_immediately(self):
        pipeline = self._make_pipeline([(None, 0)])
        result = pipeline.generate("anything")
        assert result["brick_count"] == 0
        assert result["bricks"] == ""

    def test_output_dict_has_required_keys(self):
        from backend.brick.parser import Brick

        pipeline = self._make_pipeline([
            (Brick(1, 1, 0, 0, 0, "FF0000"), 0),
            (None, 0),
        ])
        result = pipeline.generate("test")
        assert "bricks" in result
        assert "brick_count" in result
        assert "stable" in result
        assert "metadata" in result
        assert "model_version" in result["metadata"]
        assert "generation_time_ms" in result["metadata"]
        assert "rejections" in result["metadata"]
        assert "rollbacks" in result["metadata"]

    def test_rejections_counted(self):
        from backend.brick.parser import Brick

        pipeline = self._make_pipeline([
            (Brick(1, 1, 0, 0, 0, "FF0000"), 5),
            (Brick(1, 1, 1, 0, 0, "00FF00"), 3),
            (None, 0),
        ])
        result = pipeline.generate("test")
        assert result["metadata"]["rejections"] == 8
