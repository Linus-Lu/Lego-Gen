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

# The regex from brick_pipeline.py — duplicated here so we can test
# even when torch is not installed.
_BRICK_RE = re.compile(r"(\d+)x(\d+) \((\d+),(\d+),(\d+)\) #([0-9A-Fa-f]{6})")

# brick_pipeline's module body does NOT import torch — torch is imported
# inside BrickPipeline methods. Import the module-level symbols unconditionally
# so tests that don't need torch (grammar regex, outlines fallback, BoN-rank
# via __new__ stubs) can still run on CI where torch is absent.
try:
    from backend.inference.brick_pipeline import (
        BASE_TEMPERATURE,
        BRICK_PATTERN,
        MAX_BRICKS,
        MAX_REJECTIONS,
        MAX_ROLLBACKS,
        BrickPipeline,
    )
    _MODULE_IMPORTABLE = True
except (ImportError, FileNotFoundError, OSError):
    _MODULE_IMPORTABLE = False

# Separate check for tests that actually exercise torch (model.generate, etc.).
try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
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


@pytest.mark.skipif(not _MODULE_IMPORTABLE, reason="brick_pipeline module not importable")
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


# ── TestGrammarPattern (requires module import) ──────────────────────


@pytest.mark.skipif(not _MODULE_IMPORTABLE, reason="brick_pipeline module not importable")
class TestGrammarPattern:
    """The grammar regex handed to the logits processor must accept exactly the
    set of brick lines that `_BRICK_RE` + allowed dims also accept. Grammar
    widening or narrowing is a silent correctness hazard, so pin both sides."""

    def test_accepts_all_allowed_dim_combinations(self):
        pat = re.compile(BRICK_PATTERN)
        for dims in ["1x1", "1x2", "2x1", "2x2", "1x4", "4x1", "2x4",
                     "4x2", "1x6", "6x1", "2x6", "6x2", "1x8", "8x1"]:
            line = f"{dims} (0,0,0) #C91A09\n"
            assert pat.fullmatch(line) is not None, f"grammar rejected allowed dim {dims}"

    def test_rejects_disallowed_dims(self):
        pat = re.compile(BRICK_PATTERN)
        # 3x3 and 1x3 are not in BRICK_SHAPES
        assert pat.fullmatch("3x3 (0,0,0) #C91A09\n") is None
        assert pat.fullmatch("1x3 (0,0,0) #C91A09\n") is None

    def test_rejects_malformed_lines(self):
        pat = re.compile(BRICK_PATTERN)
        assert pat.fullmatch("2x4 (0, 0, 0) #C91A09\n") is None  # internal spaces
        assert pat.fullmatch("2x4 (0,0,0) C91A09\n") is None     # missing hash
        assert pat.fullmatch("2x4 (0,0,0) #C91A0\n") is None     # short hex
        assert pat.fullmatch("") is None

    def test_grammar_matches_are_parseable_by_backend_regex(self):
        """Anything the grammar accepts must also match the parser regex,
        so that `parse_brick` cannot raise after a grammar-constrained decode."""
        pat = re.compile(BRICK_PATTERN)
        samples = [
            "2x4 (5,3,0) #C91A09\n",
            "1x1 (0,0,0) #000000\n",
            "8x1 (19,19,19) #FFFFFF\n",
            "2x6 (10,0,5) #abcdef\n",
        ]
        for line in samples:
            assert pat.fullmatch(line) is not None
            # _BRICK_RE has no trailing newline, so strip before matching
            assert _BRICK_RE.fullmatch(line.rstrip("\n")) is not None


# ── TestGenerateLoop (requires torch — test body uses torch.tensor) ──


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch required")
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

        def mock_generate_one(input_ids, grid, logits_processor=None):
            try:
                return next(call_iter)
            except StopIteration:
                return None, 0

        pipeline._generate_one_brick = mock_generate_one
        # _fresh_logits_processor is a real method on the class and tries to
        # import outlines; returning None matches the absent-outlines path.
        pipeline._fresh_logits_processor = lambda: None

        # Mock tokenizer. encode() behaves differently depending on kwargs:
        #   - return_tensors="pt"  -> 2-D tensor (for the initial prompt)
        #   - otherwise            -> plain list (for per-brick appends,
        #     which are then wrapped into a tensor by the caller)
        def mock_encode(text, return_tensors=None, add_special_tokens=True):
            if return_tensors == "pt":
                return torch.tensor([[1, 2, 3]])
            return [4, 5]

        mock_tok = MagicMock()
        mock_tok.apply_chat_template.return_value = "mock prompt"
        mock_tok.encode.side_effect = mock_encode
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


def test_mock_generate_best_of_n_returns_valid_shape():
    from backend.inference.brick_pipeline import MockBrickPipeline

    pipe = MockBrickPipeline()
    out = pipe.generate_best_of_n("a small red house", n=4, strategy="rank")
    assert "bricks" in out and "brick_count" in out and "stable" in out
    assert out["metadata"]["n"] == 4
    assert out["metadata"]["picked_index"] in range(4)


def test_generate_best_of_n_strips_bricks_parsed_from_returned_dict():
    """Regression test: bricks_parsed is an internal-only key and must not
    leak onto the returned dict (breaks FastAPI JSON serialization)."""
    from backend.inference.brick_pipeline import BrickPipeline

    # Use __new__ to skip __init__ (which loads torch/models). We override
    # just the methods generate_best_of_n needs.
    pipe = BrickPipeline.__new__(BrickPipeline)
    pipe.generate = lambda caption, on_progress=None: {
        "bricks": "2x4 (0,0,0) #C91A09\n2x4 (0,0,1) #C91A09",
        "brick_count": 2,
        "stable": True,
        "metadata": {"model_version": "stub"},
    }

    out = pipe.generate_best_of_n("test", n=2, strategy="rank")
    assert "bricks_parsed" not in out
    assert isinstance(out["bricks"], str)
    assert out["metadata"]["n"] == 2


# ── TestGrammarOrdering (requires module import via _TORCH_AVAILABLE) ──


@pytest.mark.skipif(not _MODULE_IMPORTABLE, reason="brick_pipeline module not importable")
class TestGrammarOrdering:
    def test_longest_dim_prefix_matches_first(self):
        """A naive shortest-first regex would match '1x1' when '1x10' is the
        intended dim. 1x10 isn't in the allowed set, so this is actually a
        negative test against the ordering being wrong."""
        import re as _re
        from backend.inference.brick_pipeline import BRICK_PATTERN
        pat = _re.compile(BRICK_PATTERN)
        # 2x4 is allowed; 2x40 is not; if ordering were shortest-first and
        # the coord pattern were broken, this could false-match.
        assert pat.fullmatch("2x4 (0,0,0) #C91A09\n") is not None

    def test_coord_pattern_rejects_out_of_range(self):
        import re as _re
        from backend.inference.brick_pipeline import BRICK_PATTERN
        pat = _re.compile(BRICK_PATTERN)
        # WORLD_DIM is 20 → 20 is out of range, 19 is the max.
        assert pat.fullmatch("2x4 (20,0,0) #C91A09\n") is None
        assert pat.fullmatch("2x4 (0,0,19) #C91A09\n") is not None


@pytest.mark.skipif(not _MODULE_IMPORTABLE, reason="brick_pipeline module not importable")
class TestLogitsProcessorFallback:
    def test_returns_none_when_outlines_absent(self, monkeypatch):
        """When outlines is not importable, _build_logits_processor returns None.
        The caller treats this as 'no grammar constraint' and validates
        parse failures via try/except instead."""
        import sys as _sys
        from backend.inference.brick_pipeline import _build_logits_processor
        # Simulate ImportError by wiping outlines from sys.modules.
        monkeypatch.setitem(_sys.modules, "outlines", None)
        monkeypatch.setitem(_sys.modules, "outlines.processors", None)
        result = _build_logits_processor(tokenizer=object(), pattern=r"x")
        assert result is None


@pytest.mark.skipif(not _MODULE_IMPORTABLE, reason="brick_pipeline module not importable")
class TestBestOfNRankStrategy:
    def test_rank_strategy_picks_most_bricks_among_stable(self):
        """strategy='rank' skips the clustering entirely and returns
        rank_candidates(candidates)[0]."""
        from backend.inference.brick_pipeline import BrickPipeline

        pipe = BrickPipeline.__new__(BrickPipeline)
        counter = {"n": 0}

        def fake_generate(caption, on_progress=None):
            counter["n"] += 1
            # Return different brick counts per call so ranking has signal.
            i = counter["n"]
            return {
                "bricks": f"2x4 (0,0,0) #C91A09\n" * i,
                "brick_count": i,
                "stable": True,
                "metadata": {},
            }

        pipe.generate = fake_generate  # type: ignore[method-assign]
        out = pipe.generate_best_of_n("x", n=3, strategy="rank")
        # Ranking picks the largest (last call, brick_count=3).
        assert out["brick_count"] == 3
        assert out["metadata"]["picked_index"] == 2


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch required")
def test_generate_one_brick_rejects_malformed_line_without_raising():
    """Regression for bug_024: when the grammar logits processor is absent
    (outlines missing) the model can emit non-brick text. The rejection
    loop must treat a parse failure the same as a voxel collision rather
    than propagating ValueError out of the pipeline.
    """
    import torch
    from backend.inference.brick_pipeline import BrickPipeline
    from backend.brick.occupancy import VoxelGrid

    pipe = BrickPipeline.__new__(BrickPipeline)
    pipe.device = "cpu"
    pipe.logits_processor = None  # simulates outlines missing

    decode_outputs = [
        "this is not a brick line\n",  # raises ValueError
        "2x4 (0,0,0) #C91A09\n",       # valid
    ]

    class StubTok:
        pad_token_id = 0
        eos_token_id = 1
        eos_token = "<|endoftext|>"

        def __init__(self):
            self._i = 0

        def decode(self, tokens, skip_special_tokens=False):
            out = decode_outputs[self._i]
            self._i = min(self._i + 1, len(decode_outputs) - 1)
            return out

    class StubModel:
        def generate(self, input_ids, **kwargs):
            return torch.cat(
                [input_ids, torch.tensor([[42]], device=input_ids.device)], dim=1
            )

    pipe.tokenizer = StubTok()
    pipe.model = StubModel()

    grid = VoxelGrid()
    input_ids = torch.tensor([[0, 1, 2, 3]])

    brick, rejections = pipe._generate_one_brick(input_ids, grid)
    assert brick is not None
    assert (brick.h, brick.w, brick.x, brick.y, brick.z) == (2, 4, 0, 0, 0)
    assert rejections == 1  # skipped exactly one malformed line
