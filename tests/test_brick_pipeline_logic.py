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
        DONE_TOKEN,
        MAX_BRICKS,
        MAX_REJECTIONS,
        MAX_ROLLBACKS,
        STEP_PATTERN,
        BrickPipeline,
        _allowed_color_list,
        _color_is_allowed,
        _color_pattern,
        _normalize_generate_step_result,
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

    def test_rejects_colors_outside_palette(self):
        pat = re.compile(BRICK_PATTERN)
        assert pat.fullmatch("2x4 (0,0,0) #123456\n") is None

    def test_grammar_matches_are_parseable_by_backend_regex(self):
        """Anything the grammar accepts must also match the parser regex,
        so that `parse_brick` cannot raise after a grammar-constrained decode."""
        pat = re.compile(BRICK_PATTERN)
        samples = [
            "2x4 (5,3,0) #C91A09\n",
            "1x1 (0,0,0) #05131D\n",
            "8x1 (19,19,19) #FFFFFF\n",
            "2x6 (10,0,5) #0055BF\n",
        ]
        for line in samples:
            assert pat.fullmatch(line) is not None
            # _BRICK_RE has no trailing newline, so strip before matching
            assert _BRICK_RE.fullmatch(line.rstrip("\n")) is not None


@pytest.mark.skipif(not _MODULE_IMPORTABLE, reason="brick_pipeline module not importable")
class TestStepGrammarPattern:
    def test_accepts_done_stop_line(self):
        pat = re.compile(STEP_PATTERN)
        assert pat.fullmatch(f"{DONE_TOKEN}\n") is not None

    def test_accepts_brick_line(self):
        pat = re.compile(STEP_PATTERN)
        assert pat.fullmatch("2x4 (0,0,0) #C91A09\n") is not None

    def test_rejects_malformed_stop_text(self):
        pat = re.compile(STEP_PATTERN)
        assert pat.fullmatch("DONE") is None
        assert pat.fullmatch("FINISHED\n") is None


@pytest.mark.skipif(not _MODULE_IMPORTABLE, reason="brick_pipeline module not importable")
class TestPaletteValidation:
    def test_palette_accepts_known_color(self):
        assert _color_is_allowed("C91A09") is True

    def test_palette_rejects_unknown_color(self):
        assert _color_is_allowed("123456") is False

    def test_palette_helpers_fall_back_to_generic_hex_when_palette_unavailable(self, monkeypatch):
        import backend.inference.brick_pipeline as bp

        class MissingPalette:
            def __iter__(self):
                raise OSError("missing")

        monkeypatch.setattr(bp, "ALLOWED_COLORS", MissingPalette())

        assert _allowed_color_list() is None
        assert _color_pattern() == r"[0-9A-Fa-f]{6}"
        assert _color_is_allowed("123abc") is True

    def test_palette_helpers_fall_back_to_generic_hex_when_palette_data_is_malformed(self, monkeypatch):
        import backend.inference.brick_pipeline as bp

        class MalformedPalette:
            def __iter__(self):
                raise ValueError("bad json")

        monkeypatch.setattr(bp, "ALLOWED_COLORS", MalformedPalette())

        assert _allowed_color_list() is None
        assert _color_pattern() == r"[0-9A-Fa-f]{6}"
        assert _color_is_allowed("123abc") is True


@pytest.mark.skipif(not _MODULE_IMPORTABLE, reason="brick_pipeline module not importable")
def test_normalize_generate_step_result_rejects_unexpected_shape():
    with pytest.raises(TypeError, match="Unexpected _generate_one_brick result"):
        _normalize_generate_step_result("not a tuple")



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
        assert "termination_reason" in result["metadata"]
        assert "final_stable" in result["metadata"]
        assert "outlines_enabled" in result["metadata"]
        assert "palette_validation_enabled" in result["metadata"]
        assert "hit_max_rejections" in result["metadata"]
        assert "hit_max_bricks" in result["metadata"]
        assert "hit_max_seconds" in result["metadata"]
        assert "hit_max_rollbacks" in result["metadata"]
        assert "hit_done" in result["metadata"]
        assert "requested_max_bricks" in result["metadata"]
        assert "requested_max_seconds" in result["metadata"]
        assert "requested_stability_check_interval" in result["metadata"]

    def test_rejections_counted(self):
        from backend.brick.parser import Brick

        pipeline = self._make_pipeline([
            (Brick(1, 1, 0, 0, 0, "FF0000"), 5),
            (Brick(1, 1, 1, 0, 0, "00FF00"), 3),
            (None, 0),
        ])
        result = pipeline.generate("test")
        assert result["metadata"]["rejections"] == 8

    def test_max_rejections_sets_termination_reason(self):
        pipeline = self._make_pipeline([(None, MAX_REJECTIONS, "max_rejections")])
        result = pipeline.generate("test")
        assert result["metadata"]["termination_reason"] == "max_rejections"
        assert result["metadata"]["hit_max_rejections"] is True

    def test_done_stops_generation_before_max_bricks(self):
        from backend.brick.parser import Brick

        pipeline = self._make_pipeline([
            (Brick(2, 4, 0, 0, 0, "C91A09"), 0),
            (None, 0, "done"),
            (Brick(2, 4, 0, 4, 0, "FFFFFF"), 0),
        ])

        result = pipeline.generate("test", max_bricks=10)

        assert result["brick_count"] == 1
        assert result["metadata"]["termination_reason"] == "done"
        assert result["metadata"]["hit_done"] is True
        assert result["metadata"]["hit_max_bricks"] is False

    def test_max_bricks_caps_generation_and_metadata(self):
        from backend.brick.parser import Brick

        pipeline = self._make_pipeline([
            (Brick(2, 4, 0, 0, 0, "C91A09"), 0),
            (Brick(2, 4, 0, 4, 0, "FFFFFF"), 0),
            (None, 0),
        ])

        result = pipeline.generate("test", max_bricks=1)

        assert result["brick_count"] == 1
        assert result["metadata"]["termination_reason"] == "max_bricks"
        assert result["metadata"]["hit_max_bricks"] is True
        assert result["metadata"]["requested_max_bricks"] == 1

    def test_max_seconds_caps_after_slow_step_and_metadata(self):
        import time
        from backend.brick.parser import Brick

        pipeline = self._make_pipeline([])

        def slow_generate_one(input_ids, grid, logits_processor=None):
            time.sleep(0.1)
            return Brick(2, 4, 0, 0, 0, "C91A09"), 0, None

        pipeline._generate_one_brick = slow_generate_one

        result = pipeline.generate("test", max_seconds=0.05)

        assert result["brick_count"] == 1
        assert result["metadata"]["termination_reason"] == "max_seconds"
        assert result["metadata"]["hit_max_seconds"] is True
        assert result["metadata"]["requested_max_seconds"] == 0.05


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
        assert out["metadata"]["selection_strategy"] == "rank"


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

    brick, rejections, stop_reason = pipe._generate_one_brick(input_ids, grid)
    assert brick is not None
    assert (brick.h, brick.w, brick.x, brick.y, brick.z) == (2, 4, 0, 0, 0)
    assert rejections == 1  # skipped exactly one malformed line
    assert stop_reason is None


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch required")
def test_generate_one_brick_accepts_valid_line_before_eos():
    """Regression: constrained decoding often returns one valid brick line
    followed by the chat EOS token. The parser should keep that brick instead
    of treating any EOS occurrence as immediate termination.
    """
    import torch
    from backend.inference.brick_pipeline import BrickPipeline
    from backend.brick.occupancy import VoxelGrid

    pipe = BrickPipeline.__new__(BrickPipeline)
    pipe.device = "cpu"

    class StubTok:
        pad_token_id = 0
        eos_token_id = 1
        eos_token = "<|im_end|>"

        def decode(self, tokens, skip_special_tokens=False):
            return "1x2 (4,18,0) #C91A09\n<|im_end|>"

    class StubModel:
        def generate(self, input_ids, **kwargs):
            return torch.cat(
                [input_ids, torch.tensor([[42]], device=input_ids.device)], dim=1
            )

    pipe.tokenizer = StubTok()
    pipe.model = StubModel()

    brick, rejections, stop_reason = pipe._generate_one_brick(
        torch.tensor([[0, 1, 2, 3]]), VoxelGrid()
    )

    assert brick is not None
    assert (brick.h, brick.w, brick.x, brick.y, brick.z) == (1, 2, 4, 18, 0)
    assert brick.color == "C91A09"
    assert rejections == 0
    assert stop_reason is None


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch required")
def test_generate_one_brick_returns_done_stop_reason():
    import torch
    from backend.inference.brick_pipeline import BrickPipeline
    from backend.brick.occupancy import VoxelGrid

    pipe = BrickPipeline.__new__(BrickPipeline)
    pipe.device = "cpu"

    class StubTok:
        pad_token_id = 0
        eos_token_id = 1
        eos_token = "<|im_end|>"

        def decode(self, tokens, skip_special_tokens=False):
            return "DONE\n<|im_end|>"

    class StubModel:
        def generate(self, input_ids, **kwargs):
            return torch.cat(
                [input_ids, torch.tensor([[42]], device=input_ids.device)], dim=1
            )

    pipe.tokenizer = StubTok()
    pipe.model = StubModel()

    brick, rejections, stop_reason = pipe._generate_one_brick(
        torch.tensor([[0, 1, 2, 3]]), VoxelGrid()
    )

    assert brick is None
    assert rejections == 0
    assert stop_reason == "done"


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch required")
def test_generate_one_brick_uses_fresh_logits_processor_for_retries():
    """RegexLogitsProcessor carries generation state, so collision retries
    need a fresh processor rather than reusing the completed one-line FSM.
    """
    import torch
    from backend.brick.parser import Brick
    from backend.inference.brick_pipeline import BrickPipeline
    from backend.brick.occupancy import VoxelGrid

    pipe = BrickPipeline.__new__(BrickPipeline)
    pipe.device = "cpu"

    decode_outputs = [
        "2x4 (0,0,0) #C91A09\n<|im_end|>",  # collides with pre-placed brick
        "2x4 (0,4,0) #C91A09\n<|im_end|>",  # accepted
    ]
    processors = []
    processors_seen_by_model = []

    class StubTok:
        pad_token_id = 0
        eos_token_id = 1
        eos_token = "<|im_end|>"

        def __init__(self):
            self._i = 0

        def decode(self, tokens, skip_special_tokens=False):
            out = decode_outputs[self._i]
            self._i = min(self._i + 1, len(decode_outputs) - 1)
            return out

    class StubModel:
        def generate(self, input_ids, **kwargs):
            processors_seen_by_model.append(kwargs["logits_processor"][0])
            return torch.cat(
                [input_ids, torch.tensor([[42]], device=input_ids.device)], dim=1
            )

    def processor_factory():
        processor = object()
        processors.append(processor)
        return processor

    pipe.tokenizer = StubTok()
    pipe.model = StubModel()
    grid = VoxelGrid()
    grid.place(Brick(2, 4, 0, 0, 0, "C91A09"))

    brick, rejections, stop_reason = pipe._generate_one_brick(
        torch.tensor([[0, 1, 2, 3]]), grid, processor_factory
    )

    assert brick is not None
    assert (brick.h, brick.w, brick.x, brick.y, brick.z) == (2, 4, 0, 4, 0)
    assert rejections == 1
    assert stop_reason is None
    assert len(processors) == 2
    assert processors_seen_by_model == processors
