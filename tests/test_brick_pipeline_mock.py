"""End-to-end coverage of MockBrickPipeline and the singleton factories.

The real BrickPipeline is pragma'd; the mock path is the one the API exercises
under LEGOGEN_DEV=1, so its behavior is worth pinning explicitly."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.inference import brick_pipeline as bp
from backend.inference.brick_pipeline import MockBrickPipeline, _MockStage1


def test_mock_generate_emits_expected_bricks():
    pipe = MockBrickPipeline()
    events = []
    out = pipe.generate("a small house", on_progress=events.append)
    assert out["brick_count"] == 12
    assert out["stable"] is True
    assert out["metadata"]["model_version"] == "mock-brick-v1"
    # One progress event per brick.
    assert [e["count"] for e in events] == list(range(1, 13))


def test_mock_generate_no_progress_callback():
    """The callback path is optional; omitting it must not raise."""
    pipe = MockBrickPipeline()
    out = pipe.generate("a small house")
    assert out["brick_count"] == 12


def test_mock_generate_best_of_n_stamps_metadata():
    pipe = MockBrickPipeline()
    out = pipe.generate_best_of_n("x", n=5)
    assert out["metadata"]["n"] == 5
    assert out["metadata"]["picked_index"] == 0
    assert out["metadata"]["stable_rate"] == 1.0


def test_mock_generate_from_image_returns_caption():
    pipe = MockBrickPipeline()
    out = pipe.generate_from_image(object())
    assert out["caption"].startswith("a small red house")
    assert out["brick_count"] == 12


def test_mock_generate_from_image_emits_caption_event():
    pipe = MockBrickPipeline()
    events = []
    pipe.generate_from_image(object(), on_progress=events.append)
    types = [e["type"] for e in events]
    assert types[0] == "caption"
    assert events[1:] == [{"type": "brick", "count": i} for i in range(1, 13)]


def test_mock_stage1_returns_fixed_caption():
    assert _MockStage1().describe(object()).startswith("a small red house")


def test_get_brick_pipeline_returns_singleton(reset_pipeline_singletons):
    a = bp.get_brick_pipeline()
    b = bp.get_brick_pipeline()
    assert a is b
    assert isinstance(a, MockBrickPipeline)


def test_get_stage1_pipeline_returns_singleton(reset_pipeline_singletons):
    a = bp._get_stage1_pipeline()
    b = bp._get_stage1_pipeline()
    assert a is b
    assert isinstance(a, _MockStage1)
