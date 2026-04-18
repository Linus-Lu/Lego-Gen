"""End-to-end coverage of MockBrickPipeline and the singleton factories.

The real BrickPipeline is pragma'd; the mock path is the one the API exercises
under LEGOGEN_DEV=1, so its behavior is worth pinning explicitly."""

import sys
import types
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
    assert out["metadata"]["termination_reason"] == "done"
    assert out["metadata"]["hit_done"] is True
    assert out["metadata"]["final_stable"] is True
    # One progress event per brick.
    assert [e["count"] for e in events] == list(range(1, 13))


def test_mock_generate_no_progress_callback():
    """The callback path is optional; omitting it must not raise."""
    pipe = MockBrickPipeline()
    out = pipe.generate("a small house")
    assert out["brick_count"] == 12


def test_mock_generate_reports_caps_and_validates_limits():
    pipe = MockBrickPipeline()

    capped = pipe.generate("a small house", max_bricks=2)
    assert capped["brick_count"] == 2
    assert capped["metadata"]["termination_reason"] == "max_bricks"
    assert capped["metadata"]["hit_done"] is False

    timed_out = pipe.generate("a small house", max_seconds=0)
    assert timed_out["brick_count"] == 0
    assert timed_out["metadata"]["termination_reason"] == "max_seconds"

    with pytest.raises(ValueError, match="max_bricks"):
        pipe.generate("x", max_bricks=-1)
    with pytest.raises(ValueError, match="max_seconds"):
        pipe.generate("x", max_seconds=-1)
    with pytest.raises(ValueError, match="stability_check_interval"):
        pipe.generate("x", stability_check_interval=0)


def test_mock_generate_best_of_n_stamps_metadata():
    pipe = MockBrickPipeline()
    out = pipe.generate_best_of_n("x", n=5)
    assert out["metadata"]["n"] == 5
    assert out["metadata"]["picked_index"] == 0
    assert out["metadata"]["stable_rate"] == 1.0
    assert out["metadata"]["selection_strategy"] == "rank"


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


def test_get_brick_pipeline_returns_real_singleton_in_prod(monkeypatch, reset_pipeline_singletons):
    monkeypatch.setattr(bp, "LEGOGEN_DEV", False)

    created = []

    class FakeBrickPipeline:
        def __init__(self):
            created.append(self)

    monkeypatch.setattr(bp, "BrickPipeline", FakeBrickPipeline)

    a = bp.get_brick_pipeline()
    b = bp.get_brick_pipeline()

    assert a is b
    assert isinstance(a, FakeBrickPipeline)
    assert len(created) == 1


def test_get_stage1_pipeline_returns_real_singleton_in_prod(monkeypatch, reset_pipeline_singletons):
    monkeypatch.setattr(bp, "LEGOGEN_DEV", False)

    fake_stage1_module = types.ModuleType("backend.inference.stage1_pipeline")

    class FakeStage1Pipeline:
        pass

    fake_stage1_module.Stage1Pipeline = FakeStage1Pipeline
    monkeypatch.setitem(sys.modules, "backend.inference.stage1_pipeline", fake_stage1_module)

    a = bp._get_stage1_pipeline()
    b = bp._get_stage1_pipeline()

    assert a is b
    assert isinstance(a, FakeStage1Pipeline)
