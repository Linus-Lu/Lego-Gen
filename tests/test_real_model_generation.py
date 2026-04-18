"""Opt-in GPU smoke tests for real LEGOGen model generation.

These tests exercise the actual `LEGOGEN_DEV=0` inference stack, including
the prod-mode singleton factories and real Qwen forward passes. They are
intentionally opt-in because they require a CUDA GPU, the heavyweight runtime
dependencies, and locally available model weights / cache.
"""

from __future__ import annotations

import os

import pytest
from PIL import Image, ImageDraw


REAL_MODEL_TEST_ENV = "LEGOGEN_RUN_REAL_MODEL_TESTS"
REAL_STAGE2_PROMPT = "a small red house with a dark red tiled roof"


def _require_real_model_opt_in():
    if os.environ.get(REAL_MODEL_TEST_ENV) != "1":
        pytest.skip(
            f"Set {REAL_MODEL_TEST_ENV}=1 to run real-model GPU generation tests."
        )

    import bitsandbytes  # noqa: F401
    import peft  # noqa: F401
    import torch
    import transformers  # noqa: F401

    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for real-model generation tests.")

    return torch


def _make_test_image() -> Image.Image:
    image = Image.new("RGB", (256, 256), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((52, 124, 204, 220), fill=(200, 36, 36))
    draw.polygon(((40, 124), (128, 56), (216, 124)), fill=(110, 12, 12))
    draw.rectangle((112, 172, 144, 220), fill=(65, 35, 15))
    draw.rectangle((72, 148, 102, 180), fill=(235, 235, 255))
    draw.rectangle((154, 148, 184, 180), fill=(235, 235, 255))
    return image


def _assert_valid_generation_payload(result: dict, *, expected_max_bricks: int) -> None:
    from backend.brick.parser import parse_brick_sequence

    assert isinstance(result["bricks"], str)
    assert isinstance(result["brick_count"], int)
    assert isinstance(result["stable"], bool)
    assert isinstance(result["metadata"], dict)
    assert result["brick_count"] == len(
        [line for line in result["bricks"].splitlines() if line.strip()]
    )
    assert result["brick_count"] == expected_max_bricks
    assert result["stable"] is True
    assert result["metadata"]["requested_max_bricks"] == expected_max_bricks
    assert result["metadata"]["termination_reason"] in {
        "done",
        "max_bricks",
        "max_rejections",
        "max_rollbacks",
        "max_seconds",
    }

    bricks = parse_brick_sequence(result["bricks"])
    assert len(bricks) == result["brick_count"] == expected_max_bricks


@pytest.fixture(scope="module")
def real_model_stack():
    torch = _require_real_model_opt_in()

    import backend.inference.brick_pipeline as bp

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("LEGOGEN_DEV", "0")
    monkeypatch.setattr(bp, "LEGOGEN_DEV", False)

    bp._brick_instance = None
    bp._stage1_instance = None

    brick_pipeline = bp.get_brick_pipeline()
    stage1_pipeline = bp._get_stage1_pipeline()

    yield {
        "torch": torch,
        "bp": bp,
        "brick_pipeline": brick_pipeline,
        "stage1_pipeline": stage1_pipeline,
    }

    bp._brick_instance = None
    bp._stage1_instance = None
    monkeypatch.undo()
    torch.cuda.empty_cache()


@pytest.mark.gpu
def test_real_stage1_pipeline_describe_returns_clean_caption(real_model_stack):
    image = _make_test_image()

    caption = real_model_stack["stage1_pipeline"].describe(image)

    assert isinstance(caption, str)
    assert caption.strip()
    assert "<think>" not in caption
    assert "</think>" not in caption
    assert len(caption) <= 512


@pytest.mark.gpu
def test_real_brick_pipeline_generate_returns_parseable_stable_payload(real_model_stack):
    events = []

    result = real_model_stack["brick_pipeline"].generate(
        REAL_STAGE2_PROMPT,
        on_progress=events.append,
        max_bricks=1,
        max_seconds=90,
        stability_check_interval=1,
    )

    _assert_valid_generation_payload(result, expected_max_bricks=1)
    assert any(evt["type"] == "brick" for evt in events)


@pytest.mark.gpu
def test_real_generate_from_image_runs_stage1_and_stage2(real_model_stack, monkeypatch):
    brick_pipeline = real_model_stack["brick_pipeline"]
    real_generate = brick_pipeline.generate

    def bounded_generate(caption, on_progress=None, *, should_cancel=None):
        return real_generate(
            caption,
            on_progress=on_progress,
            max_bricks=1,
            max_seconds=90,
            stability_check_interval=1,
            should_cancel=should_cancel,
        )

    monkeypatch.setattr(brick_pipeline, "generate", bounded_generate)

    events = []
    result = brick_pipeline.generate_from_image(_make_test_image(), on_progress=events.append)

    assert isinstance(result["caption"], str)
    assert result["caption"].strip()
    assert "<think>" not in result["caption"]
    assert events
    assert events[0]["type"] == "caption"
    assert events[0]["caption"] == result["caption"]
    _assert_valid_generation_payload(result, expected_max_bricks=1)
