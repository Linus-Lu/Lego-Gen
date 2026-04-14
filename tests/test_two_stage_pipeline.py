"""Tests for two-stage inference pipeline."""
import os
import pytest


@pytest.fixture(autouse=True)
def dev_mode():
    """Run tests in dev mode (mock pipeline)."""
    os.environ["LEGOGEN_DEV"] = "1"
    yield
    # Reset singleton so next test gets fresh pipeline
    import backend.inference.pipeline as p
    p._pipeline_instance = None


def test_text_input_generates_bricks():
    """Text-only input should produce brick coordinates via Stage 2."""
    from backend.inference.pipeline import get_pipeline

    pipeline = get_pipeline()
    result = pipeline.generate_brick_build("a small red house")

    assert "bricks" in result
    assert "brick_count" in result
    assert "stable" in result
    assert "metadata" in result


def test_image_input_generates_bricks():
    """Image input should go through Stage 1 caption then Stage 2 bricks."""
    from backend.inference.pipeline import get_pipeline
    from PIL import Image

    pipeline = get_pipeline()
    dummy_image = Image.new("RGB", (224, 224), (128, 128, 128))
    result = pipeline.generate_brick_build_from_image(dummy_image)

    assert "bricks" in result
    assert "caption" in result
    assert result["brick_count"] > 0


def test_mock_pipeline_returns_valid_structure():
    """Mock pipeline should return complete brick structure for dev testing."""
    from backend.inference.pipeline import get_pipeline

    pipeline = get_pipeline()
    result = pipeline.generate_brick_build("test")

    assert result["brick_count"] > 0
    assert result["stable"] is True
    assert "model_version" in result["metadata"]
    # Verify brick format is parseable
    lines = result["bricks"].strip().split("\n")
    assert len(lines) == result["brick_count"]


def test_stage1_description():
    """Stage 1 should return a text description from an image."""
    from backend.inference.pipeline import get_pipeline
    from PIL import Image

    pipeline = get_pipeline()
    dummy_image = Image.new("RGB", (224, 224), (128, 128, 128))
    description = pipeline.describe_image_stage1(dummy_image)

    assert isinstance(description, str)
    assert len(description) > 0
