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
    p._unified_instance = None


def test_text_input_skips_stage1():
    """Text-only input should go directly to Stage 2."""
    from backend.inference.pipeline import get_pipeline

    pipeline = get_pipeline()
    result = pipeline.generate_build_from_text("a small red house")

    assert "description" in result
    assert "steps" in result
    assert "metadata" in result


def test_generate_response_has_required_fields():
    """Pipeline output should have description, steps, metadata, validation."""
    from backend.inference.pipeline import get_pipeline

    pipeline = get_pipeline()
    result = pipeline.generate_build_from_text("a blue car")

    assert isinstance(result["description"], dict)
    assert isinstance(result["steps"], list)
    assert "model_version" in result["metadata"]
    assert "validation" in result


def test_mock_pipeline_returns_valid_structure():
    """Mock pipeline should return complete structure for dev testing."""
    from backend.inference.pipeline import get_pipeline

    pipeline = get_pipeline()
    result = pipeline.generate_build_from_text("test")

    desc = result["description"]
    assert "object" in desc
    assert "subassemblies" in desc
    assert len(result["steps"]) > 0
    assert result["steps"][0]["step_number"] == 1
