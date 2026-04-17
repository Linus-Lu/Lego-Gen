"""Smoke test for Stage 1 inference.

The real Stage1Pipeline (transformers loader) is omitted from coverage; this
file only exercises the dev-mode _MockStage1 shim that get_stage1_pipeline()
returns under LEGOGEN_DEV=1."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.inference.brick_pipeline import _MockStage1


def test_mock_stage1_describe_returns_string():
    mock = _MockStage1()
    caption = mock.describe(object())
    assert isinstance(caption, str)
    assert len(caption) > 0


def test_mock_stage1_describe_is_stable():
    """Callers rely on the caption being deterministic for dev-mode testing."""
    assert _MockStage1().describe(None) == _MockStage1().describe(None)
