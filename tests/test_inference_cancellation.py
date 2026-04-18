import builtins
import sys
import types

import pytest

from backend.inference.cancellation import (
    GenerationCancelled,
    build_stopping_criteria,
    is_cancelled,
    raise_if_cancelled,
)


def test_is_cancelled_handles_missing_and_callable_inputs():
    assert is_cancelled(None) is False
    assert is_cancelled(lambda: False) is False
    assert is_cancelled(lambda: True) is True


def test_raise_if_cancelled_raises_generation_cancelled():
    with pytest.raises(GenerationCancelled):
        raise_if_cancelled(lambda: True)


def test_build_stopping_criteria_returns_none_without_callback():
    assert build_stopping_criteria(None) is None


def test_build_stopping_criteria_returns_none_when_transformers_is_unavailable(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("missing transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert build_stopping_criteria(lambda: True) is None


def test_build_stopping_criteria_uses_transformers_stopping_criteria(monkeypatch):
    fake_transformers = types.ModuleType("transformers")

    class FakeStoppingCriteria:
        pass

    class FakeStoppingCriteriaList(list):
        pass

    fake_transformers.StoppingCriteria = FakeStoppingCriteria
    fake_transformers.StoppingCriteriaList = FakeStoppingCriteriaList
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    criteria = build_stopping_criteria(lambda: True)

    assert isinstance(criteria, FakeStoppingCriteriaList)
    assert len(criteria) == 1
    assert criteria[0](None, None) is True
