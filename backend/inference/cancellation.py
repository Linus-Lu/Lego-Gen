"""Helpers for cooperative cancellation inside transformer generation loops."""

from __future__ import annotations

from typing import Callable

ShouldCancel = Callable[[], bool]


class GenerationCancelled(Exception):
    """Raised when a generation request has been cancelled cooperatively."""


def is_cancelled(should_cancel: ShouldCancel | None) -> bool:
    return bool(should_cancel()) if callable(should_cancel) else False


def raise_if_cancelled(should_cancel: ShouldCancel | None) -> None:
    if is_cancelled(should_cancel):
        raise GenerationCancelled()


def build_stopping_criteria(should_cancel: ShouldCancel | None):
    """Return a transformers StoppingCriteriaList that stops on cancellation.

    The helper returns ``None`` when cancellation is disabled or transformers
    is unavailable in the current environment.
    """
    if should_cancel is None:
        return None

    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except ImportError:
        return None

    class _CancelStoppingCriteria(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return is_cancelled(should_cancel)

    return StoppingCriteriaList([_CancelStoppingCriteria()])
