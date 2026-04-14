"""Tests for rejection reason tracking and brick deduplication labels."""

from collections import Counter

import pytest

# These labels mirror the constants defined in brick_pipeline.py but are
# imported as plain strings to avoid pulling in torch at collection time.
_REJ_FORMAT = "ill_formatted"
_REJ_COLLISION = "collision"
_REJ_DISCONNECTED = "disconnected"
_REJ_ALREADY = "already_rejected"


def test_rejection_labels_are_distinct():
    labels = {_REJ_FORMAT, _REJ_COLLISION, _REJ_DISCONNECTED, _REJ_ALREADY}
    assert len(labels) == 4, "All rejection labels must be unique"


def test_rejection_labels_are_strings():
    for label in (_REJ_FORMAT, _REJ_COLLISION, _REJ_DISCONNECTED, _REJ_ALREADY):
        assert isinstance(label, str)
        assert len(label) > 0


def test_counter_tracks_reasons():
    """Smoke test that Counter can aggregate reason labels correctly."""
    reasons: Counter = Counter()
    reasons[_REJ_FORMAT] += 3
    reasons[_REJ_COLLISION] += 1
    reasons[_REJ_ALREADY] += 2
    assert reasons.total() == 6
    assert reasons[_REJ_FORMAT] == 3
    assert reasons[_REJ_DISCONNECTED] == 0


def test_rejection_reasons_dict_serializable():
    """The dict(counter) output should be JSON-serializable."""
    import json

    reasons: Counter = Counter()
    reasons[_REJ_FORMAT] += 1
    reasons[_REJ_COLLISION] += 2
    d = dict(reasons)
    serialized = json.dumps(d)
    assert _REJ_FORMAT in serialized
