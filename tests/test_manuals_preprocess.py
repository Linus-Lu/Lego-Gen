"""Tests for backend.data_pipeline.manuals_preprocess — Rebrickable data transformation."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.data_pipeline.manuals_preprocess import (
    _generate_build_hints,
    classify_complexity,
    estimate_dimensions,
    validate_label,
)


# ── classify_complexity ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "num_parts,expected",
    [
        (0, "simple"),
        (30, "simple"),
        (49, "simple"),
        (50, "intermediate"),
        (199, "intermediate"),
        (200, "advanced"),
        (499, "advanced"),
        (500, "expert"),
        (1000, "expert"),
    ],
)
def test_classify_complexity(num_parts, expected):
    assert classify_complexity(num_parts) == expected


# ── estimate_dimensions ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "num_parts,expected_width",
    [
        (10, "small"),
        (49, "small"),
        (50, "medium"),
        (149, "medium"),
        (150, "medium"),
        (299, "medium"),
        (300, "large"),
    ],
)
def test_estimate_dimensions(num_parts, expected_width):
    dims = estimate_dimensions(num_parts)
    assert dims["width"] == expected_width
    assert set(dims.keys()) == {"width", "height", "depth"}


# ── _generate_build_hints ────────────────────────────────────────────


def test_generate_build_hints_always_includes_foundation():
    hints = _generate_build_hints(10, ["City"])
    assert any("foundation" in h.lower() or "base" in h.lower() for h in hints)


def test_generate_build_hints_large_build_sorting():
    hints = _generate_build_hints(150, [])
    assert any("sort" in h.lower() or "Sort" in h for h in hints)


def test_generate_build_hints_very_large_build_sections():
    hints = _generate_build_hints(250, [])
    assert any("section" in h.lower() for h in hints)


def test_generate_build_hints_category_specific():
    city_hints = _generate_build_hints(10, ["City"])
    assert any("wall" in h.lower() or "roof" in h.lower() for h in city_hints)

    technic_hints = _generate_build_hints(10, ["Technic"])
    assert any("gear" in h.lower() or "axle" in h.lower() for h in technic_hints)

    sw_hints = _generate_build_hints(10, ["Star Wars"])
    assert any("hull" in h.lower() for h in sw_hints)


# ── validate_label ───────────────────────────────────────────────────


def test_validate_label_valid(make_desc):
    is_valid, errors = validate_label(make_desc())
    assert is_valid is True
    assert errors == []


def test_validate_label_missing_fields():
    is_valid, errors = validate_label({"set_id": "001"})
    assert is_valid is False
    assert len(errors) >= 5  # many required fields missing


def test_validate_label_invalid_complexity(make_desc):
    desc = make_desc(complexity="mega")
    is_valid, errors = validate_label(desc)
    assert is_valid is False
    assert any("complexity" in e.lower() for e in errors)
