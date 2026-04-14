"""Tests for training metric functions."""

import pytest

from backend.training.utils import (
    compute_json_validity_rate,
    compute_field_accuracy,
    compute_color_f1,
    compute_parts_f1,
    compute_all_metrics,
    compute_structural_coherence,
    compute_part_realism,
    compute_build_feasibility,
)


# ── compute_json_validity_rate ───────────────────────────────────────

def test_json_validity_all_valid():
    preds = ['{"a": 1}', '{"b": 2}']
    assert compute_json_validity_rate(preds) == 1.0


def test_json_validity_none_valid():
    preds = ["garbage", "not json"]
    assert compute_json_validity_rate(preds) == 0.0


def test_json_validity_mixed():
    preds = ['{"a": 1}', "bad", '{"b": 2}', "worse"]
    assert compute_json_validity_rate(preds) == 0.5


def test_json_validity_empty_list():
    assert compute_json_validity_rate([]) == 0.0


def test_json_validity_json_in_text():
    """JSON embedded in surrounding text should still be detected."""
    preds = ['Some text {"key": "val"} trailing']
    assert compute_json_validity_rate(preds) == 1.0


# ── compute_field_accuracy ───────────────────────────────────────────

def test_field_accuracy_all_match():
    preds = [{"cat": "A"}, {"cat": "B"}]
    refs = [{"cat": "A"}, {"cat": "B"}]
    assert compute_field_accuracy(preds, refs, "cat") == 1.0


def test_field_accuracy_none_match():
    preds = [{"cat": "X"}, {"cat": "Y"}]
    refs = [{"cat": "A"}, {"cat": "B"}]
    assert compute_field_accuracy(preds, refs, "cat") == 0.0


def test_field_accuracy_partial():
    preds = [{"cat": "A"}, {"cat": "X"}]
    refs = [{"cat": "A"}, {"cat": "B"}]
    assert compute_field_accuracy(preds, refs, "cat") == 0.5


def test_field_accuracy_missing_field():
    """Field absent from references should be skipped (returns 0.0)."""
    preds = [{"other": "X"}]
    refs = [{"other": "X"}]
    assert compute_field_accuracy(preds, refs, "cat") == 0.0


# ── compute_color_f1 ────────────────────────────────────────────────

def test_color_f1_perfect():
    assert compute_color_f1(["Red", "Blue"], ["Red", "Blue"]) == 1.0


def test_color_f1_no_overlap():
    assert compute_color_f1(["Red"], ["Blue"]) == 0.0


def test_color_f1_partial():
    """pred={red,blue}, ref={blue,green} -> tp=1, precision=1/2, recall=1/2, F1=0.5"""
    result = compute_color_f1(["Red", "Blue"], ["Blue", "Green"])
    assert abs(result - 0.5) < 1e-9


def test_color_f1_both_empty():
    assert compute_color_f1([], []) == 1.0


def test_color_f1_pred_empty_ref_not():
    assert compute_color_f1([], ["Red"]) == 0.0


def test_color_f1_case_insensitive():
    assert compute_color_f1(["RED"], ["red"]) == 1.0


# ── compute_parts_f1 ────────────────────────────────────────────────

def test_parts_f1_perfect():
    parts = [{"part_id": "3001", "quantity": 2}]
    assert compute_parts_f1(parts, parts) == 1.0


def test_parts_f1_no_overlap():
    pred = [{"part_id": "3001", "quantity": 1}]
    ref = [{"part_id": "9999", "quantity": 1}]
    assert compute_parts_f1(pred, ref) == 0.0


def test_parts_f1_partial_quantity():
    """pred has 3 of part 3001, ref has 2 -> tp=2, precision=2/3, recall=2/2=1.0"""
    pred = [{"part_id": "3001", "quantity": 3}]
    ref = [{"part_id": "3001", "quantity": 2}]
    result = compute_parts_f1(pred, ref)
    expected = 2 * (2 / 3) * 1.0 / ((2 / 3) + 1.0)
    assert abs(result - expected) < 1e-9


# ── compute_structural_coherence ─────────────────────────────────────

def test_structural_coherence_well_ordered():
    pred = {
        "subassemblies": [
            {"name": "base", "spatial": {"position": "bottom", "connects_to": []}},
            {"name": "body", "spatial": {"position": "center", "connects_to": ["base"]}},
            {"name": "roof", "spatial": {"position": "top", "connects_to": ["body"]}},
        ]
    }
    score = compute_structural_coherence(pred)
    assert score == 1.0


def test_structural_coherence_no_subassemblies():
    assert compute_structural_coherence({}) == 0.0
    assert compute_structural_coherence({"subassemblies": []}) == 0.0


def test_structural_coherence_no_bottom():
    """Missing 'bottom' position reduces the score."""
    pred = {
        "subassemblies": [
            {"name": "a", "spatial": {"position": "center", "connects_to": []}},
        ]
    }
    score = compute_structural_coherence(pred)
    assert score < 1.0


# ── compute_part_realism ────────────────────────────────────────────

def test_part_realism_all_known():
    pred = {
        "subassemblies": [
            {"parts": [{"part_id": "3001"}, {"part_id": "3003"}]}
        ]
    }
    known = {"3001", "3003", "3005"}
    assert compute_part_realism(pred, known) == 1.0


def test_part_realism_none_known():
    pred = {
        "subassemblies": [
            {"parts": [{"part_id": "9999"}]}
        ]
    }
    known = {"3001"}
    assert compute_part_realism(pred, known) == 0.0


def test_part_realism_empty_catalog():
    pred = {"subassemblies": [{"parts": [{"part_id": "3001"}]}]}
    assert compute_part_realism(pred, set()) == 0.0


def test_part_realism_no_parts():
    pred = {"subassemblies": [{"parts": []}]}
    assert compute_part_realism(pred, {"3001"}) == 0.0


# ── compute_build_feasibility ───────────────────────────────────────

def test_build_feasibility_consistent():
    pred = {
        "total_parts": 3,
        "subassemblies": [
            {"parts": [{"quantity": 2}]},
            {"parts": [{"quantity": 1}]},
        ],
    }
    assert compute_build_feasibility(pred) == 1.0


def test_build_feasibility_mismatch():
    """total_parts way off from actual sum reduces score."""
    pred = {
        "total_parts": 100,
        "subassemblies": [{"parts": [{"quantity": 1}]}],
    }
    score = compute_build_feasibility(pred)
    assert score < 1.0


def test_build_feasibility_no_subassemblies():
    pred = {"total_parts": 5}
    assert compute_build_feasibility(pred) == 0.0


def test_build_feasibility_empty_subassembly():
    """Subassembly with no parts fails the 'each sub has parts' check."""
    pred = {
        "total_parts": 0,
        "subassemblies": [{"parts": []}],
    }
    score = compute_build_feasibility(pred)
    assert score < 1.0


# ── compute_all_metrics ──────────────────────────────────────────────

def test_compute_all_metrics_smoke():
    """Smoke test: runs end-to-end and returns expected keys."""
    pred = {
        "category": "City",
        "subcategory": "Vehicles",
        "complexity": "simple",
        "dominant_colors": ["Red", "Blue"],
        "subassemblies": [
            {"parts": [{"part_id": "3001", "quantity": 2}]}
        ],
    }
    ref = {
        "category": "City",
        "subcategory": "Vehicles",
        "complexity": "medium",
        "dominant_colors": ["Red", "Green"],
        "subassemblies": [
            {"parts": [{"part_id": "3001", "quantity": 2}]}
        ],
    }
    metrics = compute_all_metrics([pred], [ref])
    assert "category_accuracy" in metrics
    assert "subcategory_accuracy" in metrics
    assert "complexity_accuracy" in metrics
    assert "color_f1" in metrics
    assert "parts_f1" in metrics
    assert float(metrics["category_accuracy"]) == 1.0
    assert float(metrics["parts_f1"]) == 1.0
