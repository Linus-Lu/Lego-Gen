"""Tests for the Stage 1 dataset builder (COCO + ST2B caption matching)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.data_pipeline.build_stage1_dataset import (
    load_st2b_captions_by_category,
    match_coco_to_st2b,
    generate_description_from_label,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

CHAIR_LABEL = {
    "set_id": "chair-001",
    "object": "Simple office chair with four legs and a backrest.",
    "category": "Creator",
    "subcategory": "Furniture",
    "complexity": "simple",
    "total_parts": 30,
    "dominant_colors": ["Red", "Black"],
    "dimensions_estimate": {"width": "small", "height": "medium", "depth": "small"},
    "subassemblies": [],
    "build_hints": [],
}

CAR_LABEL = {
    "set_id": "car-001",
    "object": "Compact car with rounded roof and four wheels.",
    "category": "City",
    "subcategory": "Vehicles",
    "complexity": "medium",
    "total_parts": 80,
    "dominant_colors": ["Blue", "White"],
    "dimensions_estimate": {"width": "medium", "height": "small", "depth": "large"},
    "subassemblies": [],
    "build_hints": [],
}

TABLE_LABEL = {
    "set_id": "table-001",
    "object": "Rectangular table with a flat top and four legs.",
    "category": "Creator",
    "subcategory": "Furniture",
    "complexity": "simple",
    "total_parts": 20,
    "dominant_colors": ["Tan", "Dark Brown"],
    "dimensions_estimate": {"width": "large", "height": "medium", "depth": "large"},
    "subassemblies": [],
    "build_hints": [],
}


# ── Tests ─────────────────────────────────────────────────────────────────


def test_match_coco_to_st2b_chair():
    """Matching 'chair' COCO category should return a caption about chairs."""
    st2b_by_category = {
        "chair": [CHAIR_LABEL],
        "car": [CAR_LABEL],
        "table": [TABLE_LABEL],
    }
    result = match_coco_to_st2b("chair", st2b_by_category, seed=42)
    assert result is not None
    assert isinstance(result, str)
    # The returned caption should come from a chair label
    assert "chair" in result.lower()


def test_match_coco_to_st2b_unknown_category():
    """Unknown COCO category (not in COCO_TO_ST2B_CATEGORY) returns None."""
    st2b_by_category = {
        "chair": [CHAIR_LABEL],
        "car": [CAR_LABEL],
    }
    # "zebra" is not in COCO_TO_ST2B_CATEGORY
    result = match_coco_to_st2b("zebra", st2b_by_category, seed=42)
    assert result is None


def test_generate_description_from_label():
    """Generated description contains object name and colors."""
    description = generate_description_from_label(CHAIR_LABEL)
    assert isinstance(description, str)
    assert len(description) > 0
    # Should mention the object concept
    assert "chair" in description.lower()
    # Should mention at least one color
    has_color = any(
        color.lower() in description.lower()
        for color in CHAIR_LABEL["dominant_colors"]
    )
    assert has_color, f"Expected a color in: {description!r}"


def test_load_st2b_captions_by_category(tmp_path):
    """Grouping ST2B labels by object type works on a small test set."""
    # Write minimal ST2B label files to a temp directory
    labels = [CHAIR_LABEL, CAR_LABEL, TABLE_LABEL]
    for label in labels:
        p = tmp_path / f"{label['set_id']}.json"
        p.write_text(json.dumps(label))

    result = load_st2b_captions_by_category(tmp_path)

    assert isinstance(result, dict)
    # Should have at least a 'chair' and 'car' key
    assert "chair" in result
    assert "car" in result
    # Each value is a list of label dicts
    assert len(result["chair"]) >= 1
    assert len(result["car"]) >= 1
    # Labels for 'chair' category should contain object text about chairs
    chair_objects = [lbl["object"].lower() for lbl in result["chair"]]
    assert any("chair" in obj for obj in chair_objects)


def test_load_st2b_captions_skips_invalid_json(tmp_path):
    """Files that fail JSON parsing must be silently skipped, not crash."""
    # A valid chair label alongside a broken file.
    (tmp_path / "chair.json").write_text(json.dumps(CHAIR_LABEL))
    (tmp_path / "broken.json").write_text("not json {")
    result = load_st2b_captions_by_category(tmp_path)
    # Valid one still picked up; broken one ignored.
    assert "chair" in result
    assert len(result["chair"]) == 1


def test_load_st2b_captions_skips_empty_object_field(tmp_path):
    """Labels with an empty/missing 'object' field must be skipped."""
    empty_label = {"set_id": "empty-1", "object": ""}
    (tmp_path / "empty.json").write_text(json.dumps(empty_label))
    result = load_st2b_captions_by_category(tmp_path)
    # Nothing landed in any category — the empty label was skipped.
    assert all(len(v) == 0 for v in result.values()) or result == {}


def test_match_coco_to_st2b_mapped_but_no_candidates():
    """COCO category maps to an ST2B type but that type has no entries → None."""
    # 'chair' is in COCO_TO_ST2B_CATEGORY but the dict is empty so candidates=[].
    result = match_coco_to_st2b("chair", {}, seed=42)
    assert result is None


# ── Branch coverage for generate_description_from_label ──


def test_generate_description_empty_label_returns_empty_string():
    """Label with no obj/colors/dims fields produces empty string (parts is empty)."""
    description = generate_description_from_label({})
    assert description == ""


def test_generate_description_obj_only():
    """Only 'object' present → skips colors and dims branches."""
    description = generate_description_from_label({"object": "a small chair"})
    assert description == "a small chair."
    assert "primarily" not in description
    assert "form factor" not in description


def test_generate_description_skips_colors_when_absent():
    """No dominant_colors → the colors clause is skipped (132->136 branch)."""
    label = {
        "object": "a red chair",
        "dimensions_estimate": {"width": "small"},
    }
    description = generate_description_from_label(label)
    assert "primarily" not in description
    assert "form factor" in description


def test_generate_description_skips_dims_when_absent():
    """No dimensions_estimate → the dims clause is skipped (136->144 branch)."""
    label = {
        "object": "a red chair",
        "dominant_colors": ["Red"],
    }
    description = generate_description_from_label(label)
    assert "primarily" in description
    assert "form factor" not in description


def test_generate_description_skips_dims_when_all_empty():
    """dims present but all fields empty → size_parts is [] (141->144 branch)."""
    label = {
        "object": "a chair",
        "dimensions_estimate": {"width": "", "height": "", "depth": ""},
    }
    description = generate_description_from_label(label)
    assert "form factor" not in description
    assert description == "a chair."
