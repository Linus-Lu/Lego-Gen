"""Tests for backend.inference.constraint_engine — JSON validation, repair, and enforcement."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.inference.constraint_engine import (
    POSITION_ORDER,
    REQUIRED_FIELDS,
    VALID_COMPLEXITY,
    VALID_DIMENSIONS,
    VALID_ORIENTATIONS,
    VALID_POSITIONS,
    enforce_valid_values,
    repair_connects_to,
    repair_json_string,
    safe_parse_and_validate,
    validate_and_repair_dict,
    validate_lego_json,
    validate_part_ids,
    validate_structural_order,
)


# ── TestValidateLegoJson ─────────────────────────────────────────────


class TestValidateLegoJson:
    def test_valid_description_passes(self, make_desc):
        is_valid, errors = validate_lego_json(make_desc())
        assert is_valid is True
        assert errors == []

    def test_missing_required_field(self, make_desc):
        desc = make_desc()
        del desc["object"]
        is_valid, errors = validate_lego_json(desc)
        assert is_valid is False
        assert any("Missing field: object" in e for e in errors)

    def test_wrong_type_detected(self, make_desc):
        desc = make_desc(total_parts="not_an_int")
        is_valid, errors = validate_lego_json(desc)
        assert is_valid is False
        assert any("total_parts" in e and "int" in e for e in errors)

    def test_invalid_complexity_flagged(self, make_desc):
        desc = make_desc(complexity="mega")
        is_valid, errors = validate_lego_json(desc)
        assert is_valid is False
        assert any("Invalid complexity" in e for e in errors)

    def test_invalid_dimension_flagged(self, make_desc):
        desc = make_desc(
            dimensions_estimate={"width": "huge", "height": "small", "depth": "small"}
        )
        is_valid, errors = validate_lego_json(desc)
        assert is_valid is False
        assert any("Invalid dimension" in e for e in errors)

    def test_subassembly_missing_key(self, make_desc):
        desc = make_desc(subassemblies=[{"name": "base", "type": "Plates"}])
        is_valid, errors = validate_lego_json(desc)
        assert is_valid is False
        assert any("missing 'parts'" in e for e in errors)
        assert any("missing 'spatial'" in e for e in errors)

    def test_invalid_spatial_position(self, make_desc):
        desc = make_desc()
        desc["subassemblies"][0]["spatial"]["position"] = "underground"
        is_valid, errors = validate_lego_json(desc)
        assert is_valid is False
        assert any("invalid position" in e for e in errors)

    def test_empty_dict_reports_all_missing(self):
        is_valid, errors = validate_lego_json({})
        assert is_valid is False
        assert len(errors) == len(REQUIRED_FIELDS)


# ── TestRepairJsonString ─────────────────────────────────────────────


class TestRepairJsonString:
    def test_valid_json_unchanged(self):
        raw = '{"a": 1, "b": [2, 3]}'
        result = repair_json_string(raw)
        assert json.loads(result) == {"a": 1, "b": [2, 3]}

    def test_trailing_comma_before_bracket(self):
        raw = '{"items": [1, 2, 3,]}'
        result = repair_json_string(raw)
        parsed = json.loads(result)
        assert parsed["items"] == [1, 2, 3]

    def test_trailing_comma_before_brace(self):
        raw = '{"a": 1, "b": 2,}'
        result = repair_json_string(raw)
        parsed = json.loads(result)
        assert parsed == {"a": 1, "b": 2}

    def test_unclosed_brackets(self):
        raw = '{"items": [1, 2, 3'
        result = repair_json_string(raw)
        parsed = json.loads(result)
        assert parsed["items"] == [1, 2, 3]

    def test_unclosed_braces(self):
        raw = '{"a": {"b": 1}'
        result = repair_json_string(raw)
        parsed = json.loads(result)
        assert parsed["a"]["b"] == 1

    def test_odd_quotes_closed(self):
        raw = '{"object": "House'
        result = repair_json_string(raw)
        # Should be parseable after repair
        parsed = json.loads(result)
        assert "House" in str(parsed)

    def test_truncated_json_end_to_end(self):
        raw = '{"set_id": "001", "total_parts": 5,'
        result = repair_json_string(raw)
        parsed = json.loads(result)
        assert parsed["set_id"] == "001"
        assert parsed["total_parts"] == 5


# ── TestEnforceValidValues ───────────────────────────────────────────


class TestEnforceValidValues:
    def test_complexity_inferred_from_total_parts(self, make_desc):
        thresholds = [
            (30, "simple"),
            (50, "intermediate"),
            (199, "intermediate"),
            (200, "advanced"),
            (499, "advanced"),
            (500, "expert"),
        ]
        for parts, expected in thresholds:
            desc = make_desc(total_parts=parts, complexity="bogus")
            result = enforce_valid_values(desc)
            assert result["complexity"] == expected, f"Failed for total_parts={parts}"

    def test_invalid_dimensions_default_to_medium(self, make_desc):
        desc = make_desc(
            dimensions_estimate={"width": "xxx", "height": "yyy", "depth": "zzz"}
        )
        result = enforce_valid_values(desc)
        for key in ("width", "height", "depth"):
            assert result["dimensions_estimate"][key] == "medium"

    def test_invalid_position_and_orientation_defaulted(self, make_desc):
        desc = make_desc()
        desc["subassemblies"][0]["spatial"]["position"] = "underground"
        desc["subassemblies"][0]["spatial"]["orientation"] = "diagonal"
        result = enforce_valid_values(desc)
        assert result["subassemblies"][0]["spatial"]["position"] == "center"
        assert result["subassemblies"][0]["spatial"]["orientation"] == "upright"

    def test_negative_total_parts_becomes_zero(self, make_desc):
        desc = make_desc(total_parts=-5)
        result = enforce_valid_values(desc)
        assert result["total_parts"] == 0

    def test_float_total_parts_becomes_zero(self, make_desc):
        desc = make_desc(total_parts=3.14)
        result = enforce_valid_values(desc)
        assert result["total_parts"] == 0


# ── TestValidatePartIds ──────────────────────────────────────────────


class TestValidatePartIds:
    def test_known_parts_pass_unknown_flagged(self, make_desc):
        desc = make_desc()
        known = {"3020"}
        warnings = validate_part_ids(desc, known)
        # 3020 is known; 3001 (in walls) is not
        assert any("3001" in w for w in warnings)
        assert not any("3020" in w for w in warnings)

    def test_empty_known_set_flags_all(self, make_desc):
        desc = make_desc()
        warnings = validate_part_ids(desc, set())
        assert len(warnings) >= 2  # at least one per subassembly


# ── TestRepairConnectsTo ─────────────────────────────────────────────


class TestRepairConnectsTo:
    def test_invalid_refs_removed_valid_kept(self, make_desc):
        desc = make_desc()
        # "walls" is a real subassembly name; "nonexistent" is not
        desc["subassemblies"][0]["spatial"]["connects_to"] = [
            "walls",
            "nonexistent",
        ]
        result = repair_connects_to(desc)
        connects = result["subassemblies"][0]["spatial"]["connects_to"]
        assert "walls" in connects
        assert "nonexistent" not in connects

    def test_self_reference_preserved(self, make_desc):
        desc = make_desc()
        desc["subassemblies"][0]["spatial"]["connects_to"] = ["base"]
        result = repair_connects_to(desc)
        assert "base" in result["subassemblies"][0]["spatial"]["connects_to"]


# ── TestValidateStructuralOrder ──────────────────────────────────────


class TestValidateStructuralOrder:
    def test_correct_order_passes(self, make_desc):
        # Default make_desc has bottom then center — correct
        warnings = validate_structural_order(make_desc())
        assert warnings == []

    def test_incorrect_order_warned(self, make_desc):
        desc = make_desc()
        desc["subassemblies"][0]["spatial"]["position"] = "top"
        desc["subassemblies"][1]["spatial"]["position"] = "bottom"
        warnings = validate_structural_order(desc)
        assert any("not ordered" in w for w in warnings)

    def test_empty_subassemblies_passes(self, make_desc):
        desc = make_desc(subassemblies=[])
        warnings = validate_structural_order(desc)
        assert warnings == []


# ── TestSafeParseAndValidate ─────────────────────────────────────────


class TestSafeParseAndValidate:
    def test_valid_json_roundtrip(self, make_desc):
        desc = make_desc()
        raw = json.dumps(desc)
        result, errors = safe_parse_and_validate(raw)
        assert result is not None
        assert result["set_id"] == "test-001"

    def test_malformed_json_repaired(self, make_desc):
        desc = make_desc()
        raw = json.dumps(desc)
        # Truncate to create malformed JSON
        truncated = raw[:len(raw) // 2]
        result, errors = safe_parse_and_validate(truncated)
        # Should either repair successfully or return None
        # The key is it doesn't crash
        assert isinstance(errors, list)

    def test_unparseable_returns_none(self):
        result, errors = safe_parse_and_validate("not json at all")
        assert result is None
        assert any("Failed to parse" in e for e in errors)


# ── TestValidateAndRepairDict ────────────────────────────────────────


class TestValidateAndRepairDict:
    def test_valid_dict_returns_no_errors(self, make_desc):
        desc = make_desc()
        result, errors = validate_and_repair_dict(desc)
        assert errors == []
        assert result["set_id"] == "test-001"

    def test_invalid_values_repaired(self, make_desc):
        desc = make_desc(complexity="bogus", total_parts=30)
        result, errors = validate_and_repair_dict(desc)
        # Validation reports the invalid complexity, but enforce fixes it
        assert result["complexity"] == "simple"
