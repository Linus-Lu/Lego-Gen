"""Tests for backend.inference.postprocess_manual — JSON-to-build-steps conversion."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.inference.postprocess_manual import (
    _format_instruction,
    _format_title,
    json_to_steps,
    steps_to_summary,
)


# ── TestJsonToSteps ──────────────────────────────────────────────────


class TestJsonToSteps:
    def test_empty_description_returns_empty(self):
        assert json_to_steps({}) == []
        assert json_to_steps(None) == []

    def test_no_subassemblies_returns_single_fallback(self):
        desc = {"object": "Castle", "total_parts": 42}
        steps = json_to_steps(desc)
        assert len(steps) == 1
        assert steps[0]["step_number"] == 1
        assert "Castle" in steps[0]["title"]
        assert steps[0]["part_count"] == 42
        assert steps[0]["parts"] == []

    def test_fallback_missing_object(self):
        desc = {"total_parts": 10}
        steps = json_to_steps(desc)
        assert "model" in steps[0]["title"]

    def test_multiple_subassemblies_sorted_bottom_to_top(self):
        desc = {
            "subassemblies": [
                {
                    "name": "roof",
                    "type": "Roof Tiles",
                    "parts": [{"quantity": 4}],
                    "spatial": {"position": "top"},
                },
                {
                    "name": "base",
                    "type": "Plates",
                    "parts": [{"quantity": 2}],
                    "spatial": {"position": "bottom"},
                },
                {
                    "name": "walls",
                    "type": "Bricks",
                    "parts": [{"quantity": 6}],
                    "spatial": {"position": "center"},
                },
            ]
        }
        steps = json_to_steps(desc)
        assert len(steps) == 3
        assert "Base" in steps[0]["title"]
        assert "Walls" in steps[1]["title"]
        assert "Roof" in steps[2]["title"]

    def test_part_count_sums_quantities(self):
        desc = {
            "subassemblies": [
                {
                    "name": "base",
                    "type": "Plates",
                    "parts": [
                        {"quantity": 3},
                        {"quantity": 5},
                        {},  # missing quantity defaults to 1
                    ],
                    "spatial": {"position": "bottom"},
                }
            ]
        }
        steps = json_to_steps(desc)
        assert steps[0]["part_count"] == 9  # 3 + 5 + 1

    def test_step_number_starts_at_one(self, make_desc):
        steps = json_to_steps(make_desc())
        assert steps[0]["step_number"] == 1
        if len(steps) > 1:
            assert steps[1]["step_number"] == 2


# ── TestFormatTitle ──────────────────────────────────────────────────


class TestFormatTitle:
    def test_underscore_to_space_and_title_case(self):
        sa = {"name": "base_plate_assembly"}
        assert _format_title(sa) == "Build the Base Plate Assembly"

    def test_missing_name_defaults_to_section(self):
        assert _format_title({}) == "Build the Section"


# ── TestFormatInstruction ────────────────────────────────────────────


class TestFormatInstruction:
    def test_includes_position_hint(self):
        sa = {
            "type": "Plates",
            "parts": [{"quantity": 5}],
            "spatial": {"position": "bottom", "connects_to": []},
        }
        instr = _format_instruction(sa, {})
        assert "at the bottom" in instr
        assert "5 Plates" in instr

    def test_connection_hint_max_two_targets(self):
        sa = {
            "type": "Bricks",
            "parts": [{"quantity": 3}],
            "spatial": {
                "position": "",
                "connects_to": ["base_plate", "side_wall", "extra_thing"],
            },
        }
        instr = _format_instruction(sa, {})
        assert "base plate" in instr
        assert "side wall" in instr
        # Third item should not appear
        assert "extra thing" not in instr

    def test_no_position_no_connection(self):
        sa = {
            "type": "Bricks",
            "parts": [{"quantity": 2}],
            "spatial": {"position": "", "connects_to": []},
        }
        instr = _format_instruction(sa, {})
        assert "at the" not in instr
        assert "connects to" not in instr.lower()


# ── TestStepsToSummary ───────────────────────────────────────────────


class TestStepsToSummary:
    def test_correct_format(self):
        steps = [
            {
                "step_number": 1,
                "title": "Build the Base",
                "instruction": "Place the plate.",
                "part_count": 3,
            }
        ]
        summary = steps_to_summary(steps)
        assert "Step 1: Build the Base" in summary
        assert "Place the plate." in summary
        assert "Parts needed: 3" in summary

    def test_empty_steps_returns_empty(self):
        assert steps_to_summary([]) == ""
