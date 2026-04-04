"""Tests for the build stability and legality checker."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from dataclasses import asdict

from backend.inference.stability_checker import StabilityChecker, ValidationReport


# ── Fixtures ──────────────────────────────────────────────────────────

def _make_desc(**overrides):
    """Build a minimal valid description dict."""
    base = {
        "set_id": "test-001",
        "object": "Test Build",
        "category": "City",
        "subcategory": "Test",
        "complexity": "simple",
        "total_parts": 10,
        "dominant_colors": ["Red"],
        "dimensions_estimate": {"width": "small", "height": "small", "depth": "small"},
        "subassemblies": [
            {
                "name": "base",
                "type": "Plates",
                "parts": [
                    {"part_id": "3020", "name": "Plate 2x4", "category": "Plates", "color": "Red", "color_hex": "#C91A09", "is_trans": False, "quantity": 5},
                ],
                "spatial": {"position": "bottom", "orientation": "flat", "connects_to": ["walls"]},
            },
            {
                "name": "walls",
                "type": "Bricks",
                "parts": [
                    {"part_id": "3001", "name": "Brick 2x4", "category": "Bricks", "color": "White", "color_hex": "#FFFFFF", "is_trans": False, "quantity": 5},
                ],
                "spatial": {"position": "center", "orientation": "upright", "connects_to": ["base"]},
            },
        ],
        "build_hints": ["Start with the base"],
    }
    base.update(overrides)
    return base


def _mock_house():
    """The mock house from MockPipeline — should score high."""
    return {
        "set_id": "mock-001",
        "object": "Cozy Family House",
        "category": "City",
        "subcategory": "Residential",
        "complexity": "intermediate",
        "total_parts": 86,
        "dominant_colors": ["Red", "White", "Bright Orange"],
        "dimensions_estimate": {"width": "medium", "height": "medium", "depth": "small"},
        "subassemblies": [
            {
                "name": "base_plate",
                "type": "Baseplates",
                "parts": [
                    {"part_id": "3811", "name": "Baseplate 32x32", "category": "Baseplates", "color": "Green", "color_hex": "#237841", "is_trans": False, "quantity": 1},
                    {"part_id": "3020", "name": "Plate 2x4", "category": "Plates", "color": "Dark Tan", "color_hex": "#958A73", "is_trans": False, "quantity": 4},
                ],
                "spatial": {"position": "bottom", "orientation": "flat", "connects_to": ["walls_lower"]},
            },
            {
                "name": "walls_lower",
                "type": "Bricks",
                "parts": [
                    {"part_id": "3001", "name": "Brick 2x4", "category": "Bricks", "color": "Red", "color_hex": "#C91A09", "is_trans": False, "quantity": 8},
                    {"part_id": "3001", "name": "Brick 2x4", "category": "Bricks", "color": "White", "color_hex": "#FFFFFF", "is_trans": False, "quantity": 6},
                    {"part_id": "3010", "name": "Brick 1x4", "category": "Bricks", "color": "Red", "color_hex": "#C91A09", "is_trans": False, "quantity": 4},
                ],
                "spatial": {"position": "bottom", "orientation": "upright", "connects_to": ["walls_upper"]},
            },
            {
                "name": "walls_upper",
                "type": "Bricks",
                "parts": [
                    {"part_id": "3001", "name": "Brick 2x4", "category": "Bricks", "color": "Yellow", "color_hex": "#F2CD37", "is_trans": False, "quantity": 6},
                    {"part_id": "3001", "name": "Brick 2x4", "category": "Bricks", "color": "Red", "color_hex": "#C91A09", "is_trans": False, "quantity": 4},
                    {"part_id": "3622", "name": "Brick 1x3", "category": "Bricks", "color": "White", "color_hex": "#FFFFFF", "is_trans": False, "quantity": 4},
                ],
                "spatial": {"position": "center", "orientation": "upright", "connects_to": ["windows_and_doors", "roof"]},
            },
            {
                "name": "windows_and_doors",
                "type": "Windows and Doors",
                "parts": [
                    {"part_id": "60594", "name": "Window 1x2x3 Pane", "category": "Windows and Doors", "color": "Trans-Clear", "color_hex": "#FCFCFC", "is_trans": True, "quantity": 4},
                    {"part_id": "60593", "name": "Window 1x2x3 Frame", "category": "Windows and Doors", "color": "Blue", "color_hex": "#0055BF", "is_trans": False, "quantity": 4},
                    {"part_id": "60596", "name": "Door 1x4x6 Frame", "category": "Windows and Doors", "color": "Reddish Brown", "color_hex": "#582A12", "is_trans": False, "quantity": 1},
                    {"part_id": "60616", "name": "Door 1x4x6 Panel", "category": "Windows and Doors", "color": "Dark Azure", "color_hex": "#078BC9", "is_trans": False, "quantity": 1},
                ],
                "spatial": {"position": "center", "orientation": "upright", "connects_to": ["walls_upper"]},
            },
            {
                "name": "roof",
                "type": "Roof Tiles",
                "parts": [
                    {"part_id": "3037", "name": "Slope 45 2x4", "category": "Roof Tiles", "color": "Bright Orange", "color_hex": "#FE8A18", "is_trans": False, "quantity": 8},
                    {"part_id": "3038", "name": "Slope 45 2x3", "category": "Roof Tiles", "color": "Bright Orange", "color_hex": "#FE8A18", "is_trans": False, "quantity": 4},
                    {"part_id": "3048", "name": "Slope 45 1x2 Triple", "category": "Roof Tiles", "color": "Dark Red", "color_hex": "#720E0F", "is_trans": False, "quantity": 2},
                ],
                "spatial": {"position": "top", "orientation": "angled", "connects_to": ["walls_upper"]},
            },
        ],
        "build_hints": [],
    }


@pytest.fixture
def checker():
    return StabilityChecker()


# ── Legality checks ──────────────────────────────────────────────────

class TestPartExistence:
    def test_valid_parts_pass(self, checker):
        desc = _make_desc()
        result = checker.check_part_existence(desc)
        assert result.status in ("pass", "warn")  # warn if catalog not loaded

    def test_unknown_parts_flagged(self, checker):
        desc = _make_desc()
        desc["subassemblies"][0]["parts"][0]["part_id"] = "FAKE_999999"
        result = checker.check_part_existence(desc)
        # If catalog is loaded, should flag; if not, skip with warn
        assert result.status in ("warn", "fail")


class TestPartCompatibility:
    def test_matching_types_pass(self, checker):
        desc = _make_desc()
        result = checker.check_part_compatibility(desc)
        assert result.status == "pass"

    def test_mismatched_types_warn(self, checker):
        desc = _make_desc()
        # Put a "Technic" part in a "Plates" subassembly
        desc["subassemblies"][0]["parts"].append(
            {"part_id": "9999", "name": "Technic Pin", "category": "Technic", "color": "Black", "color_hex": "#000", "is_trans": False, "quantity": 1}
        )
        result = checker.check_part_compatibility(desc)
        assert result.status == "warn"


class TestColorValidity:
    def test_valid_colors_pass(self, checker):
        desc = _make_desc()
        result = checker.check_color_validity(desc)
        assert result.status in ("pass", "warn")  # warn if catalog not loaded

    def test_invalid_color_flagged(self, checker):
        desc = _make_desc()
        desc["subassemblies"][0]["parts"][0]["color"] = "Sparkle Rainbow Unicorn"
        result = checker.check_color_validity(desc)
        assert result.status == "warn"


class TestQuantityReasonableness:
    def test_normal_quantities_pass(self, checker):
        desc = _make_desc()
        result = checker.check_quantity_reasonableness(desc)
        assert result.status == "pass"

    def test_high_quantity_warns(self, checker):
        desc = _make_desc()
        desc["subassemblies"][0]["parts"][0]["quantity"] = 60
        result = checker.check_quantity_reasonableness(desc)
        assert result.status == "warn"

    def test_extreme_quantity_fails(self, checker):
        desc = _make_desc()
        desc["subassemblies"][0]["parts"][0]["quantity"] = 300
        result = checker.check_quantity_reasonableness(desc)
        assert result.status == "fail"


# ── Stability checks ─────────────────────────────────────────────────

class TestFoundation:
    def test_has_foundation_pass(self, checker):
        desc = _make_desc()
        result = checker.check_foundation(desc)
        assert result.status == "pass"

    def test_no_foundation_fail(self, checker):
        desc = _make_desc()
        for sa in desc["subassemblies"]:
            sa["spatial"]["position"] = "center"
        result = checker.check_foundation(desc)
        assert result.status == "fail"


class TestConnectivity:
    def test_connected_pass(self, checker):
        desc = _make_desc()
        result = checker.check_connectivity(desc)
        assert result.status == "pass"

    def test_disconnected_fail(self, checker):
        desc = _make_desc()
        # Add an island subassembly with no connections
        desc["subassemblies"].append({
            "name": "island",
            "type": "Bricks",
            "parts": [{"part_id": "3001", "name": "Brick 2x4", "category": "Bricks", "color": "Red", "color_hex": "#C91A09", "is_trans": False, "quantity": 1}],
            "spatial": {"position": "center", "orientation": "flat", "connects_to": []},
        })
        result = checker.check_connectivity(desc)
        assert result.status == "fail"
        assert "island" in result.message

    def test_single_subassembly_pass(self, checker):
        desc = _make_desc()
        desc["subassemblies"] = [desc["subassemblies"][0]]
        desc["subassemblies"][0]["spatial"]["connects_to"] = []
        result = checker.check_connectivity(desc)
        assert result.status == "pass"


class TestSupportRatio:
    def test_balanced_pass(self, checker):
        desc = _make_desc()
        result = checker.check_support_ratio(desc)
        assert result.status == "pass"

    def test_top_heavy_warns(self, checker):
        desc = _make_desc()
        # Make upper subassembly much heavier than lower
        desc["subassemblies"][1]["parts"][0]["quantity"] = 100
        desc["subassemblies"][0]["parts"][0]["quantity"] = 5
        desc["subassemblies"][1]["spatial"]["position"] = "top"
        result = checker.check_support_ratio(desc)
        assert result.status == "warn"


class TestBuildOrder:
    def test_valid_order_pass(self, checker):
        desc = _make_desc()
        result = checker.check_build_order(desc)
        assert result.status == "pass"

    def test_invalid_order_warns(self, checker):
        desc = _make_desc()
        # Reverse: put walls first, base second
        desc["subassemblies"] = list(reversed(desc["subassemblies"]))
        result = checker.check_build_order(desc)
        assert result.status == "warn"


class TestCenterOfMass:
    def test_balanced_pass(self, checker):
        desc = _make_desc()
        result = checker.check_center_of_mass(desc)
        assert result.status == "pass"

    def test_top_heavy_warns(self, checker):
        desc = _make_desc()
        desc["subassemblies"][0]["spatial"]["position"] = "bottom"
        desc["subassemblies"][0]["parts"][0]["quantity"] = 2
        desc["subassemblies"][1]["spatial"]["position"] = "top"
        desc["subassemblies"][1]["parts"][0]["quantity"] = 50
        result = checker.check_center_of_mass(desc)
        assert result.status == "warn"


class TestCantilever:
    def test_no_sides_pass(self, checker):
        desc = _make_desc()
        result = checker.check_cantilever(desc)
        assert result.status == "pass"

    def test_unsupported_side_fails(self, checker):
        desc = _make_desc()
        desc["subassemblies"].append({
            "name": "wing",
            "type": "Bricks",
            "parts": [{"part_id": "3001", "name": "Brick 2x4", "category": "Bricks", "color": "Red", "color_hex": "#C91A09", "is_trans": False, "quantity": 3}],
            "spatial": {"position": "left", "orientation": "flat", "connects_to": []},
        })
        result = checker.check_cantilever(desc)
        assert result.status == "fail"

    def test_supported_side_pass(self, checker):
        desc = _make_desc()
        desc["subassemblies"].append({
            "name": "wing",
            "type": "Bricks",
            "parts": [{"part_id": "3001", "name": "Brick 2x4", "category": "Bricks", "color": "Red", "color_hex": "#C91A09", "is_trans": False, "quantity": 3}],
            "spatial": {"position": "left", "orientation": "flat", "connects_to": ["base", "walls"]},
        })
        result = checker.check_cantilever(desc)
        assert result.status == "pass"


# ── Full validation ───────────────────────────────────────────────────

class TestValidate:
    def test_empty_description(self, checker):
        report = checker.validate({})
        assert report.score == 0
        assert len(report.checks) == 1
        assert report.checks[0].status == "fail"

    def test_valid_build_scores_high(self, checker):
        desc = _make_desc()
        report = checker.validate(desc)
        assert report.score >= 70
        assert "failure" in report.summary or "passed" in report.summary

    def test_mock_house_golden(self, checker):
        """The mock house should score well on all checks."""
        desc = _mock_house()
        report = checker.validate(desc)
        fails = [c for c in report.checks if c.status == "fail"]
        assert len(fails) == 0, f"Mock house has failures: {[f.name for f in fails]}"
        assert report.score >= 75

    def test_score_clamped(self, checker):
        """Score should never go below 0."""
        desc = _make_desc()
        # Make everything fail
        for sa in desc["subassemblies"]:
            sa["spatial"]["position"] = "top"
            sa["spatial"]["connects_to"] = []
            sa["parts"][0]["quantity"] = 300
            sa["parts"][0]["part_id"] = "FAKE"
            sa["parts"][0]["color"] = "FAKE"
        report = checker.validate(desc)
        assert report.score >= 0

    def test_report_serializable(self, checker):
        """ValidationReport should be serializable via asdict."""
        desc = _make_desc()
        report = checker.validate(desc)
        d = asdict(report)
        assert isinstance(d, dict)
        assert "score" in d
        assert "checks" in d
        assert "summary" in d
        assert all(isinstance(c, dict) for c in d["checks"])
