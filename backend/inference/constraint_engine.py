"""JSON output validation and repair for LEGO descriptions."""

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ── Schema definition ──────────────────────────────────────────────────

REQUIRED_FIELDS = {
    "set_id": str,
    "object": str,
    "category": str,
    "subcategory": str,
    "complexity": str,
    "total_parts": int,
    "dominant_colors": list,
    "dimensions_estimate": dict,
    "subassemblies": list,
    "build_hints": list,
}

VALID_COMPLEXITY = {"simple", "intermediate", "advanced", "expert"}
VALID_POSITIONS = {"top", "bottom", "left", "right", "center", "front", "back"}
VALID_ORIENTATIONS = {"flat", "upright", "angled"}
VALID_DIMENSIONS = {"small", "medium", "large"}


def validate_lego_json(data: dict) -> tuple[bool, list[str]]:
    """Validate a JSON description against the LEGO schema.

    Returns (is_valid, list_of_error_messages).
    """
    errors = []

    # Check required fields and types
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in data:
            errors.append(f"Missing field: {field}")
        elif not isinstance(data[field], expected_type):
            errors.append(f"Field '{field}' should be {expected_type.__name__}, got {type(data[field]).__name__}")

    # Validate complexity
    if data.get("complexity") and data["complexity"] not in VALID_COMPLEXITY:
        errors.append(f"Invalid complexity: '{data['complexity']}'")

    # Validate dimensions_estimate
    dims = data.get("dimensions_estimate", {})
    for key in ("width", "height", "depth"):
        val = dims.get(key)
        if val and val not in VALID_DIMENSIONS:
            errors.append(f"Invalid dimension '{key}': '{val}'")

    # Validate subassemblies
    for i, sa in enumerate(data.get("subassemblies", [])):
        if not isinstance(sa, dict):
            errors.append(f"subassemblies[{i}] is not a dict")
            continue
        for key in ("name", "type", "parts", "spatial"):
            if key not in sa:
                errors.append(f"subassemblies[{i}] missing '{key}'")
        spatial = sa.get("spatial", {})
        if spatial.get("position") and spatial["position"] not in VALID_POSITIONS:
            errors.append(f"subassemblies[{i}] invalid position: '{spatial['position']}'")

    return len(errors) == 0, errors


def repair_json_string(raw: str) -> str:
    """Attempt to repair malformed JSON string."""
    try:
        import json_repair
        return json_repair.repair_json(raw)
    except ImportError:
        # Manual basic repairs
        repaired = raw.strip()
        # Fix trailing commas
        repaired = repaired.replace(",]", "]").replace(",}", "}")
        # Ensure closing braces
        open_braces = repaired.count("{") - repaired.count("}")
        repaired += "}" * max(0, open_braces)
        open_brackets = repaired.count("[") - repaired.count("]")
        repaired += "]" * max(0, open_brackets)
        return repaired


def enforce_valid_values(data: dict) -> dict:
    """Clamp values to valid ranges and fix invalid enum values."""
    # Fix complexity
    if data.get("complexity") not in VALID_COMPLEXITY:
        total = data.get("total_parts", 0)
        if total < 50:
            data["complexity"] = "simple"
        elif total < 200:
            data["complexity"] = "intermediate"
        elif total < 500:
            data["complexity"] = "advanced"
        else:
            data["complexity"] = "expert"

    # Fix dimensions
    dims = data.get("dimensions_estimate", {})
    for key in ("width", "height", "depth"):
        if dims.get(key) not in VALID_DIMENSIONS:
            dims[key] = "medium"
    data["dimensions_estimate"] = dims

    # Fix spatial positions in subassemblies
    for sa in data.get("subassemblies", []):
        spatial = sa.get("spatial", {})
        if spatial.get("position") not in VALID_POSITIONS:
            spatial["position"] = "center"
        if spatial.get("orientation") not in VALID_ORIENTATIONS:
            spatial["orientation"] = "upright"
        sa["spatial"] = spatial

    # Ensure total_parts is positive
    if not isinstance(data.get("total_parts"), int) or data["total_parts"] < 0:
        data["total_parts"] = 0

    return data


def safe_parse_and_validate(raw: str) -> tuple[dict | None, list[str]]:
    """Full pipeline: repair -> parse -> validate -> enforce valid values.

    Returns (parsed_dict_or_None, errors).
    """
    # Step 1: Try direct parse
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Step 2: Try repair
        repaired = repair_json_string(raw)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            return None, ["Failed to parse JSON even after repair"]

    # Step 3: Validate
    is_valid, errors = validate_lego_json(data)

    # Step 4: Enforce valid values (fix what we can)
    data = enforce_valid_values(data)

    return data, errors
