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

# Canonical position ordering (lower = earlier in build)
POSITION_ORDER = {"bottom": 0, "center": 1, "front": 2, "back": 2, "left": 2, "right": 2, "top": 3}


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
    """Attempt to repair malformed or truncated JSON string.

    Handles output truncated by token limits (common with long LEGO JSONs)
    as well as minor formatting issues like trailing commas.
    """
    import re

    # Try json_repair library first (best results)
    try:
        import json_repair
        return json_repair.repair_json(raw)
    except ImportError:
        pass

    # Manual repair for truncated / malformed JSON
    repaired = raw.strip()

    # Fix trailing commas before closing brackets
    repaired = re.sub(r',\s*]', ']', repaired)
    repaired = re.sub(r',\s*}', '}', repaired)

    # Strip trailing comma at end of string (truncation artifact)
    repaired = re.sub(r',\s*$', '', repaired)

    # Close any open string literal
    if repaired.count('"') % 2 == 1:
        repaired += '"'

    # Close unclosed brackets then braces (order matters for valid JSON)
    open_brackets = repaired.count("[") - repaired.count("]")
    repaired += "]" * max(0, open_brackets)
    open_braces = repaired.count("{") - repaired.count("}")
    repaired += "}" * max(0, open_braces)

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


def validate_part_ids(data: dict, known_parts: set[str]) -> list[str]:
    """Flag part IDs not found in the known catalog."""
    warnings = []
    for i, sa in enumerate(data.get("subassemblies", [])):
        for j, part in enumerate(sa.get("parts", [])):
            pid = part.get("part_id", "")
            if pid and pid not in known_parts:
                warnings.append(f"subassemblies[{i}].parts[{j}]: unknown part_id '{pid}'")
    return warnings


def repair_connects_to(data: dict) -> dict:
    """Ensure connects_to references match actual subassembly names."""
    valid_names = {sa.get("name", "") for sa in data.get("subassemblies", [])}
    for sa in data.get("subassemblies", []):
        spatial = sa.get("spatial", {})
        connects = spatial.get("connects_to", [])
        spatial["connects_to"] = [c for c in connects if c in valid_names]
        sa["spatial"] = spatial
    return data


def validate_structural_order(data: dict) -> list[str]:
    """Check that subassemblies follow a bottom-to-top ordering."""
    warnings = []
    subs = data.get("subassemblies", [])
    orders = [POSITION_ORDER.get(sa.get("spatial", {}).get("position", "center"), 1) for sa in subs]
    if orders != sorted(orders):
        warnings.append("Subassemblies are not ordered bottom-to-top")
    return warnings


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

    # Step 5: Repair connects_to references
    data = repair_connects_to(data)

    return data, errors
