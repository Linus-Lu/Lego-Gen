"""Transforms raw Rebrickable API data into structured JSON labels for training."""

import json
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from backend.data_pipeline.part_library import PartLibrary


# ── Spatial heuristics ─────────────────────────────────────────────────
# Map part category names to likely spatial positions in a build.
CATEGORY_SPATIAL_MAP = {
    "Baseplates": {"position": "bottom", "orientation": "flat"},
    "Plates": {"position": "bottom", "orientation": "flat"},
    "Bricks": {"position": "center", "orientation": "upright"},
    "Bricks Sloped": {"position": "top", "orientation": "angled"},
    "Slopes": {"position": "top", "orientation": "angled"},
    "Roof Tiles": {"position": "top", "orientation": "angled"},
    "Tiles": {"position": "center", "orientation": "flat"},
    "Windows and Doors": {"position": "center", "orientation": "upright"},
    "Wheels and Tyres": {"position": "bottom", "orientation": "upright"},
    "Technic Beams": {"position": "center", "orientation": "upright"},
    "Technic Pins": {"position": "center", "orientation": "upright"},
    "Minifig Accessories": {"position": "top", "orientation": "upright"},
    "Minifig Headwear": {"position": "top", "orientation": "upright"},
    "Minifig Upper Body": {"position": "center", "orientation": "upright"},
    "Minifig Lower Body": {"position": "bottom", "orientation": "upright"},
    "Plants and Animals": {"position": "center", "orientation": "upright"},
    "Flags, Signs, Plastics": {"position": "top", "orientation": "upright"},
}

DEFAULT_SPATIAL = {"position": "center", "orientation": "upright"}


def classify_complexity(num_parts: int) -> str:
    if num_parts < 50:
        return "simple"
    elif num_parts < 200:
        return "intermediate"
    elif num_parts < 500:
        return "advanced"
    return "expert"


def estimate_dimensions(num_parts: int) -> dict:
    """Coarse dimension estimate based on total part count."""
    if num_parts < 50:
        return {"width": "small", "height": "small", "depth": "small"}
    elif num_parts < 150:
        return {"width": "medium", "height": "small", "depth": "small"}
    elif num_parts < 300:
        return {"width": "medium", "height": "medium", "depth": "medium"}
    return {"width": "large", "height": "large", "depth": "medium"}


def extract_dominant_colors(
    inventory: list[dict], part_library: PartLibrary, top_n: int = 3
) -> list[str]:
    """Return the top-N color names by total quantity in the inventory."""
    color_counts: Counter = Counter()
    for item in inventory:
        color_id = item["color"]["id"]
        qty = item.get("quantity", 1)
        color_name = part_library.get_color_name(color_id)
        if color_name != "Unknown":
            color_counts[color_name] += qty
    return [name for name, _ in color_counts.most_common(top_n)]


def group_into_subassemblies(
    inventory: list[dict], part_library: PartLibrary
) -> list[dict]:
    """Group inventory parts by their category into subassemblies."""
    groups: dict[str, list[dict]] = {}

    for item in inventory:
        part = item.get("part", {})
        cat_id = part.get("part_cat_id", 0)
        cat_name = part_library.get_category_name(cat_id)

        color_id = item["color"]["id"]
        color_info = part_library.colors.get(color_id, {})
        part_entry = {
            "part_id": part.get("part_num", ""),
            "name": part.get("name", "Unknown Part"),
            "category": cat_name,
            "color": part_library.get_color_name(color_id),
            "color_hex": f"#{part_library.get_color_hex(color_id)}",
            "is_trans": color_info.get("is_trans", False),
            "quantity": item.get("quantity", 1),
        }

        if cat_name not in groups:
            groups[cat_name] = []
        groups[cat_name].append(part_entry)

    # Build subassembly list
    subassemblies = []
    group_names = sorted(groups.keys())
    for cat_name in group_names:
        spatial = CATEGORY_SPATIAL_MAP.get(cat_name, DEFAULT_SPATIAL)
        safe_name = cat_name.lower().replace(" ", "_").replace(",", "")
        subassemblies.append(
            {
                "name": safe_name,
                "type": cat_name,
                "parts": groups[cat_name],
                "spatial": {
                    **spatial,
                    "connects_to": [],  # filled heuristically below
                },
            }
        )

    # Simple heuristic: bottom connects to center, center connects to top
    position_map = {}
    for sa in subassemblies:
        pos = sa["spatial"]["position"]
        if pos not in position_map:
            position_map[pos] = []
        position_map[pos].append(sa["name"])

    for sa in subassemblies:
        pos = sa["spatial"]["position"]
        if pos == "bottom":
            sa["spatial"]["connects_to"] = position_map.get("center", [])
        elif pos == "center":
            sa["spatial"]["connects_to"] = position_map.get("top", [])
        elif pos == "top":
            sa["spatial"]["connects_to"] = position_map.get("center", [])

    return subassemblies


def build_json_label(
    set_info: dict,
    inventory: list[dict],
    theme_chain: list[str],
    part_library: PartLibrary,
) -> dict:
    """Build the complete JSON label for a LEGO set.

    Args:
        set_info: Set metadata from Rebrickable API.
        inventory: Part inventory from Rebrickable API.
        theme_chain: Theme hierarchy [root, ..., leaf].
        part_library: Loaded PartLibrary instance.

    Returns:
        Structured JSON dict matching our training schema.
    """
    num_parts = set_info.get("num_parts", len(inventory))
    category = theme_chain[0] if theme_chain else "Unknown"
    subcategory = theme_chain[-1] if len(theme_chain) > 1 else category

    label = {
        "set_id": set_info["set_num"],
        "object": set_info["name"],
        "category": category,
        "subcategory": subcategory,
        "complexity": classify_complexity(num_parts),
        "total_parts": num_parts,
        "dominant_colors": extract_dominant_colors(inventory, part_library),
        "dimensions_estimate": estimate_dimensions(num_parts),
        "subassemblies": group_into_subassemblies(inventory, part_library),
        "build_hints": _generate_build_hints(num_parts, theme_chain),
    }
    return label


def _generate_build_hints(num_parts: int, theme_chain: list[str]) -> list[str]:
    """Generate basic build hints based on set properties."""
    hints = ["Start with the base plate or foundation pieces"]

    if num_parts > 100:
        hints.append("Sort pieces by color before building")

    category = theme_chain[0] if theme_chain else ""
    if category in ("City", "Creator", "Architecture"):
        hints.append("Build walls before attaching the roof")
    elif category in ("Technic", "Mindstorms"):
        hints.append("Assemble the gear train and axles first")
    elif category in ("Star Wars", "Space"):
        hints.append("Build the main hull before adding details")

    if num_parts > 200:
        hints.append("Work in sections — complete each subassembly before joining")

    return hints


def validate_label(label: dict) -> tuple[bool, list[str]]:
    """Validate a JSON label against the expected schema.

    Returns (is_valid, list_of_errors).
    """
    errors = []
    required_fields = [
        "set_id", "object", "category", "subcategory", "complexity",
        "total_parts", "dominant_colors", "dimensions_estimate",
        "subassemblies", "build_hints",
    ]
    for field in required_fields:
        if field not in label:
            errors.append(f"Missing required field: {field}")

    if "complexity" in label and label["complexity"] not in (
        "simple", "intermediate", "advanced", "expert"
    ):
        errors.append(f"Invalid complexity: {label['complexity']}")

    if "dominant_colors" in label and not isinstance(label["dominant_colors"], list):
        errors.append("dominant_colors must be a list")

    if "subassemblies" in label:
        for i, sa in enumerate(label["subassemblies"]):
            for key in ("name", "type", "parts", "spatial"):
                if key not in sa:
                    errors.append(f"subassemblies[{i}] missing '{key}'")

    return len(errors) == 0, errors


def save_label(label: dict, set_num: str, output_dir: Path) -> Path:
    """Save a JSON label to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{set_num}.json"
    with open(path, "w") as f:
        json.dump(label, f, indent=2)
    return path
