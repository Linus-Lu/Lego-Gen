"""Convert structured JSON descriptions into frontend-ready build steps."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.inference.constraint_engine import POSITION_ORDER

# ── Color name → hex fallback (common LEGO colors) ───────────────────
# Used to fill in missing color_hex from model output
_COLOR_NAME_TO_HEX: dict[str, str] = {
    "black": "#05131D", "blue": "#0055BF", "bright green": "#4B9F4A",
    "bright light blue": "#9FC3E9", "bright light orange": "#F8BB3D",
    "bright light yellow": "#FFF03A", "bright orange": "#FE8A18",
    "bright pink": "#E4ADC8", "coral": "#FF698F", "dark azure": "#078BC9",
    "dark blue": "#143044", "dark bluish gray": "#6C6E68",
    "dark brown": "#352100", "dark green": "#184632",
    "dark orange": "#A95500", "dark pink": "#C870A0",
    "dark red": "#720E0F", "dark tan": "#958A73",
    "dark turquoise": "#008F9B", "green": "#237841", "lavender": "#E1D5ED",
    "light aqua": "#ADC3C0", "light bluish gray": "#A0A5A9",
    "light gray": "#9BA19D", "light nougat": "#FCC39E",
    "lime": "#BBE90B", "magenta": "#923978", "medium azure": "#36AEBF",
    "medium blue": "#5A93DB", "medium lavender": "#AC78BA",
    "medium nougat": "#AA7D55", "nougat": "#D09168",
    "olive green": "#9B9A5A", "orange": "#FE8A18",
    "pearl gold": "#AA7F2E", "red": "#C91A09", "reddish brown": "#582A12",
    "sand blue": "#596072", "sand green": "#A0BCAC",
    "tan": "#E4CD9E", "teal": "#008F9B",
    "trans-clear": "#FCFCFC", "trans-light blue": "#C1DFF0",
    "trans-red": "#C91A09", "trans-green": "#84B68D",
    "trans-orange": "#F08F1C", "trans-yellow": "#F5CD2F",
    "white": "#FFFFFF", "yellow": "#F2CD37",
}


def _enrich_parts(parts: list[dict]) -> list[dict]:
    """Fill in missing color_hex and is_trans fields on part dicts."""
    for part in parts:
        # Ensure color_hex is present
        if not part.get("color_hex"):
            name = (part.get("color") or "").lower().strip()
            part["color_hex"] = _COLOR_NAME_TO_HEX.get(name, "#A0A5A9")
        # Ensure is_trans is present
        if "is_trans" not in part:
            part["is_trans"] = "trans" in (part.get("color") or "").lower()
        # Ensure quantity is at least 1
        if not part.get("quantity") or part["quantity"] < 1:
            part["quantity"] = 1
    return parts


def json_to_steps(description: dict) -> list[dict]:
    """Convert a LEGO JSON description into ordered build steps.

    Each step contains:
        - step_number: int
        - title: str
        - instruction: str
        - parts: list of part dicts
        - part_count: total parts in this step
    """
    if not description:
        return []

    subassemblies = description.get("subassemblies", [])
    if not subassemblies:
        return _single_step_from_description(description)

    # Sort subassemblies by spatial position (bottom-up)
    sorted_subs = sorted(
        subassemblies,
        key=lambda sa: POSITION_ORDER.get(
            sa.get("spatial", {}).get("position", "center"), 1
        ),
    )

    steps = []
    for i, sa in enumerate(sorted_subs, start=1):
        parts = _enrich_parts(sa.get("parts", []))
        part_count = sum(p.get("quantity", 1) for p in parts)

        step = {
            "step_number": i,
            "title": _format_title(sa),
            "instruction": _format_instruction(sa, description),
            "parts": parts,
            "part_count": part_count,
        }
        steps.append(step)

    return steps


def _single_step_from_description(description: dict) -> list[dict]:
    """Create a single build step when there are no subassemblies."""
    return [
        {
            "step_number": 1,
            "title": f"Build the {description.get('object', 'model')}",
            "instruction": f"Assemble all {description.get('total_parts', 0)} parts.",
            "parts": [],
            "part_count": description.get("total_parts", 0),
        }
    ]


def _format_title(subassembly: dict) -> str:
    """Generate a human-readable title for a build step."""
    name = subassembly.get("name", "section").replace("_", " ").title()
    return f"Build the {name}"


def _format_instruction(subassembly: dict, description: dict) -> str:
    """Generate a human-readable instruction for a build step."""
    parts = subassembly.get("parts", [])
    part_count = sum(p.get("quantity", 1) for p in parts)
    sa_type = subassembly.get("type", "section")
    position = subassembly.get("spatial", {}).get("position", "")

    # Describe what connects where
    connects_to = subassembly.get("spatial", {}).get("connects_to", [])
    connection_hint = ""
    if connects_to:
        targets = ", ".join(c.replace("_", " ") for c in connects_to[:2])
        connection_hint = f" This connects to the {targets}."

    position_hint = f" at the {position}" if position else ""
    return (
        f"Assemble {part_count} {sa_type} pieces{position_hint}.{connection_hint}"
    )


def steps_to_summary(steps: list[dict]) -> str:
    """Convert steps to a plain-text summary."""
    lines = []
    for step in steps:
        lines.append(f"Step {step['step_number']}: {step['title']}")
        lines.append(f"  {step['instruction']}")
        lines.append(f"  Parts needed: {step['part_count']}")
        lines.append("")
    return "\n".join(lines)
