"""Convert structured JSON descriptions into frontend-ready build steps."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ── Build order priority (lower = earlier in build) ────────────────────
POSITION_ORDER = {"bottom": 0, "center": 1, "front": 2, "back": 2, "left": 2, "right": 2, "top": 3}


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
        parts = sa.get("parts", [])
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
