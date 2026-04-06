"""add_grid_pos.py — Preprocessing script that adds grid_pos: [x, z] to every
part entry in every ST2B label file.

grid_pos represents coarse stud coordinates so the 3D viewer can place bricks.
Within each layer the parts are packed left-to-right based on stud width.

Usage
-----
python -m backend.data_pipeline.add_grid_pos [--labels-dir PATH] [--dry-run]
"""

import copy
import json
import math
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

# Allow `python add_grid_pos.py` to find the project root packages.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import ST2B_CONVERTED_DIR

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

_DIM_RE = re.compile(r"(\d+)\s*x\s*(\d+)", re.IGNORECASE)

_DEFAULT_WIDTH = 2


def parse_brick_width(name: str) -> int:
    """Return the stud width (first dimension) parsed from a part name.

    Examples
    --------
    >>> parse_brick_width("Brick 2x2")
    2
    >>> parse_brick_width("Plate 1x4")
    1
    >>> parse_brick_width("Something weird")
    2
    """
    m = _DIM_RE.search(name)
    if m:
        return int(m.group(1))
    return _DEFAULT_WIDTH


def compute_grid_positions(layer: dict) -> dict:
    """Return a deep copy of *layer* with ``grid_pos`` added to every part.

    The packing algorithm:
    1. Compute ``total_studs = sum(width * quantity)`` across all parts.
    2. ``layer_width = max(4, ceil(sqrt(total_studs)))``
    3. Walk parts left-to-right; for each part:
       ``grid_pos = [cursor_x % layer_width, cursor_x // layer_width]``
       then advance ``cursor_x += width * quantity``.

    The input dict is **not** mutated.
    """
    result = copy.deepcopy(layer)
    parts = result.get("parts", [])

    if not parts:
        return result

    # --- compute layer_width ---
    total_studs = sum(parse_brick_width(p["name"]) * p["quantity"] for p in parts)
    layer_width = max(4, math.ceil(math.sqrt(total_studs)))

    # --- assign grid positions ---
    cursor_x = 0
    for part in parts:
        width = parse_brick_width(part["name"])
        part["grid_pos"] = [cursor_x % layer_width, cursor_x // layer_width]
        cursor_x += width * part["quantity"]

    return result


def add_grid_pos_to_label(label: dict) -> dict:
    """Return a deep copy of *label* with ``grid_pos`` added to all parts in
    every subassembly layer.
    """
    result = copy.deepcopy(label)
    for subassembly in result.get("subassemblies", []):
        updated = compute_grid_positions(subassembly)
        subassembly.update(updated)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = ArgumentParser(
        description="Add grid_pos: [x, z] to every part in every ST2B label file."
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=ST2B_CONVERTED_DIR,
        help="Directory containing ST2B .json label files (default: ST2B_CONVERTED_DIR from config).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and compute grid positions but do not write any files.",
    )
    args = parser.parse_args()

    labels_dir: Path = args.labels_dir
    if not labels_dir.is_dir():
        print(f"ERROR: labels directory not found: {labels_dir}", file=sys.stderr)
        sys.exit(1)

    json_files = sorted(labels_dir.glob("*.json"))
    if not json_files:
        print(f"No .json files found in {labels_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(json_files)} label files in {labels_dir} ...")
    if args.dry_run:
        print("(dry-run mode — files will not be modified)")

    errors = 0
    for idx, path in enumerate(json_files, start=1):
        try:
            with path.open("r", encoding="utf-8") as fh:
                label = json.load(fh)

            updated = add_grid_pos_to_label(label)

            if not args.dry_run:
                with path.open("w", encoding="utf-8") as fh:
                    json.dump(updated, fh, indent=2, ensure_ascii=False)

            if idx % 5000 == 0 or idx == len(json_files):
                print(f"  [{idx}/{len(json_files)}] done")

        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR processing {path.name}: {exc}", file=sys.stderr)
            errors += 1

    status = "dry-run complete" if args.dry_run else "complete"
    print(f"Finished ({status}). Errors: {errors}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
