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
_DEFAULT_DEPTH = 2


def parse_brick_dims(name: str) -> tuple[int, int]:
    """Return ``(width, depth)`` parsed from a part name.

    Examples
    --------
    >>> parse_brick_dims("Brick 2x4")
    (2, 4)
    >>> parse_brick_dims("Plate 1x6")
    (1, 6)
    >>> parse_brick_dims("Something weird")
    (2, 2)
    """
    m = _DIM_RE.search(name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return _DEFAULT_WIDTH, _DEFAULT_DEPTH


def parse_brick_width(name: str) -> int:
    """Return the stud width (first dimension) parsed from a part name."""
    return parse_brick_dims(name)[0]


def compute_grid_positions(layer: dict) -> dict:
    """Return a deep copy of *layer* with ``grid_positions`` added to every part.

    Each part receives ``grid_positions``: a list of ``[x, z]`` **stud
    coordinates**, one entry per quantity.  ``grid_pos`` is kept as a shortcut
    pointing at the first instance for backward compatibility.

    The packing algorithm:
    1. Expand every part instance (respecting *quantity*).
    2. Compute ``layer_width`` from total stud area for a roughly square footprint.
    3. Lay instances left-to-right; start a new row when the next brick would
       exceed ``layer_width``.  Row depth equals the deepest brick in that row.

    The input dict is **not** mutated.
    """
    result = copy.deepcopy(layer)
    parts = result.get("parts", [])

    if not parts:
        return result

    # Expand all brick instances: (part_index, width, depth)
    instances: list[tuple[int, int, int]] = []
    for pi, part in enumerate(parts):
        w, d = parse_brick_dims(part["name"])
        for _ in range(part.get("quantity", 1)):
            instances.append((pi, w, d))

    if not instances:
        return result

    # Target a roughly square footprint
    total_area = sum(w * d for _, w, d in instances)
    layer_width = max(4, math.ceil(math.sqrt(total_area)))

    # Row-based packing with depth tracking
    cursor_x = 0
    cursor_z = 0
    row_max_depth = 0
    per_part: dict[int, list[list[int]]] = {}

    for pi, w, d in instances:
        if cursor_x + w > layer_width and cursor_x > 0:
            cursor_z += row_max_depth
            cursor_x = 0
            row_max_depth = 0

        per_part.setdefault(pi, []).append([cursor_x, cursor_z])
        cursor_x += w
        row_max_depth = max(row_max_depth, d)

    for pi, part in enumerate(parts):
        positions = per_part.get(pi, [])
        part["grid_positions"] = positions
        part["grid_pos"] = positions[0] if positions else [0, 0]

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
                # Atomic write: write to temp file then rename, so a crash
                # mid-write doesn't leave the original file truncated/empty.
                tmp_path = path.with_suffix(".json.tmp")
                with tmp_path.open("w", encoding="utf-8") as fh:
                    json.dump(updated, fh, indent=2, ensure_ascii=False)
                tmp_path.replace(path)

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
