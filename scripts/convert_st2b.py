#!/usr/bin/env python3
"""Convert StableText2Brick dataset to LegoDescription JSON format.

Downloads AvaLovelace/StableText2Brick from HuggingFace and converts each
example's brick placements + captions into our frontend-compatible JSON schema.

Usage:
    python scripts/convert_st2b.py
    python scripts/convert_st2b.py --max-examples 100  # quick test
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from random import Random

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import ST2B_CONVERTED_DIR, ST2B_PROMPTS_DIR, ST2B_CACHE_DIR

# ── Brick size → real LEGO part ID mapping ────────────────────────────

BRICK_SIZE_TO_PART = {
    "1x1": {"part_id": "3005", "name": "Brick 1x1", "category": "Bricks"},
    "1x2": {"part_id": "3004", "name": "Brick 1x2", "category": "Bricks"},
    "1x3": {"part_id": "3622", "name": "Brick 1x3", "category": "Bricks"},
    "1x4": {"part_id": "3010", "name": "Brick 1x4", "category": "Bricks"},
    "1x6": {"part_id": "3009", "name": "Brick 1x6", "category": "Bricks"},
    "1x8": {"part_id": "3008", "name": "Brick 1x8", "category": "Bricks"},
    "1x10": {"part_id": "6111", "name": "Brick 1x10", "category": "Bricks"},
    "2x1": {"part_id": "3004", "name": "Brick 1x2", "category": "Bricks"},  # treat as 1x2
    "2x2": {"part_id": "3003", "name": "Brick 2x2", "category": "Bricks"},
    "2x3": {"part_id": "3002", "name": "Brick 2x3", "category": "Bricks"},
    "2x4": {"part_id": "3001", "name": "Brick 2x4", "category": "Bricks"},
    "2x6": {"part_id": "2456", "name": "Brick 2x6", "category": "Bricks"},
    "2x8": {"part_id": "3007", "name": "Brick 2x8", "category": "Bricks"},
    "2x10": {"part_id": "3006", "name": "Brick 2x10", "category": "Bricks"},
    "3x1": {"part_id": "3622", "name": "Brick 1x3", "category": "Bricks"},
    "3x2": {"part_id": "3002", "name": "Brick 2x3", "category": "Bricks"},
    "4x1": {"part_id": "3010", "name": "Brick 1x4", "category": "Bricks"},
    "4x2": {"part_id": "3001", "name": "Brick 2x4", "category": "Bricks"},
    "4x4": {"part_id": "2356", "name": "Brick 4x4", "category": "Bricks"},
    "6x1": {"part_id": "3009", "name": "Brick 1x6", "category": "Bricks"},
    "6x2": {"part_id": "2456", "name": "Brick 2x6", "category": "Bricks"},
    "8x1": {"part_id": "3008", "name": "Brick 1x8", "category": "Bricks"},
    "8x2": {"part_id": "3007", "name": "Brick 2x8", "category": "Bricks"},
}

# Fallback: map unknown sizes to nearest known part
def _get_part_for_size(size_str: str) -> dict:
    if size_str in BRICK_SIZE_TO_PART:
        return BRICK_SIZE_TO_PART[size_str]
    # Try reversed (e.g. "3x1" -> "1x3")
    parts = size_str.split("x")
    if len(parts) == 2:
        rev = f"{parts[1]}x{parts[0]}"
        if rev in BRICK_SIZE_TO_PART:
            return BRICK_SIZE_TO_PART[rev]
    # Default to generic brick
    return {"part_id": "3005", "name": f"Brick {size_str}", "category": "Bricks"}


# ── ShapeNet category → LEGO category/subcategory + name ─────────────

SHAPENET_CATEGORIES = {
    "02801938": {"name": "basket", "category": "Creator", "subcategory": "Household"},
    "02818832": {"name": "bed", "category": "Creator", "subcategory": "Furniture"},
    "02828884": {"name": "bench", "category": "City", "subcategory": "Furniture"},
    "02843684": {"name": "birdhouse", "category": "Creator", "subcategory": "Animals"},
    "02871439": {"name": "bookshelf", "category": "Creator", "subcategory": "Furniture"},
    "02876657": {"name": "bottle", "category": "Creator", "subcategory": "Household"},
    "02880940": {"name": "bowl", "category": "Creator", "subcategory": "Household"},
    "02924116": {"name": "bus", "category": "City", "subcategory": "Vehicles"},
    "02942699": {"name": "camera", "category": "Creator", "subcategory": "Technology"},
    "02958343": {"name": "car", "category": "City", "subcategory": "Vehicles"},
    "03001627": {"name": "chair", "category": "Creator", "subcategory": "Furniture"},
    "03467517": {"name": "guitar", "category": "Creator", "subcategory": "Music"},
    "03593526": {"name": "jar", "category": "Creator", "subcategory": "Household"},
    "03797390": {"name": "mug", "category": "Creator", "subcategory": "Household"},
    "03928116": {"name": "piano", "category": "Creator", "subcategory": "Music"},
    "03991062": {"name": "pot", "category": "Creator", "subcategory": "Household"},
    "04256520": {"name": "sofa", "category": "Creator", "subcategory": "Furniture"},
    "04379243": {"name": "table", "category": "Creator", "subcategory": "Furniture"},
    "04460130": {"name": "tower", "category": "Creator", "subcategory": "Architecture"},
    "04468005": {"name": "train", "category": "City", "subcategory": "Vehicles"},
    "04530566": {"name": "vessel", "category": "Creator", "subcategory": "Vehicles"},
}

# ── Semantic color palettes per subcategory ───────────────────────────
# Each scheme: [base_color, body_color, accent_color]

COLOR_PALETTES = {
    "Vehicles": {
        "schemes": [
            [{"color": "Black", "hex": "#05131D", "is_trans": False},
             {"color": "Red", "hex": "#C91A09", "is_trans": False},
             {"color": "Light Bluish Gray", "hex": "#A0A5A9", "is_trans": False}],
            [{"color": "Dark Bluish Gray", "hex": "#6C6E68", "is_trans": False},
             {"color": "Blue", "hex": "#0055BF", "is_trans": False},
             {"color": "White", "hex": "#FFFFFF", "is_trans": False}],
            [{"color": "Black", "hex": "#05131D", "is_trans": False},
             {"color": "Yellow", "hex": "#F2CD37", "is_trans": False},
             {"color": "Red", "hex": "#C91A09", "is_trans": False}],
            [{"color": "Dark Bluish Gray", "hex": "#6C6E68", "is_trans": False},
             {"color": "White", "hex": "#FFFFFF", "is_trans": False},
             {"color": "Blue", "hex": "#0055BF", "is_trans": False}],
            [{"color": "Black", "hex": "#05131D", "is_trans": False},
             {"color": "Bright Green", "hex": "#4B9F4A", "is_trans": False},
             {"color": "White", "hex": "#FFFFFF", "is_trans": False}],
        ],
        "window_color": {"color": "Trans-Clear", "hex": "#FCFCFC", "is_trans": True},
    },
    "Furniture": {
        "schemes": [
            [{"color": "Dark Brown", "hex": "#352100", "is_trans": False},
             {"color": "Reddish Brown", "hex": "#582A12", "is_trans": False},
             {"color": "Tan", "hex": "#E4CD9E", "is_trans": False}],
            [{"color": "Black", "hex": "#05131D", "is_trans": False},
             {"color": "Dark Tan", "hex": "#958A73", "is_trans": False},
             {"color": "White", "hex": "#FFFFFF", "is_trans": False}],
            [{"color": "Reddish Brown", "hex": "#582A12", "is_trans": False},
             {"color": "Tan", "hex": "#E4CD9E", "is_trans": False},
             {"color": "Dark Tan", "hex": "#958A73", "is_trans": False}],
            [{"color": "Dark Brown", "hex": "#352100", "is_trans": False},
             {"color": "Nougat", "hex": "#D09168", "is_trans": False},
             {"color": "Tan", "hex": "#E4CD9E", "is_trans": False}],
        ],
    },
    "Household": {
        "schemes": [
            [{"color": "White", "hex": "#FFFFFF", "is_trans": False},
             {"color": "Red", "hex": "#C91A09", "is_trans": False},
             {"color": "White", "hex": "#FFFFFF", "is_trans": False}],
            [{"color": "Light Bluish Gray", "hex": "#A0A5A9", "is_trans": False},
             {"color": "Blue", "hex": "#0055BF", "is_trans": False},
             {"color": "White", "hex": "#FFFFFF", "is_trans": False}],
            [{"color": "Tan", "hex": "#E4CD9E", "is_trans": False},
             {"color": "Dark Green", "hex": "#184632", "is_trans": False},
             {"color": "Bright Green", "hex": "#4B9F4A", "is_trans": False}],
            [{"color": "White", "hex": "#FFFFFF", "is_trans": False},
             {"color": "Medium Azure", "hex": "#36AEBF", "is_trans": False},
             {"color": "Light Bluish Gray", "hex": "#A0A5A9", "is_trans": False}],
        ],
    },
    "Architecture": {
        "schemes": [
            [{"color": "Dark Bluish Gray", "hex": "#6C6E68", "is_trans": False},
             {"color": "Light Bluish Gray", "hex": "#A0A5A9", "is_trans": False},
             {"color": "Tan", "hex": "#E4CD9E", "is_trans": False}],
            [{"color": "Reddish Brown", "hex": "#582A12", "is_trans": False},
             {"color": "Dark Tan", "hex": "#958A73", "is_trans": False},
             {"color": "Tan", "hex": "#E4CD9E", "is_trans": False}],
            [{"color": "Dark Bluish Gray", "hex": "#6C6E68", "is_trans": False},
             {"color": "White", "hex": "#FFFFFF", "is_trans": False},
             {"color": "Light Bluish Gray", "hex": "#A0A5A9", "is_trans": False}],
        ],
    },
    "Music": {
        "schemes": [
            [{"color": "Black", "hex": "#05131D", "is_trans": False},
             {"color": "Reddish Brown", "hex": "#582A12", "is_trans": False},
             {"color": "Tan", "hex": "#E4CD9E", "is_trans": False}],
            [{"color": "Black", "hex": "#05131D", "is_trans": False},
             {"color": "Dark Red", "hex": "#720E0F", "is_trans": False},
             {"color": "Tan", "hex": "#E4CD9E", "is_trans": False}],
        ],
    },
    "Technology": {
        "schemes": [
            [{"color": "Black", "hex": "#05131D", "is_trans": False},
             {"color": "Dark Bluish Gray", "hex": "#6C6E68", "is_trans": False},
             {"color": "Light Bluish Gray", "hex": "#A0A5A9", "is_trans": False}],
            [{"color": "Black", "hex": "#05131D", "is_trans": False},
             {"color": "White", "hex": "#FFFFFF", "is_trans": False},
             {"color": "Red", "hex": "#C91A09", "is_trans": False}],
        ],
    },
    "Animals": {
        "schemes": [
            [{"color": "Dark Brown", "hex": "#352100", "is_trans": False},
             {"color": "Reddish Brown", "hex": "#582A12", "is_trans": False},
             {"color": "Bright Green", "hex": "#4B9F4A", "is_trans": False}],
            [{"color": "Tan", "hex": "#E4CD9E", "is_trans": False},
             {"color": "Medium Nougat", "hex": "#AA7D55", "is_trans": False},
             {"color": "White", "hex": "#FFFFFF", "is_trans": False}],
        ],
    },
}

# Fallback palette for unknown subcategories
DEFAULT_PALETTE = {
    "schemes": [
        [{"color": "Light Bluish Gray", "hex": "#A0A5A9", "is_trans": False},
         {"color": "Red", "hex": "#C91A09", "is_trans": False},
         {"color": "White", "hex": "#FFFFFF", "is_trans": False}],
    ],
}


# ── Brick string parser ──────────────────────────────────────────────

BRICK_PATTERN = re.compile(r"(\d+x\d+)\s*\((\d+),(\d+),(\d+)\)")


def parse_bricks(brick_str: str) -> list[dict]:
    """Parse ST2B brick string into list of {size, x, y, z} dicts."""
    bricks = []
    for match in BRICK_PATTERN.finditer(brick_str):
        bricks.append({
            "size": match.group(1),
            "x": int(match.group(2)),
            "y": int(match.group(3)),
            "z": int(match.group(4)),
        })
    return bricks


# ── Layer grouping and subassembly creation ───────────────────────────

def group_into_subassemblies(bricks: list[dict], max_subassemblies: int = 5) -> list[dict]:
    """Group bricks by z-layer, merge into 3-5 subassemblies."""
    if not bricks:
        return []

    # Group by z
    by_z = defaultdict(list)
    for b in bricks:
        by_z[b["z"]].append(b)

    z_layers = sorted(by_z.keys())
    if not z_layers:
        return []

    # Merge layers into subassemblies (target 3-5 groups)
    num_layers = len(z_layers)
    if num_layers <= max_subassemblies:
        groups = [[z] for z in z_layers]
    else:
        # Split evenly
        chunk_size = max(1, num_layers // max_subassemblies)
        groups = []
        for i in range(0, num_layers, chunk_size):
            groups.append(z_layers[i:i + chunk_size])
        # Merge overflow into last group
        if len(groups) > max_subassemblies:
            groups[-2].extend(groups[-1])
            groups.pop()

    # Assign positions based on vertical location
    subassemblies = []
    for i, z_group in enumerate(groups):
        if i == 0:
            position = "bottom"
        elif i == len(groups) - 1:
            position = "top"
        else:
            position = "center"

        layer_bricks = []
        for z in z_group:
            layer_bricks.extend(by_z[z])

        subassemblies.append({
            "z_group": z_group,
            "bricks": layer_bricks,
            "position": position,
        })

    return subassemblies


# ── Color assignment ──────────────────────────────────────────────────

def assign_colors(subassemblies: list[dict], subcategory: str, rng: Random) -> tuple[list[dict], list[str]]:
    """Assign semantic colors to subassemblies based on category and position.

    Returns (colored_subassemblies, dominant_colors).
    """
    palette = COLOR_PALETTES.get(subcategory, DEFAULT_PALETTE)
    scheme = rng.choice(palette["schemes"])
    base_color, body_color, accent_color = scheme

    window_color = palette.get("window_color")

    colored = []
    for i, sa in enumerate(subassemblies):
        # Pick layer color based on position
        if sa["position"] == "bottom":
            layer_color = base_color
        elif sa["position"] == "top":
            layer_color = accent_color
        else:
            layer_color = body_color

        # Aggregate bricks by (part_id, color)
        part_counts = Counter()
        part_info = {}

        for brick in sa["bricks"]:
            part = _get_part_for_size(brick["size"])
            # 80% layer color, 20% accent for contrast
            if rng.random() < 0.8:
                color = layer_color
            else:
                color = accent_color if sa["position"] != "top" else body_color

            key = (part["part_id"], color["color"])
            part_counts[key] += 1
            part_info[key] = {
                "part_id": part["part_id"],
                "name": part["name"],
                "category": part["category"],
                "color": color["color"],
                "color_hex": f"#{color['hex'].lstrip('#')}",
                "is_trans": color.get("is_trans", False),
            }

        # Add window parts for vehicles in middle/top layers
        if window_color and sa["position"] in ("center", "top") and len(sa["bricks"]) > 3:
            num_windows = max(1, len(sa["bricks"]) // 6)
            key = ("60594", window_color["color"])
            part_counts[key] += num_windows
            part_info[key] = {
                "part_id": "60594",
                "name": "Window 1x2x3 Pane",
                "category": "Windows and Doors",
                "color": window_color["color"],
                "color_hex": f"#{window_color['hex'].lstrip('#')}",
                "is_trans": window_color.get("is_trans", True),
            }

        # Build parts list
        parts = []
        for key, qty in part_counts.items():
            info = part_info[key].copy()
            info["quantity"] = qty
            parts.append(info)

        # Sort parts by quantity descending
        parts.sort(key=lambda p: p["quantity"], reverse=True)

        colored.append({
            "parts": parts,
            "position": sa["position"],
        })

    # Dominant colors from scheme
    dominant_colors = [body_color["color"], base_color["color"], accent_color["color"]]
    # Deduplicate while preserving order
    seen = set()
    unique_colors = []
    for c in dominant_colors:
        if c not in seen:
            seen.add(c)
            unique_colors.append(c)

    return colored, unique_colors


# ── Full conversion ──────────────────────────────────────────────────

def convert_example(example: dict, rng: Random) -> tuple[dict | None, list[str]]:
    """Convert a single ST2B example to LegoDescription JSON + prompt list.

    Returns (label_dict, prompts_list) or (None, []) if invalid.
    """
    # Parse bricks
    bricks = parse_bricks(example["bricks"])
    if len(bricks) < 3 or len(bricks) > 300:
        return None, []

    # Get category info
    cat_id = example["category_id"]
    cat_info = SHAPENET_CATEGORIES.get(cat_id, {
        "name": "object", "category": "Creator", "subcategory": "Household"
    })

    # Group into subassemblies
    subassemblies = group_into_subassemblies(bricks)
    if not subassemblies:
        return None, []

    # Assign colors
    colored_subs, dominant_colors = assign_colors(
        subassemblies, cat_info["subcategory"], rng
    )

    # Build subassembly dicts
    sub_dicts = []
    for i, csub in enumerate(colored_subs):
        position = csub["position"]
        # Generate connections
        connects_to = []
        if i > 0:
            connects_to.append(f"layer_{i - 1}")
        if i < len(colored_subs) - 1:
            connects_to.append(f"layer_{i + 1}")

        sub_dicts.append({
            "name": f"layer_{i}",
            "type": "Bricks",
            "parts": csub["parts"],
            "spatial": {
                "position": position,
                "orientation": "flat" if position == "bottom" else "upright",
                "connects_to": connects_to,
            },
        })

    # Compute metadata
    total_parts = sum(
        p["quantity"] for sd in sub_dicts for p in sd["parts"]
    )
    num_bricks = len(bricks)
    if num_bricks < 20:
        complexity = "simple"
    elif num_bricks < 80:
        complexity = "intermediate"
    elif num_bricks < 200:
        complexity = "advanced"
    else:
        complexity = "expert"

    # Dimension estimate from grid extents
    xs = [b["x"] for b in bricks]
    ys = [b["y"] for b in bricks]
    zs = [b["z"] for b in bricks]

    def _dim(extent):
        if extent < 8:
            return "small"
        elif extent < 14:
            return "medium"
        return "large"

    # Pick first caption as object name
    captions = example.get("captions", ["LEGO model"])
    object_name = captions[0] if captions else "LEGO model"
    # Truncate long captions for the object field
    if len(object_name) > 80:
        object_name = object_name[:77] + "..."

    label = {
        "set_id": example.get("structure_id", "st2b-unknown"),
        "object": object_name,
        "category": cat_info["category"],
        "subcategory": cat_info["subcategory"],
        "complexity": complexity,
        "total_parts": total_parts,
        "dominant_colors": dominant_colors,
        "dimensions_estimate": {
            "width": _dim(max(xs) - min(xs) + 1),
            "height": _dim(max(zs) - min(zs) + 1),
            "depth": _dim(max(ys) - min(ys) + 1),
        },
        "subassemblies": sub_dicts,
        "build_hints": [
            f"Start with the {sub_dicts[0]['spatial']['position']} layer",
            f"This {cat_info['name']} uses {total_parts} pieces total",
            "Work from bottom to top for stability",
        ],
    }

    # Generate prompt variants from captions
    # Plain captions + color-aware versions
    prompts = list(captions)  # 5 plain captions
    body_color = dominant_colors[0] if dominant_colors else "Red"
    for cap in captions[:2]:
        # Truncate for prompt use
        short_cap = cap.split(".")[0] if "." in cap else cap
        if len(short_cap) > 60:
            short_cap = short_cap[:57] + "..."
        prompts.append(f"Build me a {body_color.lower()} {short_cap.lower()}")
        prompts.append(f"I want a LEGO {short_cap.lower()}")

    return label, prompts


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert StableText2Brick to LegoGen format")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit examples for testing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading StableText2Brick ({args.split} split)...")
    dataset = load_dataset("AvaLovelace/StableText2Brick", split=args.split)

    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))
    print(f"  {len(dataset)} examples loaded")

    # Create output dirs
    ST2B_CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
    ST2B_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    rng = Random(args.seed)
    converted = 0
    skipped = 0

    for i, example in enumerate(dataset):
        label, prompts = convert_example(example, rng)

        if label is None:
            skipped += 1
            continue

        sid = label["set_id"]

        # Save label
        label_path = ST2B_CONVERTED_DIR / f"{sid}.json"
        with open(label_path, "w") as f:
            json.dump(label, f, indent=2)

        # Save prompts
        prompts_path = ST2B_PROMPTS_DIR / f"{sid}.json"
        with open(prompts_path, "w") as f:
            json.dump(prompts, f, indent=2)

        converted += 1
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} — {converted} converted, {skipped} skipped")

    print(f"\nDone! Converted {converted}, skipped {skipped}")
    print(f"  Labels: {ST2B_CONVERTED_DIR}")
    print(f"  Prompts: {ST2B_PROMPTS_DIR}")

    # Save split info for dataset loading
    split_info = {
        "split": args.split,
        "count": converted,
        "ids": [f.stem for f in sorted(ST2B_CONVERTED_DIR.glob("*.json"))],
    }
    split_path = ST2B_CONVERTED_DIR.parent / f"st2b_{args.split}_split.json"
    with open(split_path, "w") as f:
        json.dump(split_info, f)
    print(f"  Split info: {split_path}")


if __name__ == "__main__":
    main()
