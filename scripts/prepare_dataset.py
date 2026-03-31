#!/usr/bin/env python3
"""Orchestrates LEGO dataset collection using Rebrickable CSV database dumps.

Downloads CSV files directly from Rebrickable (no API key needed for CSVs),
then builds JSON labels and downloads set images.

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --max-sets 1000 --skip-images
"""

import argparse
import csv
import gzip
import io
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import DATA_DIR, MIN_PARTS, MAX_PARTS, VAL_RATIO

# ── Rebrickable CSV download URLs ──────────────────────────────────────
CSV_BASE = "https://cdn.rebrickable.com/media/downloads"
CSV_URLS = {
    "sets":       f"{CSV_BASE}/sets.csv.gz",
    "themes":     f"{CSV_BASE}/themes.csv.gz",
    "colors":     f"{CSV_BASE}/colors.csv.gz",
    "parts":      f"{CSV_BASE}/parts.csv.gz",
    "part_cats":  f"{CSV_BASE}/part_categories.csv.gz",
    "inventories":    f"{CSV_BASE}/inventories.csv.gz",
    "inventory_parts": f"{CSV_BASE}/inventory_parts.csv.gz",
}


def download_csv(name: str, url: str, cache_dir: Path) -> list[dict]:
    """Download a gzipped CSV and return as list of dicts."""
    cache_path = cache_dir / f"{name}.csv.gz"

    # Use cached file if it exists (less than 7 days old)
    if cache_path.exists():
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400
        if age_days < 7:
            print(f"  Using cached {name}.csv.gz ({age_days:.1f} days old)")
            with gzip.open(cache_path, "rt", encoding="utf-8") as f:
                return list(csv.DictReader(f))

    print(f"  Downloading {name}.csv.gz ...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(resp.content)

    with gzip.open(io.BytesIO(resp.content), "rt", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def download_image(url: str, save_path: Path) -> bool:
    """Download a single image. Returns True on success."""
    if save_path.exists():
        return True
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        save_path.write_bytes(resp.content)
        return True
    except Exception:
        return False


def classify_complexity(num_parts: int) -> str:
    if num_parts < 50:
        return "simple"
    elif num_parts < 200:
        return "intermediate"
    elif num_parts < 500:
        return "advanced"
    return "expert"


def estimate_dimensions(num_parts: int) -> dict:
    if num_parts < 50:
        return {"width": "small", "height": "small", "depth": "small"}
    elif num_parts < 150:
        return {"width": "medium", "height": "small", "depth": "small"}
    elif num_parts < 300:
        return {"width": "medium", "height": "medium", "depth": "medium"}
    return {"width": "large", "height": "large", "depth": "medium"}


# Category-to-spatial mapping
CATEGORY_SPATIAL = {
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
    "Plants and Animals": {"position": "center", "orientation": "upright"},
}
DEFAULT_SPATIAL = {"position": "center", "orientation": "upright"}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare LEGO training dataset from Rebrickable CSV dumps")
    parser.add_argument("--output-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--max-sets", type=int, default=5000)
    parser.add_argument("--min-parts", type=int, default=MIN_PARTS)
    parser.add_argument("--max-parts", type=int, default=MAX_PARTS)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-images", action="store_true", help="Skip image downloads (for testing)")
    parser.add_argument("--api-key", type=str, default=None, help="(Unused, kept for backward compat)")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache"
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Download all CSV tables ────────────────────────────────
    print("Downloading Rebrickable database CSVs...")
    raw = {}
    for name, url in CSV_URLS.items():
        raw[name] = download_csv(name, url, cache_dir)
    print()

    # ── Step 2: Build lookup tables ────────────────────────────────────
    print("Building lookup tables...")

    # Themes: id -> {name, parent_id}
    themes = {}
    for row in raw["themes"]:
        themes[int(row["id"])] = {
            "name": row["name"],
            "parent_id": int(row["parent_id"]) if row["parent_id"] else None,
        }

    def resolve_theme_chain(theme_id: int) -> list[str]:
        chain = []
        current = theme_id
        while current and current in themes:
            chain.append(themes[current]["name"])
            current = themes[current].get("parent_id")
        chain.reverse()
        return chain

    # Colors: id -> {name, rgb, is_trans}
    colors = {}
    for row in raw["colors"]:
        colors[int(row["id"])] = {
            "name": row["name"],
            "rgb": row["rgb"],
            "is_trans": row["is_trans"].lower() == "t",
        }

    # Part categories: id -> name
    part_cats = {}
    for row in raw["part_cats"]:
        part_cats[int(row["id"])] = row["name"]

    # Parts: part_num -> {name, cat_id}
    parts_lookup = {}
    for row in raw["parts"]:
        parts_lookup[row["part_num"]] = {
            "name": row["name"],
            "part_cat_id": int(row["part_cat_id"]),
        }

    # Inventories: map set_num -> inventory_id (use the latest/highest version)
    set_to_inv = {}
    for row in raw["inventories"]:
        set_num = row["set_num"]
        inv_id = int(row["id"])
        version = int(row["version"])
        if set_num not in set_to_inv or version > set_to_inv[set_num][1]:
            set_to_inv[set_num] = (inv_id, version)

    inv_id_to_set = {v[0]: k for k, v in set_to_inv.items()}

    # Inventory parts: inventory_id -> [parts...]
    inv_parts = defaultdict(list)
    for row in raw["inventory_parts"]:
        inv_id = int(row["inventory_id"])
        if inv_id in inv_id_to_set:
            inv_parts[inv_id].append(row)

    print(f"  {len(themes)} themes, {len(colors)} colors, {len(part_cats)} part categories")
    print(f"  {len(parts_lookup)} parts, {len(set_to_inv)} sets with inventories")

    # Save colors/categories cache (used by PartLibrary at inference time)
    cat_cache = {str(k): v for k, v in part_cats.items()}
    col_cache = {str(k): v for k, v in colors.items()}
    with open(cache_dir / "categories.json", "w") as f:
        json.dump(cat_cache, f, indent=2)
    with open(cache_dir / "colors.json", "w") as f:
        json.dump(col_cache, f, indent=2)

    # ── Step 3: Filter sets ────────────────────────────────────────────
    print(f"\nFiltering sets ({args.min_parts}-{args.max_parts} parts)...")
    eligible_sets = []
    for row in raw["sets"]:
        num_parts = int(row["num_parts"])
        if args.min_parts <= num_parts <= args.max_parts and row.get("img_url"):
            set_num = row["set_num"]
            if set_num in set_to_inv:
                eligible_sets.append(row)

    # Sort by year descending (newer sets first)
    eligible_sets.sort(key=lambda s: int(s.get("year", 0)), reverse=True)
    if args.max_sets and len(eligible_sets) > args.max_sets:
        eligible_sets = eligible_sets[:args.max_sets]

    print(f"  {len(eligible_sets)} eligible sets")

    # ── Step 4: Build labels and download images ───────────────────────
    print(f"\nProcessing sets...")
    successful = []
    failed = []

    for i, set_row in enumerate(tqdm(eligible_sets, desc="Building dataset")):
        set_num = set_row["set_num"]
        num_parts = int(set_row["num_parts"])
        theme_id = int(set_row["theme_id"]) if set_row.get("theme_id") else None
        image_url = set_row.get("img_url", "")

        # Download image
        if not args.skip_images and image_url:
            ext = Path(image_url).suffix or ".jpg"
            img_path = images_dir / f"{set_num}{ext}"
            if not download_image(image_url, img_path):
                failed.append(set_num)
                continue
        elif not args.skip_images:
            failed.append(set_num)
            continue

        # Get inventory
        inv_id = set_to_inv[set_num][0]
        inv_items = inv_parts.get(inv_id, [])
        if not inv_items:
            failed.append(set_num)
            continue

        # Build subassemblies grouped by category
        groups: dict[str, list[dict]] = {}
        color_counter: Counter = Counter()

        for item in inv_items:
            part_num = item["part_num"]
            color_id = int(item["color_id"])
            quantity = int(item["quantity"])
            is_spare = item.get("is_spare", "f").lower() == "t"
            if is_spare:
                continue

            part_info = parts_lookup.get(part_num, {"name": "Unknown", "part_cat_id": 0})
            cat_name = part_cats.get(part_info["part_cat_id"], "Other")
            color_info = colors.get(color_id, {"name": "Unknown", "rgb": "000000", "is_trans": False})

            color_counter[color_info["name"]] += quantity

            part_entry = {
                "part_id": part_num,
                "name": part_info["name"],
                "category": cat_name,
                "color": color_info["name"],
                "color_hex": f"#{color_info['rgb']}",
                "is_trans": color_info["is_trans"],
                "quantity": quantity,
            }

            if cat_name not in groups:
                groups[cat_name] = []
            groups[cat_name].append(part_entry)

        # Build subassembly list
        subassemblies = []
        for cat_name in sorted(groups.keys()):
            spatial = CATEGORY_SPATIAL.get(cat_name, DEFAULT_SPATIAL)
            safe_name = cat_name.lower().replace(" ", "_").replace(",", "")
            subassemblies.append({
                "name": safe_name,
                "type": cat_name,
                "parts": groups[cat_name],
                "spatial": {**spatial, "connects_to": []},
            })

        # Simple connectivity heuristic
        pos_map = defaultdict(list)
        for sa in subassemblies:
            pos_map[sa["spatial"]["position"]].append(sa["name"])
        for sa in subassemblies:
            pos = sa["spatial"]["position"]
            if pos == "bottom":
                sa["spatial"]["connects_to"] = pos_map.get("center", [])
            elif pos == "center":
                sa["spatial"]["connects_to"] = pos_map.get("top", [])
            elif pos == "top":
                sa["spatial"]["connects_to"] = pos_map.get("center", [])

        # Theme info
        theme_chain = resolve_theme_chain(theme_id) if theme_id else []
        category = theme_chain[0] if theme_chain else "Unknown"
        subcategory = theme_chain[-1] if len(theme_chain) > 1 else category

        # Dominant colors
        dominant_colors = [name for name, _ in color_counter.most_common(3)]

        # Build hints
        hints = ["Start with the base plate or foundation pieces"]
        if num_parts > 100:
            hints.append("Sort pieces by color before building")
        if category in ("City", "Creator", "Architecture"):
            hints.append("Build walls before attaching the roof")
        elif category in ("Technic", "Mindstorms"):
            hints.append("Assemble the gear train and axles first")
        if num_parts > 200:
            hints.append("Work in sections — complete each subassembly before joining")

        # Final label
        label = {
            "set_id": set_num,
            "object": set_row["name"],
            "category": category,
            "subcategory": subcategory,
            "complexity": classify_complexity(num_parts),
            "total_parts": num_parts,
            "dominant_colors": dominant_colors,
            "dimensions_estimate": estimate_dimensions(num_parts),
            "subassemblies": subassemblies,
            "build_hints": hints,
        }

        # Save label
        label_path = labels_dir / f"{set_num}.json"
        with open(label_path, "w") as f:
            json.dump(label, f, indent=2)

        successful.append(set_num)

    # ── Step 5: Create train/val split ─────────────────────────────────
    print(f"\nCreating train/val split (val_ratio={args.val_ratio})...")
    random.shuffle(successful)
    val_size = int(len(successful) * args.val_ratio)
    val_sets = successful[:val_size]
    train_sets = successful[val_size:]

    splits = {"train": train_sets, "val": val_sets}
    splits_path = output_dir / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Dataset Preparation Complete")
    print("=" * 60)
    print(f"  Total sets attempted: {len(eligible_sets)}")
    print(f"  Successful:           {len(successful)}")
    print(f"  Failed:               {len(failed)}")
    print(f"  Train split:          {len(train_sets)}")
    print(f"  Val split:            {len(val_sets)}")
    print(f"  Images dir:           {images_dir}")
    print(f"  Labels dir:           {labels_dir}")
    print(f"  Splits file:          {splits_path}")


if __name__ == "__main__":
    main()
