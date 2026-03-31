#!/usr/bin/env python3
"""Orchestrates LEGO dataset collection from Rebrickable API.

Usage:
    python scripts/prepare_dataset.py --api-key YOUR_KEY --max-sets 5000
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (
    DATA_DIR,
    MIN_PARTS,
    MAX_PARTS,
    VAL_RATIO,
)
from backend.data_pipeline.part_library import PartLibrary
from backend.data_pipeline.manuals_loader import RebrickableLoader
from backend.data_pipeline.manuals_preprocess import (
    build_json_label,
    validate_label,
    save_label,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare LEGO training dataset")
    parser.add_argument("--api-key", type=str, required=True, help="Rebrickable API key")
    parser.add_argument("--output-dir", type=str, default=str(DATA_DIR), help="Output directory")
    parser.add_argument("--max-sets", type=int, default=5000, help="Maximum number of sets to download")
    parser.add_argument("--min-parts", type=int, default=MIN_PARTS, help="Minimum parts per set")
    parser.add_argument("--max-parts", type=int, default=MAX_PARTS, help="Maximum parts per set")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # ── Initialize ─────────────────────────────────────────────────────
    print("Initializing Rebrickable loader and part library...")
    loader = RebrickableLoader(api_key=args.api_key)
    part_lib = PartLibrary(api_key=args.api_key)
    part_lib.ensure_loaded()

    print(f"  Loaded {len(part_lib.categories)} part categories")
    print(f"  Loaded {len(part_lib.colors)} colors")

    # ── Fetch themes ───────────────────────────────────────────────────
    print("Fetching theme hierarchy...")
    themes = loader.fetch_themes()
    print(f"  Found {len(themes)} themes")

    # ── Fetch sets ─────────────────────────────────────────────────────
    print(f"Fetching sets ({args.min_parts}-{args.max_parts} parts, max {args.max_sets})...")
    sets = loader.fetch_all_sets(
        min_parts=args.min_parts,
        max_parts=args.max_parts,
        max_sets=args.max_sets,
    )
    print(f"  Found {len(sets)} sets")

    # ── Process each set ───────────────────────────────────────────────
    successful = []
    failed = []

    for i, set_info in enumerate(sets):
        set_num = set_info["set_num"]
        progress = f"[{i + 1}/{len(sets)}]"

        # Download image
        image_url = set_info.get("set_img_url")
        if not image_url:
            print(f"  {progress} {set_num}: No image URL, skipping")
            failed.append(set_num)
            continue

        image_path = loader.download_set_image(set_num, image_url, Path(args.output_dir) / "images")
        if not image_path:
            print(f"  {progress} {set_num}: Image download failed, skipping")
            failed.append(set_num)
            continue

        # Fetch inventory
        try:
            inventory = loader.fetch_set_inventory(set_num)
        except Exception as e:
            print(f"  {progress} {set_num}: Inventory fetch failed ({e}), skipping")
            failed.append(set_num)
            continue

        if not inventory:
            print(f"  {progress} {set_num}: Empty inventory, skipping")
            failed.append(set_num)
            continue

        # Build JSON label
        theme_id = set_info.get("theme_id")
        theme_chain = loader.resolve_theme_hierarchy(theme_id, themes) if theme_id else []

        label = build_json_label(set_info, inventory, theme_chain, part_lib)
        is_valid, errors = validate_label(label)

        if not is_valid:
            print(f"  {progress} {set_num}: Invalid label ({errors}), skipping")
            failed.append(set_num)
            continue

        save_label(label, set_num, Path(args.output_dir) / "labels")
        successful.append(set_num)

        if (i + 1) % 50 == 0:
            print(f"  {progress} Processed {len(successful)} successful, {len(failed)} failed")

    # ── Create train/val split ─────────────────────────────────────────
    print(f"\nCreating train/val split (val_ratio={args.val_ratio})...")
    random.shuffle(successful)
    val_size = int(len(successful) * args.val_ratio)
    val_sets = successful[:val_size]
    train_sets = successful[val_size:]

    splits = {"train": train_sets, "val": val_sets}
    splits_path = Path(args.output_dir) / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Dataset Preparation Complete")
    print("=" * 60)
    print(f"  Total sets attempted: {len(sets)}")
    print(f"  Successful:           {len(successful)}")
    print(f"  Failed:               {len(failed)}")
    print(f"  Train split:          {len(train_sets)}")
    print(f"  Val split:            {len(val_sets)}")
    print(f"  Images dir:           {Path(args.output_dir) / 'images'}")
    print(f"  Labels dir:           {Path(args.output_dir) / 'labels'}")
    print(f"  Splits file:          {splits_path}")


if __name__ == "__main__":
    main()
