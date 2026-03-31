#!/usr/bin/env python3
"""Collect MOC (My Own Creation) data from Rebrickable API.

MOCs are fan-built custom LEGO sets with real photos and part inventories.
This gives us real creative photos → LEGO parts mapping, which is closer
to the 'any image → LEGO build' use case.

Requires a Rebrickable API key (free at rebrickable.com/api).

Usage:
    python scripts/prepare_moc_dataset.py --api-key YOUR_KEY --max-mocs 2000
"""

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import DATA_DIR, MIN_PARTS, MAX_PARTS, VAL_RATIO

# MOC categories map to LEGO themes
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
}
DEFAULT_SPATIAL = {"position": "center", "orientation": "upright"}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare MOC dataset from Rebrickable API")
    parser.add_argument("--api-key", type=str, required=True, help="Rebrickable API key")
    parser.add_argument("--output-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--max-mocs", type=int, default=2000)
    parser.add_argument("--min-parts", type=int, default=MIN_PARTS)
    parser.add_argument("--max-parts", type=int, default=MAX_PARTS)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-images", action="store_true")
    return parser.parse_args()


class RebrickableClient:
    def __init__(self, api_key: str):
        self.headers = {"Authorization": f"key {api_key}"}
        self._last = 0.0

    def _get(self, url: str, params: dict = None, retries: int = 5) -> dict:
        for attempt in range(retries):
            elapsed = time.time() - self._last
            if elapsed < 1.5:
                time.sleep(1.5 - elapsed)
            self._last = time.time()

            resp = requests.get(url, headers=self.headers, params=params or {}, timeout=30)
            if resp.status_code == 429:
                wait = min(2 ** attempt * 2, 60)
                print(f"\n  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()

    def fetch_mocs(self, page_size: int = 100) -> list[dict]:
        """Paginate through all MOCs on Rebrickable."""
        mocs = []
        url = "https://rebrickable.com/api/v3/lego/mocs/"
        page = 1
        while True:
            data = self._get(url, {"page": page, "page_size": page_size})
            mocs.extend(data["results"])
            print(f"  Fetched {len(mocs)} MOCs...", end="\r")
            if not data.get("next"):
                break
            page += 1
        return mocs

    def fetch_moc_parts(self, moc_num: str) -> list[dict]:
        """Get parts for a specific MOC."""
        parts = []
        url = f"https://rebrickable.com/api/v3/lego/mocs/{moc_num}/parts/"
        page = 1
        while True:
            data = self._get(url, {"page": page, "page_size": 200})
            parts.extend(data["results"])
            if not data.get("next"):
                break
            page += 1
        return parts

    def download_image(self, url: str, save_path: Path) -> bool:
        if save_path.exists():
            return True
        if not url:
            return False
        try:
            elapsed = time.time() - self._last
            if elapsed < 0.5:
                time.sleep(0.5 - elapsed)
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            save_path.write_bytes(resp.content)
            return True
        except Exception:
            return False


def classify_complexity(num_parts: int) -> str:
    if num_parts < 50:   return "simple"
    elif num_parts < 200: return "intermediate"
    elif num_parts < 500: return "advanced"
    return "expert"


def estimate_dimensions(num_parts: int) -> dict:
    if num_parts < 50:   return {"width": "small", "height": "small", "depth": "small"}
    elif num_parts < 150: return {"width": "medium", "height": "small", "depth": "small"}
    elif num_parts < 300: return {"width": "medium", "height": "medium", "depth": "medium"}
    return {"width": "large", "height": "large", "depth": "medium"}


def build_label_from_moc(moc: dict, parts: list[dict]) -> dict:
    """Build a JSON label from MOC metadata + parts list."""
    num_parts = moc.get("num_parts", len(parts))

    # Group parts by category
    groups: dict[str, list] = defaultdict(list)
    color_counter: Counter = Counter()

    for item in parts:
        part = item.get("part", {})
        color = item.get("color", {})
        quantity = item.get("quantity", 1)
        is_spare = item.get("is_spare", False)
        if is_spare:
            continue

        cat_name = part.get("part_cat_id", "Other")
        # Rebrickable returns category as an object in some endpoints
        if isinstance(item.get("part", {}).get("part_cat_id"), dict):
            cat_name = item["part"]["part_cat_id"].get("name", "Other")

        color_name = color.get("name", "Unknown")
        color_hex = f"#{color.get('rgb', '000000')}"
        is_trans = color.get("is_trans", False)

        color_counter[color_name] += quantity
        groups[str(cat_name)].append({
            "part_id": part.get("part_num", ""),
            "name": part.get("name", "Unknown"),
            "category": str(cat_name),
            "color": color_name,
            "color_hex": color_hex,
            "is_trans": is_trans,
            "quantity": quantity,
        })

    # Build subassemblies
    subassemblies = []
    pos_map = defaultdict(list)
    for cat_name, cat_parts in sorted(groups.items()):
        spatial = CATEGORY_SPATIAL.get(cat_name, DEFAULT_SPATIAL)
        safe_name = str(cat_name).lower().replace(" ", "_").replace(",", "")
        sa = {"name": safe_name, "type": cat_name, "parts": cat_parts,
              "spatial": {**spatial, "connects_to": []}}
        subassemblies.append(sa)
        pos_map[spatial["position"]].append(safe_name)

    for sa in subassemblies:
        pos = sa["spatial"]["position"]
        if pos == "bottom":
            sa["spatial"]["connects_to"] = pos_map.get("center", [])
        elif pos == "center":
            sa["spatial"]["connects_to"] = pos_map.get("top", [])
        elif pos == "top":
            sa["spatial"]["connects_to"] = pos_map.get("center", [])

    dominant_colors = [n for n, _ in color_counter.most_common(3)]
    theme = moc.get("theme_name", "Unknown")

    hints = ["Start with the base plate or foundation pieces"]
    if num_parts > 100:
        hints.append("Sort pieces by color before building")
    if num_parts > 200:
        hints.append("Work in sections — complete each subassembly before joining")

    return {
        "set_id": moc["moc_id"],
        "object": moc["name"],
        "category": theme,
        "subcategory": theme,
        "complexity": classify_complexity(num_parts),
        "total_parts": num_parts,
        "dominant_colors": dominant_colors,
        "dimensions_estimate": estimate_dimensions(num_parts),
        "subassemblies": subassemblies,
        "build_hints": hints,
    }


def main():
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    client = RebrickableClient(args.api_key)

    # ── Fetch MOC list ─────────────────────────────────────────────────
    print("Fetching MOC list from Rebrickable...")
    all_mocs = client.fetch_mocs()
    print(f"\n  Found {len(all_mocs)} total MOCs")

    # Filter by part count and image availability
    eligible = [
        m for m in all_mocs
        if args.min_parts <= m.get("num_parts", 0) <= args.max_parts
        and m.get("moc_img_url")
    ]
    random.shuffle(eligible)
    eligible = eligible[:args.max_mocs]
    print(f"  {len(eligible)} eligible MOCs ({args.min_parts}-{args.max_parts} parts, with image)")

    # ── Process each MOC ───────────────────────────────────────────────
    successful = []
    failed = []

    for i, moc in enumerate(tqdm(eligible, desc="Processing MOCs")):
        moc_id = moc["moc_id"]

        # Download image
        if not args.skip_images:
            img_url = moc.get("moc_img_url", "")
            ext = Path(img_url).suffix or ".jpg"
            img_path = images_dir / f"{moc_id}{ext}"
            if not client.download_image(img_url, img_path):
                failed.append(moc_id)
                continue

        # Fetch parts
        try:
            parts = client.fetch_moc_parts(moc_id)
        except Exception as e:
            failed.append(moc_id)
            continue

        if not parts:
            failed.append(moc_id)
            continue

        # Build label
        label = build_label_from_moc(moc, parts)
        label_path = labels_dir / f"{moc_id}.json"
        with open(label_path, "w") as f:
            json.dump(label, f, indent=2)

        successful.append(moc_id)

    # ── Merge with existing splits or create new ───────────────────────
    splits_path = output_dir / "splits.json"
    existing_train, existing_val = [], []
    if splits_path.exists():
        with open(splits_path) as f:
            existing = json.load(f)
            existing_train = existing.get("train", [])
            existing_val = existing.get("val", [])

    random.shuffle(successful)
    val_size = int(len(successful) * args.val_ratio)
    new_val = successful[:val_size]
    new_train = successful[val_size:]

    splits = {
        "train": existing_train + new_train,
        "val": existing_val + new_val,
    }
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MOC Dataset Preparation Complete")
    print("=" * 60)
    print(f"  MOCs attempted:   {len(eligible)}")
    print(f"  Successful:       {len(successful)}")
    print(f"  Failed:           {len(failed)}")
    print(f"  New train:        {len(new_train)}")
    print(f"  New val:          {len(new_val)}")
    print(f"  Total train now:  {len(splits['train'])}")
    print(f"  Total val now:    {len(splits['val'])}")


if __name__ == "__main__":
    main()
