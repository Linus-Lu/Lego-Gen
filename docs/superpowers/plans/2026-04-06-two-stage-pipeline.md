# Two-Stage LEGO Generation Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-stage unified pipeline with a two-stage approach: Stage 1 (image→description) and Stage 2 (text→JSON), with structure-aware training and coarse grid positions.

**Architecture:** Two separate LoRA adapters on Qwen3.5-27B. Stage 1 is a lightweight rank-32 adapter for image→text description. Stage 2 is the existing rank-128 adapter retrained on ST2B-only data with structure-aware loss weighting, compact JSON, and grid_pos per part. At inference, adapters are swapped via PEFT `set_adapter()`.

**Tech Stack:** Python 3.12, PyTorch, Transformers, PEFT, bitsandbytes, Qwen3.5-27B, COCO 2017, FastAPI, React/Three.js

---

## File Structure

### New files
| File | Responsibility |
|---|---|
| `backend/data_pipeline/add_grid_pos.py` | Preprocessing script: add `grid_pos: [x, z]` to all ST2B labels |
| `backend/data_pipeline/build_stage1_dataset.py` | Download COCO images, match to ST2B captions, generate Stage 1 training data |
| `backend/data_pipeline/dataset_stage1.py` | Stage 1 PyTorch Dataset + Collator |
| `backend/training/train_stage1.py` | Stage 1 training script (image→description LoRA) |
| `tests/test_grid_pos.py` | Tests for grid_pos computation |
| `tests/test_stage1_dataset.py` | Tests for Stage 1 dataset loading |
| `tests/test_structure_aware_loss.py` | Tests for structure-aware loss weighting |
| `tests/test_two_stage_pipeline.py` | Integration test for two-stage inference |

### Modified files
| File | Changes |
|---|---|
| `backend/config.py` | Add Stage 1 config (checkpoint dir, LoRA rank, LR, category mappings) |
| `backend/data_pipeline/dataset_unified.py` | Remove Rebrickable from Stage 2 dataset |
| `backend/training/train_unified.py` | Structure-aware loss, ProgressCallback (already partially done in working tree) |
| `backend/inference/pipeline.py` | Two-stage inference with adapter swapping |
| `backend/models/unified_model.py` | Support loading multiple named LoRA adapters |
| `backend/app.py` | Preload both adapters at startup |
| `frontend/src/components/LegoViewer.tsx` | Use `grid_pos` from parts data when available |

---

## Task 1: Commit existing working-tree improvements

The working tree already has partially-implemented changes from this session (structure-aware loss, compact JSON, viewer improvements, app preloading, pipeline warmup fix). Commit these as the foundation.

**Files:**
- Modified: `backend/app.py`, `backend/config.py`, `backend/data_pipeline/dataset_unified.py`, `backend/inference/pipeline.py`, `backend/models/unified_model.py`, `backend/training/train_unified.py`, `frontend/src/api/legogen.ts`, `frontend/src/components/BrickMesh.tsx`, `frontend/src/components/LegoViewer.tsx`

- [ ] **Step 1: Stage and commit all working-tree changes**

```bash
git add backend/app.py backend/config.py backend/data_pipeline/dataset_unified.py \
  backend/inference/pipeline.py backend/models/unified_model.py \
  backend/training/train_unified.py frontend/src/api/legogen.ts \
  frontend/src/components/BrickMesh.tsx frontend/src/components/LegoViewer.tsx
git commit -m "feat: structure-aware loss, compact JSON, viewer improvements, API fixes

- Add StructureAwareWeights + custom compute_loss (50x structural signal boost)
- Switch text samples to compact JSON (28% fewer tokens)
- Improve 3D viewer: row-packing layout, proper brick sizing from part names
- Fix tokenization cache warmup for Qwen3.5 chat template
- Preload model at startup to avoid first-request timeout
- Fix frontend API_BASE for remote access"
```

---

## Task 2: Add grid_pos to ST2B labels

Preprocessing script that adds `grid_pos: [x, z]` (coarse stud coordinates) to every part entry in every ST2B label file.

**Files:**
- Create: `backend/data_pipeline/add_grid_pos.py`
- Create: `tests/test_grid_pos.py`

- [ ] **Step 1: Write tests for grid_pos computation**

Create `tests/test_grid_pos.py`:

```python
"""Tests for grid_pos computation logic."""
import json
import pytest

from backend.data_pipeline.add_grid_pos import compute_grid_positions


def test_single_part_gets_origin():
    """Single part should be placed at [0, 0]."""
    layer = {
        "name": "layer_0",
        "type": "Bricks",
        "parts": [
            {"part_id": "3003", "name": "Brick 2x2", "quantity": 1}
        ],
        "spatial": {"position": "bottom", "orientation": "flat", "connects_to": ["layer_1"]},
    }
    result = compute_grid_positions(layer)
    assert result["parts"][0]["grid_pos"] == [0, 0]


def test_multiple_parts_pack_left_to_right():
    """Parts should fill left-to-right, wrapping at layer_width."""
    layer = {
        "name": "layer_0",
        "type": "Bricks",
        "parts": [
            {"part_id": "3004", "name": "Brick 1x2", "quantity": 3},
            {"part_id": "3003", "name": "Brick 2x2", "quantity": 2},
        ],
        "spatial": {"position": "bottom", "orientation": "flat", "connects_to": []},
    }
    result = compute_grid_positions(layer)
    # Each part type gets a single grid_pos (not per-instance)
    assert "grid_pos" in result["parts"][0]
    assert "grid_pos" in result["parts"][1]
    # Second part should be offset from first
    assert result["parts"][1]["grid_pos"][0] > result["parts"][0]["grid_pos"][0] or \
           result["parts"][1]["grid_pos"][1] > result["parts"][0]["grid_pos"][1]


def test_preserves_existing_fields():
    """grid_pos should be added without removing any existing fields."""
    layer = {
        "name": "layer_0",
        "type": "Bricks",
        "parts": [
            {"part_id": "3003", "name": "Brick 2x2", "category": "Bricks",
             "color": "Red", "color_hex": "#C91A09", "is_trans": False, "quantity": 2}
        ],
        "spatial": {"position": "bottom", "orientation": "flat", "connects_to": []},
    }
    result = compute_grid_positions(layer)
    part = result["parts"][0]
    assert part["part_id"] == "3003"
    assert part["color"] == "Red"
    assert part["quantity"] == 2
    assert "grid_pos" in part


def test_empty_parts_list():
    """Layer with no parts should return unchanged."""
    layer = {
        "name": "layer_0",
        "type": "Bricks",
        "parts": [],
        "spatial": {"position": "bottom", "orientation": "flat", "connects_to": []},
    }
    result = compute_grid_positions(layer)
    assert result["parts"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/Lego-Gen && python -m pytest tests/test_grid_pos.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'backend.data_pipeline.add_grid_pos'`

- [ ] **Step 3: Implement add_grid_pos.py**

Create `backend/data_pipeline/add_grid_pos.py`:

```python
#!/usr/bin/env python3
"""Add coarse grid_pos [x, z] to every part entry in ST2B label files.

For each subassembly (layer), parts are packed left-to-right within
a square footprint estimated from total stud area.

Usage:
    python -m backend.data_pipeline.add_grid_pos
    python -m backend.data_pipeline.add_grid_pos --labels-dir data/st2b_labels --dry-run
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import ST2B_CONVERTED_DIR


def parse_brick_width(name: str) -> int:
    """Extract stud width from part name like 'Brick 2x2' -> 2."""
    m = re.search(r"(\d+)\s*x\s*(\d+)", name)
    if m:
        return int(m.group(1))
    return 2  # default 2-stud width


def compute_grid_positions(layer: dict) -> dict:
    """Add grid_pos to each part in a single layer/subassembly.

    Returns a new dict with grid_pos added to each part entry.
    """
    layer = json.loads(json.dumps(layer))  # deep copy
    parts = layer.get("parts", [])
    if not parts:
        return layer

    # Estimate total stud area to compute layer width
    total_studs = 0
    for p in parts:
        w = parse_brick_width(p.get("name", ""))
        total_studs += w * p.get("quantity", 1)

    layer_width = max(4, math.ceil(math.sqrt(total_studs)))

    # Pack left-to-right, assign grid_pos per part type
    cursor_x = 0
    cursor_z = 0
    for p in parts:
        w = parse_brick_width(p.get("name", ""))
        span = w * p.get("quantity", 1)

        # Record position of this part type's first brick
        p["grid_pos"] = [cursor_x % layer_width, cursor_x // layer_width]

        cursor_x += span

    return layer


def add_grid_pos_to_label(label: dict) -> dict:
    """Add grid_pos to all subassemblies in a label."""
    label = json.loads(json.dumps(label))  # deep copy
    for i, sa in enumerate(label.get("subassemblies", [])):
        label["subassemblies"][i] = compute_grid_positions(sa)
    return label


def main():
    parser = argparse.ArgumentParser(description="Add grid_pos to ST2B labels")
    parser.add_argument("--labels-dir", type=str, default=str(ST2B_CONVERTED_DIR))
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing")
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    files = sorted(labels_dir.glob("*.json"))
    print(f"Processing {len(files)} label files in {labels_dir}...")

    modified = 0
    for f in files:
        label = json.load(open(f))
        updated = add_grid_pos_to_label(label)

        if not args.dry_run:
            with open(f, "w") as out:
                json.dump(updated, out)
        modified += 1

    print(f"{'Would modify' if args.dry_run else 'Modified'} {modified} files.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspace/Lego-Gen && python -m pytest tests/test_grid_pos.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Run the script on ST2B labels**

```bash
cd /workspace/Lego-Gen && python -m backend.data_pipeline.add_grid_pos --dry-run
```

Expected: `Would modify 46143 files.`

Then run for real:

```bash
cd /workspace/Lego-Gen && python -m backend.data_pipeline.add_grid_pos
```

- [ ] **Step 6: Verify a label was updated**

```bash
python3 -c "import json; d=json.load(open('data/st2b_labels/0000446a-c55e-4ec2-815e-590ac3d22f6f.json')); print(json.dumps(d['subassemblies'][0]['parts'][0], indent=2))"
```

Expected: Part entry now includes `"grid_pos": [0, 0]`

- [ ] **Step 7: Commit**

```bash
git add backend/data_pipeline/add_grid_pos.py tests/test_grid_pos.py
git commit -m "feat: add grid_pos preprocessing for ST2B labels

Adds coarse [x, z] stud coordinates per part type within each layer.
15% token increase, enables better 3D viewer brick placement."
```

---

## Task 3: Update Stage 2 config and remove Rebrickable from training

Update config for the two-stage setup and modify the dataset to use ST2B-only for Stage 2.

**Files:**
- Modify: `backend/config.py`
- Modify: `backend/data_pipeline/dataset_unified.py`

- [ ] **Step 1: Add Stage 1 config and update Stage 2 config in config.py**

Add after the existing unified model section in `backend/config.py`:

```python
# ── Stage 1: Image → Description (lightweight LoRA) ──────────────────
STAGE1_CHECKPOINT_DIR = CHECKPOINT_DIR / "qwen35-27b-lego-stage1-lora"
STAGE1_LORA_R = 32
STAGE1_LORA_ALPHA = 64
STAGE1_LEARNING_RATE = 5e-5
STAGE1_NUM_EPOCHS = 3
STAGE1_WARMUP_STEPS = 100
STAGE1_BATCH_SIZE = 4
STAGE1_GRADIENT_ACCUMULATION = 4  # effective batch = 16
STAGE1_MAX_SEQ_LENGTH = 512       # descriptions are short

# ── COCO → ST2B category mapping for Stage 1 data ────────────────────
COCO_TO_ST2B_CATEGORY = {
    "chair": "chair",
    "couch": "sofa",
    "bed": "bed",
    "dining table": "table",
    "car": "car",
    "bus": "bus",
    "train": "train",
    "truck": "truck",
    "boat": "vessel",
    "motorcycle": "motorbike",
    "airplane": "airplane",
    "bench": "bench",
    "vase": "vase",
    "cup": "mug",
    "laptop": "laptop",
}

# ── Stage 1 system prompt ─────────────────────────────────────────────
STAGE1_SYSTEM_PROMPT = (
    "You are a LEGO design assistant. Describe this object's shape, structure, "
    "colors, and proportions in a way useful for building it with LEGO bricks. "
    "Focus on geometry and spatial relationships, not materials or artistic style. "
    "Be concise — one to three sentences."
)
```

- [ ] **Step 2: Remove Rebrickable from Stage 2 dataset**

In `backend/data_pipeline/dataset_unified.py`, modify `__init__` to skip Rebrickable when in Stage 2 mode. Replace the `vision_set_nums` and `rebrickable_ids` handling:

Find the block that starts with `# ── Vision samples (image + JSON)` and wraps through `# ── Rebrickable text samples`. Wrap both in a condition:

```python
        # ── Vision samples (image + JSON) ─────────────────────────────
        # Skipped for Stage 2 (text-only) — vision handled by Stage 1
        if vision_set_nums:
            images_dir = self.data_dir / "images"
            labels_dir = self.data_dir / "labels"
            vision_samples = []
            for set_num in vision_set_nums:
                label_path = labels_dir / f"{set_num}.json"
                if not label_path.exists():
                    continue
                image_path = self._find_image(images_dir, set_num)
                if image_path:
                    vision_samples.append(("vision", {
                        "image_path": image_path,
                        "label_path": label_path,
                    }))
            # Upsample vision to balance with planner data
            self.samples.extend(vision_samples * max(1, vision_upsample))

        # ── Rebrickable text samples ──────────────────────────────────
        if rebrickable_ids:
            labels_dir = self.data_dir / "labels"
            rb_samples = []
            for sid in rebrickable_ids:
                label_path = labels_dir / f"{sid}.json"
                if not label_path.exists():
                    continue
                rb_samples.append(("text", {
                    "label_path": label_path,
                    "prompts_path": PLANNER_PROMPTS_DIR / f"{sid}.json",
                    "source": "rebrickable",
                }))
            self.samples.extend(rb_samples * max(1, rebrickable_upsample))
```

No code change needed in the dataset class itself — the caller in `train_unified.py` controls what gets passed. Update `train_unified.py` `main()` to pass empty lists for vision and rebrickable:

In `main()`, change the `load_unified_splits` usage to skip Rebrickable and vision:

```python
    splits = load_unified_splits(data_dir)

    train_ds = UnifiedLegoDataset(
        vision_set_nums=[],          # Stage 2: text-only, no vision
        rebrickable_ids=[],          # Stage 2: ST2B-only, no Rebrickable
        st2b_ids=splits["st2b_train"],
        data_dir=data_dir,
        processor=processor,
        max_length=args.max_seq_length,
        split="train",
        rebrickable_upsample=1,
        vision_upsample=1,
    )
    val_ds = UnifiedLegoDataset(
        vision_set_nums=[],
        rebrickable_ids=[],
        st2b_ids=splits["st2b_val"],
        data_dir=data_dir,
        processor=processor,
        max_length=args.max_seq_length,
        split="val",
        rebrickable_upsample=1,
        vision_upsample=1,
    )
```

- [ ] **Step 3: Commit**

```bash
git add backend/config.py backend/data_pipeline/dataset_unified.py backend/training/train_unified.py
git commit -m "feat: Stage 1 config + remove Rebrickable from Stage 2 training

Stage 2 now trains on ST2B-only data for consistent layer-based format.
Add COCO→ST2B category mapping and Stage 1 LoRA config."
```

---

## Task 4: Build Stage 1 dataset (COCO + ST2B caption matching)

Download COCO 2017 annotations, filter to matching categories, pair with ST2B captions.

**Files:**
- Create: `backend/data_pipeline/build_stage1_dataset.py`
- Create: `tests/test_stage1_dataset.py`

- [ ] **Step 1: Write test for COCO→ST2B matching**

Create `tests/test_stage1_dataset.py`:

```python
"""Tests for Stage 1 dataset building."""
import pytest

from backend.data_pipeline.build_stage1_dataset import (
    match_coco_to_st2b,
    generate_description_from_label,
)


def test_match_coco_to_st2b_chair():
    """COCO chair should match ST2B chair captions."""
    st2b_by_category = {
        "chair": [
            {"captions": ["A chair with four legs and a flat seat."]},
            {"captions": ["Simple wooden chair with backrest."]},
        ],
        "table": [
            {"captions": ["Rectangular table with legs."]},
        ],
    }
    result = match_coco_to_st2b("chair", st2b_by_category, seed=42)
    assert result is not None
    assert "chair" in result.lower() or "seat" in result.lower() or "legs" in result.lower()


def test_match_coco_to_st2b_unknown_category():
    """Unknown COCO category should return None."""
    st2b_by_category = {"chair": [{"captions": ["A chair."]}]}
    result = match_coco_to_st2b("giraffe", st2b_by_category, seed=42)
    assert result is None


def test_generate_description_from_label():
    """Generate description from Rebrickable label fields."""
    label = {
        "object": "Fire Station",
        "dominant_colors": ["Red", "White", "Dark Gray"],
        "dimensions_estimate": {"width": "large", "height": "medium", "depth": "medium"},
        "complexity": "advanced",
    }
    desc = generate_description_from_label(label)
    assert "Fire Station" in desc
    assert "Red" in desc
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/Lego-Gen && python -m pytest tests/test_stage1_dataset.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement build_stage1_dataset.py**

Create `backend/data_pipeline/build_stage1_dataset.py`:

```python
#!/usr/bin/env python3
"""Build Stage 1 training data: pair COCO images with ST2B-style captions.

Outputs a JSON manifest file listing image paths and target descriptions.

Usage:
    python -m backend.data_pipeline.build_stage1_dataset
    python -m backend.data_pipeline.build_stage1_dataset --coco-dir data/coco --output data/stage1_manifest.json
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (
    DATA_DIR,
    ST2B_CONVERTED_DIR,
    COCO_TO_ST2B_CATEGORY,
)


def load_st2b_captions_by_category(st2b_dir: Path) -> dict[str, list[dict]]:
    """Load ST2B labels grouped by subcategory (lowercased)."""
    by_cat: dict[str, list[dict]] = {}
    for f in st2b_dir.glob("*.json"):
        try:
            d = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        subcat = d.get("subcategory", "").lower()
        # Extract the object type (e.g., "chair", "table") from the subcategory or object field
        obj = d.get("object", "").lower()
        for known_cat in COCO_TO_ST2B_CATEGORY.values():
            if known_cat in obj or known_cat in subcat:
                by_cat.setdefault(known_cat, []).append({
                    "captions": [d.get("object", "")],
                    "set_id": d.get("set_id", ""),
                })
                break
    return by_cat


def match_coco_to_st2b(
    coco_category: str,
    st2b_by_category: dict[str, list[dict]],
    seed: int = 0,
) -> str | None:
    """Pick a random ST2B caption matching a COCO category."""
    st2b_cat = COCO_TO_ST2B_CATEGORY.get(coco_category)
    if not st2b_cat or st2b_cat not in st2b_by_category:
        return None

    rng = random.Random(seed)
    entries = st2b_by_category[st2b_cat]
    entry = rng.choice(entries)
    captions = entry.get("captions", [])
    return rng.choice(captions) if captions else None


def generate_description_from_label(label: dict) -> str:
    """Generate a structural description from a Rebrickable label."""
    obj = label.get("object", "LEGO model")
    colors = label.get("dominant_colors", [])
    dims = label.get("dimensions_estimate", {})
    color_str = ", ".join(colors[:3]) if colors else "mixed colors"
    size = dims.get("width", "medium")
    return f"{obj}. Dominant colors: {color_str}. Size: {size}."


def load_coco_annotations(coco_dir: Path) -> list[dict]:
    """Load COCO 2017 instance annotations."""
    ann_file = coco_dir / "annotations" / "instances_train2017.json"
    if not ann_file.exists():
        print(f"COCO annotations not found at {ann_file}")
        print("Download from: https://cocodataset.org/#download")
        print("  wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
        sys.exit(1)

    print(f"Loading COCO annotations from {ann_file}...")
    data = json.load(open(ann_file))

    # Build category ID → name mapping
    cat_map = {c["id"]: c["name"] for c in data["categories"]}

    # Build image ID → file_name mapping
    img_map = {img["id"]: img["file_name"] for img in data["images"]}

    # Group annotations by image, pick dominant category
    from collections import Counter
    img_cats: dict[int, list[str]] = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        cat_name = cat_map.get(ann["category_id"], "")
        if cat_name in COCO_TO_ST2B_CATEGORY:
            img_cats.setdefault(img_id, []).append(cat_name)

    # For each image, pick the most common matching category
    results = []
    for img_id, cats in img_cats.items():
        dominant = Counter(cats).most_common(1)[0][0]
        results.append({
            "image_id": img_id,
            "file_name": img_map.get(img_id, ""),
            "category": dominant,
        })

    return results


def build_stage1_manifest(
    coco_dir: Path,
    st2b_dir: Path,
    rebrickable_dir: Path | None,
    output_path: Path,
    max_per_category: int = 2000,
):
    """Build the Stage 1 training manifest."""
    # Load ST2B captions by category
    print("Loading ST2B captions...")
    st2b_by_cat = load_st2b_captions_by_category(st2b_dir)
    for cat, entries in sorted(st2b_by_cat.items()):
        print(f"  {cat}: {len(entries)} ST2B entries")

    # Load COCO annotations
    coco_images = load_coco_annotations(coco_dir)
    print(f"Found {len(coco_images)} COCO images matching ST2B categories")

    # Match and build manifest
    manifest = []
    cat_counts: dict[str, int] = {}

    for i, img in enumerate(coco_images):
        cat = img["category"]
        if cat_counts.get(cat, 0) >= max_per_category:
            continue

        caption = match_coco_to_st2b(cat, st2b_by_cat, seed=i)
        if caption is None:
            continue

        image_path = str(coco_dir / "train2017" / img["file_name"])
        if not os.path.exists(image_path):
            continue

        manifest.append({
            "image_path": image_path,
            "description": caption,
            "category": cat,
            "source": "coco",
        })
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    # Add Rebrickable images if available
    if rebrickable_dir:
        labels_dir = rebrickable_dir / "labels"
        images_dir = rebrickable_dir / "images"
        if labels_dir.exists() and images_dir.exists():
            print("Adding Rebrickable images...")
            for label_file in labels_dir.glob("*.json"):
                set_num = label_file.stem
                # Find matching image
                img_path = None
                for ext in (".jpg", ".jpeg", ".png", ".webp"):
                    candidate = images_dir / f"{set_num}{ext}"
                    if candidate.exists():
                        img_path = candidate
                        break
                if img_path is None:
                    continue

                label = json.load(open(label_file))
                desc = generate_description_from_label(label)
                manifest.append({
                    "image_path": str(img_path),
                    "description": desc,
                    "category": label.get("subcategory", "unknown").lower(),
                    "source": "rebrickable",
                })

    # Shuffle
    random.Random(42).shuffle(manifest)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {output_path}")
    print(f"Total samples: {len(manifest)}")
    for source in ("coco", "rebrickable"):
        n = len([m for m in manifest if m["source"] == source])
        print(f"  {source}: {n}")


def main():
    parser = argparse.ArgumentParser(description="Build Stage 1 training dataset")
    parser.add_argument("--coco-dir", type=str, default=str(DATA_DIR / "coco"))
    parser.add_argument("--st2b-dir", type=str, default=str(ST2B_CONVERTED_DIR))
    parser.add_argument("--rebrickable-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--output", type=str, default=str(DATA_DIR / "stage1_manifest.json"))
    parser.add_argument("--max-per-category", type=int, default=2000)
    args = parser.parse_args()

    build_stage1_manifest(
        coco_dir=Path(args.coco_dir),
        st2b_dir=Path(args.st2b_dir),
        rebrickable_dir=Path(args.rebrickable_dir),
        output_path=Path(args.output),
        max_per_category=args.max_per_category,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspace/Lego-Gen && python -m pytest tests/test_stage1_dataset.py -v
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/data_pipeline/build_stage1_dataset.py tests/test_stage1_dataset.py
git commit -m "feat: Stage 1 dataset builder (COCO + ST2B caption matching)"
```

---

## Task 5: Create Stage 1 Dataset class and training script

PyTorch Dataset for Stage 1 (image → description) and its training script.

**Files:**
- Create: `backend/data_pipeline/dataset_stage1.py`
- Create: `backend/training/train_stage1.py`

- [ ] **Step 1: Implement dataset_stage1.py**

Create `backend/data_pipeline/dataset_stage1.py`:

```python
"""Stage 1 dataset: image → structural description.

Loads image-description pairs from a manifest JSON file and
tokenizes them for causal LM training with prompt masking.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.config import STAGE1_SYSTEM_PROMPT, STAGE1_MAX_SEQ_LENGTH
from backend.data_pipeline.dataset import TRAIN_TRANSFORMS, VAL_TRANSFORMS


class Stage1Dataset(Dataset):
    """Image → description dataset for Stage 1 LoRA training."""

    def __init__(
        self,
        manifest_path: str | Path,
        processor,
        max_length: int = STAGE1_MAX_SEQ_LENGTH,
        split: str = "train",
    ):
        self.processor = processor
        self.max_length = max_length
        self.split = split
        self.augment = split == "train"

        with open(manifest_path, "r") as f:
            self.samples = json.load(f)

        # Train/val split: 90/10
        n = len(self.samples)
        split_idx = int(n * 0.9)
        if split == "train":
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        if self.augment:
            image = TRAIN_TRANSFORMS(image)
        else:
            image = VAL_TRANSFORMS(image)

        description = sample["description"]

        # Build chat messages
        messages = [
            {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this object for LEGO building."},
                ],
            },
            {"role": "assistant", "content": description},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        no_squeeze = {"pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}
        inputs = {k: v.squeeze(0) if k not in no_squeeze else v for k, v in inputs.items()}

        # Mask prompt tokens — only train on the description
        inputs["labels"] = self._mask_prompt(inputs)
        return inputs

    def _mask_prompt(self, inputs: dict) -> torch.Tensor:
        """Mask everything before the assistant response with -100."""
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()

        assistant_marker = self.processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False,
        )
        marker_len = len(assistant_marker)
        marker_t = torch.tensor(assistant_marker, dtype=input_ids.dtype)

        prompt_len = 0
        for i in range(len(input_ids) - marker_len, -1, -1):
            if torch.equal(input_ids[i : i + marker_len], marker_t):
                prompt_len = i + marker_len
                break

        labels[:prompt_len] = -100

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        return labels


class Stage1Collator:
    """Collator for Stage 1 batches with vision data."""

    def __call__(self, features: list[dict]) -> dict:
        batch = {}

        for key in ("input_ids", "attention_mask", "labels"):
            if key in features[0]:
                batch[key] = torch.stack([f[key] for f in features])

        has_pixels = [f for f in features if "pixel_values" in f]
        if has_pixels:
            batch["pixel_values"] = torch.cat([f["pixel_values"] for f in has_pixels], dim=0)
        if any("image_grid_thw" in f for f in features):
            grids = [f["image_grid_thw"] for f in features if "image_grid_thw" in f]
            batch["image_grid_thw"] = torch.cat(grids, dim=0)

        return batch
```

- [ ] **Step 2: Implement train_stage1.py**

Create `backend/training/train_stage1.py`:

```python
#!/usr/bin/env python3
"""Stage 1 training: fine-tune lightweight LoRA for image → structural description.

Usage:
    python -m backend.training.train_stage1
    python -m backend.training.train_stage1 --manifest data/stage1_manifest.json
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (
    DATA_DIR,
    UNIFIED_MODEL_NAME,
    STAGE1_CHECKPOINT_DIR,
    STAGE1_LORA_R,
    STAGE1_LORA_ALPHA,
    STAGE1_LEARNING_RATE,
    STAGE1_NUM_EPOCHS,
    STAGE1_WARMUP_STEPS,
    STAGE1_BATCH_SIZE,
    STAGE1_GRADIENT_ACCUMULATION,
    STAGE1_MAX_SEQ_LENGTH,
    SAVE_TOTAL_LIMIT,
    WEIGHT_DECAY,
    USE_BF16,
    LOGGING_STEPS,
)
from backend.data_pipeline.dataset_stage1 import Stage1Dataset, Stage1Collator
from backend.training.utils import seed_everything, setup_wandb


def main():
    parser = argparse.ArgumentParser(description="Stage 1 LoRA training (image→description)")
    parser.add_argument("--manifest", type=str, default=str(DATA_DIR / "stage1_manifest.json"))
    parser.add_argument("--output-dir", type=str, default=str(STAGE1_CHECKPOINT_DIR))
    parser.add_argument("--epochs", type=int, default=STAGE1_NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=STAGE1_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)

    if not args.no_wandb:
        setup_wandb("legogen-stage1", config=vars(args))

    # ── Model ──────────────────────────────────────────────────────────
    print(f"Loading {UNIFIED_MODEL_NAME} with Stage 1 LoRA (rank {STAGE1_LORA_R})...")
    from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        UNIFIED_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    processor = AutoProcessor.from_pretrained(
        UNIFIED_MODEL_NAME,
        trust_remote_code=True,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )

    # ── LoRA ───────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=STAGE1_LORA_R,
        lora_alpha=STAGE1_LORA_ALPHA,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ────────────────────────────────────────────────────────
    print("Loading Stage 1 dataset...")
    train_ds = Stage1Dataset(
        manifest_path=args.manifest,
        processor=processor,
        max_length=STAGE1_MAX_SEQ_LENGTH,
        split="train",
    )
    val_ds = Stage1Dataset(
        manifest_path=args.manifest,
        processor=processor,
        max_length=STAGE1_MAX_SEQ_LENGTH,
        split="val",
    )
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")

    # ── Training ───────────────────────────────────────────────────────
    effective_batch = STAGE1_BATCH_SIZE * STAGE1_GRADIENT_ACCUMULATION
    total_steps = (len(train_ds) // effective_batch) * args.epochs
    save_steps = max(50, total_steps // 5)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=STAGE1_BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=STAGE1_GRADIENT_ACCUMULATION,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=min(STAGE1_WARMUP_STEPS, total_steps // 5),
        weight_decay=WEIGHT_DECAY,
        bf16=USE_BF16,
        optim="paged_adamw_8bit",
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=save_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if not args.no_wandb else "none",
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=Stage1Collator(),
    )

    print("Starting Stage 1 training...")
    trainer.train()

    # Save final adapter
    print("Saving Stage 1 adapter...")
    model.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add backend/data_pipeline/dataset_stage1.py backend/training/train_stage1.py
git commit -m "feat: Stage 1 dataset class and training script (image→description)"
```

---

## Task 6: Two-stage inference pipeline

Update the inference pipeline to support adapter swapping between Stage 1 and Stage 2.

**Files:**
- Modify: `backend/inference/pipeline.py`
- Modify: `backend/models/unified_model.py`
- Modify: `backend/app.py`
- Create: `tests/test_two_stage_pipeline.py`

- [ ] **Step 1: Write integration test**

Create `tests/test_two_stage_pipeline.py`:

```python
"""Tests for two-stage inference pipeline."""
import pytest


def test_text_input_skips_stage1():
    """Text-only input should go directly to Stage 2."""
    from backend.inference.pipeline import UnifiedPipeline

    # This test verifies the routing logic, not actual model inference.
    # With LEGOGEN_DEV=1, the mock pipeline handles this.
    import os
    os.environ["LEGOGEN_DEV"] = "1"

    from backend.inference.pipeline import get_pipeline
    pipeline = get_pipeline()
    result = pipeline.generate_build_from_text("a small red house")

    assert "description" in result
    assert "steps" in result
    assert "metadata" in result


def test_generate_response_has_required_fields():
    """Pipeline output should have description, steps, metadata, validation."""
    import os
    os.environ["LEGOGEN_DEV"] = "1"

    from backend.inference.pipeline import get_pipeline
    pipeline = get_pipeline()
    result = pipeline.generate_build_from_text("a blue car")

    assert isinstance(result["description"], dict)
    assert isinstance(result["steps"], list)
    assert "model_version" in result["metadata"]
```

- [ ] **Step 2: Update unified_model.py to support named adapters**

In `backend/models/unified_model.py`, add a method to load a second adapter:

```python
    def load_named_adapter(self, name: str, adapter_path: str | Path):
        """Load an additional named LoRA adapter for adapter swapping."""
        from peft import PeftModel
        adapter_path = Path(adapter_path)
        if not adapter_path.exists():
            print(f"Adapter {name} not found at {adapter_path}, skipping")
            return False
        self.model.load_adapter(str(adapter_path), adapter_name=name)
        print(f"Loaded adapter '{name}' from {adapter_path}")
        return True

    def set_adapter(self, name: str):
        """Switch to a named adapter."""
        self.model.set_adapter(name)
```

- [ ] **Step 3: Update pipeline.py for two-stage inference**

In `backend/inference/pipeline.py`, update `UnifiedPipeline.__init__` to optionally load both adapters, and add a `describe_image_stage1` method:

Add to `__init__`, after the existing model loading:

```python
        # Try to load Stage 1 adapter for two-stage pipeline
        from backend.config import STAGE1_CHECKPOINT_DIR
        stage1_path = Path(STAGE1_CHECKPOINT_DIR)
        self.has_stage1 = False
        if stage1_path.exists() and (stage1_path / "adapter_config.json").exists():
            self.has_stage1 = self.wrapper.load_named_adapter("stage1", stage1_path)
            if self.has_stage1:
                # Switch back to default (stage2) adapter
                self.wrapper.set_adapter("default")
```

Add a new method for Stage 1:

```python
    def describe_image_stage1(self, image) -> str:
        """Stage 1: Generate a structural description from an image."""
        import torch
        from backend.config import STAGE1_SYSTEM_PROMPT
        from backend.models.tokenizer import strip_thinking_blocks

        if self.has_stage1:
            self.wrapper.set_adapter("stage1")

        with torch.inference_mode():
            messages = [
                {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe this object for LEGO building."},
                    ],
                },
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )

            inputs = self.processor(
                text=[text], images=[image], return_tensors="pt",
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs, max_new_tokens=256, temperature=0.7,
                top_p=0.9, do_sample=True,
            )
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            raw = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        raw = strip_thinking_blocks(raw)

        # Switch back to Stage 2 adapter
        if self.has_stage1:
            self.wrapper.set_adapter("default")

        return raw.strip()
```

Update `generate_build` to use two-stage when Stage 1 is available:

```python
    def generate_build(self, image, cache_key: str | None = None) -> dict:
        """Full pipeline: image -> description -> build steps.

        If Stage 1 adapter is loaded, uses two-stage pipeline:
          Stage 1: image -> text description
          Stage 2: text description -> LEGO JSON
        Otherwise falls back to single-stage image -> JSON.
        """
        if self.has_stage1:
            # Two-stage pipeline
            description_text = self.describe_image_stage1(image)
            result = self.describe_from_text(description_text)
        else:
            # Legacy single-stage
            result = self.describe_image(image, cache_key=cache_key)

        from backend.inference.postprocess_manual import json_to_steps
        from backend.inference.stability_checker import StabilityChecker
        from dataclasses import asdict

        description = result["description"]
        steps = json_to_steps(description) if description else []
        validation = asdict(StabilityChecker().validate(description))

        return {
            "description": description,
            "steps": steps,
            "metadata": {
                "model_version": "qwen35-lego-two-stage-v1" if self.has_stage1 else "qwen35-lego-unified-v1",
                "generation_time_ms": result["generation_time_ms"],
                "json_valid": result["is_valid"],
                "errors": result["errors"],
                "cached": result.get("cached", False),
            },
            "validation": validation,
        }
```

- [ ] **Step 4: Run tests**

```bash
cd /workspace/Lego-Gen && LEGOGEN_DEV=1 python -m pytest tests/test_two_stage_pipeline.py -v
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/inference/pipeline.py backend/models/unified_model.py tests/test_two_stage_pipeline.py
git commit -m "feat: two-stage inference pipeline with adapter swapping

Stage 1 (image→description) + Stage 2 (text→JSON) with PEFT set_adapter().
Falls back to single-stage if Stage 1 adapter not found."
```

---

## Task 7: Update 3D viewer to use grid_pos

When parts include `grid_pos`, use it for placement instead of auto-packing.

**Files:**
- Modify: `frontend/src/components/LegoViewer.tsx`
- Modify: `frontend/src/api/legogen.ts`

- [ ] **Step 1: Add grid_pos to Part type**

In `frontend/src/api/legogen.ts`, add `grid_pos` to the `Part` interface:

```typescript
export interface Part {
  part_id: string;
  name: string;
  category: string;
  color: string;
  color_hex: string;
  is_trans?: boolean;
  quantity: number;
  grid_pos?: [number, number];  // [x, z] stud coordinates within layer
}
```

- [ ] **Step 2: Update LegoViewer to use grid_pos when available**

In `frontend/src/components/LegoViewer.tsx`, update the `packLayer` function to check for `grid_pos`:

At the start of `packLayer`, after expanding parts by quantity, add:

```typescript
  // If parts have grid_pos, use it for placement instead of auto-packing
  const hasGridPos = parts.some(p => p.gridPos != null);
  if (hasGridPos) {
    for (const p of parts) {
      const [w, h, d] = p.size;
      const gx = p.gridPos?.[0] ?? 0;
      const gz = p.gridPos?.[1] ?? 0;
      for (let q = 0; q < p.quantity; q++) {
        const x = (gx + q * w) * 1.0;  // 1 stud = 1.0 unit
        const z = gz * 1.0;
        bricks.push({
          key: `brick-${idx}`,
          position: [x, baseY + h / 2, z],
          size: [w, h, d],
          color: p.color,
          isTrans: p.isTrans,
          stepNum,
        });
        idx++;
      }
    }
    // Center the layer
    // ... (same centering logic as auto-pack)
    return { bricks, layerHeight: Math.max(...parts.map(p => p.size[1]), BRICK_H), nextIdx: idx };
  }
```

Update the `partsWithSize` mapping in `Scene` to pass `gridPos`:

```typescript
      const partsWithSize = step.parts.map(part => ({
        size: getBrickSize(part),
        color: part.color_hex,
        isTrans: part.is_trans ?? part.color.toLowerCase().includes('trans'),
        quantity: part.quantity,
        gridPos: part.grid_pos,
      }));
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/api/legogen.ts frontend/src/components/LegoViewer.tsx
git commit -m "feat: use grid_pos for 3D viewer brick placement when available"
```

---

## Task 8: Download COCO data and build Stage 1 manifest

Download COCO 2017 annotations and images, then run the Stage 1 dataset builder.

**Files:** None (data download + script execution)

- [ ] **Step 1: Download COCO 2017 annotations**

```bash
mkdir -p /workspace/Lego-Gen/data/coco
cd /workspace/Lego-Gen/data/coco
wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q annotations_trainval2017.zip
rm annotations_trainval2017.zip
```

- [ ] **Step 2: Download COCO 2017 train images**

```bash
cd /workspace/Lego-Gen/data/coco
wget -q http://images.cocodataset.org/zips/train2017.zip
unzip -q train2017.zip
rm train2017.zip
```

Note: This is ~18GB and will take time.

- [ ] **Step 3: Run the Stage 1 dataset builder**

```bash
cd /workspace/Lego-Gen && python -m backend.data_pipeline.build_stage1_dataset
```

Expected output: Manifest saved with ~15-20k samples.

- [ ] **Step 4: Verify manifest**

```bash
python3 -c "import json; d=json.load(open('data/stage1_manifest.json')); print(f'Total: {len(d)}'); print('Sample:', json.dumps(d[0], indent=2))"
```

---

## Task 9: Run Stage 2 training (ST2B-only with structure-aware loss)

Start the retrained Stage 2 from scratch on clean ST2B data.

**Files:** None (training execution)

- [ ] **Step 1: Run grid_pos preprocessing on ST2B labels**

```bash
cd /workspace/Lego-Gen && python -m backend.data_pipeline.add_grid_pos
```

- [ ] **Step 2: Start Stage 2 training**

```bash
cd /workspace/Lego-Gen && nohup python -m backend.training.train_unified \
  --output-dir backend/models/checkpoints/qwen35-27b-lego-stage2-lora \
  --epochs 3 \
  --no-wandb \
  > training_stage2.log 2>&1 &
echo "Stage 2 PID: $!"
```

Monitor with: `tail -f training_stage2.log`

---

## Task 10: Run Stage 1 training

After COCO data is downloaded and manifest is built, train the Stage 1 adapter.

**Files:** None (training execution)

- [ ] **Step 1: Start Stage 1 training**

```bash
cd /workspace/Lego-Gen && nohup python -m backend.training.train_stage1 \
  --manifest data/stage1_manifest.json \
  --no-wandb \
  > training_stage1.log 2>&1 &
echo "Stage 1 PID: $!"
```

Note: Stage 1 and Stage 2 cannot train simultaneously (single GPU). Run Stage 2 first since it takes longer, then Stage 1.

---

## Task 11: Integration test — full two-stage pipeline

After both adapters are trained, test the full pipeline end-to-end.

**Files:** None (testing)

- [ ] **Step 1: Update config to point to new checkpoints**

In `backend/config.py`, update `UNIFIED_CHECKPOINT_DIR` to point to the Stage 2 checkpoint:

```python
UNIFIED_CHECKPOINT_DIR = CHECKPOINT_DIR / "qwen35-27b-lego-stage2-lora"
```

`STAGE1_CHECKPOINT_DIR` is already set correctly.

- [ ] **Step 2: Start the backend**

```bash
cd /workspace/Lego-Gen && LEGOGEN_DEV=0 uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

- [ ] **Step 3: Test text-to-JSON (Stage 2 only)**

```bash
curl -s -X POST http://localhost:8000/api/generate-from-text \
  -F "prompt=a small red chair with four legs" | python3 -m json.tool | head -30
```

Verify: Output uses `layer_0`...`layer_N` naming with `grid_pos` per part.

- [ ] **Step 4: Test image-to-JSON (two-stage)**

```bash
curl -s -X POST http://localhost:8000/api/generate \
  -F "image=@data/images/10343-1.jpg" | python3 -m json.tool | head -30
```

Verify: Output includes both a description and layer-based JSON.

- [ ] **Step 5: Final commit**

```bash
git add backend/config.py
git commit -m "feat: point config to new two-stage checkpoints"
```
