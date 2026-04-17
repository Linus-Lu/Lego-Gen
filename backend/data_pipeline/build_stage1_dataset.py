"""Stage 1 dataset builder: matches COCO images with ST2B captions.

Produces data/stage1_manifest.json — a list of
    {"image_path": str, "description": str, "category": str, "source": "coco"|"rebrickable"}
entries used to fine-tune the Stage 1 image-to-description model.
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Make sure top-level package is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.config import (
    COCO_TO_ST2B_CATEGORY,
    DATA_DIR,
    ST2B_CONVERTED_DIR,
)


# ── ST2B helpers ──────────────────────────────────────────────────────────


def load_st2b_captions_by_category(st2b_dir: Path) -> dict[str, list[dict]]:
    """Load ST2B label files and group them by the object type they describe.

    Matching is done by checking whether each ST2B *target* word (the values
    in ``COCO_TO_ST2B_CATEGORY``) appears as a whole word inside the label's
    ``object`` field text.

    Args:
        st2b_dir: Directory containing ST2B JSON label files.

    Returns:
        Mapping from ST2B object-type string (e.g. "chair") to a list of the
        matching label dicts.
    """
    st2b_dir = Path(st2b_dir)
    by_category: dict[str, list[dict]] = {}

    # Build a set of all target object-type words we care about.
    target_words: set[str] = set(COCO_TO_ST2B_CATEGORY.values())

    for json_file in st2b_dir.glob("*.json"):
        try:
            label = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        object_text: str = label.get("object", "").lower()
        if not object_text:
            continue

        # A label can match multiple target words (e.g. "sofa chair" → both).
        for word in target_words:
            # Simple whole-word containment check.
            if _word_in_text(word.lower(), object_text):
                by_category.setdefault(word, []).append(label)

    return by_category


def _word_in_text(word: str, text: str) -> bool:
    """Return True if *word* appears as a token inside *text*.

    Splits on common delimiters so that e.g. "car" matches "car," or "car."
    but not "cargo".
    """
    import re

    pattern = r"\b" + re.escape(word) + r"\b"
    return bool(re.search(pattern, text))


def match_coco_to_st2b(
    coco_category: str,
    st2b_by_category: dict[str, list[dict]],
    seed: int | None = None,
) -> str | None:
    """Pick a random ST2B caption for a given COCO category name.

    Args:
        coco_category: COCO category name (e.g. "chair", "dining table").
        st2b_by_category: Output of :func:`load_st2b_captions_by_category`.
        seed: Optional RNG seed for reproducibility.

    Returns:
        The ``object`` text of a randomly selected ST2B label, or ``None``
        if the category has no mapping or no matching labels exist.
    """
    st2b_type = COCO_TO_ST2B_CATEGORY.get(coco_category)
    if st2b_type is None:
        return None

    candidates = st2b_by_category.get(st2b_type, [])
    if not candidates:
        return None

    rng = random.Random(seed)
    chosen = rng.choice(candidates)
    return chosen.get("object", "")


# ── Rebrickable helpers ───────────────────────────────────────────────────


def generate_description_from_label(label: dict) -> str:
    """Create a natural-language description from a Rebrickable-style label.

    Combines the ``object`` description with dominant colors and rough size
    information to produce a sentence suitable for Stage 1 training.

    Args:
        label: Label dict with at least ``object``, ``dominant_colors``, and
               ``dimensions_estimate`` fields.

    Returns:
        A short descriptive sentence about the build.
    """
    obj: str = label.get("object", "").rstrip(".")
    colors: list[str] = label.get("dominant_colors", [])
    dims: dict = label.get("dimensions_estimate", {})

    parts: list[str] = []

    if obj:
        parts.append(obj)

    if colors:
        color_str = ", ".join(colors[:3])  # at most three colors
        parts.append(f"primarily in {color_str}")

    if dims:
        width = dims.get("width", "")
        height = dims.get("height", "")
        depth = dims.get("depth", "")
        size_parts = [s for s in [width, height, depth] if s]
        if size_parts:
            parts.append(f"with a {' × '.join(size_parts)} form factor")

    return ", ".join(parts) + "." if parts else ""


# ── COCO helpers ──────────────────────────────────────────────────────────


# Filesystem orchestration — requires the COCO annotations JSON on disk.
def load_coco_annotations(coco_dir: Path) -> list[dict]:  # pragma: no cover
    """Load COCO 2017 train annotations and return filtered image records.

    Filters to images that contain at least one annotation whose category
    name is in ``COCO_TO_ST2B_CATEGORY``.

    Args:
        coco_dir: Path to the COCO dataset root (e.g. ``data/coco``).
                  Expects ``annotations/instances_train2017.json`` inside it.

    Returns:
        List of dicts, each with keys: ``image_id``, ``file_name``,
        ``coco_category``.

    Raises:
        SystemExit: If the annotations file is not found.
    """
    coco_dir = Path(coco_dir)
    ann_file = coco_dir / "annotations" / "instances_train2017.json"

    if not ann_file.exists():
        print(
            f"[build_stage1_dataset] COCO annotations not found at: {ann_file}\n"
            "Download instructions:\n"
            "  wget http://images.cocodataset.org/annotations/"
            "annotations_trainval2017.zip\n"
            "  unzip annotations_trainval2017.zip -d data/coco/\n"
            "  # Images (optional — only needed for actual training):\n"
            "  wget http://images.cocodataset.org/zips/train2017.zip\n"
            "  unzip train2017.zip -d data/coco/",
            file=sys.stderr,
        )
        sys.exit(1)

    with ann_file.open(encoding="utf-8") as fh:
        coco = json.load(fh)

    # Build id → name lookup for categories.
    valid_coco_names: set[str] = set(COCO_TO_ST2B_CATEGORY.keys())
    cat_id_to_name: dict[int, str] = {
        cat["id"]: cat["name"]
        for cat in coco.get("categories", [])
        if cat["name"] in valid_coco_names
    }

    # Build image_id → file_name lookup.
    img_id_to_filename: dict[int, str] = {
        img["id"]: img["file_name"] for img in coco.get("images", [])
    }

    # Collect one representative category per image (first match wins).
    seen: dict[int, str] = {}
    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        if img_id not in seen and cat_id in cat_id_to_name:
            seen[img_id] = cat_id_to_name[cat_id]

    records = []
    for img_id, coco_cat in seen.items():
        file_name = img_id_to_filename.get(img_id)
        if file_name:
            records.append(
                {
                    "image_id": img_id,
                    "file_name": file_name,
                    "coco_category": coco_cat,
                }
            )

    return records


# ── Main builder ──────────────────────────────────────────────────────────


# End-to-end dataset assembly — reads COCO + ST2B + Rebrickable off disk and
# writes the manifest; covered by the full training pipeline, not unit tests.
def build_stage1_manifest(  # pragma: no cover
    coco_dir: Path,
    st2b_dir: Path,
    rebrickable_dir: Path | None,
    output_path: Path,
    max_per_category: int = 2000,
    seed: int = 42,
) -> None:
    """Build and write the Stage 1 training manifest.

    Args:
        coco_dir: COCO dataset root directory.
        st2b_dir: Directory with ST2B JSON label files.
        rebrickable_dir: Optional directory with Rebrickable JSON labels.
        output_path: Where to write the manifest JSON file.
        max_per_category: Maximum number of entries per COCO category.
        seed: RNG seed for reproducible sampling.
    """
    print("[build_stage1_dataset] Loading ST2B captions …")
    st2b_by_category = load_st2b_captions_by_category(st2b_dir)
    print(
        f"[build_stage1_dataset] ST2B categories found: "
        f"{sorted(st2b_by_category.keys())}"
    )

    print("[build_stage1_dataset] Loading COCO annotations …")
    coco_records = load_coco_annotations(coco_dir)
    print(f"[build_stage1_dataset] COCO images after filtering: {len(coco_records)}")

    manifest: list[dict] = []
    rng = random.Random(seed)

    # ── COCO entries ─────────────────────────────────────────────────────
    by_coco_cat: dict[str, list[dict]] = {}
    for rec in coco_records:
        by_coco_cat.setdefault(rec["coco_category"], []).append(rec)

    for coco_cat, records in by_coco_cat.items():
        rng.shuffle(records)
        for rec in records[:max_per_category]:
            description = match_coco_to_st2b(coco_cat, st2b_by_category, seed=rng.randint(0, 2**31))
            if description is None:
                continue
            image_path = str(
                Path(coco_dir) / "train2017" / rec["file_name"]
            )
            manifest.append(
                {
                    "image_path": image_path,
                    "description": description,
                    "category": coco_cat,
                    "source": "coco",
                }
            )

    print(f"[build_stage1_dataset] COCO entries: {len(manifest)}")

    # ── Rebrickable entries ───────────────────────────────────────────────
    rebrickable_count = 0
    if rebrickable_dir is not None:
        rebrickable_dir = Path(rebrickable_dir)
        image_exts = {".jpg", ".jpeg", ".png"}

        for json_file in rebrickable_dir.glob("*.json"):
            try:
                label = json.loads(json_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            # Look for a matching image (same stem, any image extension).
            image_path: Path | None = None
            for ext in image_exts:
                candidate = json_file.with_suffix(ext)
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is None:
                # Try images sub-directory.
                for ext in image_exts:
                    candidate = rebrickable_dir / "images" / (json_file.stem + ext)
                    if candidate.exists():
                        image_path = candidate
                        break

            if image_path is None:
                continue

            description = generate_description_from_label(label)
            if not description:
                continue

            category = label.get("subcategory") or label.get("category") or "unknown"
            manifest.append(
                {
                    "image_path": str(image_path),
                    "description": description,
                    "category": category,
                    "source": "rebrickable",
                }
            )
            rebrickable_count += 1

    print(f"[build_stage1_dataset] Rebrickable entries: {rebrickable_count}")

    # ── Write output ──────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[build_stage1_dataset] Manifest written to {output_path} "
        f"({len(manifest)} entries)"
    )


# ── CLI ───────────────────────────────────────────────────────────────────


# CLI entry point — wraps build_stage1_manifest with argparse; not unit-tested.
def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Build Stage 1 training manifest (COCO + ST2B caption matching)."
    )
    parser.add_argument(
        "--coco-dir",
        type=Path,
        default=DATA_DIR / "coco",
        help="Path to COCO dataset root (default: data/coco)",
    )
    parser.add_argument(
        "--st2b-dir",
        type=Path,
        default=ST2B_CONVERTED_DIR,
        help="Directory with ST2B JSON label files (default: data/st2b_labels)",
    )
    parser.add_argument(
        "--rebrickable-dir",
        type=Path,
        default=None,
        help="Directory with Rebrickable JSON labels + images (optional)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "stage1_manifest.json",
        help="Output manifest path (default: data/stage1_manifest.json)",
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=2000,
        help="Max entries per COCO category (default: 2000)",
    )
    args = parser.parse_args()

    build_stage1_manifest(
        coco_dir=args.coco_dir,
        st2b_dir=args.st2b_dir,
        rebrickable_dir=args.rebrickable_dir,
        output_path=args.output,
        max_per_category=args.max_per_category,
    )


if __name__ == "__main__":
    main()
