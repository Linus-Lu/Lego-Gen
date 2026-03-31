#!/usr/bin/env python3
"""Test the trained model against ground-truth labels from the val set.

Usage:
    # On Colab with real model:
    python scripts/test_inference.py --num-samples 20

    # Quick check with mock pipeline (no GPU needed):
    LEGOGEN_DEV=1 python scripts/test_inference.py --mock
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import DATA_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=10, help="Number of val samples to test")
    parser.add_argument("--mock", action="store_true", help="Use mock pipeline (no GPU)")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--output", type=str, default="test_results.json", help="Save results to JSON")
    return parser.parse_args()


def score_prediction(pred: dict, ref: dict) -> dict:
    """Score a single prediction against the ground truth label."""
    scores = {}

    # ── JSON validity ──────────────────────────────────────────────────
    scores["valid_json"] = isinstance(pred, dict) and len(pred) > 0

    # ── Required fields present ────────────────────────────────────────
    required = ["set_id", "object", "category", "complexity",
                "total_parts", "dominant_colors", "subassemblies", "build_hints"]
    scores["fields_present"] = sum(1 for f in required if f in pred) / len(required)

    # ── Exact field matches ────────────────────────────────────────────
    scores["category_correct"] = pred.get("category") == ref.get("category")
    scores["complexity_correct"] = pred.get("complexity") == ref.get("complexity")
    scores["object_name_match"] = _name_similarity(
        pred.get("object", ""), ref.get("object", "")
    )

    # ── Part count accuracy ────────────────────────────────────────────
    pred_parts = pred.get("total_parts", 0)
    ref_parts = ref.get("total_parts", 1)
    scores["part_count_error_pct"] = abs(pred_parts - ref_parts) / max(ref_parts, 1) * 100

    # ── Color overlap ──────────────────────────────────────────────────
    pred_colors = set(c.lower() for c in pred.get("dominant_colors", []))
    ref_colors = set(c.lower() for c in ref.get("dominant_colors", []))
    if ref_colors:
        scores["color_overlap"] = len(pred_colors & ref_colors) / len(ref_colors)
    else:
        scores["color_overlap"] = 1.0

    # ── Subassembly count ──────────────────────────────────────────────
    pred_sa = len(pred.get("subassemblies", []))
    ref_sa = len(ref.get("subassemblies", []))
    scores["subassembly_count_error"] = abs(pred_sa - ref_sa)

    # ── Part ID overlap (across all subassemblies) ─────────────────────
    pred_part_ids = set()
    ref_part_ids = set()
    for sa in pred.get("subassemblies", []):
        for p in sa.get("parts", []):
            if p.get("part_id"):
                pred_part_ids.add(p["part_id"])
    for sa in ref.get("subassemblies", []):
        for p in sa.get("parts", []):
            if p.get("part_id"):
                ref_part_ids.add(p["part_id"])

    if ref_part_ids:
        scores["part_id_precision"] = len(pred_part_ids & ref_part_ids) / max(len(pred_part_ids), 1)
        scores["part_id_recall"] = len(pred_part_ids & ref_part_ids) / len(ref_part_ids)
    else:
        scores["part_id_precision"] = 0.0
        scores["part_id_recall"] = 0.0

    # ── Overall score (weighted) ───────────────────────────────────────
    scores["overall"] = (
        scores["valid_json"] * 0.2 +
        scores["fields_present"] * 0.15 +
        scores["category_correct"] * 0.15 +
        scores["complexity_correct"] * 0.10 +
        scores["color_overlap"] * 0.15 +
        scores["part_id_recall"] * 0.25
    )

    return scores


def _name_similarity(a: str, b: str) -> float:
    """Simple word overlap similarity."""
    if not a or not b:
        return 0.0
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    overlap = len(words_a & words_b)
    return overlap / max(len(words_b), 1)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    # ── Load val split ─────────────────────────────────────────────────
    splits_path = data_dir / "splits.json"
    if not splits_path.exists():
        print("ERROR: No splits.json found. Run prepare_dataset.py first.")
        sys.exit(1)

    with open(splits_path) as f:
        splits = json.load(f)

    val_sets = splits.get("val", [])
    if not val_sets:
        print("ERROR: No validation sets found in splits.json")
        sys.exit(1)

    # Filter to sets that have both image and label
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    test_samples = []
    for set_num in val_sets:
        label_path = labels_dir / f"{set_num}.json"
        if not label_path.exists():
            continue
        image_path = None
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = images_dir / f"{set_num}{ext}"
            if p.exists():
                image_path = p
                break
        if image_path:
            test_samples.append((set_num, image_path, label_path))

    test_samples = test_samples[:args.num_samples]
    print(f"Testing {len(test_samples)} samples from val set")

    # ── Load pipeline ──────────────────────────────────────────────────
    if args.mock:
        import os
        os.environ["LEGOGEN_DEV"] = "1"

    from backend.inference.pipeline import get_pipeline
    pipeline = get_pipeline()
    print("Pipeline ready\n")

    # ── Run inference ──────────────────────────────────────────────────
    from PIL import Image as PILImage

    results = []
    all_scores = []

    for i, (set_num, image_path, label_path) in enumerate(test_samples):
        print(f"[{i+1}/{len(test_samples)}] {set_num}", end=" ... ", flush=True)

        # Load ground truth
        with open(label_path) as f:
            ref = json.load(f)

        # Run inference
        try:
            image = PILImage.open(image_path).convert("RGB")
            t0 = time.time()
            result = pipeline.generate_build(image)
            elapsed = int((time.time() - t0) * 1000)
            pred = result["description"]
        except Exception as e:
            print(f"FAILED ({e})")
            results.append({"set_num": set_num, "error": str(e)})
            continue

        # Score it
        scores = score_prediction(pred, ref)
        all_scores.append(scores)

        print(f"overall={scores['overall']:.2f} | "
              f"cat={'✓' if scores['category_correct'] else '✗'} | "
              f"complexity={'✓' if scores['complexity_correct'] else '✗'} | "
              f"colors={scores['color_overlap']:.0%} | "
              f"parts_recall={scores['part_id_recall']:.0%} | "
              f"{elapsed}ms")

        results.append({
            "set_num": set_num,
            "ref_name": ref.get("object"),
            "pred_name": pred.get("object"),
            "scores": scores,
            "pred": pred,
            "ref": ref,
        })

    # ── Aggregate ──────────────────────────────────────────────────────
    if all_scores:
        print("\n" + "=" * 60)
        print("AGGREGATE RESULTS")
        print("=" * 60)
        metrics = ["valid_json", "fields_present", "category_correct",
                   "complexity_correct", "color_overlap",
                   "part_id_precision", "part_id_recall", "overall"]
        for m in metrics:
            avg = sum(s.get(m, 0) for s in all_scores) / len(all_scores)
            bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
            print(f"  {m:<25} {bar} {avg:.1%}")

        avg_part_err = sum(s.get("part_count_error_pct", 0) for s in all_scores) / len(all_scores)
        print(f"\n  Avg part count error: {avg_part_err:.1f}%")
        print(f"  Samples tested:       {len(all_scores)}")

    # ── Save results ───────────────────────────────────────────────────
    output_path = PROJECT_ROOT / args.output
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
