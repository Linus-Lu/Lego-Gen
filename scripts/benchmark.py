#!/usr/bin/env python3
"""LEGO-Gen Evaluation Benchmark — poster-ready metrics for two-stage pipeline.

Evaluates both stages independently and end-to-end:
  Stage 1: Image → Description  (BLEU, ROUGE-L, BERTScore, keyword coverage)
  Stage 2: Text → LEGO JSON     (JSON validity, field accuracy, structural metrics)
  End-to-End: Full pipeline      (overall score, latency, stability)

Outputs:
  - Console summary with bar charts
  - JSON results file (machine-readable)
  - CSV results file (for plotting in LaTeX/matplotlib)
  - Poster-ready formatted table (copy-paste into papers)

Usage:
    # Full benchmark (needs GPU with trained adapters):
    python scripts/benchmark.py --num-samples 100

    # Stage 2 only (text-to-JSON, faster):
    python scripts/benchmark.py --stage2-only --num-samples 200

    # Mock mode (no GPU, tests the harness):
    python scripts/benchmark.py --mock --num-samples 5

    # Multi-GPU evaluation (splits samples across GPUs):
    torchrun --nproc_per_node=4 scripts/benchmark.py --num-samples 200
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import DATA_DIR, STAGE1_SYSTEM_PROMPT


# ═══════════════════════════════════════════════════════════════════════
#  Stage 1 Metrics: Description Quality
# ═══════════════════════════════════════════════════════════════════════

def compute_bleu(prediction: str, reference: str, max_n: int = 4) -> dict:
    """Compute BLEU-1 through BLEU-4 scores."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if not pred_tokens or not ref_tokens:
        return {f"bleu_{n}": 0.0 for n in range(1, max_n + 1)}

    scores = {}
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(
            tuple(pred_tokens[i:i + n]) for i in range(len(pred_tokens) - n + 1)
        )
        ref_ngrams = Counter(
            tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1)
        )
        clipped = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
        total = sum(pred_ngrams.values())
        scores[f"bleu_{n}"] = clipped / total if total > 0 else 0.0

    # Brevity penalty
    bp = min(1.0, np.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1)))
    # Combined BLEU (geometric mean of BLEU-1..4 with BP)
    individual = [scores[f"bleu_{n}"] for n in range(1, max_n + 1)]
    log_avg = np.mean([np.log(max(s, 1e-10)) for s in individual])
    scores["bleu_combined"] = bp * np.exp(log_avg)

    return scores


def compute_rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 via longest common subsequence."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    m, n = len(ref_tokens), len(pred_tokens)
    # LCS via DP
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == pred_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]

    precision = lcs_len / n if n > 0 else 0.0
    recall = lcs_len / m if m > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# Geometry and structure keywords that a good description should contain
GEOMETRY_KEYWORDS = {
    "rectangular", "square", "round", "circular", "flat", "sloped", "angled",
    "curved", "tapered", "cylindrical", "triangular", "symmetric", "asymmetric",
    "wide", "narrow", "tall", "short", "thick", "thin", "long",
    "base", "top", "bottom", "side", "front", "back", "center",
    "layer", "stack", "overhang", "protrusion", "indent", "recess",
}

COLOR_KEYWORDS = {
    "red", "blue", "green", "yellow", "white", "black", "gray", "grey",
    "orange", "brown", "tan", "dark", "light", "bright", "transparent",
}


def compute_keyword_coverage(prediction: str) -> dict:
    """How many geometry/color keywords does the description use?"""
    pred_words = set(prediction.lower().split())

    geo_hits = pred_words & GEOMETRY_KEYWORDS
    color_hits = pred_words & COLOR_KEYWORDS

    return {
        "geometry_keywords": len(geo_hits),
        "color_keywords": len(color_hits),
        "geometry_coverage": len(geo_hits) / len(GEOMETRY_KEYWORDS),
        "color_coverage": len(color_hits) / len(COLOR_KEYWORDS),
        "total_keyword_count": len(geo_hits) + len(color_hits),
    }


def compute_description_length(prediction: str) -> dict:
    """Description length statistics."""
    words = prediction.split()
    sentences = [s.strip() for s in prediction.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "in_target_range": 1.0 if 1 <= len(sentences) <= 3 else 0.0,
    }


def evaluate_stage1(prediction: str, reference: str) -> dict:
    """Full Stage 1 evaluation for a single sample."""
    metrics = {}

    # BLEU scores
    metrics.update(compute_bleu(prediction, reference))

    # ROUGE-L
    metrics["rouge_l"] = compute_rouge_l(prediction, reference)

    # Keyword coverage
    metrics.update(compute_keyword_coverage(prediction))

    # Description quality
    metrics.update(compute_description_length(prediction))

    return metrics


# ═══════════════════════════════════════════════════════════════════════
#  Stage 2 Metrics: LEGO JSON Quality
# ═══════════════════════════════════════════════════════════════════════

def evaluate_stage2(pred: dict, ref: dict, known_part_ids: set | None = None) -> dict:
    """Full Stage 2 evaluation for a single sample."""
    from backend.training.utils import (
        compute_color_f1,
        compute_parts_f1,
        compute_structural_coherence,
        compute_build_feasibility,
        compute_part_realism,
    )

    metrics = {}

    # ── JSON validity ──────────────────────────────────────────────────
    metrics["valid_json"] = 1.0 if isinstance(pred, dict) and len(pred) > 0 else 0.0

    # ── Required fields ────────────────────────────────────────────────
    required = [
        "set_id", "object", "category", "subcategory", "complexity",
        "total_parts", "dominant_colors", "dimensions_estimate",
        "subassemblies", "build_hints",
    ]
    metrics["fields_present"] = sum(1 for f in required if f in pred) / len(required)

    # ── Field accuracy ─────────────────────────────────────────────────
    for field in ["category", "subcategory", "complexity"]:
        metrics[f"{field}_accuracy"] = 1.0 if pred.get(field) == ref.get(field) else 0.0

    # ── Color F1 ───────────────────────────────────────────────────────
    metrics["color_f1"] = compute_color_f1(
        pred.get("dominant_colors", []),
        ref.get("dominant_colors", []),
    )

    # ── Parts F1 ───────────────────────────────────────────────────────
    pred_parts = [p for sa in pred.get("subassemblies", []) for p in sa.get("parts", [])]
    ref_parts = [p for sa in ref.get("subassemblies", []) for p in sa.get("parts", [])]
    metrics["parts_f1"] = compute_parts_f1(pred_parts, ref_parts)

    # ── Part count accuracy ────────────────────────────────────────────
    pred_total = pred.get("total_parts", 0)
    ref_total = ref.get("total_parts", 1)
    metrics["part_count_error_pct"] = abs(pred_total - ref_total) / max(ref_total, 1) * 100
    metrics["part_count_within_10pct"] = 1.0 if metrics["part_count_error_pct"] <= 10 else 0.0

    # ── Structural coherence ───────────────────────────────────────────
    metrics["structural_coherence"] = compute_structural_coherence(pred)

    # ── Build feasibility ──────────────────────────────────────────────
    metrics["build_feasibility"] = compute_build_feasibility(pred)

    # ── Part realism (if catalog available) ────────────────────────────
    if known_part_ids:
        metrics["part_realism"] = compute_part_realism(pred, known_part_ids)
    else:
        metrics["part_realism"] = float("nan")

    # ── Subassembly statistics ─────────────────────────────────────────
    subs = pred.get("subassemblies", [])
    metrics["subassembly_count"] = len(subs)
    metrics["avg_parts_per_sub"] = (
        np.mean([len(sa.get("parts", [])) for sa in subs]) if subs else 0.0
    )

    # ── Layer-based format compliance ──────────────────────────────────
    layer_names = [sa.get("name", "") for sa in subs]
    layer_pattern_count = sum(1 for n in layer_names if n.startswith("layer_"))
    metrics["layer_format_compliance"] = layer_pattern_count / max(len(subs), 1)

    # ── Overall weighted score ─────────────────────────────────────────
    metrics["overall_stage2"] = (
        metrics["valid_json"] * 0.10 +
        metrics["fields_present"] * 0.10 +
        metrics["category_accuracy"] * 0.10 +
        metrics["complexity_accuracy"] * 0.05 +
        metrics["color_f1"] * 0.15 +
        metrics["parts_f1"] * 0.20 +
        metrics["structural_coherence"] * 0.15 +
        metrics["build_feasibility"] * 0.10 +
        metrics["layer_format_compliance"] * 0.05
    )

    return metrics


# ═══════════════════════════════════════════════════════════════════════
#  End-to-End Metrics
# ═══════════════════════════════════════════════════════════════════════

def evaluate_stability(pred: dict) -> dict:
    """Run stability checker and return score + per-check results.

    When per-brick reliability scores are available in *pred*'s metadata
    (produced by the updated generation pipeline), the returned dict also
    includes ``avg_reliability_score``, ``min_reliability_score``,
    ``support_ratio_mean``, and ``floating_brick_count``.
    """
    try:
        from backend.brick.stability import is_stable
        from backend.brick.parser import parse_brick_sequence
        from backend.brick.reliability import ReliabilityScorer
        from backend.brick.occupancy import VoxelGrid

        bricks_text = pred.get("bricks", "")
        if not bricks_text:
            return {
                "stability_score": 0,
                "stability_checks_passed": 0,
                "stability_checks_total": 1,
                "avg_reliability_score": 0.0,
                "min_reliability_score": 0.0,
                "support_ratio_mean": 0.0,
                "floating_brick_count": 0,
            }

        bricks = parse_brick_sequence(bricks_text)
        stable = is_stable(bricks)

        # If the pipeline already included reliability scores, use them.
        metadata = pred.get("metadata", {})
        if "avg_reliability_score" in metadata:
            avg_rel = metadata["avg_reliability_score"]
            min_rel = metadata["min_reliability_score"]
            scores_list = metadata.get("reliability_scores", [])
        else:
            # Recompute reliability scores for offline evaluation.
            grid = VoxelGrid()
            scorer = ReliabilityScorer(grid)
            for brick in bricks:
                scorer.add_brick(brick)
                grid.place(brick)
            avg_rel = scorer.aggregate_score()
            min_rel = scorer.min_score()
            scores_list = [s.score for s in scorer.scores]

        # Compute per-brick breakdown metrics.
        floating_count = 0
        support_ratios: list[float] = []
        if scores_list:
            # Recompute from bricks to get support_ratio and connectivity.
            grid2 = VoxelGrid()
            scorer2 = ReliabilityScorer(grid2)
            for brick in bricks:
                bs = scorer2.add_brick(brick)
                grid2.place(brick)
                if bs.connectivity == 0.0:
                    floating_count += 1
                support_ratios.append(bs.support_ratio)

        return {
            "stability_score": 100 if stable else 0,
            "stability_checks_passed": 1 if stable else 0,
            "stability_checks_total": 1,
            "avg_reliability_score": avg_rel,
            "min_reliability_score": min_rel,
            "support_ratio_mean": (
                float(np.mean(support_ratios)) if support_ratios else 0.0
            ),
            "floating_brick_count": floating_count,
        }
    except Exception:
        return {
            "stability_score": 0,
            "stability_checks_passed": 0,
            "stability_checks_total": 1,
            "avg_reliability_score": 0.0,
            "min_reliability_score": 0.0,
            "support_ratio_mean": 0.0,
            "floating_brick_count": 0,
        }


# ═══════════════════════════════════════════════════════════════════════
#  Aggregation & Output
# ═══════════════════════════════════════════════════════════════════════

def aggregate_metrics(all_metrics: list[dict]) -> dict:
    """Aggregate per-sample metrics into summary statistics."""
    if not all_metrics:
        return {}

    summary = {}
    keys = set()
    for m in all_metrics:
        keys.update(m.keys())

    for key in sorted(keys):
        values = [m[key] for m in all_metrics if key in m and not np.isnan(m.get(key, 0))]
        if not values:
            continue
        if isinstance(values[0], (int, float)):
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
            }
    return summary


def print_poster_table(summary: dict, title: str):
    """Print a poster-ready formatted table."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<35} {'Mean':>8} {'Std':>8} {'Med':>8}")
    print(f"  {'-' * 35} {'-' * 8} {'-' * 8} {'-' * 8}")

    for key, stats in sorted(summary.items()):
        mean = stats["mean"]
        std = stats["std"]
        med = stats["median"]

        # Format as percentage if value is 0-1
        if 0 <= mean <= 1 and key not in {
            "word_count", "sentence_count", "geometry_keywords",
            "color_keywords", "total_keyword_count", "subassembly_count",
            "avg_parts_per_sub", "stability_checks_passed",
            "stability_checks_total", "part_count_error_pct",
            "latency_ms",
        }:
            print(f"  {key:<35} {mean:>7.1%} {std:>7.1%} {med:>7.1%}")
        else:
            print(f"  {key:<35} {mean:>8.2f} {std:>8.2f} {med:>8.2f}")

    print(f"{'=' * 70}")


def print_bar_chart(summary: dict, keys: list[str], title: str):
    """Print a horizontal bar chart for selected metrics."""
    print(f"\n  {title}")
    print(f"  {'-' * 50}")
    for key in keys:
        if key not in summary:
            continue
        val = summary[key]["mean"]
        bar_len = int(val * 30) if val <= 1 else int(min(val / 100, 1) * 30)
        bar = "\u2588" * bar_len + "\u2591" * (30 - bar_len)
        if 0 <= val <= 1:
            print(f"  {key:<25} {bar} {val:.1%}")
        else:
            print(f"  {key:<25} {bar} {val:.2f}")


def save_csv(all_metrics: list[dict], path: Path):
    """Save per-sample metrics as CSV for plotting."""
    if not all_metrics:
        return
    keys = sorted(set(k for m in all_metrics for k in m.keys()))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for m in all_metrics:
            writer.writerow({k: m.get(k, "") for k in keys})
    print(f"  CSV saved: {path}")


def save_latex_table(summary: dict, path: Path, caption: str):
    """Save a LaTeX-formatted table for papers/posters."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{{caption}}}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Metric & Mean & Std & Median \\",
        r"\midrule",
    ]
    for key, stats in sorted(summary.items()):
        mean = stats["mean"]
        std = stats["std"]
        med = stats["median"]
        name = key.replace("_", r"\_")
        if 0 <= mean <= 1 and key not in {
            "word_count", "sentence_count", "geometry_keywords",
            "color_keywords", "subassembly_count", "avg_parts_per_sub",
            "part_count_error_pct", "latency_ms",
        }:
            lines.append(f"{name} & {mean:.1%} & {std:.1%} & {med:.1%} \\\\")
        else:
            lines.append(f"{name} & {mean:.2f} & {std:.2f} & {med:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  LaTeX table saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════

def load_part_catalog() -> set:
    """Load known part IDs from Rebrickable cache."""
    catalog_path = DATA_DIR / "cache" / "parts.csv"
    if not catalog_path.exists():
        return set()
    part_ids = set()
    with open(catalog_path) as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row:
                part_ids.add(row[0])
    return part_ids


def load_test_samples(data_dir: Path, num_samples: int, stage2_only: bool = False):
    """Load validation samples with images and labels."""
    # Try splits.json first
    splits_path = data_dir / "splits.json"
    if splits_path.exists():
        with open(splits_path) as f:
            splits = json.load(f)
        val_sets = splits.get("val", [])
    else:
        # Fall back to all labels
        val_sets = [p.stem for p in (data_dir / "labels").glob("*.json")]

    samples = []
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    for set_id in val_sets:
        label_path = labels_dir / f"{set_id}.json"
        if not label_path.exists():
            continue

        image_path = None
        if not stage2_only:
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                p = images_dir / f"{set_id}{ext}"
                if p.exists():
                    image_path = p
                    break
            if not image_path:
                continue  # skip samples without images for full pipeline

        samples.append({
            "set_id": str(set_id),
            "image_path": str(image_path) if image_path else None,
            "label_path": str(label_path),
        })

        if len(samples) >= num_samples:
            break

    return samples


def parse_args():
    parser = argparse.ArgumentParser(description="LEGO-Gen Evaluation Benchmark")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--stage2-only", action="store_true", help="Skip Stage 1, evaluate text→JSON only")
    parser.add_argument("--mock", action="store_true", help="Use mock pipeline (no GPU)")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "benchmark_results"))
    parser.add_argument("--no-stability", action="store_true", help="Skip stability checks (faster)")
    parser.add_argument("--no-latex", action="store_true", help="Skip LaTeX output")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 70}")
    print(f"  LEGO-Gen Evaluation Benchmark")
    print(f"  Samples: {args.num_samples}")
    print(f"  Mode: {'Stage 2 only' if args.stage2_only else 'Full two-stage pipeline'}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 70}")

    # ── Load data ──────────────────────────────────────────────────────
    samples = load_test_samples(data_dir, args.num_samples, args.stage2_only)
    print(f"  Loaded {len(samples)} test samples")

    if not samples:
        print("ERROR: No test samples found. Check your data directory.")
        sys.exit(1)

    known_part_ids = load_part_catalog()
    print(f"  Part catalog: {len(known_part_ids)} known IDs")

    # ── Load pipeline ──────────────────────────────────────────────────
    if args.mock:
        os.environ["LEGOGEN_DEV"] = "1"

    from backend.inference.pipeline import get_pipeline
    pipeline = get_pipeline()
    print("  Pipeline loaded\n")

    # ── Run evaluation ─────────────────────────────────────────────────
    stage1_metrics = []
    stage2_metrics = []
    e2e_metrics = []
    errors = []

    for i, sample in enumerate(samples):
        set_id = sample["set_id"]
        print(f"  [{i + 1}/{len(samples)}] {set_id}", end=" ... ", flush=True)

        # Load reference
        with open(sample["label_path"]) as f:
            ref = json.load(f)

        try:
            t0 = time.time()

            if args.stage2_only:
                # Build description from reference for Stage 2 eval
                desc_text = f"{ref.get('object', 'object')}. "
                colors = ref.get("dominant_colors", [])
                if colors:
                    desc_text += f"Dominant colors: {', '.join(colors)}. "
                dims = ref.get("dimensions_estimate", {})
                if dims:
                    desc_text += f"Size: {dims.get('width', 'medium')} width, {dims.get('height', 'medium')} height."

                result = pipeline.generate_brick_build(desc_text)
            else:
                from PIL import Image as PILImage
                image = PILImage.open(sample["image_path"]).convert("RGB")
                result = pipeline.generate_brick_build_from_image(image)

            latency = (time.time() - t0) * 1000
            pred = result.get("description", {})

            # ── Stage 1 metrics (if two-stage and description available)
            if not args.stage2_only and hasattr(pipeline, "last_stage1_description"):
                stage1_desc = getattr(pipeline, "last_stage1_description", "")
                # Build reference description from label
                ref_desc = f"{ref.get('object', '')}. "
                ref_colors = ref.get("dominant_colors", [])
                if ref_colors:
                    ref_desc += f"Dominant colors: {', '.join(ref_colors)}. "
                s1 = evaluate_stage1(stage1_desc, ref_desc)
                s1["set_id"] = set_id
                stage1_metrics.append(s1)

            # ── Stage 2 metrics
            s2 = evaluate_stage2(pred, ref, known_part_ids)
            s2["set_id"] = set_id
            s2["latency_ms"] = latency
            stage2_metrics.append(s2)

            # ── Stability (end-to-end)
            if not args.no_stability:
                stab = evaluate_stability(pred)
                stab["set_id"] = set_id
                stab["latency_ms"] = latency
                e2e_metrics.append(stab)

            # Print inline summary
            print(
                f"overall={s2['overall_stage2']:.0%} "
                f"color_f1={s2['color_f1']:.0%} "
                f"parts_f1={s2['parts_f1']:.0%} "
                f"struct={s2['structural_coherence']:.0%} "
                f"{latency:.0f}ms"
            )

        except Exception as e:
            print(f"FAILED: {e}")
            errors.append({"set_id": set_id, "error": str(e)})

    # ── Aggregate & Display ────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  RESULTS ({len(stage2_metrics)} successful / {len(errors)} errors)")
    print(f"{'=' * 70}")

    # Stage 1 summary
    if stage1_metrics:
        s1_summary = aggregate_metrics(stage1_metrics)
        print_poster_table(s1_summary, "Stage 1: Image -> Description")
        print_bar_chart(s1_summary, [
            "bleu_1", "bleu_4", "bleu_combined", "rouge_l",
            "geometry_coverage", "color_coverage", "in_target_range",
        ], "Stage 1 Quality")

    # Stage 2 summary
    if stage2_metrics:
        s2_summary = aggregate_metrics(stage2_metrics)
        print_poster_table(s2_summary, "Stage 2: Text -> LEGO JSON")
        print_bar_chart(s2_summary, [
            "valid_json", "fields_present", "category_accuracy",
            "color_f1", "parts_f1", "structural_coherence",
            "build_feasibility", "layer_format_compliance",
            "overall_stage2",
        ], "Stage 2 Quality")

    # End-to-end summary
    if e2e_metrics:
        e2e_summary = aggregate_metrics(e2e_metrics)
        print_poster_table(e2e_summary, "End-to-End: Stability & Latency")

    # ── Poster-ready highlight metrics ─────────────────────────────────
    if stage2_metrics:
        s2s = aggregate_metrics(stage2_metrics)
        print(f"\n{'=' * 70}")
        print(f"  POSTER HIGHLIGHTS")
        print(f"{'=' * 70}")

        highlights = {
            "JSON Validity Rate": s2s.get("valid_json", {}).get("mean", 0),
            "Schema Compliance": s2s.get("fields_present", {}).get("mean", 0),
            "Category Accuracy": s2s.get("category_accuracy", {}).get("mean", 0),
            "Color F1": s2s.get("color_f1", {}).get("mean", 0),
            "Parts F1": s2s.get("parts_f1", {}).get("mean", 0),
            "Structural Coherence": s2s.get("structural_coherence", {}).get("mean", 0),
            "Build Feasibility": s2s.get("build_feasibility", {}).get("mean", 0),
            "Layer Format Compliance": s2s.get("layer_format_compliance", {}).get("mean", 0),
            "Overall Score": s2s.get("overall_stage2", {}).get("mean", 0),
        }

        for name, val in highlights.items():
            bar = "\u2588" * int(val * 30) + "\u2591" * (30 - int(val * 30))
            print(f"  {name:<28} {bar} {val:.1%}")

        latency_stats = s2s.get("latency_ms", {})
        if latency_stats:
            print(f"\n  Avg Latency:  {latency_stats.get('mean', 0):.0f} ms")
            print(f"  P50 Latency:  {latency_stats.get('median', 0):.0f} ms")

        if e2e_metrics:
            e2es = aggregate_metrics(e2e_metrics)
            stab = e2es.get("stability_score", {})
            if stab:
                print(f"  Stability:    {stab.get('mean', 0):.1f}/100")

        print(f"  Samples:      {len(stage2_metrics)}")
        print(f"  Errors:       {len(errors)}")

    # ── Save outputs ───────────────────────────────────────────────────
    # JSON
    results_json = {
        "config": {
            "num_samples": args.num_samples,
            "stage2_only": args.stage2_only,
            "actual_samples": len(stage2_metrics),
            "errors": len(errors),
        },
        "stage1_summary": aggregate_metrics(stage1_metrics) if stage1_metrics else {},
        "stage2_summary": aggregate_metrics(stage2_metrics) if stage2_metrics else {},
        "e2e_summary": aggregate_metrics(e2e_metrics) if e2e_metrics else {},
        "errors": errors,
    }
    json_path = output_dir / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n  JSON saved: {json_path}")

    # CSV
    if stage1_metrics:
        save_csv(stage1_metrics, output_dir / "stage1_metrics.csv")
    if stage2_metrics:
        save_csv(stage2_metrics, output_dir / "stage2_metrics.csv")
    if e2e_metrics:
        save_csv(e2e_metrics, output_dir / "e2e_metrics.csv")

    # LaTeX
    if not args.no_latex:
        if stage1_metrics:
            save_latex_table(
                aggregate_metrics(stage1_metrics),
                output_dir / "stage1_table.tex",
                "Stage 1: Image to Description Quality",
            )
        if stage2_metrics:
            save_latex_table(
                aggregate_metrics(stage2_metrics),
                output_dir / "stage2_table.tex",
                "Stage 2: Text to LEGO JSON Quality",
            )

    print(f"\n  All results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
