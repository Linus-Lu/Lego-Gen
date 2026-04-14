"""Shared training utilities: metrics, seeding, logging."""

import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_wandb(project_name: str = "legogen", config: dict | None = None):
    """Initialize Weights & Biases logging."""
    try:
        import wandb
        wandb.init(project=project_name, config=config or {})
        return True
    except ImportError:
        print("wandb not installed, skipping logging")
        return False


# ── Metrics ────────────────────────────────────────────────────────────

def compute_json_validity_rate(predictions: list[str]) -> float:
    """What fraction of predictions parse as valid JSON?"""
    valid = 0
    for pred in predictions:
        try:
            # Try to extract JSON from the string
            start = pred.find("{")
            end = pred.rfind("}") + 1
            if start >= 0 and end > start:
                json.loads(pred[start:end])
                valid += 1
        except (json.JSONDecodeError, ValueError):
            pass
    return valid / len(predictions) if predictions else 0.0


def compute_field_accuracy(
    predictions: list[dict], references: list[dict], field: str
) -> float:
    """Exact match accuracy for a specific field."""
    correct = 0
    total = 0
    for pred, ref in zip(predictions, references):
        if field in ref:
            total += 1
            if pred.get(field) == ref.get(field):
                correct += 1
    return correct / total if total > 0 else 0.0


def compute_color_f1(
    pred_colors: list[str], ref_colors: list[str]
) -> float:
    """F1 score between predicted and reference color lists."""
    if not ref_colors:
        return 1.0 if not pred_colors else 0.0
    if not pred_colors:
        return 0.0

    pred_set = set(c.lower() for c in pred_colors)
    ref_set = set(c.lower() for c in ref_colors)

    tp = len(pred_set & ref_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(ref_set) if ref_set else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_parts_f1(pred_parts: list[dict], ref_parts: list[dict]) -> float:
    """F1 score on predicted vs actual part IDs (with quantities)."""
    def to_counter(parts):
        c = Counter()
        for p in parts:
            c[p.get("part_id", "")] += p.get("quantity", 1)
        return c

    pred_c = to_counter(pred_parts)
    ref_c = to_counter(ref_parts)

    # Intersection = min of each
    tp = sum((pred_c & ref_c).values())
    pred_total = sum(pred_c.values())
    ref_total = sum(ref_c.values())

    precision = tp / pred_total if pred_total > 0 else 0.0
    recall = tp / ref_total if ref_total > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_all_metrics(
    pred_dicts: list[dict], ref_dicts: list[dict]
) -> dict:
    """Compute all evaluation metrics."""
    metrics = {}

    # Field-level accuracy
    for field in ["category", "subcategory", "complexity"]:
        metrics[f"{field}_accuracy"] = compute_field_accuracy(
            pred_dicts, ref_dicts, field
        )

    # Color F1 (averaged)
    color_f1s = []
    for pred, ref in zip(pred_dicts, ref_dicts):
        f1 = compute_color_f1(
            pred.get("dominant_colors", []),
            ref.get("dominant_colors", []),
        )
        color_f1s.append(f1)
    metrics["color_f1"] = np.mean(color_f1s) if color_f1s else 0.0

    # Parts F1 (averaged over subassemblies)
    parts_f1s = []
    for pred, ref in zip(pred_dicts, ref_dicts):
        pred_all_parts = []
        ref_all_parts = []
        for sa in pred.get("subassemblies", []):
            pred_all_parts.extend(sa.get("parts", []))
        for sa in ref.get("subassemblies", []):
            ref_all_parts.extend(sa.get("parts", []))
        parts_f1s.append(compute_parts_f1(pred_all_parts, ref_all_parts))
    metrics["parts_f1"] = np.mean(parts_f1s) if parts_f1s else 0.0

    return metrics


# ── Planner-specific metrics ──────────────────────────────────────────

POSITION_ORDER = {
    "bottom": 0, "left": 1, "right": 1, "front": 1, "back": 1,
    "center": 1, "top": 2,
}


def compute_structural_coherence(pred: dict) -> float:
    """Score structural coherence: bottom-to-top ordering + valid connects_to refs."""
    subs = pred.get("subassemblies", [])
    if not subs:
        return 0.0

    score = 0.0
    checks = 0

    # Check bottom-to-top ordering
    positions = [sa.get("spatial", {}).get("position", "center") for sa in subs]
    orders = [POSITION_ORDER.get(p, 1) for p in positions]
    if orders == sorted(orders):
        score += 1.0
    checks += 1

    # Check at least one bottom subassembly
    if "bottom" in positions:
        score += 1.0
    checks += 1

    # Check connects_to references are valid subassembly names
    valid_names = {sa.get("name", "") for sa in subs}
    for sa in subs:
        connects = sa.get("spatial", {}).get("connects_to", [])
        checks += 1
        if all(c in valid_names for c in connects):
            score += 1.0

    return score / checks if checks > 0 else 0.0


def compute_part_realism(pred: dict, known_part_ids: set[str]) -> float:
    """Fraction of predicted part_ids that exist in the Rebrickable catalog."""
    if not known_part_ids:
        return 0.0

    total = 0
    valid = 0
    for sa in pred.get("subassemblies", []):
        for part in sa.get("parts", []):
            total += 1
            if part.get("part_id", "") in known_part_ids:
                valid += 1

    return valid / total if total > 0 else 0.0


def compute_build_feasibility(pred: dict) -> float:
    """Score build feasibility: parts count consistency, reasonable subassembly count."""
    score = 0.0
    checks = 0

    # Check total_parts matches sum of quantities
    claimed_total = pred.get("total_parts", 0)
    actual_total = sum(
        p.get("quantity", 1)
        for sa in pred.get("subassemblies", [])
        for p in sa.get("parts", [])
    )
    checks += 1
    if claimed_total > 0 and abs(claimed_total - actual_total) <= max(2, claimed_total * 0.1):
        score += 1.0

    # Check reasonable subassembly count
    num_subs = len(pred.get("subassemblies", []))
    checks += 1
    if 1 <= num_subs <= 10:
        score += 1.0

    # Check each subassembly has at least one part
    checks += 1
    if num_subs > 0 and all(
        len(sa.get("parts", [])) > 0 for sa in pred.get("subassemblies", [])
    ):
        score += 1.0

    return score / checks if checks > 0 else 0.0
