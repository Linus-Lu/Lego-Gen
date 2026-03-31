"""Shared training utilities: metrics, seeding, logging."""

import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
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
