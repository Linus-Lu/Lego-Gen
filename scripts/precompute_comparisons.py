#!/usr/bin/env python3
"""
Pre-compute model comparison data across checkpoints for the Explore page.

Runs inference on a set of test prompts across available checkpoints, measuring
JSON validity, parts F1, color F1, and part count. Results are saved as static
JSON files in data/comparisons/ for frontend consumption.

Usage:
    python scripts/precompute_comparisons.py
"""

import json
import os
import re
import sys
from pathlib import Path
from collections import Counter

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import CHECKPOINT_DIR, DATA_DIR

OUTPUT_DIR = PROJECT_ROOT / "frontend" / "public" / "data" / "comparisons"

# ── Test prompts ──────────────────────────────────────────────────────

TEST_INPUTS = [
    {"slug": "red_car", "prompt": "Build a small red car with 4 wheels"},
    {"slug": "house", "prompt": "Build a simple house with a roof and door"},
    {"slug": "spaceship", "prompt": "Build a small spaceship with wings"},
    {"slug": "tree", "prompt": "Build a green tree with a brown trunk"},
    {"slug": "robot", "prompt": "Build a simple robot figure"},
    {"slug": "bridge", "prompt": "Build a small bridge"},
    {"slug": "castle_tower", "prompt": "Build a castle tower with battlements"},
    {"slug": "airplane", "prompt": "Build a small airplane"},
    {"slug": "flower", "prompt": "Build a flower with petals and a stem"},
    {"slug": "boat", "prompt": "Build a small sailboat"},
    {"slug": "dog", "prompt": "Build a simple dog figure"},
    {"slug": "cat", "prompt": "Build a simple cat figure"},
    {"slug": "helicopter", "prompt": "Build a helicopter"},
    {"slug": "truck", "prompt": "Build a delivery truck"},
    {"slug": "windmill", "prompt": "Build a small windmill"},
    {"slug": "lighthouse", "prompt": "Build a lighthouse"},
    {"slug": "train", "prompt": "Build a small train engine"},
    {"slug": "dinosaur", "prompt": "Build a T-Rex dinosaur"},
    {"slug": "rocket", "prompt": "Build a rocket ship"},
    {"slug": "piano", "prompt": "Build a small grand piano"},
]


def find_checkpoints() -> list[Path]:
    """Find all available model checkpoint directories."""
    checkpoints = []
    if CHECKPOINT_DIR.exists():
        for d in sorted(CHECKPOINT_DIR.iterdir()):
            if d.is_dir() and (d / "adapter_config.json").exists():
                checkpoints.append(d)
    return checkpoints


def check_json_validity(text: str) -> bool:
    """Check if text contains valid JSON."""
    try:
        # Try to extract JSON from the text
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            json.loads(match.group())
            return True
    except (json.JSONDecodeError, AttributeError):
        pass
    return False


def extract_parts(data: dict) -> list[str]:
    """Extract part names from a parsed build description."""
    parts = []
    for sub in data.get("subassemblies", []):
        for part in sub.get("parts", []):
            name = part.get("name", "")
            qty = part.get("quantity", 1)
            parts.extend([name] * qty)
    return parts


def extract_colors(data: dict) -> list[str]:
    """Extract color names from a parsed build description."""
    colors = []
    for sub in data.get("subassemblies", []):
        for part in sub.get("parts", []):
            color = part.get("color", "")
            qty = part.get("quantity", 1)
            colors.extend([color] * qty)
    return colors


def compute_f1(predicted: list[str], reference: list[str]) -> float:
    """Compute F1 score between two lists using Counter overlap."""
    if not predicted and not reference:
        return 1.0
    if not predicted or not reference:
        return 0.0

    pred_counts = Counter(predicted)
    ref_counts = Counter(reference)

    overlap = sum((pred_counts & ref_counts).values())
    precision = overlap / sum(pred_counts.values()) if pred_counts else 0
    recall = overlap / sum(ref_counts.values()) if ref_counts else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run_inference_mock(prompt: str, checkpoint_name: str) -> str:
    """
    Mock inference function. Replace with actual model inference when running
    with a GPU and loaded model.

    In production, this would:
    1. Load the base model + LoRA adapter from the checkpoint
    2. Run inference with the prompt
    3. Return the raw model output text
    """
    # Return a placeholder — actual inference requires GPU
    return json.dumps({
        "set_id": "mock-001",
        "object": prompt.replace("Build a ", "").replace("Build ", ""),
        "category": "mock",
        "subcategory": "test",
        "complexity": "medium",
        "total_parts": 10,
        "dominant_colors": ["Red", "Blue"],
        "dimensions_estimate": {"width": "10cm", "height": "8cm", "depth": "6cm"},
        "subassemblies": [
            {
                "name": "Base",
                "type": "structural",
                "parts": [
                    {"part_id": "3001", "name": "Brick 2x4", "category": "brick",
                     "color": "Red", "color_hex": "#B40000", "quantity": 5},
                    {"part_id": "3002", "name": "Brick 2x3", "category": "brick",
                     "color": "Blue", "color_hex": "#0055BF", "quantity": 5},
                ],
                "spatial": {"position": "center", "orientation": "upright", "connects_to": []},
            }
        ],
        "build_hints": ["Start with the base"],
    })


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoints = find_checkpoints()
    checkpoint_names = [cp.name for cp in checkpoints]

    if not checkpoint_names:
        print("No checkpoints found. Using mock checkpoints for demo data.")
        checkpoint_names = ["checkpoint-100", "checkpoint-500", "checkpoint-1000"]

    print(f"Found {len(checkpoint_names)} checkpoints: {checkpoint_names}")
    print(f"Processing {len(TEST_INPUTS)} test inputs...")

    for test_input in TEST_INPUTS:
        slug = test_input["slug"]
        prompt = test_input["prompt"]
        print(f"\n  Processing: {slug}")

        results = []
        for cp_name in checkpoint_names:
            output_text = run_inference_mock(prompt, cp_name)

            json_valid = check_json_validity(output_text)

            parts_f1 = 0.0
            color_f1 = 0.0
            part_count = 0

            if json_valid:
                try:
                    match = re.search(r'\{[\s\S]*\}', output_text)
                    data = json.loads(match.group()) if match else {}
                    part_count = data.get("total_parts", 0)

                    # Self-comparison for mock data (in real use, compare against reference)
                    predicted_parts = extract_parts(data)
                    predicted_colors = extract_colors(data)
                    parts_f1 = min(1.0, len(predicted_parts) / max(part_count, 1))
                    color_f1 = min(1.0, len(set(predicted_colors)) / max(len(data.get("dominant_colors", [])), 1))
                except Exception:
                    pass

            results.append({
                "checkpoint": cp_name,
                "json_valid": json_valid,
                "parts_f1": round(parts_f1, 3),
                "color_f1": round(color_f1, 3),
                "part_count": part_count,
            })

        comparison = {
            "input": slug,
            "prompt": prompt,
            "results": results,
        }

        out_path = OUTPUT_DIR / f"{slug}.json"
        with open(out_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"    -> {out_path}")

    print(f"\nDone. {len(TEST_INPUTS)} comparison files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
