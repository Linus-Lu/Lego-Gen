#!/usr/bin/env python3
"""Generate prompt variants for Rebrickable labels for text-to-JSON training.

Reads all labels from data/labels/ and creates 8-10 prompt variants per label
using PROMPT_TEMPLATES. Saves to data/prompts/{set_num}.json.

Usage:
    python scripts/prepare_planner_prompts.py
"""

import json
import sys
from pathlib import Path
from random import Random

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import LABELS_DIR, PLANNER_PROMPTS_DIR
from backend.models.tokenizer import PROMPT_TEMPLATES, sample_prompt_template


def main():
    PLANNER_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    label_files = sorted(LABELS_DIR.glob("*.json"))
    print(f"Found {len(label_files)} labels in {LABELS_DIR}")

    rng = Random(42)
    total_prompts = 0

    for lf in label_files:
        with open(lf) as f:
            label = json.load(f)

        set_num = lf.stem

        # Generate prompts using different epoch values to get diverse templates
        prompts = []
        seen = set()
        for epoch in range(20):
            prompt = sample_prompt_template(label, epoch=epoch, rng=Random(rng.randint(0, 999999)))
            if prompt not in seen:
                seen.add(prompt)
                prompts.append(prompt)
            if len(prompts) >= 10:
                break

        # Ensure minimum of 8 prompts by adding simple variants
        simple_variants = [
            label.get("object", "LEGO model"),
            f"Build me a {label.get('object', 'LEGO model')}",
            f"LEGO {label.get('object', 'model')}",
        ]
        for sv in simple_variants:
            if sv not in seen and len(prompts) < 10:
                seen.add(sv)
                prompts.append(sv)

        out_path = PLANNER_PROMPTS_DIR / f"{set_num}.json"
        with open(out_path, "w") as f:
            json.dump(prompts, f, indent=2)

        total_prompts += len(prompts)

    print(f"Generated {total_prompts} prompts for {len(label_files)} labels")
    print(f"  Average: {total_prompts / max(1, len(label_files)):.1f} per label")
    print(f"  Output: {PLANNER_PROMPTS_DIR}")


if __name__ == "__main__":
    main()
