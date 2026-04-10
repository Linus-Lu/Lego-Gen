#!/usr/bin/env python3
"""Ablation comparison: Qwen3.5-4B (few-shot) vs Qwen3.5-9B (few-shot) vs Ours (fine-tuned).

Produces poster-ready comparison tables and per-model breakdowns.

VRAM requirements (4-bit NF4):
  - Qwen3.5-4B: ~2.5GB  (fits on any GPU)
  - Qwen3.5-9B: ~5.5GB  (fits on any 8GB+ GPU)
  - Ours (9B + LoRA):  ~6GB  (fits on any 8GB+ GPU)
  - All three sequential: ~6GB peak (loads one at a time)

Usage:
    # Full comparison (needs GPU):
    python scripts/eval_comparison.py --num-samples 50

    # Quick test:
    python scripts/eval_comparison.py --num-samples 5

    # Skip 4B baseline (faster):
    python scripts/eval_comparison.py --num-samples 50 --skip-4b

    # Use local model paths (no internet):
    python scripts/eval_comparison.py --model-4b /path/to/Qwen3.5-4B --model-9b /path/to/Qwen3.5-9B
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── HF mirror auto-detect ─────────────────────────────────────────────
if not os.environ.get("HF_ENDPOINT") and not os.environ.get("HF_HUB_OFFLINE"):
    import socket
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3)
    except OSError:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("[AutoDL] Using hf-mirror.com")

from backend.config import (
    DATA_DIR,
    UNIFIED_CHECKPOINT_DIR,
    UNIFIED_MODEL_NAME,
    BRICK_MODEL_NAME,
)
from backend.models.tokenizer import (
    PLANNER_SYSTEM_PROMPT,
    extract_json_from_text,
    strip_thinking_blocks,
)
from backend.training.utils import (
    compute_color_f1,
    compute_parts_f1,
    compute_structural_coherence,
    compute_build_feasibility,
)


# ═══════════════════════════════════════════════════════════════════════
#  Few-Shot Examples (5 compact examples for baseline models)
# ═══════════════════════════════════════════════════════════════════════

def load_few_shot_examples(data_dir: Path, n: int = 5) -> list[dict]:
    """Load n training samples as few-shot examples."""
    labels_dir = data_dir / "labels"
    st2b_dir = data_dir / "st2b_labels"

    examples = []

    # Try ST2B labels first (consistent format)
    if st2b_dir.exists():
        for p in sorted(st2b_dir.glob("*.json"))[:n * 3]:
            try:
                label = json.loads(p.read_text())
                if label.get("subassemblies") and label.get("total_parts", 0) > 5:
                    examples.append(label)
                    if len(examples) >= n:
                        break
            except (json.JSONDecodeError, OSError):
                continue

    # Fall back to Rebrickable labels
    if len(examples) < n and labels_dir.exists():
        for p in sorted(labels_dir.glob("*.json")):
            try:
                label = json.loads(p.read_text())
                if label.get("subassemblies") and label.get("total_parts", 0) > 5:
                    examples.append(label)
                    if len(examples) >= n:
                        break
            except (json.JSONDecodeError, OSError):
                continue

    return examples[:n]


def build_few_shot_prompt(examples: list[dict], query: str) -> str:
    """Build a few-shot prompt with examples + query."""
    parts = []
    parts.append(PLANNER_SYSTEM_PROMPT)
    parts.append("")
    parts.append("Here are some examples of LEGO build descriptions:")
    parts.append("")

    for i, ex in enumerate(examples):
        obj = ex.get("object", "LEGO model")
        # Compact JSON (no indentation) to save tokens
        compact = json.dumps(ex, separators=(",", ":"))
        # Truncate very long examples
        if len(compact) > 1500:
            compact = compact[:1500] + '..."}'
        parts.append(f"Example {i + 1}:")
        parts.append(f"Input: Build me a {obj}")
        parts.append(f"Output: {compact}")
        parts.append("")

    parts.append("Now generate a LEGO build for this request:")
    parts.append(f"Input: {query}")
    parts.append("Output:")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
#  Model Loading & Inference
# ═══════════════════════════════════════════════════════════════════════

def load_baseline_model(model_name: str, device: int = 0):
    """Load a raw Qwen model with 4-bit quantization for few-shot eval."""
    print(f"  Loading {model_name} (4-bit)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    print(f"  Loaded {model_name}")
    return model, tokenizer


def load_finetuned_model(base_name: str, adapter_path: str, device: int = 0):
    """Load our fine-tuned model (base + LoRA adapter)."""
    from peft import PeftModel

    print(f"  Loading fine-tuned {base_name} + adapter...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        quantization_config=bnb_config,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    print(f"  Loaded fine-tuned model from {adapter_path}")
    return model, tokenizer


def generate_from_model(model, tokenizer, prompt: str, max_new_tokens: int = 2048) -> str:
    """Generate text from a model given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return strip_thinking_blocks(raw)


def generate_finetuned(model, tokenizer, query: str, max_new_tokens: int = 2048) -> str:
    """Generate from fine-tuned model using the proper chat template."""
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return strip_thinking_blocks(raw)


def unload_model(model, tokenizer):
    """Free GPU memory after evaluation."""
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
#  Scoring
# ═══════════════════════════════════════════════════════════════════════

def score_prediction(pred: dict, ref: dict) -> dict:
    """Score a single prediction against ground truth."""
    scores = {}

    # JSON validity
    scores["valid_json"] = 1.0 if isinstance(pred, dict) and len(pred) > 0 else 0.0

    # Required fields
    required = [
        "set_id", "object", "category", "subcategory", "complexity",
        "total_parts", "dominant_colors", "dimensions_estimate",
        "subassemblies", "build_hints",
    ]
    scores["fields_present"] = sum(1 for f in required if f in pred) / len(required)

    # Field accuracy
    scores["category_acc"] = 1.0 if pred.get("category") == ref.get("category") else 0.0
    scores["complexity_acc"] = 1.0 if pred.get("complexity") == ref.get("complexity") else 0.0

    # Color F1
    scores["color_f1"] = compute_color_f1(
        pred.get("dominant_colors", []),
        ref.get("dominant_colors", []),
    )

    # Parts F1
    pred_parts = [p for sa in pred.get("subassemblies", []) for p in sa.get("parts", [])]
    ref_parts = [p for sa in ref.get("subassemblies", []) for p in sa.get("parts", [])]
    scores["parts_f1"] = compute_parts_f1(pred_parts, ref_parts)

    # Part count error
    pred_total = pred.get("total_parts", 0)
    ref_total = ref.get("total_parts", 1)
    scores["part_count_err"] = abs(pred_total - ref_total) / max(ref_total, 1) * 100

    # Structural coherence
    scores["struct_coherence"] = compute_structural_coherence(pred)

    # Build feasibility
    scores["build_feasibility"] = compute_build_feasibility(pred)

    # Layer format
    subs = pred.get("subassemblies", [])
    layer_count = sum(1 for sa in subs if sa.get("name", "").startswith("layer_"))
    scores["layer_format"] = layer_count / max(len(subs), 1)

    # Overall weighted
    scores["overall"] = (
        scores["valid_json"] * 0.10 +
        scores["fields_present"] * 0.10 +
        scores["category_acc"] * 0.10 +
        scores["complexity_acc"] * 0.05 +
        scores["color_f1"] * 0.15 +
        scores["parts_f1"] * 0.20 +
        scores["struct_coherence"] * 0.15 +
        scores["build_feasibility"] * 0.10 +
        scores["layer_format"] * 0.05
    )

    return scores


# ═══════════════════════════════════════════════════════════════════════
#  Load Test Data
# ═══════════════════════════════════════════════════════════════════════

def load_test_samples(data_dir: Path, num_samples: int) -> list[dict]:
    """Load validation samples (text prompts + reference labels)."""
    # Try splits.json
    splits_path = data_dir / "splits.json"
    if splits_path.exists():
        with open(splits_path) as f:
            val_ids = json.load(f).get("val", [])
    else:
        val_ids = [p.stem for p in (data_dir / "labels").glob("*.json")]

    samples = []
    labels_dir = data_dir / "labels"
    st2b_dir = data_dir / "st2b_labels"

    # Rebrickable samples
    for set_id in val_ids:
        label_path = labels_dir / f"{set_id}.json"
        if not label_path.exists():
            continue
        try:
            label = json.loads(label_path.read_text())
            obj = label.get("object", "")
            if obj and label.get("subassemblies"):
                samples.append({
                    "id": str(set_id),
                    "query": f"Build me a {obj}",
                    "ref": label,
                    "source": "rebrickable",
                })
        except (json.JSONDecodeError, OSError):
            continue
        if len(samples) >= num_samples:
            break

    # If not enough, add ST2B samples
    if len(samples) < num_samples and st2b_dir.exists():
        for p in sorted(st2b_dir.glob("*.json")):
            if len(samples) >= num_samples:
                break
            try:
                label = json.loads(p.read_text())
                obj = label.get("object", "")
                if obj and label.get("subassemblies"):
                    samples.append({
                        "id": p.stem,
                        "query": f"Build me a {obj}",
                        "ref": label,
                        "source": "st2b",
                    })
            except (json.JSONDecodeError, OSError):
                continue

    return samples[:num_samples]


# ═══════════════════════════════════════════════════════════════════════
#  Run Evaluation for One Model
# ═══════════════════════════════════════════════════════════════════════

def evaluate_model(
    model_name: str,
    generate_fn,
    samples: list[dict],
    few_shot_prompt_fn=None,
) -> list[dict]:
    """Run evaluation on all samples for a single model."""
    all_scores = []

    for i, sample in enumerate(samples):
        print(f"    [{i + 1}/{len(samples)}] {sample['id']}", end=" ... ", flush=True)

        try:
            t0 = time.time()

            if few_shot_prompt_fn:
                prompt = few_shot_prompt_fn(sample["query"])
                raw = generate_fn(prompt)
            else:
                raw = generate_fn(sample["query"])

            latency = (time.time() - t0) * 1000

            # Parse JSON from output
            pred = extract_json_from_text(raw) or {}
            scores = score_prediction(pred, sample["ref"])
            scores["latency_ms"] = latency
            scores["id"] = sample["id"]
            all_scores.append(scores)

            print(f"overall={scores['overall']:.0%} {latency:.0f}ms")

        except Exception as e:
            print(f"FAILED: {e}")
            all_scores.append({
                "id": sample["id"],
                "valid_json": 0, "fields_present": 0, "category_acc": 0,
                "complexity_acc": 0, "color_f1": 0, "parts_f1": 0,
                "part_count_err": 100, "struct_coherence": 0,
                "build_feasibility": 0, "layer_format": 0, "overall": 0,
                "latency_ms": 0,
            })

    return all_scores


# ═══════════════════════════════════════════════════════════════════════
#  Aggregation & Display
# ═══════════════════════════════════════════════════════════════════════

def aggregate(scores: list[dict]) -> dict:
    """Compute mean for each metric."""
    if not scores:
        return {}
    keys = [k for k in scores[0] if k != "id" and isinstance(scores[0].get(k), (int, float))]
    return {k: float(np.mean([s.get(k, 0) for s in scores])) for k in keys}


def print_comparison_table(results: dict[str, dict]):
    """Print a poster-ready comparison table."""
    models = list(results.keys())
    metrics = [
        "valid_json", "fields_present", "category_acc", "color_f1",
        "parts_f1", "struct_coherence", "build_feasibility",
        "layer_format", "overall", "latency_ms",
    ]

    # Header
    header = f"{'Metric':<25}"
    for m in models:
        header += f" {m:>18}"
    print(f"\n{'=' * (25 + 19 * len(models))}")
    print("  MODEL COMPARISON (Ablation Study)")
    print(f"{'=' * (25 + 19 * len(models))}")
    print(f"  {header}")
    print(f"  {'-' * 25}" + f" {'-' * 18}" * len(models))

    for metric in metrics:
        row = f"  {metric:<25}"
        # Find best value for highlighting
        vals = [results[m].get(metric, 0) for m in models]
        if metric == "latency_ms":
            best_idx = np.argmin(vals)  # lower is better
        elif metric == "part_count_err":
            best_idx = np.argmin(vals)
        else:
            best_idx = np.argmax(vals)  # higher is better

        for i, m in enumerate(models):
            val = results[m].get(metric, 0)
            if metric in ("latency_ms", "part_count_err"):
                cell = f"{val:>15.1f}ms" if metric == "latency_ms" else f"{val:>16.1f}%"
            elif 0 <= val <= 1:
                cell = f"{val:>17.1%}"
            else:
                cell = f"{val:>18.2f}"

            if i == best_idx:
                cell += "*"  # mark best
            else:
                cell += " "

            row += cell
        print(row)

    print(f"{'=' * (25 + 19 * len(models))}")
    print("  * = best in row")


def print_bar_comparison(results: dict[str, dict]):
    """Print visual bar chart comparison for key metrics."""
    models = list(results.keys())
    key_metrics = ["valid_json", "color_f1", "parts_f1", "struct_coherence", "overall"]

    print(f"\n  KEY METRICS COMPARISON")
    print(f"  {'-' * 60}")

    for metric in key_metrics:
        print(f"\n  {metric}:")
        for model_name in models:
            val = results[model_name].get(metric, 0)
            bar_len = int(val * 30)
            bar = "\u2588" * bar_len + "\u2591" * (30 - bar_len)
            print(f"    {model_name:<20} {bar} {val:.1%}")


def save_latex_comparison(results: dict[str, dict], path: Path):
    """Save LaTeX table for poster."""
    models = list(results.keys())
    metrics = [
        ("JSON Validity", "valid_json"),
        ("Schema Compliance", "fields_present"),
        ("Category Acc.", "category_acc"),
        ("Color F1", "color_f1"),
        ("Parts F1", "parts_f1"),
        ("Structural Coherence", "struct_coherence"),
        ("Build Feasibility", "build_feasibility"),
        ("Layer Format", "layer_format"),
        ("Overall Score", "overall"),
        ("Latency (ms)", "latency_ms"),
    ]

    cols = "l" + "r" * len(models)
    header_cells = " & ".join([m.replace("_", r"\_") for m in models])

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Ablation study: Few-shot baselines vs. fine-tuned model}",
        f"\\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        f"Metric & {header_cells} \\\\",
        r"\midrule",
    ]

    for display_name, key in metrics:
        vals = [results[m].get(key, 0) for m in models]

        if key == "latency_ms":
            best_idx = int(np.argmin(vals))
            cells = []
            for i, v in enumerate(vals):
                s = f"{v:.0f}"
                cells.append(f"\\textbf{{{s}}}" if i == best_idx else s)
        elif key == "part_count_err":
            best_idx = int(np.argmin(vals))
            cells = []
            for i, v in enumerate(vals):
                s = f"{v:.1f}\\%"
                cells.append(f"\\textbf{{{s}}}" if i == best_idx else s)
        else:
            best_idx = int(np.argmax(vals))
            cells = []
            for i, v in enumerate(vals):
                s = f"{v:.1%}"
                cells.append(f"\\textbf{{{s}}}" if i == best_idx else s)

        row = f"{display_name} & {' & '.join(cells)} \\\\"
        lines.append(row)

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"  LaTeX table saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation: Qwen 4B few-shot vs 9B few-shot vs Ours (fine-tuned)"
    )
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--num-few-shot", type=int, default=5)
    parser.add_argument("--skip-4b", action="store_true", help="Skip Qwen3.5-4B baseline")
    parser.add_argument("--skip-9b-raw", action="store_true", help="Skip raw 9B baseline")
    parser.add_argument("--model-4b", type=str, default=BRICK_MODEL_NAME)
    parser.add_argument("--model-9b", type=str, default=UNIFIED_MODEL_NAME)
    parser.add_argument("--adapter-path", type=str, default=str(UNIFIED_CHECKPOINT_DIR))
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "eval_results"))
    parser.add_argument("--device", type=int, default=0, help="GPU device index")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 70}")
    print(f"  LEGO-Gen Ablation Study")
    print(f"  Samples: {args.num_samples}")
    print(f"  Few-shot examples: {args.num_few_shot}")
    print(f"  GPU: {args.device}")
    print(f"{'=' * 70}")

    # ── Load test data ─────────────────────────────────────────────────
    samples = load_test_samples(data_dir, args.num_samples)
    print(f"  Test samples: {len(samples)}")

    if not samples:
        print("ERROR: No test samples found.")
        sys.exit(1)

    # ── Load few-shot examples ─────────────────────────────────────────
    few_shot_examples = load_few_shot_examples(data_dir, args.num_few_shot)
    print(f"  Few-shot examples loaded: {len(few_shot_examples)}")

    results = {}

    # ══════════════════════════════════════════════════════════════════
    #  Model 1: Qwen3.5-4B (5-shot baseline)
    # ══════════════════════════════════════════════════════════════════
    if not args.skip_4b:
        print(f"\n{'─' * 70}")
        print(f"  [1/3] Qwen3.5-4B (5-shot)")
        print(f"{'─' * 70}")

        model_4b, tok_4b = load_baseline_model(args.model_4b, args.device)

        def few_shot_4b(prompt):
            return generate_from_model(model_4b, tok_4b, prompt, max_new_tokens=2048)

        def build_prompt_4b(query):
            return build_few_shot_prompt(few_shot_examples, query)

        scores_4b = evaluate_model(
            "4B-5shot", few_shot_4b, samples,
            few_shot_prompt_fn=build_prompt_4b,
        )
        results["Qwen3.5-4B\n(5-shot)"] = aggregate(scores_4b)

        # Save per-sample
        with open(output_dir / "scores_4b_fewshot.json", "w") as f:
            json.dump(scores_4b, f, indent=2)

        unload_model(model_4b, tok_4b)

    # ══════════════════════════════════════════════════════════════════
    #  Model 2: Qwen3.5-9B (5-shot baseline)
    # ══════════════════════════════════════════════════════════════════
    if not args.skip_9b_raw:
        print(f"\n{'─' * 70}")
        print(f"  [2/3] Qwen3.5-9B (5-shot)")
        print(f"{'─' * 70}")

        model_9b, tok_9b = load_baseline_model(args.model_9b, args.device)

        def few_shot_9b(prompt):
            return generate_from_model(model_9b, tok_9b, prompt, max_new_tokens=2048)

        def build_prompt_9b(query):
            return build_few_shot_prompt(few_shot_examples, query)

        scores_9b = evaluate_model(
            "9B-5shot", few_shot_9b, samples,
            few_shot_prompt_fn=build_prompt_9b,
        )
        results["Qwen3.5-9B\n(5-shot)"] = aggregate(scores_9b)

        with open(output_dir / "scores_9b_fewshot.json", "w") as f:
            json.dump(scores_9b, f, indent=2)

        unload_model(model_9b, tok_9b)

    # ══════════════════════════════════════════════════════════════════
    #  Model 3: Ours (Qwen3.5-9B + LoRA fine-tuned)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print(f"  [3/3] Ours (Qwen3.5-9B + LoRA fine-tuned)")
    print(f"{'─' * 70}")

    adapter_path = args.adapter_path
    if not Path(adapter_path).exists():
        print(f"  WARNING: Adapter not found at {adapter_path}")
        print(f"  Skipping fine-tuned model evaluation.")
    else:
        model_ft, tok_ft = load_finetuned_model(args.model_9b, adapter_path, args.device)

        def gen_finetuned(query):
            return generate_finetuned(model_ft, tok_ft, query, max_new_tokens=2048)

        scores_ft = evaluate_model("Ours", gen_finetuned, samples)
        results["Ours\n(9B+LoRA)"] = aggregate(scores_ft)

        with open(output_dir / "scores_finetuned.json", "w") as f:
            json.dump(scores_ft, f, indent=2)

        unload_model(model_ft, tok_ft)

    # ══════════════════════════════════════════════════════════════════
    #  Results
    # ══════════════════════════════════════════════════════════════════
    if len(results) < 2:
        print("\nNot enough models to compare. Need at least 2.")
        sys.exit(1)

    print_comparison_table(results)
    print_bar_comparison(results)

    # Save
    summary = {name: agg for name, agg in results.items()}
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  JSON: {output_dir / 'comparison_results.json'}")

    save_latex_comparison(results, output_dir / "comparison_table.tex")

    # CSV
    csv_path = output_dir / "comparison_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        metrics = list(next(iter(results.values())).keys())
        writer.writerow(["model"] + metrics)
        for name, agg in results.items():
            writer.writerow([name.replace("\n", " ")] + [f"{agg.get(m, 0):.4f}" for m in metrics])
    print(f"  CSV: {csv_path}")

    print(f"\n  All results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
