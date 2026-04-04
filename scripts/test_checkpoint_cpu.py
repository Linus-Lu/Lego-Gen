"""Test planner checkpoint without BitsAndBytes quantization.

Qwen3.5-9B uses DeltaNet (hybrid linear attention) with Triton kernels,
so CUDA is required. This script loads in float16 (no 4-bit quant) for
a clean checkpoint validation.

Usage:
    python scripts/test_checkpoint_cpu.py
    python scripts/test_checkpoint_cpu.py --checkpoint backend/models/checkpoints/qwen35-lego-planner-lora/checkpoint-600
    python scripts/test_checkpoint_cpu.py --prompt "Build me a red fire truck"
"""

import argparse
import json
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_model(model_name: str, adapter_path: str):
    """Load base model in float16 (no quantization) and attach LoRA adapter.

    Qwen3.5-9B requires CUDA for its DeltaNet Triton kernels.
    Using float16 instead of 4-bit quant for a clean checkpoint test.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading tokenizer from {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model on {device} ({dtype}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    )

    print(f"Loading LoRA adapter from {adapter_path} ...")
    model = PeftModel.from_pretrained(model, adapter_path, torch_dtype=dtype)

    print("Merging LoRA weights into base model ...")
    model = model.merge_and_unload()

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params / 1e9:.1f}B params on {device}")

    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 1024):
    """Run generation on CPU."""
    import torch
    from backend.models.tokenizer import (
        build_planner_chat_messages,
        extract_json_from_text,
        strip_thinking_blocks,
    )

    messages = build_planner_chat_messages(prompt)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print(f"\nPrompt: {prompt}")
    print(f"Input tokens: {inputs['input_ids'].shape[1]}")
    print("Generating ...")

    start = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    elapsed = time.time() - start

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    raw_output = strip_thinking_blocks(raw_output)

    tokens_generated = len(generated_ids)
    tps = tokens_generated / elapsed if elapsed > 0 else 0

    print(f"Generated {tokens_generated} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)")

    parsed = extract_json_from_text(raw_output)
    if parsed is None:
        print("Standard parse failed, attempting truncated JSON repair ...")
        parsed = repair_truncated_json(raw_output)
        if parsed:
            print("JSON repaired successfully!")
    return raw_output, parsed


def repair_truncated_json(raw: str) -> dict | None:
    """Repair JSON truncated by token limit — close open structures."""
    import re

    # Find the start of JSON
    start = raw.find("{")
    if start == -1:
        return None
    text = raw[start:]

    # Try parsing as-is first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try json_repair library
    try:
        import json_repair
        repaired = json_repair.repair_json(text)
        return json.loads(repaired)
    except Exception:
        pass

    # Manual: close brackets/braces and strip trailing comma
    text = text.rstrip()
    text = re.sub(r',\s*$', '', text)

    # Count unclosed structures
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')

    # Close any open string
    if text.count('"') % 2 == 1:
        text += '"'

    text += ']' * max(0, open_brackets)
    text += '}' * max(0, open_braces)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def validate_output(parsed: dict | None):
    """Quick validation of the generated JSON structure."""
    if parsed is None:
        print("\n[FAIL] Could not parse JSON from output")
        return False

    required = ["object", "category", "subassemblies"]
    missing = [f for f in required if f not in parsed]
    if missing:
        print(f"\n[WARN] Missing fields: {missing}")

    subs = parsed.get("subassemblies", [])
    print(f"\n--- Validation ---")
    print(f"  Object:          {parsed.get('object', 'N/A')}")
    print(f"  Category:        {parsed.get('category', 'N/A')}")
    print(f"  Complexity:      {parsed.get('complexity', 'N/A')}")
    print(f"  Total parts:     {parsed.get('total_parts', 'N/A')}")
    print(f"  Colors:          {parsed.get('dominant_colors', [])}")
    print(f"  Subassemblies:   {len(subs)}")

    total_parts_actual = 0
    for sa in subs:
        parts = sa.get("parts", [])
        sa_qty = sum(p.get("quantity", 0) for p in parts)
        total_parts_actual += sa_qty
        print(f"    - {sa.get('name', '?')}: {len(parts)} unique parts, {sa_qty} total")

    print(f"  Counted parts:   {total_parts_actual}")

    has_part_ids = all(
        p.get("part_id") for sa in subs for p in sa.get("parts", [])
    )
    print(f"  All part IDs:    {'Yes' if has_part_ids else 'No'}")
    print(f"  JSON valid:      Yes")

    return True


TEST_PROMPTS = [
    "Build me a small red race car",
    "I want LEGO instructions for a medieval castle tower",
    "Design a LEGO model of a cute robot",
]


def main():
    parser = argparse.ArgumentParser(description="Test planner checkpoint on CPU")
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "backend/models/checkpoints/qwen35-lego-planner-lora/checkpoint-600"),
        help="Path to LoRA adapter checkpoint",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-9B",
        help="Base model name",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Single prompt to test (otherwise runs built-in test prompts)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max new tokens to generate",
    )
    args = parser.parse_args()

    # Verify checkpoint exists
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"Error: Checkpoint not found at {ckpt}")
        sys.exit(1)

    adapter_config = ckpt / "adapter_config.json"
    if not adapter_config.exists():
        print(f"Error: No adapter_config.json in {ckpt}")
        sys.exit(1)

    print(f"Checkpoint: {ckpt}")
    print(f"Base model: {args.model}")
    print("=" * 60)

    model, tokenizer = load_model(args.model, str(ckpt))

    prompts = [args.prompt] if args.prompt else TEST_PROMPTS
    results = []

    for i, prompt in enumerate(prompts):
        print(f"\n{'=' * 60}")
        print(f"Test {i + 1}/{len(prompts)}")
        print("=" * 60)

        raw_output, parsed = generate(model, tokenizer, prompt, args.max_tokens)

        if parsed:
            print(f"\n--- Raw JSON (truncated) ---")
            print(json.dumps(parsed, indent=2)[:2000])
            valid = validate_output(parsed)
        else:
            print(f"\n--- Raw output ---")
            print(raw_output[:2000])
            valid = False

        results.append({"prompt": prompt, "valid_json": parsed is not None, "valid_structure": valid})

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        status = "PASS" if r["valid_structure"] else ("JSON_OK" if r["valid_json"] else "FAIL")
        print(f"  [{status}] {r['prompt']}")

    valid_count = sum(1 for r in results if r["valid_json"])
    print(f"\nJSON parse rate: {valid_count}/{len(results)}")


if __name__ == "__main__":
    main()
