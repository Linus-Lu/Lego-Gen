"""Prompt templates and JSON encoding/decoding utilities for BLIP-2 fine-tuning."""

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

PROMPT_TEMPLATE = (
    "Given the LEGO set image, describe it as a structured JSON object with the "
    "following fields: object, category, subcategory, complexity, total_parts, "
    "dominant_colors, dimensions_estimate, subassemblies, and build_hints.\n\nJSON:\n"
)


def get_json_prompt() -> str:
    """Return the fixed instruction prompt for BLIP-2 generation."""
    return PROMPT_TEMPLATE


def encode_json_label(label: dict, tokenizer, max_length: int = 1024) -> dict:
    """Serialize a JSON label dict and tokenize it for training targets.

    Returns dict with 'input_ids' and 'attention_mask' tensors.
    """
    json_str = json.dumps(label, indent=2)
    encoded = tokenizer(
        json_str,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    return encoded


def decode_and_parse(token_ids, tokenizer) -> dict | None:
    """Decode token IDs back to a string, extract JSON, and parse it.

    Returns the parsed dict, or None if parsing fails.
    """
    raw = tokenizer.decode(token_ids, skip_special_tokens=True)
    return extract_json_from_text(raw)


def extract_json_from_text(text: str) -> dict | None:
    """Extract and parse the first JSON object from a text string."""
    # Try to find JSON between braces
    start = text.find("{")
    if start == -1:
        return None

    # Find the matching closing brace
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                json_str = text[start : i + 1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    return _try_repair_json(json_str)
    return None


def _try_repair_json(raw: str) -> dict | None:
    """Attempt to repair malformed JSON using json_repair if available."""
    try:
        import json_repair
        repaired = json_repair.repair_json(raw)
        return json.loads(repaired)
    except Exception:
        return None
