"""Prompt templates and JSON encoding/decoding utilities for Qwen2.5-VL fine-tuning."""

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

SYSTEM_PROMPT = (
    "You are a LEGO set analysis expert. Given an image of a LEGO set, "
    "output a structured JSON description with the following fields: "
    "object, category, subcategory, complexity, total_parts, dominant_colors, "
    "dimensions_estimate, subassemblies (each with name, type, parts list "
    "including part_id, name, category, color, color_hex, is_trans, quantity, "
    "and spatial info), and build_hints."
)

USER_PROMPT = "Analyze this LEGO set image and output a structured JSON description."


def build_chat_messages(image_url: str | None = None) -> list[dict]:
    """Build Qwen2.5-VL chat messages for LEGO analysis.

    For training, image_url can be a local file path.
    For inference, it can be a PIL image passed via the processor.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    if image_url:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": USER_PROMPT},
            ],
        })
    else:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": USER_PROMPT},
            ],
        })

    return messages


def get_json_prompt() -> str:
    """Return the system + user prompt as a single string (for compatibility)."""
    return f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}\n\nJSON:\n"


def encode_json_label(label: dict, tokenizer, max_length: int = 1024) -> dict:
    """Serialize a JSON label dict and tokenize it for training targets."""
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
    """Decode token IDs back to a string, extract JSON, and parse it."""
    raw = tokenizer.decode(token_ids, skip_special_tokens=True)
    return extract_json_from_text(raw)


def extract_json_from_text(text: str) -> dict | None:
    """Extract and parse the first JSON object from a text string."""
    start = text.find("{")
    if start == -1:
        return None

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
