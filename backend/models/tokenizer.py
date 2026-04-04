"""Prompt templates and JSON encoding/decoding utilities for Qwen3-VL fine-tuning."""

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

SYSTEM_PROMPT = (
    "You are a LEGO master builder and set designer. Given ANY image — whether "
    "it is a real-world photo, a drawing, or an existing LEGO set — your job is "
    "to design how it would be built as a LEGO set. "
    "Analyse the shape, colours, and structure of the object in the image, then "
    "output a structured JSON description of the equivalent LEGO build with these fields: "
    "object, category, subcategory, complexity, total_parts, dominant_colors, "
    "dimensions_estimate, subassemblies (each with name, type, parts list "
    "including part_id, name, category, color, color_hex, is_trans, quantity, "
    "and spatial info), and build_hints. "
    "Use real LEGO part numbers where possible. Be creative but realistic about "
    "what can be built with standard LEGO bricks."
)

USER_PROMPT = "Design a LEGO build inspired by this image and output a structured JSON description."


# ── Planner (text-to-JSON) prompts ─────────────────────────────────

PLANNER_SYSTEM_PROMPT = (
    "You are a LEGO master builder AI. Given a text description of an object, "
    "design a buildable LEGO model and output a structured JSON description. "
    "Use real LEGO part IDs and realistic color combinations. "
    "Order subassemblies from foundation (bottom) to top. "
    "Output fields: set_id, object, category, subcategory, complexity, total_parts, "
    "dominant_colors, dimensions_estimate, subassemblies (each with name, type, "
    "parts list with part_id/name/category/color/color_hex/is_trans/quantity, "
    "and spatial info including position/orientation/connects_to), and build_hints. "
    "/no_think"
)

PROMPT_TEMPLATES = [
    # Plain prompts (model picks default colors from learned associations)
    "Build me a {object}",
    "I want LEGO instructions for {object}",
    "{object}",
    "Design a LEGO model of {object}",
    "Create step-by-step LEGO building instructions for {object}",
    "How would I build {object} with LEGO bricks?",
    "Can you make {object} out of LEGO?",
    "I'd like to build {object} from LEGO",
    "LEGO instructions: {object}",
    "Make me a LEGO version of {object}",
    "Design {object} as a LEGO set",
    # Color-aware prompts (model learns to follow color instructions)
    "Build me a {dominant_color} {object}",
    "Build {object} in {dominant_color} and {secondary_color}",
    "Build {object} using mostly {color_hint}",
    "I want a {dominant_color} LEGO {object}",
    "Make {object} with {dominant_color} bricks",
    # Context-rich prompts
    "Generate a {complexity} LEGO build for {object}",
    "{category} theme: build {object}",
    "Help me build {object} with LEGO bricks, about {total_parts} pieces",
]


def build_planner_chat_messages(user_prompt: str) -> list[dict]:
    """Build chat messages for text-to-JSON LEGO generation."""
    return [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def sample_prompt_template(label: dict, epoch: int = 0, rng=None) -> str:
    """Pick a prompt template and fill it from label fields.

    Rotates through templates based on epoch to maximize diversity.
    Color-aware templates use the label's dominant_colors field.
    """
    import random
    rng = rng or random.Random(epoch * 10000 + hash(label.get("set_id", "")))

    # Extract color info from label
    dom_colors = label.get("dominant_colors", [])
    dominant_color = dom_colors[0] if len(dom_colors) > 0 else "Red"
    secondary_color = dom_colors[1] if len(dom_colors) > 1 else "White"
    color_hint = " and ".join(dom_colors[:2]) if dom_colors else "Red and White"

    # Build substitution dict
    subs = {
        "object": label.get("object", "LEGO model"),
        "category": label.get("category", "Creator"),
        "subcategory": label.get("subcategory", "General"),
        "complexity": label.get("complexity", "intermediate"),
        "total_parts": str(label.get("total_parts", 50)),
        "dominant_color": dominant_color,
        "secondary_color": secondary_color,
        "color_hint": color_hint,
    }

    # Rotate template selection by epoch
    idx = (epoch + rng.randint(0, len(PROMPT_TEMPLATES) - 1)) % len(PROMPT_TEMPLATES)
    template = PROMPT_TEMPLATES[idx]

    try:
        return template.format(**subs)
    except KeyError:
        # Fallback to simple template if substitution fails
        return f"Build me a {subs['object']}"


# ── Vision (image-to-JSON) chat messages ──────────────────────────

def build_chat_messages(image_url: str | None = None) -> list[dict]:
    """Build Qwen3-VL chat messages for LEGO analysis.

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


def strip_thinking_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


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
