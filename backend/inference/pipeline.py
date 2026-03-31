"""End-to-end inference pipeline: image -> JSON description -> build steps."""

import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.config import (
    MODEL_NAME,
    CHECKPOINT_DIR,
    MAX_NEW_TOKENS,
    NUM_BEAMS,
    TEMPERATURE,
    TOP_P,
    LEGOGEN_DEV,
)

# Lazy imports — only needed when real pipeline is used
# from backend.models.vision_encoder import LegoVisionEncoder
# from backend.models.tokenizer import get_json_prompt, extract_json_from_text
# from backend.inference.constraint_engine import safe_parse_and_validate, enforce_valid_values


# ── Singleton ──────────────────────────────────────────────────────────

_pipeline_instance = None


def get_pipeline():
    """Get or create the singleton pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        if LEGOGEN_DEV:
            _pipeline_instance = MockPipeline()
        else:
            _pipeline_instance = LegoGenPipeline()
    return _pipeline_instance


# ── Mock pipeline for frontend development ─────────────────────────────

class MockPipeline:
    """Returns a realistic hardcoded LEGO house response for dev/testing."""

    def generate_build(self, image) -> dict:
        from backend.inference.postprocess_manual import json_to_steps

        description = {
            "set_id": "mock-001",
            "object": "Cozy Family House",
            "category": "City",
            "subcategory": "Residential",
            "complexity": "intermediate",
            "total_parts": 86,
            "dominant_colors": ["Red", "White", "Bright Orange"],
            "dimensions_estimate": {"width": "medium", "height": "medium", "depth": "small"},
            "subassemblies": [
                {
                    "name": "base_plate",
                    "type": "Baseplates",
                    "parts": [
                        {"part_id": "3811", "name": "Baseplate 32x32", "category": "Baseplates", "color": "Green", "color_hex": "#237841", "is_trans": False, "quantity": 1},
                        {"part_id": "3020", "name": "Plate 2x4", "category": "Plates", "color": "Dark Tan", "color_hex": "#958A73", "is_trans": False, "quantity": 4},
                    ],
                    "spatial": {"position": "bottom", "orientation": "flat", "connects_to": ["walls_lower"]},
                },
                {
                    "name": "walls_lower",
                    "type": "Bricks",
                    "parts": [
                        {"part_id": "3001", "name": "Brick 2x4", "category": "Bricks", "color": "Red", "color_hex": "#C91A09", "is_trans": False, "quantity": 8},
                        {"part_id": "3001", "name": "Brick 2x4", "category": "Bricks", "color": "White", "color_hex": "#FFFFFF", "is_trans": False, "quantity": 6},
                        {"part_id": "3010", "name": "Brick 1x4", "category": "Bricks", "color": "Red", "color_hex": "#C91A09", "is_trans": False, "quantity": 4},
                    ],
                    "spatial": {"position": "bottom", "orientation": "upright", "connects_to": ["walls_upper"]},
                },
                {
                    "name": "walls_upper",
                    "type": "Bricks",
                    "parts": [
                        {"part_id": "3001", "name": "Brick 2x4", "category": "Bricks", "color": "Yellow", "color_hex": "#F2CD37", "is_trans": False, "quantity": 6},
                        {"part_id": "3001", "name": "Brick 2x4", "category": "Bricks", "color": "Red", "color_hex": "#C91A09", "is_trans": False, "quantity": 4},
                        {"part_id": "3622", "name": "Brick 1x3", "category": "Bricks", "color": "White", "color_hex": "#FFFFFF", "is_trans": False, "quantity": 4},
                    ],
                    "spatial": {"position": "center", "orientation": "upright", "connects_to": ["windows_and_doors", "roof"]},
                },
                {
                    "name": "windows_and_doors",
                    "type": "Windows and Doors",
                    "parts": [
                        {"part_id": "60594", "name": "Window 1x2x3 Pane", "category": "Windows and Doors", "color": "Trans-Clear", "color_hex": "#FCFCFC", "is_trans": True, "quantity": 4},
                        {"part_id": "60593", "name": "Window 1x2x3 Frame", "category": "Windows and Doors", "color": "Blue", "color_hex": "#0055BF", "is_trans": False, "quantity": 4},
                        {"part_id": "60596", "name": "Door 1x4x6 Frame", "category": "Windows and Doors", "color": "Reddish Brown", "color_hex": "#582A12", "is_trans": False, "quantity": 1},
                        {"part_id": "60616", "name": "Door 1x4x6 Panel", "category": "Windows and Doors", "color": "Dark Azure", "color_hex": "#078BC9", "is_trans": False, "quantity": 1},
                    ],
                    "spatial": {"position": "center", "orientation": "upright", "connects_to": ["walls_upper"]},
                },
                {
                    "name": "roof",
                    "type": "Roof Tiles",
                    "parts": [
                        {"part_id": "3037", "name": "Slope 45 2x4", "category": "Roof Tiles", "color": "Bright Orange", "color_hex": "#FE8A18", "is_trans": False, "quantity": 8},
                        {"part_id": "3038", "name": "Slope 45 2x3", "category": "Roof Tiles", "color": "Bright Orange", "color_hex": "#FE8A18", "is_trans": False, "quantity": 4},
                        {"part_id": "3048", "name": "Slope 45 1x2 Triple", "category": "Roof Tiles", "color": "Dark Red", "color_hex": "#720E0F", "is_trans": False, "quantity": 2},
                    ],
                    "spatial": {"position": "top", "orientation": "angled", "connects_to": ["walls_upper"]},
                },
            ],
            "build_hints": [
                "Start with the green base plate",
                "Build the lower walls with alternating red and white bricks",
                "Add upper walls, leaving gaps for windows",
                "Insert window frames and panes",
                "Attach the orange roof slopes last",
            ],
        }

        steps = json_to_steps(description)

        return {
            "description": description,
            "steps": steps,
            "metadata": {
                "model_version": "mock-dev-v1",
                "generation_time_ms": 42,
                "json_valid": True,
                "errors": [],
            },
        }


class LegoGenPipeline:
    """Full inference pipeline from image to validated JSON description."""

    def __init__(
        self,
        adapter_path: str | Path | None = None,
        model_name: str = MODEL_NAME,
    ):
        import torch
        from backend.models.vision_encoder import LegoVisionEncoder

        if adapter_path is None:
            adapter_path = CHECKPOINT_DIR / "blip2-lego-lora"

        adapter_path = Path(adapter_path)
        load_adapter = str(adapter_path) if adapter_path.exists() else None

        self.encoder = LegoVisionEncoder(
            model_name=model_name,
            load_adapter=load_adapter,
        )
        self.model = self.encoder.get_model()
        self.processor = self.encoder.get_processor()
        self.model.eval()

    def describe_image(
        self,
        image,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> dict:
        """Generate a structured JSON description from a LEGO image."""
        import torch
        import json
        from backend.models.tokenizer import get_json_prompt, extract_json_from_text
        from backend.inference.constraint_engine import safe_parse_and_validate, enforce_valid_values

        start = time.time()

        with torch.inference_mode():
            prompt = get_json_prompt()
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=NUM_BEAMS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=NUM_BEAMS == 1,
            )

            raw_output = self.processor.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

        # Parse and validate (outside inference_mode)
        parsed = extract_json_from_text(raw_output)
        if parsed:
            parsed = enforce_valid_values(parsed)
            is_valid, errors = safe_parse_and_validate(json.dumps(parsed))
        else:
            parsed, errors = safe_parse_and_validate(raw_output)
            is_valid = parsed is not None

        elapsed_ms = int((time.time() - start) * 1000)

        return {
            "description": parsed or {},
            "raw_output": raw_output,
            "is_valid": is_valid,
            "errors": errors,
            "generation_time_ms": elapsed_ms,
        }

    def generate_build(self, image) -> dict:
        """Full pipeline: image -> description -> build steps.

        Returns dict with description, steps, and metadata.
        """
        result = self.describe_image(image)

        # Convert description to build steps
        from backend.inference.postprocess_manual import json_to_steps
        steps = json_to_steps(result["description"]) if result["description"] else []

        return {
            "description": result["description"],
            "steps": steps,
            "metadata": {
                "model_version": "blip2-lego-lora-v1",
                "generation_time_ms": result["generation_time_ms"],
                "json_valid": result["is_valid"],
                "errors": result["errors"],
            },
        }
