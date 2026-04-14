"""Two-phase inference pipeline.

Stage 1: Image → text description (Qwen VL 7B)
Stage 2: Text → brick coordinates (Qwen 4B)
"""

import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.config import (
    UNIFIED_MODEL_NAME,
    UNIFIED_CHECKPOINT_DIR,
    STAGE1_CHECKPOINT_DIR,
    STAGE1_SYSTEM_PROMPT,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    LEGOGEN_DEV,
)


# ── Singletons ────────────────────────────────────────────────────────

_pipeline_instance = None


def get_pipeline():
    """Get or create the pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        if LEGOGEN_DEV:
            _pipeline_instance = MockPipeline()
        else:
            _pipeline_instance = TwoStagePipeline()
    return _pipeline_instance


def get_planner_pipeline():
    """Returns the same pipeline instance."""
    return get_pipeline()


# ── Mock pipeline for frontend development ─────────────────────────────

class MockPipeline:
    """Returns a realistic hardcoded brick response for dev/testing."""

    has_stage1 = True

    def generate_brick_build(self, caption: str) -> dict:
        """Mock text-to-bricks pipeline for dev mode."""
        from backend.brick.parser import serialize_brick, Brick

        bricks = [
            Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09"),
            Brick(h=2, w=4, x=2, y=0, z=0, color="0055BF"),
            Brick(h=2, w=4, x=4, y=0, z=0, color="C91A09"),
            Brick(h=2, w=4, x=1, y=0, z=1, color="FFFFFF"),
            Brick(h=2, w=4, x=3, y=0, z=1, color="FFFFFF"),
            Brick(h=2, w=4, x=0, y=0, z=2, color="FEC401"),
            Brick(h=2, w=4, x=2, y=0, z=2, color="FEC401"),
            Brick(h=2, w=4, x=4, y=0, z=2, color="FEC401"),
        ]

        return {
            "bricks": "\n".join(serialize_brick(b) for b in bricks),
            "caption": caption or "A small colorful house",
            "brick_count": len(bricks),
            "stable": True,
            "metadata": {
                "model_version": "mock-dev-v1",
                "generation_time_ms": 42,
                "rejections": 0,
                "rollbacks": 0,
            },
        }

    def generate_brick_build_from_image(self, image) -> dict:
        """Mock image-to-bricks pipeline for dev mode."""
        result = self.generate_brick_build("A small colorful house")
        result["caption"] = "A small colorful house"
        return result

    def describe_image_stage1(self, image) -> str:
        """Mock Stage 1: image → description."""
        return "A small colorful house with red walls, white trim, and a yellow roof."


# ── Two-Stage Pipeline ─────────────────────────────────────────────────

class TwoStagePipeline:
    """Two-phase pipeline: image → text description → brick coordinates.

    Stage 1: Uses Qwen VL model with LoRA adapter for image → text
    Stage 2: Uses Qwen 4B model for text → brick coordinates
    """

    def __init__(
        self,
        adapter_path: str | Path | None = None,
        model_name: str = UNIFIED_MODEL_NAME,
    ):
        import torch
        from backend.models.unified_model import LegoUnifiedModel

        if adapter_path is None:
            adapter_path = UNIFIED_CHECKPOINT_DIR

        adapter_path = Path(adapter_path)
        if adapter_path.exists():
            load_adapter = str(adapter_path)
        else:
            print(f"Checkpoint not found at {adapter_path}, using base model")
            load_adapter = None

        self.wrapper = LegoUnifiedModel(
            model_name=model_name,
            load_adapter=load_adapter,
            is_trainable=False,
        )
        self.model = self.wrapper.get_model()
        self.processor = self.wrapper.get_processor()
        self.model.eval()

        # Try to load Stage 1 adapter for two-stage pipeline
        stage1_path = Path(STAGE1_CHECKPOINT_DIR)
        self.has_stage1 = False
        if stage1_path.exists() and (stage1_path / "adapter_config.json").exists():
            self.has_stage1 = self.wrapper.load_named_adapter("stage1", stage1_path)
            if self.has_stage1:
                self.wrapper.set_adapter("default")
                print("Two-stage pipeline enabled (Stage 1 + Stage 2)")

    def describe_image_stage1(self, image) -> str:
        """Stage 1: Generate a structural description from an image."""
        import torch
        from backend.models.tokenizer import strip_thinking_blocks

        if self.has_stage1:
            self.wrapper.set_adapter("stage1")

        with torch.inference_mode():
            messages = [
                {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe this object for LEGO building."},
                    ],
                },
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )

            inputs = self.processor(
                text=[text], images=[image], return_tensors="pt",
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            raw = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        raw = strip_thinking_blocks(raw)

        # Switch back to default adapter
        if self.has_stage1:
            self.wrapper.set_adapter("default")

        return raw.strip()

    def generate_brick_build(self, caption: str) -> dict:
        """Stage 2: Generate brick coordinates from text using Qwen 4B."""
        from backend.inference.brick_pipeline import BrickPipeline
        if not hasattr(self, '_brick_pipeline'):
            self._brick_pipeline = BrickPipeline(device=str(self.model.device))
        return self._brick_pipeline.generate(caption)

    def generate_brick_build_from_image(self, image) -> dict:
        """Full two-stage: image → Stage 1 caption → Stage 2 bricks."""
        caption = self.describe_image_stage1(image)
        result = self.generate_brick_build(caption)
        result["caption"] = caption
        return result
