"""Stage 1 inference: image → short LEGO-relevant description.

Loads Qwen3.5-9B (+ Stage 1 LoRA if present) in 4-bit NF4 and runs a single
image-to-caption forward pass. The caption is then fed into BrickPipeline
for Stage 2 brick-coordinate generation.
"""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.config import (
    STAGE1_MODEL_NAME,
    STAGE1_CHECKPOINT_DIR,
    STAGE1_SYSTEM_PROMPT,
    USE_BF16,
)


STAGE1_USER_PROMPT = "Describe this object for LEGO building."
STAGE1_MAX_NEW_TOKENS = 256
STAGE1_TEMPERATURE = 0.7
STAGE1_TOP_P = 0.9


def _strip_thinking_blocks(text: str) -> str:
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class Stage1Pipeline:
    """Qwen3.5-9B + Stage 1 LoRA: produces a short geometry description from an image."""

    def __init__(self) -> None:
        import torch
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen3_5ForConditionalGeneration,
        )
        from peft import PeftModel

        self.processor = AutoProcessor.from_pretrained(
            STAGE1_MODEL_NAME,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            STAGE1_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        )

        ckpt = Path(STAGE1_CHECKPOINT_DIR)
        if ckpt.exists() and (ckpt / "adapter_config.json").exists():
            self.model = PeftModel.from_pretrained(base, str(ckpt))
            print(f"[stage1] Loaded LoRA adapter from {ckpt}")
        else:
            self.model = base
            print(f"[stage1] No adapter at {ckpt}, running base model")

        self.model.eval()

    def describe(self, image) -> str:
        """Run Stage 1: image → short description string."""
        import torch

        messages = [
            {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": STAGE1_USER_PROMPT},
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

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=STAGE1_MAX_NEW_TOKENS,
                temperature=STAGE1_TEMPERATURE,
                top_p=STAGE1_TOP_P,
                do_sample=True,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return _strip_thinking_blocks(raw)
