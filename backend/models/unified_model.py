"""Unified Qwen3.5-9B model for both image-to-JSON and text-to-JSON LEGO generation.

Replaces the separate LegoVisionEncoder (vision_encoder.py) and LegoPlannerLM
(planner_lm.py) with a single model that handles both modalities via one LoRA adapter.
"""

from pathlib import Path

import torch
from transformers import (
    Qwen3_5ForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from backend.config import (
    UNIFIED_MODEL_NAME,
    QUANTIZATION_BITS,
    UNIFIED_CHECKPOINT_DIR,
    USE_BF16,
)

# Unified LoRA config — matches planner (Qwen3.5 hybrid arch needs all-linear)
UNIFIED_LORA_R = 64
UNIFIED_LORA_ALPHA = 128
UNIFIED_LORA_DROPOUT = 0.05
UNIFIED_LORA_TARGET_MODULES = "all-linear"


class LegoUnifiedModel:
    """Single Qwen3.5-9B model with one LoRA adapter for vision + text tasks.

    - Vision encoder is frozen (only LM layers are fine-tuned via LoRA)
    - For image inputs: processor produces pixel_values + text tokens
    - For text-only inputs: processor produces text tokens only (no pixel_values)
    """

    def __init__(
        self,
        model_name: str = UNIFIED_MODEL_NAME,
        load_adapter: str | None = None,
    ):
        self.model_name = model_name

        # ── Quantization config ────────────────────────────────────────
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # ── Load processor (handles both image and text) ──────────────
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=256 * 28 * 28,
            max_pixels=384 * 28 * 28,  # reduced from 512 to fit 32GB VRAM
        )

        # ── Load base model (multimodal: vision + LM) ─────────────────
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=self.bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        )

        # ── Enable input gradients for gradient checkpointing compat ──
        self.model.enable_input_require_grads()

        # ── Apply LoRA or load existing adapter ────────────────────────
        if load_adapter:
            self.model = PeftModel.from_pretrained(self.model, load_adapter)
        else:
            self._apply_lora()

        # ── Freeze vision encoder and any LoRA params on vision layers ─
        self._freeze_vision()

    def _apply_lora(self):
        """Apply LoRA adapters to language model layers."""
        lora_config = LoraConfig(
            r=UNIFIED_LORA_R,
            lora_alpha=UNIFIED_LORA_ALPHA,
            target_modules=UNIFIED_LORA_TARGET_MODULES,
            lora_dropout=UNIFIED_LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)

    def _freeze_vision(self):
        """Freeze all vision encoder parameters including any LoRA matrices."""
        for name, param in self.model.named_parameters():
            if "visual" in name or "vision" in name:
                param.requires_grad = False

    def get_model(self):
        return self.model

    def get_processor(self):
        return self.processor

    def get_tokenizer(self):
        return self.processor.tokenizer

    def save_adapter(self, path: str | Path | None = None):
        """Save only the LoRA adapter weights."""
        save_path = Path(path) if path else UNIFIED_CHECKPOINT_DIR
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(save_path))
        self.processor.save_pretrained(str(save_path))

    def print_trainable_params(self):
        """Print the number of trainable vs total parameters."""
        trainable = 0
        total = 0
        for _, param in self.model.named_parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
        pct = 100 * trainable / total if total > 0 else 0
        print(f"Trainable: {trainable:,} / {total:,} ({pct:.2f}%)")
