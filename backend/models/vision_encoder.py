"""Qwen3-VL model wrapper with QLoRA for LEGO image-to-JSON fine-tuning."""

from pathlib import Path

import torch
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from backend.config import (
    MODEL_NAME,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    QUANTIZATION_BITS,
    CHECKPOINT_DIR,
    USE_BF16,
)


class LegoVisionEncoder:
    """Wraps Qwen3-VL with QLoRA adapters for efficient fine-tuning."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
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

        # ── Load base model ────────────────────────────────────────────
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=self.bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        )

        # ── Freeze vision encoder ──────────────────────────────────────
        # Freeze by parameter name pattern (robust across transformers versions)
        for name, param in self.model.named_parameters():
            if "visual" in name or "vision" in name:
                param.requires_grad = False

        # ── Enable input gradients for gradient checkpointing compat ──
        self.model.enable_input_require_grads()

        # ── Apply LoRA or load existing adapter ────────────────────────
        if load_adapter:
            self.model = PeftModel.from_pretrained(self.model, load_adapter)
        else:
            self._apply_lora()

    def _apply_lora(self):
        """Apply LoRA adapters to the language model layers."""
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)

    def get_model(self):
        return self.model

    def get_processor(self):
        return self.processor

    def save_adapter(self, path: str | Path | None = None):
        """Save only the LoRA adapter weights (small, ~50MB)."""
        save_path = Path(path) if path else CHECKPOINT_DIR / "qwen-lego-lora"
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
