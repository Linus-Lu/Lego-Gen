"""Qwen3.5-9B text-only model wrapper with QLoRA for LEGO text-to-JSON fine-tuning."""

from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from backend.config import (
    PLANNER_MODEL_NAME,
    PLANNER_LORA_R,
    PLANNER_LORA_ALPHA,
    PLANNER_LORA_DROPOUT,
    PLANNER_LORA_TARGET_MODULES,
    QUANTIZATION_BITS,
    PLANNER_CHECKPOINT_DIR,
    USE_BF16,
)


class LegoPlannerLM:
    """Wraps Qwen3.5-9B with QLoRA adapters for text-to-JSON fine-tuning.

    Uses 'all-linear' target_modules to handle the hybrid DeltaNet/Attention
    architecture where different layer types have different linear projection names.
    """

    def __init__(
        self,
        model_name: str = PLANNER_MODEL_NAME,
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
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

    def _apply_lora(self):
        """Apply LoRA adapters to all linear layers (handles hybrid architecture)."""
        lora_config = LoraConfig(
            r=PLANNER_LORA_R,
            lora_alpha=PLANNER_LORA_ALPHA,
            target_modules=PLANNER_LORA_TARGET_MODULES,
            lora_dropout=PLANNER_LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def save_adapter(self, path: str | Path | None = None):
        """Save only the LoRA adapter weights."""
        save_path = Path(path) if path else PLANNER_CHECKPOINT_DIR
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))

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
