"""Fine-tune Qwen3.5-4B with LoRA for text → colored brick sequence generation.

Features ported from train_unified.py:
  - Structure-aware loss weighting (upweight coordinates/dimensions)
  - Curriculum ordering (untruncated samples first)
  - Chunked cross-entropy (avoids OOM on 248K vocab)
"""

import argparse
import inspect
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    TrainerCallback,
)
from trl import SFTTrainer

# Check if SFTConfig exists (TRL >= 0.12)
try:
    from trl import SFTConfig
except ImportError:
    SFTConfig = None

from backend.config import (
    BRICK_MODEL_NAME, BRICK_CHECKPOINT_DIR, BRICK_LEARNING_RATE,
    BRICK_BATCH_SIZE, BRICK_GRADIENT_ACCUMULATION, BRICK_MAX_SEQ_LENGTH,
    BRICK_NUM_EPOCHS, BRICK_LORA_R, BRICK_LORA_ALPHA, BRICK_LORA_DROPOUT,
    BRICK_TRAINING_DATA,
)


# ── Structure-aware loss for brick format ─────────────────────────────────

class BrickStructureWeights:
    """Per-token loss weights for brick coordinate format.

    Brick format: "2x4 (5,3,0) #C91A09"
      - Boilerplate: parentheses, commas, 'x', '#', newlines, spaces → weight 0.1
      - Coordinates/dimensions (the actual numbers) → weight 1.0 (default)
      - Structural tokens: dimension combos like "2x4", coordinate digits → weight 3.0
    """

    def __init__(self, tokenizer, boilerplate_weight=0.1, structure_weight=3.0):
        self.boilerplate_weight = boilerplate_weight
        self.structure_weight = structure_weight

        # Boilerplate: syntax characters that repeat in every brick line
        self.boilerplate_ids: set[int] = set()
        for char in ["(", ")", ",", "x", "#", "\n", " ", "(,", "),",
                     "  ", "    ", "\n\n"]:
            ids = tokenizer.encode(char, add_special_tokens=False)
            self.boilerplate_ids.update(ids)

        # Structure: brick dimensions and coordinate patterns
        self.structure_ids: set[int] = set()
        for word in ["2x4", "4x2", "2x6", "6x2", "1x2", "2x1",
                     "1x4", "4x1", "1x6", "6x1", "1x8", "8x1",
                     "1x1", "2x2"]:
            ids = tokenizer.encode(word, add_special_tokens=False)
            self.structure_ids.update(ids)

        # Remove overlap (structure wins)
        self.boilerplate_ids -= self.structure_ids

        print(f"[BrickStructureWeights] boilerplate token IDs: {len(self.boilerplate_ids)}, "
              f"structure token IDs: {len(self.structure_ids)}")

    def get_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """Return per-token weights. Shape matches labels. Vectorized."""
        weights = torch.ones_like(labels, dtype=torch.float32)

        if self.boilerplate_ids:
            bp_ids = torch.tensor(list(self.boilerplate_ids), device=labels.device)
            bp_mask = (labels.unsqueeze(-1) == bp_ids).any(-1)
            weights[bp_mask] = self.boilerplate_weight

        if self.structure_ids:
            st_ids = torch.tensor(list(self.structure_ids), device=labels.device)
            st_mask = (labels.unsqueeze(-1) == st_ids).any(-1)
            weights[st_mask] = self.structure_weight

        # Masked tokens (-100) get weight 0
        weights[labels == -100] = 0.0
        return weights


# ── Curriculum dataset & sampler ──────────────────────────────────────────

def apply_curriculum_ordering(hf_dataset, max_seq_length: int):
    """Reorder HF dataset: untruncated samples first, then truncated.

    Returns the reordered dataset and the count of untruncated samples.
    """
    PROMPT_OVERHEAD = 80
    CHARS_PER_TOKEN = 3.0

    fits, truncated = [], []
    for i in range(len(hf_dataset)):
        messages = hf_dataset[i]["messages"]
        assistant_len = len(messages[-1]["content"]) if messages else 0
        est_tokens = int(assistant_len / CHARS_PER_TOKEN) + PROMPT_OVERHEAD
        if est_tokens <= max_seq_length:
            fits.append(i)
        else:
            truncated.append(i)

    # Reorder the HF dataset directly (preserves all HF Dataset methods)
    ordered_indices = fits + truncated
    reordered = hf_dataset.select(ordered_indices)

    print(f"  Curriculum: {len(fits)} untruncated, "
          f"{len(truncated)} truncated ({len(truncated)*100/max(1,len(ordered_indices)):.1f}%) "
          f"at max_seq_length={max_seq_length}")

    return reordered, len(fits)


class MemoryCleanupCallback(TrainerCallback):
    """Free GPU memory after evaluation."""

    def on_evaluate(self, args, state, control, **kwargs):
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _inspect_params(cls):
    """Return set of parameter names accepted by cls.__init__."""
    try:
        return set(inspect.signature(cls.__init__).parameters.keys())
    except (ValueError, TypeError):
        return set()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path to resume from, or 'auto' for latest")
    args, _ = parser.parse_known_args()

    # Auto-detect latest checkpoint
    if args.resume is None or args.resume == "auto":
        ckpt_dir = Path(str(BRICK_CHECKPOINT_DIR))
        if ckpt_dir.exists():
            checkpoints = sorted(
                ckpt_dir.glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[-1])
                if p.name.split("-")[-1].isdigit() else 0)
            if checkpoints:
                args.resume = str(checkpoints[-1])
                print(f"Auto-resuming from: {args.resume}", flush=True)
            else:
                args.resume = None
        else:
            args.resume = None

    train_path = BRICK_TRAINING_DATA / "train.jsonl"
    test_path = BRICK_TRAINING_DATA / "test.jsonl"

    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run prepare_brick_dataset.py first.")
        sys.exit(1)

    # Print TRL/transformers versions for debugging
    import trl, transformers
    print(f"TRL version: {trl.__version__}", flush=True)
    print(f"Transformers version: {transformers.__version__}", flush=True)

    print(f"Loading model: {BRICK_MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BRICK_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BRICK_MODEL_NAME, torch_dtype="auto", trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=BRICK_LORA_R, lora_alpha=BRICK_LORA_ALPHA,
        lora_dropout=BRICK_LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        use_dora=True,
        use_rslora=True,
        init_lora_weights="pissa",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    print("Loading datasets...", flush=True)
    ds = load_dataset("json", data_files={
        "train": str(train_path),
        "test": str(test_path),
    })

    # ── Curriculum ordering ──────────────────────────────────────────
    print("Building curriculum ordering...", flush=True)
    train_ds, n_untruncated = apply_curriculum_ordering(ds["train"], BRICK_MAX_SEQ_LENGTH)

    # ── Structure-aware loss weights ─────────────────────────────────
    structure_weights = BrickStructureWeights(tokenizer)

    output_dir = str(BRICK_CHECKPOINT_DIR)

    # ── Build training args (version-adaptive) ────────────────────────
    base_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=BRICK_NUM_EPOCHS,
        per_device_train_batch_size=BRICK_BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=BRICK_GRADIENT_ACCUMULATION,
        learning_rate=BRICK_LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        max_grad_norm=0.5,
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
    )

    # Decide which config class to use
    ConfigClass = SFTConfig if SFTConfig is not None else TrainingArguments
    config_params = _inspect_params(ConfigClass)

    config_kwargs = dict(base_kwargs)
    if "max_seq_length" in config_params:
        config_kwargs["max_seq_length"] = BRICK_MAX_SEQ_LENGTH
    elif "max_length" in config_params:
        config_kwargs["max_length"] = BRICK_MAX_SEQ_LENGTH
    if "dataset_text_field" in config_params:
        config_kwargs["dataset_text_field"] = None

    print(f"Using config class: {ConfigClass.__name__}", flush=True)
    training_args = ConfigClass(**config_kwargs)

    # ── Build custom trainer with structure-aware loss + curriculum ───
    class BrickTrainer(SFTTrainer):
        """SFTTrainer with structure-aware loss and chunked cross-entropy."""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            # Shift labels for causal LM (predict next token)
            shift_labels = labels[..., 1:].contiguous()
            seq_len = shift_labels.size(1)

            # Apply structure-aware weights
            weights = structure_weights.get_weights(shift_labels).view(-1)

            # Cross-entropy loss — single vectorized call
            shift_logits = logits[:, :seq_len, :].contiguous().view(-1, logits.size(-1))
            flat_labels = shift_labels.view(-1)
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            per_token_loss = loss_fct(shift_logits, flat_labels)

            weighted_loss = (per_token_loss * weights).sum() / weights.sum().clamp(min=1.0)

            return (weighted_loss, outputs) if return_outputs else weighted_loss

    # ── Build trainer kwargs (version-adaptive) ──────────────────────
    trainer_params = _inspect_params(SFTTrainer)
    print(f"SFTTrainer accepts: processing_class={'processing_class' in trainer_params}, "
          f"tokenizer={'tokenizer' in trainer_params}, "
          f"max_seq_length={'max_seq_length' in trainer_params}", flush=True)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=ds["test"],
        callbacks=[
            MemoryCleanupCallback(),
        ],
    )

    # Pass tokenizer with the right param name
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    # Pass max_seq_length to trainer if config didn't accept it
    if "max_seq_length" in trainer_params and "max_seq_length" not in config_kwargs:
        trainer_kwargs["max_seq_length"] = BRICK_MAX_SEQ_LENGTH

    trainer = BrickTrainer(**trainer_kwargs)

    print("Starting training...", flush=True)
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
