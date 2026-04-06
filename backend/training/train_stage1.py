#!/usr/bin/env python3
"""Stage 1 training: fine-tune Qwen3.5-27B LoRA on image → description.

Trains a lightweight LoRA adapter on the Stage 1 manifest so the model
learns to produce concise LEGO-relevant geometry descriptions from photos.

Usage:
    python -m backend.training.train_stage1
    python -m backend.training.train_stage1 --manifest data/stage1_manifest.json
    python -m backend.training.train_stage1 --no-wandb --epochs 1
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (
    DATA_DIR,
    LOGGING_STEPS,
    SAVE_TOTAL_LIMIT,
    STAGE1_BATCH_SIZE,
    STAGE1_CHECKPOINT_DIR,
    STAGE1_GRADIENT_ACCUMULATION,
    STAGE1_LEARNING_RATE,
    STAGE1_LORA_ALPHA,
    STAGE1_LORA_R,
    STAGE1_MAX_SEQ_LENGTH,
    STAGE1_NUM_EPOCHS,
    STAGE1_WARMUP_STEPS,
    UNIFIED_MODEL_NAME,
    USE_BF16,
    WEIGHT_DECAY,
)
from backend.data_pipeline.dataset_stage1 import Stage1Collator, Stage1Dataset
from backend.training.utils import seed_everything, setup_wandb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1 LoRA training: image → LEGO description"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(DATA_DIR / "stage1_manifest.json"),
        help="Path to stage1_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(STAGE1_CHECKPOINT_DIR),
        help="Directory to save adapter checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=STAGE1_NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=STAGE1_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def load_model_and_processor(model_name: str):
    """Load Qwen3.5-27B with 4-bit NF4 quantization and a Stage 1 LoRA adapter."""

    # ── 4-bit quantization ─────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Processor (handles image + text tokenization) ──────────────────
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )

    # ── Base model ─────────────────────────────────────────────────────
    # Import here to avoid loading torch.nn at module level before cuda check
    from transformers import Qwen3_5ForConditionalGeneration

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
    )
    model.enable_input_require_grads()

    # ── LoRA adapter ───────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=STAGE1_LORA_R,
        lora_alpha=STAGE1_LORA_ALPHA,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # Freeze vision encoder — only fine-tune language model layers
    for name, param in model.named_parameters():
        if "visual" in name or "vision" in name:
            param.requires_grad = False

    # Report trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    return model, processor


def main():
    args = parse_args()
    seed_everything(args.seed)

    # ── W&B ────────────────────────────────────────────────────────────
    if not args.no_wandb:
        setup_wandb("legogen-stage1", config=vars(args))

    # ── Model ──────────────────────────────────────────────────────────
    print(f"Loading {UNIFIED_MODEL_NAME} with 4-bit quantization + Stage 1 LoRA...")
    model, processor = load_model_and_processor(UNIFIED_MODEL_NAME)

    # ── Dataset ────────────────────────────────────────────────────────
    print(f"Loading Stage 1 manifest from {args.manifest} ...")
    train_ds = Stage1Dataset(
        manifest_path=args.manifest,
        processor=processor,
        max_length=STAGE1_MAX_SEQ_LENGTH,
        split="train",
    )
    val_ds = Stage1Dataset(
        manifest_path=args.manifest,
        processor=processor,
        max_length=STAGE1_MAX_SEQ_LENGTH,
        split="val",
    )
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")

    # ── Training arguments ─────────────────────────────────────────────
    effective_batch = STAGE1_BATCH_SIZE * STAGE1_GRADIENT_ACCUMULATION
    total_steps = (len(train_ds) // effective_batch) * args.epochs
    adaptive_save_steps = max(10, total_steps // 5)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=STAGE1_BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=STAGE1_GRADIENT_ACCUMULATION,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=min(STAGE1_WARMUP_STEPS, max(1, total_steps // 10)),
        weight_decay=WEIGHT_DECAY,
        bf16=USE_BF16,
        fp16=not USE_BF16 and torch.cuda.is_available(),
        optim="paged_adamw_8bit",
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=adaptive_save_steps,
        save_strategy="steps",
        save_steps=adaptive_save_steps,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if not args.no_wandb else "none",
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        # Avoid OOM during eval — only compute loss, not full logits
        prediction_loss_only=True,
    )

    # ── Trainer ────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=Stage1Collator(),
    )

    # ── Train ──────────────────────────────────────────────────────────
    print("Starting Stage 1 training...")
    trainer.train()

    # ── Save adapter ───────────────────────────────────────────────────
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving Stage 1 adapter to {output_path} ...")
    model.save_pretrained(str(output_path))
    processor.save_pretrained(str(output_path))
    print("Done.")

    # ── Final evaluation ───────────────────────────────────────────────
    print("Running final evaluation...")
    metrics = trainer.evaluate()
    print("\nFinal Metrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
