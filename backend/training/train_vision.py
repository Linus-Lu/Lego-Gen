#!/usr/bin/env python3
"""Fine-tune Qwen2.5-VL with QLoRA on LEGO image-to-JSON dataset.

Usage:
    python -m backend.training.train_vision
    python -m backend.training.train_vision --resume checkpoints/qwen-lego-lora/checkpoint-400
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (
    MODEL_NAME,
    DATA_DIR,
    CHECKPOINT_DIR,
    MAX_SEQ_LENGTH,
    LEARNING_RATE,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    NUM_EPOCHS,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    LOGGING_STEPS,
    EVAL_STEPS,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    USE_BF16,
    MAX_NEW_TOKENS,
)
from backend.models.vision_encoder import LegoVisionEncoder
from backend.models.tokenizer import get_json_prompt, decode_and_parse
from backend.data_pipeline.dataset import LegoDataset, load_splits
from backend.training.utils import (
    seed_everything,
    setup_wandb,
    compute_json_validity_rate,
    compute_all_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen2.5-VL LoRA for LEGO")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--output-dir", type=str, default=str(CHECKPOINT_DIR / "qwen-lego-lora"))
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


class JsonValidityCallback(TrainerCallback):
    """Logs JSON validity rate during evaluation."""

    def __init__(self, processor, num_samples: int = 20):
        self.processor = processor
        self.num_samples = num_samples

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        print(f"  Step {state.global_step}: evaluation complete")


def build_compute_metrics(processor):
    """Build a compute_metrics function for the Trainer."""

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # Decode predictions
        pred_strs = processor.tokenizer.batch_decode(
            preds, skip_special_tokens=True
        )

        # JSON validity
        validity = compute_json_validity_rate(pred_strs)

        # Parse and compute field metrics
        pred_dicts = []
        ref_dicts = []
        for pred_str in pred_strs:
            parsed = decode_and_parse(
                processor.tokenizer.encode(pred_str), processor.tokenizer
            )
            pred_dicts.append(parsed if parsed else {})

        # Labels: replace -100 with pad token before decoding
        labels[labels == -100] = processor.tokenizer.pad_token_id
        label_strs = processor.tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )
        for label_str in label_strs:
            parsed = decode_and_parse(
                processor.tokenizer.encode(label_str), processor.tokenizer
            )
            ref_dicts.append(parsed if parsed else {})

        metrics = compute_all_metrics(pred_dicts, ref_dicts)
        metrics["json_validity_rate"] = validity

        return metrics

    return compute_metrics


def main():
    args = parse_args()
    seed_everything(args.seed)

    # ── W&B ────────────────────────────────────────────────────────────
    if not args.no_wandb:
        setup_wandb("legogen-qwen25vl", config=vars(args))

    # ── Model ──────────────────────────────────────────────────────────
    print("Loading Qwen2.5-VL with QLoRA...")
    encoder = LegoVisionEncoder(
        model_name=MODEL_NAME,
        load_adapter=args.resume if args.resume else None,
    )
    model = encoder.get_model()
    processor = encoder.get_processor()
    encoder.print_trainable_params()

    # ── Dataset ────────────────────────────────────────────────────────
    print("Loading dataset...")
    data_dir = Path(args.data_dir)
    splits = load_splits(data_dir)

    train_ds = LegoDataset(
        set_nums=splits["train"],
        data_dir=data_dir,
        processor=processor,
        max_length=MAX_SEQ_LENGTH,
        split="train",
    )
    val_ds = LegoDataset(
        set_nums=splits["val"],
        data_dir=data_dir,
        processor=processor,
        max_length=MAX_SEQ_LENGTH,
        split="val",
    )
    print(f"  Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # ── Training arguments ─────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        bf16=USE_BF16,
        fp16=not USE_BF16 and torch.cuda.is_available(),
        optim="paged_adamw_8bit",
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if not args.no_wandb else "none",
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    # ── Trainer ────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[JsonValidityCallback(processor)],
    )

    # ── Train ──────────────────────────────────────────────────────────
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume)

    # ── Save final adapter ─────────────────────────────────────────────
    print("Saving final adapter...")
    encoder.save_adapter(args.output_dir)

    # ── Final evaluation ───────────────────────────────────────────────
    print("Running final evaluation...")
    metrics = trainer.evaluate()
    print("\nFinal Metrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
