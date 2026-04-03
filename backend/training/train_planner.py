#!/usr/bin/env python3
"""Fine-tune Qwen3-8B with QLoRA on LEGO text-to-JSON dataset.

Trains on combined Rebrickable (1.6k upsampled) + StableText2Brick (40k) data.

Usage:
    python -m backend.training.train_planner
    python -m backend.training.train_planner --resume checkpoints/qwen-lego-planner-lora/checkpoint-400
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
    PLANNER_MODEL_NAME,
    DATA_DIR,
    PLANNER_CHECKPOINT_DIR,
    PLANNER_MAX_SEQ_LENGTH,
    PLANNER_LEARNING_RATE,
    PLANNER_NUM_EPOCHS,
    PLANNER_WARMUP_STEPS,
    BATCH_SIZE,
    WEIGHT_DECAY,
    LOGGING_STEPS,
    EVAL_STEPS,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    USE_BF16,
)
from backend.models.planner_lm import LegoPlannerLM
from backend.models.tokenizer import sample_prompt_template
from backend.data_pipeline.dataset_planner import (
    PlannerDataset,
    load_planner_splits,
)
from backend.training.utils import (
    seed_everything,
    setup_wandb,
    compute_json_validity_rate,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen3 LoRA for LEGO text-to-JSON")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--output-dir", type=str, default=str(PLANNER_CHECKPOINT_DIR))
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--epochs", type=int, default=PLANNER_NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=PLANNER_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    return parser.parse_args()


class EpochUpdateCallback(TrainerCallback):
    """Updates dataset epoch counter for curriculum scheduling."""

    def __init__(self, train_dataset: PlannerDataset):
        self.train_dataset = train_dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch else 0
        self.train_dataset.epoch = epoch
        print(f"  Epoch {epoch}: updated dataset prompt rotation")


def main():
    args = parse_args()
    seed_everything(args.seed)

    # ── W&B ────────────────────────────────────────────────────────────
    if not args.no_wandb:
        setup_wandb("legogen-planner", config=vars(args))

    # ── Model ──────────────────────────────────────────────────────────
    print("Loading Qwen3-8B with QLoRA...")
    planner = LegoPlannerLM(
        model_name=PLANNER_MODEL_NAME,
        load_adapter=args.resume if args.resume else None,
    )
    model = planner.get_model()
    tokenizer = planner.get_tokenizer()
    planner.print_trainable_params()

    # ── Dataset ────────────────────────────────────────────────────────
    print("Loading dataset...")
    data_dir = Path(args.data_dir)
    splits = load_planner_splits(data_dir)

    train_ds = PlannerDataset(
        rebrickable_ids=splits["rebrickable_train"],
        st2b_ids=splits["st2b_train"],
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=PLANNER_MAX_SEQ_LENGTH,
        split="train",
        rebrickable_upsample=3,
    )
    val_ds = PlannerDataset(
        rebrickable_ids=splits["rebrickable_val"],
        st2b_ids=splits["st2b_val"],
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=PLANNER_MAX_SEQ_LENGTH,
        split="val",
        rebrickable_upsample=1,
    )
    print(f"  Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # ── Training arguments ─────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=PLANNER_WARMUP_STEPS,
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
        callbacks=[EpochUpdateCallback(train_ds)],
    )

    # ── Train ──────────────────────────────────────────────────────────
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume)

    # ── Save final adapter ─────────────────────────────────────────────
    print("Saving final adapter...")
    planner.save_adapter(args.output_dir)

    # ── Final evaluation ───────────────────────────────────────────────
    print("Running final evaluation...")
    metrics = trainer.evaluate()
    print("\nFinal Metrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
