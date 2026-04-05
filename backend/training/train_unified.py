#!/usr/bin/env python3
"""Unified training: fine-tune one Qwen3.5-9B LoRA on both image→JSON and text→JSON.

Usage:
    python -m backend.training.train_unified
    python -m backend.training.train_unified --resume checkpoints/qwen35-lego-unified-lora/checkpoint-400
    python -m backend.training.train_unified --no-wandb --epochs 3
"""

import argparse
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
    UNIFIED_MODEL_NAME,
    UNIFIED_MAX_SEQ_LENGTH,
    UNIFIED_CHECKPOINT_DIR,
    UNIFIED_LEARNING_RATE,
    UNIFIED_NUM_EPOCHS,
    UNIFIED_WARMUP_STEPS,
    UNIFIED_BATCH_SIZE,
    UNIFIED_GRADIENT_ACCUMULATION,
    VISION_UPSAMPLE,
    WEIGHT_DECAY,
    LOGGING_STEPS,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    USE_BF16,
    DATA_DIR,
)
from backend.models.unified_model import LegoUnifiedModel
from backend.models.tokenizer import decode_and_parse
from backend.data_pipeline.dataset_unified import (
    UnifiedLegoDataset,
    UnifiedCollator,
    CurriculumSampler,
    load_unified_splits,
)
from backend.training.utils import (
    seed_everything,
    setup_wandb,
    compute_json_validity_rate,
    compute_all_metrics,
)


class EpochUpdateCallback(TrainerCallback):
    """Update dataset epoch counter for prompt rotation diversity."""

    def __init__(self, train_dataset, val_dataset=None, sampler=None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.sampler = sampler

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch else 0
        self.train_dataset.epoch = epoch
        if self.val_dataset is not None:
            self.val_dataset.epoch = epoch
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
        print(f"  Epoch {epoch}: updated dataset prompt rotation")


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Qwen3.5 LoRA training (vision + planner)")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--output-dir", type=str, default=str(UNIFIED_CHECKPOINT_DIR))
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--epochs", type=int, default=UNIFIED_NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=UNIFIED_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=UNIFIED_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--vision-upsample", type=int, default=VISION_UPSAMPLE)
    parser.add_argument("--rebrickable-upsample", type=int, default=3)
    parser.add_argument("--max-seq-length", type=int, default=UNIFIED_MAX_SEQ_LENGTH)
    return parser.parse_args()


def build_compute_metrics(tokenizer):
    """Build a compute_metrics function for the Trainer."""

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds are raw logits [batch, seq, vocab] — argmax to get token IDs
        if preds.ndim == 3:
            preds = preds.argmax(axis=-1)
        pred_strs = tokenizer.batch_decode(preds, skip_special_tokens=True)

        validity = compute_json_validity_rate(pred_strs)

        pred_dicts = []
        for pred_str in pred_strs:
            parsed = decode_and_parse(
                tokenizer.encode(pred_str), tokenizer
            )
            pred_dicts.append(parsed if parsed else {})

        labels[labels == -100] = tokenizer.pad_token_id
        label_strs = tokenizer.batch_decode(labels, skip_special_tokens=True)
        ref_dicts = []
        for label_str in label_strs:
            parsed = decode_and_parse(
                tokenizer.encode(label_str), tokenizer
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
        setup_wandb("legogen-unified", config=vars(args))

    # ── Model ──────────────────────────────────────────────────────────
    print("Loading unified Qwen3.5 model with QLoRA...")
    model_wrapper = LegoUnifiedModel(
        model_name=UNIFIED_MODEL_NAME,
        load_adapter=args.resume if args.resume else None,
    )
    model = model_wrapper.get_model()
    processor = model_wrapper.get_processor()
    tokenizer = model_wrapper.get_tokenizer()
    model_wrapper.print_trainable_params()

    # ── Dataset ────────────────────────────────────────────────────────
    print("Loading unified dataset (vision + planner)...")
    data_dir = Path(args.data_dir)
    splits = load_unified_splits(data_dir)

    train_ds = UnifiedLegoDataset(
        vision_set_nums=splits["vision_train"],
        rebrickable_ids=splits["rebrickable_train"],
        st2b_ids=splits["st2b_train"],
        data_dir=data_dir,
        processor=processor,
        max_length=args.max_seq_length,
        split="train",
        rebrickable_upsample=args.rebrickable_upsample,
        vision_upsample=args.vision_upsample,
    )
    val_ds = UnifiedLegoDataset(
        vision_set_nums=splits["vision_val"],
        rebrickable_ids=splits["rebrickable_val"],
        st2b_ids=splits["st2b_val"],
        data_dir=data_dir,
        processor=processor,
        max_length=args.max_seq_length,
        split="val",
        rebrickable_upsample=1,
        vision_upsample=1,
    )

    n_vision = len([s for s in train_ds.samples if s[0] == "vision"])
    n_text = len([s for s in train_ds.samples if s[0] == "text"])
    print(f"  Train: {len(train_ds)} samples ({n_vision} vision, {n_text} text)")
    print(f"  Val:   {len(val_ds)} samples")

    # ── Training arguments ─────────────────────────────────────────────
    effective_batch = args.batch_size * UNIFIED_GRADIENT_ACCUMULATION
    total_steps = (len(train_ds) // effective_batch) * args.epochs
    adaptive_save_steps = max(10, min(SAVE_STEPS, total_steps // 3))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=UNIFIED_GRADIENT_ACCUMULATION,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=min(UNIFIED_WARMUP_STEPS, total_steps // 5),
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
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        eval_accumulation_steps=8,
    )

    # ── Curriculum sampler ──────────────────────────────────────────────
    curriculum_sampler = CurriculumSampler(train_ds, seed=args.seed)

    class CurriculumTrainer(Trainer):
        """Override sampler so untruncated samples are seen first each epoch."""
        def _get_train_sampler(self):
            return curriculum_sampler

    # ── Trainer ────────────────────────────────────────────────────────
    trainer = CurriculumTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=UnifiedCollator(),
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=[EpochUpdateCallback(train_ds, val_ds, sampler=curriculum_sampler)],
    )

    # ── Train ──────────────────────────────────────────────────────────
    print("Starting unified training...")
    trainer.train(resume_from_checkpoint=args.resume)

    # ── Truncation report ─────────────────────────────────────────────
    print("\nTruncation Report:")
    for name, ds in [("Train", train_ds), ("Val", val_ds)]:
        stats = ds.truncation_stats
        print(f"  {name}: {stats['estimated_untruncated']} untruncated, "
              f"{stats['estimated_truncated']} truncated "
              f"({stats['estimated_truncation_pct']}%), "
              f"actual truncated during iteration: {stats['actual_truncated']}")

    # ── Save final adapter ─────────────────────────────────────────────
    print("\nSaving unified adapter...")
    model_wrapper.save_adapter(args.output_dir)

    # ── Final evaluation ───────────────────────────────────────────────
    print("Running final evaluation...")
    metrics = trainer.evaluate()
    print("\nFinal Metrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
