#!/usr/bin/env python3
"""Unified training: fine-tune one Qwen3.5-27B LoRA on both image→JSON and text→JSON.

Usage:
    python -m backend.training.train_unified
    python -m backend.training.train_unified --resume checkpoints/qwen35-27b-lego-unified-lora/checkpoint-400
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


class StructureAwareWeights:
    """Build per-token loss weights that upweight structural decisions.

    JSON labels are ~62% boilerplate (syntax + whitespace) and only ~1.4%
    structural layout tokens (layer names, positions, connects_to).
    Standard cross-entropy drowns the structural signal.

    This class pre-computes token ID sets for:
      - boilerplate (JSON syntax, whitespace) → weight 0.1
      - structural keywords (layer names, positions, connectivity) → weight 5.0
      - everything else (part IDs, colors, quantities) → weight 1.0
    """

    def __init__(self, tokenizer, boilerplate_weight=0.1, structure_weight=5.0):
        self.boilerplate_weight = boilerplate_weight
        self.structure_weight = structure_weight

        # Collect token IDs for JSON syntax characters
        boilerplate_ids: set[int] = set()
        for char in ["{", "}", "[", "]", ":", ",", '{"', '"}', '["', '"]',
                     '":', '",', "},", "],", "  ", "    ", "\n", "\n  ",
                     "\n    ", "\n      ", "\n        "]:
            ids = tokenizer.encode(char, add_special_tokens=False)
            boilerplate_ids.update(ids)

        # Also add common JSON field keys that repeat in every sample
        for key in ['"part_id"', '"name"', '"category"', '"color"',
                    '"color_hex"', '"is_trans"', '"quantity"', '"type"',
                    '"spatial"', '"subassemblies"', '"parts"',
                    '"dimensions_estimate"', '"dominant_colors"',
                    '"build_hints"', '"set_id"', '"object"',
                    '"subcategory"', '"complexity"', '"total_parts"',
                    '"width"', '"height"', '"depth"',
                    ': "', ': {', ': [', ': true', ': false']:
            ids = tokenizer.encode(key, add_special_tokens=False)
            boilerplate_ids.update(ids)

        # Structural keywords — the decisions that matter most
        structure_ids: set[int] = set()
        for word in ["layer_0", "layer_1", "layer_2", "layer_3", "layer_4",
                     "layer_5", "layer_6", "layer_7", "layer_8", "layer_9",
                     "bottom", "center", "top", "left", "right",
                     "connects_to", "position", "orientation",
                     "flat", "upright", "angled", "inverted"]:
            ids = tokenizer.encode(word, add_special_tokens=False)
            structure_ids.update(ids)

        # Remove any overlap (structure wins over boilerplate)
        boilerplate_ids -= structure_ids

        # Pre-compute sorted tensors for torch.isin() — O(1) per call instead of O(n*m)
        self.boilerplate_tensor = torch.tensor(sorted(boilerplate_ids), dtype=torch.long)
        self.structure_tensor = torch.tensor(sorted(structure_ids), dtype=torch.long)

        print(f"[StructureAwareWeights] boilerplate token IDs: {len(boilerplate_ids)}, "
              f"structure token IDs: {len(structure_ids)}")

    def get_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """Return per-token weights. Shape matches labels.

        Uses torch.isin() with pre-computed lookup tensors instead of
        iterating over each token ID. This is O(seq_len * log(n)) instead
        of O(seq_len * n_ids) per batch.
        """
        weights = torch.ones_like(labels, dtype=torch.float32)

        # Move lookup tensors to labels device (cached after first call)
        if self.boilerplate_tensor.device != labels.device:
            self.boilerplate_tensor = self.boilerplate_tensor.to(labels.device)
            self.structure_tensor = self.structure_tensor.to(labels.device)

        # Vectorized membership tests
        is_boilerplate = torch.isin(labels, self.boilerplate_tensor)
        is_structure = torch.isin(labels, self.structure_tensor)

        weights[is_boilerplate] = self.boilerplate_weight
        weights[is_structure] = self.structure_weight

        # Masked tokens (-100) get weight 0
        weights[labels == -100] = 0.0
        return weights


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


class MemoryCleanupCallback(TrainerCallback):
    """Free GPU memory after evaluation to prevent OOM on next training steps."""

    def on_evaluate(self, args, state, control, **kwargs):
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [MemoryCleanup] freed GPU cache after eval at step {state.global_step}")


class ProgressCallback(TrainerCallback):
    """Log a milestone every 10% of training."""

    def __init__(self):
        self.next_pct = 10

    def on_step_end(self, args, state, control, **kwargs):
        if state.max_steps <= 0:
            return
        pct = 100 * state.global_step / state.max_steps
        if pct >= self.next_pct:
            loss = state.log_history[-1].get("loss", "?") if state.log_history else "?"
            elapsed_h = (state.log_history[-1].get("epoch", 0) / state.num_train_epochs * 100) if state.log_history else pct
            print(f"\n{'='*60}")
            print(f"  MILESTONE: {int(self.next_pct)}% complete — "
                  f"step {state.global_step}/{state.max_steps}, loss={loss}")
            print(f"{'='*60}\n", flush=True)
            self.next_pct += 10


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
    print(f"Loading unified {UNIFIED_MODEL_NAME} model with LoRA...")
    model_wrapper = LegoUnifiedModel(
        model_name=UNIFIED_MODEL_NAME,
        load_adapter=args.resume if args.resume else None,
    )
    model = model_wrapper.get_model()
    processor = model_wrapper.get_processor()
    tokenizer = model_wrapper.get_tokenizer()
    model_wrapper.print_trainable_params()

    # ── Dataset ────────────────────────────────────────────────────────
    print("Loading Stage 2 dataset (ST2B text-only)...")
    data_dir = Path(args.data_dir)
    splits = load_unified_splits(data_dir)

    train_ds = UnifiedLegoDataset(
        vision_set_nums=[],          # Stage 2: text-only, no vision
        rebrickable_ids=[],          # Stage 2: ST2B-only, no Rebrickable
        st2b_ids=splits["st2b_train"],
        data_dir=data_dir,
        processor=processor,
        max_length=args.max_seq_length,
        split="train",
        rebrickable_upsample=1,
        vision_upsample=1,
    )
    val_ds = UnifiedLegoDataset(
        vision_set_nums=[],
        rebrickable_ids=[],
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

    # Cap val set with balanced modality coverage
    MAX_VAL_SAMPLES = 500
    if len(val_ds) > MAX_VAL_SAMPLES:
        vision_samples = [s for s in val_ds.samples if s[0] == "vision"]
        text_samples = [s for s in val_ds.samples if s[0] == "text"]
        # Keep all vision (minority class), fill rest with text
        n_vision = min(len(vision_samples), MAX_VAL_SAMPLES // 3)
        n_text = MAX_VAL_SAMPLES - n_vision
        balanced = vision_samples[:n_vision] + text_samples[:n_text]
        val_ds.samples = balanced
        print(f"  Capped val set to {len(balanced)} ({n_vision} vision, {n_text} text)")

    # ── Training arguments ─────────────────────────────────────────────
    effective_batch = args.batch_size * UNIFIED_GRADIENT_ACCUMULATION
    total_steps = (len(train_ds) // effective_batch) * args.epochs
    adaptive_save_steps = max(10, min(SAVE_STEPS, total_steps // 3))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,  # BS=1 for eval to avoid OOM from logit accumulation
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
        eval_accumulation_steps=4,  # accumulate fewer steps to reduce peak memory
    )

    # ── Structure-aware loss weighting ───────────────────────────────────
    structure_weights = StructureAwareWeights(tokenizer)

    # ── Curriculum sampler ──────────────────────────────────────────────
    curriculum_sampler = CurriculumSampler(train_ds, seed=args.seed)

    class CurriculumTrainer(Trainer):
        """Custom trainer with curriculum sampling and structure-aware loss."""

        def _get_train_sampler(self, *args, **kwargs):
            return curriculum_sampler

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            # Shift for causal LM: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Apply structure-aware weights
            weights = structure_weights.get_weights(shift_labels).view(-1)

            # Chunked cross-entropy to avoid OOM on large vocab
            CHUNK_SIZE = 256  # process 256 tokens at a time
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            seq_len = shift_logits.size(1)
            all_losses = []
            for i in range(0, seq_len, CHUNK_SIZE):
                chunk_logits = shift_logits[:, i:i+CHUNK_SIZE, :].reshape(-1, shift_logits.size(-1))
                chunk_labels = shift_labels[:, i:i+CHUNK_SIZE].reshape(-1)
                all_losses.append(loss_fct(chunk_logits, chunk_labels))
            per_token_loss = torch.cat(all_losses)

            weighted_loss = (per_token_loss * weights).sum() / weights.sum().clamp(min=1.0)

            return (weighted_loss, outputs) if return_outputs else weighted_loss

    # ── Trainer ────────────────────────────────────────────────────────
    # compute_metrics disabled for 27B model — accumulating full logits across
    # the 46k val set causes OOM.  eval_loss is tracked automatically.
    trainer = CurriculumTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=UnifiedCollator(),
        callbacks=[
            EpochUpdateCallback(train_ds, val_ds, sampler=curriculum_sampler),
            MemoryCleanupCallback(),
            ProgressCallback(),
        ],
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
