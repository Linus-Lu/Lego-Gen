#!/usr/bin/env python3
"""Stage 1 training: fine-tune Qwen3.5-9B LoRA on image → description.

Trains a lightweight LoRA adapter on the Stage 1 manifest so the model
learns to produce concise LEGO-relevant geometry descriptions from photos.

Optimized for multi-GPU (4x RTX 5090) with DDP.

Usage:
    # Single GPU:
    python -m backend.training.train_stage1

    # Multi-GPU (4x RTX 5090):
    torchrun --nproc_per_node=4 -m backend.training.train_stage1

    # With local model path (no internet needed):
    torchrun --nproc_per_node=4 -m backend.training.train_stage1 --model-path /root/autodl-tmp/models/Qwen3.5-9B

    # AutoDL / China (uses hf-mirror.com automatically):
    HF_ENDPOINT=https://hf-mirror.com torchrun --nproc_per_node=4 -m backend.training.train_stage1
"""

import argparse
import gc
import os
import sys
from pathlib import Path

# ── HuggingFace mirror for China (AutoDL, etc.) ───────────────────────
# Set before any HF imports so the SDK picks it up immediately.
# Only probe on rank 0 to avoid 4 parallel socket checks.
if not os.environ.get("HF_ENDPOINT") and not os.environ.get("HF_HUB_OFFLINE"):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        import socket
        try:
            socket.create_connection(("huggingface.co", 443), timeout=3)
        except OSError:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            print("[AutoDL] HuggingFace unreachable, using hf-mirror.com")

# ── Disable W&B on non-main ranks BEFORE any import can trigger init ──
_local_rank = int(os.environ.get("LOCAL_RANK", -1))
if _local_rank > 0:
    os.environ["WANDB_DISABLED"] = "true"

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
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


# ── DDP helpers ───────────────────────────────────────────────────────

def _is_torchrun() -> bool:
    """True when launched via torchrun (LOCAL_RANK env var is set)."""
    return "LOCAL_RANK" in os.environ


def _init_distributed():
    """Initialize the process group early so we can use barriers during
    model loading.  No-op if already initialized or not using torchrun."""
    if not _is_torchrun():
        return
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


# ── Callbacks ─────────────────────────────────────────────────────────

class MemoryCleanupCallback(TrainerCallback):
    """Free GPU memory after evaluation to prevent OOM during training."""

    def on_evaluate(self, args, state, control, **kwargs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ProgressCallback(TrainerCallback):
    """Log milestone progress every 10%."""

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.last_pct = -1

    def on_step_end(self, args, state, control, **kwargs):
        if self.total_steps <= 0:
            return
        pct = int(state.global_step / self.total_steps * 100)
        if pct % 10 == 0 and pct != self.last_pct:
            self.last_pct = pct
            if state.is_local_process_zero:
                print(f"  [{pct}%] Step {state.global_step}/{self.total_steps}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1 LoRA training: image → LEGO description (multi-GPU)"
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
    parser.add_argument("--batch-size", type=int, default=STAGE1_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Local path to pre-downloaded model (skips HF download)",
    )
    return parser.parse_args()


def load_model_and_processor(model_name: str, local_rank: int):
    """Load Qwen3.5-9B with 4-bit NF4 quantization and a Stage 1 LoRA adapter.

    DDP-safe: rank 0 downloads first, others wait at a barrier, then all
    load from the HF cache simultaneously onto their own GPU.
    """
    is_distributed = torch.distributed.is_initialized()

    # ── DDP: rank 0 downloads, others wait ─────────────────────────────
    if is_distributed and local_rank != 0:
        torch.distributed.barrier()

    # ── 4-bit quantization ─────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Processor (lower resolution — description task doesn't need 512 tiles)
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=128 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )

    # ── Base model — each rank loads onto its own GPU ──────────────────
    from transformers import Qwen3_5ForConditionalGeneration

    target_device = local_rank if is_distributed else 0
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": target_device},
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
    )
    model.enable_input_require_grads()

    # ── DDP: rank 0 done downloading, release others ───────────────────
    if is_distributed and local_rank == 0:
        torch.distributed.barrier()

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

    # NOTE: torch.compile() is incompatible with quantized model training
    # (HF Trainer rejects it). Only useful at inference time.

    # Report trainable parameters (rank 0 only)
    if local_rank <= 0:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    return model, processor


def main():
    args = parse_args()
    seed_everything(args.seed)

    # ── Initialize DDP early so barriers work during model loading ──────
    _init_distributed()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank <= 0

    # Disable W&B on non-main ranks (belt-and-suspenders with top-of-file)
    if not is_main or args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"

    if is_main:
        print(f"{'=' * 60}")
        print(f"  Stage 1 Training: Image -> LEGO Description")
        print(f"  Model: {args.model_path or UNIFIED_MODEL_NAME}")
        print(f"  GPUs: {world_size}")
        print(f"  Per-device batch: {args.batch_size}")
        print(f"  Grad accumulation: {STAGE1_GRADIENT_ACCUMULATION}")
        print(f"  Effective batch: {args.batch_size * world_size * STAGE1_GRADIENT_ACCUMULATION}")
        print(f"  Epochs: {args.epochs}")
        print(f"  LR: {args.lr}")
        print(f"  LoRA: r={STAGE1_LORA_R}, alpha={STAGE1_LORA_ALPHA}")
        print(f"{'=' * 60}")

    # ── W&B (rank 0 only) ─────────────────────────────────────────────
    if not args.no_wandb and is_main:
        setup_wandb("legogen-stage1", config={
            **vars(args),
            "world_size": world_size,
            "effective_batch": args.batch_size * world_size * STAGE1_GRADIENT_ACCUMULATION,
        })

    # ── Model ──────────────────────────────────────────────────────────
    model_name = args.model_path or UNIFIED_MODEL_NAME
    if is_main:
        print(f"Loading {model_name} with 4-bit quantization + Stage 1 LoRA...")
    model, processor = load_model_and_processor(
        model_name,
        local_rank=max(local_rank, 0),
    )

    # ── Dataset ────────────────────────────────────────────────────────
    if is_main:
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
    if is_main:
        print(f"  Train: {len(train_ds)} samples")
        print(f"  Val:   {len(val_ds)} samples")

    # ── Training arguments ─────────────────────────────────────────────
    effective_batch = args.batch_size * world_size * STAGE1_GRADIENT_ACCUMULATION
    total_steps = (len(train_ds) // effective_batch) * args.epochs
    adaptive_save_steps = max(10, total_steps // 5)

    if is_main:
        print(f"  Total steps: {total_steps}")
        print(f"  Save/eval every: {adaptive_save_steps} steps")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=STAGE1_GRADIENT_ACCUMULATION,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=min(STAGE1_WARMUP_STEPS, max(1, total_steps // 10)),
        weight_decay=WEIGHT_DECAY,
        bf16=USE_BF16,
        fp16=not USE_BF16 and torch.cuda.is_available(),
        optim="adamw_torch",
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=adaptive_save_steps,
        save_strategy="steps",
        save_steps=adaptive_save_steps,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if (not args.no_wandb and is_main) else "none",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        prediction_loss_only=True,
        # Multi-GPU: DDP settings
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=50,
    )

    # ── Trainer ────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=Stage1Collator(),
        callbacks=[
            MemoryCleanupCallback(),
            ProgressCallback(total_steps),
        ],
    )

    # ── Train ──────────────────────────────────────────────────────────
    if is_main:
        print("Starting Stage 1 training...")
    trainer.train()

    # ── Save adapter ───────────────────────────────────────────────────
    if is_main:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving Stage 1 adapter to {output_path} ...")
        model.save_pretrained(str(output_path))
        processor.save_pretrained(str(output_path))
        print("Done.")

    # ── Final evaluation ───────────────────────────────────────────────
    if is_main:
        print("Running final evaluation...")
        metrics = trainer.evaluate()
        print("\nFinal Metrics:")
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
