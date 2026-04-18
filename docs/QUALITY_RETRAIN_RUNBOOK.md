# LEGOGen Quality Retrain Runbook

This runbook is for the v2 Stage 2 brick model. It fixes three quality blockers before retraining: runtime-safe colors, multi-color supervision, and explicit `DONE` stopping.

## What Changed

- Stage 2 training data now ends every assistant answer with `DONE`.
- Caption colors are extracted with word boundaries and multiple colors are preserved instead of applying the first color to every brick.
- Training colors are normalized to the runtime palette, including `yellow -> F2CD37`.
- Runtime constrained decoding accepts either one brick line or `DONE`.
- Quality canary evaluation writes artifacts under `benchmark_runs/<timestamp>/`.

## Generate V2 Data

Run on the cloud box:

```bash
source /etc/network_turbo
cd /root/autodl-tmp/Lego-Gen

export HF_HOME=/root/autodl-tmp/cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/cache/hub

.venv/bin/python -m backend.data_pipeline.prepare_brick_dataset \
  --output-dir data/brick_training_v2 \
  --include-canary \
  --canary-repeat 20 \
  --palette-audit strict
```

`--palette-audit strict` must pass before training. If it fails, the data contains colors the runtime grammar cannot emit.

## One-GPU Gate

Use one 5090 first. This is only a broken-pipeline gate, not final model selection.

```bash
export CUDA_VISIBLE_DEVICES=0
export LEGOGEN_DEV=0
export BRICK_TRAINING_DATA=/root/autodl-tmp/Lego-Gen/data/brick_training_v2

.venv/bin/python -m backend.training.train_brick \
  --data-dir data/brick_training_v2 \
  --output-dir backend/models/checkpoints/qwen35-4b-brick-lora-v2-canary \
  --max-steps 80 \
  --train-samples 4096 \
  --eval-samples 256 \
  --resume none \
  --save-total-limit 5 \
  --no-wandb
```

The first run with a new data/tokenizer/subset combination builds
`data/brick_training_v2/.tokenized_cache/`. Later runs with the same settings
load that cache and skip the expensive tokenization pass. Use
`--rebuild-tokenized-cache` only when you intentionally want to regenerate the
same cache key.

Evaluate:

```bash
export BRICK_CHECKPOINT_DIR=/root/autodl-tmp/Lego-Gen/backend/models/checkpoints/qwen35-4b-brick-lora-v2-canary

.venv/bin/python scripts/eval_generation_quality.py \
  --timestamp 20260418_v2_canary_quality \
  --prompts benchmarks/prompts/quality_canary_prompts.jsonl \
  --max-bricks-per-sample 64 \
  --sample-timeout-s 120 \
  --stability-check-interval 8
```

Gate criteria:

- `quality_raw.jsonl` has successful rows.
- Outputs have `brick_count > 0`.
- `parse_valid=true`, `collision_free=true`, and `export_success=true`.
- Expected color coverage is nonzero on multi-color prompts.
- The run does not crash or hang.

## Full 2x5090 Retrain

Start clean from the base model, not from the old checkpoint.

```bash
export CUDA_VISIBLE_DEVICES=0,1
export LEGOGEN_DEV=0
export BRICK_TRAINING_DATA=/root/autodl-tmp/Lego-Gen/data/brick_training_v2

torchrun --nproc_per_node=2 -m backend.training.train_brick \
  --data-dir data/brick_training_v2 \
  --output-dir backend/models/checkpoints/qwen35-4b-brick-lora-v2 \
  --epochs 3 \
  --eval-samples 1024 \
  --eval-steps 500 \
  --save-steps 500 \
  --gradient-accumulation-steps 8 \
  --resume none \
  --save-total-limit 5
```

The trainer uses a deterministic eval subset by default so checkpoint eval does
not turn into a full-test-set bottleneck every few hundred optimizer steps. Use
the separate quality eval below for final checkpoint selection. On 8x5090, keep
the effective batch close to the single-GPU default by using
`--gradient-accumulation-steps 2`.

On 96GB RTX PRO 6000 nodes, run a short speed/OOM test before full training with
`--batch-size 2 --gradient-accumulation-steps 1 --no-gradient-checkpointing`.
If that is stable, it should train faster than the 5090-safe checkpointing
configuration.

Evaluate promising checkpoints with both:

```bash
.venv/bin/python scripts/eval_generation_quality.py \
  --timestamp 20260418_v2_full_quality \
  --prompts benchmarks/prompts/quality_canary_prompts.jsonl \
  --max-bricks-per-sample 64 \
  --sample-timeout-s 120 \
  --stability-check-interval 8

.venv/bin/python scripts/benchmark_legogen.py \
  --quick-smoke \
  --timestamp 20260418_v2_full_quick_smoke
```

## Selection Rule

Do not choose the deploy checkpoint by eval loss alone. Prefer the checkpoint that:

- Stops with `DONE` on simple prompts.
- Does not frequently hit `max_bricks`.
- Preserves expected colors on canary prompts.
- Parses, avoids collisions, and exports successfully.
- Has acceptable stability without excessive rollbacks.
