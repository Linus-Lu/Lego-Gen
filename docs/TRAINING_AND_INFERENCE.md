# Training and Inference

Reference documentation for the LEGOGen training pipeline and the runtime
inference stack. Reflects the brick-only state of the tree after the legacy
JSON path was removed.

---

## 1. Two-Stage Architecture

The runtime decomposes "image → buildable LEGO" into two independently
fine-tuned Qwen stages followed by a runtime-only physics validation layer.

| Stage | Job | Base model | Input | Output |
|-------|-----|-----------|-------|--------|
| Stage 1 | Perceive | `Qwen/Qwen3.5-9B` (multimodal)       | RGB image      | 1–3 sentence geometry / colour / scale description |
| Stage 2 | Build   | `Qwen/Qwen3.5-4B`                    | Text caption   | Newline-separated `HxW (x,y,z) #RRGGBB` sequence    |

Two entry points live on top of those stages:

- **Image → Bricks** — Stage 1 caption feeds Stage 2; returns placed bricks.
- **Text → Bricks** — skip Stage 1; user prompt goes straight to Stage 2.

There is no JSON head and no structured-description path. The 10-check
stability checker, constraint engine, and post-hoc validator are gone —
their role is now filled by per-brick grammar + voxel + LP enforcement
**during** generation.

---

## 2. Configuration

All training and inference knobs live in `backend/config.py`.

### 2.1 Model IDs

| Constant | Value | Used by |
|----------|-------|---------|
| `STAGE1_MODEL_NAME` | `"Qwen/Qwen3.5-9B"` | Stage 1 training + `Stage1Pipeline` |
| `BRICK_MODEL_NAME`  | `"Qwen/Qwen3.5-4B"` | Stage 2 training + `BrickPipeline` |

### 2.2 Checkpoint paths

- Stage 1 LoRA adapter: `CHECKPOINT_DIR / "qwen35-9b-lego-stage1-lora"` → `STAGE1_CHECKPOINT_DIR`
- Stage 2 Brick adapter: `CHECKPOINT_DIR / "qwen35-4b-brick-lora"` → `BRICK_CHECKPOINT_DIR`

### 2.3 Hyperparameters (verbatim from `config.py`)

**Stage 1**

| Knob | Value |
|------|-------|
| `STAGE1_LORA_R` | `32` |
| `STAGE1_LORA_ALPHA` | `64` |
| `STAGE1_LEARNING_RATE` | `5e-5` |
| `STAGE1_BATCH_SIZE` | `8` (per GPU; effective = 8 × N_GPUS with `gradient_accumulation=1`) |
| `STAGE1_MAX_SEQ_LENGTH` | `512` |
| `STAGE1_NUM_EPOCHS` | `3` |
| `STAGE1_WARMUP_STEPS` | `100` |

**Stage 2 Brick**

| Knob | Value |
|------|-------|
| `BRICK_LORA_R` | `32` |
| `BRICK_LORA_ALPHA` | `64` |
| `BRICK_LORA_DROPOUT` | `0.05` |
| `BRICK_LEARNING_RATE` | `1e-3` |
| `BRICK_BATCH_SIZE` | `1` (effective 16 with `BRICK_GRADIENT_ACCUMULATION=16`) |
| `BRICK_MAX_SEQ_LENGTH` | `4096` |
| `BRICK_NUM_EPOCHS` | `3` |

**Inference**

- `INFERENCE_TIMEOUT_SECONDS = 120` (route-level timeout)
- Stage 1 generation: `max_new_tokens=256`, `temperature=0.7`, `top_p=0.9`
- Stage 2 generation: `BASE_TEMPERATURE=0.6` (fixed; no ramp)

---

## 3. Training

### 3.1 Stage 1 — Image → Description (`backend/training/train_stage1.py`)

Fine-tunes a LoRA adapter on the 9B Qwen3.5 multimodal model so it emits a
short, LEGO-relevant caption from a photograph.

**Entry points**:

```bash
# Single GPU
python -m backend.training.train_stage1

# Multi-GPU (author targets 4× RTX 5090)
torchrun --nproc_per_node=4 -m backend.training.train_stage1

# With a pre-downloaded base model
torchrun --nproc_per_node=4 -m backend.training.train_stage1 \
  --model-path /root/autodl-tmp/models/Qwen3.5-9B

# AutoDL / China — falls back to hf-mirror.com automatically
HF_ENDPOINT=https://hf-mirror.com torchrun --nproc_per_node=4 \
  -m backend.training.train_stage1
```

**Data** — `Stage1Dataset` / `Stage1Collator` in
`backend/data_pipeline/dataset_stage1.py`. Reads
`data/stage1_manifest.json` (list of `{image_path, description, category,
source}`) built by `backend/data_pipeline/build_stage1_dataset.py`, which
matches COCO images with StableText2Brick captions through the
`COCO_TO_ST2B_CATEGORY` map in `config.py`. Image augmentations come from
`TRAIN_TRANSFORMS` / `VAL_TRANSFORMS` in `backend/data_pipeline/dataset.py`.

The collator stacks `input_ids`, `attention_mask`, `labels`,
`mm_token_type_ids` and concatenates `pixel_values`. The dataset masks all
tokens before the last `"<|im_start|>assistant\n"` marker with `-100` so
loss is only computed on the description, not the prompt.

**LoRA config** — `r=32`, `alpha=64`, `target_modules="all-linear"`,
`dropout=0.05`, with `use_dora=True` and `use_rslora=True`. Vision encoder
parameters (names containing `visual` / `vision`) are frozen.

PiSSA is intentionally **not** enabled on Stage 1: PiSSA initializes the
adapter by SVD on the full-precision base weights, which is incompatible
with the 4-bit NF4 quantized load used here. DoRA and rsLoRA work on top
of the quantized base without the SVD step.

**Trainer settings** (HuggingFace `Trainer` + `TrainingArguments`):

- `lr_scheduler_type="cosine"`, warmup = `max(1, total_steps // 10)` (capped at 100)
- `gradient_checkpointing=True`, `use_reentrant=False`
- `ddp_bucket_cap_mb=50`, `ddp_find_unused_parameters=False`
- `save_total_limit=2`, `load_best_model_at_end=True`
- W&B logging via `backend.training.utils.setup_wandb` (auto-disabled on non-rank-0)

Checkpoint written to `STAGE1_CHECKPOINT_DIR` (+ `adapter_config.json` so
`Stage1Pipeline` can detect its presence).

### 3.2 Stage 2 Brick — Text → Brick coordinates (`backend/training/train_brick.py`)

Fine-tunes a LoRA adapter on Qwen3.5-4B to emit the brick sequence format
consumed by `BrickPipeline`.

**Entry point**:

```bash
python -m backend.training.train_brick \
  --output-dir backend/models/checkpoints/qwen35-4b-brick-lora/ \
  --epochs 3 \
  [--resume none|auto|/path/to/checkpoint]
  [--data-dir data/brick_training_v2]
  [--max-steps 80]
  [--save-total-limit 5]
```

**Data** — JSONL files in `BRICK_TRAINING_DATA` (default
`data/brick_training/train.jsonl`, `test.jsonl`), loaded through HF
`datasets.load_dataset`. Each row is a chat-format message list whose
assistant reply ends with brick lines followed by `DONE`.
`prepare_brick_dataset.py` generates these from StableText2Brick and can
append source-controlled v2 canary examples.

**Curriculum** — samples are split into "untruncated" (estimated tokens
`(len(content)/3.0) + 80` ≤ `max_seq_length`) and "truncated", then
concatenated so the model sees the clean prefix first.

**Structure-aware loss** (`train_brick.py`, class `BrickStructureWeights`).
Per-token weights:

| Token class | Weight |
|-------------|--------|
| Boilerplate (`(`, `)`, `,`, `x`, `#`, spaces, newlines) | `0.1` |
| Dimension combos (`2x4`, `1x2`, …, `2x2`) | `3.0` |
| Everything else (coordinate digits, hex, etc.) | `1.0` |
| Masked prompt tokens (`-100`) | `0.0` |

Implemented via an override of `Trainer.compute_loss` that applies the
weights to a single vectorized cross-entropy call.

**Trainer settings** — `trl.SFTTrainer` (with HF `TrainingArguments`):

- `per_device_train_batch_size=1`, `gradient_accumulation_steps=16` ⇒ effective batch 16
- `learning_rate=1e-3`, `lr_scheduler="cosine"`, `warmup_steps=100`
- `max_seq_length=4096`, `num_epochs=3`
- `bf16=True`, `gradient_checkpointing=True`, `optim="adamw_torch"`
- `save_steps=200`, `save_total_limit=5` by default, W&B when available

**LoRA config** — `r=32`, `alpha=64`, `dropout=0.05`,
`target_modules=["q_proj", "v_proj"]` (narrower than Stage 1 because the
task is structurally simpler), with `use_dora=True`, `use_rslora=True`,
and **without** `init_lora_weights="pissa"`. The Stage 2 trainer keeps
the base in bf16, so PiSSA would be possible in principle, but the
shipped configuration leaves it off because PEFT warns on
`DoRA + PiSSA` and the more stable choice in this repo is
`DoRA + rsLoRA`.

Checkpoint written to `BRICK_CHECKPOINT_DIR`.

### 3.3 Orchestration (`scripts/train_full_pipeline.sh`)

Single entry point for a clean GPU box. Four steps, all overridable:

1. **Download COCO 2017** (`annotations_trainval2017.zip`, `train2017.zip` ~18 GB) into `data/coco/`. Skipped if present, or with `--skip-coco`.
2. **Build Stage 1 manifest** by running `python -m backend.data_pipeline.build_stage1_dataset`.
3. **Train Stage 2 Brick** (`python -m backend.training.train_brick --output-dir <…> --epochs 3`).
4. **Train Stage 1** (`python -m backend.training.train_stage1 --manifest data/stage1_manifest.json`).

### 3.4 Shared training utilities (`backend/training/utils.py`)

- `seed_everything(seed)` — deterministic torch/numpy/random seeding.
- `setup_wandb(project, run_name, ...)` — W&B init gated on `WANDB_DISABLED` and rank.

---

## 4. Inference

### 4.1 HTTP surface (`backend/app.py`, `backend/api/routes_*.py`)

`backend/app.py` wires a FastAPI app with CORS, a `/health` probe, and two
routers.

**Generate** (`backend/api/routes_generate.py`):

- `POST /api/generate-bricks` — accepts image or text → `BrickResponse`
  (`{bricks, caption?, brick_count, stable, metadata}`). Non-streaming.
- `POST /api/generate-stream` — SSE stream of events:
  - `progress` `{stage: "stage1"|"stage2", message, caption?}`
  - `brick`    `{count}` — after each placement
  - `rollback` `{count}` — after each physics rollback
  - `result`   full `BrickResponse`
  - `error`    `{detail}`

Both endpoints dispatch the blocking call through
`asyncio.get_event_loop().run_in_executor(None, …)` and wrap it in
`asyncio.wait_for(..., INFERENCE_TIMEOUT_SECONDS)` to avoid indefinite GPU
holds. The streaming route passes an `on_progress` callback into the
pipeline; the pipeline queues brick/rollback events, the route drains the
queue between model steps and emits SSE frames.

**Gallery** (`routes_gallery.py`): `GET/POST /api/gallery`,
`GET /api/gallery/{id}`, `PATCH /api/gallery/{id}/star`. Backed by SQLite
through `backend/storage/gallery_db.py` (async, `aiosqlite`). Schema:
`{id, title, caption, bricks, brick_count, stable, thumbnail_b64,
stars, star_count, created_at}`.

### 4.2 `Stage1Pipeline` (`backend/inference/stage1_pipeline.py`)

Minimal wrapper around Qwen3.5-9B in 4-bit NF4:

1. Builds `AutoProcessor` with vision pixel bounds `[256*28², 512*28²]`.
2. Loads `Qwen3_5ForConditionalGeneration` with `BitsAndBytesConfig` (NF4,
   double-quant, bf16 compute) and `device_map="auto"`.
3. If `STAGE1_CHECKPOINT_DIR/adapter_config.json` exists, wraps the base
   with `PeftModel.from_pretrained(...)`.
4. Exposes `describe(image) → str`: builds the Stage 1 chat template with
   the system prompt from `config.STAGE1_SYSTEM_PROMPT`, generates up to
   256 tokens at `temperature=0.7, top_p=0.9`, decodes, strips
   `<think>…</think>` blocks, returns the trimmed caption.

Singletons live in `brick_pipeline._get_stage1_pipeline()` (not
`stage1_pipeline.py` — the factory is colocated with Brick so Brick can
import it cleanly). In dev mode (`LEGOGEN_DEV=1`) it returns a fixed
caption via `_MockStage1`.

### 4.3 `BrickPipeline` (`backend/inference/brick_pipeline.py`)

The Stage 2 runtime. Loads `Qwen/Qwen3.5-4B` with the same 4-bit NF4
config and, if `BRICK_CHECKPOINT_DIR` exists, wraps it with the LoRA
adapter via `peft.PeftModel.from_pretrained`.

Generation constants:

- `MAX_BRICKS = 500`
- `MAX_REJECTIONS = 500` per brick
- `MAX_ROLLBACKS = 100`
- Temperature: fixed at `BASE_TEMPERATURE=0.6`. The old rejection-driven
  temperature ramp is gone — grammar-constrained decoding makes parse
  failures impossible, so the only remaining rejection cause is a voxel
  collision, which doesn't benefit from higher temperature.

The decoder runs under an `outlines.processors.RegexLogitsProcessor` built
from `STEP_PATTERN`: either `DONE\n` or `BRICK_PATTERN` (the 14 allowed dim
strings × coord tuple × hex color). This forces every generated step to be
either a syntactically valid brick line before the voxel check or an
explicit learned stop.

Prompt format:

```
System: You are a LEGO master builder.
User:   Create a colored LEGO model. Format: <dims> (<x>,<y>,<z>) <#hex>.
        Allowed dims: 2x4, 4x2, 2x6, 6x2, 1x2, 2x1, 1x4, 4x1, 1x6, 6x1, 1x8, 8x1, 1x1, 2x2.
        All bricks are 1 unit tall.
        ### Input:
        {caption}
```

Each line is parsed with the regex
`(\d+)x(\d+) \((\d+),(\d+),(\d+)\) #([0-9A-Fa-f]{6})`.

The generation loop combines three correctness mechanisms:

- **Grammar-constrained decoding** — the logits processor rules out any
  token that would break the brick-line regex, so every decoded line
  parses and uses only allowed dimensions.
- **Voxel rejection sampling** — after decoding, the brick is checked
  against `VoxelGrid.can_place` (`backend/brick/occupancy.py`) for
  bounds / collision / shape-set membership. On failure the brick is
  resampled at the same fixed temperature.
- **Physics rollback** — after each placement, `is_stable`
  (`backend/brick/stability.py`) runs a force-equilibrium LP: for every
  non-ground brick, `scipy.optimize.linprog` (HiGHS) must find per-stud
  contact forces satisfying vertical balance and zero moment about the
  brick's COM, subject to a per-stud tension bound (`STUD_STRENGTH`) and
  compression-only ground reactions. On failure,
  `find_first_unstable` binary-searches prefixes for the earliest brick
  that broke the LP; the sequence is truncated to that prefix, the voxel
  grid cleared, and generation resumes. Up to `MAX_ROLLBACKS` full
  rollbacks are permitted.

`generate(caption, on_progress=…)` and `generate_from_image(image,
on_progress=…)` both accept a streaming callback; the callback receives
`{"type": "caption", "caption": …}` (image path only, after Stage 1
completes), `{"type": "brick", "count": n}` after each placement, and
`{"type": "rollback", "count": n}` after each rollback.

Return shape: `{bricks: str, caption?: str, brick_count: int, stable: bool,
metadata: {model_version, generation_time_ms, rejections, rollbacks}}`.

In dev mode (`LEGOGEN_DEV=1`), `MockBrickPipeline` returns a fixed 12-brick
red house deterministically.

### 4.4 End-to-end request (image upload → streaming)

1. `POST /api/generate-stream` with a multipart image.
2. `routes_generate.generate_stream` validates content-type, decodes to
   `PIL.Image`, and opens the SSE response.
3. Emits `progress {stage: "stage1"}`, then runs
   `BrickPipeline.generate_from_image(image, on_progress)` on a thread.
4. `generate_from_image` first calls `Stage1Pipeline.describe(image)` →
   enqueues `{type: "caption", caption}`. The route drains the queue and
   emits a `progress {stage: "stage1", caption}` event, then
   `progress {stage: "stage2"}`.
5. `BrickPipeline.generate(caption, on_progress)` runs the
   grammar/voxel/LP loop. Each placed brick enqueues a `brick` event;
   each rollback enqueues a `rollback` event. The route drains the queue
   in ~50 ms ticks.
6. When generation completes, the route emits `result <BrickResponse>`
   and closes the stream.

---

## 5. Directory map

### `backend/`

| Path | Purpose |
|------|---------|
| `app.py` | FastAPI factory, lifespan (async model preload), CORS, `/health` |
| `config.py` | Central config: model IDs, hyperparameters, paths |
| `api/routes_generate.py` | `/api/generate-bricks`, `/api/generate-stream` |
| `api/routes_gallery.py` | `/api/gallery*` CRUD |
| `inference/stage1_pipeline.py` | `Stage1Pipeline` (9B multimodal → caption) |
| `inference/brick_pipeline.py` | `BrickPipeline` + factories + `MockBrickPipeline` + `_MockStage1` |
| `brick/` | `constants.py`, `parser.py`, `decoder.py`, `stability.py`, `occupancy.py` — brick math + physics primitives |
| `data_pipeline/dataset.py` | Shared `TRAIN_TRANSFORMS` / `VAL_TRANSFORMS` |
| `data_pipeline/dataset_stage1.py` | `Stage1Dataset` + `Stage1Collator` |
| `data_pipeline/build_stage1_dataset.py` | COCO + ST2B manifest builder |
| `data_pipeline/prepare_brick_dataset.py` | Rebrickable → brick JSONL |
| `training/train_stage1.py` | Stage 1 LoRA trainer (DDP-ready) |
| `training/train_brick.py` | Stage 2 Brick LoRA trainer (SFTTrainer + structure-aware loss) |
| `training/utils.py` | `seed_everything`, `setup_wandb` |
| `storage/gallery_db.py` | Async SQLite gallery store (brick schema) |

### `scripts/`

| File | Purpose |
|------|---------|
| `train_full_pipeline.sh` | 4-step orchestrator: COCO → manifest → Stage 2 → Stage 1 |
| `train_brick_runpod.sh` | RunPod-specific helper for the brick trainer |
| `prepare_dataset.py` | Rebrickable CSVs + set JSON + image collection |
| `bootstrap.sh` | Environment setup (deps, dirs, env vars) |

### `frontend/` (surface only)

- Pages: `/` (Home), `/build` (BuildSession → `/api/generate-stream`),
  `/guide/:buildId` (GuidancePage — layer walkthrough), `/explore`
  (ExplorePage → `/api/gallery`), `/about` (system spec).
- 3D viewer: `BrickCoordViewer` + `BrickMesh`.
- API client: `frontend/src/api/legogen.ts` — `BrickResponse`,
  `BrickCoord`, `GalleryBuild`, `StreamEvent` types plus
  `parseBrickString`, `bricksToLayers`, `generateBricksStream`.

---

## 6. Gotchas

- **Gallery schema doesn't store a JSON description anymore.** The
  `description_json` / `category` / `complexity` / `parts_count` columns
  are gone; a gallery row is `{title, caption, bricks, brick_count,
  stable, thumbnail_b64, stars, …}`. Anything reading the old columns
  will 404.
- **Two different backbones at runtime.** `Stage1Pipeline` holds a 9B
  Qwen3.5 (+ Stage 1 adapter) and `BrickPipeline` holds a separate 4B
  Qwen3.5 (+ Brick adapter). Both use 4-bit NF4 quantization. Expect
  ~10 GB VRAM combined.
- **Model IDs** `Qwen/Qwen3.5-9B` and `Qwen/Qwen3.5-4B` are strings
  consumed literally by `from_pretrained`. If the HF repo you are
  mirroring uses a different name, set `HF_ENDPOINT` or pass
  `--model-path` to the trainers.
- **No post-hoc validator endpoint.** `/api/validate` is removed. All
  stability/legality enforcement happens during Stage 2 generation
  (grammar + voxel + LP). If you want to re-check a gallery entry,
  re-parse its `bricks` string and call `is_stable` directly.
