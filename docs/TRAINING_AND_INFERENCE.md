# Training and Inference

Reference documentation for the Lego-Gen training pipeline and the runtime
inference stack, reflecting the post-"strip to two-stage" state of the tree.
All file paths and line numbers refer to the repo at the main branch tip.

---

## 1. Two-Stage Architecture

The system decomposes "image → buildable LEGO" into two independently
fine-tuned Qwen stages plus a runtime-only validation layer:

| Stage | Job | Base model | Input | Output |
|-------|-----|-----------|-------|--------|
| Stage 1 | Perceive | `Qwen/Qwen3.5-9B` (+ vision) | RGB image | 1–3 sentence geometry/color/scale description |
| Stage 2 | Build | `Qwen/Qwen3.5-4B` | Text caption | Newline-separated brick sequence `HxW (x,y,z) #RRGGBB` |

Two deployment paths live on top of those stages:

- **Image → Bricks** — Stage 1 caption feeds Stage 2; returns placed bricks.
- **Text → Bricks** — skip Stage 1, user prompt goes straight to Stage 2.

A legacy **image → structured JSON** path (`UnifiedPipeline`) also remains in
`backend/inference/pipeline.py` and is used by `/api/generate`. It reuses
the Stage 1 adapter for the image → text step, then prompts the same 9B
backbone against a JSON schema (see §4.3). There is no active training
script producing the Stage 2 JSON adapter in the current tree — the adapter
at `UNIFIED_CHECKPOINT_DIR` is loaded if it exists on disk, otherwise the
pipeline runs the base 9B model without an adapter (`pipeline.py:213-219`).

---

## 2. Configuration

All training and inference knobs live in `backend/config.py`.

### 2.1 Model IDs (`config.py:28-102`)

| Constant | Value | Used by |
|----------|-------|---------|
| `MODEL_NAME` | `"Qwen/Qwen3-VL-8B-Instruct"` | Legacy — no longer trained; left for back-compat imports |
| `UNIFIED_MODEL_NAME` | `"Qwen/Qwen3.5-9B"` | Stage 1 training, `UnifiedPipeline` base |
| `PLANNER_MODEL_NAME` | `"Qwen/Qwen3.5-9B"` | JSON planner path inside `UnifiedPipeline` |
| `BRICK_MODEL_NAME` | `"Qwen/Qwen3.5-4B"` | Stage 2 brick training + `BrickPipeline` |

### 2.2 Checkpoint paths (`config.py`)

- Stage 1 LoRA adapter: `CHECKPOINT_DIR / "qwen35-9b-lego-stage1-lora"` → `STAGE1_CHECKPOINT_DIR`
- Stage 2 JSON adapter (legacy): `CHECKPOINT_DIR / "qwen35-9b-lego-stage2-lora"` → `UNIFIED_CHECKPOINT_DIR`
- Stage 2 Brick adapter: `CHECKPOINT_DIR / "qwen35-4b-brick-lora"` → `BRICK_CHECKPOINT_DIR`

### 2.3 Hyperparameters (verbatim from `config.py`)

**Stage 1** (lines 91-99)

| Knob | Value |
|------|-------|
| `STAGE1_LORA_R` | `32` |
| `STAGE1_LORA_ALPHA` | `64` |
| `STAGE1_LEARNING_RATE` | `5e-5` |
| `STAGE1_BATCH_SIZE` | `8` (per GPU; effective = 8 × N_GPUS with `gradient_accumulation=1`) |
| `STAGE1_MAX_SEQ_LENGTH` | `512` |
| `STAGE1_NUM_EPOCHS` | `3` |
| `STAGE1_WARMUP_STEPS` | `100` |

**Stage 2 Brick** (lines 102-112)

| Knob | Value |
|------|-------|
| `BRICK_LORA_R` | `32` |
| `BRICK_LORA_ALPHA` | `64` |
| `BRICK_LORA_DROPOUT` | `0.05` |
| `BRICK_LEARNING_RATE` | `1e-3` |
| `BRICK_BATCH_SIZE` | `1` (effective 16 with `BRICK_GRADIENT_ACCUMULATION=16`) |
| `BRICK_MAX_SEQ_LENGTH` | `4096` |
| `BRICK_NUM_EPOCHS` | `3` |

**Stage 2 JSON (legacy, inference-only)** (lines 80-87)

| Knob | Value |
|------|-------|
| `UNIFIED_LEARNING_RATE` | `5e-5` |
| `UNIFIED_BATCH_SIZE` | `1` (effective 32 with `UNIFIED_GRADIENT_ACCUMULATION=32`) |
| `UNIFIED_MAX_SEQ_LENGTH` | `4096` |
| `UNIFIED_QUANTIZATION_BITS` | `4` |

**Shared / LoRA / quantization** (lines 35-45)

- `LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- `QUANTIZATION_BITS = 4` (NF4, with double-quantization)
- `USE_BF16` — auto-detected from the GPU

**Inference** (around line 151)

- `MAX_NEW_TOKENS = 2048`
- `TEMPERATURE = 0.7`
- `TOP_P = 0.9`
- `INFERENCE_TIMEOUT_SECONDS = 120`

---

## 3. Training

### 3.1 Stage 1 — Image → Description (`backend/training/train_stage1.py`)

Fine-tunes a LoRA adapter on the 9B Qwen3.5 VL-style model so it emits a
short, LEGO-relevant caption from a photograph.

**Entry points** (`train_stage1.py:9-20`):

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
`COCO_TO_ST2B_CATEGORY` map in `config.py`.

The collator stacks `input_ids`, `attention_mask`, `labels`,
`mm_token_type_ids` and concatenates `pixel_values`. The dataset masks all
tokens before the last `"<|im_start|>assistant\n"` marker with `-100` so
loss is only computed on the description, not the prompt.

**LoRA config** — `r=32`, `alpha=64`, `target_modules="all-linear"`,
`dropout=0.05`. Vision encoder parameters (names containing `visual` /
`vision`) are frozen.

**Trainer settings** (HuggingFace `Trainer` + `TrainingArguments`):

- `lr_scheduler_type="cosine"`, warmup = `max(1, total_steps // 10)` (capped at 100)
- `gradient_checkpointing=True`, `use_reentrant=False`
- `ddp_bucket_cap_mb=50`, `ddp_find_unused_parameters=False`
- `save_total_limit=2`, `load_best_model_at_end=True`
- W&B logging via `backend.training.utils.setup_wandb` (auto-disabled on non-rank-0)

Checkpoint written to `STAGE1_CHECKPOINT_DIR` (+ `adapter_config.json` so
the inference pipeline can detect its presence).

### 3.2 Stage 2 Brick — Text → Brick coordinates (`backend/training/train_brick.py`)

Fine-tunes a LoRA adapter on Qwen3.5-4B to emit the brick sequence format
consumed by `BrickPipeline`.

**Entry point** (`train_brick.py:313-317`):

```bash
python -m backend.training.train_brick \
  --output-dir backend/models/checkpoints/qwen35-4b-brick-lora/ \
  --epochs 3 \
  [--resume auto]
```

**Data** — JSONL files in `BRICK_TRAINING_DATA` (default
`data/brick_training/train.jsonl`, `test.jsonl`), loaded through HF
`datasets.load_dataset`. Each row is a chat-format message list ending in
the brick sequence as the assistant reply. `prepare_brick_dataset.py`
generates these from Rebrickable CSVs.

**Curriculum** (`train_brick.py:99-125`) — samples are split into
"untruncated" (estimated tokens `(len(content)/3.0) + 80` ≤ `max_seq_length`)
and "truncated", then concatenated so the model sees the clean prefix
first. A single log line like

```
Curriculum: 2500 untruncated, 500 truncated (16.7%) at max_seq_length=4096
```

reports the split.

**Structure-aware loss** (`train_brick.py:44-94`, class
`BrickStructureWeights`). Per-token weights:

| Token class | Weight |
|-------------|--------|
| Boilerplate (`(`, `)`, `,`, `x`, `#`, spaces, newlines) | `0.1` |
| Dimension combos (`2x4`, `1x2`, …, `2x2`) | `3.0` |
| Everything else (coordinate digits, hex, etc.) | `1.0` (default) |
| Masked prompt tokens (`-100`) | `0.0` |

Implemented via an override of `Trainer.compute_loss` (`train_brick.py:260-283`)
that applies the weights to a single vectorized cross-entropy call.

**Trainer settings** — `trl.SFTTrainer` (with HF `TrainingArguments`):

- `per_device_train_batch_size=1`, `gradient_accumulation_steps=16` ⇒ effective batch 16
- `learning_rate=1e-3`, `lr_scheduler="cosine"`, `warmup_steps=100`
- `max_seq_length=4096`, `num_epochs=3`
- `bf16=True`, `gradient_checkpointing=True`, `optim="adamw_torch"`
- `save_steps=200`, `save_total_limit=2`, W&B when available

**LoRA config** — `r=32`, `alpha=64`, `dropout=0.05`,
`target_modules=["q_proj", "v_proj"]` (narrower than Stage 1 because the
task is structurally simpler).

Checkpoint written to `BRICK_CHECKPOINT_DIR`.

### 3.3 Orchestration (`scripts/train_full_pipeline.sh`)

Single entry point for a clean GPU box. Four steps, all overridable:

1. **Download COCO 2017** (`annotations_trainval2017.zip`, `train2017.zip` ~18 GB) into `data/coco/`. Skipped if present, or with `--skip-coco`.
2. **Build Stage 1 manifest** by running `python -m backend.data_pipeline.build_stage1_dataset` (skipped when `data/stage1_manifest.json` already exists).
3. **Train Stage 2 Brick** (`python -m backend.training.train_brick --output-dir <…> --epochs 3`), logs to `training_stage2.log`.
4. **Train Stage 1** (`python -m backend.training.train_stage1 --manifest data/stage1_manifest.json`), logs to `training_stage1.log`.

Flags: `--skip-coco`, `--stage2-only`, `--stage1-only`, `--no-wandb`,
`--resume=<path>` (forwarded to the brick trainer).

### 3.4 Shared training utilities (`backend/training/utils.py`)

- `seed_everything(seed)` — deterministic torch/numpy/random seeding.
- `setup_wandb(project, run_name, ...)` — W&B init gated on `WANDB_DISABLED` and rank.
- Metric helpers used by callbacks: `json_validity_rate`, `color_f1`, `parts_f1`, `structural_coherence`.

### 3.5 Data pipeline module (`backend/data_pipeline/`)

| File | Purpose |
|------|---------|
| `dataset_stage1.py` | `Stage1Dataset`, `Stage1Collator` for image→description |
| `dataset.py` | `LegoDataset` + `TRAIN_TRANSFORMS` / `VAL_TRANSFORMS` (legacy image→JSON path) |
| `build_stage1_dataset.py` | Build `stage1_manifest.json` from COCO + ST2B |
| `prepare_brick_dataset.py` | Download Rebrickable CSVs, emit brick JSONL |

---

## 4. Inference

### 4.1 HTTP surface (`backend/app.py`, `backend/api/routes_*.py`)

`backend/app.py:1-64` wires a FastAPI app with CORS, a `/health` probe, and
three routers.

**Generate** (`backend/api/routes_generate.py`):

- `POST /api/generate` — multipart `image` (+ optional `prompt`) → full build (description JSON + steps + validation).
- `POST /api/generate-from-text` — JSON body `{prompt}` → same shape as above, no Stage 1.
- `POST /api/generate-bricks` — accepts image or text → `BrickPipeline` output (`bricks`, `brick_count`, `stable`, metadata).
- `POST /api/generate-stream` — SSE; emits `{"stage": "stage1"|"stage2"|"validating"}` progress events, then a final `{"result": …}`.

All generate endpoints hash the raw bytes/prompt with SHA-256, dispatch the
blocking call through `asyncio.get_event_loop().run_in_executor(None, …)`,
and wrap the call in `asyncio.wait_for(..., INFERENCE_TIMEOUT_SECONDS)` to
avoid indefinite GPU holds (`routes_generate.py:47-60`).

**Validate** (`routes_validate.py`): `POST /api/validate` with a
`LegoDescription` JSON body — returns a `ValidationReport` from the
stability checker (see §4.5).

**Gallery** (`routes_gallery.py`): `GET/POST /api/gallery`,
`GET /api/gallery/{id}`, `PATCH /api/gallery/{id}/star`. Backed by SQLite
through `backend/storage/gallery_db.py` (async, `aiosqlite`).

### 4.2 `UnifiedPipeline` (`backend/inference/pipeline.py:192-710`)

The singleton returned by `get_pipeline()` wraps the 9B Qwen3.5 model
(`LegoUnifiedModel` in `backend/models/unified_model.py`) loaded with 4-bit
NF4 quantization and supports **named LoRA adapters** (`set_adapter("default"|"stage1")`).

Construction (`pipeline.py:201-252`):

1. Build the base wrapper with 4-bit quant.
2. Load the JSON (Stage 2) adapter from `UNIFIED_CHECKPOINT_DIR` as the
   `default` adapter; if missing, run the base model (no fallback to the
   legacy `Qwen3-VL-8B` adapter — it's incompatible with the 9B backbone).
3. If `STAGE1_CHECKPOINT_DIR/adapter_config.json` exists, attach the Stage 1
   adapter under the name `stage1`, set `has_stage1=True`, and log
   `"Two-stage pipeline enabled (Stage 1 + Stage 2)"`.
4. Initialize and warm up the three caches (§4.4).
5. Pre-compute the Stage 1 KV prefix if two-stage is available.

Key methods:

- `describe_image_stage1(image)` (`pipeline.py:532-625`) — swap to
  `stage1` adapter, generate ≤256 tokens from
  `[STAGE1_SYSTEM_PROMPT, image, "Describe this object for LEGO building."]`,
  strip thinking blocks, swap back to `default`. Adapter swap overhead is
  logged when it exceeds 5 ms.
- `describe_from_text(prompt)` (`pipeline.py:434-530`) — text → JSON using
  `PLANNER_SYSTEM_PROMPT`. 2048 tokens, response cache keyed on
  `SHA-256(prompt.strip())`.
- `describe_image(image)` (`pipeline.py:316-432`) — direct single-stage
  image → JSON, used when `has_stage1=False`.
- `generate_build(image, cache_key=…)` (`pipeline.py:627-671`) — top-level
  entry. If `has_stage1`: two-stage response cache on key `f"twostage:{cache_key}"`,
  then Stage 1 → Stage 2 JSON → `json_to_steps` → stability report. Otherwise
  falls back to `describe_image`.

A `MockPipeline` (used when `LEGOGEN_DEV=1`) lives just above at
`pipeline.py:77-190`; it returns a canned description/steps/validation in
~42 ms and never touches a GPU.

### 4.3 `BrickPipeline` (`backend/inference/brick_pipeline.py:35-146+`)

The Stage 2 Brick runtime. Loads `Qwen/Qwen3.5-4B` with the same 4-bit NF4
config and, if `BRICK_CHECKPOINT_DIR` exists, wraps it with the LoRA
adapter via `peft.PeftModel.from_pretrained`.

Generation constants (top of file):

- `MAX_BRICKS = 500`
- `MAX_REJECTIONS = 500` per brick
- `MAX_ROLLBACKS = 100`
- Temperature curve: `BASE_TEMPERATURE=0.6` → `+TEMP_INCREMENT=0.01` per
  rejection, capped at `MAX_TEMPERATURE=2.0`.

Prompt format (lines 27-32):

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

The generation loop combines two correctness mechanisms:

- **Rejection sampling** — each brick is generated one at a time. If the
  emitted line doesn't parse or collides with the voxel grid
  (`backend/brick/occupancy.py::VoxelGrid.can_place` — shape / bounds /
  collision), the temperature is bumped and the brick is re-sampled.
- **Physics rollback** — after each placement, `is_stable` (BFS from
  `z=0` through overlapping layers, `backend/brick/stability.py`) is
  evaluated. On failure, `find_first_unstable` returns the index of the
  earliest brick that broke stability; the sequence is truncated to that
  prefix, the voxel grid cleared, and generation resumes. Up to
  `MAX_ROLLBACKS` full rollbacks are permitted.

Return shape: `{bricks: str, brick_count: int, stable: bool, metadata: {rejections, rollbacks, generation_time_ms}}`.

### 4.4 Caching (`backend/inference/cache.py`)

Three layers, all gated by flags in `config.py` (`CACHE_ENABLED`,
`CACHE_KV_PREFIX_ENABLED`, `CACHE_RESPONSE_ENABLED`,
`CACHE_TOKENIZATION_ENABLED`) and warmed at pipeline startup.

- **KV prefix cache** — pre-runs the forward pass on static system prompts
  (`SYSTEM_PROMPT`, `PLANNER_SYSTEM_PROMPT`, `STAGE1_SYSTEM_PROMPT`),
  stores the `DynamicCache`, and clones it into `past_key_values` on each
  `generate()` call. Saves roughly the cost of ~100 prefix tokens per
  request.
- **Response cache** — thread-safe `OrderedDict` with LRU + TTL
  (`CACHE_RESPONSE_MAX_SIZE=256`, `CACHE_RESPONSE_TTL_SECONDS=3600`). Only
  activated when outputs are deterministic — the gate is
  `CACHE_RESPONSE_ENABLED and (TEMPERATURE == 0 or not TOP_P or CACHE_RESPONSE_FOR_SAMPLING)` (`pipeline.py:27-29`). With the shipped
  `TEMPERATURE=0.7` and `TOP_P=0.9`, this is off by default.
- **Tokenization cache** — keyed dict for `apply_chat_template` outputs of
  static prompt skeletons (`vision_template`, planner prefix, etc.).

### 4.5 Validation layer

Two post-generation components used by the JSON path.

**Constraint engine** (`backend/inference/constraint_engine.py`). Entry
point `safe_parse_and_validate(raw_output)`:

1. `repair_json_string` — regex fixups for trailing commas, unclosed
   strings/brackets (for truncated outputs).
2. `json.loads` → dict.
3. `validate_lego_json` — enforce `REQUIRED_FIELDS`, typed fields,
   enums (e.g. `complexity ∈ {simple, intermediate, advanced, expert}`).
4. `enforce_valid_values` — clamp invalid enums, fix spatial positions.
5. `repair_connects_to` — ensure `connects_to` references point to real
   subassembly names.
6. `validate_structural_order` — bottom-to-top ordering check.

A shortcut `validate_and_repair_dict` skips the JSON round-trip when the
caller already has a parsed dict (see `pipeline.py:54-62`).

**Stability checker** (`backend/inference/stability_checker.py`).
Singleton (`pipeline.py:34-43`) producing a `ValidationReport` with an
integer `score ∈ [0, 100]`, a list of `CheckResult`, and a summary. The
ten checks are split into:

- *Legality*: `part_existence` (against `parts.csv.gz`),
  `part_compatibility`, `color_validity` (against `colors.json`),
  `quantity_legality` (thresholds
  `QUANTITY_WARN_THRESHOLD=50`, `QUANTITY_FAIL_THRESHOLD=200`).
- *Stability*: `foundation` (bottom subassembly present), `connectivity`
  (subassembly adjacency graph), `cantilever`
  (≥`MIN_CANTILEVER_CONNECTIONS=2` supports for overhangs), `top_heavy`
  (support ratio ≥ `SUPPORT_RATIO_WARN=3.0`), `center_of_mass` within the
  base footprint.

**Postprocess** (`backend/inference/postprocess_manual.py`).
`json_to_steps` sorts subassemblies by `POSITION_ORDER` (bottom → center
→ sides → top) and emits `BuildStep` entries with `step_number`, `title`,
`instruction`, `parts`, `part_count`.

### 4.6 End-to-end request (image upload)

1. `POST /api/generate` with a multipart image.
2. `routes_generate.generate` validates content-type, reads bytes,
   decodes to `PIL.Image`, computes `cache_key = sha256(bytes)`, frees the
   raw bytes, and dispatches to the executor.
3. `UnifiedPipeline.generate_build` checks the two-stage response cache,
   then runs Stage 1 (adapter swap, ≤256-token caption, swap back).
4. The caption feeds `describe_from_text`, which runs the Stage 2 JSON
   generation against the planner system prompt (KV prefix cached), parses
   with the constraint engine, and caches on the prompt hash.
5. `json_to_steps` produces an ordered `BuildStep[]` and the stability
   checker produces a `ValidationReport`.
6. The combined result is cached under `f"twostage:{cache_key}"` and
   returned as:

   ```json
   {
     "description": { /* LegoDescription */ },
     "steps":       [ /* BuildStep */ ],
     "metadata":    {
       "model_version": "qwen35-lego-two-stage-v1",
       "generation_time_ms": 3200,
       "json_valid": true,
       "errors": [],
       "cached": false
     },
     "validation":  { /* ValidationReport */ }
   }
   ```

For `/api/generate-bricks` the Stage 1 caption is instead handed to
`BrickPipeline.generate`, whose return shape is the brick-sequence dict
described in §4.3.

---

## 5. Directory map

### `backend/`

| Path | Purpose |
|------|---------|
| `app.py` | FastAPI app factory, lifespan (async model preload), CORS, `/health` |
| `config.py` | Central config: model IDs, hyperparameters, paths, feature flags |
| `api/routes_generate.py` | `/api/generate`, `/generate-from-text`, `/generate-bricks`, `/generate-stream` |
| `api/routes_validate.py` | `/api/validate` (stability/legality report) |
| `api/routes_gallery.py` | `/api/gallery*` CRUD |
| `models/unified_model.py` | `LegoUnifiedModel` — 9B Qwen wrapper with named-adapter swap |
| `models/tokenizer.py` | Chat templates, system prompts, JSON extraction, thinking-block stripping |
| `inference/pipeline.py` | `UnifiedPipeline`, `MockPipeline`, singletons |
| `inference/brick_pipeline.py` | `BrickPipeline` — rejection sampling + physics rollback |
| `inference/cache.py` | KV prefix, response, and tokenization caches |
| `inference/constraint_engine.py` | JSON schema validation / repair |
| `inference/stability_checker.py` | 10-check legality + stability report |
| `inference/postprocess_manual.py` | `json_to_steps` (JSON → ordered `BuildStep` list) |
| `brick/` | `constants.py`, `parser.py`, `decoder.py`, `stability.py`, `occupancy.py` — brick math and physics primitives |
| `data_pipeline/dataset_stage1.py` | `Stage1Dataset` + `Stage1Collator` |
| `data_pipeline/build_stage1_dataset.py` | COCO + ST2B manifest builder |
| `data_pipeline/prepare_brick_dataset.py` | Rebrickable → brick JSONL |
| `data_pipeline/dataset.py` | `LegoDataset` + augmentations (legacy image→JSON) |
| `training/train_stage1.py` | Stage 1 LoRA trainer (DDP-ready) |
| `training/train_brick.py` | Stage 2 Brick LoRA trainer (SFTTrainer + structure-aware loss) |
| `training/utils.py` | Seeding, W&B, metric helpers |
| `storage/gallery_db.py` | Async SQLite gallery store |

### `scripts/`

| File | Purpose |
|------|---------|
| `train_full_pipeline.sh` | 4-step orchestrator: COCO → manifest → Stage 2 → Stage 1 |
| `train_brick_runpod.sh` | RunPod-specific helper for the brick trainer |
| `prepare_dataset.py` | Rebrickable CSVs + set JSON + image collection |
| `benchmark.py` | Inference latency benchmarking |
| `bootstrap.sh` | Environment setup (deps, dirs, env vars) |

### `frontend/` (surface only)

- Pages: `/` (Home), `/build` (BuildSession → `/api/generate`), `/guide/:buildId` (GuidancePage), `/explore` (ExplorePage → `/api/gallery`), `/about`.
- The 3D viewer components (`LegoViewer`, `BrickCoordViewer`) render either the `LegoDescription` subassemblies or the brick-coordinate output; `ValidationPanel` renders the `ValidationReport`; `StepList` / `StepDetail` / `PartsChecklist` render the `BuildStep[]`.
- `frontend/src/api/legogen.ts` defines the TS interfaces (`LegoDescription`, `BuildStep`, `BrickResponse`, `ValidationReport`) and helpers (`parseBrickString`, `bricksToSteps`).

---

## 6. Gotchas

- **Stage 2 JSON adapter is not trained by anything in the current tree.**
  `train_full_pipeline.sh` trains Stage 1 and Stage 2 *Brick*. The JSON path
  remains in `UnifiedPipeline` for back-compat and will run against the base
  9B model unless an adapter is manually placed at `UNIFIED_CHECKPOINT_DIR`.
- **Response cache is off under default sampling settings.** `TEMPERATURE=0.7`
  and `TOP_P=0.9` gate out Layer 2 of `cache.py`. Flip
  `CACHE_RESPONSE_FOR_SAMPLING` to enable approximate caching, or run with
  `TEMPERATURE=0` for strict caching.
- **Two different backbones at runtime.** `UnifiedPipeline` holds a 9B
  Qwen3.5 (+ Stage 1 adapter) and `BrickPipeline` holds a separate 4B
  Qwen3.5. Both use 4-bit NF4 quantization. Expect ~10 GB VRAM combined.
- **Model IDs** `Qwen/Qwen3.5-9B` and `Qwen/Qwen3.5-4B` are strings consumed
  literally by `from_pretrained`. If the HF repo you are mirroring uses a
  different name, set `HF_ENDPOINT` or pass `--model-path` to the trainers.
