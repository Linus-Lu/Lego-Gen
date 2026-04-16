# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running things

This project uses a `.venv` at the repo root. Always invoke Python through `.venv/bin/python` — the system `python` won't have the right deps, and no `pyproject.toml`/`setup.cfg`/`pytest.ini` exists to hook them up otherwise.

```bash
# Tests (dev mode bypasses the GPU model load in conftest/app lifespan)
LEGOGEN_DEV=1 .venv/bin/python -m pytest -q

# Single test file / single test
LEGOGEN_DEV=1 .venv/bin/python -m pytest tests/test_brick_stability.py -v
LEGOGEN_DEV=1 .venv/bin/python -m pytest tests/test_brick_stability.py::test_name -v

# Backend (dev — MockPipeline, no GPU)
LEGOGEN_DEV=1 .venv/bin/uvicorn backend.app:app --reload --port 8000

# Backend (prod — loads Qwen3.5-9B + adapters; requires ≥24 GB VRAM)
LEGOGEN_DEV=0 .venv/bin/uvicorn backend.app:app --host 0.0.0.0 --port 8000

# Frontend
cd frontend && npm run dev        # Vite dev server
cd frontend && npm run build      # runs `tsc && vite build` — both must pass
```

GPU-only tests are auto-skipped on machines without CUDA (`tests/conftest.py` adds a `gpu` marker and skips marked tests when `torch.cuda.is_available()` is false).

There is no lint/format config in-tree (no ruff, eslint, prettier, mypy). Don't invent one.

## Training entry points

```bash
# Stage 1 (image → description): Qwen3.5-9B + LoRA r=32
python -m backend.training.train_stage1
torchrun --nproc_per_node=4 -m backend.training.train_stage1   # multi-GPU

# Stage 2 Brick (text → brick coords): Qwen3.5-4B + LoRA r=32
python -m backend.training.train_brick --output-dir backend/models/checkpoints/qwen35-4b-brick-lora/ --epochs 3

# Full pipeline: downloads COCO, builds Stage 1 manifest, trains Stage 2 then Stage 1
bash scripts/train_full_pipeline.sh
```

All training knobs (LR, LoRA ranks, batch sizes, seq lengths, checkpoint paths) live in `backend/config.py`. Change them there, not inline.

## Architecture — what you need to know before editing

The system is **two independent Qwen3.5 pipelines**, both loaded with 4-bit NF4 quantization, held as module-level singletons.

### `UnifiedPipeline` (`backend/inference/pipeline.py`) — the JSON path
- Wraps `Qwen/Qwen3.5-9B` (`backend/models/unified_model.py`) with **named LoRA adapters**: `default` (Stage 2 JSON at `UNIFIED_CHECKPOINT_DIR`) and `stage1` (Stage 1 caption adapter at `STAGE1_CHECKPOINT_DIR`).
- `describe_image_stage1` → `describe_from_text` → constraint engine → `json_to_steps` → stability checker. Each request swaps adapters twice; swap overhead is logged if >5 ms.
- `MockPipeline` (same file, ~line 77) returns canned data in ~42 ms when `LEGOGEN_DEV=1`. All tests run through this.
- **Gotcha**: no training script in-tree produces the Stage 2 JSON adapter. If `UNIFIED_CHECKPOINT_DIR` is empty, it falls back to the base 9B with no adapter — do not fall back to the legacy `Qwen3-VL-8B` adapter (incompatible architecture).

### `BrickPipeline` (`backend/inference/brick_pipeline.py`) — the brick-coord path
- Separate `Qwen/Qwen3.5-4B` + LoRA, loaded independently of `UnifiedPipeline`. Both pipelines coexist at runtime (~10 GB VRAM combined).
- Emits lines matching `(\d+)x(\d+) \((\d+),(\d+),(\d+)\) #([0-9A-Fa-f]{6})`. Correctness comes from three stacked mechanisms:
  1. **Grammar-constrained decoding** via `outlines.processors.RegexLogitsProcessor` attached in `__init__` — token-level; bad dims/formats can't be emitted.
  2. **Voxel rejection** via `VoxelGrid.can_place` (`backend/brick/occupancy.py`) — collision/bounds check after decode.
  3. **Physics rollback** via `is_stable` (`backend/brick/stability.py`) running a per-stud LP (`scipy.optimize.linprog`). `find_first_unstable` binary-searches prefixes; up to `MAX_ROLLBACKS=100` full rollbacks allowed.
- **Temperature is fixed at `BASE_TEMPERATURE=0.6`**. Don't reintroduce temperature ramping on rejections — the grammar makes parse failures impossible, and voxel collisions don't benefit from higher temperature.

### Validation layer (JSON path only)
- `constraint_engine.safe_parse_and_validate` → regex JSON repair → `json.loads` → schema + enum enforcement → `connects_to` repair → structural-order check. Use `validate_and_repair_dict` if you already have a parsed dict.
- `stability_checker` produces a `ValidationReport` with `score ∈ [0, 100]` from 10 checks (part existence, color validity, quantity thresholds, foundation, connectivity, cantilever, top-heavy, center-of-mass, etc.). Singleton; fixtures monkey-patch it.

### Caching (`backend/inference/cache.py`)
Three flag-gated layers, warmed at pipeline startup:
- **KV-prefix cache** on the three static system prompts — cloned into `past_key_values` on each `generate()` call.
- **Response cache** (LRU + TTL) — gated on `TEMPERATURE == 0 or not TOP_P or CACHE_RESPONSE_FOR_SAMPLING`. **Off by default** under the shipped `TEMPERATURE=0.7`/`TOP_P=0.9`.
- **Tokenization cache** for `apply_chat_template` outputs.

### Frontend surface
React 19 + Vite + Three.js (R3F). `frontend/src/api/legogen.ts` holds the TS interfaces (`LegoDescription`, `BuildStep`, `BrickResponse`, `ValidationReport`) that mirror the backend contracts — keep them in sync when changing API shapes. Routes: `/`, `/build`, `/guide/:buildId`, `/explore`, `/about`.

## When in doubt

`docs/TRAINING_AND_INFERENCE.md` is the detailed, line-number-anchored reference for the training pipeline, the three caches, the validation layer, and the end-to-end request flow. Read it before refactoring any of those areas.
