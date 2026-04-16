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

# Backend (dev — MockBrickPipeline, no GPU)
LEGOGEN_DEV=1 .venv/bin/uvicorn backend.app:app --reload --port 8000

# Backend (prod — loads Qwen3.5-9B + Qwen3.5-4B with LoRA; ≥24 GB VRAM)
LEGOGEN_DEV=0 .venv/bin/uvicorn backend.app:app --host 0.0.0.0 --port 8000

# Frontend
cd frontend && npm run dev        # Vite dev server
cd frontend && npm run build      # runs `tsc && vite build` — both must pass
```

GPU-only tests are auto-skipped on machines without CUDA (`tests/conftest.py` adds a `gpu` marker and skips marked tests when `torch.cuda.is_available()` is false).

There is no lint/format config in-tree (no ruff, eslint, prettier, mypy). Don't invent one.

**Known pre-existing test failure**: `tests/test_brick_decoder.py::test_allowed_colors_populated` fails because `colors.json` isn't checked in. It's in the explicit skip list in `.claude/settings.local.json` — ignore it when running the suite.

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

The runtime is **two independent Qwen3.5 pipelines**, both loaded with 4-bit NF4 quantization, held as module-level singletons. There is no legacy JSON path — if you find references to `UnifiedPipeline`, `LegoUnifiedModel`, `constraint_engine`, or `/api/validate`, they are gone.

### `Stage1Pipeline` (`backend/inference/stage1_pipeline.py`) — image → caption
- Loads `Qwen/Qwen3.5-9B` in 4-bit NF4 and, if `STAGE1_CHECKPOINT_DIR/adapter_config.json` exists, wraps it with the Stage 1 LoRA adapter.
- Single method: `describe(image) → str`. Generates up to 256 tokens, strips `<think>…</think>` blocks.
- Mocked by `_MockStage1` (same file's factory) when `LEGOGEN_DEV=1` — returns a fixed caption in no time.

### `BrickPipeline` (`backend/inference/brick_pipeline.py`) — caption → brick lines
- Separate `Qwen/Qwen3.5-4B` + LoRA. Combined with Stage 1, ~10 GB VRAM total.
- Emits `HxW (x,y,z) #RRGGBB\n` lines. Correctness comes from three stacked mechanisms:
  1. **Grammar-constrained decoding** via `outlines.processors.RegexLogitsProcessor` — token-level; parse failures impossible.
  2. **Voxel rejection** via `VoxelGrid.can_place` (`backend/brick/occupancy.py`) — collision/bounds check.
  3. **Physics rollback** via `is_stable` (`backend/brick/stability.py`) running a per-stud LP (`scipy.optimize.linprog`, HiGHS). `find_first_unstable` binary-searches prefixes; up to `MAX_ROLLBACKS=100` full rollbacks allowed.
- **Temperature is fixed at `BASE_TEMPERATURE=0.6`**. Don't reintroduce temperature ramping — the grammar makes parse failures impossible, and voxel collisions don't benefit from higher temperature.
- `generate(caption, on_progress=…)` and `generate_from_image(image, on_progress=…)` accept a callback that fires `{type:"caption"|"brick"|"rollback", …}` events — used by the SSE route to stream progress.
- Mocked by `MockBrickPipeline` (same file) when `LEGOGEN_DEV=1`: returns a fixed 12-brick house.

### Factories
`get_brick_pipeline()` and `_get_stage1_pipeline()` in `brick_pipeline.py` are the only entry points into the inference stack. Both return singletons; both branch on `LEGOGEN_DEV`.

### API routes
- `POST /api/generate-bricks` — image or text → `BrickResponse`. Non-streaming.
- `POST /api/generate-stream` — same inputs → SSE with `progress | brick | rollback | result | error` events. The frontend's `BuildSession` uses this.
- `GET/POST /api/gallery`, `GET /api/gallery/{id}`, `PATCH /api/gallery/{id}/star` — archive storing `{title, caption, bricks, brick_count, stable}`.
- `GET /health`.

### Frontend surface
React 19 + Vite + Three.js (R3F). Aesthetic is "Blueprint Console" — JetBrains Mono labels, IBM Plex Sans body, Fraunces display; near-black base with an acid-lime (`#c6f432`) accent. `frontend/src/api/legogen.ts` holds the TS contracts (`BrickResponse`, `BrickCoord`, `GalleryBuild`, `StreamEvent`). Routes: `/`, `/build`, `/guide/:buildId`, `/explore`, `/about`. Styling is in `frontend/src/index.css` via `@theme` CSS variables — use those instead of hardcoding hex.

## When in doubt

`docs/TRAINING_AND_INFERENCE.md` is the detailed, line-number-anchored reference for training and the brick-generation runtime. Read it before refactoring training or stability/occupancy code.
