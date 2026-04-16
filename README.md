# LEGOGen

Two-stage AI pipeline that turns a photo or text prompt into a LEGO build. Grammar-constrained brick generation, per-stud physics validation, a 3D viewer, and a gallery of saved builds.

## Architecture

```
 Image  ─────────┐
                 ▼
          ┌────────────┐
          │  Stage 1   │   Qwen3.5-9B  (+ LoRA r=32, 4-bit NF4)
          │ image→text │   Short geometry/colour description
          └──────┬─────┘
                 │  caption
 Text prompt ────┤
                 ▼
          ┌────────────┐
          │  Stage 2   │   Qwen3.5-4B  (+ LoRA r=32, 4-bit NF4)
          │ text→bricks│   Grammar-constrained regex decoder →
          │            │   HxW (x,y,z) #RRGGBB
          └──────┬─────┘
                 │
            ┌────┴────┐
            ▼         ▼
      VoxelGrid    Stud-LP
      (collision / bounds)   (force-equilibrium; rollback on fail)
                 │
                 ▼
      React + Three.js 3D viewer + Gallery
```

Stage 1 can be skipped — a text prompt goes straight to Stage 2. Stage 2 emits bricks one at a time; each is voxel-checked, and a rolling force-equilibrium LP verifies stability. Unstable prefixes are truncated and resumed.

## Features

- **Image-to-Build** — Stage 1 captions a photo, Stage 2 emits buildable bricks
- **Text-to-Build** — prompt goes straight into the Stage 2 brick head
- **Grammar-constrained decoding** — the only 14 LEGO dims + hex-color format can be emitted
- **Physics-validated output** — every placement is checked for collision; every structure passes a per-stud LP before returning
- **Streaming UI** — the SSE endpoint pushes `progress / brick / rollback / result` events in real time; the frontend visualises brick count and rollback events as they happen
- **Layer walkthrough** — 3D viewer with past/current/future brick opacity tiers and a keyboard-controlled layer stepper
- **Gallery** — save, star, and revisit past builds; shareable deep links (`/guide/:buildId`)
- **Dev mode** — `LEGOGEN_DEV=1` uses `MockBrickPipeline` so the frontend can be built without a GPU

## Project Structure

```
backend/
  api/
    routes_generate.py   POST /api/generate-bricks, /api/generate-stream (SSE)
    routes_gallery.py    /api/gallery CRUD
  inference/
    stage1_pipeline.py   Qwen3.5-9B + Stage 1 LoRA (image → caption)
    brick_pipeline.py    Qwen3.5-4B + brick LoRA + factories + MockBrickPipeline
  brick/
    constants.py, decoder.py, occupancy.py, parser.py, stability.py
  data_pipeline/
    dataset.py                 Image transforms (shared with Stage 1 dataset)
    dataset_stage1.py          Image→caption dataset + collator
    build_stage1_dataset.py    Build Stage 1 manifest from COCO + ST2B
    prepare_brick_dataset.py   Emit brick JSONL from Rebrickable
  training/
    train_stage1.py      Fine-tune Stage 1 (Qwen3.5-9B)
    train_brick.py       Fine-tune Stage 2 brick head (Qwen3.5-4B)
    utils.py             seed_everything + setup_wandb
  storage/
    gallery_db.py        Async SQLite (bricks, caption, brick_count, stable)
  config.py              Models, paths, LoRA ranks, training hyperparameters
  app.py                 FastAPI app (CORS, lifespan, router registration, /health)

frontend/
  src/
    pages/               Home, BuildSession, GuidancePage, ExplorePage, About
    components/          Header, Footer, UploadPanel, PromptInput,
                         ProgressIndicator, BrickCoordViewer, BrickMesh,
                         GalleryCard, ErrorBoundary
    api/legogen.ts       Typed API client + brick-layer projection helpers
    index.css            Blueprint Console theme (CSS vars via @theme)

scripts/
  prepare_dataset.py         Download Rebrickable data + images
  bootstrap.sh               Environment setup
  train_full_pipeline.sh     Train Stage 2 brick + Stage 1 (in that order)
  train_brick_runpod.sh      Brick head training on RunPod

tests/
  Unit + integration tests for brick (parser / occupancy / stability / decoder),
  Stage 1 dataset, brick-dataset prep, brick-pipeline logic, api routes,
  gallery_db.
```

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- For full inference: CUDA GPU with ≥24 GB VRAM (e.g. RTX 5090). Dev mode has no GPU requirement.

### Install

```bash
pip install -r requirements.txt
cd frontend && npm install && cd -
```

### Run

```bash
# Dev (mock pipeline, no GPU)
LEGOGEN_DEV=1 uvicorn backend.app:app --reload --port 8000

# Production (loads Qwen3.5-9B + Qwen3.5-4B with LoRA adapters)
LEGOGEN_DEV=0 uvicorn backend.app:app --host 0.0.0.0 --port 8000

# Frontend
cd frontend && npm run dev
```

## Training

### Stage 1 — image → description

```bash
# Single GPU
python -m backend.training.train_stage1

# Multi-GPU (e.g. 4× RTX 5090)
torchrun --nproc_per_node=4 -m backend.training.train_stage1
```

| Param | Value |
|-------|-------|
| Base | Qwen3.5-9B |
| LoRA r / α | 32 / 64 (DoRA + rsLoRA, all-linear) |
| LR | 5e-5 (cosine) |
| Batch / accum | 8 / 1 (effective 8 × N_GPUS) |
| Max seq length | 512 |
| Epochs | 3 |
| Quantization | 4-bit NF4 |

### Stage 2 — text → brick coordinates

```bash
python -m backend.training.train_brick
```

| Param | Value |
|-------|-------|
| Base | Qwen3.5-4B |
| LoRA r / α | 32 / 64 (DoRA + rsLoRA + PiSSA; q_proj/v_proj) |
| LR | 1e-3 |
| Batch / accum | 1 / 16 (effective 16) |
| Max seq length | 4096 |
| Epochs | 3 |
| Extras | Structure-aware loss weighting, curriculum ordering, chunked cross-entropy |

Full pipeline: `bash scripts/train_full_pipeline.sh`.

## API Reference

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/generate-bricks` | Image or text → `{bricks, caption?, brick_count, stable, metadata}` |
| POST | `/api/generate-stream` | SSE — `progress / brick / rollback / result / error` events |
| GET  | `/api/gallery` | List saved builds (sort: newest / bricks / stars; q: title+caption search) |
| POST | `/api/gallery` | Save a build (`title, caption, bricks, brick_count, stable`) |
| GET  | `/api/gallery/{id}` | Fetch a saved build |
| PATCH | `/api/gallery/{id}/star` | Rate 1–5 (running average) |
| GET  | `/health` | Health check |

## Tech Stack

| Layer | Stack |
|-------|-------|
| Frontend | React 19, TypeScript, Vite, Tailwind CSS v4, Three.js, React-Three-Fiber, React Router |
| Backend | FastAPI, Python 3.11, asyncio, aiosqlite |
| Models | Qwen3.5-9B (Stage 1), Qwen3.5-4B (Stage 2), QLoRA (DoRA + rsLoRA ± PiSSA), BitsAndBytes |
| Decoding | Outlines regex-constrained logits processor |
| Physics | SciPy HiGHS linear-program solver |
| Training | HuggingFace Transformers + Trainer / TRL SFTTrainer, PEFT, W&B |
