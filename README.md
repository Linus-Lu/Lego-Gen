# LEGOGen

Two-stage AI pipeline that turns a photo or text prompt into a LEGO build — structured JSON, ordered build steps, a brick-accurate 3D viewer, stability validation, and a gallery of saved builds.

## Architecture

```
 Image  ─────────────┐
                     ▼
              ┌──────────────┐
              │   Stage 1    │   Qwen3.5-9B  (+ LoRA r=32)
              │ image → text │   Short geometry description
              └──────┬───────┘
                     │  (concise prompt)
 Text prompt ────────┤
                     ▼
              ┌──────────────┐
              │   Stage 2    │   Qwen3.5-9B  (+ LoRA)   → LEGO JSON
              │  text → …    │   Qwen3.5-4B  (+ LoRA)   → brick coords
              └──────┬───────┘
                     ▼
        Constraint engine (validate / repair)
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   Build-step generator    Stability checker
          │                     │
          ▼                     ▼
       React + Three.js (3D viewer, gallery, guidance)
```

The pipeline has two entry points — describe an image (Stage 1 feeds Stage 2) or take a text prompt directly into Stage 2. Stage 2 has two heads: a JSON head that produces a part-list description, and a brick-coordinate head that produces placement coordinates for the 3D viewer.

## Features

- **Image-to-Build** — upload a photo, Stage 1 writes a concise prompt, Stage 2 designs a LEGO model
- **Text-to-Build** — describe what you want; Stage 2 generates directly
- **Brick-accurate 3D viewer** — render the model from brick coordinates with orbit controls, transparency, step-by-step progression
- **Guidance mode** — walk through the build one step at a time with part lists, voice narration, and a parts checklist
- **Gallery** — save, star, and revisit past builds; shareable deep links (`/guide/:buildId`)
- **Explore page** — browse community-style saved builds
- **Compare tab** — diff two builds side-by-side
- **Validation** — stability checker scores structural integrity (foundation, connectivity, cantilever, center-of-mass…) and legality (known parts, valid colors, legal quantities); constraint engine auto-repairs malformed JSON
- **Caching** — KV-prefix, response, and tokenization caches (see `backend/inference/cache.py`)
- **Dev mode** — `LEGOGEN_DEV=1` uses `MockPipeline` so the frontend can be built without a GPU

## Project Structure

```
backend/
  api/
    routes_generate.py     POST /api/generate, /generate-from-text, /generate-bricks, /generate-stream (SSE)
    routes_validate.py     POST /api/validate
    routes_gallery.py      GET/POST /api/gallery, GET /api/gallery/{id}, PATCH /api/gallery/{id}/star
  models/
    unified_model.py       Qwen3.5-9B + QLoRA wrapper (serves both Stage 1 and Stage 2 JSON head)
    tokenizer.py           Chat templates, JSON extraction, thinking-block stripping
  inference/
    pipeline.py            UnifiedPipeline (Stage 1 + Stage 2 JSON) + MockPipeline
    brick_pipeline.py      Brick-coordinate head (Qwen3.5-4B)
    constraint_engine.py   JSON schema validation + repair
    stability_checker.py   10-check structural/legality scoring (0–100)
    postprocess_manual.py  JSON → ordered build steps
    cache.py               KV-prefix / response / tokenization caches
  brick/
    constants.py, decoder.py, occupancy.py, parser.py, stability.py
  data_pipeline/
    dataset.py, dataset_stage1.py
    build_stage1_dataset.py    Build Stage 1 image→description manifest
    prepare_brick_dataset.py   Prepare Stage 2 brick-coordinate training data
  training/
    train_stage1.py        Fine-tune Stage 1 (image → description)
    train_brick.py         Fine-tune Stage 2 brick head (text → bricks)
    utils.py               Shared training utils + metrics
  storage/
    gallery_db.py          Async SQLite store for saved builds
  config.py                Models, paths, LoRA ranks, training hyperparameters, cache flags
  app.py                   FastAPI app (CORS, lifespan, router registration, /health)

frontend/
  src/
    pages/
      Home.tsx             Landing
      BuildSession.tsx     Upload / prompt → results
      GuidancePage.tsx     Step-by-step guided build (/guide/:buildId)
      ExplorePage.tsx      Browse saved builds (/explore)
      About.tsx            Tech/about
    components/
      UploadPanel, PromptInput, ProgressIndicator
      LegoViewer, BrickCoordViewer, BrickMesh, ThreeDPlaceholder
      StepList, StepDetail, StepControls, PartsChecklist
      GalleryTab, GalleryCard, CompareTab, GuidanceViewer
      ValidationPanel, ColorLegend, MetricsBar, VoiceNarrator
      Header, Footer, ErrorBoundary
    api/legogen.ts         Typed API client

scripts/
  prepare_dataset.py         Download Rebrickable data + images
  benchmark.py               End-to-end latency benchmark
  bootstrap.sh               Environment setup
  train_full_pipeline.sh     Train Stage 1 + Stage 2 brick head
  train_brick_runpod.sh      Brick head training on RunPod

tests/
  Unit + integration tests for gallery_db, cache, constraint_engine,
  stability_checker, postprocess_manual, tokenizer, brick pipeline,
  api routes, two-stage pipeline.
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

# Production (loads Qwen3.5-9B + LoRA adapters)
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
| LoRA r / α | 32 / 64 |
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
| LoRA r / α | 32 / 64 |
| LR | 1e-3 |
| Batch / accum | 1 / 16 (effective 16) |
| Max seq length | 4096 |
| Epochs | 3 |
| Extras | Structure-aware loss weighting, curriculum ordering, chunked cross-entropy |

Full pipeline: `bash scripts/train_full_pipeline.sh`.

## API Reference

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/generate` | Image → full build (JSON + steps + validation) |
| POST | `/api/generate-from-text` | Text → full build |
| POST | `/api/generate-bricks` | Image or text → brick-coordinate model (for 3D viewer) |
| POST | `/api/generate-stream` | SSE stream with per-stage progress + final result |
| POST | `/api/validate` | Validate a LEGO description (stability + legality) |
| GET  | `/api/gallery` | List saved builds |
| POST | `/api/gallery` | Save a build |
| GET  | `/api/gallery/{id}` | Fetch a saved build |
| PATCH | `/api/gallery/{id}/star` | Star / unstar |
| GET  | `/health` | Health check |

## Tech Stack

| Layer | Stack |
|-------|-------|
| Frontend | React 19, TypeScript, Vite, Tailwind CSS, Three.js, React-Three-Fiber, React Router |
| Backend | FastAPI, Python 3.11, asyncio, aiosqlite |
| Models | Qwen3.5-9B (Stage 1 + Stage 2 JSON), Qwen3.5-4B (Stage 2 brick), QLoRA, BitsAndBytes |
| Training | HuggingFace Transformers + Trainer, PEFT, W&B |
