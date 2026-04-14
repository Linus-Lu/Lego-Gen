# LEGOGen

AI-powered LEGO model generator using a two-phase pipeline. Upload a photo or describe what you want to build, and LEGOGen generates a 3D LEGO model with precise brick coordinates, colors, and an interactive viewer.

## Architecture

```
Image / Text Prompt
       |
  Two-Phase Pipeline
       |
  +---------+---------+
  |                   |
  Stage 1             |
  Image -> Text       |
  (Qwen 7B VL)        |
  |                   |
  +----> Text Description
              |
         Stage 2
         Text -> Brick Coordinates
         (Qwen 4B)
              |
         Brick Sequence with
         Rejection Sampling +
         Physics Rollback
              |
         React Frontend +
         Three.js 3D Viewer
```

### Two-Phase Pipeline

| Stage | Task | Model | Output |
|-------|------|-------|--------|
| **Stage 1** | Image -> Text Description | Qwen 7B VL + LoRA | Concise structural description |
| **Stage 2** | Text -> Brick Coordinates | Qwen 4B + LoRA | `HxW (x,y,z) #RRGGBB` per brick |

- **Stage 1** takes an image and produces a concise text description of the object's shape, structure, colors, and proportions. The vision encoder is frozen; only the language model layers are fine-tuned with a lightweight LoRA adapter (rank 32, alpha 64).

- **Stage 2** takes the text description and generates a sequence of brick placements with exact coordinates and colors. Uses rejection sampling (invalid bricks are discarded and regenerated) and physics rollback (unstable structures are truncated and rebuilt).

Both stages use **QLoRA** (4-bit NF4 quantization) for efficient fine-tuning on consumer GPUs.

## Brick Output Format

Each brick is a single line:

```
2x4 (5,3,0) #C91A09
1x2 (7,3,1) #FFFFFF
2x4 (5,3,1) #0055BF
```

Format: `<height>x<width> (<x>,<y>,<z>) #<hex_color>`

Allowed dimensions: `1x1, 1x2, 2x1, 1x4, 4x1, 1x6, 6x1, 1x8, 8x1, 2x2, 2x4, 4x2, 2x6, 6x2`

All bricks are 1 unit tall. Grid is 20x20x20.

## Project Structure

```
backend/
  api/
    routes_generate.py       # POST /api/generate-bricks (image or text)
  models/
    unified_model.py         # Qwen VL model wrapper (Stage 1 + adapter swapping)
    tokenizer.py             # Prompt templates, JSON parsing
  inference/
    pipeline.py              # TwoStagePipeline + MockPipeline
    brick_pipeline.py        # Stage 2: text -> brick coordinates (Qwen 4B)
  brick/
    constants.py             # Brick shapes, grid size, color palette
    parser.py                # Parse/serialize brick text format
    decoder.py               # Constrained decoding state machine
    occupancy.py             # Voxel grid collision detection
    stability.py             # Physics stability checks
  data_pipeline/
    dataset_stage1.py        # Stage 1 dataset (image-description pairs)
    build_stage1_dataset.py  # Build Stage 1 manifest from COCO + ST2B
    prepare_brick_dataset.py # Build Stage 2 training data from ST2B
    dataset.py               # Image augmentation transforms
  training/
    train_stage1.py          # Stage 1 training (multi-GPU DDP)
    train_brick.py           # Stage 2 training (structure-aware loss)
    utils.py                 # Metrics, seeding, W&B logging
  config.py                  # All configuration: models, paths, hyperparameters
  app.py                     # FastAPI app with CORS and lifespan

frontend/
  src/
    pages/
      Home.tsx               # Landing page
      BuildSession.tsx       # Main build interface (upload/prompt -> 3D viewer)
      About.tsx              # Tech stack info
    components/
      UploadPanel.tsx        # Drag-and-drop image upload
      PromptInput.tsx        # Text prompt input
      BrickCoordViewer.tsx   # Three.js 3D brick viewer
      BrickMesh.tsx          # Individual brick 3D mesh
    api/
      legogen.ts             # API client (generateBricks)

scripts/
  prepare_dataset.py         # Download Rebrickable CSVs, build labels, fetch images
  bootstrap.sh               # Environment setup
  train_full_pipeline.sh     # Full training orchestration
  train_brick_runpod.sh      # RunPod training script
  benchmark.py               # Evaluation benchmark
```

## Setup

### Prerequisites

- Python 3.11+
- CUDA GPU with 16GB+ VRAM (24GB+ recommended for training)
- Node.js 18+

### Install Dependencies

```bash
pip install -r requirements.txt
cd frontend && npm install
```

### Prepare Training Data

```bash
# 1. Download Rebrickable data and images
python scripts/prepare_dataset.py --max-sets 2000

# 2. Build Stage 1 manifest (COCO + ST2B image-description pairs)
python -m backend.data_pipeline.build_stage1_dataset

# 3. Build Stage 2 brick training data from StableText2Brick
python -m backend.data_pipeline.prepare_brick_dataset
```

### Train Models

```bash
# Stage 1: Image -> Text Description (Qwen VL 7B + LoRA)
# Single GPU:
python -m backend.training.train_stage1

# Multi-GPU (4x GPUs):
torchrun --nproc_per_node=4 -m backend.training.train_stage1

# Stage 2: Text -> Brick Coordinates (Qwen 4B + LoRA)
python -m backend.training.train_brick
```

### Run the App

```bash
# Dev mode (mock pipeline, no GPU needed)
LEGOGEN_DEV=1 uvicorn backend.app:app --reload

# Production mode (loads trained models)
LEGOGEN_DEV=0 uvicorn backend.app:app --host 0.0.0.0 --port 8000

# Frontend
cd frontend && npm run dev
```

## Training Details

### Stage 1: Image -> Text (Qwen 7B VL)

| Param | Value |
|-------|-------|
| Base model | Qwen VL 7B |
| LoRA rank / alpha | 32 / 64 |
| LoRA targets | all-linear (vision encoder frozen) |
| Learning rate | 5e-5 (cosine schedule) |
| Batch size | 8 per device |
| Epochs | 3 |
| Warmup | 100 steps |
| Quantization | 4-bit NF4 |
| Max sequence length | 512 tokens |

### Stage 2: Text -> Brick Coordinates (Qwen 4B)

| Param | Value |
|-------|-------|
| Base model | Qwen 4B |
| LoRA rank / alpha | 32 / 64 |
| LoRA targets | q_proj, v_proj |
| Learning rate | 1e-3 (cosine schedule) |
| Batch size | 1 (grad accum 16, effective 16) |
| Epochs | 3 |
| Quantization | full precision (bf16) |
| Max sequence length | 4096 tokens |
| Loss weighting | Structure-aware (3x on coordinates, 0.1x on syntax) |
| Curriculum | Untruncated samples first |

### Stage 2 Inference Features

- **Rejection sampling**: Invalid bricks (out of bounds, collisions, bad dimensions) are discarded and regenerated with increasing temperature
- **Physics rollback**: When an unstable brick is detected, the sequence is truncated to the last stable point and generation resumes
- **Temperature scheduling**: Starts at 0.6, increments by 0.01 per rejection up to 2.0

## API Reference

### `POST /api/generate-bricks`

Generate a LEGO model from an image or text prompt.

- **Body**: `multipart/form-data` with optional `image` file and/or `prompt` string
- **Response**:
```json
{
  "bricks": "2x4 (5,3,0) #C91A09\n1x2 (7,3,1) #FFFFFF\n...",
  "caption": "A small red house with white trim",
  "brick_count": 42,
  "stable": true,
  "metadata": {
    "model_version": "qwen35-4b-brick-v1",
    "generation_time_ms": 3200,
    "rejections": 5,
    "rollbacks": 0
  }
}
```

### `GET /health`

Health check endpoint.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, TypeScript, Tailwind CSS, Three.js, React-Three-Fiber, Vite |
| Backend | FastAPI, Python 3.11 |
| Stage 1 | Qwen 7B VL, QLoRA, BitsAndBytes |
| Stage 2 | Qwen 4B, LoRA, rejection sampling, physics rollback |
| Training | HuggingFace Transformers + TRL SFTTrainer, PEFT, W&B |
| Data | StableText2Brick (HuggingFace), COCO 2017, Rebrickable |
