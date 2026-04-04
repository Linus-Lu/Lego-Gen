# LEGOGen

AI-powered LEGO set generator that creates step-by-step building instructions from images or text prompts. Upload a photo of any object or describe what you want to build, and LEGOGen designs a complete LEGO model with real part IDs, color assignments, subassembly-level spatial relationships, and an interactive 3D build viewer.

## Architecture

```
Image / Text Prompt
       |
  FastAPI Backend
       |
  +----+----+
  |         |
  V2        V3
  Vision    Planner
  Encoder   LM
  (Qwen3    (Qwen3.5
   -VL-8B)   -9B)
  |         |
  +----+----+
       |
  Structured JSON Description
       |
  Constraint Engine (validate + repair)
       |
  Build Step Generator
       |
  React Frontend + Three.js 3D Viewer
```

### Two Model Pipeline

| Model | Task | Base | Training Data |
|-------|------|------|---------------|
| **V2 Vision Encoder** | Image -> JSON | Qwen3-VL-8B-Instruct | 1.9k Rebrickable sets with images |
| **V3 Planner LM** | Text -> JSON | Qwen3.5-9B | 1.9k Rebrickable (3x upsampled) + 41k StableText2Brick |

Both models are fine-tuned with **QLoRA** (4-bit NF4 quantization). Vision encoder uses LoRA rank 32/alpha 64 on attention + MLP layers. Planner uses LoRA rank 64/alpha 128 with `all-linear` targeting for Qwen3.5's hybrid DeltaNet/Attention architecture.

## Features

- **Image-to-Build**: Upload any image (photo, drawing, existing LEGO set) and get a full LEGO build plan
- **Text-to-Build**: Describe what you want ("Build me a red sports car") and get a complete parts list and instructions
- **Interactive 3D Viewer**: Step-by-step build visualization with color-coded bricks, orbit controls, and transparency support (schematic layout — not exact brick geometry)
- **Real Part IDs**: Uses actual LEGO part numbers from the Rebrickable catalog
- **JSON Validation & Repair**: Constraint engine validates model outputs against a strict schema and auto-repairs common issues
- **Diverse Prompt Training**: 17 prompt templates with color-aware variants for robust text understanding
- **Curriculum Learning**: Prompt rotation per epoch for training diversity

## Output Schema

The models produce structured JSON with:

```json
{
  "set_id": "custom-001",
  "object": "Cozy Family House",
  "category": "City",
  "subcategory": "Residential",
  "complexity": "intermediate",
  "total_parts": 86,
  "dominant_colors": ["Red", "White", "Bright Orange"],
  "dimensions_estimate": {"width": "medium", "height": "medium", "depth": "small"},
  "subassemblies": [
    {
      "name": "base_plate",
      "type": "Baseplates",
      "parts": [
        {"part_id": "3811", "name": "Baseplate 32x32", "color": "Green", "color_hex": "#237841", "quantity": 1}
      ],
      "spatial": {"position": "bottom", "orientation": "flat", "connects_to": ["walls_lower"]}
    }
  ],
  "build_hints": ["Start with the green base plate", "Build walls before attaching the roof"]
}
```

## Project Structure

```
backend/
  api/
    routes_generate.py       # POST /api/generate (image), /api/generate-from-text
    routes_validate.py       # POST /api/validate (build stability/legality check)
  models/
    vision_encoder.py        # Qwen3-VL + QLoRA wrapper
    planner_lm.py            # Qwen3.5-9B + QLoRA wrapper
    tokenizer.py             # Prompt templates, JSON parsing, chat message builders
  inference/
    pipeline.py              # LegoGenPipeline (image) + PlannerPipeline (text) + MockPipeline (dev)
    constraint_engine.py     # Schema validation, JSON repair, value enforcement
    stability_checker.py     # Build stability/legality checker (10 checks, 0-100 scoring)
    postprocess_manual.py    # Convert JSON descriptions to ordered build steps
  data_pipeline/
    dataset.py               # Vision dataset (image-JSON pairs with augmentation)
    dataset_planner.py       # Planner dataset (text-JSON pairs, Rebrickable + ST2B)
    part_library.py          # Rebrickable part/color/category cache
    manuals_loader.py        # Rebrickable API data fetcher
    manuals_preprocess.py    # Raw API data -> structured JSON labels
  training/
    train_vision.py          # V2 vision model training script
    train_planner.py         # V3 planner model training script
    utils.py                 # Metrics (JSON validity, field accuracy, color F1, parts F1)
  config.py                  # All configuration: models, paths, hyperparameters
  app.py                     # FastAPI app with CORS, lifespan, static file serving

frontend/
  src/
    pages/
      Home.tsx               # Landing page with hero and feature cards
      BuildSession.tsx       # Main build interface (upload/prompt -> steps -> 3D viewer)
      About.tsx              # Tech stack and project info
    components/
      UploadPanel.tsx        # Drag-and-drop image upload
      PromptInput.tsx        # Text prompt input
      StepList.tsx           # Step navigation sidebar
      StepDetail.tsx         # Current step parts and instructions
      LegoViewer.tsx         # Three.js 3D progressive build viewer
      ValidationPanel.tsx    # Build stability/legality validation results
      ColorLegend.tsx        # Color key display
    api/
      legogen.ts             # API client (generateBuild, generateBuildFromText)

scripts/
  prepare_dataset.py         # Download Rebrickable CSVs, build labels, fetch images
  convert_st2b.py            # Convert HuggingFace StableText2Brick to our schema
  prepare_planner_prompts.py # Generate diverse prompt variants per label
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

# 2. Convert StableText2Brick dataset (for planner training)
python scripts/convert_st2b.py --split train
python scripts/convert_st2b.py --split test

# 3. Generate prompt variants for planner training
python scripts/prepare_planner_prompts.py
```

### Train Models

```bash
# V2: Image-to-JSON (Qwen3-VL + QLoRA)
python -m backend.training.train_vision --epochs 3 --batch-size 2

# V3: Text-to-JSON (Qwen3.5-9B + QLoRA)
python -m backend.training.train_planner --epochs 5 --batch-size 2
```

Both scripts support `--resume <checkpoint-path>` to continue from a saved checkpoint and `--no-wandb` to disable W&B logging.

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

### V2 Vision Encoder

| Param | Value |
|-------|-------|
| Base model | Qwen3-VL-8B-Instruct |
| LoRA rank / alpha | 32 / 64 |
| Learning rate | 1e-4 (cosine schedule) |
| Batch size | 2 (grad accum 16, effective 32) |
| Epochs | 3 |
| Warmup | 200 steps |
| Quantization | 4-bit NF4 |
| Vision encoder | Frozen |

### V3 Planner LM

| Param | Value |
|-------|-------|
| Base model | Qwen3.5-9B |
| LoRA rank / alpha | 64 / 128 |
| LoRA targets | all-linear (hybrid DeltaNet/Attention) |
| Learning rate | 3e-5 (cosine schedule) |
| Batch size | 2 (grad accum 8, effective 16) |
| Epochs | 5 |
| Warmup | 300 steps |
| Quantization | 4-bit NF4 |
| Training data | ~47k samples (Rebrickable 3x upsampled + StableText2Brick) |

### Evaluation Metrics

Computed during evaluation steps via `compute_metrics` in both training scripts:

- **JSON Validity Rate**: % of outputs that parse as valid JSON
- **Field Accuracy**: Exact match on category, subcategory, complexity
- **Color F1**: Set-based F1 on predicted vs reference dominant colors
- **Parts F1**: Quantity-weighted F1 on predicted vs reference part lists

## API Reference

### `POST /api/generate`

Upload an image to generate a LEGO build plan.

- **Body**: `multipart/form-data` with `image` file and optional `prompt` string
- **Response**: `{description, steps[], metadata: {model_version, generation_time_ms, json_valid, errors}, validation: {score, checks[], summary}}`

### `POST /api/generate-from-text`

Generate a LEGO build plan from a text description.

- **Body**: `multipart/form-data` with `prompt` string
- **Response**: Same as above

### `POST /api/validate`

Validate an existing LEGO build description for structural stability and part legality.

- **Body**: `application/json` with a `LegoDescription` object
- **Response**: `{score: 0-100, checks: [{name, category, status, message, details?}], summary}`
- **Check categories**: `legality` (part existence, compatibility, colors, quantities) and `stability` (foundation, connectivity, support ratio, build order, center of mass, cantilever)

### `GET /health`

Health check endpoint.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, TypeScript, Tailwind CSS, Three.js, React-Three-Fiber, Vite |
| Backend | FastAPI, Python 3.11 |
| Models | Qwen3-VL-8B, Qwen3.5-9B, QLoRA, BitsAndBytes |
| Training | HuggingFace Transformers + Trainer, PEFT, W&B |
| Data | Rebrickable database, StableText2Brick (HuggingFace) |

## What's New in V3

- **Qwen3-VL-8B vision model** replacing Qwen2.5-VL-7B for image-to-JSON generation
- **Qwen3.5-9B planner model** with hybrid DeltaNet/Attention architecture for text-to-JSON generation
- **StableText2Brick integration**: 41k additional training examples (per-brick coordinates are aggregated into subassembly-level descriptions during conversion)
- **Expanded LoRA**: rank 64/alpha 128 with `all-linear` targeting for Qwen3.5's hybrid layers
- **Lower learning rates** (2e-4 -> 1e-4 for vision, 3e-5 for planner) for more stable convergence
- **Optimized training**: 5 epochs, batch 2 with grad accum 8 (effective 16), gradient checkpointing enabled (RTX 5090)
- **Prompt diversity**: 17 template variants with color-aware prompts and curriculum rotation
- **Gradient checkpointing fix**: `enable_input_require_grads()` for QLoRA + gradient checkpointing compatibility
- **Dual pipeline**: Independent vision and planner inference paths with shared constraint engine
