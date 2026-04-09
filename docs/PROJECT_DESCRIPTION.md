# LEGO-Gen: AI-Powered LEGO Set Generator

## Overview

LEGO-Gen is an AI system that generates step-by-step LEGO building instructions from images or text prompts. It uses a **two-stage fine-tuned pipeline** on **Qwen3.5-9B** (both stages) with adapter swapping, producing structured JSON with real part IDs, layer-based assembly, and spatial relationships — rendered in an interactive 3D viewer.

---

## Two-Stage Architecture

```
┌──────────────┐         ┌──────────────────────────┐
│  Image Input │         │     Text Input            │
└──────┬───────┘         └────────────┬──────────────┘
       │                              │
       V                              │ (skip Stage 1)
┌──────────────────────┐              │
│  STAGE 1             │              │
│  Image → Description │              │
│  Qwen3.5-9B + LoRA   │              │
│  (r=32, α=64)        │              │
│  "stage1" adapter    │              │
└──────────┬───────────┘              │
           │                          │
           V                          V
   ┌───────────────────────────────────────┐
   │  Structural Description (2-3 sentences)│
   │  Shape, colors, proportions, geometry  │
   └───────────────────┬───────────────────┘
                       │
                       V
            ┌──────────────────────┐
            │  STAGE 2              │
            │  Text → LEGO JSON     │
            │  Qwen3.5-9B + LoRA    │
            │  (r=64, α=128)        │
            │  "default" adapter   │
            └──────────┬───────────┘
                       │
          ┌────────────┼────────────┐
          V            V            V
   ┌───────────┐ ┌──────────┐ ┌──────────────┐
   │ Constraint│ │Stability │ │ Build Steps  │
   │ Engine    │ │Checker   │ │ Generator    │
   │ (repair)  │ │(10 checks│ │ (ordering)   │
   └───────────┘ └──────────┘ └──────────────┘
                       │
                       V
            ┌──────────────────────┐
            │  Frontend (React +   │
            │  Three.js 3D Viewer) │
            └──────────────────────┘
```

---

## Stage 1: Image → Structural Description (Qwen3.5-9B)

**Purpose**: Extract LEGO-relevant features from a real-world image.

| Parameter | Value |
|-----------|-------|
| Base model | Qwen3.5-9B-Instruct (multimodal) |
| LoRA rank/alpha | 32 / 64 |
| Target modules | all-linear (hybrid DeltaNet/Attention) |
| Epochs | 3 |
| Learning rate | 5e-5 (cosine schedule) |
| Max seq length | 512 |
| Effective batch | 16 (4 x 4 grad accum) |

### Training Data (~20k image-description pairs)

- **COCO 2017 + StableText2Brick matching** (~12-15k) — real photos paired with structural LEGO descriptions via category overlap (chair↔chair, car↔car, bus↔bus, table↔dining table, sofa↔couch, etc.)
- **Rebrickable set images** (~1,440) — extracted descriptions from labels: `"{object}. Dominant colors: {colors}. Size: {dims}. Components: {types}."`
- **LEGO minifigure captions** (~13k) — ready-to-use image+caption pairs

### System Prompt

> You are a LEGO design assistant. Describe this object's shape, structure, colors, and proportions in a way useful for building it with LEGO bricks. Focus on geometry and spatial relationships, not materials or artistic style. Be concise — one to three sentences.

### Example

- **Input**: Photo of a house
- **Output**: `"A two-story house with a rectangular base, sloped roof, front door, and two square windows. Dominant colors: red walls, dark gray roof, white window frames."`

---

## Stage 2: Text → Layer-Based LEGO JSON (Qwen3.5-9B)

**Purpose**: Convert a structural description into a complete, buildable LEGO set definition.

| Parameter | Value |
|-----------|-------|
| Base model | Qwen3.5-9B-Instruct (text-only mode) |
| LoRA rank/alpha | 64 / 128 |
| Target modules | all-linear |
| Epochs | 5 |
| Learning rate | 3e-5 (cosine schedule) |
| Max seq length | 2048 |
| Effective batch | 16 (2 x 8 grad accum) |
| Gradient checkpointing | Enabled |

### Training Data (~47k text-JSON pairs)

- **StableText2Brick** (46,143 samples) — 21 object categories with consistent `layer_0...layer_N` bottom-to-top format, proper position/orientation/connects_to chains
- **Rebrickable** (3x upsampled → ~4,320 samples) — converted from category-based to layer-based format

### Structure-Aware Loss Weighting

| Token Type | Weight | Examples |
|-----------|--------|---------|
| JSON boilerplate | 0.1x | `{`, `}`, `"`, whitespace |
| Structural decisions | 5.0x | layer names, positions, connectivity |
| Content | 1.0x | part IDs, colors, quantities |

This gives a **50x relative signal boost** for layout decisions over syntax tokens.

---

## Adapter Swapping Mechanism

Both stages share the same base Qwen3.5-9B model. At inference:

```python
# Load once at startup
model = load("Qwen3.5-9B")
model.load_adapter("stage1", stage1_checkpoint)   # r=32
model.load_adapter("default", stage2_checkpoint)  # r=64

# Image path (two-stage):
model.set_adapter("stage1")        # ~1ms swap, no reload
description = generate(image)      # Stage 1
model.set_adapter("default")       # ~1ms swap
lego_json = generate(description)  # Stage 2

# Text path (single-stage):
# Already on "default" adapter
lego_json = generate(user_prompt)  # Stage 2 only
```

---

## Complete Data Flow

### Image-to-Build

1. User uploads image → `UploadPanel.tsx`
2. `POST /api/generate` → `UnifiedPipeline.generate_build()`
3. **Stage 1**: Swap to "stage1" adapter → image processed through Qwen3.5-9B vision encoder → generates 1-3 sentence description
4. **Stage 2**: Swap to "default" adapter → description tokenized with planner system prompt → generates structured JSON (up to 2048 tokens)
5. **Constraint Engine**: Repair malformed JSON → validate schema → enforce valid enums/ranges → fix connects_to references
6. **Stability Checker**: 10 checks (part existence, color validity, foundation strength, connectivity, support ratio, center of mass, cantilevers, build order) → score 0-100
7. **Build Steps Generator**: Sort subassemblies bottom-to-top → generate sequential instructions
8. Response → Frontend renders step navigator + Three.js 3D viewer

### Text-to-Build

Same as above but **skips Stage 1** — text prompt goes directly to Stage 2.

---

## Three-Layer Caching System

| Layer | What | Speed Impact |
|-------|------|-------------|
| **KV Prefix Cache** | Pre-computed KV states for static system prompts; clone at inference (~1ms) | ~5% speedup |
| **Response Cache** | LRU + 1hr TTL for identical inputs (SHA256 keyed, max 256 entries) | ~10x on cache hit |
| **Tokenization Cache** | Pre-computed chat template strings | Avoids redundant tokenization |

---

## Output JSON Schema

```json
{
  "set_id": "custom-001",
  "object": "Cozy Family House",
  "category": "City",
  "subcategory": "Residential",
  "complexity": "intermediate",
  "total_parts": 86,
  "dominant_colors": ["Red", "White", "Bright Orange"],
  "dimensions_estimate": {
    "width": "medium",
    "height": "medium",
    "depth": "small"
  },
  "subassemblies": [
    {
      "name": "layer_0_foundation",
      "type": "Baseplates",
      "parts": [
        {
          "part_id": "3811",
          "name": "Baseplate 32x32",
          "category": "Baseplates",
          "color": "Green",
          "color_hex": "#237841",
          "is_trans": false,
          "quantity": 1,
          "grid_pos": [0, 0]
        }
      ],
      "spatial": {
        "position": "bottom",
        "orientation": "flat",
        "connects_to": ["layer_1_walls"]
      }
    }
  ],
  "build_hints": [
    "Start with the green base plate",
    "Build walls before attaching the roof"
  ]
}
```

---

## Stability Checker (10 Checks)

| # | Check | Description |
|---|-------|-------------|
| 1 | **part_existence** | All part IDs exist in Rebrickable catalog |
| 2 | **part_compatibility** | Part categories match subassembly types |
| 3 | **color_validity** | All colors are valid LEGO colors |
| 4 | **quantity_reasonableness** | No single part >200 (fail) or >50 (warn) |
| 5 | **foundation** | Base has adequate support (min 20 studs) |
| 6 | **connectivity** | Subassemblies properly connected |
| 7 | **support_ratio** | Upper layers not top-heavy (ratio <3.0) |
| 8 | **center_of_mass** | COM within 60% of base footprint |
| 9 | **cantilever** | Overhang < 2x connector count |
| 10 | **build_order** | Follows logical bottom-to-top sequence |

---

## Training Pipeline

### Training Order

1. **Stage 2 first** (~41hrs, ~5,646 steps on RTX 5090) — text→JSON on ST2B + Rebrickable with structure-aware loss
2. **Stage 1 second** (~2-3hrs) — image→description on COCO + Rebrickable + minifig data
3. **Integration test** — full two-stage end-to-end validation

### Curriculum Sampler

Estimates token counts from label file sizes, partitions into "fits" (≤ max_length) vs "truncated", always trains on complete samples first — so early-stopped runs see highest-quality data. Shuffles within each partition every epoch.

### Evaluation Metrics

- **JSON validity rate** — % of outputs that parse as valid JSON
- **Field accuracy** — exact match on category, subcategory, complexity
- **Color F1** — set-based F1 on dominant colors
- **Parts F1** — quantity-weighted F1 on part lists per subassembly

---

## Constraint Engine

### JSON Repair Pipeline

1. Try `json_repair` library first
2. Manual repair: fix trailing commas, close unclosed strings, balance brackets/braces
3. Validate schema: check all required fields present
4. Enforce valid values: clamp complexity by total_parts, default dimensions, fix spatial positions
5. Repair connects_to: validate references match actual subassembly names

---

## Frontend Stack

**React 19 + TypeScript + Vite**

| Component | Purpose |
|-----------|---------|
| `UploadPanel` | Drag-and-drop image upload |
| `PromptInput` | Text prompt input |
| `StepList` / `StepDetail` | Step-by-step build navigation |
| `LegoViewer` | Three.js 3D progressive build viewer with orbit controls, transparency toggle, grid placement |
| `ValidationPanel` | Stability score + per-check details |
| `ColorLegend` | Brick color key |

---

## Backend Stack

**FastAPI** with async handlers (GPU inference via `run_in_executor`), SQLite gallery storage, session tracking, 120s request timeout.

### API Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/api/generate` | POST | Image + optional prompt → build |
| `/api/generate-from-text` | POST | Text prompt → build |
| `/api/generate-bricks` | POST | Image or text → brick coordinates |
| `/api/generate-stream` | POST | SSE streaming with stage progress |
| `/api/validate` | POST | Validate existing build |
| `/health` | GET | Health check |

---

## Model Summary

| Model | Task | Base | Training Data | LoRA Config | Status |
|-------|------|------|---------------|-------------|--------|
| **Stage 1** | Image → Description | Qwen3.5-9B | COCO + Rebrickable + minifig (~20k) | r=32, α=64 | Active |
| **Stage 2** | Text → LEGO JSON | Qwen3.5-9B | ST2B + Rebrickable 3x (~47k) | r=64, α=128 | Active |
| **Brick Model** | Text → Brick coords | Qwen3.5-4B | Custom brick format | r=32, α=64 | Experimental |

---

## Key Design Decisions

- **Two-stage separation** isolates image understanding from LEGO structure generation, avoiding conflicting training signals between Rebrickable and ST2B formats
- **Same base model, different adapters** — memory efficient, ~1ms swap time
- **Structure-aware loss** — 50x signal boost for layout decisions over JSON syntax
- **Layer-based format** — consistent `layer_0...layer_N` bottom-to-top naming enables proper build ordering
- **4-bit NF4 quantization** — fits 9B model on single GPU with LoRA training
- **Curriculum learning** — untruncated samples first ensures early-stopped runs train on highest-quality data

---

## Inference Performance

| Stage | Input | Output | Latency |
|-------|-------|--------|---------|
| Stage 1 | PIL Image (224-512px) | 50-150 tokens (~2-3 sentences) | ~5-10s |
| Stage 2 | 50-150 tokens description | 500-2000 tokens (JSON) | ~10-20s |
| **Total two-stage** | Image | Full build JSON | **~20-30s** |
| Cache hit | Any | Cached response | ~2-3s |
| Stability check | JSON | Validation report | ~100-200ms |

---

## Project Structure

```
Lego-Gen/
├── backend/
│   ├── api/                    # FastAPI routes
│   ├── models/                 # Model wrappers (unified_model.py)
│   ├── inference/              # Pipeline, constraint engine, stability checker
│   ├── data_pipeline/          # Dataset classes, preprocessing, data builders
│   ├── training/               # Training scripts (train_unified.py, train_stage1.py)
│   ├── brick/                  # Brick coordinate domain (constants, parser, occupancy)
│   ├── storage/                # SQLite gallery, session store
│   └── config.py               # Centralized configuration
├── frontend/
│   ├── src/
│   │   ├── pages/              # Home, BuildSession, About
│   │   ├── components/         # UploadPanel, LegoViewer, StepList, etc.
│   │   └── api/                # API client
│   └── vite.config.ts
├── scripts/                    # Dataset preparation utilities
├── data/                       # Images, labels, caches (not in repo)
├── docs/                       # Design specs and plans
└── README.md
```
