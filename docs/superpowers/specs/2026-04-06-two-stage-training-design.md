# Two-Stage LEGO Generation Pipeline

**Date:** 2026-04-06
**Status:** Approved

## Problem

The current single-stage pipeline (image/text → LEGO JSON) has three issues:

1. **Conflicting training data formats**: Rebrickable uses category-based subassemblies (`"bricks_sloped"`) while ST2B uses layer-based (`"layer_0"`). The model sees contradictory structural patterns.
2. **Structural signal drowned by boilerplate**: JSON syntax/formatting tokens are 62.5% of the loss, while spatial layout decisions (layer names, positions, connectivity) are only 1.4%. The model learns JSON fluency but not build structure.
3. **Vision task is too hard in one step**: Going directly from pixels to structured multi-layer LEGO JSON requires the model to simultaneously understand visual content AND produce a complex structured output.

## Solution: Two-Stage Pipeline

Split generation into two stages with separate LoRA adapters on the same Qwen3.5-27B base model.

### Stage 1: Image → Structural Description

**Task**: Given a real-world image, produce a text description focused on shape, structure, colors, and proportions — optimized for downstream LEGO building.

**Example**:
- Input: Photo of a house
- Output: *"A two-story house with a rectangular base, sloped roof, front door, and two square windows. Dominant colors are red walls, dark gray roof, and white window frames."*

**LoRA config**:
- Rank: 32
- Alpha: 64
- Target modules: all-linear
- Dropout: 0.05
- Separate adapter from Stage 2
- Learning rate: 5e-5
- Epochs: 3
- Batch size: 4, gradient accumulation: 4 (effective batch 16)
- Both adapters registered via PEFT `model.load_adapter(name, path)` — swap at inference with `model.set_adapter(name)` (no reload needed, ~1ms)

**System prompt**:
```
You are a LEGO design assistant. Describe this object's shape, structure, colors,
and proportions in a way useful for building it with LEGO bricks. Focus on
geometry and spatial relationships, not materials or artistic style. Be concise.
```

### Stage 2: Text → Layer-Based LEGO JSON

**Task**: Given a text description of an object, produce a structured LEGO build as layer-based JSON.

**Example**:
- Input: *"A two-story house with a rectangular base, sloped roof..."*
- Output: JSON with `layer_0` (foundation), `layer_1` (first floor walls), `layer_2` (second floor), `layer_3` (roof slopes), each with parts, `grid_pos`, and connectivity.

**LoRA config**:
- Rank: 128
- Alpha: 256
- Target modules: all-linear
- Dropout: 0.05
- Same adapter currently being trained, retrained with improvements below

### Inference Flow

```python
# Stage 1: describe the image (base model + stage1 LoRA)
model.load_adapter("stage1-lora")
description = model.generate(image, system_prompt=STAGE1_PROMPT)

# Stage 2: generate LEGO JSON (swap to stage2 LoRA)
model.load_adapter("stage2-lora")
lego_json = model.generate(description, system_prompt=STAGE2_PROMPT)
```

For text-only input (user types "build me a red car"), skip Stage 1 and go directly to Stage 2.

## Training Data

### Stage 1 Data (~15-20k image-description pairs)

**Source 1: COCO 2017 + ST2B caption matching** (~12-15k pairs)
- Download COCO 2017 train images + category annotations
- Filter to categories overlapping ST2B's 21 categories:
  - COCO `chair` → ST2B `chair`
  - COCO `car` → ST2B `car`
  - COCO `bus` → ST2B `bus`
  - COCO `dining table` → ST2B `table`
  - COCO `couch` → ST2B `sofa`
  - COCO `bed` → ST2B `bed`
  - COCO `bench` → ST2B `bench`
  - COCO `train` → ST2B `train`
  - COCO `boat` → ST2B `vessel`
  - COCO `motorcycle` → ST2B `motorbike`
  - COCO `airplane` → ST2B `airplane`
  - COCO `truck` → ST2B `truck`
  - COCO `vase` → ST2B `vase`
  - COCO `cup` → ST2B `mug`
  - COCO `laptop` → ST2B `laptop`
  - Categories without COCO match (basket, guitar, piano, tower, bookshelf, etc.) are skipped — ST2B text-only captions still cover them in Stage 2
- For each COCO image, sample a ST2B caption from a matching-category object
- This gives real photos paired with structural descriptions in the style Stage 2 expects

**Source 2: Rebrickable set images + generated descriptions** (~1,440 pairs)
- Use existing 1,440 LEGO set images
- Generate structural descriptions from existing Rebrickable labels: extract `object`, `dominant_colors`, `dimensions_estimate`, subassembly types into a natural language sentence
- Template: *"{object}. Dominant colors: {colors}. Size: {dimensions}. Key components: {subassembly_types}."*

**Source 3: lego_minifigure_captions** (HuggingFace, ~13k pairs)
- Ready-to-use image + caption pairs for LEGO minifigures
- Adds diversity to the LEGO domain

### Stage 2 Data (~46k text-JSON pairs)

**Source: StableText2Brick only** (drop Rebrickable)
- 46,143 text→JSON pairs with consistent layer-based format
- All labels use `layer_0`...`layer_N` naming with bottom→top ordering
- All labels have proper `connects_to` chains and `position` values

**Data format changes**:
- Compact JSON (no indent) — reduces token count by 28%
- Add `grid_pos: [x, z]` per part type — coarse placement hint for 3D viewer (+15% tokens)
  - Computed by the `add_grid_pos.py` preprocessing script
  - Within each layer, parts are assigned grid positions by packing left-to-right: `x = running_stud_offset % layer_width`, `z = running_stud_offset // layer_width`
  - `running_stud_offset` increments by `part_width * quantity` for each part entry
  - `layer_width` is estimated from total stud area in the layer: `ceil(sqrt(total_area))`

**Example enhanced part entry**:
```json
{
  "part_id": "3003",
  "name": "Brick 2x2",
  "category": "Bricks",
  "color": "Red",
  "color_hex": "#C91A09",
  "is_trans": false,
  "quantity": 3,
  "grid_pos": [0, 2]
}
```

## Training Improvements for Stage 2

### Structure-Aware Loss Weighting (already implemented)

Custom `compute_loss` in the Trainer that applies per-token weights:

| Token type | Weight | Examples |
|---|---|---|
| JSON boilerplate | 0.1 | `{`, `}`, `[`, `]`, `:`, `,`, `"`, whitespace, field keys |
| Structural decisions | 5.0 | `layer_0`–`layer_9`, `bottom`, `center`, `top`, `connects_to`, `position` |
| Content (default) | 1.0 | Part IDs, colors, quantities |

Net effect: structural tokens get 50x more relative signal than boilerplate.

### Compact JSON

Text samples use `json.dumps(label)` instead of `json.dumps(label, indent=2)`. Saves ~890 tokens per sample (28% reduction). Already implemented.

### No Rebrickable in Stage 2

Removes the source of format conflicts. Stage 2 sees only layer-based ST2B data — one consistent structural pattern to learn.

## Files to Create/Modify

### New files
- `backend/data_pipeline/build_stage1_dataset.py` — COCO download, filtering, caption matching
- `backend/data_pipeline/dataset_stage1.py` — Stage 1 Dataset class
- `backend/training/train_stage1.py` — Stage 1 training script
- `backend/models/stage1_model.py` — Stage 1 LoRA model wrapper (rank 32)
- `backend/data_pipeline/add_grid_pos.py` — Script to add `grid_pos` to ST2B labels

### Modified files
- `backend/inference/pipeline.py` — Two-stage inference with adapter swapping
- `backend/data_pipeline/dataset_unified.py` — Remove Rebrickable from Stage 2 dataset (already uses compact JSON)
- `backend/training/train_unified.py` — Structure-aware loss (already implemented), train as Stage 2 only
- `backend/config.py` — Stage 1 config (model paths, LoRA rank, learning rate)
- `frontend/src/components/LegoViewer.tsx` — Use `grid_pos` for brick placement (already improved)
- `frontend/src/components/BrickMesh.tsx` — Parse dimensions from part names (already improved)

### Not modified
- ST2B label files — `grid_pos` is added by a preprocessing script, not changed at training time
- Frontend API types — `grid_pos` is optional, viewer uses it if present

## Training Schedule

1. **Stage 2 first**: Retrain text→JSON on ST2B-only with structure-aware loss, grid_pos data. Resume from scratch (the current checkpoint learned conflicting formats). ~5,646 steps, ~41 hours at 26s/step.
2. **Stage 1 second**: Train image→description on COCO+Rebrickable+minifig data. Lighter task, lower rank, faster training. ~2-3 hours estimated.
3. **Integration test**: Run full two-stage pipeline end-to-end.

## Success Criteria

- Stage 2 output consistently uses `layer_0`...`layer_N` naming (>95% of generations)
- Stage 2 output has `position: bottom` for layer_0, `position: top` for final layer
- Stage 2 output has valid `connects_to` chains between adjacent layers
- Stage 1 produces descriptions that focus on shape/structure/color, not material/style
- Full pipeline: real photo → reasonable LEGO JSON in <120 seconds total
- 3D viewer renders builds that visually resemble the input (layers stack, bricks sized correctly)
