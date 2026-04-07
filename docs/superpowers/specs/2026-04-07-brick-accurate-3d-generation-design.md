# Brick-Accurate 3D Generation Design

**Date**: 2026-04-07
**Status**: Draft
**Goal**: Make the 3D viewer render LEGO models that actually look like the described object by adopting autoregressive per-brick coordinate generation.

---

## Problem

The current system outputs parts lists with quantities but no per-brick 3D coordinates.
The viewer packs each subassembly as a flat grid and stacks layers vertically.
A chair looks like a stack of pancakes, not a chair.

## Approach

Adopt the BrickGPT approach (CMU, ICCV 2025): autoregressive brick-by-brick generation
on a 20x20x20 stud grid with per-brick `[x,y,z]` coordinates. Extend with color support
(our unique contribution) and keep our existing image-to-text Stage 1.

---

## Architecture

### Two-stage pipeline

```
Stage 1 (existing):  Qwen3.5-9B + LoRA
                      Image → text description
                      Already trained, no changes needed.

Stage 2 (new):        Qwen3.5-4B + LoRA
                      Text → colored brick sequence (autoregressive, one brick per line)
```

Stage 1 handles vision understanding (what the object looks like).
Stage 2 handles spatial reasoning (where each brick goes).

### Brick representation

One line per brick:
```
hxw (x,y,z) #RRGGBB
```

Examples:
```
2x4 (5,3,0) #C91A09
1x2 (3,7,1) #05131D
2x2 (10,10,3) #00852B
```

~11-13 tokens per brick. A 100-brick model is ~1200 tokens.

### Grid

20x20x20 stud grid. Each axis 0-19. z=0 is the ground plane.

### Brick library

8 shapes (all 1-unit tall), 14 including orientation variants:

| Dims | LDraw ID | Description |
|------|----------|-------------|
| 1x1  | 3005     | Brick 1x1   |
| 1x2  | 3004     | Brick 1x2   |
| 2x2  | 3003     | Brick 2x2   |
| 1x4  | 3010     | Brick 1x4   |
| 2x4  | 3001     | Brick 2x4   |
| 1x6  | 3009     | Brick 1x6   |
| 2x6  | 2456     | Brick 2x6   |
| 1x8  | 3008     | Brick 1x8   |

Allowed dimension strings in prompts:
`2x4, 4x2, 2x6, 6x2, 1x2, 2x1, 1x4, 4x1, 1x6, 6x1, 1x8, 8x1, 1x1, 2x2`

### Color palette

~50 common LEGO colors mapped to hex codes, sourced from our existing
color data in `data/cache/colors.json`. Colors are constrained via logit
masking to this fixed palette during generation.

---

## Data Pipeline

### Source data

1. Download original StableText2Brick from HuggingFace
   (`AvaLovelace/StableText2Brick`) — 47,389 structures with 3D coordinates.
2. Our existing ST2B labels in `data/st2b_labels/` — 46,143 files with
   part colors, categories, and metadata.

### Color mapping

Match structures between the two datasets by cross-referencing object
descriptions and brick counts. Assign colors from our labels to the
original coordinate data.

For unmatched structures, use category-based color palettes:
- Vehicles: red, blue, black, white
- Furniture: brown, tan, dark tan
- Nature: green, dark green, brown
- Buildings: gray, white, red, blue

Additional heuristics:
- Ground-touching bricks biased toward darker colors
- Parse caption for explicit color words
- Random variation within palette for diversity

### Training example format

JSONL, one conversation per line:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a LEGO master builder."
    },
    {
      "role": "user",
      "content": "Create a colored LEGO model. Format: <dims> (<x>,<y>,<z>) <#hex>.\nAllowed dims: 2x4, 4x2, 2x6, 6x2, 1x2, 2x1, 1x4, 4x1, 1x6, 6x1, 1x8, 8x1, 1x1, 2x2.\nAll bricks are 1 unit tall.\n\n### Input:\nA red chair with four legs and a tall backrest"
    },
    {
      "role": "assistant",
      "content": "2x2 (8,8,0) #C91A09\n2x2 (12,8,0) #C91A09\n..."
    }
  ]
}
```

### Dataset size

- 5 captions per structure from original dataset: 47k x 5 = ~235k
- Enriched prompts from our ST2B labels (color-aware, complexity-aware,
  category-specific): ~8-10 prompts per matched structure
- Total: ~350-400k training examples

---

## Training

### Model

- Base: Qwen3.5-4B (hybrid DeltaNet + Attention architecture)
- Fine-tuning: LoRA
  - Rank: 32
  - Alpha: 64
  - Dropout: 0.05
  - Target modules: q_proj, v_proj
  - Trainable params: ~5M

### Hyperparameters

- Optimizer: AdamW
- Learning rate: 2e-3
- Scheduler: Cosine with 100 warmup steps
- Batch size: 4 per GPU, gradient accumulation 4 (effective batch 64)
- Epochs: 3
- Max sequence length: 8192
- Precision: bf16

### Structure-aware loss weighting

- Coordinate tokens (x,y,z values): weight 3.0
- Color tokens (#RRGGBB): weight 2.0
- Dimension tokens (hxw): weight 1.5
- Syntax tokens (parens, commas, newlines): weight 0.1

### Hardware

Primary (cloud):
- 4x RTX 4090 on RunPod — $1.36/hr
- Estimated time: 18-24 hours
- Estimated cost: $24-33

Local alternative:
- 2x RTX 5090 — $0/hr (owned hardware)
- Estimated time: 36 hours

Fallback (if 4B underperforms):
- Scale to Qwen3.5-9B with 4-bit LoRA on 4x RTX A6000
- $1.32/hr, ~2-3 days, ~$60

---

## Inference

### Constrained decoding (logit masking)

Each brick is generated token-by-token with only valid tokens allowed:

```
1.  Dimensions or EOS    → {2x4,4x2,2x6,6x2,1x2,2x1,1x4,4x1,1x6,6x1,1x8,8x1,1x1,2x2,<eos>}
2.  Literal " ("
3.  X position           → {0..19}
4.  Literal ","
5.  Y position           → {0..19}
6.  Literal ","
7.  Z position           → {0..19}
8.  Literal ") #"
9.  Color hex            → {C91A09,05131D,00852B,...} (palette ~50 colors)
10. Literal "\n"
```

### Per-brick rejection sampling

After generating each brick:
1. Regex valid? (`\d+x\d+ \(\d+,\d+,\d+\) #[0-9A-F]{6}`)
2. Dimensions in brick library?
3. In bounds? (x+h <= 20, y+w <= 20, z < 20)
4. No collision with existing bricks? (check voxel occupancy)

If invalid: rollback KV cache, resample.
Temperature starts at 0.6, increases by 0.01 per rejection, max 2.0.
Max 500 attempts per brick.

### Physics-aware rollback

After EOS (full structure generated):
1. Build connectivity graph (bricks as nodes, edges for vertical adjacency
   with overlapping footprints)
2. Check all bricks are connected to ground
3. If unstable: find first unstable brick, truncate everything after it
4. Resume generation from stable prefix
5. Max 100 rollback cycles (median in practice: ~2)

### Integration

```
Image → Stage 1 (Qwen3.5-9B) → "A red compact chair with four legs"
      → Stage 2 (Qwen3.5-4B) → "2x2 (8,8,0) #C91A09\n2x2 (12,8,0) #C91A09\n..."
```

Text-only input skips Stage 1 and goes directly to Stage 2.

### Performance

Estimated ~30-60 seconds per model.
Median ~59 bricks per structure, ~98 brick rejections, ~2 rollback cycles.

---

## 3D Viewer Changes

### What gets deleted

- `packLayer()` function in LegoViewer.tsx
- `grid_pos` / `grid_positions` handling
- All packing heuristics and row-based layout code
- `add_grid_pos.py` becomes unnecessary for new pipeline

### What stays

- `BrickMesh.tsx` — box + studs rendering (works as-is)
- OrbitControls, lighting, grid, camera setup
- Step-based rendering (past/current/future opacity)

### New viewer logic

Parse brick string into array, place directly:

```typescript
interface Brick {
  h: number; w: number;
  x: number; y: number; z: number;
  color: string;
}

function parseBricks(raw: string): Brick[] {
  return raw.trim().split('\n').map(line => {
    const m = line.match(/(\d+)x(\d+) \((\d+),(\d+),(\d+)\) #([0-9A-Fa-f]{6})/);
    return { h: +m[1], w: +m[2], x: +m[3], y: +m[4], z: +m[5], color: '#'+m[6] };
  });
}

// Three.js placement (Y-up):
const STUD = 1.0;
const BRICK_H = 1.0;
position = [(brick.x + brick.h/2) * STUD, brick.z * BRICK_H + BRICK_H/2, (brick.y + brick.w/2) * STUD];
size = [brick.h * STUD, BRICK_H, brick.w * STUD];
```

Center the model around origin by computing bounding box midpoint.

---

## API Changes

### New response format

```json
{
  "bricks": "2x4 (5,3,0) #C91A09\n1x2 (3,7,1) #05131D\n...",
  "caption": "A red compact chair with four legs",
  "brick_count": 87,
  "stable": true,
  "metadata": {
    "model_version": "qwen35-4b-brick-v1",
    "generation_time_ms": 42000,
    "rejections": 98,
    "rollbacks": 2
  }
}
```

### Backward compatibility

The existing `/api/generate` endpoint returns both old format (description + steps)
and new format (bricks string) during transition. The frontend switches rendering
based on whether `bricks` field is present.

---

## Output formats

Three outputs per generation (matching BrickGPT):
- `.txt` — brick text format (one `hxw (x,y,z) #color` per line)
- `.ldr` — LDraw format for LEGO CAD software
  - Coordinate transform: x_ldr = (x+h*0.5)*20, z_ldr = (y+w*0.5)*20, y_ldr = -z*24
  - Color mapping from hex to LDraw color codes
- `.png` — rendered visualization (Three.js canvas capture)

---

## Migration plan

### Phase 1: Data preparation (1-2 days)
- Download StableText2Brick from HuggingFace
- Build color mapping between our ST2B labels and original coordinates
- Generate JSONL training dataset with colored brick sequences

### Phase 2: Training (1-2 days)
- Fine-tune Qwen3.5-4B on 4x RTX 4090 (RunPod) or 2x RTX 5090 (local)
- Validate on held-out test set (4.79k structures)

### Phase 3: Inference pipeline (2-3 days)
- Implement constrained decoding with logit masking
- Implement per-brick rejection sampling
- Implement connectivity-based stability checking
- Implement rollback-regeneration loop
- Wire up Stage 1 → Stage 2 pipeline

### Phase 4: Frontend (1 day)
- New brick parser and direct-placement viewer
- Update API types and response handling
- Backward compat with old format during transition

### Phase 5: Validation (1 day)
- Manual inspection of generated models across categories
- Stability rate measurement (target: >95%)
- A/B comparison: old flat-pack viewer vs new coordinate viewer

**Total estimated timeline: 6-9 days**
