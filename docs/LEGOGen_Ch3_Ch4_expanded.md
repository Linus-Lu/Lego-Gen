# Chapter 3 — System Architecture

## 3.1 Overview
The LEGOGen system is structured as a two-stage pipeline augmented with three runtime guardrails to ensure syntactic, spatial, and physical correctness of generated LEGO models.

```
Image ──────────────┐
                     ▼
              ┌──────────────┐   Qwen3.5-9B + LoRA r=32 (4-bit NF4)
              │   Stage 1    │   System prompt → 1–3 sentence caption
              │ image→text   │
              └──────┬───────┘
                     │  caption
Text prompt ─────────┤
                     ▼
              ┌──────────────┐   Qwen3.5-4B + LoRA r=32 (4-bit NF4)
              │   Stage 2    │   Outlines RegexLogitsProcessor decoder
              │ text→bricks  │
              └──────┬───────┘
                     │  brick line (proposed)
           ┌─────────┴─────────┐
           ▼                   ▼
     ┌──────────┐      ┌───────────────┐
     │ VoxelGrid│      │    Stud-LP    │
     │ can_place│      │  is_stable    │
     │ (reject) │      │  (rollback)   │
     └──────────┘      └───────────────┘
                     │
                     ▼
          React 19 + R3F 3D viewer + SQLite gallery
```

**[IMAGE PLACEHOLDER: Figure 3.1 — Full pipeline flow diagram]**

## 3.2 Stage 1 — Image → Caption
Stage 1 is responsible for converting user-uploaded images into short textual descriptions suitable for Stage 2 input.

- Base model: `Qwen3.5-9B` multimodal
- LoRA adapter: rank=32, alpha=64, all-linear target
- Quantization: 4-bit NF4 (double quantized, bf16 compute)
- Input: image preprocessed to 448×448 with augmentations
- Output: 1–3 sentence caption

**Dataset:** COCO 2017 images matched with StableText2Brick captions by category.

**Training:**
- DDP with `torchrun` on 4× RTX 5090
- Epochs: 3, batch size: 8 per GPU
- Learning rate: 5e-5, cosine schedule with warmup
- Gradient checkpointing enabled
- Vision encoder frozen

**[IMAGE PLACEHOLDER: Figure 3.2 — Stage 1 data flow]**

## 3.3 Stage 2 — Caption → Bricks
Stage 2 converts the Stage 1 caption (or a user prompt) into a newline-separated `HxW (x,y,z) #RRGGBB` brick sequence.

- Base model: `Qwen3.5-4B` text-only
- LoRA adapter: rank=32, alpha=64, target modules `q_proj/v_proj`
- Precision: bf16
- Max sequence length: 4096 tokens

**Generation guardrails:**
1. Grammar-constrained decoding via `RegexLogitsProcessor`
2. Voxel occupancy check (`VoxelGrid.can_place`)
3. LP stability check with rollback

**[IMAGE PLACEHOLDER: Figure 3.3 — Stage 2 guardrail loop]**

## 3.4 Best-of-N Sampling
- Generates N candidates for the same prompt
- Filters stable candidates
- Computes 9-D structural feature vector per candidate
- K-means clustering; selects candidate closest to largest cluster centroid

**[IMAGE PLACEHOLDER: Figure 3.4 — Best-of-N sampling workflow]**

## Chapter 4 — Data and Training

## 4.1 Stage 1 Dataset
- Manifest: `data/stage1_manifest.json`
- Each entry: `{image_path, description, category, source}`
- Train/val split: 90/10 deterministic
- Transforms: random crop, horizontal flip, rotation, color jitter

## 4.2 Stage 1 Trainer
- LoRA config: r=32, alpha=64, all-linear, dropout=0.05, DoRA+rsLoRA
- Optimizer: adamw_torch
- LR: 5e-5, batch size: 8 per GPU
- Epochs: 3
- Gradient checkpointing enabled

## 4.3 Stage 2 Dataset
- Input: StableText2Brick pairs `(caption, brick_list)`
- Colorization:
  1. Caption-driven if color word present
  2. Category palette otherwise
  3. Ground-weighted dark colors at z=0
  4. Deterministic seed based on structure_id + caption_index + brick_index
- Output: chat-style JSONL messages with assistant turn as brick sequence

## 4.4 Stage 2 Trainer
- LoRA config: r=32, alpha=64, q_proj/v_proj, dropout=0.05, DoRA+rsLoRA
- Structure-aware loss: boilerplate=0.1, dimension=3.0, content=1.0
- Curriculum ordering: untruncated first, truncated later
- Optimizer: adamw_torch, bf16, gradient checkpointing, cosine LR
- Effective batch: 16
- Epochs: 3

**[IMAGE PLACEHOLDER: Figure 4.1 — Training data and curriculum diagram]**

## 4.5 Tables — Key Hyperparameters

| Stage | Base | LoRA r | α | Target | Precision | Max Seq | Epochs | Batch |
|-------|------|--------|---|--------|-----------|---------|--------|-------|
| 1     | Qwen3.5-9B | 32     | 64 | all-linear | 4-bit NF4 | 512 | 3 | 8/GPU |
| 2     | Qwen3.5-4B | 32     | 64 | q_proj/v_proj | bf16      | 4096 | 3 | 16 effective |

**[IMAGE PLACEHOLDER: Figure 4.2 — Stage 2 structure-aware loss schematic]**