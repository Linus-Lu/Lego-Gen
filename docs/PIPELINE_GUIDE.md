# LEGO-Gen Pipeline Guide — How Training and Inference Actually Work

> **Audience.** You have taken an intro ML course (loss functions, SGD,
> cross-entropy), a DL course (backprop, transformers, attention), and
> an LLM course (decoder-only models, autoregressive sampling, fine-tuning,
> tokenizers). You have **not** necessarily seen LoRA/DoRA/PiSSA,
> 4-bit quantization, grammar-constrained decoding, or linear programming.
> This document teaches every non-obvious technique from first principles
> with pointers into the code.
>
> **Relationship to `TRAINING_AND_INFERENCE.md`.** That file is a
> line-anchored reference: where each thing lives, what values each knob
> takes. This file is a *guide*: why each thing exists, what would break
> without it, and how the pieces fit together.

---

## 0. What is the system trying to do?

A user uploads a photo of a couch (or types "a small red house"). The
system returns a sequence of LEGO brick placements, something like:

```
2x4 (0,0,0) #C91A09
2x4 (2,0,0) #C91A09
2x4 (0,0,1) #FFFFFF
…
```

Each line says: *a brick of dimensions H×W, at grid cell (x, y, z), with
this RGB hex colour*. The sequence must be:

1. **Syntactically valid** — parseable back into bricks.
2. **Non-overlapping** — no two bricks claim the same grid cell.
3. **Physically stable** — the structure does not fall over under
   gravity.
4. **Semantically relevant** — it resembles the input image or caption.

A vanilla LLM fine-tuned on brick sequences will get (4) approximately
right and (1)–(3) frequently wrong. The whole design of this project is
about getting (1)–(3) to be *guaranteed*, not merely "usually", while
(4) remains the LLM's job.

---

## 1. The big picture: two stages plus runtime guardrails

```
         ┌────────────────────┐     ┌──────────────────────┐
  image ─┤  Stage 1 (Qwen-9B) ├─── caption ─┤  Stage 2 (Qwen-4B) ├── bricks
         │   image → caption  │             │  caption → bricks  │
         └────────────────────┘             └────────────────────┘
                                                      │
                                                      ▼
                                     ┌────────────────────────────────┐
                                     │  Runtime guardrails:            │
                                     │    1. Grammar-constrained       │
                                     │       decoding  (outlines)      │
                                     │    2. Voxel rejection           │
                                     │    3. LP physics rollback       │
                                     └────────────────────────────────┘
```

Two models, two roles:

| Stage | Model | Input | Output | Why |
|-------|-------|-------|--------|-----|
| 1 | `Qwen/Qwen3.5-9B` (multimodal) | RGB image | 1-3 sentence caption | Vision-language understanding |
| 2 | `Qwen/Qwen3.5-4B` (text-only) | Caption | Brick lines | Spatial/geometric reasoning |

If the user types a prompt directly, we skip Stage 1 entirely. That is
the text-only path. The image path is `image → Stage 1 → caption →
Stage 2 → bricks`.

**Why split into two stages?** Two reasons.

1. **Capacity routing.** Vision-language understanding benefits from a
   big multimodal model (9B). The "translate a caption into a brick
   grid" task is narrower and fits comfortably in a smaller 4B text-only
   model. Running one 13B multimodal end-to-end would cost more VRAM for
   worse specialization.
2. **Independent fine-tuning.** Each stage has its own LoRA adapter
   trained on its own dataset. We can retrain one without touching the
   other.

Runtime VRAM on a 24 GB GPU: ~4 GB for Stage 1 (9B @ 4-bit NF4) + ~2 GB
for Stage 2 (4B @ 4-bit NF4) + overheads ≈ 10 GB.

---

## 2. Why we can't just fine-tune the full 9B model

A 9B-parameter model is 18 GB in bf16. The Adam optimizer keeps two
extra tensors per parameter (momentum and variance), so a full fine-tune
needs ~54 GB just for optimizer state, plus the model weights, plus
activations. That is way beyond a single 24 GB GPU.

Two techniques combine to make fine-tuning fit: **4-bit NF4
quantization** and **LoRA**. You only need to understand each one to
understand the system.

### 2.1 4-bit NF4 quantization

**The problem.** Storing 9B parameters in 16-bit bf16 is 18 GB. We want
to fit them in less than 6 GB.

**The idea.** Most neural-network weights are approximately normally
distributed. If you only have 4 bits per number (16 possible values),
you want those 16 levels spaced such that frequently-occurring weight
magnitudes are well-represented. NF4 ("NormalFloat 4") picks the 16
quantiles of a standard normal distribution as its levels. That is
better than uniform spacing for weight tensors that really are normal-ish.

**Implementation.** We use the `bitsandbytes` library:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

- `quant_type="nf4"` — the quantile-based levels described above.
- `compute_dtype=bf16` — the weights are stored in 4-bit but dequantized
  back to bf16 *on the fly* during matrix multiplications. You train in
  bf16 math even though the storage is 4-bit.
- `use_double_quant=True` — the per-group scales themselves are also
  quantized. Saves another ~0.4 bits per parameter on average.

**Cost.** Some quality loss — typically < 1 pp on academic benchmarks
for most models. Acceptable here because the LoRA adapter (next
section) can absorb small base-model imperfections.

**Where in the code.** Both pipelines use the same pattern:
`backend/inference/stage1_pipeline.py:50` and
`backend/inference/brick_pipeline.py:72`.

### 2.2 LoRA: Low-Rank Adaptation

**The intuition.** Fine-tuning updates a weight matrix `W` by an update
`ΔW`. LoRA's core observation: for most fine-tuning tasks, `ΔW` is
*low-rank*. You can decompose it as `ΔW = A × B` where `A` is
`(d_in, r)` and `B` is `(r, d_out)`, with `r` much smaller than either
`d_in` or `d_out`. Instead of training all `d_in × d_out` parameters,
you only train `r × (d_in + d_out)`.

**Concretely.** Pick `r=32`. For a 4096×4096 matrix, full fine-tuning
updates 16.7M parameters. LoRA updates `32 × (4096 + 4096) = 262K`
parameters — 64× fewer. And the base `W` stays frozen and quantized —
it never gets a gradient.

**Forward pass.**

```
y = W x                # the original base matrix
y += (α / r) · B(A x)  # the LoRA adapter, α is a scaling hyperparameter
```

`α` controls how strongly the adapter influences the output relative to
the base. We use `α = 2r = 64`.

**Why it works for us.** Fine-tuning a 9B multimodal model to produce
short descriptive captions is a narrow specialization. The *information*
we are adding — "this kind of photo should yield this kind of
caption" — really is low-rank. LoRA captures it cleanly.

**Where in the code.**

- Stage 1: `backend/training/train_stage1.py:200`
- Stage 2: `backend/training/train_brick.py:189`

Both use `peft.LoraConfig`. Stage 1 targets `"all-linear"` (every
linear layer in the model). Stage 2 targets only `["q_proj", "v_proj"]`
— the Q and V projections inside each attention block. The Stage 2 task
is narrower, so the narrower target is enough.

### 2.3 Three tiny but important PEFT upgrades

Vanilla LoRA has three known weaknesses. Each has a one-flag fix in
modern `peft`. We use all three where possible.

#### DoRA — Weight-Decomposed LoRA (ICML 2024)

A weight matrix has two kinds of information: **magnitude** (how big is
each column?) and **direction** (which way does each column point?).
Vanilla LoRA updates both through a single low-rank perturbation. DoRA
decomposes `W` as `m × (W / ||W||)` — a per-column magnitude vector
times a normalized direction matrix — and fine-tunes them separately.
The magnitude update is a small 1-D vector; the direction update is
LoRA. Result: ~1–2 pp accuracy gains for free.

Enable with `use_dora=True` on `LoraConfig`. We use it on both stages.

#### rsLoRA — Rank-Stabilized LoRA (NeurIPS 2023)

Vanilla LoRA scales the adapter by `α / r`. At higher ranks (say
`r=128`), this makes the adapter's *effective* contribution too small
compared to its parameter count. rsLoRA uses `α / √r` instead, which is
the scaling that keeps the variance of adapter outputs roughly constant
as you grow `r`.

For us (r=32) the difference is modest, but it costs nothing and makes
the model robust if we ever want to try `r=64` or `r=128`.

Enable with `use_rslora=True`.

#### PiSSA — Principal Singular values and Singular vectors Adaptation (NeurIPS 2024)

Vanilla LoRA initializes `B=0` and `A` to small random values, so the
adapter starts as a no-op. PiSSA instead does an SVD of the base weight
`W` and uses the **top `r` singular directions** to initialize `A` and
`B`. So the adapter starts out representing the most important
directions of the base matrix. Empirically this converges faster and
reaches better final quality (+3–5 pp on small math benchmarks).

**Catch.** PiSSA needs `W` in full precision to SVD. If you quantize
`W` to NF4 *before* running PiSSA, the SVD is noisy and the
initialization is wrong. In this repo:

- **Stage 2** (text → bricks) keeps the base in **bf16**
  (`train_brick.py`) so PiSSA remains *possible in principle*, but the
  shipped trainer does **not** enable it because PEFT warns on
  `DoRA + PiSSA` and the combination was unstable in practice.
- **Stage 1** (image → caption) loads the base in **4-bit NF4**
  (`train_stage1.py:166`) because 9B bf16 would not fit for training.
  PiSSA is intentionally **not** enabled here; only DoRA + rsLoRA
  (`train_stage1.py:200-209`).

This asymmetry is documented in both trainers and in
`TRAINING_AND_INFERENCE.md` §3.

---

## 3. Stage 1 training: image → caption

**File:** `backend/training/train_stage1.py`.

**Goal.** Given an image, output a 1-3-sentence description in LEGO
vocabulary — "a green bench with three wooden planks and a tall backrest"
— so Stage 2 has a concrete target.

**Base model.** `Qwen/Qwen3.5-9B`, the multimodal variant. The vision
encoder sees the image; the language model reads a chat template and
autoregressively emits a description.

### 3.1 The dataset problem

We need paired `(image, description)` samples. Two existing datasets
get close:

- **COCO 2017** — millions of labeled photographs with category
  annotations (chair, couch, car, …) but no LEGO-relevant captions.
- **StableText2Brick** (HuggingFace `AvaLovelace/StableText2Brick`) —
  for ~22k LEGO structures, human-written captions in LEGO vocabulary,
  but no photographs.

`backend/data_pipeline/build_stage1_dataset.py` **matches them**: for
each COCO image whose category appears in `COCO_TO_ST2B_CATEGORY`
(`config.py:39`), it picks a random ST2B caption from the corresponding
category. The output manifest is `data/stage1_manifest.json`:

```json
{
  "image_path": "data/coco/train2017/000000123456.jpg",
  "description": "a green bench with slatted backrest",
  "category": "bench",
  "source": "coco"
}
```

This is imperfect supervision — the caption was written about a
*different* bench than the photo shows — but it teaches the model the
right *style and vocabulary*. A random matching is fine because we are
training a distribution mapping, not a pixel-perfect captioner.

### 3.2 The loss: mask the prompt

A standard fine-tuning gotcha. The training input to a chat model looks
like:

```
<|im_start|>system
You are a LEGO design assistant. Describe this object's shape…
<|im_end|>
<|im_start|>user
<image>Describe this object for LEGO building.<|im_end|>
<|im_start|>assistant
a green bench with slatted backrest<|im_end|>
```

Cross-entropy loss on this whole sequence would also penalize the model
for not predicting "*system*" or "*assistant*". We don't want that —
we want the loss only on the assistant's *answer*.

`Stage1Dataset` handles this by setting `labels[i] = -100` for every
token up to and including the final `<|im_start|>assistant\n` marker.
PyTorch's `CrossEntropyLoss` ignores `-100` targets by convention. So
loss only flows on the description tokens. See
`backend/data_pipeline/dataset_stage1.py` for the exact masking code.

### 3.3 Trainer mechanics

Hyperparameters that matter (all in `backend/config.py`):

- `STAGE1_LORA_R = 32` — adapter rank.
- `STAGE1_LORA_ALPHA = 64` — adapter scale `α = 2r`.
- `STAGE1_LEARNING_RATE = 5e-5` — standard LoRA LR on large models.
- `STAGE1_BATCH_SIZE = 8` per GPU. With 4 GPUs and
  `gradient_accumulation=1`, effective batch is 32.
- `STAGE1_MAX_SEQ_LENGTH = 512` — captions are short; this is plenty.
- `STAGE1_NUM_EPOCHS = 3`.

Other knobs set in `train_stage1.py:302-334`:

- `lr_scheduler_type="cosine"` with warmup = `max(1, total_steps // 10)`,
  capped at 100 steps.
- `gradient_checkpointing=True` — trades compute for memory by
  re-running forward activations during backward. Essential for
  training a 9B model on 24 GB cards.
- `load_best_model_at_end=True` — HF Trainer evaluates periodically and
  keeps the best checkpoint by `eval_loss`.
- Vision encoder is frozen (`train_stage1.py:213-215`): we only
  fine-tune the language model layers. The vision encoder already
  knows what "chair" looks like; we just need the language model to
  learn the output style.

### 3.4 Multi-GPU

The trainer is DDP-ready (`torchrun --nproc_per_node=4 -m
backend.training.train_stage1`). Rank-zero prints status, downloads
from HuggingFace, writes checkpoints. Other ranks wait at barriers. If
`huggingface.co` is unreachable (AutoDL in China), the script
auto-falls-back to `https://hf-mirror.com`
(`train_stage1.py:32-39`).

---

## 4. Stage 2 training: caption → bricks

**File:** `backend/training/train_brick.py`.

**Goal.** Given a caption, emit a brick sequence that parses, fits in a
20×20×20 grid, and is physically stable.

**Base model.** `Qwen/Qwen3.5-4B`, text-only. Loaded in bf16 (not
4-bit — the trainer keeps the base full precision and leaves PiSSA off).

### 4.1 The dataset: StableText2Brick + deterministic colorization

ST2B gives us `(caption, brick_list)` pairs. Each brick is `H x W (x,
y, z)` — no colour. LEGO-Gen wants coloured bricks.

`backend/data_pipeline/prepare_brick_dataset.py` solves this with a
**deterministic colour map**:

1. **Caption-driven.** If the caption contains a colour word (`red`,
   `blue`, etc.), all bricks get that hex code
   (`prepare_brick_dataset.py:116`).
2. **Category-driven.** Otherwise pick from the category's palette
   (`CATEGORY_PALETTES` at `prepare_brick_dataset.py:60`). E.g., cars
   draw from `{"C91A09", "0055BF", "05131D", "FFFFFF", "A0A5A9"}`.
3. **Ground-weighted dark colours.** For `z=0` bricks (the foundation),
   dark palette entries get double weight. This teaches the model that
   bases tend to be dark (foundations look like ground, not sky).
4. **Deterministic seeding.** The seed is `md5(structure_id,
   caption_index)[:4]`, so the same structure always gets the same
   colourization across training runs.

Each row of the JSONL is a chat-style message list:

```json
{"messages": [
    {"role": "system",    "content": "You are a LEGO master builder."},
    {"role": "user",      "content": "Create a colored LEGO model...{caption}"},
    {"role": "assistant", "content": "2x4 (0,0,0) #C91A09\n2x4 (2,0,0) #C91A09\n..."}
]}
```

At training time, `trl.SFTTrainer` tokenizes this chat template and
learns to emit the assistant message. Same prompt-masking idea as Stage 1.

### 4.2 Structure-aware loss weighting

Here is the subtle trick. A brick line like `"2x4 (0,0,0) #C91A09"`
contains three different *kinds* of token:

- **Boilerplate** — the parentheses, commas, spaces, and the letter `x`
  that appears in every line. If the model fails on these, the output
  doesn't parse — but they are easy to learn; they are always there.
- **Structural** — the dimension combo `2x4`. There are only 14 legal
  choices (see `BRICK_SHAPES` in `backend/brick/constants.py:14`).
  Getting this wrong means emitting a nonexistent brick.
- **Content** — the actual integers (coordinates) and the hex colour.
  These are the hard part — spatial reasoning, color choice.

Vanilla cross-entropy weights every token equally. That's wrong:
boilerplate is easy and uninformative, structural tokens are
constrained (only 14 possibilities), and content tokens are where the
real learning happens.

`BrickStructureWeights` (`train_brick.py:44`) assigns:

| Token class | Weight |
|-------------|--------|
| Boilerplate (`(`, `)`, `,`, `x`, `#`, spaces, `\n`) | **0.1** |
| Dimension combos (`2x4`, `4x2`, …, `2x2`) | **3.0** |
| Everything else (coordinate digits, hex digits) | 1.0 |
| Masked prompt tokens (`-100`) | 0.0 |

The 30× ratio between dimension tokens and boilerplate tokens is
aggressive on purpose: it nudges the model to really get the dimension
combos right, since those have a closed vocabulary and any mistake is
unrecoverable.

Implementation: a subclass of `SFTTrainer` overrides `compute_loss`
(`train_brick.py:263`). It runs one vectorized
`CrossEntropyLoss(reduction="none")`, multiplies by per-token weights,
then takes a weighted mean. The whole custom loss is ~10 lines.

### 4.3 Curriculum ordering

A naive trainer sees batches in random order. Many structures in ST2B
are long enough that tokenized they exceed `max_seq_length=4096` and
get truncated. Truncated samples are *silent failures*: the model sees
an incomplete output and learns from a wrong label.

`apply_curriculum_ordering` (`train_brick.py:99`) estimates each
sample's tokenized length (cheap heuristic: `len(content)/3.0 + 80`)
and reorders the dataset to put all untruncated samples first, then
the truncated ones. During the first ~`n_untruncated/batch_size` steps
the model only sees clean examples. This is a mild form of curriculum
learning — learn the pattern on easy cases first, then harder ones.

### 4.4 Final LoRA config

```python
LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
    use_dora=True,
    use_rslora=True,
)
```

Two modern PEFT upgrades are on. PiSSA is intentionally left off even
with a bf16 base because PEFT warns on `DoRA + PiSSA`, and the current
trainer follows the more stable `DoRA + rsLoRA` combination.
`target_modules=["q_proj", "v_proj"]` is the narrowest useful target
set — it is the classical LoRA recipe from the original paper. It works
here because the task is narrow.

---

## 5. Inference: the three-mechanism guardrail

Fine-tuning *improves* Stage 2's output, but any LLM that sometimes
hallucinates "3x9" (not a legal brick) or places bricks in midair is
unacceptable. We fix this at generation time with **three stacked
guardrails** in `backend/inference/brick_pipeline.py`.

```
 LLM proposes token           Grammar rules out invalid tokens
       │                                │
       ▼                                ▼
 ┌──────────────────┐          ┌─────────────────────────┐
 │ outlines logits  │ ◄────── │ RegexLogitsProcessor    │
 │  processor       │          │ (forces valid bricks)   │
 └──────────────────┘          └─────────────────────────┘
       │
       ▼
 Brick parsed — check VoxelGrid.can_place — else resample
       │
       ▼ placed
 Check is_stable (LP) — else rollback to first unstable prefix
       │
       ▼ stable
 Continue with next brick
```

### 5.1 Mechanism 1 — Grammar-constrained decoding

**Problem.** An LLM samples the next token from a softmax over all
~150,000 possible tokens. Many of those tokens, in many positions,
would violate the brick grammar. E.g., after `"2x4 ("` the next token
*must* be a digit in 0-19. Without intervention the model could emit
"the" or a newline and we'd have to throw the line away.

**Classical fix.** Rejection sampling: generate the whole line, try to
parse it with a regex, retry if it fails. Works but is slow and the
parse-failure rate compounds across hundreds of bricks.

**Modern fix (what we use).** `outlines` provides a
`RegexLogitsProcessor`: it compiles a regex to an automaton and, at
every generation step, masks the logits of any token that would take
the automaton to a dead state. The model literally *cannot* emit a
token that would break the grammar. Parse failures become mathematically
impossible.

**Our regex** (built in `brick_pipeline.py:22`):

```
(2x4|4x2|2x6|6x2|1x2|2x1|1x4|4x1|1x6|6x1|1x8|8x1|1x1|2x2) \(\d{1,2},\d{1,2},\d{1,2}\) #[0-9A-Fa-f]{6}\n
```

It encodes all of:

- Only the 14 allowed dimension combos (alternation).
- Coords are 1-2 digits (our grid is 20×20×20 so always < 100).
- Colour is exactly 6 hex chars.
- Lines end in `\n`.

**Dev-mode note.** If `outlines` is absent, `_build_logits_processor`
returns `None` and generation falls back to unconstrained sampling
plus regex rejection. This keeps tests runnable without the package.
See `brick_pipeline.py:43`.

**Effect on temperature.** Under grammar-constrained decoding, every
generated line parses. The old rejection-driven temperature ramp
(start low, bump up on parse failures) is gone. `BASE_TEMPERATURE=0.6`
is fixed (`brick_pipeline.py:33`). Voxel collisions (the remaining
rejection cause below) don't benefit from higher temperature because
they are structural, not sampling-noise.

### 5.2 Mechanism 2 — Voxel rejection

Grammar ensures the line parses. It does **not** ensure the brick can
physically go there.

`backend/brick/occupancy.py` maintains a `VoxelGrid` — a 20×20×20
boolean array where `grid[x, y, z]` is True iff that unit cube is
already occupied.

`VoxelGrid.can_place(brick)` checks five things:

1. `(h, w)` is a legal shape (in `BRICK_SHAPES`). [grammar also enforces this but defence-in-depth]
2. Coordinates are non-negative.
3. The brick fits in `[0, WORLD_DIM) ^ 3` (no overflow).
4. No voxel in the brick's footprint is already occupied.

If `can_place` returns False, `_generate_one_brick`
(`brick_pipeline.py:180`) simply resamples the next brick line with the
same temperature. It tries up to `MAX_REJECTIONS=500` times.

**Why resampling works.** The grammar makes sure the resampled line is
*syntactically* valid. Stochastic sampling gives the model another shot
at picking a different dimension or location that fits. For a model that
has seen thousands of ST2B training examples, most of its probability
mass is on physically reasonable placements.

### 5.3 Mechanism 3 — Physics rollback via linear programming

Grammar + voxel gives us *syntactically valid, non-overlapping*
structures. Those can still collapse. A tall tower where each brick is
offset 1 stud from the one below it might place legally but fall over
under gravity.

We use a **linear program** (LP) to check static equilibrium. If
gravity-balance cannot be satisfied, we roll back.

#### The physics

Every brick is 1 unit tall. That means contact between two bricks is a
horizontal plane. For each *stud-cell* where bricks `a` (lower) and
`b` (upper) overlap in XY, there is one unknown — the vertical
force `f_{a,b,stud}` that `a` exerts upward on `b`. By Newton's third
law, `b` exerts `-f_{a,b,stud}` downward on `a`.

For each brick `b` we write three equations:

1. **Vertical force balance**: sum of all forces on `b` = weight of `b`.
2. **Moment-x balance**: torque about `b`'s COM in the x-axis = 0.
3. **Moment-y balance**: torque about `b`'s COM in the y-axis = 0.

Per-stud contact forces have bounds:

- Stud-stud contacts: in `[-STUD_STRENGTH, +∞)`. Negative = tension
  (pulling apart — LEGO studs can resist a bit of tension via friction,
  but not much, so we bound it).
- Ground contacts: in `[0, +∞)`. Compression-only — the ground can
  push up, but cannot pull down.

Set `STUD_STRENGTH = 1.0` (arbitrary units) and
`BRICK_DENSITY = 1.0` (mass = footprint area). See
`backend/brick/stability.py:26-28`.

This is a **linear feasibility problem**: find any assignment of forces
satisfying `3n` equality constraints and `|num_studs|` bound
constraints. We use `scipy.optimize.linprog(method="highs")` with
`c = 0` (pure feasibility, no objective).

If the LP is feasible → the structure is stable. If infeasible → no
force assignment balances gravity, meaning some brick gets pushed out
of equilibrium. The structure would fall.

**Why LP and not Newton's method?** Balance equations are linear in the
forces, and the bounds are box constraints. LP solves this in
milliseconds with HiGHS. Non-linear methods are overkill.

**Why check after every brick?** Because of monotonicity: adding weight
on top can only *tighten* the lower bricks' equilibrium, never loosen
it. So if the current sequence is stable, we only need to re-check
after the *next* placement, not recheck the whole prefix. That is what
the rollback loop in `brick_pipeline.py:121-157` does.

#### The rollback loop

```python
for _ in range(MAX_ROLLBACKS):   # up to 100 times
    while len(bricks) < MAX_BRICKS:
        brick = _generate_one_brick(...)   # grammar + voxel
        if brick is None:
            break
        bricks.append(brick); grid.place(brick)

    if is_stable(bricks):
        break                               # stable — return

    idx = find_first_unstable(bricks)       # linear scan
    bricks = bricks[:idx]                   # truncate
    grid.clear(); [grid.place(b) for b in bricks]   # rebuild grid
    # go back to the while loop and try again from idx
```

`find_first_unstable` (`stability.py:195`) is a **linear scan over
prefixes**. The scan is deliberate: instability is *not* monotone
because a later counterweight brick can stabilize an earlier unstable
prefix, so binary search would be unsound.

Up to 100 full rollbacks are allowed before the generator gives up and
returns whatever it has. In practice, with a well-trained adapter and
grammar-constrained decoding, fewer than 3 rollbacks per sample is
typical.

### 5.4 Why all three mechanisms?

Each one catches a different failure mode, and each is cheaper than the
next:

| Mechanism | Catches | Cost |
|-----------|---------|------|
| Grammar | Syntactic nonsense | ~0 (logit mask during decode) |
| Voxel | Overlaps, out-of-bounds | O(brick footprint) bool check |
| LP | Gravity failure | O(num_studs) LP solve (~ms) |

Running only the expensive one would work but is slow. Running only
the cheap one misses the hard failures. Stacking them gives
**mathematical correctness guarantees** for the first two and
**physical correctness guarantees** for the third, at the lowest
combined cost.

---

## 6. Best-of-N sampling (coming with PR #11)

**Not yet on `main`; lives on `feature/best-of-n`.**

An obvious question: what if we run Stage 2 *multiple times* and pick
the best result? For tasks where the oracle (stability checker) is
cheap, this is a well-studied technique called **Best-of-N sampling**
or **inference-time scaling**.

**Core loop.** Run `generate()` `n` times in parallel, get `n`
candidates. Filter to stable ones. Rank by `brick_count` (more = richer
structure). Return the top.

**Cluster-and-vote variant.** For higher `n`, simple ranking can pick
outliers. Instead:

1. Compute a 9-dim **structural feature vector** per candidate (brick
   count, XYZ extents, centre of mass, mean footprint area, unique
   colour count — see `best_of_n.py:27`).
2. Z-normalize across candidates.
3. Run KMeans with `k=2-3` clusters.
4. Return the candidate **closest to the centroid of the largest
   cluster**.

This returns the "consensus" structure — the one closest to the mode
of the candidate distribution — instead of an outlier. No renderer is
needed; the 9-dim descriptor is pure math on the coordinates.

**Why the plan chose this.** Other Best-of-N schemes cluster by CLIP
image embeddings, which would require rendering each candidate to a
PNG. We have no backend renderer. The structural descriptor is a
rendering-free alternative that captures what "looks similar" means for
brick structures.

**Where it lands in the API.** `/api/generate-bricks` accepts an
optional `n: int = 1`. When `n > 1`, the route calls
`pipeline.generate_best_of_n(prompt, n=n)` and stamps
`metadata.{n, picked_index, stable_rate}` on the response. Default
`n=1` path is byte-identical to today's behaviour.

**See also.** `docs/superpowers/plans/2026-04-17-roadmap-items-4-to-9.md`
is the implementation plan; PR #11 is the implementation.

---

## 7. End-to-end: following one request through the system

```
POST /api/generate-stream  (multipart: image=<photo.png>)
└── backend/api/routes_generate.py : generate_stream
    ├── validate content-type, decode to PIL.Image
    ├── open an SSE response
    ├── emit: progress {stage:"stage1"}
    └── run on a thread pool:
        └── backend/inference/brick_pipeline.py : generate_from_image
            ├── _get_stage1_pipeline().describe(image)
            │   └── Qwen3.5-9B forward pass (256 tokens)
            │       → strip <think> blocks → caption
            ├── emit (via callback): caption event
            └── self.generate(caption, on_progress)
                ├── loop up to MAX_ROLLBACKS=100 times:
                │   └── loop up to MAX_BRICKS=500 times:
                │       └── _generate_one_brick:
                │           ├── Qwen3.5-4B sample 1 brick line
                │           │   (under outlines RegexLogitsProcessor)
                │           ├── parse → VoxelGrid.can_place?
                │           ├── YES: place; emit brick event
                │           └── NO: retry (up to MAX_REJECTIONS=500)
                │   └── is_stable(bricks)?
                │       ├── YES: return
                │       └── NO: find_first_unstable binary-searches prefix,
                │           truncate, rebuild VoxelGrid, emit rollback event,
                │           continue outer loop
                └── return {bricks, caption, brick_count, stable, metadata}
    └── emit: result <full BrickResponse>
```

The whole thing runs inside `asyncio.wait_for(…, 120s)` so a stuck GPU
can't hold the web worker forever.

---

## 8. Where to look next

- **Reference for config values, file paths, exact function signatures:**
  `docs/TRAINING_AND_INFERENCE.md` — line-anchored.
- **Runtime physics primitives:** `backend/brick/stability.py` and
  `backend/brick/occupancy.py`. Short files, worth reading end-to-end.
- **Grammar regex:** `backend/inference/brick_pipeline.py:22-25`.
- **Dataset prep:** `backend/data_pipeline/prepare_brick_dataset.py`
  (ST2B → brick JSONL) and `build_stage1_dataset.py` (COCO + ST2B →
  Stage 1 manifest).
- **Best-of-N plan:** `docs/superpowers/plans/2026-04-17-roadmap-items-4-to-9.md`
  (proposes RAFT, RAG, PRM, MCTS, ORPO as follow-ups; each item
  includes a feasibility read and effort estimate).
- **Frontend:** React 19 + Vite + R3F. The SSE consumer lives in
  `frontend/src/api/legogen.ts`; the 3D viewer in
  `frontend/src/components/BrickCoordViewer.tsx`.

---

## 9. Summary

What we built is a **two-stage LLM pipeline with runtime guardrails** that
produces buildable LEGO from text or images.

**Training** is parameter-efficient (LoRA r=32), uses the shipped
`DoRA + rsLoRA` PEFT stack, and employs a structure-aware loss that
weights tokens by their structural importance. PiSSA remains discussed
in the guide as an explored variant, but it is not enabled in the
current trainers.

**Inference** is three-layered: grammar-constrained decoding makes
syntax errors impossible, voxel rejection prevents overlaps, and an LP
physics solver rolls back unstable prefixes. Together they turn a
probabilistic LLM into a deterministic-enough generator of physically
valid LEGO structures.

Every technique on this list exists because *something specific broke
without it*. Quantization to fit memory. LoRA to fit optimizer state.
Prompt masking so loss only penalizes the answer. Structure-aware
weighting so the model doesn't waste capacity on boilerplate. Grammar
constraints so we don't waste inference compute on parse failures.
Voxel checks because grammars don't know about collisions. LP rollback
because autoregression doesn't know about gravity.

If you are extending the system, the rule of thumb is: **add new
guardrails at inference, not new losses at training**. Training is
expensive and hard to iterate. Inference-time correctness mechanisms
compose cleanly and never regress past-trained capability.
