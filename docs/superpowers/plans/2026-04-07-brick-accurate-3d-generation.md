# Brick-Accurate 3D Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace flat-pack layout with autoregressive per-brick coordinate generation so LEGO models look like the described object.

**Architecture:** Two-stage pipeline — existing Qwen3.5-9B (image→text, already trained), new Qwen3.5-4B with LoRA (text→colored brick sequence on 20x20x20 grid). Constrained decoding with logit masking ensures valid syntax. Connectivity-based rollback ensures stability.

**Tech Stack:** Python, PyTorch, HuggingFace Transformers/TRL/PEFT, React, Three.js, datasets library

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `backend/brick/__init__.py` | Package init |
| `backend/brick/constants.py` | Brick library (8 shapes), color palette, grid dims (20x20x20) |
| `backend/brick/parser.py` | Parse/serialize `hxw (x,y,z) #RRGGBB` format |
| `backend/brick/occupancy.py` | Voxel occupancy grid, collision detection, bounds checking |
| `backend/brick/stability.py` | Connectivity-based stability analysis, find-first-unstable |
| `backend/brick/decoder.py` | Constrained decoding state machine for logit masking |
| `backend/data_pipeline/prepare_brick_dataset.py` | Download ST2B from HF, assign colors, emit JSONL |
| `backend/training/train_brick.py` | Fine-tune Qwen3.5-4B with LoRA on brick sequences |
| `backend/inference/brick_pipeline.py` | Stage 2 inference with rejection sampling + rollback |
| `frontend/src/components/BrickCoordViewer.tsx` | 3D viewer rendering from exact brick coordinates |
| `tests/test_brick_parser.py` | Parser tests |
| `tests/test_brick_occupancy.py` | Occupancy grid tests |
| `tests/test_brick_stability.py` | Stability checker tests |
| `tests/test_brick_decoder.py` | Constrained decoding tests |
| `tests/test_prepare_brick_dataset.py` | Data pipeline tests |

### Modified files

| File | Change |
|------|--------|
| `backend/config.py` | Add BRICK_* config constants for Stage 2 model |
| `backend/inference/pipeline.py` | Add `generate_brick_build` methods to `UnifiedPipeline` |
| `backend/api/routes_generate.py` | Add `/api/generate-bricks` endpoint |
| `frontend/src/api/legogen.ts` | Add `BrickCoord`, `BrickResponse` types, `parseBrickString`, `generateBricks` API |
| `frontend/src/components/LegoViewer.tsx` | Accept optional `brickString` prop, delegate to `BrickCoordViewer` |
| `frontend/src/components/GuidanceViewer.tsx` | Same optional `brickString` support |

---

## Task 1: Brick constants and parser

**Files:**
- Create: `backend/brick/__init__.py`
- Create: `backend/brick/constants.py`
- Create: `backend/brick/parser.py`
- Test: `tests/test_brick_parser.py`

- [ ] **Step 1: Write failing tests for brick parser**

```python
# tests/test_brick_parser.py
import pytest
from backend.brick.parser import parse_brick, serialize_brick, parse_brick_sequence, Brick


def test_parse_single_brick():
    b = parse_brick("2x4 (5,3,0) #C91A09")
    assert b == Brick(h=2, w=4, x=5, y=3, z=0, color="C91A09")


def test_parse_brick_no_hash():
    b = parse_brick("1x1 (0,0,0) #05131D")
    assert b.color == "05131D"


def test_parse_invalid_format():
    with pytest.raises(ValueError, match="Invalid brick"):
        parse_brick("garbage")


def test_serialize_brick():
    b = Brick(h=2, w=4, x=5, y=3, z=0, color="C91A09")
    assert serialize_brick(b) == "2x4 (5,3,0) #C91A09"


def test_roundtrip():
    line = "1x6 (10,12,3) #237841"
    assert serialize_brick(parse_brick(line)) == line


def test_parse_sequence():
    raw = "2x4 (5,3,0) #C91A09\n1x2 (3,7,1) #05131D\n"
    bricks = parse_brick_sequence(raw)
    assert len(bricks) == 2
    assert bricks[0].h == 2
    assert bricks[1].color == "05131D"


def test_parse_sequence_empty():
    assert parse_brick_sequence("") == []
    assert parse_brick_sequence("  \n  ") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /workspace/Lego-Gen && python -m pytest tests/test_brick_parser.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.brick'`

- [ ] **Step 3: Implement constants**

```python
# backend/brick/__init__.py
```

```python
# backend/brick/constants.py
"""Brick library, color palette, and grid constants."""

import json
from pathlib import Path

WORLD_DIM = 20

BRICK_SHAPES: set[tuple[int, int]] = {
    (1, 1), (1, 2), (2, 1), (2, 2),
    (1, 4), (4, 1), (2, 4), (4, 2),
    (1, 6), (6, 1), (2, 6), (6, 2),
    (1, 8), (8, 1),
}

ALLOWED_DIMS: list[str] = sorted(
    [f"{h}x{w}" for h, w in BRICK_SHAPES],
    key=lambda s: (int(s.split("x")[0]), int(s.split("x")[1])),
)

LDRAW_IDS: dict[tuple[int, int], str] = {
    (1, 1): "3005", (1, 2): "3004", (2, 2): "3003",
    (1, 4): "3010", (2, 4): "3001", (1, 6): "3009",
    (2, 6): "2456", (1, 8): "3008",
}


def _load_color_palette() -> dict[str, str]:
    colors_path = Path(__file__).resolve().parent.parent.parent / "data" / "cache" / "colors.json"
    if not colors_path.exists():
        return {}
    with colors_path.open() as f:
        raw = json.load(f)
    palette: dict[str, str] = {}
    for entry in raw.values():
        if entry.get("is_trans"):
            continue
        name = entry.get("name", "")
        rgb = entry.get("rgb", "")
        if rgb and name and name != "[Unknown]":
            palette[rgb] = name
    return palette


COLOR_PALETTE: dict[str, str] = _load_color_palette()
ALLOWED_COLORS: list[str] = sorted(COLOR_PALETTE.keys())
```

- [ ] **Step 4: Implement parser**

```python
# backend/brick/parser.py
"""Parse and serialize the brick text format: hxw (x,y,z) #RRGGBB"""

import re
from dataclasses import dataclass

_BRICK_RE = re.compile(r"(\d+)x(\d+) \((\d+),(\d+),(\d+)\) #([0-9A-Fa-f]{6})")


@dataclass(frozen=True, slots=True)
class Brick:
    h: int
    w: int
    x: int
    y: int
    z: int
    color: str  # 6-char hex, no #


def parse_brick(line: str) -> Brick:
    m = _BRICK_RE.fullmatch(line.strip())
    if not m:
        raise ValueError(f"Invalid brick format: {line!r}")
    return Brick(
        h=int(m.group(1)), w=int(m.group(2)),
        x=int(m.group(3)), y=int(m.group(4)), z=int(m.group(5)),
        color=m.group(6).upper(),
    )


def serialize_brick(b: Brick) -> str:
    return f"{b.h}x{b.w} ({b.x},{b.y},{b.z}) #{b.color}"


def parse_brick_sequence(raw: str) -> list[Brick]:
    bricks = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if line:
            bricks.append(parse_brick(line))
    return bricks
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /workspace/Lego-Gen && python -m pytest tests/test_brick_parser.py -v`
Expected: All 7 tests PASS

- [ ] **Step 6: Commit**

```bash
git add backend/brick/ tests/test_brick_parser.py
git commit -m "feat(brick): add brick constants, parser, and serializer"
```

---

## Task 2: Voxel occupancy grid

**Files:**
- Create: `backend/brick/occupancy.py`
- Test: `tests/test_brick_occupancy.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_brick_occupancy.py
import pytest
from backend.brick.parser import Brick
from backend.brick.occupancy import VoxelGrid


def test_empty_grid():
    g = VoxelGrid()
    assert g.is_empty(0, 0, 0)


def test_place_brick():
    g = VoxelGrid()
    b = Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09")
    assert g.can_place(b)
    g.place(b)
    assert not g.is_empty(0, 0, 0)
    assert not g.is_empty(1, 3, 0)
    assert g.is_empty(2, 0, 0)


def test_collision():
    g = VoxelGrid()
    b1 = Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09")
    b2 = Brick(h=2, w=2, x=1, y=1, z=0, color="05131D")
    g.place(b1)
    assert not g.can_place(b2)


def test_out_of_bounds():
    g = VoxelGrid()
    b = Brick(h=2, w=4, x=19, y=0, z=0, color="C91A09")
    assert not g.can_place(b)


def test_stacking():
    g = VoxelGrid()
    b1 = Brick(h=2, w=2, x=0, y=0, z=0, color="C91A09")
    b2 = Brick(h=2, w=2, x=0, y=0, z=1, color="05131D")
    g.place(b1)
    assert g.can_place(b2)
    g.place(b2)
    assert not g.is_empty(0, 0, 1)


def test_invalid_dims():
    g = VoxelGrid()
    b = Brick(h=3, w=3, x=0, y=0, z=0, color="C91A09")
    assert not g.can_place(b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /workspace/Lego-Gen && python -m pytest tests/test_brick_occupancy.py -v`
Expected: FAIL

- [ ] **Step 3: Implement occupancy grid**

```python
# backend/brick/occupancy.py
"""20x20x20 voxel occupancy grid for collision detection."""

import numpy as np
from backend.brick.constants import WORLD_DIM, BRICK_SHAPES
from backend.brick.parser import Brick


class VoxelGrid:
    def __init__(self) -> None:
        self.grid = np.zeros((WORLD_DIM, WORLD_DIM, WORLD_DIM), dtype=np.bool_)

    def is_empty(self, x: int, y: int, z: int) -> bool:
        return not self.grid[x, y, z]

    def can_place(self, brick: Brick) -> bool:
        h, w, x, y, z = brick.h, brick.w, brick.x, brick.y, brick.z
        if (h, w) not in BRICK_SHAPES:
            return False
        if x < 0 or y < 0 or z < 0:
            return False
        if x + h > WORLD_DIM or y + w > WORLD_DIM or z >= WORLD_DIM:
            return False
        return not np.any(self.grid[x : x + h, y : y + w, z])

    def place(self, brick: Brick) -> None:
        self.grid[brick.x : brick.x + brick.h, brick.y : brick.y + brick.w, brick.z] = True

    def remove(self, brick: Brick) -> None:
        self.grid[brick.x : brick.x + brick.h, brick.y : brick.y + brick.w, brick.z] = False

    def clear(self) -> None:
        self.grid[:] = False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /workspace/Lego-Gen && python -m pytest tests/test_brick_occupancy.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/brick/occupancy.py tests/test_brick_occupancy.py
git commit -m "feat(brick): add voxel occupancy grid with collision detection"
```

---

## Task 3: Connectivity-based stability checker

**Files:**
- Create: `backend/brick/stability.py`
- Test: `tests/test_brick_stability.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_brick_stability.py
import pytest
from backend.brick.parser import Brick
from backend.brick.stability import is_stable, find_first_unstable


def test_single_ground_brick_stable():
    bricks = [Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09")]
    assert is_stable(bricks)


def test_floating_brick_unstable():
    bricks = [Brick(h=2, w=2, x=0, y=0, z=3, color="C91A09")]
    assert not is_stable(bricks)


def test_supported_brick_stable():
    bricks = [
        Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09"),
        Brick(h=2, w=2, x=0, y=0, z=1, color="05131D"),
    ]
    assert is_stable(bricks)


def test_unsupported_second_layer():
    bricks = [
        Brick(h=2, w=2, x=0, y=0, z=0, color="C91A09"),
        Brick(h=2, w=2, x=10, y=10, z=1, color="05131D"),
    ]
    assert not is_stable(bricks)


def test_find_first_unstable():
    bricks = [
        Brick(h=2, w=2, x=0, y=0, z=0, color="C91A09"),
        Brick(h=2, w=2, x=0, y=0, z=1, color="05131D"),
        Brick(h=1, w=1, x=15, y=15, z=1, color="237841"),
    ]
    assert find_first_unstable(bricks) == 2


def test_find_first_unstable_all_stable():
    bricks = [
        Brick(h=2, w=2, x=0, y=0, z=0, color="C91A09"),
        Brick(h=2, w=2, x=0, y=0, z=1, color="05131D"),
    ]
    assert find_first_unstable(bricks) == -1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /workspace/Lego-Gen && python -m pytest tests/test_brick_stability.py -v`
Expected: FAIL

- [ ] **Step 3: Implement stability checker**

```python
# backend/brick/stability.py
"""Connectivity-based stability analysis for brick structures."""

from backend.brick.parser import Brick


def _overlaps_xy(a: Brick, b: Brick) -> bool:
    return (
        a.x < b.x + b.h and a.x + a.h > b.x and
        a.y < b.y + b.w and a.y + a.w > b.y
    )


def is_stable(bricks: list[Brick]) -> bool:
    if not bricks:
        return True

    n = len(bricks)
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(bricks[i].z - bricks[j].z) == 1 and _overlaps_xy(bricks[i], bricks[j]):
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    queue: list[int] = []
    for i, b in enumerate(bricks):
        if b.z == 0:
            visited[i] = True
            queue.append(i)

    head = 0
    while head < len(queue):
        cur = queue[head]
        head += 1
        for nb in adj[cur]:
            if not visited[nb]:
                visited[nb] = True
                queue.append(nb)

    return all(visited)


def find_first_unstable(bricks: list[Brick]) -> int:
    if is_stable(bricks):
        return -1
    for i in range(1, len(bricks) + 1):
        if not is_stable(bricks[:i]):
            return i - 1
    return -1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /workspace/Lego-Gen && python -m pytest tests/test_brick_stability.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/brick/stability.py tests/test_brick_stability.py
git commit -m "feat(brick): add connectivity-based stability checker"
```

---

## Task 4: Data pipeline — download ST2B and prepare training JSONL

**Files:**
- Create: `backend/data_pipeline/prepare_brick_dataset.py`
- Test: `tests/test_prepare_brick_dataset.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_prepare_brick_dataset.py
import pytest
from backend.data_pipeline.prepare_brick_dataset import (
    parse_st2b_bricks,
    pick_color_for_brick,
    colorize_structure,
    format_training_example,
)
from backend.brick.parser import Brick


def test_parse_st2b_bricks():
    raw = "1x1 (15,17,0)\n1x4 (15,13,0)\n"
    bricks = parse_st2b_bricks(raw)
    assert len(bricks) == 2
    assert bricks[0] == (1, 1, 15, 17, 0)


def test_parse_st2b_bricks_empty():
    assert parse_st2b_bricks("") == []


def test_pick_color_returns_valid_hex():
    color = pick_color_for_brick(caption="A red car", category="car", z=0, seed=42)
    assert len(color) == 6


def test_colorize_structure():
    raw = [(2, 4, 5, 3, 0), (1, 2, 3, 7, 1)]
    colored = colorize_structure(raw, caption="A blue table", category="table", seed=0)
    assert len(colored) == 2
    assert all(isinstance(b, Brick) for b in colored)


def test_format_training_example():
    bricks = [
        Brick(h=2, w=4, x=5, y=3, z=0, color="C91A09"),
        Brick(h=1, w=2, x=3, y=7, z=1, color="05131D"),
    ]
    example = format_training_example("A red chair", bricks)
    assert example["messages"][0]["role"] == "system"
    assert "A red chair" in example["messages"][1]["content"]
    assert "2x4 (5,3,0) #C91A09" in example["messages"][2]["content"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /workspace/Lego-Gen && python -m pytest tests/test_prepare_brick_dataset.py -v`
Expected: FAIL

- [ ] **Step 3: Implement prepare_brick_dataset.py**

Create `backend/data_pipeline/prepare_brick_dataset.py` with the full implementation including `parse_st2b_bricks`, `pick_color_for_brick`, `colorize_structure`, `format_training_example`, and `main()` CLI that downloads from HuggingFace and emits JSONL. See Task 4 in the detailed plan above for complete code.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /workspace/Lego-Gen && python -m pytest tests/test_prepare_brick_dataset.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/data_pipeline/prepare_brick_dataset.py tests/test_prepare_brick_dataset.py
git commit -m "feat(data): add brick dataset preparation with color assignment"
```

- [ ] **Step 6: Run data pipeline**

Run: `cd /workspace/Lego-Gen && python -m backend.data_pipeline.prepare_brick_dataset`
Expected: Downloads ST2B, outputs `data/brick_training/train.jsonl` (~235k lines) and `test.jsonl` (~24k lines).

- [ ] **Step 7: Verify output**

Run: `head -1 data/brick_training/train.jsonl | python -m json.tool | head -20`
Expected: Valid JSON with messages array, assistant content has colored brick lines.

- [ ] **Step 8: Commit**

```bash
echo "Generated by prepare_brick_dataset.py" > data/brick_training/README.md
git add data/brick_training/README.md
git commit -m "docs: add brick training data directory"
```

---

## Task 5: Training script

**Files:**
- Modify: `backend/config.py`
- Create: `backend/training/train_brick.py`

- [ ] **Step 1: Add brick config constants**

Add to `backend/config.py` after the Stage 1 config block:

```python
# ── Stage 2: Brick coordinate model ────────────────────────────────────
BRICK_MODEL_NAME = "Qwen/Qwen3.5-4B"
BRICK_CHECKPOINT_DIR = CHECKPOINT_DIR / "qwen35-4b-brick-lora"
BRICK_LEARNING_RATE = 2e-3
BRICK_BATCH_SIZE = 4
BRICK_GRADIENT_ACCUMULATION = 4
BRICK_MAX_SEQ_LENGTH = 8192
BRICK_NUM_EPOCHS = 3
BRICK_LORA_R = 32
BRICK_LORA_ALPHA = 64
BRICK_LORA_DROPOUT = 0.05
BRICK_TRAINING_DATA = DATA_DIR / "brick_training"
```

- [ ] **Step 2: Create training script**

Create `backend/training/train_brick.py` with LoRA fine-tuning using TRL SFTTrainer. See Task 5 in the detailed plan above for complete code.

- [ ] **Step 3: Verify import**

Run: `cd /workspace/Lego-Gen && python -c "from backend.training.train_brick import main; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add backend/config.py backend/training/train_brick.py
git commit -m "feat(training): add Qwen3.5-4B brick sequence training script"
```

---

## Task 6: Constrained decoding and inference pipeline

**Files:**
- Create: `backend/brick/decoder.py`
- Create: `backend/inference/brick_pipeline.py`
- Test: `tests/test_brick_decoder.py`

- [ ] **Step 1: Write failing tests for decoder**

```python
# tests/test_brick_decoder.py
import pytest
from backend.brick.decoder import BrickTokenConstraint
from backend.brick.constants import ALLOWED_DIMS, ALLOWED_COLORS


def test_allowed_dims_populated():
    assert len(ALLOWED_DIMS) == 14


def test_allowed_colors_populated():
    assert len(ALLOWED_COLORS) > 40


def test_constraint_initial_state():
    c = BrickTokenConstraint()
    allowed = c.get_allowed_strings()
    for dim in ALLOWED_DIMS:
        assert dim in allowed


def test_constraint_after_dim():
    c = BrickTokenConstraint()
    c.feed("2x4")
    assert c.get_allowed_strings() == [" ("]


def test_constraint_full_sequence():
    c = BrickTokenConstraint()
    for tok in ["2x4", " (", "5", ",", "3", ",", "0", ") #", "C91A09", "\n"]:
        c.feed(tok)
    for dim in ALLOWED_DIMS:
        assert dim in c.get_allowed_strings()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /workspace/Lego-Gen && python -m pytest tests/test_brick_decoder.py -v`
Expected: FAIL

- [ ] **Step 3: Implement decoder and inference pipeline**

Create `backend/brick/decoder.py` (state machine) and `backend/inference/brick_pipeline.py` (full generation loop with rejection sampling and rollback). See Task 6 in the detailed plan above for complete code.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /workspace/Lego-Gen && python -m pytest tests/test_brick_decoder.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/brick/decoder.py backend/inference/brick_pipeline.py tests/test_brick_decoder.py
git commit -m "feat(inference): add constrained brick decoder and inference pipeline"
```

---

## Task 7: API integration

**Files:**
- Modify: `backend/inference/pipeline.py`
- Modify: `backend/api/routes_generate.py`

- [ ] **Step 1: Add brick methods to UnifiedPipeline**

Add `generate_brick_build(caption)` and `generate_brick_build_from_image(image)` to the `UnifiedPipeline` class in `backend/inference/pipeline.py`. See Task 7 in the detailed plan above.

- [ ] **Step 2: Add /api/generate-bricks endpoint**

Add the new endpoint to `backend/api/routes_generate.py`. See Task 7 above.

- [ ] **Step 3: Commit**

```bash
git add backend/inference/pipeline.py backend/api/routes_generate.py
git commit -m "feat(api): add /api/generate-bricks endpoint"
```

---

## Task 8: Frontend — brick coordinate viewer

**Files:**
- Create: `frontend/src/components/BrickCoordViewer.tsx`
- Modify: `frontend/src/api/legogen.ts`

- [ ] **Step 1: Add types and API function**

Add `BrickCoord`, `BrickResponse`, `parseBrickString`, and `generateBricks` to `frontend/src/api/legogen.ts`. See Task 8 in the detailed plan above.

- [ ] **Step 2: Create BrickCoordViewer component**

Create `frontend/src/components/BrickCoordViewer.tsx` with Three.js direct placement from coordinates. See Task 8 above.

- [ ] **Step 3: Type-check**

Run: `cd /workspace/Lego-Gen/frontend && npx tsc --noEmit --pretty`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/BrickCoordViewer.tsx frontend/src/api/legogen.ts
git commit -m "feat(frontend): add BrickCoordViewer for coordinate-based rendering"
```

---

## Task 9: Wire brick viewer into existing UI

**Files:**
- Modify: `frontend/src/components/LegoViewer.tsx`
- Modify: `frontend/src/components/GuidanceViewer.tsx`

- [ ] **Step 1: Update LegoViewer**

Add optional `brickString` prop. When present, render `BrickCoordViewer` instead of the legacy pack-based viewer. See Task 9 in the detailed plan above.

- [ ] **Step 2: Update GuidanceViewer**

Same pattern — optional `brickString` prop delegates to `BrickCoordViewer`.

- [ ] **Step 3: Type-check**

Run: `cd /workspace/Lego-Gen/frontend && npx tsc --noEmit --pretty`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/LegoViewer.tsx frontend/src/components/GuidanceViewer.tsx
git commit -m "feat(frontend): integrate BrickCoordViewer into existing viewers"
```

---

## Task 10: End-to-end validation

- [ ] **Step 1: Run all backend tests**

Run: `cd /workspace/Lego-Gen && python -m pytest tests/test_brick_parser.py tests/test_brick_occupancy.py tests/test_brick_stability.py tests/test_brick_decoder.py tests/test_prepare_brick_dataset.py -v`
Expected: All tests PASS

- [ ] **Step 2: Type-check frontend**

Run: `cd /workspace/Lego-Gen/frontend && npx tsc --noEmit --pretty`
Expected: No errors

- [ ] **Step 3: Verify training data**

Run: `cd /workspace/Lego-Gen && python -c "
from backend.brick.parser import parse_brick_sequence
import json
with open('data/brick_training/train.jsonl') as f:
    line = f.readline()
example = json.loads(line)
bricks = parse_brick_sequence(example['messages'][2]['content'])
print(f'Bricks: {len(bricks)}, First: {bricks[0]}, Last: {bricks[-1]}')
print('PASS')
"`
Expected: Prints brick info then PASS.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete brick-accurate 3D generation pipeline"
```

---

## Note: LDraw export

The spec mentions `.ldr` export. This is a nice-to-have that can be added after the core pipeline works. The coordinate transform is: `x_ldr = (x+h*0.5)*20, z_ldr = (y+w*0.5)*20, y_ldr = -z*24` with color mapping from hex to LDraw color codes.
