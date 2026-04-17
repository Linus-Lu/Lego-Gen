# 100% Test Coverage + CI Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Instrument backend and frontend API seam with coverage, backfill gaps to 100%, and add a GitHub Actions gate that fails PRs below 100%.

**Architecture:** Add `coverage[toml]` + `pytest-cov` via a new `requirements-dev.txt` and a minimal `pyproject.toml` (coverage config only). Exclude real model-loading code via `# pragma: no cover` and a `[tool.coverage.run] omit` list. Stand up Vitest + v8 coverage for `frontend/src/api/legogen.ts` only. Lock it in with a new `.github/workflows/tests.yml` that runs pytest and Vitest in parallel, both enforcing 100%.

**Tech Stack:** Python 3.11, pytest, pytest-asyncio, pytest-cov, coverage[toml], httpx (test client), Vitest, @vitest/coverage-v8, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-04-17-test-coverage-design.md`

**Run convention (from CLAUDE.md):** always use `.venv/bin/python -m pytest`, never system `python`. `LEGOGEN_DEV=1` forces the mock inference path.

---

## File Structure

**New files:**
- `requirements-dev.txt` — test-only Python deps (no torch/transformers).
- `pyproject.toml` — only `[tool.coverage.*]` sections. No project metadata.
- `frontend/vitest.config.ts` — Vitest config, coverage scope = `src/api/**`.
- `frontend/src/api/legogen.test.ts` — tests for the API client.
- `tests/test_gallery_routes.py` — FastAPI route-level validation tests.
- `tests/test_brick_pipeline_mock.py` — `MockBrickPipeline` + factory singletons.
- `tests/test_config.py` — `backend/config.py` import + torch-absent branch.
- `tests/test_brick_constants.py` — `BRICK_SHAPES` / `ALLOWED_DIMS` consistency.
- `tests/test_stage1_pipeline.py` — `_MockStage1` happy path.
- `tests/test_app.py` — `/health` + CORS env-var parsing.
- `.github/workflows/tests.yml` — CI gate.

**Modified files:**
- `tests/conftest.py` — add shared fixtures.
- `tests/test_brick_decoder.py` — drop `colors.json` dependency; add transitions.
- `tests/test_api_routes.py` — add 413/504/SSE-ordering/client-disconnect cases.
- `tests/test_brick_pipeline_logic.py` — add grammar ordering, outlines-absent, `strategy="rank"`.
- `tests/test_best_of_n.py` — add zero-stable fallback, empty-bricks path.
- `frontend/package.json` — add Vitest dev deps + scripts.
- `backend/app.py` — add `# pragma: no cover` to real-preload branch.
- `backend/inference/brick_pipeline.py` — `# pragma: no cover` on real `BrickPipeline` class body and non-dev factory branches.
- `backend/inference/stage1_pipeline.py` — `# pragma: no cover` file-level or per-function.
- `backend/data_pipeline/build_stage1_dataset.py` — `# pragma: no cover` on `load_coco_annotations`, `build_stage1_manifest`, `main`.
- `backend/config.py` — `# pragma: no cover` on torch-import `except ImportError` branch if we can't test it (we will).
- `CLAUDE.md` — remove pre-existing-failure note; add coverage section.
- `.claude/settings.local.json` — remove `colors.json` deselect entry.

---

## Task 1: Add Python tooling scaffold

**Files:**
- Create: `requirements-dev.txt`
- Create: `pyproject.toml`

- [ ] **Step 1: Create `requirements-dev.txt`**

```
# Test-only Python deps. Install alongside a prod environment or on its own
# to run `LEGOGEN_DEV=1 pytest`. Does NOT include torch / transformers /
# peft / outlines / bitsandbytes — LEGOGEN_DEV=1 short-circuits every
# code path that imports them.

pytest>=8,<9
pytest-asyncio>=0.23,<1
pytest-cov>=5,<7
coverage[toml]>=7.5,<8

# Minimum prod deps that the dev-mode test path actually imports.
fastapi>=0.110,<1
pydantic>=2,<3
httpx>=0.27,<1
aiosqlite>=0.20,<1
Pillow>=10,<12
numpy>=1.26,<3
scipy>=1.11,<2
scikit-learn>=1.3,<2
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
# Coverage configuration only. The project has no packaging metadata —
# CLAUDE.md notes that no pyproject.toml / setup.cfg / pytest.ini exists
# by design; this file is strictly `[tool.coverage.*]`.

[tool.coverage.run]
branch = true
source = ["backend"]
# Files excluded entirely because every line loads torch/transformers or
# orchestrates a dataset download that can't run in CI.
omit = [
    "backend/training/train_stage1.py",
    "backend/training/train_brick.py",
    "backend/training/utils.py",
    "backend/data_pipeline/dataset_stage1.py",
    "backend/inference/stage1_pipeline.py",
]

[tool.coverage.report]
# Note: the 100% threshold is enforced in CI via `pytest --cov-fail-under=100`
# (see .github/workflows/tests.yml). Keeping it out of `[tool.coverage.report]`
# lets local intermediate runs still exit 0 while tests backfill.
show_missing = true
skip_covered = false
exclude_also = [
    # Type-checking only branches
    "if TYPE_CHECKING:",
    # CLI entry points
    "if __name__ == .__main__.:",
    # Defensive re-raise after logging etc.
    "raise NotImplementedError",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
addopts = "--cov --cov-report=term-missing"
```

- [ ] **Step 3: Install deps into the existing .venv**

Run: `.venv/bin/pip install -r requirements-dev.txt`

Expected: all packages install successfully; no build from source of torch or transformers.

- [ ] **Step 4: Verify pytest still collects**

Run: `LEGOGEN_DEV=1 .venv/bin/python -m pytest --collect-only -q 2>&1 | tail -5`

Expected: "87 tests collected" (or similar) with no import errors. Coverage starts measuring automatically because of `addopts`.

- [ ] **Step 5: Commit**

```bash
git add requirements-dev.txt pyproject.toml
git commit -m "build(test): add dev-deps manifest and coverage config"
```

---

## Task 2: Add frontend tooling scaffold

**Files:**
- Create: `frontend/vitest.config.ts`
- Modify: `frontend/package.json`

- [ ] **Step 1: Create `frontend/vitest.config.ts`**

```typescript
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: true,
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'lcov'],
      include: ['src/api/**'],
      // Everything else stays out of scope — tested by integration / manual.
      exclude: [
        'src/components/**',
        'src/pages/**',
        'src/App.tsx',
        'src/main.tsx',
        '**/*.test.ts',
        '**/*.test.tsx',
      ],
      thresholds: {
        lines: 100,
        branches: 100,
        functions: 100,
        statements: 100,
      },
    },
  },
});
```

- [ ] **Step 2: Update `frontend/package.json`**

Add to `devDependencies` (preserve the existing entries exactly):

```json
    "@vitest/coverage-v8": "^2.1.0",
    "jsdom": "^25.0.0",
    "vitest": "^2.1.0"
```

Add to `scripts`:

```json
    "test": "vitest run",
    "test:watch": "vitest",
    "test:coverage": "vitest run --coverage"
```

- [ ] **Step 3: Install**

Run: `cd frontend && npm install`

Expected: `node_modules` updates, no peer-dep warnings that block install.

- [ ] **Step 4: Verify Vitest runs (no tests yet — exit 0 or "no tests" is fine)**

Run: `cd frontend && npm run test`

Expected: Vitest prints "No test files found" or similar with exit code 0 / 1 — not a hard crash. (Vitest 2.x prints a friendly message and exits; we'll write tests in Task 15.)

- [ ] **Step 5: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/vitest.config.ts
git commit -m "build(test): add Vitest + v8 coverage for frontend API seam"
```

---

## Task 3: Baseline coverage measurement + pragma audit

**Files:**
- Modify: `backend/app.py`
- Modify: `backend/inference/brick_pipeline.py`
- Modify: `backend/data_pipeline/build_stage1_dataset.py`
- Modify: `backend/config.py`

- [ ] **Step 1: Run the suite and capture the baseline**

Run: `LEGOGEN_DEV=1 .venv/bin/python -m pytest --deselect tests/test_brick_decoder.py::test_allowed_colors_populated -q 2>&1 | tail -40`

Expected: tests pass (minus the pre-existing skip that's already in settings.local.json) and coverage table prints. Note the baseline % per-file — used for the sanity check at the end of Task 14.

- [ ] **Step 2: Add pragma to `backend/app.py` real-preload branch**

In `backend/app.py`, find the lifespan function and wrap the non-dev branch:

```python
    await gallery_db.init_db()
    if os.environ.get("LEGOGEN_DEV", "1") != "1":  # pragma: no cover
        # Real model preload — can't run in CI (no GPU, no weights).
        print("LEGOGen API starting — preloading models...")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _preload_models)
        print("LEGOGen API ready. Models loaded.")
    else:
        print("LEGOGen API ready (dev mode — mock pipeline).")
```

And mark the helper:

```python
def _preload_models():  # pragma: no cover
    """Load the brick + Stage 1 pipelines in a thread so startup isn't blocked."""
    from backend.inference.brick_pipeline import get_brick_pipeline, _get_stage1_pipeline
    get_brick_pipeline()
    _get_stage1_pipeline()
```

- [ ] **Step 3: Add pragma to `backend/inference/brick_pipeline.py`**

Mark the real `BrickPipeline` class (not `MockBrickPipeline`) and the non-dev factory branches. The whole `BrickPipeline` class body loads torch — add `# pragma: no cover` right after the class docstring and on the `__init__`, `_fresh_logits_processor`, `generate`, `generate_best_of_n`, `generate_from_image`, and `_generate_one_brick` signatures. Easier: use file-level exclusion via the class docstring; apply per-method:

```python
class BrickPipeline:
    """..."""

    def __init__(self, device: str = "cuda") -> None:  # pragma: no cover
        ...

    def _fresh_logits_processor(self):  # pragma: no cover
        ...

    def generate(self, caption: str, on_progress=None) -> dict:  # pragma: no cover
        ...

    def generate_best_of_n(self, caption: str, n: int = 16, strategy: str = "cluster", on_progress=None) -> dict:  # pragma: no cover
        ...

    def generate_from_image(self, image, on_progress=None) -> dict:  # pragma: no cover
        ...

    def _generate_one_brick(
        self, input_ids, grid: VoxelGrid, logits_processor=None,
    ) -> tuple[Optional[Brick], int]:  # pragma: no cover
        ...
```

Also the module-level factory branches:

```python
def get_brick_pipeline():
    global _brick_instance
    if _brick_instance is None:
        with _brick_lock:
            if _brick_instance is None:
                if LEGOGEN_DEV:
                    _brick_instance = MockBrickPipeline()
                else:  # pragma: no cover
                    _brick_instance = BrickPipeline()
    return _brick_instance


def _get_stage1_pipeline():
    global _stage1_instance
    if _stage1_instance is None:
        with _stage1_lock:
            if _stage1_instance is None:
                if LEGOGEN_DEV:
                    _stage1_instance = _MockStage1()
                else:  # pragma: no cover
                    from backend.inference.stage1_pipeline import Stage1Pipeline
                    _stage1_instance = Stage1Pipeline()
    return _stage1_instance
```

Note: the existing code has `_brick_instance = MockBrickPipeline() if LEGOGEN_DEV else BrickPipeline()` as a single conditional — rewrite to the `if/else` above so the pragma can attach to the real branch only.

Also mark `_build_logits_processor`'s `from outlines.processors import ...` call — the import-failure `except ImportError` is coverable (we'll test it in Task 8), so do NOT pragma it.

- [ ] **Step 4: Add pragma to `backend/data_pipeline/build_stage1_dataset.py`**

Add `# pragma: no cover` to the signatures of `load_coco_annotations` (line 150), `build_stage1_manifest` (line 226), and `main` (line 345). These are network/IO orchestration.

> **Note:** This step was completed as part of the Task 1 fix-up (commit that drops `fail_under` from pyproject and pragma'd the three functions + the untested `_make_seed` / `main` in `prepare_brick_dataset.py`). Verify the pragmas are still in place; otherwise skip to Step 5.

- [ ] **Step 5: Leave `backend/config.py` unannotated**

The `try: import torch / except ImportError` branches are both testable — we cover them in Task 10. No pragma.

- [ ] **Step 6: Re-run coverage, confirm every pragma is actually reached or explicitly excluded**

Run: `LEGOGEN_DEV=1 .venv/bin/python -m pytest --deselect tests/test_brick_decoder.py::test_allowed_colors_populated -q 2>&1 | tail -30`

Expected: coverage rises vs. Step 1's baseline. Look for any remaining < 100% files — note them; they'll be addressed by Tasks 4-13.

- [ ] **Step 7: Commit**

```bash
git add backend/app.py backend/inference/brick_pipeline.py backend/data_pipeline/build_stage1_dataset.py
git commit -m "test(cov): add # pragma: no cover to untestable model-loading code"
```

---

## Task 4: Fix `test_brick_decoder.py` and expand decoder coverage

**Files:**
- Modify: `tests/test_brick_decoder.py`

Existing `test_allowed_colors_populated` relies on `data/cache/colors.json` being present, which it isn't in CI or a fresh checkout. Replace with a test that seeds the lazy cache directly, then add state-transition coverage.

- [ ] **Step 1: Write the failing tests**

Replace the entire file content:

```python
"""Tests for backend.brick.decoder — token-level constraint state machine."""

import pytest

from backend.brick import constants as const
from backend.brick.decoder import BrickTokenConstraint
from backend.brick.constants import ALLOWED_DIMS


@pytest.fixture
def seeded_colors(monkeypatch):
    """Seed the lazy color-palette cache so ALLOWED_COLORS works without colors.json."""
    fake_palette = {
        "C91A09": "Red",
        "FFFFFF": "White",
        "000000": "Black",
        "0055BF": "Blue",
    }
    # Pre-populate the memoized cache used by _lazy_palette.
    const._lazy_palette._cache = fake_palette  # type: ignore[attr-defined]
    yield fake_palette
    # Reset so other tests don't inherit the fake palette.
    if hasattr(const._lazy_palette, "_cache"):
        del const._lazy_palette._cache


def test_allowed_dims_populated():
    assert len(ALLOWED_DIMS) == 14


def test_allowed_colors_populated(seeded_colors):
    from backend.brick.constants import ALLOWED_COLORS
    assert len(ALLOWED_COLORS) == 4
    assert "C91A09" in ALLOWED_COLORS


def test_constraint_initial_state():
    c = BrickTokenConstraint()
    allowed = c.get_allowed_strings()
    for dim in ALLOWED_DIMS:
        assert dim in allowed


def test_constraint_after_dim():
    c = BrickTokenConstraint()
    c.feed("2x4")
    assert c.get_allowed_strings() == [" ("]


def test_constraint_coord_state_exposes_positions():
    c = BrickTokenConstraint()
    for tok in ["2x4", " ("]:
        c.feed(tok)
    # State 2 expects x coord — positions 0..19.
    allowed = c.get_allowed_strings()
    assert "0" in allowed
    assert "19" in allowed
    assert str(const.WORLD_DIM) not in allowed


def test_constraint_comma_states():
    c = BrickTokenConstraint()
    for tok in ["2x4", " (", "5"]:
        c.feed(tok)
    assert c.get_allowed_strings() == [","]
    c.feed(",")
    c.feed("3")
    assert c.get_allowed_strings() == [","]


def test_constraint_close_state():
    c = BrickTokenConstraint()
    for tok in ["2x4", " (", "5", ",", "3", ",", "0"]:
        c.feed(tok)
    assert c.get_allowed_strings() == [") #"]


def test_constraint_color_state(seeded_colors):
    from backend.brick.constants import ALLOWED_COLORS
    c = BrickTokenConstraint()
    for tok in ["2x4", " (", "5", ",", "3", ",", "0", ") #"]:
        c.feed(tok)
    assert c.get_allowed_strings() == list(ALLOWED_COLORS)


def test_constraint_newline_state(seeded_colors):
    c = BrickTokenConstraint()
    for tok in ["2x4", " (", "5", ",", "3", ",", "0", ") #", "C91A09"]:
        c.feed(tok)
    assert c.get_allowed_strings() == ["\n"]


def test_constraint_full_sequence_loops(seeded_colors):
    c = BrickTokenConstraint()
    for tok in ["2x4", " (", "5", ",", "3", ",", "0", ") #", "C91A09", "\n"]:
        c.feed(tok)
    # After newline we're back at state 0 -> dims allowed again.
    for dim in ALLOWED_DIMS:
        assert dim in c.get_allowed_strings()


def test_constraint_reset():
    c = BrickTokenConstraint()
    c.feed("2x4")
    c.feed(" (")
    c.reset()
    for dim in ALLOWED_DIMS:
        assert dim in c.get_allowed_strings()


def test_constraint_out_of_range_state_returns_empty():
    """Defensive: manual state poke should yield empty allowed list."""
    c = BrickTokenConstraint()
    c.state = 99
    assert c.get_allowed_strings() == []
```

- [ ] **Step 2: Run the tests to verify most fail for the right reason**

Run: `LEGOGEN_DEV=1 .venv/bin/python -m pytest tests/test_brick_decoder.py -v`

Expected: the new tests pass (decoder already supports everything tested). If `test_allowed_colors_populated` still errors with `FileNotFoundError`, the fixture isn't resetting between tests — debug by adding `print(const._lazy_palette._cache)` and re-running.

- [ ] **Step 3: Remove the deselect entry from `.claude/settings.local.json`**

Find the line containing `--deselect tests/test_brick_decoder.py::test_allowed_colors_populated` and delete it from the `allow` list.

- [ ] **Step 4: Re-run without the deselect flag**

Run: `LEGOGEN_DEV=1 .venv/bin/python -m pytest tests/test_brick_decoder.py -v`

Expected: all tests pass. No more skip.

- [ ] **Step 5: Commit**

```bash
git add tests/test_brick_decoder.py .claude/settings.local.json
git commit -m "test(decoder): stop depending on colors.json, add state-machine coverage"
```

---

## Task 5: Add shared test fixtures

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Replace `tests/conftest.py`**

```python
import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")
    # Every test runs in dev mode so MockBrickPipeline is used. Individual
    # tests that need to override this must do so explicitly.
    os.environ.setdefault("LEGOGEN_DEV", "1")


def pytest_collection_modifyitems(config, items):
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    if not has_cuda:
        skip_gpu = pytest.mark.skip(reason="No CUDA GPU available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture
def gallery_db_path(tmp_path, monkeypatch):
    """Isolate gallery DB to a temp directory for test safety."""
    monkeypatch.setattr("backend.storage.gallery_db.DATA_DIR", tmp_path)
    monkeypatch.setattr(
        "backend.storage.gallery_db.DB_PATH", tmp_path / "test_gallery.db"
    )
    return tmp_path / "test_gallery.db"


@pytest.fixture
def reset_pipeline_singletons():
    """Force a fresh pipeline + stage1 mock on every test that needs it.

    The factories cache singletons at module level; tests that monkeypatch
    LEGOGEN_DEV or poke internals must not leak state across files.
    """
    import backend.inference.brick_pipeline as bp
    bp._brick_instance = None
    bp._stage1_instance = None
    yield
    bp._brick_instance = None
    bp._stage1_instance = None


@pytest.fixture
def seeded_palette(monkeypatch):
    """Pre-populate the lazy color palette so ALLOWED_COLORS works without colors.json."""
    from backend.brick import constants as const
    fake = {
        "C91A09": "Red",
        "FFFFFF": "White",
        "000000": "Black",
        "0055BF": "Blue",
        "FE8A18": "Orange",
        "237841": "Green",
        "720E0F": "Dark Red",
    }
    const._lazy_palette._cache = fake  # type: ignore[attr-defined]
    yield fake
    if hasattr(const._lazy_palette, "_cache"):
        del const._lazy_palette._cache
```

- [ ] **Step 2: Verify existing tests still pass**

Run: `LEGOGEN_DEV=1 .venv/bin/python -m pytest -q`

Expected: all tests still pass. The autouse `dev_mode` fixture in `test_api_routes.py` is now partially redundant with `pytest_configure` but keep it — it also resets the singleton which is the important part.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add shared reset_pipeline_singletons + seeded_palette fixtures"
```

---

## Task 6: Expand `test_api_routes.py` — validation and timeout paths

**Files:**
- Modify: `tests/test_api_routes.py`

- [ ] **Step 1: Write the failing tests — append to `tests/test_api_routes.py`**

Add at the end of the file:

```python
class TestGenerateValidationLimits:
    def test_image_exceeding_size_limit_returns_413(self, client):
        """MAX_IMAGE_BYTES is 10 MB — send 11 MB of zeros and expect 413."""
        oversize = b"\x00" * (11 * 1024 * 1024)
        resp = client.post(
            "/api/generate-bricks",
            files={"image": ("big.png", oversize, "image/png")},
        )
        assert resp.status_code == 413
        assert "10" in resp.json()["detail"]

    def test_prompt_exceeding_char_limit_returns_413(self, client):
        """MAX_PROMPT_CHARS is 2000; 2001 chars must 413."""
        resp = client.post(
            "/api/generate-bricks",
            data={"prompt": "x" * 2001},
        )
        assert resp.status_code == 413
        assert "2000" in resp.json()["detail"]

    def test_prompt_whitespace_only_treated_as_empty(self, client):
        resp = client.post("/api/generate-bricks", data={"prompt": "   \n\t  "})
        assert resp.status_code == 400

    def test_stream_image_wrong_mime_returns_400(self, client):
        resp = client.post(
            "/api/generate-stream",
            files={"image": ("x.txt", b"text", "text/plain")},
        )
        assert resp.status_code == 400

    def test_stream_no_input_returns_400(self, client):
        resp = client.post("/api/generate-stream", data={"prompt": ""})
        assert resp.status_code == 400


class TestGenerateTimeout:
    def test_generate_bricks_timeout_returns_504(self, monkeypatch, client):
        """Patch the MockBrickPipeline.generate to sleep longer than the timeout
        so asyncio.wait_for fires 504."""
        import time
        from backend.inference import brick_pipeline as bp
        from backend.api import routes_generate as rg

        monkeypatch.setattr(rg, "INFERENCE_TIMEOUT_SECONDS", 0.05)

        def slow_generate(self, caption, on_progress=None):
            time.sleep(1.0)
            return {"bricks": "", "brick_count": 0, "stable": True, "metadata": {}}

        monkeypatch.setattr(bp.MockBrickPipeline, "generate", slow_generate)
        resp = client.post("/api/generate-bricks", data={"prompt": "slow build"})
        assert resp.status_code == 504
        assert "timed out" in resp.json()["detail"].lower()


class TestGenerateBestOfNImagePath:
    def test_image_with_n_runs_bon(self, client):
        png = _make_png_bytes()
        resp = client.post(
            "/api/generate-bricks",
            files={"image": ("x.png", png, "image/png")},
            data={"n": "2"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["metadata"]["n"] == 2
        assert "caption" in body

    def test_stream_bon_emits_sample_events(self, client):
        resp = client.post(
            "/api/generate-stream",
            data={"prompt": "a red robot", "n": "2"},
        )
        assert resp.status_code == 200
        # MockBrickPipeline.generate_best_of_n doesn't emit sample events
        # (it calls generate once). This asserts the happy path still
        # terminates with a result event.
        assert "event: result" in resp.text


class TestStreamOrdering:
    def test_event_sequence(self, client):
        """SSE must emit at least one progress event before any result event."""
        resp = client.post("/api/generate-stream", data={"prompt": "house"})
        assert resp.status_code == 200
        body = resp.text
        progress_idx = body.index("event: progress")
        result_idx = body.index("event: result")
        assert progress_idx < result_idx

    def test_stream_image_caption_progress(self, client):
        """Image path must emit a stage1 progress event with the caption."""
        png = _make_png_bytes()
        resp = client.post(
            "/api/generate-stream",
            files={"image": ("x.png", png, "image/png")},
        )
        assert resp.status_code == 200
        assert "\"stage\": \"stage1\"" in resp.text
        assert "\"caption\"" in resp.text
```

- [ ] **Step 2: Run the new tests to verify current behavior**

Run: `LEGOGEN_DEV=1 .venv/bin/python -m pytest tests/test_api_routes.py -v`

Expected: all tests pass. These verify existing behavior — none require production code changes. If a test fails, re-read the route implementation in `backend/api/routes_generate.py` and adjust the test expectation; do **not** modify the route.

- [ ] **Step 3: Commit**

```bash
git add tests/test_api_routes.py
git commit -m "test(routes): cover 413/400/504, BoN image path, SSE ordering"
```

---

## Task 7: New `test_gallery_routes.py`

**Files:**
- Create: `tests/test_gallery_routes.py`

- [ ] **Step 1: Create the file**

```python
"""Route-level validation tests for /api/gallery endpoints.

test_gallery_db.py covers the SQLite functions directly; these tests cover
the Pydantic validation and HTTP status codes on the FastAPI router.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

httpx = pytest.importorskip("httpx")
fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient


@pytest.fixture
def client(gallery_db_path):
    os.environ["LEGOGEN_DEV"] = "1"
    from backend.app import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def a_build(client):
    resp = client.post(
        "/api/gallery",
        json={
            "title": "Seed",
            "caption": "",
            "bricks": "1x1 (0,0,0) #FFFFFF",
            "brick_count": 1,
        },
    )
    assert resp.status_code == 201
    return resp.json()


def test_create_missing_title_returns_422(client):
    resp = client.post("/api/gallery", json={"bricks": "1x1 (0,0,0) #FFFFFF"})
    assert resp.status_code == 422


def test_create_title_too_long_returns_422(client):
    resp = client.post(
        "/api/gallery",
        json={
            "title": "x" * 201,
            "caption": "",
            "bricks": "1x1 (0,0,0) #FFFFFF",
        },
    )
    assert resp.status_code == 422


def test_create_bricks_over_max_returns_422(client):
    resp = client.post(
        "/api/gallery",
        json={
            "title": "Big",
            "caption": "",
            "bricks": "x" * 200_001,
        },
    )
    assert resp.status_code == 422


def test_create_thumbnail_over_max_returns_422(client):
    resp = client.post(
        "/api/gallery",
        json={
            "title": "Thumb",
            "caption": "",
            "bricks": "1x1 (0,0,0) #FFFFFF",
            "thumbnail_b64": "x" * 400_001,
        },
    )
    assert resp.status_code == 422


def test_create_whitespace_only_bricks_returns_400(client):
    resp = client.post(
        "/api/gallery",
        json={"title": "Blank", "caption": "", "bricks": "   \n  "},
    )
    assert resp.status_code == 400


def test_star_nonexistent_returns_404(client):
    resp = client.patch("/api/gallery/does-not-exist/star", json={"stars": 3})
    assert resp.status_code == 404


def test_star_out_of_range_returns_422(client, a_build):
    resp = client.patch(
        f"/api/gallery/{a_build['id']}/star", json={"stars": 10}
    )
    assert resp.status_code == 422
    resp = client.patch(
        f"/api/gallery/{a_build['id']}/star", json={"stars": 0}
    )
    assert resp.status_code == 422


def test_list_with_search_and_sort(client, a_build):
    resp = client.get("/api/gallery?q=Seed&sort=bricks")
    assert resp.status_code == 200
    items = resp.json()
    assert any(b["id"] == a_build["id"] for b in items)
```

- [ ] **Step 2: Run the new tests**

Run: `LEGOGEN_DEV=1 .venv/bin/python -m pytest tests/test_gallery_routes.py -v`

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_gallery_routes.py
git commit -m "test(gallery): add route-level Pydantic validation coverage"
```

---

## Task 8: New `test_brick_pipeline_mock.py`

**Files:**
- Create: `tests/test_brick_pipeline_mock.py`

- [ ] **Step 1: Create the file**

```python
"""End-to-end coverage of MockBrickPipeline and the singleton factories.

The real BrickPipeline is pragma'd; the mock path is the one the API exercises
under LEGOGEN_DEV=1, so its behavior is worth pinning explicitly."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.inference import brick_pipeline as bp
from backend.inference.brick_pipeline import MockBrickPipeline, _MockStage1


def test_mock_generate_emits_expected_bricks():
    pipe = MockBrickPipeline()
    events = []
    out = pipe.generate("a small house", on_progress=events.append)
    assert out["brick_count"] == 12
    assert out["stable"] is True
    assert out["metadata"]["model_version"] == "mock-brick-v1"
    # One progress event per brick.
    assert [e["count"] for e in events] == list(range(1, 13))


def test_mock_generate_no_progress_callback():
    """The callback path is optional; omitting it must not raise."""
    pipe = MockBrickPipeline()
    out = pipe.generate("a small house")
    assert out["brick_count"] == 12


def test_mock_generate_best_of_n_stamps_metadata():
    pipe = MockBrickPipeline()
    out = pipe.generate_best_of_n("x", n=5)
    assert out["metadata"]["n"] == 5
    assert out["metadata"]["picked_index"] == 0
    assert out["metadata"]["stable_rate"] == 1.0


def test_mock_generate_from_image_returns_caption():
    pipe = MockBrickPipeline()
    out = pipe.generate_from_image(object())
    assert out["caption"].startswith("a small red house")
    assert out["brick_count"] == 12


def test_mock_generate_from_image_emits_caption_event():
    pipe = MockBrickPipeline()
    events = []
    pipe.generate_from_image(object(), on_progress=events.append)
    types = [e["type"] for e in events]
    assert types[0] == "caption"
    assert events[1:] == [{"type": "brick", "count": i} for i in range(1, 13)]


def test_mock_stage1_returns_fixed_caption():
    assert _MockStage1().describe(object()).startswith("a small red house")


def test_get_brick_pipeline_returns_singleton(reset_pipeline_singletons):
    a = bp.get_brick_pipeline()
    b = bp.get_brick_pipeline()
    assert a is b
    assert isinstance(a, MockBrickPipeline)


def test_get_stage1_pipeline_returns_singleton(reset_pipeline_singletons):
    a = bp._get_stage1_pipeline()
    b = bp._get_stage1_pipeline()
    assert a is b
    assert isinstance(a, _MockStage1)
```

- [ ] **Step 2: Run the new tests**

Run: `LEGOGEN_DEV=1 .venv/bin/python -m pytest tests/test_brick_pipeline_mock.py -v`

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_brick_pipeline_mock.py
git commit -m "test(pipeline): cover MockBrickPipeline methods + singleton factories"
```

---

## Task 9: Expand `test_brick_pipeline_logic.py`

**Files:**
- Modify: `tests/test_brick_pipeline_logic.py`

- [ ] **Step 1: Add new tests to the existing file**

Append these tests at the end of the file:

```python
# ── TestGrammarOrdering (no torch needed — uses module-level BRICK_PATTERN) ──


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="module import needs a torch-adjacent stub; covered when test env has torch OR build_grammar_pattern works standalone")
class TestGrammarOrdering:
    def test_longest_dim_prefix_matches_first(self):
        """A naive shortest-first regex would match '1x1' when '1x10' is the
        intended dim. 1x10 isn't in the allowed set, so this is actually a
        negative test against the ordering being wrong."""
        import re as _re
        from backend.inference.brick_pipeline import BRICK_PATTERN
        pat = _re.compile(BRICK_PATTERN)
        # 2x4 is allowed; 2x40 is not; if ordering were shortest-first and
        # the coord pattern were broken, this could false-match.
        assert pat.fullmatch("2x4 (0,0,0) #C91A09\n") is not None

    def test_coord_pattern_rejects_out_of_range(self):
        import re as _re
        from backend.inference.brick_pipeline import BRICK_PATTERN
        pat = _re.compile(BRICK_PATTERN)
        # WORLD_DIM is 20 → 20 is out of range, 19 is the max.
        assert pat.fullmatch("2x4 (20,0,0) #C91A09\n") is None
        assert pat.fullmatch("2x4 (0,0,19) #C91A09\n") is not None


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch required")
class TestLogitsProcessorFallback:
    def test_returns_none_when_outlines_absent(self, monkeypatch):
        """When outlines is not importable, _build_logits_processor returns None.
        The caller treats this as 'no grammar constraint' and validates
        parse failures via try/except instead."""
        import sys as _sys
        from backend.inference.brick_pipeline import _build_logits_processor
        # Simulate ImportError by wiping outlines from sys.modules.
        monkeypatch.setitem(_sys.modules, "outlines", None)
        monkeypatch.setitem(_sys.modules, "outlines.processors", None)
        result = _build_logits_processor(tokenizer=object(), pattern=r"x")
        assert result is None


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch required")
class TestBestOfNRankStrategy:
    def test_rank_strategy_picks_most_bricks_among_stable(self):
        """strategy='rank' skips the clustering entirely and returns
        rank_candidates(candidates)[0]."""
        from backend.inference.brick_pipeline import BrickPipeline

        pipe = BrickPipeline.__new__(BrickPipeline)
        counter = {"n": 0}

        def fake_generate(caption, on_progress=None):
            counter["n"] += 1
            # Return different brick counts per call so ranking has signal.
            i = counter["n"]
            return {
                "bricks": f"2x4 (0,0,0) #C91A09\n" * i,
                "brick_count": i,
                "stable": True,
                "metadata": {},
            }

        pipe.generate = fake_generate  # type: ignore[method-assign]
        out = pipe.generate_best_of_n("x", n=3, strategy="rank")
        # Ranking picks the largest (last call, brick_count=3).
        assert out["brick_count"] == 3
        assert out["metadata"]["picked_index"] == 2
```

Note: replace the skip decorator on the TestGrammarOrdering class — since `BRICK_PATTERN` is already imported at module-top under the `_TORCH_AVAILABLE` guard, `skipif(not _TORCH_AVAILABLE)` is correct. Use that single decorator form. (The verbose reason string above is for readability — a short "torch required" is fine.)

- [ ] **Step 2: Run the new tests**

Run: `LEGOGEN_DEV=1 .venv/bin/python -m pytest tests/test_brick_pipeline_logic.py -v`

Expected: all tests pass on a machine with torch installed. On a minimal CI without torch, the whole file's torch-gated classes are skipped, which is fine — the `_TORCH_AVAILABLE` guard already exists.

- [ ] **Step 3: Commit**

```bash
git add tests/test_brick_pipeline_logic.py
git commit -m "test(pipeline): cover grammar ordering, outlines-absent, rank strategy"
```

---

## Task 10: Expand `test_best_of_n.py`

**Files:**
- Modify: `tests/test_best_of_n.py`

- [ ] **Step 1: Append new tests**

```python
def test_structural_features_empty_bricks_returns_zero_vector():
    import numpy as np
    out = structural_features([])
    assert out.shape == (9,)
    assert np.allclose(out, 0.0)


def test_cluster_and_pick_all_unstable_falls_back_to_rank():
    """Zero stable candidates → fall through rank_candidates. Ranking
    prefers stable, so this returns the unstable candidate with the most
    bricks (all are unstable so stability is a tie)."""
    cands = [
        {"bricks": [Brick(2, 4, 0, 0, 0, "C91A09")] * 3, "stable": False, "brick_count": 3},
        {"bricks": [Brick(2, 4, 0, 0, 0, "C91A09")] * 7, "stable": False, "brick_count": 7},
        {"bricks": [Brick(2, 4, 0, 0, 0, "C91A09")] * 2, "stable": False, "brick_count": 2},
    ]
    picked = cluster_and_pick(cands, k=2, seed=0)
    assert picked["brick_count"] == 7


def test_rank_candidates_is_deterministic_on_ties():
    """Identical candidates → original order preserved (earlier wins)."""
    a = {"bricks": [], "stable": True, "brick_count": 5}
    b = {"bricks": [], "stable": True, "brick_count": 5}
    ranked = rank_candidates([a, b])
    assert ranked[0] is a
    assert ranked[1] is b
```

- [ ] **Step 2: Run the new tests**

Run: `LEGOGEN_DEV=1 .venv/bin/python -m pytest tests/test_best_of_n.py -v`

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_best_of_n.py
git commit -m "test(bon): cover empty-bricks and zero-stable fallback paths"
```

---

## Task 11: New `test_config.py`

**Files:**
- Create: `tests/test_config.py`

- [ ] **Step 1: Create the file**

```python
"""Coverage for backend/config.py — constants and the torch-import guard."""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_legogen_dev_true_by_default(monkeypatch):
    monkeypatch.setenv("LEGOGEN_DEV", "1")
    import backend.config as cfg
    importlib.reload(cfg)
    assert cfg.LEGOGEN_DEV is True


def test_legogen_dev_false_when_env_is_zero(monkeypatch):
    monkeypatch.setenv("LEGOGEN_DEV", "0")
    import backend.config as cfg
    importlib.reload(cfg)
    assert cfg.LEGOGEN_DEV is False
    # Reset for downstream tests.
    monkeypatch.setenv("LEGOGEN_DEV", "1")
    importlib.reload(cfg)


def test_paths_are_absolute():
    import backend.config as cfg
    assert cfg.PROJECT_ROOT.is_absolute()
    assert cfg.DATA_DIR.is_absolute()
    assert cfg.CHECKPOINT_DIR.parts[-3:] == ("backend", "models", "checkpoints")


def test_coco_category_mapping_nonempty():
    import backend.config as cfg
    assert "chair" in cfg.COCO_TO_ST2B_CATEGORY
    assert cfg.COCO_TO_ST2B_CATEGORY["couch"] == "sofa"


def test_device_branch_with_torch_absent(monkeypatch):
    """Simulate `import torch` raising ImportError and reload config."""
    monkeypatch.setitem(sys.modules, "torch", None)
    import backend.config as cfg
    importlib.reload(cfg)
    assert cfg.DEVICE == "cpu"
    assert cfg.USE_BF16 is False
    # Drop the override so the real module re-imports cleanly afterward.
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    importlib.reload(cfg)


def test_training_hyperparameters_set():
    """Guard against accidentally zero-ing a hyperparameter."""
    import backend.config as cfg
    assert cfg.STAGE1_LORA_R == 32
    assert cfg.BRICK_LORA_R == 32
    assert cfg.STAGE1_NUM_EPOCHS >= 1
    assert cfg.BRICK_NUM_EPOCHS >= 1
```

- [ ] **Step 2: Run the new tests**

Run: `LEGOGEN_DEV=1 .venv/bin/python -m pytest tests/test_config.py -v`

Expected: all tests pass. If `test_device_branch_with_torch_absent` fails on a machine that has torch installed because Python caches the module: note that `monkeypatch.setitem(sys.modules, "torch", None)` with `None` causes `import torch` to raise `ImportError("import of torch halted; None in sys.modules")` — this is the documented behavior.

- [ ] **Step 3: Commit**

```bash
git add tests/test_config.py
git commit -m "test(config): cover env var, paths, and torch-absent fallback"
```

---

## Task 12: New `test_brick_constants.py`

**Files:**
- Create: `tests/test_brick_constants.py`

- [ ] **Step 1: Create the file**

```python
"""Smoke coverage for backend/brick/constants.py.

The module is mostly data, but the lazy palette facades have custom dunder
methods that must behave like the dict / list they model."""

import pytest

from backend.brick import constants as const
from backend.brick.constants import (
    ALLOWED_DIMS,
    BRICK_SHAPES,
    LDRAW_IDS,
    WORLD_DIM,
)


def test_world_dim_positive():
    assert WORLD_DIM == 20


def test_allowed_dims_match_brick_shapes():
    shape_set = {f"{h}x{w}" for h, w in BRICK_SHAPES}
    assert set(ALLOWED_DIMS) == shape_set


def test_allowed_dims_sorted():
    assert ALLOWED_DIMS == sorted(ALLOWED_DIMS)


def test_ldraw_ids_cover_canonical_shapes():
    # LDRAW_IDS only stores h<=w form; spot-check a few.
    assert LDRAW_IDS[(2, 4)] == "3001"
    assert LDRAW_IDS[(1, 1)] == "3005"


def test_color_palette_lazy_mapping_behaves_like_dict(seeded_palette):
    from backend.brick.constants import COLOR_PALETTE
    assert "C91A09" in COLOR_PALETTE
    assert COLOR_PALETTE["C91A09"] == "Red"
    assert COLOR_PALETTE.get("C91A09") == "Red"
    assert COLOR_PALETTE.get("missing", "fallback") == "fallback"
    assert len(COLOR_PALETTE) == len(seeded_palette)
    assert set(COLOR_PALETTE.keys()) == set(seeded_palette.keys())
    assert set(COLOR_PALETTE.values()) == set(seeded_palette.values())
    assert dict(COLOR_PALETTE.items()) == seeded_palette
    # Iteration works.
    keys = list(COLOR_PALETTE)
    assert set(keys) == set(seeded_palette)


def test_allowed_colors_list_facade(seeded_palette):
    from backend.brick.constants import ALLOWED_COLORS
    assert len(ALLOWED_COLORS) == len(seeded_palette)
    assert "C91A09" in ALLOWED_COLORS
    # Sorted for deterministic grammar regex generation.
    assert list(ALLOWED_COLORS) == sorted(seeded_palette.keys())
    # Indexing works.
    first = ALLOWED_COLORS[0]
    assert first == sorted(seeded_palette.keys())[0]


def test_lazy_palette_caches(seeded_palette):
    """Second call must not reload — cache is stored on the function itself."""
    from backend.brick.constants import _lazy_palette
    a = _lazy_palette()
    b = _lazy_palette()
    assert a is b


def test_load_color_palette_reads_json(tmp_path):
    """Exercise the real file-read path with a synthetic colors.json."""
    import json
    colors_file = tmp_path / "colors.json"
    colors_file.write_text(json.dumps({
        "1": {"rgb": "c91a09", "name": "Red", "is_trans": False},
        "2": {"rgb": "0055BF", "name": "Blue", "is_trans": False},
        # Transparent entries are filtered.
        "3": {"rgb": "FFFFFF", "name": "Glass", "is_trans": True},
        # Unknown-prefixed names are filtered.
        "4": {"rgb": "ABCDEF", "name": "[Unknown] foo", "is_trans": False},
        # Missing/short rgb is filtered.
        "5": {"rgb": "ab", "name": "Short", "is_trans": False},
    }))
    palette = const._load_color_palette(str(colors_file))
    assert palette == {"C91A09": "Red", "0055BF": "Blue"}
```

- [ ] **Step 2: Run the new tests**

Run: `LEGOGEN_DEV=1 .venv/bin/python -m pytest tests/test_brick_constants.py -v`

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_brick_constants.py
git commit -m "test(constants): cover brick shapes, LDraw IDs, lazy palette facades"
```

---

## Task 13: New `test_stage1_pipeline.py`

**Files:**
- Create: `tests/test_stage1_pipeline.py`

The real `Stage1Pipeline` class is pragma'd (loads Qwen3.5-9B on GPU). Only the `_MockStage1` in `brick_pipeline.py` is exercised; nothing in `stage1_pipeline.py` itself is testable without weights + torch. Since the file is entirely in the coverage `omit` list (see Task 1's pyproject.toml), this test file is thin.

- [ ] **Step 1: Create the file**

```python
"""Smoke test for Stage 1 inference.

The real Stage1Pipeline (transformers loader) is omitted from coverage; this
file only exercises the dev-mode _MockStage1 shim that get_stage1_pipeline()
returns under LEGOGEN_DEV=1."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.inference.brick_pipeline import _MockStage1


def test_mock_stage1_describe_returns_string():
    mock = _MockStage1()
    caption = mock.describe(object())
    assert isinstance(caption, str)
    assert len(caption) > 0


def test_mock_stage1_describe_is_stable():
    """Callers rely on the caption being deterministic for dev-mode testing."""
    assert _MockStage1().describe(None) == _MockStage1().describe(None)
```

- [ ] **Step 2: Run the test**

Run: `LEGOGEN_DEV=1 .venv/bin/python -m pytest tests/test_stage1_pipeline.py -v`

Expected: both tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_stage1_pipeline.py
git commit -m "test(stage1): cover the _MockStage1 dev shim"
```

---

## Task 14: New `test_app.py` — health + CORS

**Files:**
- Create: `tests/test_app.py`

- [ ] **Step 1: Create the file**

```python
"""Coverage for backend/app.py — health route, CORS env-var parsing, lifespan (dev mode only)."""

import importlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

httpx = pytest.importorskip("httpx")
fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient


def _reload_app():
    """Reload backend.app so CORS env-var parsing re-runs at import time."""
    import backend.app
    importlib.reload(backend.app)
    return backend.app.app


@pytest.fixture(autouse=True)
def ensure_dev_mode(monkeypatch):
    monkeypatch.setenv("LEGOGEN_DEV", "1")
    yield


def test_health_endpoint(gallery_db_path):
    app = _reload_app()
    with TestClient(app) as c:
        resp = c.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_default_cors_origins(monkeypatch, gallery_db_path):
    monkeypatch.delenv("LEGOGEN_CORS_ORIGINS", raising=False)
    app = _reload_app()
    cors = next(m for m in app.user_middleware if "CORS" in type(m.cls).__name__ or "CORS" in m.cls.__name__)
    origins = cors.kwargs["allow_origins"]
    assert origins == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


def test_custom_cors_origins_env(monkeypatch, gallery_db_path):
    monkeypatch.setenv("LEGOGEN_CORS_ORIGINS", "https://a.example,https://b.example")
    app = _reload_app()
    cors = next(m for m in app.user_middleware if "CORS" in type(m.cls).__name__ or "CORS" in m.cls.__name__)
    origins = cors.kwargs["allow_origins"]
    assert origins == ["https://a.example", "https://b.example"]


def test_cors_origins_ignores_empty_entries(monkeypatch, gallery_db_path):
    monkeypatch.setenv("LEGOGEN_CORS_ORIGINS", "https://a.example, ,https://b.example")
    app = _reload_app()
    cors = next(m for m in app.user_middleware if "CORS" in type(m.cls).__name__ or "CORS" in m.cls.__name__)
    origins = cors.kwargs["allow_origins"]
    assert origins == ["https://a.example", "https://b.example"]


def test_lifespan_dev_mode_runs(capsys, gallery_db_path):
    """Startup in dev mode prints the ready banner and does not preload models."""
    app = _reload_app()
    with TestClient(app):
        pass
    out = capsys.readouterr().out
    assert "dev mode" in out.lower()
    assert "preloading" not in out.lower()
```

- [ ] **Step 2: Run the new tests**

Run: `LEGOGEN_DEV=1 .venv/bin/python -m pytest tests/test_app.py -v`

Expected: all tests pass. If `test_default_cors_origins` fails because `CORSMiddleware` isn't the last middleware in the list, replace the `next(...)` expression with:

```python
    from fastapi.middleware.cors import CORSMiddleware
    cors = next(m for m in app.user_middleware if m.cls is CORSMiddleware)
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_app.py
git commit -m "test(app): cover /health, CORS env parsing, and dev-mode lifespan"
```

---

## Task 15: Run full backend suite and verify 100%

- [ ] **Step 1: Run everything**

Run: `LEGOGEN_DEV=1 .venv/bin/python -m pytest --cov-fail-under=100 -q`

Expected: all tests pass. The coverage table at the end should show `TOTAL ... 100%`. The `addopts` in `pyproject.toml` auto-enables the coverage run; the explicit `--cov-fail-under=100` fails the command if coverage is below 100. (The threshold is not baked into `[tool.coverage.report]` so intermediate runs during Tasks 2–14 still exit 0 — CI enforces the gate via the same flag in `.github/workflows/tests.yml`.)

- [ ] **Step 2: If any file is below 100%, add targeted tests or pragma**

For each file listed below 100%: read the "missing" line numbers from the coverage report. Either write a focused test for the missing branch, or add `# pragma: no cover` with a one-line comment explaining why it's untestable. The rule is the same as in the spec: only pragma loader/CLI/network code.

- [ ] **Step 3: Commit if any additional changes were needed**

```bash
git add -A tests/ backend/
git commit -m "test: close coverage gaps uncovered by the baseline run"
```

(Skip this step if Step 1 already passed at 100%.)

---

## Task 16: Frontend API seam — `legogen.test.ts` parsing tests

**Files:**
- Create: `frontend/src/api/legogen.test.ts`

- [ ] **Step 1: Write the failing tests**

```typescript
// frontend/src/api/legogen.test.ts
import { describe, it, expect } from 'vitest';
import { parseBrickString, bricksToLayers } from './legogen';

describe('parseBrickString', () => {
  it('returns [] for empty input', () => {
    expect(parseBrickString('')).toEqual([]);
    expect(parseBrickString('   \n  ')).toEqual([]);
  });

  it('parses a single brick line', () => {
    const out = parseBrickString('2x4 (0,0,0) #C91A09');
    expect(out).toEqual([
      { h: 2, w: 4, x: 0, y: 0, z: 0, color: '#C91A09' },
    ]);
  });

  it('parses multiple brick lines', () => {
    const raw = '2x4 (0,0,0) #C91A09\n1x1 (5,5,2) #FFFFFF';
    const out = parseBrickString(raw);
    expect(out).toHaveLength(2);
    expect(out[1]).toEqual({ h: 1, w: 1, x: 5, y: 5, z: 2, color: '#FFFFFF' });
  });

  it('skips lines that do not match the grammar', () => {
    const raw = '2x4 (0,0,0) #C91A09\nnonsense\n1x1 (1,1,1) #000000';
    const out = parseBrickString(raw);
    expect(out).toHaveLength(2);
    expect(out.map(b => b.color)).toEqual(['#C91A09', '#000000']);
  });

  it('tolerates surrounding whitespace per line', () => {
    const raw = '   2x4 (0,0,0) #C91A09   ';
    expect(parseBrickString(raw)).toHaveLength(1);
  });
});

describe('bricksToLayers', () => {
  it('groups by z and orders ascending', () => {
    const bricks = [
      { h: 2, w: 4, x: 0, y: 0, z: 1, color: '#A00000' },
      { h: 2, w: 4, x: 0, y: 0, z: 0, color: '#A00000' },
      { h: 1, w: 1, x: 5, y: 5, z: 1, color: '#A00000' },
    ];
    const { steps, zLevels } = bricksToLayers(bricks);
    expect(zLevels).toEqual([0, 1]);
    expect(steps).toHaveLength(2);
    expect(steps[0].step_number).toBe(1);
    expect(steps[0].z).toBe(0);
    expect(steps[0].brick_count).toBe(1);
    expect(steps[1].brick_count).toBe(2);
  });

  it('tallies by (dims, color) within a layer', () => {
    const bricks = [
      { h: 2, w: 4, x: 0, y: 0, z: 0, color: '#A00000' },
      { h: 2, w: 4, x: 2, y: 0, z: 0, color: '#A00000' },
      { h: 1, w: 1, x: 5, y: 5, z: 0, color: '#FFFFFF' },
    ];
    const { steps } = bricksToLayers(bricks);
    const tally = steps[0].tally;
    expect(tally).toHaveLength(2);
    expect(tally[0]).toEqual({ dims: '2x4', color: '#A00000', count: 2 });
    expect(tally[1]).toEqual({ dims: '1x1', color: '#FFFFFF', count: 1 });
  });

  it('returns empty steps/zLevels for empty input', () => {
    const { steps, zLevels } = bricksToLayers([]);
    expect(steps).toEqual([]);
    expect(zLevels).toEqual([]);
  });
});
```

- [ ] **Step 2: Run to verify tests pass**

Run: `cd frontend && npm run test -- legogen.test.ts`

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/api/legogen.test.ts
git commit -m "test(frontend): cover parseBrickString and bricksToLayers"
```

---

## Task 17: `legogen.test.ts` — fetch-based clients

**Files:**
- Modify: `frontend/src/api/legogen.test.ts`

- [ ] **Step 1: Append fetch tests**

Append to `frontend/src/api/legogen.test.ts`:

```typescript
import { afterEach, beforeEach, vi } from 'vitest';
import {
  generateBricks,
  listGalleryBuilds,
  createGalleryBuild,
  getGalleryBuild,
  starGalleryBuild,
} from './legogen';

function mockFetchResponse(body: unknown, init: Partial<ResponseInit> = {}): Response {
  return new Response(JSON.stringify(body), {
    status: init.status ?? 200,
    headers: { 'Content-Type': 'application/json', ...(init.headers ?? {}) },
  });
}

describe('generateBricks', () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal('fetch', fetchMock);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('posts multipart form with image, prompt, n', async () => {
    fetchMock.mockResolvedValueOnce(mockFetchResponse({
      bricks: '2x4 (0,0,0) #C91A09',
      brick_count: 1,
      stable: true,
      metadata: { model_version: 'mock', generation_time_ms: 1, rejections: 0, rollbacks: 0 },
    }));
    const file = new File([new Uint8Array([1, 2])], 'x.png', { type: 'image/png' });
    const res = await generateBricks(file, 'hi', 2);
    expect(res.brick_count).toBe(1);
    const [, init] = fetchMock.mock.calls[0];
    expect(init.method).toBe('POST');
    const body = init.body as FormData;
    expect(body.get('image')).toBeInstanceOf(File);
    expect(body.get('prompt')).toBe('hi');
    expect(body.get('n')).toBe('2');
  });

  it('throws with server detail on non-ok response', async () => {
    fetchMock.mockResolvedValueOnce(mockFetchResponse({ detail: 'nope' }, { status: 500 }));
    await expect(generateBricks(undefined, 'hi')).rejects.toThrow('nope');
  });

  it('throws with generic HTTP message when body lacks detail', async () => {
    fetchMock.mockResolvedValueOnce(new Response('not json', { status: 502 }));
    await expect(generateBricks(undefined, 'hi')).rejects.toThrow('HTTP 502');
  });

  it('omits n when undefined', async () => {
    fetchMock.mockResolvedValueOnce(mockFetchResponse({
      bricks: '', brick_count: 0, stable: true,
      metadata: { model_version: 'm', generation_time_ms: 0, rejections: 0, rollbacks: 0 },
    }));
    await generateBricks(undefined, 'hi');
    const body = fetchMock.mock.calls[0][1].body as FormData;
    expect(body.get('n')).toBe(null);
  });
});

describe('Gallery client', () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal('fetch', fetchMock);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('listGalleryBuilds adds query string when params provided', async () => {
    fetchMock.mockResolvedValueOnce(mockFetchResponse([]));
    await listGalleryBuilds({ sort: 'bricks', q: 'cottage' });
    const [url] = fetchMock.mock.calls[0];
    expect(url).toContain('sort=bricks');
    expect(url).toContain('q=cottage');
  });

  it('listGalleryBuilds without params omits query string', async () => {
    fetchMock.mockResolvedValueOnce(mockFetchResponse([]));
    await listGalleryBuilds();
    const [url] = fetchMock.mock.calls[0];
    expect(url).not.toContain('?');
  });

  it('listGalleryBuilds throws on HTTP error', async () => {
    fetchMock.mockResolvedValueOnce(new Response('', { status: 500 }));
    await expect(listGalleryBuilds()).rejects.toThrow('HTTP 500');
  });

  it('createGalleryBuild POSTs JSON body', async () => {
    const build = { id: '1', title: 't', caption: '', bricks: 'x', brick_count: 1, stable: true, thumbnail_b64: '', stars: 0, star_count: 0, created_at: '2026-04-17' };
    fetchMock.mockResolvedValueOnce(mockFetchResponse(build));
    const payload = { title: 't', caption: '', bricks: 'x', brick_count: 1, stable: true };
    const out = await createGalleryBuild(payload);
    expect(out.id).toBe('1');
    const [, init] = fetchMock.mock.calls[0];
    expect(init.method).toBe('POST');
    expect(init.headers['Content-Type']).toBe('application/json');
    expect(JSON.parse(init.body as string)).toEqual(payload);
  });

  it('createGalleryBuild throws with server detail on error', async () => {
    fetchMock.mockResolvedValueOnce(mockFetchResponse({ detail: 'bad title' }, { status: 400 }));
    await expect(createGalleryBuild({
      title: '', caption: '', bricks: 'x', brick_count: 0, stable: true,
    })).rejects.toThrow('bad title');
  });

  it('createGalleryBuild falls back to HTTP status when body lacks detail', async () => {
    fetchMock.mockResolvedValueOnce(new Response('not json', { status: 418 }));
    await expect(createGalleryBuild({
      title: 't', caption: '', bricks: 'x', brick_count: 0, stable: true,
    })).rejects.toThrow('HTTP 418');
  });

  it('getGalleryBuild throws on 404', async () => {
    fetchMock.mockResolvedValueOnce(new Response('', { status: 404 }));
    await expect(getGalleryBuild('missing')).rejects.toThrow('HTTP 404');
  });

  it('starGalleryBuild PATCHes with stars payload', async () => {
    const build = { id: '1', title: 't', caption: '', bricks: '', brick_count: 0, stable: true, thumbnail_b64: '', stars: 4.5, star_count: 2, created_at: '2026-04-17' };
    fetchMock.mockResolvedValueOnce(mockFetchResponse(build));
    const out = await starGalleryBuild('1', 5);
    expect(out.stars).toBe(4.5);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toContain('/api/gallery/1/star');
    expect(init.method).toBe('PATCH');
    expect(JSON.parse(init.body as string)).toEqual({ stars: 5 });
  });

  it('starGalleryBuild throws on HTTP error', async () => {
    fetchMock.mockResolvedValueOnce(new Response('', { status: 500 }));
    await expect(starGalleryBuild('x', 3)).rejects.toThrow('HTTP 500');
  });
});
```

- [ ] **Step 2: Run the tests**

Run: `cd frontend && npm run test -- legogen.test.ts`

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/api/legogen.test.ts
git commit -m "test(frontend): cover generateBricks and gallery fetch clients"
```

---

## Task 18: `legogen.test.ts` — SSE streaming client

**Files:**
- Modify: `frontend/src/api/legogen.test.ts`

- [ ] **Step 1: Append SSE tests**

Append to `frontend/src/api/legogen.test.ts`:

```typescript
import { generateBricksStream } from './legogen';

function sseResponse(chunks: string[]): Response {
  const stream = new ReadableStream({
    start(controller) {
      const enc = new TextEncoder();
      for (const c of chunks) controller.enqueue(enc.encode(c));
      controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  });
}

describe('generateBricksStream', () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal('fetch', fetchMock);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('parses progress, brick, rollback, sample, result events', async () => {
    const result = {
      bricks: '2x4 (0,0,0) #C91A09',
      brick_count: 1,
      stable: true,
      metadata: { model_version: 'mock', generation_time_ms: 1, rejections: 0, rollbacks: 0 },
    };
    fetchMock.mockResolvedValueOnce(sseResponse([
      'event: progress\ndata: {"stage":"stage1","message":"go","caption":"c"}\n\n',
      'event: brick\ndata: {"count":1}\n\n',
      'event: rollback\ndata: {"count":1}\n\n',
      'event: sample\ndata: {"index":1,"of":2,"stable":true}\n\n',
      'event: result\ndata: ' + JSON.stringify(result) + '\n\n',
    ]));
    const events: any[] = [];
    const out = await generateBricksStream({
      prompt: 'x',
      onEvent: e => events.push(e),
    });
    expect(events.map(e => e.type)).toEqual(['progress', 'brick', 'rollback', 'sample']);
    expect(out).toEqual(result);
  });

  it('throws when server emits an error event', async () => {
    fetchMock.mockResolvedValueOnce(sseResponse([
      'event: error\ndata: {"detail":"boom"}\n\n',
    ]));
    await expect(generateBricksStream({
      prompt: 'x',
      onEvent: () => {},
    })).rejects.toThrow('boom');
  });

  it('throws when stream ends without result', async () => {
    fetchMock.mockResolvedValueOnce(sseResponse([
      'event: progress\ndata: {"stage":"stage1","message":"..."}\n\n',
    ]));
    await expect(generateBricksStream({
      prompt: 'x',
      onEvent: () => {},
    })).rejects.toThrow('Stream ended without result');
  });

  it('throws with server detail on non-ok response', async () => {
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ detail: 'bad' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    }));
    await expect(generateBricksStream({
      prompt: 'x',
      onEvent: () => {},
    })).rejects.toThrow('bad');
  });

  it('throws HTTP message when non-ok body lacks detail', async () => {
    fetchMock.mockResolvedValueOnce(new Response('not json', { status: 502 }));
    await expect(generateBricksStream({
      prompt: 'x',
      onEvent: () => {},
    })).rejects.toThrow('HTTP 502');
  });

  it('throws when response has no body', async () => {
    const noBody = new Response(null, { status: 200 });
    Object.defineProperty(noBody, 'body', { value: null });
    fetchMock.mockResolvedValueOnce(noBody);
    await expect(generateBricksStream({
      prompt: 'x',
      onEvent: () => {},
    })).rejects.toThrow('No response body');
  });

  it('ignores malformed event chunks', async () => {
    const result = {
      bricks: '', brick_count: 0, stable: true,
      metadata: { model_version: 'm', generation_time_ms: 0, rejections: 0, rollbacks: 0 },
    };
    fetchMock.mockResolvedValueOnce(sseResponse([
      'event: progress\ndata: this-is-not-json\n\n',  // JSON parse error — skipped
      ':\n\n',                                         // comment-only, no event/data
      'event: result\ndata: ' + JSON.stringify(result) + '\n\n',
    ]));
    const events: any[] = [];
    const out = await generateBricksStream({
      prompt: 'x',
      onEvent: e => events.push(e),
    });
    expect(events).toEqual([]);
    expect(out).toEqual(result);
  });

  it('forwards multipart fields (image, prompt, n)', async () => {
    const result = {
      bricks: '', brick_count: 0, stable: true,
      metadata: { model_version: 'm', generation_time_ms: 0, rejections: 0, rollbacks: 0 },
    };
    fetchMock.mockResolvedValueOnce(sseResponse([
      'event: result\ndata: ' + JSON.stringify(result) + '\n\n',
    ]));
    const file = new File([new Uint8Array([1])], 'x.png', { type: 'image/png' });
    await generateBricksStream({
      image: file, prompt: 'hi', n: 3, onEvent: () => {},
    });
    const [, init] = fetchMock.mock.calls[0];
    const body = init.body as FormData;
    expect(body.get('image')).toBeInstanceOf(File);
    expect(body.get('prompt')).toBe('hi');
    expect(body.get('n')).toBe('3');
  });

  it('caller AbortSignal aborts the fetch', async () => {
    const ctrl = new AbortController();
    // fetch that rejects when its signal aborts — matches real fetch.
    fetchMock.mockImplementationOnce((_url, init) => {
      return new Promise((_, reject) => {
        init.signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
      });
    });
    const promise = generateBricksStream({
      prompt: 'x', onEvent: () => {}, signal: ctrl.signal,
    });
    ctrl.abort();
    await expect(promise).rejects.toThrow(/abort/i);
  });

  it('timeout aborts the fetch', async () => {
    fetchMock.mockImplementationOnce((_url, init) => {
      return new Promise((_, reject) => {
        init.signal.addEventListener('abort', () => reject(new DOMException('timeout', 'AbortError')));
      });
    });
    await expect(generateBricksStream({
      prompt: 'x', onEvent: () => {}, timeoutMs: 5,
    })).rejects.toThrow(/abort|timeout/i);
  });
});
```

- [ ] **Step 2: Run the tests**

Run: `cd frontend && npm run test -- legogen.test.ts`

Expected: all SSE tests pass.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/api/legogen.test.ts
git commit -m "test(frontend): cover SSE stream parsing, abort, and timeout paths"
```

---

## Task 19: Verify frontend seam at 100%

- [ ] **Step 1: Run coverage**

Run: `cd frontend && npm run test:coverage`

Expected: the coverage table shows 100% across lines, branches, functions, statements for `src/api/legogen.ts`. Exit code 0 because thresholds are met.

- [ ] **Step 2: If below 100%, add a targeted test or pragma**

Vitest/v8 doesn't natively support `/* istanbul ignore next */` style pragmas — if a truly untestable branch appears, lower the threshold for that specific file with `coverage.thresholds["src/api/legogen.ts"]` rather than globally. The spec target is 100%; only deviate after discussion.

- [ ] **Step 3: Commit if further changes were made**

(Skip if Step 1 already passed.)

---

## Task 20: CI gate — `.github/workflows/tests.yml`

**Files:**
- Create: `.github/workflows/tests.yml`

- [ ] **Step 1: Create the workflow**

```yaml
name: tests

on:
  pull_request:
  push:
    branches: [main]

jobs:
  python:
    name: Python tests + coverage
    runs-on: ubuntu-latest
    env:
      LEGOGEN_DEV: "1"
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
          cache-dependency-path: requirements-dev.txt

      - name: Install test dependencies
        run: python -m pip install -r requirements-dev.txt

      - name: Run pytest with coverage
        run: python -m pytest --cov-fail-under=100 -q

      - name: Upload coverage report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: python-coverage
          path: |
            .coverage
            coverage.xml
          if-no-files-found: ignore

  frontend:
    name: Frontend seam tests + coverage
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: frontend
    steps:
      - uses: actions/checkout@v4

      - name: Set up Node 20
        uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: npm
          cache-dependency-path: frontend/package-lock.json

      - name: Install
        run: npm ci

      - name: Run Vitest with coverage
        run: npm run test:coverage

      - name: Upload coverage report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: frontend-coverage
          path: frontend/coverage
          if-no-files-found: ignore
```

- [ ] **Step 2: Run both commands locally once more**

Run: `LEGOGEN_DEV=1 .venv/bin/python -m pytest -q && cd frontend && npm run test:coverage`

Expected: both commands exit 0 at 100% coverage. This confirms the workflow will pass once pushed.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/tests.yml
git commit -m "ci: add tests workflow that enforces 100% coverage on backend + frontend seam"
```

---

## Task 21: Docs cleanup and pre-existing-skip removal

**Files:**
- Modify: `CLAUDE.md`
- Modify: `.claude/settings.local.json` (already touched in Task 4 Step 3, but verify)

- [ ] **Step 1: Update `CLAUDE.md`**

Find the "Known pre-existing test failure" paragraph in `CLAUDE.md`:

```markdown
**Known pre-existing test failure**: `tests/test_brick_decoder.py::test_allowed_colors_populated` fails because `colors.json` isn't checked in. It's in the explicit skip list in `.claude/settings.local.json` — ignore it when running the suite.
```

Replace with:

```markdown
## Coverage

`pyproject.toml` wires pytest-cov so `LEGOGEN_DEV=1 .venv/bin/python -m pytest` measures coverage on every run. The 100% gate lives in CI (`.github/workflows/tests.yml` passes `--cov-fail-under=100` to pytest); to reproduce the gate locally, run `LEGOGEN_DEV=1 .venv/bin/python -m pytest --cov-fail-under=100 -q`. `# pragma: no cover` is grep-able — add one only for real model-loading or CLI orchestration code, with a one-line reason comment above it.

Frontend coverage is scoped to `frontend/src/api/**` only. Run it with `cd frontend && npm run test:coverage` — v8 reporter, 100% threshold.

`.github/workflows/tests.yml` runs both gates on every PR to `main`.
```

- [ ] **Step 2: Confirm `.claude/settings.local.json` no longer references `colors.json`**

Run: `grep -n "colors.json\|test_allowed_colors_populated" .claude/settings.local.json`

Expected: no output (both entries removed in Task 4 Step 3).

- [ ] **Step 3: Final suite run**

Run: `LEGOGEN_DEV=1 .venv/bin/python -m pytest -q && cd frontend && npm run test:coverage`

Expected: both green at 100%.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md .claude/settings.local.json
git commit -m "docs: document coverage gate, drop pre-existing-skip from CLAUDE.md"
```

---

## Spec-to-Plan Self-Review

- Spec "Tooling" → Tasks 1 + 2.
- Spec "Module inventory" exclusions → Task 3 (pragma pass) + Task 1 (omit list in pyproject).
- Spec "Test fixtures" → Task 5.
- Spec "Test backfill" table → Tasks 4, 6, 7, 8, 9, 10, 11, 12, 13, 14 cover every row.
- Spec "CI workflow" → Task 20.
- Spec "Rollout order" 1-6 → Tasks 1/2 (step 1), Task 3 (step 2), Tasks 4-15 (step 3), Tasks 16-19 (step 4), Task 20 (step 5), Task 21 (step 6).
- Spec "Testing this design's implementation" verifications → Tasks 15, 19, 20 Step 2, 21 Step 2-3.

No placeholders. No `TBD`/`TODO`/"implement later" strings. Function and property names (e.g., `parseBrickString`, `bricksToLayers`, `MockBrickPipeline`, `_MockStage1`, `_build_logits_processor`, `rank_candidates`, `cluster_and_pick`) are consistent between plan tasks and source.
