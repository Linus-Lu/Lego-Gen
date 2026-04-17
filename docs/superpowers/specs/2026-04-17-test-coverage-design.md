# 100% Test Coverage with CI Gate — Design

**Date:** 2026-04-17
**Owner:** linuslu
**Status:** Approved — ready for implementation planning

## Problem

The backend has 87 tests across 10 files but no coverage measurement, no CI
test gate, and a pre-existing skipped test (`test_allowed_colors_populated`)
living in the permission allowlist. The frontend has no test infrastructure
at all. Any regression in `backend/api`, `backend/inference/brick_pipeline`,
or `frontend/src/api/legogen.ts` ships unless caught by manual QA.

Goal: lock in the tested surface at 100% coverage, backfill the gaps on
code where the number is meaningful, and make CI fail on any drop.

## Scope decisions (pinned during brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Workflow | Measure + backfill + CI gate (option D) | Retroactive cleanup plus forward protection. |
| Surface | Python + thin frontend seam (option C) | RTL tests over R3F/Three are brittle and low-value. Playwright (future) is the right tool for UI confidence; Vitest covers pure-logic `src/api/legogen.ts`. |
| Target | Pragmatic 100% with `# pragma: no cover` on real model-loading code (option A) | Strict 100% everywhere requires mocking torch/transformers internals — assertions on library internals catch zero bugs and rot with version bumps. |
| CI gate | Hard 100% backend + frontend seam, pragma is the pressure valve (option B) | `# pragma: no cover` is grep-able and reviewable in PR; identical in effect to "zero tolerance" but without the cultural friction. |

## Architecture

### Tooling

**Python**
- New file: `requirements-dev.txt` — `pytest`, `pytest-asyncio`, `pytest-cov`, `coverage[toml]`, `httpx`, plus the minimum subset of prod deps actually imported in dev-mode test paths (`fastapi`, `pydantic`, `aiosqlite`, `Pillow`, `numpy`, `scipy`, `scikit-learn`). No torch, transformers, peft, outlines, or bitsandbytes.
- New file: `pyproject.toml` — contains only `[tool.coverage.*]` sections; no project metadata, no other tool config. CLAUDE.md explicitly notes the absence of these files; we keep the footprint minimal.
- Coverage settings: `branch = true`, `source = ["backend"]`, `fail_under = 100`, explicit `omit` list for training scripts and dataset build driver.
- Invocation unchanged: `LEGOGEN_DEV=1 .venv/bin/python -m pytest -q`. Coverage auto-applies via pyproject.

**Frontend**
- Add to `frontend/package.json` dev deps: `vitest`, `@vitest/coverage-v8`.
- New file: `frontend/vitest.config.ts` — `coverage.include = ["src/api/**"]`, `coverage.thresholds.lines = 100`, `coverage.thresholds.branches = 100`.
- New scripts: `npm run test`, `npm run test:coverage`.

### Module inventory

**Covered and already solid (keep):**
`backend/brick/parser.py`, `backend/brick/occupancy.py`, `backend/brick/stability.py`,
`backend/storage/gallery_db.py`, `backend/inference/best_of_n.py`,
`backend/data_pipeline/prepare_brick_dataset.py`,
`backend/data_pipeline/build_stage1_dataset.py` (caption-matching helpers only; see exclusions for the download/orchestration functions).

**Covered but with gaps — need additions:**
- `backend/api/routes_generate.py` — missing 413/400/504 paths, SSE ordering, BoN fan-out, client-disconnect.
- `backend/api/routes_gallery.py` — missing validation-failure paths.
- `backend/inference/brick_pipeline.py` — `MockBrickPipeline` paths untested end-to-end.
- `backend/brick/decoder.py` — existing test depends on missing `colors.json`; fix + expand.
- `backend/brick/constants.py`, `backend/config.py` — new smoke tests.

**Excluded via `# pragma: no cover` or coverage omit:**
- `backend/app.py` — real-model-preload branch of `lifespan()` + `_preload_models()`.
- `backend/inference/brick_pipeline.py` — real `BrickPipeline` class body; non-dev branch of `get_brick_pipeline()` and `_get_stage1_pipeline()`.
- `backend/inference/stage1_pipeline.py` — entire `Stage1Pipeline` class (pure torch/transformers loader).
- `backend/training/train_stage1.py`, `backend/training/train_brick.py`, `backend/training/utils.py` — omit in coverage config.
- `backend/data_pipeline/dataset_stage1.py` — omit (torch-dependent `Stage1Dataset` / `Stage1Collator`; torch is not installed in CI).
- `backend/data_pipeline/build_stage1_dataset.py` — pragma the COCO download + file-system-orchestration functions; keep the caption-matching helpers (`match_coco_to_st2b`, `generate_description_from_label`, `load_st2b_captions_by_category`) in coverage since `test_stage1_dataset.py` already covers them.
- `__main__` guards and CLI `argparse` entry points — one-line pragma.

### Test fixtures

`tests/conftest.py` gains:
- `tmp_gallery_db` (session scope) — points `gallery_db.DB_PATH` at a temp file, restores after.
- `mock_brick_pipeline` (function scope) — resets `_brick_instance` / `_stage1_instance` singletons between tests so factory-branch tests are independent.
- `httpx_app_client` (function scope) — yields `AsyncClient(transport=ASGITransport(app=app))` for route tests.

No network, no GPU, no filesystem writes outside `tmp_path`. Target wall-clock under 30 s on a laptop.

### Test backfill — new/expanded files

| File | New/Expand | Purpose |
|---|---|---|
| `tests/test_api_routes.py` | expand | 413/400/504, SSE ordering, BoN fan-out, client disconnect |
| `tests/test_gallery_routes.py` | new | Route-level validation failures (400/404/422) |
| `tests/test_brick_decoder.py` | fix + expand | Remove `colors.json` dependency; cover all state transitions |
| `tests/test_brick_pipeline_mock.py` | new | `MockBrickPipeline` methods end-to-end, factory singletons |
| `tests/test_brick_pipeline_logic.py` | expand | Longest-first grammar ordering, outlines-absent branch, BoN `strategy="rank"` |
| `tests/test_best_of_n.py` | expand | Zero-stable fallback, empty-bricks structural features |
| `tests/test_config.py` | new | Import success, torch-absent branch via reload |
| `tests/test_brick_constants.py` | new | `BRICK_SHAPES` / `ALLOWED_DIMS` / `WORLD_DIM` consistency |
| `tests/test_stage1_pipeline.py` | new | `_MockStage1` happy path only |
| `tests/test_app.py` | new | `GET /health`, CORS env-var parsing |
| `frontend/src/api/legogen.test.ts` | new | `parseBrickString`, `bricksToLayers`, fetch + SSE clients, gallery client |

### CI workflow

New file: `.github/workflows/tests.yml`. Triggers on `pull_request` and `push` to `main`. Two parallel jobs:

**`python` job**
- `ubuntu-latest`, Python 3.11.
- `pip install -r requirements-dev.txt`.
- `LEGOGEN_DEV=1 pytest -q` — coverage auto-enforced via pyproject `fail_under = 100`.
- Upload `coverage.xml` as artifact.

**`frontend` job**
- `ubuntu-latest`, Node 20.
- `cd frontend && npm ci && npm run test:coverage`.
- Vitest threshold enforces 100% on `src/api/**`.

Existing `claude-code-review.yml` and `claude.yml` are untouched — `tests.yml` runs alongside.

## Error handling & edge cases

- `LEGOGEN_DEV=1` is set in CI (mirrors the conftest convention) so no real model-loader ever runs.
- Pragma audit: every `# pragma: no cover` must have a one-line comment above it explaining why (loader code, CLI entry, etc.). Reviewers can grep for the string.
- The `tests/test_brick_decoder.py::test_allowed_colors_populated` fix removes its entry from `.claude/settings.local.json`'s deselect list in the same PR — leaving a stale deselect invites confusion.
- SSE client-disconnect test needs care: uses `AsyncClient.stream()` with an early break + `gc.collect()` to force the `asyncio.CancelledError` path deterministically.
- `test_config.py`'s torch-absent reload uses `importlib.reload` with `monkeypatch.setitem(sys.modules, "torch", None)` so it doesn't pollute other tests.

## Rollout order

Six landable chunks, each independently mergeable:

1. **Tooling scaffold** — `requirements-dev.txt`, `pyproject.toml`, `vitest.config.ts`, frontend dev deps. No gate yet. Verifies baseline install.
2. **Baseline + pragma audit** — run `coverage run`, annotate exclusions, commit the current number.
3. **Backend gap-fill** — all new Python tests. Backend hits 100%.
4. **Frontend seam** — Vitest setup + `legogen.test.ts`. `src/api/legogen.ts` at 100%.
5. **CI gate** — `tests.yml` with `fail_under=100` and Vitest thresholds.
6. **Docs + skip cleanup** — update `CLAUDE.md` (remove pre-existing-failure note, add coverage section), remove `colors.json` deselect from settings.

## Testing this design's implementation

- After step 3: `LEGOGEN_DEV=1 .venv/bin/python -m pytest --cov=backend --cov-report=term-missing -q` reports 100%.
- After step 4: `cd frontend && npm run test:coverage` reports 100% on `src/api/legogen.ts`.
- After step 5: a PR that deletes one test assertion fails CI.
- After step 6: `grep "pre-existing test failure" CLAUDE.md` returns no match; `.claude/settings.local.json` has no `colors.json` deselect.

## Out of scope

- RTL or R3F component tests. (Playwright E2E is a future separate design.)
- Mutation testing. (Low ROI at this team size.)
- Training-script tests. (Omitted by decision; see scope table.)
- A dev-dependency pin/lock file. (Pip-tools / uv-lock is a future call.)
