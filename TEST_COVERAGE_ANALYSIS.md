# Test Coverage Analysis — LEGOGen

## Current State

The test suite comprises **7 test files** with **~34 test cases**, all using pytest. Tests currently cover the `backend/brick/` package, parts of `backend/data_pipeline/`, and an integration-level pipeline test using the mock. No frontend tests exist.

### What Is Tested

| Module | File | Tests | Coverage |
|--------|------|-------|----------|
| `brick/parser.py` | `test_brick_parser.py` | 7 | Parse, serialize, roundtrip, sequences, error handling |
| `brick/decoder.py` | `test_brick_decoder.py` | 5 | Constraint state machine, allowed dims/colors, token sequencing |
| `brick/occupancy.py` | `test_brick_occupancy.py` | 6 | Voxel placement, collision, bounds, stacking, invalid dims |
| `brick/stability.py` | `test_brick_stability.py` | 6 | Ground contact, floating detection, supported stacking, `find_first_unstable` |
| `data_pipeline/prepare_brick_dataset.py` | `test_prepare_brick_dataset.py` | 5 | ST2B parsing, colorization, training example formatting |
| `data_pipeline/build_stage1_dataset.py` | `test_stage1_dataset.py` | 4 | COCO-to-ST2B matching, caption loading, description generation |
| `inference/pipeline.py` | `test_two_stage_pipeline.py` | 4 | Mock pipeline end-to-end, text/image input, structure validation |

### Existing Test Infrastructure Issues

1. **`test_brick_decoder.py` and `test_brick_occupancy.py` fail to collect** because `backend/brick/constants.py` loads `data/cache/colors.json` at import time. When this file is absent (e.g., in CI or a fresh clone), those tests cannot even be imported.
2. **Two pipeline tests fail** due to missing `PIL` (Pillow) dependency in the test environment.
3. **No `pytest.ini` or `pyproject.toml` `[tool.pytest]` section** — test discovery relies entirely on defaults and `conftest.py`.

---

## Gap Analysis: Where Coverage Is Missing

### Priority 1 — Completely Untested Backend Modules

#### 1. `api/routes_generate.py` — FastAPI Route Handler
**Risk: HIGH** — This is the only HTTP endpoint and the entry point for all user requests.

No tests exist for:
- Successful generation with text prompt
- Successful generation with image upload
- 400 error when neither image nor prompt is provided
- 400 error when an uploaded file is not an image (content type check)
- 400 error when an image cannot be decoded (corrupt file)
- 504 timeout error when inference exceeds `INFERENCE_TIMEOUT_SECONDS`
- Concurrent request handling

**Recommendation:** Add `httpx.AsyncClient` + `pytest-asyncio` tests using FastAPI's `TestClient`. Mock the pipeline to keep tests fast.

#### 2. `models/tokenizer.py` — Prompt Templates & JSON Utilities
**Risk: HIGH** — Prompt construction and JSON parsing are critical to model quality and robustness.

No tests exist for:
- `build_planner_chat_messages()` — message structure and content
- `sample_prompt_template()` — template selection, variable substitution, fallback on `KeyError`
- `build_chat_messages()` — with and without image URL
- `extract_json_from_text()` — valid JSON extraction, nested braces, malformed JSON, no JSON present
- `strip_thinking_blocks()` — removal of `<think>` blocks
- `_try_repair_json()` — JSON repair fallback
- `encode_json_label()` / `decode_and_parse()` — requires a tokenizer mock

**Recommendation:** Most of these are pure functions and easy to test without ML dependencies. `extract_json_from_text` is especially important since it handles model output parsing.

#### 3. `training/utils.py` — Metrics Functions
**Risk: MEDIUM** — Metrics drive evaluation quality and training decisions.

No tests exist for any of these 8 functions:
- `compute_json_validity_rate()` — JSON validity counting
- `compute_field_accuracy()` — field-level exact match
- `compute_color_f1()` — color set F1 score (edge cases: empty lists, identical lists, disjoint lists)
- `compute_parts_f1()` — part ID F1 with quantities
- `compute_all_metrics()` — aggregation of all metrics
- `compute_structural_coherence()` — ordering and connectivity validation
- `compute_part_realism()` — part ID validation against catalog
- `compute_build_feasibility()` — parts count consistency, subassembly count

**Recommendation:** These are all pure functions with well-defined inputs/outputs — ideal candidates for unit tests with no external dependencies.

#### 4. `app.py` — FastAPI Application Lifecycle
**Risk: LOW-MEDIUM** — Application startup/shutdown and middleware configuration.

No tests for:
- Health check endpoint (`GET /health`)
- CORS middleware configuration
- Lifespan context manager (dev mode vs. production mode)

**Recommendation:** Add a basic test for the health endpoint and CORS headers using TestClient.

### Priority 2 — Partial Coverage / Missing Edge Cases

#### 5. `brick/parser.py` — Missing Edge Cases
Current tests are solid but miss:
- **Boundary coordinates**: bricks at `(0,0,0)`, `(19,19,19)`, negative values
- **Large dimension strings**: dimensions not in `ALLOWED_DIMS`
- **Color format variations**: lowercase hex, missing digits, 3-char shorthand
- **Multi-line with empty lines/trailing whitespace** in `parse_brick_sequence`
- **Unicode or unexpected characters** in input

#### 6. `brick/occupancy.py` — Missing Edge Cases
Current tests miss:
- `remove()` method — never tested
- `clear()` method — never tested
- Placing a brick, removing it, then placing another in the same spot
- Negative coordinate handling
- Edge-of-grid placements where brick fits exactly (`x=16` with `h=4` for `WORLD_DIM=20`)

#### 7. `brick/stability.py` — Missing Edge Cases
Current tests miss:
- **Empty brick list** → should return `True`
- **Large structures** with complex adjacency graphs
- **Bricks at the same z-level with no vertical adjacency** (horizontal neighbors are not connected)
- `find_first_unstable()` with an empty list
- `_overlaps_xy()` edge cases: touching but non-overlapping, partial overlap

#### 8. `inference/pipeline.py` — Mock-Only Testing
The pipeline tests only exercise `MockPipeline`. There is no test that verifies:
- `get_pipeline()` returns the same singleton on repeated calls
- `get_planner_pipeline()` returns the same instance as `get_pipeline()`
- `MockPipeline` response structure matches the real pipeline's contract (schema validation)
- `TwoStagePipeline.__init__` behavior when checkpoint directory is missing (fallback logic)

### Priority 3 — Entirely Untested Areas

#### 9. Frontend — Zero Test Coverage
**Risk: MEDIUM-HIGH** — The frontend has ~1,230 lines of TypeScript/TSX with no tests at all.

Key areas to test:
- **`parseBrickString()`** (`api/legogen.ts`) — Pure function, easy to unit test with Vitest. Should cover valid input, empty input, malformed lines, missing color.
- **`bricksToSteps()`** (`api/legogen.ts`) — Pure function that groups bricks by z-level. Test layer grouping, part aggregation, LDRAW ID mapping, empty input.
- **`generateBricks()`** (`api/legogen.ts`) — API client. Mock fetch to test FormData construction, error handling, response parsing.
- **`BrickCoordViewer`** component — Renders Three.js canvas. Snapshot/smoke test to verify it renders without crashing.
- **`UploadPanel`** component — File upload and drag-and-drop. Test with React Testing Library.
- **`BuildSession`** page — Main user flow. Integration test for chat interaction, file upload, and brick display.
- **`ErrorBoundary`** component — Verify it catches errors and renders fallback.

**Recommendation:** Set up Vitest (already Vite-based project) + React Testing Library. Start with the pure utility functions in `legogen.ts`, then add component smoke tests.

#### 10. `data_pipeline/dataset_stage1.py` — Stage 1 Dataset Class
No tests for:
- `Stage1Dataset.__getitem__()` — data loading and preprocessing
- `Stage1Dataset._mask_prompt()` — prompt token masking for training
- `Stage1Collator` — batch collation

#### 11. `data_pipeline/dataset.py` — Image Transforms
No tests for `TRAIN_TRANSFORMS` or `VAL_TRANSFORMS` — verify they produce expected output shapes and types.

#### 12. `brick/constants.py` — Import-Time Data Loading
- `_load_color_palette()` fails hard when `colors.json` is missing, which breaks test collection for any module importing from `constants`. This should be tested with a fallback or graceful error.

#### 13. Scripts (`scripts/`)
`benchmark.py` and `prepare_dataset.py` have no tests. Lower priority since they are offline utilities, but `benchmark.py` metric computation should be validated.

---

## Recommended Action Plan

### Phase 1 — Quick Wins (Pure Functions, No Dependencies)

| Action | Est. Tests | Effort |
|--------|-----------|--------|
| Test `training/utils.py` metrics functions | 15-20 | Low |
| Test `models/tokenizer.py` pure functions (`extract_json_from_text`, `strip_thinking_blocks`, `sample_prompt_template`, `build_*_messages`) | 12-15 | Low |
| Test `api/routes_generate.py` with TestClient + mocked pipeline | 6-8 | Low |
| Fix `constants.py` to gracefully handle missing `colors.json` (unblocks decoder/occupancy tests in CI) | 1-2 | Low |

### Phase 2 — Edge Cases for Existing Tests

| Action | Est. Tests | Effort |
|--------|-----------|--------|
| Add `occupancy.py` tests for `remove()`, `clear()`, edge-of-grid | 5-6 | Low |
| Add `stability.py` tests for empty list, `_overlaps_xy` edges | 4-5 | Low |
| Add `parser.py` boundary and format edge cases | 4-5 | Low |

### Phase 3 — Frontend Testing Setup

| Action | Est. Tests | Effort |
|--------|-----------|--------|
| Set up Vitest + React Testing Library | — | Medium |
| Test `parseBrickString()` and `bricksToSteps()` utilities | 8-10 | Low |
| Add `ErrorBoundary` and `UploadPanel` component tests | 4-6 | Medium |
| Add `BuildSession` integration smoke test | 2-3 | Medium |

### Phase 4 — Infrastructure

| Action | Effort |
|--------|--------|
| Add `pyproject.toml` `[tool.pytest.ini_options]` with test paths, markers, and coverage config | Low |
| Add `pytest-cov` to CI pipeline for coverage reporting | Low |
| Add GitHub Actions workflow to run tests on push/PR | Medium |

---

## Summary

| Area | Current Tests | Missing Tests (est.) | Priority |
|------|--------------|---------------------|----------|
| API routes | 0 | 6-8 | P1 |
| Tokenizer/prompt utilities | 0 | 12-15 | P1 |
| Training metrics | 0 | 15-20 | P1 |
| Brick parser edge cases | 7 (good) | 4-5 | P2 |
| Occupancy edge cases | 6 (good) | 5-6 | P2 |
| Stability edge cases | 6 (good) | 4-5 | P2 |
| Frontend utilities | 0 | 8-10 | P3 |
| Frontend components | 0 | 6-9 | P3 |
| Constants/import resilience | 0 | 1-2 | P1 |
| **Total** | **~34** | **~65-85** | |

The most impactful improvements are testing the **API route handler**, the **tokenizer/JSON parsing utilities**, and the **training metrics** — all are critical-path code with zero coverage and low testing effort (pure functions or easily mocked).
