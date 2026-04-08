# Adversarial Review (April 8, 2026)

Scope reviewed:
- API routes, inference pipeline, and model configuration paths.
- Focus on runtime reliability and attack-surface behavior.

## High-risk findings

1. **Broken import in `/api/generate-bricks` (runtime 500 on first call)**
   - Route imported `get_pipeline` from `backend.app`, but that symbol is not defined in `backend/app.py`.
   - Impact: text/image brick generation endpoint would fail in production with import error.
   - Fix applied: route now uses `get_planner_pipeline()` from `backend.inference.pipeline` (already imported in-file).

2. **No validation for image content type / decode failures in `/api/generate-bricks`**
   - Route accepted any uploaded file and attempted to decode directly.
   - Impact: malformed payloads raised unhandled exceptions, producing 500s instead of user-safe 400s.
   - Fix applied: explicit MIME gate + guarded PIL decode with `HTTPException(400)`.

3. **Prompt whitespace bypass in `/api/generate-bricks`**
   - Route treated whitespace-only prompt as valid input.
   - Impact: downstream model calls receive empty semantic input, wasting GPU cycles and causing unstable outputs.
   - Fix applied: normalize with `prompt.strip()` and reject empty value.

## Architecture risks to address for your stated plan

You said your training/pipeline target is:
- **Qwen 9B for vision**, and
- **Qwen 4B for JSON output**.

Current codebase is not aligned end-to-end:
- `MODEL_NAME` still points at **Qwen3-VL-8B-Instruct**.
- JSON description generation pipeline defaults to **Qwen3.5-9B** (`PLANNER_MODEL_NAME` / `UNIFIED_MODEL_NAME`).
- 4B model is wired for the **brick-coordinate generator** (`BRICK_MODEL_NAME`), not primary JSON output.

This mismatch can create training/inference drift, incompatible adapters, and degraded eval comparability.

## Recommended next hardening steps

- Add request size limits and pixel dimension caps for image upload endpoints to reduce decompression-bomb / memory pressure risk.
- Add endpoint-level timeout + cancellation guards around long generation paths.
- Introduce explicit model-role config map (vision_model, json_model, brick_model) and enforce it at startup with compatibility checks against adapter metadata.
- Add API tests for `/api/generate-bricks` failure modes (bad MIME, unreadable image, empty prompt).
