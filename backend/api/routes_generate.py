"""FastAPI routes for LEGO image-to-JSON generation."""

import asyncio
import hashlib
import io
import json
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.inference.pipeline import get_pipeline, get_planner_pipeline
from backend.config import INFERENCE_TIMEOUT_SECONDS

router = APIRouter(prefix="/api", tags=["generate"])


@router.post("/generate")
async def generate(
    image: UploadFile = File(...),
    prompt: str = Form(default=""),
):
    """Generate LEGO description and build steps from an uploaded image.

    Uses run_in_executor to avoid blocking the event loop during GPU inference.
    Includes a timeout to prevent indefinite GPU holds.
    """
    # Validate file type
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    # Read and open image
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        from backend.config import LEGOGEN_DEV
        if LEGOGEN_DEV:
            pil_image = Image.new("RGB", (224, 224), (128, 128, 128))
        else:
            raise HTTPException(status_code=400, detail="Could not read the uploaded image")

    # Compute image hash for response caching, then free raw bytes
    cache_key = hashlib.sha256(contents).hexdigest()
    del contents

    # Run inference in a thread with timeout
    pipeline = get_pipeline()
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None, lambda: pipeline.generate_build(pil_image, cache_key=cache_key)
            ),
            timeout=INFERENCE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Inference timed out after {INFERENCE_TIMEOUT_SECONDS}s")

    return result


@router.post("/generate-from-text")
async def generate_from_text(
    prompt: str = Form(...),
):
    """Generate LEGO description and build steps from a text prompt.

    Uses run_in_executor to avoid blocking the event loop during GPU inference.
    Includes a timeout to prevent indefinite GPU holds.
    """
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    pipeline = get_planner_pipeline()
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None, lambda: pipeline.generate_build_from_text(prompt.strip())
            ),
            timeout=INFERENCE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Inference timed out after {INFERENCE_TIMEOUT_SECONDS}s")

    return result


@router.post("/generate-bricks")
async def generate_bricks(
    image: UploadFile = File(default=None),
    prompt: str = Form(default=""),
):
    """Generate a brick-coordinate LEGO model from image or text.

    Uses run_in_executor to avoid blocking the event loop during GPU inference.
    Includes a timeout to prevent indefinite GPU holds.
    """
    if image and image.filename:
        from PIL import Image as PILImage
        import io as _io
        if image.content_type and not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")
        contents = await image.read()
        try:
            pil_image = PILImage.open(_io.BytesIO(contents)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Could not read the uploaded image") from exc
        del contents
        pipeline = get_planner_pipeline()
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: pipeline.generate_brick_build_from_image(pil_image)
                ),
                timeout=INFERENCE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail=f"Inference timed out after {INFERENCE_TIMEOUT_SECONDS}s")
    elif prompt and prompt.strip():
        pipeline = get_planner_pipeline()
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: pipeline.generate_brick_build(prompt.strip())
                ),
                timeout=INFERENCE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail=f"Inference timed out after {INFERENCE_TIMEOUT_SECONDS}s")
    else:
        raise HTTPException(status_code=400, detail="Provide an image or prompt")

    return result


# ── SSE Streaming endpoint ────────────────────────────────────────────


def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/generate-stream")
async def generate_stream(
    image: UploadFile = File(default=None),
    prompt: str = Form(default=""),
):
    """SSE streaming endpoint that sends progress events during generation.

    Events:
      - progress: {stage: "stage1"|"stage2"|"validating", message: str}
      - result:   full GenerateResponse JSON
      - error:    {detail: str}
    """
    pil_image = None
    cache_key = None

    if image and image.filename:
        if image.content_type and not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")
        try:
            contents = await image.read()
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
            cache_key = hashlib.sha256(contents).hexdigest()
            del contents
        except Exception:
            from backend.config import LEGOGEN_DEV
            if LEGOGEN_DEV:
                pil_image = Image.new("RGB", (224, 224), (128, 128, 128))
            else:
                raise HTTPException(status_code=400, detail="Could not read the uploaded image")

    if not pil_image and not (prompt and prompt.strip()):
        raise HTTPException(status_code=400, detail="Provide an image or prompt")

    async def event_generator():
        loop = asyncio.get_event_loop()
        pipeline = get_pipeline()

        try:
            if pil_image and pipeline.has_stage1:
                # Two-stage: stream progress for each stage
                yield _sse_event("progress", {"stage": "stage1", "message": "Analyzing image..."})
                description_text = await loop.run_in_executor(
                    None, lambda: pipeline.describe_image_stage1(pil_image)
                )

                yield _sse_event("progress", {"stage": "stage2", "message": "Generating LEGO build..."})
                result_inner = await loop.run_in_executor(
                    None, lambda: pipeline.describe_from_text(description_text)
                )
            elif pil_image:
                yield _sse_event("progress", {"stage": "stage2", "message": "Generating from image..."})
                result_inner = await loop.run_in_executor(
                    None, lambda: pipeline.describe_image(pil_image, cache_key=cache_key)
                )
            else:
                yield _sse_event("progress", {"stage": "stage2", "message": "Generating LEGO build..."})
                result_inner = await loop.run_in_executor(
                    None, lambda: pipeline.describe_from_text(prompt.strip())
                )

            yield _sse_event("progress", {"stage": "validating", "message": "Validating structure..."})

            # Build final response
            from backend.inference.postprocess_manual import json_to_steps
            from backend.inference.pipeline import _get_stability_checker
            from dataclasses import asdict

            description = result_inner["description"]
            steps = json_to_steps(description) if description else []
            validation = asdict(_get_stability_checker().validate(description))

            final = {
                "description": description,
                "steps": steps,
                "metadata": {
                    "model_version": "qwen35-lego-two-stage-v1" if (pil_image and pipeline.has_stage1) else "qwen35-lego-unified-v1",
                    "generation_time_ms": result_inner["generation_time_ms"],
                    "json_valid": result_inner["is_valid"],
                    "errors": result_inner["errors"],
                    "cached": result_inner.get("cached", False),
                },
                "validation": validation,
            }

            yield _sse_event("result", final)

        except Exception as e:
            yield _sse_event("error", {"detail": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
