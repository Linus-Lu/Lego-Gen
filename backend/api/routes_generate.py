"""FastAPI routes for LEGO brick-coordinate generation."""

import asyncio
import io
import json
from pathlib import Path
from queue import SimpleQueue, Empty

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.inference.brick_pipeline import get_brick_pipeline
from backend.config import INFERENCE_TIMEOUT_SECONDS, LEGOGEN_DEV

router = APIRouter(prefix="/api", tags=["generate"])


def _decode_image(contents: bytes) -> Image.Image:
    """Decode uploaded image bytes to a PIL image, with a dev-mode fallback."""
    try:
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        if LEGOGEN_DEV:
            return Image.new("RGB", (224, 224), (128, 128, 128))
        raise HTTPException(status_code=400, detail="Could not read the uploaded image")


@router.post("/generate-bricks")
async def generate_bricks(
    image: UploadFile = File(default=None),
    prompt: str = Form(default=""),
    n: int = Form(default=1),
):
    """Image or text → brick-coordinate model. Non-streaming.

    If n > 1, runs Best-of-N sampling and returns the picked candidate. The
    response shape is unchanged; n/picked_index/stable_rate are stamped on
    metadata.
    """
    pipeline = get_brick_pipeline()
    loop = asyncio.get_event_loop()

    if image and image.filename:
        if image.content_type and not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")
        contents = await image.read()
        pil_image = _decode_image(contents)
        del contents
        if n > 1:
            call = lambda: _from_image_bon(pipeline, pil_image, n)
        else:
            call = lambda: pipeline.generate_from_image(pil_image)
    elif prompt and prompt.strip():
        if n > 1:
            call = lambda: pipeline.generate_best_of_n(prompt.strip(), n=n)
        else:
            call = lambda: pipeline.generate(prompt.strip())
    else:
        raise HTTPException(status_code=400, detail="Provide an image or prompt")

    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, call),
            timeout=INFERENCE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Inference timed out after {INFERENCE_TIMEOUT_SECONDS}s",
        )


def _from_image_bon(pipeline, pil_image, n: int) -> dict:
    """Image path BoN: describe once, then BoN on the caption.

    Avoids running Stage 1 n times — it's the expensive VLM call.
    """
    from backend.inference.brick_pipeline import _get_stage1_pipeline
    caption = _get_stage1_pipeline().describe(pil_image)
    result = pipeline.generate_best_of_n(caption, n=n)
    result["caption"] = caption
    return result


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/generate-stream")
async def generate_stream(
    image: UploadFile = File(default=None),
    prompt: str = Form(default=""),
):
    """SSE streaming endpoint for brick generation.

    Events:
      - progress: {stage: "stage1"|"stage2", message: str}
      - brick:    {count: int}                — after each brick placed
      - rollback: {count: int}                — after each physics rollback
      - result:   {bricks, caption, brick_count, stable, metadata}
      - error:    {detail: str}
    """
    pil_image = None

    if image and image.filename:
        if image.content_type and not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")
        contents = await image.read()
        pil_image = _decode_image(contents)
        del contents

    if pil_image is None and not (prompt and prompt.strip()):
        raise HTTPException(status_code=400, detail="Provide an image or prompt")

    text_prompt = prompt.strip() if prompt else ""

    async def event_generator():
        pipeline = get_brick_pipeline()
        loop = asyncio.get_event_loop()
        event_queue: SimpleQueue = SimpleQueue()

        def on_progress(evt: dict) -> None:
            event_queue.put(evt)

        def run() -> dict:
            if pil_image is not None:
                return pipeline.generate_from_image(pil_image, on_progress=on_progress)
            return pipeline.generate(text_prompt, on_progress=on_progress)

        if pil_image is not None:
            yield _sse_event("progress", {"stage": "stage1", "message": "Analyzing image..."})
        else:
            yield _sse_event("progress", {"stage": "stage2", "message": "Generating bricks..."})

        future = loop.run_in_executor(None, run)
        announced_stage2 = pil_image is None
        caption: str | None = None

        try:
            while True:
                try:
                    evt = event_queue.get_nowait()
                except Empty:
                    if future.done():
                        break
                    await asyncio.sleep(0.05)
                    continue

                if evt["type"] == "caption":
                    caption = evt["caption"]
                    yield _sse_event("progress", {
                        "stage": "stage1",
                        "message": "Caption ready",
                        "caption": caption,
                    })
                    if not announced_stage2:
                        yield _sse_event("progress", {"stage": "stage2", "message": "Generating bricks..."})
                        announced_stage2 = True
                elif evt["type"] == "brick":
                    yield _sse_event("brick", {"count": evt["count"]})
                elif evt["type"] == "rollback":
                    yield _sse_event("rollback", {"count": evt["count"]})

            result = await future
            # Drain any late events.
            while not event_queue.empty():
                evt = event_queue.get_nowait()
                if evt["type"] == "brick":
                    yield _sse_event("brick", {"count": evt["count"]})
                elif evt["type"] == "rollback":
                    yield _sse_event("rollback", {"count": evt["count"]})

            yield _sse_event("result", result)
        except Exception as exc:
            yield _sse_event("error", {"detail": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
