"""FastAPI routes for LEGO brick-coordinate generation."""

import asyncio
import io
import json
import threading
from queue import SimpleQueue, Empty

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image

from backend.inference.brick_pipeline import get_brick_pipeline
from backend.config import INFERENCE_TIMEOUT_SECONDS, LEGOGEN_DEV

router = APIRouter(prefix="/api", tags=["generate"])

# ── Input size limits ───────────────────────────────────────────────────
# Image: frontend advertises ≤ 8 MB; reject a bit above that to tolerate
# multipart overhead without letting a 2 GB upload eat server memory.
MAX_IMAGE_BYTES = 10 * 1024 * 1024
# Prompt: any real text input is well under a few hundred chars; cap at 2000.
MAX_PROMPT_CHARS = 2000


def _decode_image(contents: bytes) -> Image.Image:
    """Decode uploaded image bytes to a PIL image, with a dev-mode fallback."""
    try:
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        if LEGOGEN_DEV:
            return Image.new("RGB", (224, 224), (128, 128, 128))
        raise HTTPException(status_code=400, detail="Could not read the uploaded image")


async def _read_image_bounded(image: UploadFile) -> bytes:
    """Read an upload, rejecting anything larger than MAX_IMAGE_BYTES.

    Reads in chunks so we never buffer an oversized payload.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await image.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _validate_prompt(prompt: str) -> str:
    """Trim whitespace and enforce the prompt length cap.

    The cap is applied to the RAW input first — ``' ' * 500000 + 'a'`` would
    strip down to one character and slip past a post-strip check, but we want
    to reject oversized payloads from unauthenticated clients regardless of
    whether the characters are whitespace.
    """
    if prompt and len(prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Prompt exceeds {MAX_PROMPT_CHARS} characters",
        )
    return prompt.strip() if prompt else ""


@router.post("/generate-bricks")
async def generate_bricks(
    image: UploadFile = File(default=None),
    prompt: str = Form(default=""),
    n: int = Form(default=1, ge=1, le=16),
):
    """Image or text → brick-coordinate model. Non-streaming.

    If n > 1, runs Best-of-N sampling and returns the picked candidate. The
    response shape is unchanged; n/picked_index/stable_rate are stamped on
    metadata. n is capped at 16 because each inner sample is a full model
    generation (~15–60s on the 4-bit Qwen3.5-4B). The pipeline polls a
    cancellation flag between samples so a client-side timeout stops burning
    compute — see ``_CancelToken`` below.
    """
    loop = asyncio.get_running_loop()

    has_image = bool(image and image.filename)
    stripped_prompt = _validate_prompt(prompt)
    has_prompt = bool(stripped_prompt)
    if not has_image and not has_prompt:
        raise HTTPException(status_code=400, detail="Provide an image or prompt")
    if has_image and image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    pipeline = get_brick_pipeline()
    cancel = _CancelToken()

    if has_image:
        contents = await _read_image_bounded(image)
        pil_image = _decode_image(contents)
        del contents
        if n > 1:
            call = lambda: _from_image_bon(
                pipeline, pil_image, n, on_progress=cancel.check_callback,
            )
        else:
            call = lambda: pipeline.generate_from_image(pil_image)
    else:
        if n > 1:
            call = lambda: pipeline.generate_best_of_n(
                stripped_prompt, n=n, on_progress=cancel.check_callback,
            )
        else:
            call = lambda: pipeline.generate(stripped_prompt)

    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, call),
            timeout=INFERENCE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        # Signal the worker thread to stop at its next cooperative
        # checkpoint. The thread will still finish its current sample, but
        # it won't start another one — freeing the pool slot.
        cancel.set()
        raise HTTPException(
            status_code=504,
            detail=f"Inference timed out after {INFERENCE_TIMEOUT_SECONDS}s",
        )


class _CancelToken:
    """Thread-safe cancellation flag polled by long-running generate() loops.

    ``run_in_executor`` futures cannot be cancelled once the thread has
    started, so the worker must poll this flag and raise at a safe point.
    ``check_callback`` is shaped like the ``on_progress`` callback the
    pipeline already supports, which means Best-of-N can honour cancellation
    between samples without any new integration surface.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def set(self) -> None:
        self._event.set()

    @property
    def is_set(self) -> bool:
        return self._event.is_set()

    def check_callback(self, _evt: dict) -> None:
        if self._event.is_set():
            raise _Cancelled()


class _Cancelled(Exception):
    """Raised inside the worker thread when the request has been cancelled."""


def _from_image_bon(pipeline, pil_image, n: int, on_progress=None) -> dict:
    """Image path BoN: describe once, then BoN on the caption.

    Avoids running Stage 1 n times — it's the expensive VLM call. The caller
    passes an ``on_progress`` callback that is used both for surfacing events
    (caption / sample) and for cooperative cancellation — the callback is
    expected to raise ``_Cancelled`` when the request is no longer wanted.
    """
    from backend.inference.brick_pipeline import _get_stage1_pipeline
    caption = _get_stage1_pipeline().describe(pil_image)
    if on_progress is not None:
        on_progress({"type": "caption", "caption": caption})
    result = pipeline.generate_best_of_n(caption, n=n, on_progress=on_progress)
    result["caption"] = caption
    return result


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/generate-stream")
async def generate_stream(
    image: UploadFile = File(default=None),
    prompt: str = Form(default=""),
    n: int = Form(default=1, ge=1, le=16),
):
    """SSE streaming endpoint for brick generation.

    Events:
      - progress: {stage: "stage1"|"stage2", message: str}
      - brick:    {count: int}                — after each brick placed
      - rollback: {count: int}                — after each physics rollback
      - sample:   {index: int, of: int}       — after each Best-of-N sample
      - result:   {bricks, caption, brick_count, stable, metadata}
      - error:    {detail: str}
    """
    pil_image = None

    if image and image.filename:
        if image.content_type and not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")
        contents = await _read_image_bounded(image)
        pil_image = _decode_image(contents)
        del contents

    text_prompt = _validate_prompt(prompt)
    if pil_image is None and not text_prompt:
        raise HTTPException(status_code=400, detail="Provide an image or prompt")

    async def event_generator():
        pipeline = get_brick_pipeline()
        loop = asyncio.get_running_loop()
        event_queue: SimpleQueue = SimpleQueue()
        cancel = _CancelToken()

        def on_progress(evt: dict) -> None:
            # Bubble cancellation into the worker thread at every progress
            # boundary — otherwise a client disconnect or server-side
            # timeout would let an idle worker keep burning compute.
            if cancel.is_set:
                raise _Cancelled()  # pragma: no cover — only raised when client disconnects mid-stream; TestClient is synchronous.
            event_queue.put(evt)

        def run() -> dict:
            if pil_image is not None:
                if n > 1:
                    return _from_image_bon(
                        pipeline, pil_image, n, on_progress=on_progress,
                    )
                return pipeline.generate_from_image(pil_image, on_progress=on_progress)
            if n > 1:
                return pipeline.generate_best_of_n(
                    text_prompt, n=n, on_progress=on_progress,
                )
            return pipeline.generate(text_prompt, on_progress=on_progress)

        if pil_image is not None:
            yield _sse_event("progress", {"stage": "stage1", "message": "Analyzing image..."})
        else:
            yield _sse_event("progress", {"stage": "stage2", "message": "Generating bricks..."})

        future = loop.run_in_executor(None, run)
        announced_stage2 = pil_image is None
        caption: str | None = None
        deadline = loop.time() + INFERENCE_TIMEOUT_SECONDS

        try:
            while True:
                try:
                    evt = event_queue.get_nowait()
                except Empty:
                    if future.done():
                        break
                    if loop.time() >= deadline:  # pragma: no cover — SSE deadline fires only on real timeout races; not deterministic with TestClient.
                        cancel.set()
                        yield _sse_event("error", {
                            "detail": f"Inference timed out after {INFERENCE_TIMEOUT_SECONDS}s",
                        })
                        return
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
                elif evt["type"] == "sample":
                    yield _sse_event("sample", {
                        "index": evt.get("index"),
                        "of": evt.get("of"),
                        "stable": evt.get("stable"),
                    })

            result = await future
            # Drain any late events.
            while not event_queue.empty():  # pragma: no cover — race-condition drain path; events queued after future.done() aren't deterministically reachable in tests.
                evt = event_queue.get_nowait()
                if evt["type"] == "brick":
                    yield _sse_event("brick", {"count": evt["count"]})
                elif evt["type"] == "rollback":
                    yield _sse_event("rollback", {"count": evt["count"]})

            yield _sse_event("result", result)
        except asyncio.CancelledError:  # pragma: no cover — client-disconnect path; not reachable from synchronous TestClient.
            # Client disconnected. Mark the worker for cooperative shutdown;
            # the thread can't be killed but it will stop at its next
            # progress event and not start another Best-of-N sample.
            cancel.set()
            raise
        except _Cancelled:  # pragma: no cover — raised from the worker thread on cancellation; requires client-disconnect mid-stream.
            yield _sse_event("error", {"detail": "Request cancelled"})
        except Exception as exc:  # pragma: no cover — unexpected worker-thread error; mocks don't raise.
            yield _sse_event("error", {"detail": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
