"""FastAPI routes for two-phase LEGO brick generation."""

import asyncio
import io
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.inference.pipeline import get_pipeline
from backend.config import INFERENCE_TIMEOUT_SECONDS

router = APIRouter(prefix="/api", tags=["generate"])


@router.post("/generate-bricks")
async def generate_bricks(
    image: UploadFile = File(default=None),
    prompt: str = Form(default=""),
):
    """Generate a brick-coordinate LEGO model from image or text.

    Two-phase pipeline:
      Stage 1 (if image): Image → text description (Qwen 7B)
      Stage 2: Text → brick coordinates (Qwen 4B)
    """
    if image and image.filename:
        if image.content_type and not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")
        contents = await image.read()
        try:
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Could not read the uploaded image") from exc
        del contents

        pipeline = get_pipeline()
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
        pipeline = get_pipeline()
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
