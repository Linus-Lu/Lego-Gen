"""FastAPI routes for LEGO image-to-JSON generation."""

import io
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.inference.pipeline import get_pipeline, get_planner_pipeline

router = APIRouter(prefix="/api", tags=["generate"])


@router.post("/generate")
async def generate(
    image: UploadFile = File(...),
    prompt: str = Form(default=""),
):
    """Generate LEGO description and build steps from an uploaded image.

    Args:
        image: Uploaded image file (JPEG, PNG, etc.)
        prompt: Optional text prompt for additional context.

    Returns:
        JSON with description, steps, and metadata.
    """
    # Validate file type
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    # Read and open image
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        # In dev mode, allow any file (mock pipeline ignores the image)
        from backend.config import LEGOGEN_DEV
        if LEGOGEN_DEV:
            pil_image = Image.new("RGB", (224, 224), (128, 128, 128))
        else:
            raise HTTPException(status_code=400, detail="Could not read the uploaded image")

    # Run inference
    pipeline = get_pipeline()
    result = pipeline.generate_build(pil_image)

    return result


@router.post("/generate-from-text")
async def generate_from_text(
    prompt: str = Form(...),
):
    """Generate LEGO description and build steps from a text prompt.

    Args:
        prompt: Text description of the desired LEGO model.

    Returns:
        JSON with description, steps, and metadata.
    """
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    pipeline = get_planner_pipeline()
    result = pipeline.generate_build_from_text(prompt.strip())

    return result
