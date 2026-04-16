"""FastAPI application for LEGOGen backend."""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.api.routes_generate import router as generate_router
from backend.api.routes_gallery import router as gallery_router
from backend.storage import gallery_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle. Preload models so first request isn't slow."""
    await gallery_db.init_db()
    if os.environ.get("LEGOGEN_DEV", "1") != "1":
        print("LEGOGen API starting — preloading models...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _preload_models)
        print("LEGOGen API ready. Models loaded.")
    else:
        print("LEGOGen API ready (dev mode — mock pipeline).")
    yield
    print("Shutting down LEGOGen.")


def _preload_models():
    """Load the brick + Stage 1 pipelines in a thread so startup isn't blocked."""
    from backend.inference.brick_pipeline import get_brick_pipeline, _get_stage1_pipeline
    get_brick_pipeline()
    _get_stage1_pipeline()


app = FastAPI(
    title="LEGOGen API",
    description="Two-stage pipeline: image/text → LEGO brick coordinates",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate_router)
app.include_router(gallery_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
