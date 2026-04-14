"""FastAPI application for LEGOGen backend."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.api.routes_generate import router as generate_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle. Preload model so first request isn't slow."""
    import asyncio, os
    if os.environ.get("LEGOGEN_DEV", "1") != "1":
        print("LEGOGen API starting — preloading model...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _preload_model)
        print("LEGOGen API ready. Model loaded.")
    else:
        print("LEGOGen API ready (dev mode — mock pipeline).")
    yield
    print("Shutting down LEGOGen.")


def _preload_model():
    """Load the unified pipeline in a thread so startup isn't blocked."""
    from backend.inference.pipeline import get_pipeline
    get_pipeline()


app = FastAPI(
    title="LEGOGen API",
    description="Two-phase LEGO model generator: image-to-text + text-to-brick-coordinates",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate_router)


@app.get("/health")
async def health():
    from backend.inference.pipeline import get_pipeline
    pipeline = get_pipeline()
    result = {"status": "ok"}
    if hasattr(pipeline, "cache_stats"):
        result["cache"] = pipeline.cache_stats()
    return result
