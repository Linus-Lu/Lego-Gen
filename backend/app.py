"""FastAPI application for LEGOGen backend."""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes_generate import router as generate_router
from backend.api.routes_gallery import router as gallery_router
from backend.storage import gallery_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle. Preload models so first request isn't slow."""
    await gallery_db.init_db()
    if os.environ.get("LEGOGEN_DEV", "1") != "1":
        print("LEGOGen API starting — preloading models...")
        loop = asyncio.get_running_loop()
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

# CORS: explicit origin list. The Fetch spec (and every major browser)
# rejects ``Access-Control-Allow-Origin: *`` combined with credentials, so we
# list the Vite dev server and whatever is in LEGOGEN_CORS_ORIGINS (comma-
# separated) for deploys. No credentials are expected today — set
# ``allow_credentials=False`` to keep the surface minimal.
_default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
_env_origins = os.environ.get("LEGOGEN_CORS_ORIGINS", "").strip()
_allowed_origins = (
    [o.strip() for o in _env_origins.split(",") if o.strip()]
    if _env_origins
    else _default_origins
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate_router)
app.include_router(gallery_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
