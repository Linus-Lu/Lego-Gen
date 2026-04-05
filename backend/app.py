"""FastAPI application for LEGOGen backend."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.api.routes_generate import router as generate_router
from backend.api.routes_validate import router as validate_router
from backend.api.routes_gallery import router as gallery_router
from backend.storage import gallery_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle. Models load lazily on first request."""
    await gallery_db.init_db()
    print("LEGOGen API ready. Models will load on first request.")
    yield
    print("Shutting down LEGOGen.")


app = FastAPI(
    title="LEGOGen API",
    description="Generate LEGO building instructions from images",
    version="0.1.0",
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
app.include_router(validate_router)
app.include_router(gallery_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
