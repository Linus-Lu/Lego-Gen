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
    """Load the ML model on startup."""
    print("Loading LEGOGen inference pipeline...")
    from backend.inference.pipeline import get_pipeline
    get_pipeline()  # Initialize singleton
    print("Pipeline ready.")
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
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
