"""Gallery CRUD endpoints."""

import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.storage.gallery_db import (
    create_build,
    get_build,
    list_builds,
    update_star,
)

router = APIRouter(prefix="/api", tags=["gallery"])


# ── Request models ────────────────────────────────────────────────────

class CreateBuildRequest(BaseModel):
    title: str
    description_json: str
    thumbnail_b64: str = ""


class StarRequest(BaseModel):
    stars: int = Field(ge=1, le=5)


# ── Endpoints ─────────────────────────────────────────────────────────

@router.get("/gallery")
async def list_gallery(
    category: Optional[str] = None,
    sort: str = "newest",
    q: Optional[str] = None,
):
    return await list_builds(category=category, sort=sort, q=q)


@router.post("/gallery", status_code=201)
async def create_gallery_build(req: CreateBuildRequest):
    # Derive category, complexity, parts_count from description_json
    category = ""
    complexity = "medium"
    parts_count = 0
    try:
        desc = json.loads(req.description_json)
        category = desc.get("category", "")
        complexity = desc.get("complexity", "medium")
        parts_count = desc.get("total_parts", 0)
    except (json.JSONDecodeError, AttributeError):
        pass

    build = await create_build(
        title=req.title,
        category=category,
        complexity=complexity,
        parts_count=parts_count,
        description_json=req.description_json,
        thumbnail_b64=req.thumbnail_b64,
    )
    return build


@router.get("/gallery/{build_id}")
async def get_gallery_build(build_id: str):
    build = await get_build(build_id)
    if not build:
        raise HTTPException(status_code=404, detail="Build not found")
    return build


@router.patch("/gallery/{build_id}/star")
async def star_gallery_build(build_id: str, req: StarRequest):
    build = await update_star(build_id, req.stars)
    if not build:
        raise HTTPException(status_code=404, detail="Build not found")
    return build
