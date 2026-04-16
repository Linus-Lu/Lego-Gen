"""Gallery CRUD endpoints."""

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


class CreateBuildRequest(BaseModel):
    title: str
    caption: str = ""
    bricks: str
    brick_count: int = 0
    stable: bool = True
    thumbnail_b64: str = ""


class StarRequest(BaseModel):
    stars: int = Field(ge=1, le=5)


@router.get("/gallery")
async def list_gallery(
    sort: str = "newest",
    q: Optional[str] = None,
):
    return await list_builds(sort=sort, q=q)


@router.post("/gallery", status_code=201)
async def create_gallery_build(req: CreateBuildRequest):
    if not req.bricks.strip():
        raise HTTPException(status_code=400, detail="bricks must not be empty")
    return await create_build(
        title=req.title,
        caption=req.caption,
        bricks=req.bricks,
        brick_count=req.brick_count,
        stable=req.stable,
        thumbnail_b64=req.thumbnail_b64,
    )


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
