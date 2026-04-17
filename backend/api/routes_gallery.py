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


# Gallery is unauthenticated by design; cap every free-form string so one
# client can't stuff the SQLite DB with megabyte-sized blobs.
_MAX_TITLE_CHARS = 200
_MAX_CAPTION_CHARS = 2000
_MAX_BRICKS_CHARS = 200_000  # ≈ 500 bricks at 40 chars/line → plenty of slack
_MAX_THUMBNAIL_CHARS = 400_000  # base64 of a ~300 KB PNG


class CreateBuildRequest(BaseModel):
    title: str = Field(min_length=1, max_length=_MAX_TITLE_CHARS)
    caption: str = Field(default="", max_length=_MAX_CAPTION_CHARS)
    bricks: str = Field(max_length=_MAX_BRICKS_CHARS)
    brick_count: int = Field(default=0, ge=0)
    stable: bool = True
    thumbnail_b64: str = Field(default="", max_length=_MAX_THUMBNAIL_CHARS)


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
