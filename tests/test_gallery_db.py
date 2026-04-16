"""Tests for backend.storage.gallery_db — async SQLite gallery CRUD."""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import aiosqlite  # noqa: F401

    from backend.storage.gallery_db import (
        create_build,
        get_build,
        init_db,
        list_builds,
        update_star,
    )

    _AIOSQLITE = True
except ImportError:
    _AIOSQLITE = False

pytestmark = pytest.mark.skipif(not _AIOSQLITE, reason="aiosqlite not installed")

SAMPLE_BRICKS = "2x4 (0,0,0) #C91A09\n2x4 (2,0,0) #C91A09"


def _run(coro):
    return asyncio.run(coro)


def _make(title: str, **overrides) -> dict:
    kwargs = dict(
        title=title,
        caption="a small red house",
        bricks=SAMPLE_BRICKS,
        brick_count=2,
        stable=True,
        thumbnail_b64="",
    )
    kwargs.update(overrides)
    return kwargs


class TestGalleryDB:
    def test_init_db_idempotent(self, gallery_db_path):
        async def _test():
            await init_db()
            await init_db()

        _run(_test())

    def test_create_build_returns_all_fields(self, gallery_db_path):
        async def _test():
            await init_db()
            build = await create_build(**_make("Test House"))
            assert build is not None
            assert build["title"] == "Test House"
            assert build["caption"] == "a small red house"
            assert build["bricks"] == SAMPLE_BRICKS
            assert build["brick_count"] == 2
            assert build["stable"] is True
            assert "id" in build
            assert "created_at" in build

        _run(_test())

    def test_create_build_unique_ids(self, gallery_db_path):
        async def _test():
            await init_db()
            b1 = await create_build(**_make("A"))
            b2 = await create_build(**_make("B"))
            assert b1["id"] != b2["id"]
            assert len(b1["id"]) == 12

        _run(_test())

    def test_get_build_existing(self, gallery_db_path):
        async def _test():
            await init_db()
            created = await create_build(**_make("X"))
            fetched = await get_build(created["id"])
            assert fetched is not None
            assert fetched["id"] == created["id"]
            assert fetched["stable"] is True

        _run(_test())

    def test_get_build_nonexistent(self, gallery_db_path):
        async def _test():
            await init_db()
            assert await get_build("does_not_exist") is None

        _run(_test())

    def test_list_builds_returns_all(self, gallery_db_path):
        async def _test():
            await init_db()
            await create_build(**_make("A"))
            await create_build(**_make("B"))
            assert len(await list_builds()) == 2

        _run(_test())

    def test_list_builds_search_caption(self, gallery_db_path):
        async def _test():
            await init_db()
            await create_build(**_make("Red House", caption="red brick cottage"))
            await create_build(**_make("Blue Car", caption="compact blue vehicle"))
            builds = await list_builds(q="cottage")
            assert len(builds) == 1
            assert builds[0]["title"] == "Red House"

        _run(_test())

    def test_list_builds_sort_by_bricks(self, gallery_db_path):
        async def _test():
            await init_db()
            await create_build(**_make("Small", brick_count=5))
            await create_build(**_make("Big", brick_count=200))
            by_bricks = await list_builds(sort="bricks")
            assert by_bricks[0]["brick_count"] >= by_bricks[1]["brick_count"]

        _run(_test())

    def test_unstable_build_persists_flag(self, gallery_db_path):
        async def _test():
            await init_db()
            created = await create_build(**_make("Wobbly", stable=False))
            assert created["stable"] is False
            refetched = await get_build(created["id"])
            assert refetched["stable"] is False

        _run(_test())

    def test_update_star_running_average(self, gallery_db_path):
        async def _test():
            await init_db()
            build = await create_build(**_make("X"))
            bid = build["id"]
            assert (await update_star(bid, 5))["stars"] == pytest.approx(5.0)
            assert (await update_star(bid, 3))["stars"] == pytest.approx(4.0)
            assert (await update_star(bid, 1))["stars"] == pytest.approx(3.0)

        _run(_test())

    def test_update_star_nonexistent(self, gallery_db_path):
        async def _test():
            await init_db()
            assert await update_star("nonexistent", 5) is None

        _run(_test())
