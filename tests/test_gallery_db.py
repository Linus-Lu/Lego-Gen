"""Tests for backend.storage.gallery_db — async SQLite gallery CRUD."""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import aiosqlite  # noqa: F401

    _AIOSQLITE = True
except ImportError:
    _AIOSQLITE = False

pytestmark = pytest.mark.skipif(not _AIOSQLITE, reason="aiosqlite not installed")

from backend.storage.gallery_db import (
    create_build,
    get_build,
    init_db,
    list_builds,
    update_star,
)


def _run(coro):
    """Helper to run an async coroutine from a sync test."""
    return asyncio.run(coro)


class TestGalleryDB:
    """All gallery_db tests use the gallery_db_path fixture for isolation."""

    def test_init_db_idempotent(self, gallery_db_path):
        """Two init_db calls should not error."""

        async def _test():
            await init_db()
            await init_db()

        _run(_test())

    def test_create_build_returns_all_fields(self, gallery_db_path):
        async def _test():
            await init_db()
            build = await create_build(
                title="Test House",
                category="City",
                complexity="simple",
                parts_count=20,
                description_json='{"object": "House"}',
                thumbnail_b64="abc123",
            )
            assert build is not None
            assert build["title"] == "Test House"
            assert build["category"] == "City"
            assert build["complexity"] == "simple"
            assert build["parts_count"] == 20
            assert build["description_json"] == '{"object": "House"}'
            assert build["thumbnail_b64"] == "abc123"
            assert "id" in build
            assert "created_at" in build

        _run(_test())

    def test_create_build_unique_ids(self, gallery_db_path):
        async def _test():
            await init_db()
            b1 = await create_build("A", "Cat", "simple", 1, "{}", "")
            b2 = await create_build("B", "Cat", "simple", 2, "{}", "")
            assert b1["id"] != b2["id"]
            assert len(b1["id"]) == 12  # uuid hex[:12]

        _run(_test())

    def test_get_build_existing(self, gallery_db_path):
        async def _test():
            await init_db()
            created = await create_build("X", "Tech", "advanced", 100, "{}", "")
            fetched = await get_build(created["id"])
            assert fetched is not None
            assert fetched["id"] == created["id"]
            assert fetched["title"] == "X"

        _run(_test())

    def test_get_build_nonexistent(self, gallery_db_path):
        async def _test():
            await init_db()
            result = await get_build("does_not_exist")
            assert result is None

        _run(_test())

    def test_list_builds_returns_all(self, gallery_db_path):
        async def _test():
            await init_db()
            await create_build("A", "City", "simple", 10, "{}", "")
            await create_build("B", "Space", "advanced", 50, "{}", "")
            builds = await list_builds()
            assert len(builds) == 2

        _run(_test())

    def test_list_builds_filter_category(self, gallery_db_path):
        async def _test():
            await init_db()
            await create_build("A", "City", "simple", 10, "{}", "")
            await create_build("B", "Space", "advanced", 50, "{}", "")
            builds = await list_builds(category="City")
            assert len(builds) == 1
            assert builds[0]["category"] == "City"

        _run(_test())

    def test_list_builds_search_query(self, gallery_db_path):
        async def _test():
            await init_db()
            await create_build("Red House", "City", "simple", 10, "{}", "")
            await create_build("Blue Car", "City", "simple", 15, "{}", "")
            builds = await list_builds(q="House")
            assert len(builds) == 1
            assert "House" in builds[0]["title"]

        _run(_test())

    def test_list_builds_sort_options(self, gallery_db_path):
        async def _test():
            await init_db()
            await create_build("Small", "City", "simple", 5, "{}", "")
            await create_build("Big", "City", "advanced", 200, "{}", "")

            by_parts = await list_builds(sort="parts")
            assert by_parts[0]["parts_count"] >= by_parts[1]["parts_count"]

        _run(_test())

    def test_update_star_running_average(self, gallery_db_path):
        async def _test():
            await init_db()
            build = await create_build("X", "City", "simple", 10, "{}", "")
            bid = build["id"]

            # First star: avg = (0*0 + 5) / (0 + 1) = 5.0
            updated = await update_star(bid, 5)
            assert updated["stars"] == pytest.approx(5.0)

            # Second star: avg = (5.0*1 + 3) / (1 + 1) = 4.0
            updated = await update_star(bid, 3)
            assert updated["stars"] == pytest.approx(4.0)

            # Third star: avg = (4.0*2 + 1) / (2 + 1) = 3.0
            updated = await update_star(bid, 1)
            assert updated["stars"] == pytest.approx(3.0)

        _run(_test())

    def test_update_star_nonexistent(self, gallery_db_path):
        async def _test():
            await init_db()
            result = await update_star("nonexistent", 5)
            assert result is None

        _run(_test())
