"""SQLite-backed gallery storage using aiosqlite."""

import json
import uuid
from typing import Optional

import aiosqlite

from backend.config import DATA_DIR

DB_PATH = DATA_DIR / "gallery.db"


async def init_db() -> None:
    """Create the builds table if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS builds (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT '',
                complexity TEXT NOT NULL DEFAULT 'medium',
                parts_count INTEGER NOT NULL DEFAULT 0,
                description_json TEXT NOT NULL,
                thumbnail_b64 TEXT NOT NULL DEFAULT '',
                stars REAL NOT NULL DEFAULT 0,
                star_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        await db.commit()


async def list_builds(
    category: Optional[str] = None,
    sort: str = "newest",
    q: Optional[str] = None,
) -> list[dict]:
    """List builds with optional filtering and sorting."""
    clauses: list[str] = []
    params: list[str] = []

    if category:
        clauses.append("category = ?")
        params.append(category)
    if q:
        clauses.append("(title LIKE ? OR category LIKE ?)")
        params.extend([f"%{q}%", f"%{q}%"])

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    order_map = {
        "newest": "created_at DESC",
        "stars": "stars DESC",
        "parts": "parts_count DESC",
    }
    order = order_map.get(sort, "created_at DESC")

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            f"SELECT * FROM builds {where} ORDER BY {order}", params
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def create_build(
    title: str,
    category: str,
    complexity: str,
    parts_count: int,
    description_json: str,
    thumbnail_b64: str,
) -> dict:
    """Insert a new build and return it."""
    build_id = uuid.uuid4().hex[:12]
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO builds (id, title, category, complexity, parts_count, description_json, thumbnail_b64)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (build_id, title, category, complexity, parts_count, description_json, thumbnail_b64),
        )
        await db.commit()

    return await get_build(build_id)  # type: ignore[return-value]


async def get_build(build_id: str) -> Optional[dict]:
    """Get a single build by ID."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM builds WHERE id = ?", (build_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None


async def update_star(build_id: str, stars: int) -> Optional[dict]:
    """Add a star rating using running average."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT stars, star_count FROM builds WHERE id = ?", (build_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None

        old_avg = row["stars"]
        count = row["star_count"]
        new_avg = (old_avg * count + stars) / (count + 1)

        await db.execute(
            "UPDATE builds SET stars = ?, star_count = ? WHERE id = ?",
            (new_avg, count + 1, build_id),
        )
        await db.commit()

    return await get_build(build_id)
