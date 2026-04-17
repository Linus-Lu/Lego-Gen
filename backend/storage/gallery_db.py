"""SQLite-backed gallery storage using aiosqlite.

Each gallery build stores the Stage 2 brick sequence (newline-separated
``HxW (x,y,z) #RRGGBB`` lines) plus the Stage 1 caption it was generated
from.
"""

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
                caption TEXT NOT NULL DEFAULT '',
                bricks TEXT NOT NULL,
                brick_count INTEGER NOT NULL DEFAULT 0,
                stable INTEGER NOT NULL DEFAULT 1,
                thumbnail_b64 TEXT NOT NULL DEFAULT '',
                stars REAL NOT NULL DEFAULT 0,
                star_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        await db.commit()


async def list_builds(
    sort: str = "newest",
    q: Optional[str] = None,
) -> list[dict]:
    """List builds with optional search and sorting."""
    clauses: list[str] = []
    params: list[str] = []

    if q:
        clauses.append("(title LIKE ? OR caption LIKE ?)")
        params.extend([f"%{q}%", f"%{q}%"])

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    order_map = {
        "newest": "created_at DESC",
        "stars": "stars DESC",
        "bricks": "brick_count DESC",
    }
    order = order_map.get(sort, "created_at DESC")

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            f"SELECT * FROM builds {where} ORDER BY {order}", params
        )
        rows = await cursor.fetchall()
        results = [dict(row) for row in rows]
        for r in results:
            r["stable"] = bool(r["stable"])
        return results


async def create_build(
    title: str,
    caption: str,
    bricks: str,
    brick_count: int,
    stable: bool,
    thumbnail_b64: str,
) -> dict:
    """Insert a new build and return it."""
    build_id = uuid.uuid4().hex[:12]
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO builds
                   (id, title, caption, bricks, brick_count, stable, thumbnail_b64)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (build_id, title, caption, bricks, brick_count, int(stable), thumbnail_b64),
        )
        await db.commit()

    return await get_build(build_id)  # type: ignore[return-value]


async def get_build(build_id: str) -> Optional[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM builds WHERE id = ?", (build_id,))
        row = await cursor.fetchone()
        if not row:
            return None
        result = dict(row)
        result["stable"] = bool(result["stable"])
        return result


async def update_star(build_id: str, stars: int) -> Optional[dict]:
    """Add a star rating using running average.

    Atomic read-modify-write via a single UPDATE — the expression evaluates
    against the row's current values under SQLite's row-level write lock,
    so concurrent PATCH calls serialize correctly instead of racing on a
    stale SELECT.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """UPDATE builds
                  SET stars = (stars * star_count + ?) / (star_count + 1.0),
                      star_count = star_count + 1
                WHERE id = ?""",
            (stars, build_id),
        )
        await db.commit()
        if cursor.rowcount == 0:
            return None

    return await get_build(build_id)
