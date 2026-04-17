"""Route-level validation tests for /api/gallery endpoints.

test_gallery_db.py covers the SQLite functions directly; these tests cover
the Pydantic validation and HTTP status codes on the FastAPI router.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

httpx = pytest.importorskip("httpx")
fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient


@pytest.fixture
def client(gallery_db_path):
    os.environ["LEGOGEN_DEV"] = "1"
    from backend.app import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def a_build(client):
    resp = client.post(
        "/api/gallery",
        json={
            "title": "Seed",
            "caption": "",
            "bricks": "1x1 (0,0,0) #FFFFFF",
            "brick_count": 1,
        },
    )
    assert resp.status_code == 201
    return resp.json()


def test_create_missing_title_returns_422(client):
    resp = client.post("/api/gallery", json={"bricks": "1x1 (0,0,0) #FFFFFF"})
    assert resp.status_code == 422


def test_create_title_too_long_returns_422(client):
    resp = client.post(
        "/api/gallery",
        json={
            "title": "x" * 201,
            "caption": "",
            "bricks": "1x1 (0,0,0) #FFFFFF",
        },
    )
    assert resp.status_code == 422


def test_create_bricks_over_max_returns_422(client):
    resp = client.post(
        "/api/gallery",
        json={
            "title": "Big",
            "caption": "",
            "bricks": "x" * 200_001,
        },
    )
    assert resp.status_code == 422


def test_create_thumbnail_over_max_returns_422(client):
    resp = client.post(
        "/api/gallery",
        json={
            "title": "Thumb",
            "caption": "",
            "bricks": "1x1 (0,0,0) #FFFFFF",
            "thumbnail_b64": "x" * 400_001,
        },
    )
    assert resp.status_code == 422


def test_create_whitespace_only_bricks_returns_400(client):
    resp = client.post(
        "/api/gallery",
        json={"title": "Blank", "caption": "", "bricks": "   \n  "},
    )
    assert resp.status_code == 400


def test_star_nonexistent_returns_404(client):
    resp = client.patch("/api/gallery/does-not-exist/star", json={"stars": 3})
    assert resp.status_code == 404


def test_star_out_of_range_returns_422(client, a_build):
    resp = client.patch(
        f"/api/gallery/{a_build['id']}/star", json={"stars": 10}
    )
    assert resp.status_code == 422
    resp = client.patch(
        f"/api/gallery/{a_build['id']}/star", json={"stars": 0}
    )
    assert resp.status_code == 422


def test_list_with_search_and_sort(client, a_build):
    resp = client.get("/api/gallery?q=Seed&sort=bricks")
    assert resp.status_code == 200
    items = resp.json()
    assert any(b["id"] == a_build["id"] for b in items)
