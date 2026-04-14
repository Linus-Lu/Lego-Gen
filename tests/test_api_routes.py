"""Tests for FastAPI API routes — generate, validate, gallery, health."""

import io
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

httpx = pytest.importorskip("httpx")
fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def dev_mode():
    """Run all API tests in dev mode so MockPipeline is used."""
    os.environ["LEGOGEN_DEV"] = "1"
    yield
    # Reset pipeline singletons so tests are independent
    import backend.inference.pipeline as p

    p._unified_instance = None
    p._stability_checker = None


@pytest.fixture
def client(gallery_db_path):
    """Create a TestClient with isolated gallery DB.

    gallery_db_path fixture (from conftest.py) patches DB_PATH and DATA_DIR
    to tmp_path before the app's lifespan starts.
    """
    from backend.app import app

    with TestClient(app) as c:
        yield c


def _make_png_bytes():
    """Create a minimal valid PNG image in memory."""
    try:
        from PIL import Image

        img = Image.new("RGB", (64, 64), (255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf.read()
    except ImportError:
        # Fallback: minimal 1x1 red PNG (67 bytes)
        import base64

        minimal_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
            "nGP4z8BQDwAEgAF/pooBPQAAAABJRU5ErkJggg=="
        )
        return minimal_png


# ── TestHealth ───────────────────────────────────────────────────────


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ── TestGenerate ─────────────────────────────────────────────────────


class TestGenerate:
    def test_valid_image_returns_200(self, client):
        png = _make_png_bytes()
        resp = client.post(
            "/api/generate",
            files={"image": ("test.png", png, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "description" in data
        assert "steps" in data
        assert "metadata" in data
        assert "validation" in data

    def test_non_image_mime_returns_400(self, client):
        resp = client.post(
            "/api/generate",
            files={"image": ("test.pdf", b"not an image", "application/pdf")},
        )
        assert resp.status_code == 400

    def test_corrupted_image_dev_fallback(self, client):
        """In dev mode, corrupted image data should trigger fallback."""
        resp = client.post(
            "/api/generate",
            files={"image": ("test.png", b"not valid png data", "image/png")},
        )
        # Dev mode creates a dummy image, so should succeed
        assert resp.status_code == 200


# ── TestGenerateFromText ─────────────────────────────────────────────


class TestGenerateFromText:
    def test_valid_prompt(self, client):
        resp = client.post("/api/generate-from-text", data={"prompt": "a red house"})
        assert resp.status_code == 200
        data = resp.json()
        assert "description" in data
        assert "steps" in data

    def test_empty_prompt_rejected(self, client):
        resp = client.post("/api/generate-from-text", data={"prompt": ""})
        assert resp.status_code in (400, 422)  # 400 from handler or 422 from FastAPI validation

    def test_whitespace_only_prompt_returns_400(self, client):
        resp = client.post("/api/generate-from-text", data={"prompt": "   "})
        assert resp.status_code == 400


# ── TestGenerateBricks ───────────────────────────────────────────────


class TestGenerateBricks:
    def test_no_input_returns_400(self, client):
        resp = client.post("/api/generate-bricks", data={"prompt": ""})
        assert resp.status_code == 400


# ── TestValidate ─────────────────────────────────────────────────────


class TestValidate:
    def test_valid_description(self, client, make_desc):
        desc = make_desc()
        resp = client.post("/api/validate", json=desc)
        assert resp.status_code == 200
        data = resp.json()
        assert "score" in data
        assert "checks" in data
        assert "summary" in data
        assert isinstance(data["score"], int)

    def test_empty_description_low_score(self, client):
        resp = client.post("/api/validate", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["score"] <= 30  # empty desc should score very low


# ── TestGallery ──────────────────────────────────────────────────────


class TestGallery:
    def test_list_empty_initially(self, client):
        resp = client.get("/api/gallery")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_create_build(self, client):
        payload = {
            "title": "My House",
            "description_json": json.dumps(
                {"category": "City", "complexity": "simple", "total_parts": 30}
            ),
            "thumbnail_b64": "",
        }
        resp = client.post("/api/gallery", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["title"] == "My House"
        assert data["category"] == "City"
        assert "id" in data

    def test_get_build(self, client):
        payload = {
            "title": "Get Test",
            "description_json": "{}",
        }
        created = client.post("/api/gallery", json=payload).json()
        resp = client.get(f"/api/gallery/{created['id']}")
        assert resp.status_code == 200
        assert resp.json()["title"] == "Get Test"

    def test_get_nonexistent_returns_404(self, client):
        resp = client.get("/api/gallery/nonexistent_id")
        assert resp.status_code == 404

    def test_star_build(self, client):
        payload = {"title": "Star Test", "description_json": "{}"}
        created = client.post("/api/gallery", json=payload).json()
        resp = client.patch(
            f"/api/gallery/{created['id']}/star", json={"stars": 5}
        )
        assert resp.status_code == 200
        assert resp.json()["stars"] == pytest.approx(5.0)

    def test_star_nonexistent_returns_404(self, client):
        resp = client.patch("/api/gallery/nonexistent/star", json={"stars": 3})
        assert resp.status_code == 404

    def test_filter_by_category(self, client):
        client.post(
            "/api/gallery",
            json={
                "title": "City Build",
                "description_json": json.dumps({"category": "City"}),
            },
        )
        client.post(
            "/api/gallery",
            json={
                "title": "Space Build",
                "description_json": json.dumps({"category": "Space"}),
            },
        )
        resp = client.get("/api/gallery", params={"category": "City"})
        assert resp.status_code == 200
        builds = resp.json()
        assert all(b["category"] == "City" for b in builds)

    def test_search_query(self, client):
        client.post(
            "/api/gallery",
            json={"title": "Red Castle", "description_json": "{}"},
        )
        client.post(
            "/api/gallery",
            json={"title": "Blue Car", "description_json": "{}"},
        )
        resp = client.get("/api/gallery", params={"q": "Castle"})
        assert resp.status_code == 200
        builds = resp.json()
        assert len(builds) == 1
        assert "Castle" in builds[0]["title"]
