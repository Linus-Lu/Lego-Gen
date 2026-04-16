"""Tests for FastAPI API routes — generate-bricks, gallery, health."""

import io
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

httpx = pytest.importorskip("httpx")
fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def dev_mode():
    """Run all API tests in dev mode so MockBrickPipeline is used."""
    os.environ["LEGOGEN_DEV"] = "1"
    yield
    import backend.inference.brick_pipeline as bp
    bp._brick_instance = None
    bp._stage1_instance = None


@pytest.fixture
def client(gallery_db_path):
    """TestClient with an isolated gallery DB (see conftest.py fixture)."""
    from backend.app import app
    with TestClient(app) as c:
        yield c


def _make_png_bytes() -> bytes:
    from PIL import Image
    img = Image.new("RGB", (64, 64), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestGenerateBricks:
    def test_image_returns_bricks(self, client):
        png = _make_png_bytes()
        resp = client.post(
            "/api/generate-bricks",
            files={"image": ("test.png", png, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "bricks" in data
        assert "brick_count" in data
        assert "stable" in data
        assert "metadata" in data
        assert data["brick_count"] > 0

    def test_text_returns_bricks(self, client):
        resp = client.post("/api/generate-bricks", data={"prompt": "a red house"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["brick_count"] > 0
        assert isinstance(data["bricks"], str)

    def test_no_input_returns_400(self, client):
        resp = client.post("/api/generate-bricks", data={"prompt": ""})
        assert resp.status_code == 400

    def test_non_image_mime_returns_400(self, client):
        resp = client.post(
            "/api/generate-bricks",
            files={"image": ("test.pdf", b"not an image", "application/pdf")},
        )
        assert resp.status_code == 400

    def test_brick_lines_parse(self, client):
        """Each returned brick line should match the brick grammar regex."""
        import re
        brick_re = re.compile(r"^\d+x\d+ \(\d+,\d+,\d+\) #[0-9A-Fa-f]{6}$")
        resp = client.post("/api/generate-bricks", data={"prompt": "anything"})
        for line in resp.json()["bricks"].splitlines():
            assert brick_re.match(line), f"Bad brick line: {line!r}"


class TestGenerateStream:
    def test_image_stream_emits_events(self, client):
        png = _make_png_bytes()
        resp = client.post(
            "/api/generate-stream",
            files={"image": ("test.png", png, "image/png")},
        )
        assert resp.status_code == 200
        body = resp.text
        assert "event: progress" in body
        assert "event: brick" in body
        assert "event: result" in body

    def test_text_stream_emits_events(self, client):
        resp = client.post("/api/generate-stream", data={"prompt": "a small robot"})
        assert resp.status_code == 200
        body = resp.text
        assert "event: progress" in body
        assert "event: result" in body


class TestGallery:
    def test_list_empty_initially(self, client):
        resp = client.get("/api/gallery")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_create_and_get_build(self, client):
        payload = {
            "title": "My House",
            "description_json": "{}",
            "thumbnail_b64": "",
        }
        resp = client.post("/api/gallery", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["title"] == "My House"
        assert "id" in data

        got = client.get(f"/api/gallery/{data['id']}")
        assert got.status_code == 200
        assert got.json()["title"] == "My House"

    def test_get_nonexistent_returns_404(self, client):
        resp = client.get("/api/gallery/nonexistent_id")
        assert resp.status_code == 404

    def test_star_build(self, client):
        created = client.post(
            "/api/gallery",
            json={"title": "Star Test", "description_json": "{}"},
        ).json()
        resp = client.patch(
            f"/api/gallery/{created['id']}/star", json={"stars": 5}
        )
        assert resp.status_code == 200
        assert resp.json()["stars"] == pytest.approx(5.0)
