"""Tests for FastAPI routes and application endpoints."""

import io
import os

import pytest

# Enable dev mode before importing the app so the mock pipeline is used.
os.environ["LEGOGEN_DEV"] = "1"

from fastapi.testclient import TestClient
from backend.app import app


@pytest.fixture(autouse=True)
def _reset_pipeline():
    """Reset pipeline singleton between tests."""
    yield
    import backend.inference.pipeline as p
    p._pipeline_instance = None


client = TestClient(app)


# ── Health endpoint ──────────────────────────────────────────────────

def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── CORS headers ─────────────────────────────────────────────────────

def test_cors_headers():
    resp = client.options(
        "/api/generate-bricks",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" in resp.headers


# ── Generate bricks with prompt ──────────────────────────────────────

def test_generate_with_prompt():
    resp = client.post(
        "/api/generate-bricks",
        data={"prompt": "a small red house"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "bricks" in body
    assert "caption" in body
    assert body["caption"] == "a small red house"
    assert "brick_count" in body
    assert body["brick_count"] > 0
    assert "stable" in body
    assert "metadata" in body


# ── Generate bricks with image ───────────────────────────────────────

def test_generate_with_image():
    from PIL import Image

    img = Image.new("RGB", (64, 64), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    resp = client.post(
        "/api/generate-bricks",
        files={"image": ("test.png", buf, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "bricks" in body
    assert body["brick_count"] > 0


# ── Error: no input ──────────────────────────────────────────────────

def test_generate_no_input():
    resp = client.post("/api/generate-bricks")
    assert resp.status_code == 400


def test_generate_empty_prompt():
    resp = client.post(
        "/api/generate-bricks",
        data={"prompt": "   "},
    )
    assert resp.status_code == 400


# ── Error: invalid content type ──────────────────────────────────────

def test_generate_invalid_content_type():
    resp = client.post(
        "/api/generate-bricks",
        files={"image": ("test.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 400


# ── Error: corrupt image ─────────────────────────────────────────────

def test_generate_corrupt_image():
    resp = client.post(
        "/api/generate-bricks",
        files={"image": ("bad.png", b"\x89PNG\r\n\x1a\ngarbage", "image/png")},
    )
    assert resp.status_code == 400
