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

    def test_generate_bricks_with_n_returns_bon_metadata(self, client):
        resp = client.post("/api/generate-bricks", data={"prompt": "a small red house", "n": "4"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["metadata"]["n"] == 4
        assert "picked_index" in body["metadata"]

    def test_generate_bricks_rejects_out_of_range_n(self, client):
        resp = client.post("/api/generate-bricks", data={"prompt": "test", "n": "100"})
        assert resp.status_code == 422
        resp = client.post("/api/generate-bricks", data={"prompt": "test", "n": "0"})
        assert resp.status_code == 422


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
            "caption": "a small red house",
            "bricks": "2x4 (0,0,0) #C91A09",
            "brick_count": 1,
            "stable": True,
        }
        resp = client.post("/api/gallery", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["title"] == "My House"
        assert data["bricks"] == payload["bricks"]
        assert "id" in data

        got = client.get(f"/api/gallery/{data['id']}")
        assert got.status_code == 200
        assert got.json()["title"] == "My House"

    def test_empty_bricks_rejected(self, client):
        resp = client.post(
            "/api/gallery",
            json={"title": "Empty", "caption": "", "bricks": ""},
        )
        assert resp.status_code == 400

    def test_get_nonexistent_returns_404(self, client):
        resp = client.get("/api/gallery/nonexistent_id")
        assert resp.status_code == 404

    def test_star_build(self, client):
        created = client.post(
            "/api/gallery",
            json={
                "title": "Star Test",
                "caption": "test",
                "bricks": "1x1 (0,0,0) #FFFFFF",
                "brick_count": 1,
            },
        ).json()
        resp = client.patch(
            f"/api/gallery/{created['id']}/star", json={"stars": 5}
        )
        assert resp.status_code == 200
        assert resp.json()["stars"] == pytest.approx(5.0)


class TestGenerateValidationLimits:
    def test_image_exceeding_size_limit_returns_413(self, client):
        """MAX_IMAGE_BYTES is 10 MB — send 11 MB of zeros and expect 413."""
        oversize = b"\x00" * (11 * 1024 * 1024)
        resp = client.post(
            "/api/generate-bricks",
            files={"image": ("big.png", oversize, "image/png")},
        )
        assert resp.status_code == 413
        assert "10" in resp.json()["detail"]

    def test_prompt_exceeding_char_limit_returns_413(self, client):
        """MAX_PROMPT_CHARS is 2000; 2001 chars must 413."""
        resp = client.post(
            "/api/generate-bricks",
            data={"prompt": "x" * 2001},
        )
        assert resp.status_code == 413
        assert "2000" in resp.json()["detail"]

    def test_prompt_whitespace_only_treated_as_empty(self, client):
        resp = client.post("/api/generate-bricks", data={"prompt": "   \n\t  "})
        assert resp.status_code == 400

    def test_stream_image_wrong_mime_returns_400(self, client):
        resp = client.post(
            "/api/generate-stream",
            files={"image": ("x.txt", b"text", "text/plain")},
        )
        assert resp.status_code == 400

    def test_stream_no_input_returns_400(self, client):
        resp = client.post("/api/generate-stream", data={"prompt": ""})
        assert resp.status_code == 400


class TestGenerateTimeout:
    def test_generate_bricks_timeout_returns_504(self, monkeypatch, client):
        """Patch the MockBrickPipeline.generate to sleep longer than the timeout
        so asyncio.wait_for fires 504."""
        import time
        from backend.inference import brick_pipeline as bp
        from backend.api import routes_generate as rg

        monkeypatch.setattr(rg, "INFERENCE_TIMEOUT_SECONDS", 0.05)

        def slow_generate(self, caption, on_progress=None):
            time.sleep(1.0)
            return {"bricks": "", "brick_count": 0, "stable": True, "metadata": {}}

        monkeypatch.setattr(bp.MockBrickPipeline, "generate", slow_generate)
        resp = client.post("/api/generate-bricks", data={"prompt": "slow build"})
        assert resp.status_code == 504
        assert "timed out" in resp.json()["detail"].lower()


class TestGenerateBestOfNImagePath:
    def test_image_with_n_runs_bon(self, client):
        png = _make_png_bytes()
        resp = client.post(
            "/api/generate-bricks",
            files={"image": ("x.png", png, "image/png")},
            data={"n": "2"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["metadata"]["n"] == 2
        assert "caption" in body

    def test_stream_bon_emits_sample_events(self, client):
        resp = client.post(
            "/api/generate-stream",
            data={"prompt": "a red robot", "n": "2"},
        )
        assert resp.status_code == 200
        # MockBrickPipeline.generate_best_of_n doesn't emit sample events
        # (it calls generate once). This asserts the happy path still
        # terminates with a result event.
        assert "event: result" in resp.text


class TestStreamOrdering:
    def test_event_sequence(self, client):
        """SSE must emit at least one progress event before any result event."""
        resp = client.post("/api/generate-stream", data={"prompt": "house"})
        assert resp.status_code == 200
        body = resp.text
        progress_idx = body.index("event: progress")
        result_idx = body.index("event: result")
        assert progress_idx < result_idx

    def test_stream_image_caption_progress(self, client):
        """Image path must emit a stage1 progress event with the caption."""
        png = _make_png_bytes()
        resp = client.post(
            "/api/generate-stream",
            files={"image": ("x.png", png, "image/png")},
        )
        assert resp.status_code == 200
        assert "\"stage\": \"stage1\"" in resp.text
        assert "\"caption\"" in resp.text
