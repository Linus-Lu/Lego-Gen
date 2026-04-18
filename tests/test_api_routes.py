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
        assert "termination_reason" in data["metadata"]
        assert "outlines_enabled" in data["metadata"]

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

    def test_require_stable_accepts_stable_result(self, client):
        resp = client.post(
            "/api/generate-bricks",
            data={"prompt": "a red house", "require_stable": "true"},
        )
        assert resp.status_code == 200
        assert resp.json()["stable"] is True

    def test_require_stable_rejects_unstable_result(self, monkeypatch, client):
        import backend.inference.brick_pipeline as bp

        def unstable_generate(self, caption, on_progress=None):
            return {
                "bricks": "2x4 (0,0,0) #C91A09",
                "brick_count": 1,
                "stable": False,
                "metadata": {
                    "model_version": "mock-brick-v1",
                    "generation_time_ms": 1,
                    "rejections": 0,
                    "rollbacks": 0,
                },
            }

        monkeypatch.setattr(bp.MockBrickPipeline, "generate", unstable_generate)
        resp = client.post(
            "/api/generate-bricks",
            data={"prompt": "unstable", "require_stable": "true"},
        )
        assert resp.status_code == 422
        assert "stable" in resp.json()["detail"].lower()

    def test_generate_bricks_text_path_passes_progress_callback(self, monkeypatch, client):
        import backend.inference.brick_pipeline as bp

        seen = {"callback": None}

        def callback_generate(self, caption, on_progress=None):
            seen["callback"] = on_progress
            return {
                "bricks": "2x4 (0,0,0) #C91A09",
                "brick_count": 1,
                "stable": True,
                "metadata": {"model_version": "mock", "rejections": 0, "rollbacks": 0},
            }

        monkeypatch.setattr(bp.MockBrickPipeline, "generate", callback_generate)
        resp = client.post("/api/generate-bricks", data={"prompt": "callback test"})
        assert resp.status_code == 200
        assert seen["callback"] is not None

    def test_generate_bricks_image_path_passes_progress_callback(self, monkeypatch, client):
        import backend.inference.brick_pipeline as bp

        seen = {"callback": None}

        def callback_generate_from_image(self, image, on_progress=None):
            seen["callback"] = on_progress
            return {
                "bricks": "2x4 (0,0,0) #C91A09",
                "caption": "a red house",
                "brick_count": 1,
                "stable": True,
                "metadata": {"model_version": "mock", "rejections": 0, "rollbacks": 0},
            }

        monkeypatch.setattr(bp.MockBrickPipeline, "generate_from_image", callback_generate_from_image)
        resp = client.post(
            "/api/generate-bricks",
            files={"image": ("test.png", _make_png_bytes(), "image/png")},
        )
        assert resp.status_code == 200
        assert seen["callback"] is not None


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

    def test_stream_require_stable_emits_result_when_stable(self, client):
        resp = client.post(
            "/api/generate-stream",
            data={"prompt": "a small robot", "require_stable": "true"},
        )
        assert resp.status_code == 200
        assert "event: result" in resp.text

    def test_stream_require_stable_emits_error_when_unstable(self, monkeypatch, client):
        import backend.inference.brick_pipeline as bp

        def unstable_generate(self, caption, on_progress=None):
            return {
                "bricks": "2x4 (0,0,0) #C91A09",
                "brick_count": 1,
                "stable": False,
                "metadata": {
                    "model_version": "mock-brick-v1",
                    "generation_time_ms": 1,
                    "rejections": 0,
                    "rollbacks": 0,
                },
            }

        monkeypatch.setattr(bp.MockBrickPipeline, "generate", unstable_generate)
        resp = client.post(
            "/api/generate-stream",
            data={"prompt": "unstable", "require_stable": "true"},
        )
        assert resp.status_code == 200
        assert "event: error" in resp.text
        assert "stable-only requirement" in resp.text
        assert "event: result" not in resp.text


class TestExportLDraw:
    def test_export_ldr_returns_attachment(self, client):
        resp = client.post(
            "/api/export-ldr",
            json={
                "title": "My Build",
                "bricks": "2x4 (0,0,0) #C91A09\n1x2 (0,4,0) #0055BF",
            },
        )
        assert resp.status_code == 200
        assert "attachment" in resp.headers["content-disposition"]
        assert "My-Build.ldr" in resp.headers["content-disposition"]
        assert "0 FILE My-Build.ldr" in resp.text
        assert "3001.dat" in resp.text

    def test_export_ldr_rejects_empty_or_invalid_payload(self, client):
        resp = client.post("/api/export-ldr", json={"title": "Empty", "bricks": "   "})
        assert resp.status_code == 400

        resp = client.post("/api/export-ldr", json={"title": "Bad", "bricks": "garbage"})
        assert resp.status_code == 400
        assert "valid bricks" in resp.json()["detail"].lower()


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

    def test_invalid_brick_text_rejected(self, client):
        resp = client.post(
            "/api/gallery",
            json={"title": "Bad", "caption": "", "bricks": "nonsense"},
        )
        assert resp.status_code == 400
        assert "valid brick lines" in resp.json()["detail"]

    def test_invalid_geometry_rejected(self, client):
        resp = client.post(
            "/api/gallery",
            json={
                "title": "Overlap",
                "caption": "",
                "bricks": "2x4 (0,0,0) #C91A09\n2x4 (0,0,0) #0055BF",
            },
        )
        assert resp.status_code == 400
        assert "invalid placement" in resp.json()["detail"]

    def test_gallery_recomputes_brick_count_and_stability(self, monkeypatch, client):
        from backend.api import routes_gallery as rg

        monkeypatch.setattr(rg, "is_stable", lambda bricks: False)

        resp = client.post(
            "/api/gallery",
            json={
                "title": "Client Lies",
                "caption": "ignored metadata",
                "bricks": "2x4 (0,0,0) #C91A09",
                "brick_count": 99,
                "stable": True,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["brick_count"] == 1
        assert data["stable"] is False

    def test_get_nonexistent_returns_404(self, client):
        resp = client.get("/api/gallery/nonexistent_id")
        assert resp.status_code == 404

    def test_stable_is_bool_across_list_and_get(self, client):
        """Both endpoints must serialize ``stable`` as a JSON boolean."""
        created = client.post(
            "/api/gallery",
            json={
                "title": "Type Check",
                "caption": "test",
                "bricks": "1x1 (0,0,0) #FFFFFF",
                "brick_count": 1,
                "stable": True,
            },
        ).json()

        listed = client.get("/api/gallery").json()
        assert isinstance(listed, list) and listed, "expected at least one build"
        assert isinstance(listed[0]["stable"], bool)

        got = client.get(f"/api/gallery/{created['id']}").json()
        assert isinstance(got["stable"], bool)

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


class TestDecodeImageFallback:
    def test_garbled_image_bytes_fallback_in_dev(self, client):
        """In LEGOGEN_DEV=1, _decode_image falls back to a blank image when
        PIL can't parse the bytes — endpoint returns 200 rather than 400."""
        resp = client.post(
            "/api/generate-bricks",
            files={"image": ("bad.png", b"not-a-real-png", "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["brick_count"] > 0

    def test_decode_image_raises_400_when_not_dev(self, monkeypatch):
        """Production path: PIL failure raises HTTPException(400)."""
        from fastapi import HTTPException
        from backend.api import routes_generate as rg

        monkeypatch.setattr(rg, "LEGOGEN_DEV", False)
        with pytest.raises(HTTPException) as excinfo:
            rg._decode_image(b"not-a-real-png")
        assert excinfo.value.status_code == 400


class TestCancelToken:
    def test_check_callback_raises_when_set(self):
        """_CancelToken.check_callback raises _Cancelled once .set() has fired."""
        from backend.api.routes_generate import _CancelToken, _Cancelled

        tok = _CancelToken()
        # No-op when not set.
        tok.check_callback({"type": "brick"})
        assert tok.is_set is False
        tok.set()
        assert tok.is_set is True
        with pytest.raises(_Cancelled):
            tok.check_callback({"type": "brick"})


class TestStreamImageBestOfN:
    def test_stream_with_image_and_n_uses_from_image_bon(self, client):
        """Image + n>1 stream path hits _from_image_bon inside the thread (line 226)."""
        png = _make_png_bytes()
        resp = client.post(
            "/api/generate-stream",
            files={"image": ("x.png", png, "image/png")},
            data={"n": "2"},
        )
        assert resp.status_code == 200
        body = resp.text
        # The caption event is emitted by _from_image_bon itself.
        assert "\"caption\"" in body
        assert "event: result" in body


class TestFromImageBonDirect:
    def test_from_image_bon_with_no_progress_callback(self):
        """_from_image_bon(on_progress=None) takes the False branch at the
        caption check (covers branch 169->171)."""
        from backend.api.routes_generate import _from_image_bon
        from backend.inference.brick_pipeline import MockBrickPipeline
        from PIL import Image

        pipe = MockBrickPipeline()
        result = _from_image_bon(pipe, Image.new("RGB", (32, 32)), n=2, on_progress=None)
        assert "caption" in result
        assert result["metadata"]["n"] == 2

    def test_from_image_bon_forwards_should_cancel(self, monkeypatch):
        from backend.api.routes_generate import _from_image_bon
        from PIL import Image

        seen = {"stage1_should_cancel": None, "bon_should_cancel": None}

        class StubStage1:
            def describe(self, image, *, should_cancel=None):
                seen["stage1_should_cancel"] = should_cancel
                return "a red house"

        class StubPipeline:
            def generate_best_of_n(self, caption, n=1, on_progress=None, *, should_cancel=None):
                seen["bon_should_cancel"] = should_cancel
                return {
                    "bricks": "2x4 (0,0,0) #C91A09",
                    "brick_count": 1,
                    "stable": True,
                    "metadata": {"n": n},
                }

        monkeypatch.setattr("backend.inference.brick_pipeline._get_stage1_pipeline", lambda: StubStage1())
        should_cancel = lambda: False

        result = _from_image_bon(
            StubPipeline(),
            Image.new("RGB", (32, 32)),
            n=2,
            should_cancel=should_cancel,
        )

        assert result["caption"] == "a red house"
        assert seen["stage1_should_cancel"] is should_cancel
        assert seen["bon_should_cancel"] is should_cancel


class TestStreamRollbackAndUnknownEvents:
    def test_stream_emits_rejection_event(self, monkeypatch, client):
        """Mock generate() to push a rejection progress event through the SSE
        loop so the cumulative rejection counter is surfaced to the UI."""
        from backend.inference import brick_pipeline as bp

        def generate_with_rejection(self, caption, on_progress=None):
            if on_progress is not None:
                on_progress({"type": "rejection", "count": 2})
                on_progress({"type": "brick", "count": 1})
            return {
                "bricks": "2x4 (0,0,0) #C91A09",
                "brick_count": 1,
                "stable": True,
                "metadata": {"model_version": "mock", "rejections": 2},
            }

        monkeypatch.setattr(bp.MockBrickPipeline, "generate", generate_with_rejection)
        resp = client.post("/api/generate-stream", data={"prompt": "rejection test"})
        assert resp.status_code == 200
        assert "event: rejection" in resp.text
        assert "event: result" in resp.text

    def test_stream_emits_rollback_event(self, monkeypatch, client):
        """Mock generate() to push a rollback progress event through the SSE
        loop so the 'rollback' branch (lines 274-275) is covered."""
        from backend.inference import brick_pipeline as bp

        def generate_with_rollback(self, caption, on_progress=None):
            if on_progress is not None:
                on_progress({"type": "brick", "count": 1})
                on_progress({"type": "rollback", "count": 1})
            return {
                "bricks": "2x4 (0,0,0) #C91A09",
                "brick_count": 1,
                "stable": True,
                "metadata": {"model_version": "mock"},
            }

        monkeypatch.setattr(bp.MockBrickPipeline, "generate", generate_with_rollback)
        resp = client.post("/api/generate-stream", data={"prompt": "rollback test"})
        assert resp.status_code == 200
        assert "event: rollback" in resp.text
        assert "event: result" in resp.text

    def test_stream_ignores_unknown_event_types(self, monkeypatch, client):
        """Events with an unrecognised type fall through all elif branches and
        loop back to the top of the while (covers branch 269->247)."""
        from backend.inference import brick_pipeline as bp

        def generate_with_surprise(self, caption, on_progress=None):
            if on_progress is not None:
                on_progress({"type": "surprise", "payload": "ignored"})
                on_progress({"type": "brick", "count": 1})
            return {
                "bricks": "2x4 (0,0,0) #C91A09",
                "brick_count": 1,
                "stable": True,
                "metadata": {"model_version": "mock"},
            }

        monkeypatch.setattr(bp.MockBrickPipeline, "generate", generate_with_surprise)
        resp = client.post("/api/generate-stream", data={"prompt": "unknown event"})
        assert resp.status_code == 200
        # The stream completes and emits a result — "surprise" type is ignored.
        assert "event: result" in resp.text
        assert "event: surprise" not in resp.text

    def test_stream_emits_sample_event(self, monkeypatch, client):
        """Best-of-N pipelines can emit sample events; the SSE loop forwards
        them as 'sample' events (covers line 277)."""
        from backend.inference import brick_pipeline as bp

        def bon_with_sample(self, caption, n=1, strategy="rank", on_progress=None):
            if on_progress is not None:
                on_progress({"type": "sample", "index": 1, "of": n, "stable": True})
            return {
                "bricks": "2x4 (0,0,0) #C91A09",
                "brick_count": 1,
                "stable": True,
                "metadata": {"model_version": "mock", "n": n, "picked_index": 0, "stable_rate": 1.0},
            }

        monkeypatch.setattr(bp.MockBrickPipeline, "generate_best_of_n", bon_with_sample)
        resp = client.post("/api/generate-stream", data={"prompt": "sample test", "n": "2"})
        assert resp.status_code == 200
        assert "event: sample" in resp.text

    def test_stream_caption_skips_stage2_announce_when_already_announced(self, monkeypatch, client):
        """For text prompts announced_stage2 is True from the start; a stray
        caption event should take the False branch at 'if not announced_stage2'
        (covers branch 269->247)."""
        from backend.inference import brick_pipeline as bp

        def generate_with_caption_event(self, caption, on_progress=None):
            if on_progress is not None:
                # Emit a caption event even for a text-only call — the loop
                # must handle it without re-announcing stage 2.
                on_progress({"type": "caption", "caption": "synthetic caption"})
                on_progress({"type": "brick", "count": 1})
            return {
                "bricks": "2x4 (0,0,0) #C91A09",
                "brick_count": 1,
                "stable": True,
                "metadata": {"model_version": "mock"},
            }

        monkeypatch.setattr(bp.MockBrickPipeline, "generate", generate_with_caption_event)
        resp = client.post("/api/generate-stream", data={"prompt": "caption branch"})
        assert resp.status_code == 200
        # Stage2 announce appears exactly once (at the start of text-only stream).
        body = resp.text
        # The initial stage2 announce.
        stage2_announces = body.count("\"message\": \"Generating bricks...\"")
        assert stage2_announces == 1
