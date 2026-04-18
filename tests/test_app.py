"""Coverage for backend/app.py — health route, CORS env-var parsing, lifespan (dev mode only)."""

import importlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

httpx = pytest.importorskip("httpx")
fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient


def _reload_app():
    """Reload backend.app so CORS env-var parsing re-runs at import time."""
    import backend.app
    importlib.reload(backend.app)
    return backend.app.app


@pytest.fixture(autouse=True)
def ensure_dev_mode(monkeypatch):
    monkeypatch.setenv("LEGOGEN_DEV", "1")
    yield


def test_health_endpoint(gallery_db_path):
    app = _reload_app()
    with TestClient(app) as c:
        resp = c.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_default_cors_origins(monkeypatch, gallery_db_path):
    monkeypatch.delenv("LEGOGEN_CORS_ORIGINS", raising=False)
    app = _reload_app()
    from fastapi.middleware.cors import CORSMiddleware
    cors = next(m for m in app.user_middleware if m.cls is CORSMiddleware)
    origins = cors.kwargs["allow_origins"]
    assert origins == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


def test_custom_cors_origins_env(monkeypatch, gallery_db_path):
    monkeypatch.setenv("LEGOGEN_CORS_ORIGINS", "https://a.example,https://b.example")
    app = _reload_app()
    from fastapi.middleware.cors import CORSMiddleware
    cors = next(m for m in app.user_middleware if m.cls is CORSMiddleware)
    origins = cors.kwargs["allow_origins"]
    assert origins == ["https://a.example", "https://b.example"]


def test_cors_origins_ignores_empty_entries(monkeypatch, gallery_db_path):
    monkeypatch.setenv("LEGOGEN_CORS_ORIGINS", "https://a.example, ,https://b.example")
    app = _reload_app()
    from fastapi.middleware.cors import CORSMiddleware
    cors = next(m for m in app.user_middleware if m.cls is CORSMiddleware)
    origins = cors.kwargs["allow_origins"]
    assert origins == ["https://a.example", "https://b.example"]


def test_lifespan_dev_mode_runs(capsys, gallery_db_path):
    """Startup in dev mode prints the ready banner and does not preload models."""
    app = _reload_app()
    with TestClient(app):
        pass
    out = capsys.readouterr().out
    assert "dev mode" in out.lower()
    assert "preloading" not in out.lower()


def test_lifespan_prod_mode_preloads_models(monkeypatch, capsys, gallery_db_path):
    monkeypatch.setenv("LEGOGEN_DEV", "0")
    import backend.app

    importlib.reload(backend.app)
    preload_calls = []
    monkeypatch.setattr(backend.app, "_preload_models", lambda: preload_calls.append("preload"))

    with TestClient(backend.app.app):
        pass

    out = capsys.readouterr().out.lower()
    assert preload_calls == ["preload"]
    assert "preloading models" in out
    assert "models loaded" in out


def test_preload_models_calls_both_factories(monkeypatch):
    import backend.app as app_mod
    import backend.inference.brick_pipeline as bp

    calls = []
    monkeypatch.setattr(bp, "get_brick_pipeline", lambda: calls.append("brick") or object())
    monkeypatch.setattr(bp, "_get_stage1_pipeline", lambda: calls.append("stage1") or object())

    app_mod._preload_models()

    assert calls == ["brick", "stage1"]
