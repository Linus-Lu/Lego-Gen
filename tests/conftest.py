import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")
    # Every test runs in dev mode so MockBrickPipeline is used. Individual
    # tests that need to override this must do so explicitly.
    os.environ.setdefault("LEGOGEN_DEV", "1")


def pytest_collection_modifyitems(config, items):
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    if not has_cuda:
        skip_gpu = pytest.mark.skip(reason="No CUDA GPU available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture
def gallery_db_path(tmp_path, monkeypatch):
    """Isolate gallery DB to a temp directory for test safety."""
    monkeypatch.setattr("backend.storage.gallery_db.DATA_DIR", tmp_path)
    monkeypatch.setattr(
        "backend.storage.gallery_db.DB_PATH", tmp_path / "test_gallery.db"
    )
    return tmp_path / "test_gallery.db"


@pytest.fixture
def reset_pipeline_singletons():
    """Force a fresh pipeline + stage1 mock on every test that needs it.

    The factories cache singletons at module level; tests that monkeypatch
    LEGOGEN_DEV or poke internals must not leak state across files.
    """
    import backend.inference.brick_pipeline as bp
    bp._brick_instance = None
    bp._stage1_instance = None
    yield
    bp._brick_instance = None
    bp._stage1_instance = None


@pytest.fixture
def seeded_palette(monkeypatch):
    """Pre-populate the lazy color palette so ALLOWED_COLORS works without colors.json."""
    from backend.brick import constants as const
    fake = {
        "C91A09": "Red",
        "FFFFFF": "White",
        "000000": "Black",
        "0055BF": "Blue",
        "FE8A18": "Orange",
        "237841": "Green",
        "720E0F": "Dark Red",
    }
    monkeypatch.setattr(const._lazy_palette, "_cache", fake, raising=False)
    return fake
