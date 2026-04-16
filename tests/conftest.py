import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")


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
