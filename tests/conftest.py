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


# ── Shared fixtures ──────────────────────────────────────────────────


@pytest.fixture
def make_desc():
    """Factory fixture that builds a minimal valid LEGO description dict.

    Usage: desc = make_desc(total_parts=20, complexity="advanced")
    """

    def _make_desc(**overrides):
        base = {
            "set_id": "test-001",
            "object": "Test Build",
            "category": "City",
            "subcategory": "Test",
            "complexity": "simple",
            "total_parts": 10,
            "dominant_colors": ["Red"],
            "dimensions_estimate": {
                "width": "small",
                "height": "small",
                "depth": "small",
            },
            "subassemblies": [
                {
                    "name": "base",
                    "type": "Plates",
                    "parts": [
                        {
                            "part_id": "3020",
                            "name": "Plate 2x4",
                            "category": "Plates",
                            "color": "Red",
                            "color_hex": "#C91A09",
                            "is_trans": False,
                            "quantity": 5,
                        }
                    ],
                    "spatial": {
                        "position": "bottom",
                        "orientation": "flat",
                        "connects_to": ["walls"],
                    },
                },
                {
                    "name": "walls",
                    "type": "Bricks",
                    "parts": [
                        {
                            "part_id": "3001",
                            "name": "Brick 2x4",
                            "category": "Bricks",
                            "color": "White",
                            "color_hex": "#FFFFFF",
                            "is_trans": False,
                            "quantity": 5,
                        }
                    ],
                    "spatial": {
                        "position": "center",
                        "orientation": "upright",
                        "connects_to": ["base"],
                    },
                },
            ],
            "build_hints": ["Start with the base"],
        }
        base.update(overrides)
        return base

    return _make_desc


@pytest.fixture
def gallery_db_path(tmp_path, monkeypatch):
    """Isolate gallery DB to a temp directory for test safety."""
    monkeypatch.setattr("backend.storage.gallery_db.DATA_DIR", tmp_path)
    monkeypatch.setattr(
        "backend.storage.gallery_db.DB_PATH", tmp_path / "test_gallery.db"
    )
    return tmp_path / "test_gallery.db"
