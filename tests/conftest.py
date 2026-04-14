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
