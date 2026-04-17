"""Coverage for backend/config.py — constants and the torch-import guard."""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_legogen_dev_true_by_default(monkeypatch):
    monkeypatch.setenv("LEGOGEN_DEV", "1")
    import backend.config as cfg
    importlib.reload(cfg)
    assert cfg.LEGOGEN_DEV is True


def test_legogen_dev_false_when_env_is_zero(monkeypatch):
    monkeypatch.setenv("LEGOGEN_DEV", "0")
    import backend.config as cfg
    importlib.reload(cfg)
    assert cfg.LEGOGEN_DEV is False
    # Reset for downstream tests.
    monkeypatch.setenv("LEGOGEN_DEV", "1")
    importlib.reload(cfg)


def test_paths_are_absolute():
    import backend.config as cfg
    assert cfg.PROJECT_ROOT.is_absolute()
    assert cfg.DATA_DIR.is_absolute()
    assert cfg.CHECKPOINT_DIR.parts[-3:] == ("backend", "models", "checkpoints")


def test_coco_category_mapping_nonempty():
    import backend.config as cfg
    assert "chair" in cfg.COCO_TO_ST2B_CATEGORY
    assert cfg.COCO_TO_ST2B_CATEGORY["couch"] == "sofa"


def test_device_branch_with_torch_absent(monkeypatch):
    """Simulate `import torch` raising ImportError and reload config."""
    real_torch = sys.modules.get("torch")
    monkeypatch.setitem(sys.modules, "torch", None)
    import backend.config as cfg
    importlib.reload(cfg)
    assert cfg.DEVICE == "cpu"
    assert cfg.USE_BF16 is False
    # Restore the real torch module so the reload re-uses it instead of
    # re-executing torch's top-level (which would hit double-registration).
    if real_torch is not None:
        monkeypatch.setitem(sys.modules, "torch", real_torch)
    else:
        monkeypatch.delitem(sys.modules, "torch", raising=False)
    importlib.reload(cfg)


def test_training_hyperparameters_set():
    """Guard against accidentally zero-ing a hyperparameter."""
    import backend.config as cfg
    assert cfg.STAGE1_LORA_R == 32
    assert cfg.BRICK_LORA_R == 32
    assert cfg.STAGE1_NUM_EPOCHS >= 1
    assert cfg.BRICK_NUM_EPOCHS >= 1
