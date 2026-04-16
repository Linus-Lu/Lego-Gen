"""Shared training utilities: seeding, W&B setup."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_wandb(project_name: str = "legogen", config: dict | None = None):
    """Initialize Weights & Biases logging."""
    try:
        import wandb
        wandb.init(project=project_name, config=config or {})
        return True
    except ImportError:
        print("wandb not installed, skipping logging")
        return False
