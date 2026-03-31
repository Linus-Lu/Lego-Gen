import os
from pathlib import Path

# ── Dev Mode ───────────────────────────────────────────────────────────
LEGOGEN_DEV = os.environ.get("LEGOGEN_DEV", "1") == "1"

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
CACHE_DIR = DATA_DIR / "cache"
CHECKPOINT_DIR = BACKEND_DIR / "models" / "checkpoints"

# ── Rebrickable API ────────────────────────────────────────────────────
REBRICKABLE_API_KEY = os.environ.get("REBRICKABLE_API_KEY", "")
REBRICKABLE_BASE_URL = "https://rebrickable.com/api/v3/lego"
REBRICKABLE_RATE_LIMIT = 1.0  # seconds between requests

# ── Dataset ────────────────────────────────────────────────────────────
MIN_PARTS = 5
MAX_PARTS = 500
VAL_RATIO = 0.1
SPLITS_FILE = DATA_DIR / "splits.json"

# ── Model ──────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_SEQ_LENGTH = 1024

# ── QLoRA ──────────────────────────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
QUANTIZATION_BITS = 4

# ── Training ───────────────────────────────────────────────────────────
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
NUM_EPOCHS = 5
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
LOGGING_STEPS = 10
EVAL_STEPS = 200
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 3

# ── Inference ──────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 1024
NUM_BEAMS = 1
TEMPERATURE = 0.7
TOP_P = 0.9

# ── Device ─────────────────────────────────────────────────────────────
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
except ImportError:
    DEVICE = "cpu"
    USE_BF16 = False
