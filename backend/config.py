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
REBRICKABLE_RATE_LIMIT = 1.5  # seconds between requests (free tier is strict)

# ── Dataset ────────────────────────────────────────────────────────────
MIN_PARTS = 5
MAX_PARTS = 500
VAL_RATIO = 0.1
SPLITS_FILE = DATA_DIR / "splits.json"

# ── Model ──────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_SEQ_LENGTH = 1024

# ── QLoRA ──────────────────────────────────────────────────────────────
LORA_R = 32                # v1 was 16, higher rank = more capacity
LORA_ALPHA = 64            # v1 was 32, keep alpha = 2*r
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
QUANTIZATION_BITS = 4

# ── Training ───────────────────────────────────────────────────────────
LEARNING_RATE = 1e-4       # v1 was 2e-4, lower = more stable convergence
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
NUM_EPOCHS = 10            # v1 was 5, loss was still dropping
WARMUP_STEPS = 200         # v1 was 100, longer warmup for lower LR
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
LOGGING_STEPS = 10
EVAL_STEPS = 200
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 3

# ── Planner (text-to-JSON) ─────────────────────────────────────────
PLANNER_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
PLANNER_CHECKPOINT_DIR = BACKEND_DIR / "models" / "checkpoints" / "qwen-lego-planner-lora"
PLANNER_MAX_SEQ_LENGTH = 2048
PLANNER_LEARNING_RATE = 5e-5
PLANNER_NUM_EPOCHS = 10
PLANNER_WARMUP_STEPS = 500

# ── StableText2Brick dataset ──────────────────────────────────────
ST2B_DATASET = "AvaLovelace/StableText2Brick"
ST2B_CACHE_DIR = DATA_DIR / "st2b_cache"
ST2B_CONVERTED_DIR = DATA_DIR / "st2b_labels"
ST2B_PROMPTS_DIR = DATA_DIR / "st2b_prompts"
PLANNER_PROMPTS_DIR = DATA_DIR / "prompts"

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
