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

# ── Rebrickable API (used by scripts/prepare_dataset.py) ───────────────
REBRICKABLE_API_KEY = os.environ.get("REBRICKABLE_API_KEY", "")
REBRICKABLE_BASE_URL = "https://rebrickable.com/api/v3/lego"
REBRICKABLE_RATE_LIMIT = 1.5

# ── Dataset splits (used by scripts/prepare_dataset.py) ────────────────
MIN_PARTS = 5
MAX_PARTS = 500
VAL_RATIO = 0.1
SPLITS_FILE = DATA_DIR / "splits.json"

# ── Shared training knobs (Stage 1) ────────────────────────────────────
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 2

# ── StableText2Brick + Stage 1 dataset (build_stage1_dataset.py) ───────
ST2B_DATASET = "AvaLovelace/StableText2Brick"
ST2B_CACHE_DIR = DATA_DIR / "st2b_cache"
ST2B_CONVERTED_DIR = DATA_DIR / "st2b_labels"
ST2B_PROMPTS_DIR = DATA_DIR / "st2b_prompts"
PLANNER_PROMPTS_DIR = DATA_DIR / "prompts"

COCO_TO_ST2B_CATEGORY = {
    "chair": "chair",
    "couch": "sofa",
    "bed": "bed",
    "dining table": "table",
    "car": "car",
    "bus": "bus",
    "train": "train",
    "truck": "truck",
    "boat": "vessel",
    "motorcycle": "motorbike",
    "airplane": "airplane",
    "bench": "bench",
    "vase": "vase",
    "cup": "mug",
    "laptop": "laptop",
}

# ── Stage 1: Image → Description (Qwen3.5-9B + LoRA) ───────────────────
STAGE1_MODEL_NAME = "Qwen/Qwen3.5-9B"
STAGE1_CHECKPOINT_DIR = CHECKPOINT_DIR / "qwen35-9b-lego-stage1-lora"
STAGE1_LORA_R = 32
STAGE1_LORA_ALPHA = 64
STAGE1_LEARNING_RATE = 5e-5
STAGE1_NUM_EPOCHS = 3
STAGE1_WARMUP_STEPS = 100
STAGE1_BATCH_SIZE = 8
STAGE1_GRADIENT_ACCUMULATION = 1
STAGE1_MAX_SEQ_LENGTH = 512

STAGE1_SYSTEM_PROMPT = (
    "You are a LEGO design assistant. Describe this object's shape, structure, "
    "colors, and proportions in a way useful for building it with LEGO bricks. "
    "Focus on geometry and spatial relationships, not materials or artistic style. "
    "Be concise — one to three sentences."
)

# ── Stage 2: Text → Brick coordinates (Qwen3.5-4B + LoRA) ──────────────
BRICK_MODEL_NAME = "Qwen/Qwen3.5-4B"
BRICK_CHECKPOINT_DIR = Path(
    os.environ.get(
        "BRICK_CHECKPOINT_DIR",
        CHECKPOINT_DIR / "qwen35-4b-brick-lora",
    )
)
BRICK_LEARNING_RATE = 1e-3
BRICK_BATCH_SIZE = 1
BRICK_GRADIENT_ACCUMULATION = 16
BRICK_MAX_SEQ_LENGTH = 4096
BRICK_NUM_EPOCHS = 3
BRICK_LORA_R = 32
BRICK_LORA_ALPHA = 64
BRICK_LORA_DROPOUT = 0.05
BRICK_TRAINING_DATA = Path(os.environ.get("BRICK_TRAINING_DATA", DATA_DIR / "brick_training"))

# ── Inference ──────────────────────────────────────────────────────────
INFERENCE_TIMEOUT_SECONDS = 120

# ── Device ─────────────────────────────────────────────────────────────
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
except ImportError:
    DEVICE = "cpu"
    USE_BF16 = False
