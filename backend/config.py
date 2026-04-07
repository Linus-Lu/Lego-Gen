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

# ── Vision Model (backward compat with checked-in adapter) ───────────
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MAX_SEQ_LENGTH = 4096

# ── Unified Model ─────────────────────────────────────────────────────
UNIFIED_MODEL_NAME = "Qwen/Qwen3.5-9B"

# ── QLoRA (vision-only — keeps original targets for existing adapter) ─
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
QUANTIZATION_BITS = 4

# ── Training ───────────────────────────────────────────────────────────
LEARNING_RATE = 1e-4       # v1 was 2e-4, lower = more stable convergence
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32
NUM_EPOCHS = 3             # v2 was 10, reduced for faster iteration
WARMUP_STEPS = 200         # v1 was 100, longer warmup for lower LR
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
LOGGING_STEPS = 10
EVAL_STEPS = 200
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 2  # keep only 2 checkpoints — 100GB volume is tight with 27B model

# ── Planner (text-to-JSON) ─────────────────────────────────────────
PLANNER_MODEL_NAME = "Qwen/Qwen3.5-9B"
PLANNER_CHECKPOINT_DIR = BACKEND_DIR / "models" / "checkpoints" / "qwen35-lego-planner-lora"
PLANNER_MAX_SEQ_LENGTH = 2048
PLANNER_LEARNING_RATE = 3e-5
PLANNER_NUM_EPOCHS = 5
PLANNER_WARMUP_STEPS = 300

# ── Planner QLoRA (Qwen3.5-9B hybrid architecture) ────────────────
PLANNER_LORA_R = 64
PLANNER_LORA_ALPHA = 128
PLANNER_LORA_DROPOUT = 0.05
PLANNER_LORA_TARGET_MODULES = "all-linear"

# ── Planner Training (RTX 5090 optimized) ─────────────────────────
PLANNER_BATCH_SIZE = 2
PLANNER_GRADIENT_ACCUMULATION = 8

# ── StableText2Brick dataset ──────────────────────────────────────
ST2B_DATASET = "AvaLovelace/StableText2Brick"
ST2B_CACHE_DIR = DATA_DIR / "st2b_cache"
ST2B_CONVERTED_DIR = DATA_DIR / "st2b_labels"
ST2B_PROMPTS_DIR = DATA_DIR / "st2b_prompts"
PLANNER_PROMPTS_DIR = DATA_DIR / "prompts"

# ── Stage 2: Text → LEGO JSON (Qwen3.5-9B + LoRA) ───────────────────
UNIFIED_CHECKPOINT_DIR = CHECKPOINT_DIR / "qwen35-9b-lego-stage2-lora"
UNIFIED_LEARNING_RATE = 5e-5       # higher LR ok for smaller model
UNIFIED_NUM_EPOCHS = 3
UNIFIED_WARMUP_STEPS = 200
UNIFIED_BATCH_SIZE = 1             # 9B 4-bit fits bs=1 on 32GB (5090) with chunked loss
UNIFIED_GRADIENT_ACCUMULATION = 32  # effective batch = 32
UNIFIED_MAX_SEQ_LENGTH = 4096
UNIFIED_QUANTIZATION_BITS = 4      # 4-bit NF4
VISION_UPSAMPLE = 10  # upsample vision samples to balance with planner data

# ── Stage 1: Image → Description (lightweight LoRA) ──────────────────
STAGE1_CHECKPOINT_DIR = CHECKPOINT_DIR / "qwen35-9b-lego-stage1-lora"
STAGE1_LORA_R = 32
STAGE1_LORA_ALPHA = 64
STAGE1_LEARNING_RATE = 5e-5
STAGE1_NUM_EPOCHS = 3
STAGE1_WARMUP_STEPS = 100
STAGE1_BATCH_SIZE = 4
STAGE1_GRADIENT_ACCUMULATION = 4  # effective batch = 16
STAGE1_MAX_SEQ_LENGTH = 512       # descriptions are short

# ── Stage 2: Brick coordinate model ────────────────────────────────────
BRICK_MODEL_NAME = "Qwen/Qwen3.5-4B"
BRICK_CHECKPOINT_DIR = CHECKPOINT_DIR / "qwen35-4b-brick-lora"
BRICK_LEARNING_RATE = 2e-3
BRICK_BATCH_SIZE = 2
BRICK_GRADIENT_ACCUMULATION = 8
BRICK_MAX_SEQ_LENGTH = 4096
BRICK_NUM_EPOCHS = 3
BRICK_LORA_R = 32
BRICK_LORA_ALPHA = 64
BRICK_LORA_DROPOUT = 0.05
BRICK_TRAINING_DATA = DATA_DIR / "brick_training"

STAGE1_SYSTEM_PROMPT = (
    "You are a LEGO design assistant. Describe this object's shape, structure, "
    "colors, and proportions in a way useful for building it with LEGO bricks. "
    "Focus on geometry and spatial relationships, not materials or artistic style. "
    "Be concise — one to three sentences."
)

# ── COCO → ST2B category mapping for Stage 1 data ────────────────────
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

# ── Prompt Caching ────────────────────────────────────────────────────
CACHE_ENABLED = os.environ.get("LEGOGEN_CACHE_ENABLED", "1") == "1"
CACHE_KV_PREFIX_ENABLED = True       # Layer 1: KV-cache prefix reuse
CACHE_RESPONSE_ENABLED = True        # Layer 2: Full response caching (note: do_sample=True means same prompt can produce different outputs; first result is cached)
CACHE_TOKENIZATION_ENABLED = True    # Layer 3: Tokenization caching
CACHE_RESPONSE_MAX_SIZE = 256        # Max cached responses
CACHE_RESPONSE_TTL_SECONDS = 3600    # 1 hour TTL

# ── Inference ──────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 1024
NUM_BEAMS = 1
TEMPERATURE = 0.7
TOP_P = 0.9

# ── Stability Checker Thresholds ──────────────────────────────────────
QUANTITY_WARN_THRESHOLD = 50
QUANTITY_FAIL_THRESHOLD = 200
SUPPORT_RATIO_WARN = 3.0
TOP_HEAVY_RATIO = 2.0
MIN_CANTILEVER_CONNECTIONS = 2

# ── Device ─────────────────────────────────────────────────────────────
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
except ImportError:
    DEVICE = "cpu"
    USE_BF16 = False
