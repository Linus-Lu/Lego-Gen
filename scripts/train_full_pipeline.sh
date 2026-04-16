#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Full Two-Stage Training Pipeline
#
# Runs everything needed to train the LEGOGen two-stage model:
#   1. Download COCO 2017 data (if not already present)
#   2. Build Stage 1 manifest (COCO + ST2B caption matching)
#   3. Train Stage 2: text → brick coordinates (Qwen3.5-4B + LoRA)
#   4. Train Stage 1: image → description (Qwen3.5-9B + LoRA)
#
# Usage:
#   bash scripts/train_full_pipeline.sh              # run everything
#   bash scripts/train_full_pipeline.sh --skip-coco   # skip COCO download
#   bash scripts/train_full_pipeline.sh --stage2-only  # only Stage 2
#   bash scripts/train_full_pipeline.sh --stage1-only  # only Stage 1 (assumes manifest exists)
#   bash scripts/train_full_pipeline.sh --no-wandb     # disable W&B logging
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Ensure /workspace/pylibs is on PYTHONPATH (torch lives there) ────
if [ -d "/workspace/pylibs" ]; then
    export PYTHONPATH="${PYTHONPATH:-}:/workspace/pylibs"
fi

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# ── Parse args ────────────────────────────────────────────────────────
SKIP_COCO=false
STAGE2_ONLY=false
STAGE1_ONLY=false
WANDB_FLAG=""
STAGE2_RESUME=""

for arg in "$@"; do
    case $arg in
        --skip-coco)    SKIP_COCO=true ;;
        --stage2-only)  STAGE2_ONLY=true ;;
        --stage1-only)  STAGE1_ONLY=true ;;
        --no-wandb)     WANDB_FLAG="--no-wandb" ;;
        --resume=*)     STAGE2_RESUME="--resume ${arg#*=}" ;;
        *)              echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────
log() { echo -e "\n\033[1;36m══ $1 ══\033[0m\n"; }
err() { echo -e "\033[1;31mERROR: $1\033[0m"; exit 1; }
elapsed() {
    local t=$SECONDS
    printf "%dh %dm %ds" $((t/3600)) $((t%3600/60)) $((t%60))
}

SECONDS=0

# ── Check GPU ─────────────────────────────────────────────────────────
log "Checking GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || err "No GPU found"
echo ""

# ══════════════════════════════════════════════════════════════════════
# STEP 1: Download COCO 2017 (annotations + images)
# ══════════════════════════════════════════════════════════════════════
if [ "$STAGE2_ONLY" = false ] && [ "$SKIP_COCO" = false ]; then
    log "Step 1/4: Downloading COCO 2017 data"

    COCO_DIR="$PROJECT_ROOT/data/coco"
    mkdir -p "$COCO_DIR"

    # Annotations
    if [ ! -f "$COCO_DIR/annotations/instances_train2017.json" ]; then
        echo "Downloading COCO 2017 annotations..."
        cd "$COCO_DIR"
        wget -q --show-progress http://images.cocodataset.org/annotations/annotations_trainval2017.zip
        unzip -q annotations_trainval2017.zip
        rm -f annotations_trainval2017.zip
        cd "$PROJECT_ROOT"
    else
        echo "COCO annotations already present."
    fi

    # Train images
    if [ ! -d "$COCO_DIR/train2017" ] || [ "$(ls "$COCO_DIR/train2017/" 2>/dev/null | wc -l)" -lt 1000 ]; then
        echo "Downloading COCO 2017 train images (~18GB, this will take a while)..."
        cd "$COCO_DIR"
        wget -q --show-progress http://images.cocodataset.org/zips/train2017.zip
        unzip -q train2017.zip
        rm -f train2017.zip
        cd "$PROJECT_ROOT"
    else
        echo "COCO train images already present ($(ls "$COCO_DIR/train2017/" | wc -l) files)."
    fi
fi

# ══════════════════════════════════════════════════════════════════════
# STEP 2: Build Stage 1 manifest
# ══════════════════════════════════════════════════════════════════════
if [ "$STAGE2_ONLY" = false ]; then
    log "Step 2/4: Building Stage 1 training manifest"

    if [ -f "data/stage1_manifest.json" ]; then
        COUNT=$(python3 -c "import json; print(len(json.load(open('data/stage1_manifest.json'))))")
        echo "Manifest already exists with $COUNT samples. Skipping rebuild."
    else
        python -m backend.data_pipeline.build_stage1_dataset
    fi
fi

# ══════════════════════════════════════════════════════════════════════
# STEP 3: Train Stage 2 (text → brick coordinates)
# ══════════════════════════════════════════════════════════════════════
if [ "$STAGE1_ONLY" = false ]; then
    log "Step 3/4: Training Stage 2 (text → brick coordinates)"
    echo "Model: Qwen3.5-4B + LoRA"
    echo "Data: StableText2Brick (text → colored brick sequence)"
    echo "Output: backend/models/checkpoints/qwen35-4b-brick-lora/"
    echo ""

    STAGE2_DIR="backend/models/checkpoints/qwen35-4b-brick-lora"
    mkdir -p "$STAGE2_DIR"

    python -m backend.training.train_brick \
        --output-dir "$STAGE2_DIR" \
        --epochs 3 \
        $WANDB_FLAG \
        $STAGE2_RESUME \
        2>&1 | tee training_stage2.log

    echo ""
    echo "Stage 2 training complete. $(elapsed)"
fi

# ══════════════════════════════════════════════════════════════════════
# STEP 4: Train Stage 1 (image → description)
# ══════════════════════════════════════════════════════════════════════
if [ "$STAGE2_ONLY" = false ]; then
    log "Step 4/4: Training Stage 1 (image → description)"
    echo "Model: Qwen3.5-9B + LoRA rank 32"
    echo "Data: COCO + Rebrickable images"
    echo "Output: backend/models/checkpoints/qwen35-9b-lego-stage1-lora/"
    echo ""

    if [ ! -f "data/stage1_manifest.json" ]; then
        err "Stage 1 manifest not found. Run without --stage1-only first."
    fi

    python -m backend.training.train_stage1 \
        --manifest data/stage1_manifest.json \
        $WANDB_FLAG \
        2>&1 | tee training_stage1.log

    echo ""
    echo "Stage 1 training complete. $(elapsed)"
fi

# ══════════════════════════════════════════════════════════════════════
# Done
# ══════════════════════════════════════════════════════════════════════
log "All training complete! Total time: $(elapsed)"
echo ""
echo "Checkpoints:"
echo "  Stage 2: backend/models/checkpoints/qwen35-4b-brick-lora/"
echo "  Stage 1: backend/models/checkpoints/qwen35-9b-lego-stage1-lora/"
echo ""
echo "To start the server with new checkpoints:"
echo "  LEGOGEN_DEV=0 uvicorn backend.app:app --host 0.0.0.0 --port 8000"
