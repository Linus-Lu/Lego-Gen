#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Full Two-Stage Training Pipeline
#
# Runs everything needed to train the LEGOGen two-stage model:
#   1. Preprocess ST2B labels (add grid_pos)
#   2. Download COCO 2017 data (if not already present)
#   3. Build Stage 1 manifest (COCO + ST2B caption matching)
#   4. Train Stage 2: text → LEGO JSON (ST2B-only, structure-aware loss)
#   5. Train Stage 1: image → description (COCO + Rebrickable)
#
# Usage:
#   bash scripts/train_full_pipeline.sh              # run everything
#   bash scripts/train_full_pipeline.sh --skip-coco   # skip COCO download
#   bash scripts/train_full_pipeline.sh --stage2-only  # only Stage 2
#   bash scripts/train_full_pipeline.sh --stage1-only  # only Stage 1 (assumes manifest exists)
#   bash scripts/train_full_pipeline.sh --no-wandb     # disable W&B logging
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

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
# STEP 1: Preprocess ST2B labels (add grid_pos)
# ══════════════════════════════════════════════════════════════════════
if [ "$STAGE1_ONLY" = false ]; then
    log "Step 1/5: Adding grid_pos to ST2B labels"

    # Check if already processed (sample a file for grid_pos)
    SAMPLE=$(ls data/st2b_labels/*.json 2>/dev/null | head -1)
    if [ -n "$SAMPLE" ] && python3 -c "import json; d=json.load(open('$SAMPLE')); assert 'grid_pos' in d.get('subassemblies',[{}])[0].get('parts',[{}])[0]" 2>/dev/null; then
        echo "grid_pos already present, skipping."
    else
        python -m backend.data_pipeline.add_grid_pos
    fi
fi

# ══════════════════════════════════════════════════════════════════════
# STEP 2: Download COCO 2017 (annotations + images)
# ══════════════════════════════════════════════════════════════════════
if [ "$STAGE2_ONLY" = false ] && [ "$SKIP_COCO" = false ]; then
    log "Step 2/5: Downloading COCO 2017 data"

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
# STEP 3: Build Stage 1 manifest
# ══════════════════════════════════════════════════════════════════════
if [ "$STAGE2_ONLY" = false ]; then
    log "Step 3/5: Building Stage 1 training manifest"

    if [ -f "data/stage1_manifest.json" ]; then
        COUNT=$(python3 -c "import json; print(len(json.load(open('data/stage1_manifest.json'))))")
        echo "Manifest already exists with $COUNT samples."
        read -p "Rebuild? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python -m backend.data_pipeline.build_stage1_dataset
        fi
    else
        python -m backend.data_pipeline.build_stage1_dataset
    fi
fi

# ══════════════════════════════════════════════════════════════════════
# STEP 4: Train Stage 2 (text → LEGO JSON)
# ══════════════════════════════════════════════════════════════════════
if [ "$STAGE1_ONLY" = false ]; then
    log "Step 4/5: Training Stage 2 (text → LEGO JSON)"
    echo "Model: Qwen3.5-9B + LoRA rank 128"
    echo "Data: ST2B-only (~46k samples), structure-aware loss"
    echo "Output: backend/models/checkpoints/qwen35-9b-lego-stage2-lora/"
    echo ""

    STAGE2_DIR="backend/models/checkpoints/qwen35-9b-lego-stage2-lora"
    mkdir -p "$STAGE2_DIR"

    python -m backend.training.train_unified \
        --output-dir "$STAGE2_DIR" \
        --epochs 3 \
        $WANDB_FLAG \
        $STAGE2_RESUME \
        2>&1 | tee training_stage2.log

    echo ""
    echo "Stage 2 training complete. $(elapsed)"
fi

# ══════════════════════════════════════════════════════════════════════
# STEP 5: Train Stage 1 (image → description)
# ══════════════════════════════════════════════════════════════════════
if [ "$STAGE2_ONLY" = false ]; then
    log "Step 5/5: Training Stage 1 (image → description)"
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
echo "  Stage 2: backend/models/checkpoints/qwen35-9b-lego-stage2-lora/"
echo "  Stage 1: backend/models/checkpoints/qwen35-9b-lego-stage1-lora/"
echo ""
echo "To start the server with new checkpoints:"
echo "  LEGOGEN_DEV=0 uvicorn backend.app:app --host 0.0.0.0 --port 8000"
echo ""
echo "Update backend/config.py if checkpoint paths differ:"
echo "  UNIFIED_CHECKPOINT_DIR = CHECKPOINT_DIR / 'qwen35-9b-lego-stage2-lora'"
