#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# LEGOGen Bootstrap — from bare machine to trained two-stage model
#
# Tested on: RunPod / Lambda / Vast.ai with Ubuntu 22.04+ and NVIDIA GPU
# Requirements: NVIDIA GPU with >= 48GB VRAM (80-95GB recommended for 27B)
#               At least 80GB free disk space
#
# Usage:
#   curl -sL https://raw.githubusercontent.com/Linus-Lu/Lego-Gen/main/scripts/bootstrap.sh | bash
#
#   # Or clone first and run locally:
#   git clone https://github.com/Linus-Lu/Lego-Gen.git
#   cd Lego-Gen
#   bash scripts/bootstrap.sh
#
#   # Options:
#   bash scripts/bootstrap.sh --no-wandb          # skip W&B logging
#   bash scripts/bootstrap.sh --skip-training      # setup only, no training
#   bash scripts/bootstrap.sh --skip-coco          # skip COCO download (Stage 2 only)
#   bash scripts/bootstrap.sh --hf-token=hf_xxx    # set HuggingFace token
# ══════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Ensure /workspace/pylibs is on PYTHONPATH (torch lives there) ────
if [ -d "/workspace/pylibs" ]; then
    export PYTHONPATH="${PYTHONPATH:-}:/workspace/pylibs"
fi

# ── Parse args ────────────────────────────────────────────────────────
SKIP_TRAINING=false
SKIP_COCO=false
WANDB_FLAG=""
HF_TOKEN="${HF_TOKEN:-}"

for arg in "$@"; do
    case $arg in
        --skip-training)  SKIP_TRAINING=true ;;
        --skip-coco)      SKIP_COCO=true ;;
        --no-wandb)       WANDB_FLAG="--no-wandb" ;;
        --hf-token=*)     HF_TOKEN="${arg#*=}" ;;
        *)                echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[1;36m'
NC='\033[0m'

log()  { echo -e "\n${CYAN}══ $1 ══${NC}\n"; }
ok()   { echo -e "${GREEN}✓ $1${NC}"; }
err()  { echo -e "${RED}✗ $1${NC}"; exit 1; }
SECONDS=0
elapsed() { printf "%dh %dm %ds" $((SECONDS/3600)) $((SECONDS%3600/60)) $((SECONDS%60)); }

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         LEGOGen Two-Stage Pipeline Bootstrap            ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ══════════════════════════════════════════════════════════════════════
# 1. System dependencies
# ══════════════════════════════════════════════════════════════════════
log "1/8 Installing system dependencies"

apt-get update -qq 2>/dev/null || true
apt-get install -y -qq git wget unzip curl build-essential 2>/dev/null || true
ok "System packages"

# Check Python
python3 --version || err "Python 3 not found. Install Python 3.10+"
ok "Python $(python3 --version 2>&1 | cut -d' ' -f2)"

# Check NVIDIA GPU
if nvidia-smi > /dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    ok "GPU: $GPU_NAME ($GPU_MEM)"
else
    err "No NVIDIA GPU found. Training requires a GPU with >= 48GB VRAM."
fi

# Check disk space
FREE_GB=$(df -BG . | awk 'NR==2 {print $4}' | tr -d 'G')
if [ "${FREE_GB:-0}" -lt 50 ]; then
    echo "WARNING: Only ${FREE_GB}GB free disk space. Recommend 80GB+."
fi

# ══════════════════════════════════════════════════════════════════════
# 2. Clone repo (if not already in it)
# ══════════════════════════════════════════════════════════════════════
log "2/8 Setting up repository"

if [ -f "backend/config.py" ]; then
    ok "Already in LEGOGen repo"
    PROJECT_ROOT=$(pwd)
elif [ -d "Lego-Gen" ]; then
    cd Lego-Gen
    PROJECT_ROOT=$(pwd)
    ok "Found existing Lego-Gen directory"
else
    git clone https://github.com/Linus-Lu/Lego-Gen.git
    cd Lego-Gen
    PROJECT_ROOT=$(pwd)
    ok "Cloned repository"
fi

# ══════════════════════════════════════════════════════════════════════
# 3. Python dependencies
# ══════════════════════════════════════════════════════════════════════
log "3/8 Installing Python dependencies"

# Upgrade pip
pip install --upgrade pip -q

# Install PyTorch (CUDA 12.x)
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -q
fi
ok "PyTorch $(python3 -c 'import torch; print(torch.__version__)')"

# Install flash-attn (needs special handling)
if ! python3 -c "import flash_attn" 2>/dev/null; then
    echo "Installing flash-attention (this takes a few minutes)..."
    pip install flash-attn --no-build-isolation -q 2>&1 | tail -1 || {
        echo "WARNING: flash-attn install failed. Training will use default attention."
    }
fi

# Install remaining requirements
pip install -r requirements.txt -q 2>&1 | tail -3
ok "Python packages installed"

# HuggingFace token
if [ -n "$HF_TOKEN" ]; then
    python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')" 2>/dev/null
    ok "HuggingFace token set"
else
    echo "  (No HF_TOKEN set — unauthenticated downloads may be rate-limited)"
fi

# ══════════════════════════════════════════════════════════════════════
# 4. Frontend dependencies
# ══════════════════════════════════════════════════════════════════════
log "4/8 Installing frontend dependencies"

if command -v node > /dev/null 2>&1; then
    cd frontend
    npm install --silent 2>&1 | tail -1
    cd "$PROJECT_ROOT"
    ok "Frontend dependencies installed"
else
    echo "  Node.js not found — skipping frontend (install with: apt install nodejs npm)"
fi

# ══════════════════════════════════════════════════════════════════════
# 5. Prepare Rebrickable dataset
# ══════════════════════════════════════════════════════════════════════
log "5/8 Preparing Rebrickable dataset"

if [ -d "data/labels" ] && [ "$(ls data/labels/*.json 2>/dev/null | wc -l)" -gt 100 ]; then
    ok "Rebrickable labels already present ($(ls data/labels/*.json | wc -l) files)"
else
    echo "Downloading Rebrickable data and generating labels..."
    python scripts/prepare_dataset.py --max-sets 2000
    ok "Rebrickable dataset prepared"
fi

# Generate planner prompts
if [ -d "data/prompts" ] && [ "$(ls data/prompts/*.json 2>/dev/null | wc -l)" -gt 100 ]; then
    ok "Planner prompts already present"
else
    python scripts/prepare_planner_prompts.py
    ok "Planner prompts generated"
fi

# ══════════════════════════════════════════════════════════════════════
# 6. Prepare StableText2Brick dataset
# ══════════════════════════════════════════════════════════════════════
log "6/8 Preparing StableText2Brick dataset"

ST2B_COUNT=$(find data/st2b_labels -name '*.json' 2>/dev/null | wc -l)
if [ -d "data/st2b_labels" ] && [ "$ST2B_COUNT" -gt 1000 ]; then
    ok "ST2B labels already present ($ST2B_COUNT files)"
else
    echo "ST2B labels not found. Download and convert the HuggingFace"
    echo "StableText2Brick dataset into data/st2b_labels/ before training."
fi

# ══════════════════════════════════════════════════════════════════════
# 7. Prepare COCO data + Stage 1 manifest
# ══════════════════════════════════════════════════════════════════════
if [ "$SKIP_COCO" = false ]; then
    log "7/8 Preparing COCO 2017 + Stage 1 manifest"

    COCO_DIR="$PROJECT_ROOT/data/coco"
    mkdir -p "$COCO_DIR"

    # Annotations
    if [ ! -f "$COCO_DIR/annotations/instances_train2017.json" ]; then
        echo "Downloading COCO 2017 annotations (~250MB)..."
        cd "$COCO_DIR"
        wget -q --show-progress http://images.cocodataset.org/annotations/annotations_trainval2017.zip
        unzip -q annotations_trainval2017.zip
        rm -f annotations_trainval2017.zip
        cd "$PROJECT_ROOT"
        ok "COCO annotations downloaded"
    else
        ok "COCO annotations already present"
    fi

    # Train images
    if [ ! -d "$COCO_DIR/train2017" ] || [ "$(ls "$COCO_DIR/train2017/" 2>/dev/null | wc -l)" -lt 1000 ]; then
        echo "Downloading COCO 2017 train images (~18GB, this will take a while)..."
        cd "$COCO_DIR"
        wget -q --show-progress http://images.cocodataset.org/zips/train2017.zip
        unzip -q train2017.zip
        rm -f train2017.zip
        cd "$PROJECT_ROOT"
        ok "COCO train images downloaded"
    else
        ok "COCO train images already present ($(ls "$COCO_DIR/train2017/" | wc -l) files)"
    fi

    # Build manifest
    if [ ! -f "data/stage1_manifest.json" ]; then
        echo "Building Stage 1 training manifest..."
        python -m backend.data_pipeline.build_stage1_dataset
        ok "Stage 1 manifest built"
    else
        COUNT=$(python3 -c "import json; print(len(json.load(open('data/stage1_manifest.json'))))")
        ok "Stage 1 manifest already present ($COUNT samples)"
    fi
else
    log "7/8 Skipping COCO download (--skip-coco)"
fi

# ══════════════════════════════════════════════════════════════════════
# 8. Train
# ══════════════════════════════════════════════════════════════════════
if [ "$SKIP_TRAINING" = true ]; then
    log "8/8 Skipping training (--skip-training)"
    echo "Setup complete! To start training manually:"
    echo ""
    echo "  # Full pipeline:"
    echo "  bash scripts/train_full_pipeline.sh $WANDB_FLAG"
    echo ""
    echo "  # Stage 2 only (text→JSON, no COCO needed):"
    echo "  bash scripts/train_full_pipeline.sh --stage2-only $WANDB_FLAG"
    echo ""
else
    log "8/8 Starting training"

    TRAIN_FLAGS="$WANDB_FLAG"
    if [ "$SKIP_COCO" = true ]; then
        TRAIN_FLAGS="$TRAIN_FLAGS --stage2-only"
    fi

    bash scripts/train_full_pipeline.sh $TRAIN_FLAGS
fi

# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                   Bootstrap Complete                    ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Total time: $(printf '%-40s' "$(elapsed)")  ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Start the server:                                      ║"
echo "║    cd $PROJECT_ROOT"
echo "║    LEGOGEN_DEV=0 uvicorn backend.app:app \\              ║"
echo "║      --host 0.0.0.0 --port 8000                        ║"
echo "║                                                         ║"
echo "║  Start the frontend:                                    ║"
echo "║    cd frontend && npm run dev -- --host 0.0.0.0         ║"
echo "╚══════════════════════════════════════════════════════════╝"
