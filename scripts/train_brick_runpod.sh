#!/bin/bash
##############################################################################
# One-shot training script for Qwen3.5-4B brick coordinate model
#
# Target: 4x RTX 5090 RunPod pod (or any multi-GPU machine)
# Expected time: ~12-16 hours
# Expected cost: ~$43-57 at $3.56/hr
#
# Usage:
#   1. Create a RunPod pod with 4x RTX 5090, PyTorch 2.x template
#   2. SSH in or open terminal
#   3. Run:  bash <(curl -sL https://raw.githubusercontent.com/Linus-Lu/Lego-Gen/main/scripts/train_brick_runpod.sh)
#      Or:   git clone https://github.com/Linus-Lu/Lego-Gen.git && cd Lego-Gen && bash scripts/train_brick_runpod.sh
##############################################################################

set -euo pipefail

# ── HuggingFace cache (set to large disk if needed) ────────────────────
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache}"
mkdir -p "$HF_HOME"

# ── Timestamps ──────────────────────────────────────────────────────────
start_time=$(date +%s)
log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "============================================"
log "  LegoGen Brick Model Training"
log "  $(date)"
log "============================================"

# ── System info ─────────────────────────────────────────────────────────
log ""
log "=== SYSTEM INFO ==="
log "GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not found"
log "CPU: $(nproc) cores"
log "RAM: $(free -h | awk '/^Mem:/{print $2}')"
log "Disk: $(df -h / | awk 'NR==2{print $4}') free"
echo ""

# ── Clone repo if not already in it ─────────────────────────────────────
if [ ! -f "backend/config.py" ]; then
    log "=== CLONING REPO ==="
    if [ -d "Lego-Gen" ]; then
        cd Lego-Gen
        log "  Using existing Lego-Gen directory"
        git pull --ff-only || true
    else
        git clone https://github.com/Linus-Lu/Lego-Gen.git
        cd Lego-Gen
    fi
    log "  Working directory: $(pwd)"
else
    log "  Already in repo: $(pwd)"
fi
echo ""

# ── Install dependencies ────────────────────────────────────────────────
log "=== INSTALLING DEPENDENCIES ==="
log "  Installing Python packages..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 2>/dev/null || true
pip install -q transformers accelerate peft trl datasets bitsandbytes sentencepiece protobuf 2>&1 | tail -3
pip install -q huggingface_hub 2>&1 | tail -1
log "  Done."
echo ""

# ── Verify GPU access ──────────────────────────────────────────────────
log "=== VERIFYING GPU ACCESS ==="
python3 -c "
import torch
n = torch.cuda.device_count()
print(f'  PyTorch sees {n} GPU(s)')
for i in range(n):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1e9
    print(f'    GPU {i}: {name} ({mem:.1f} GB)')
if n == 0:
    print('  ERROR: No GPUs detected!')
    exit(1)
"
echo ""

# ── Step 1: Prepare training data ──────────────────────────────────────
log "=== STEP 1/3: PREPARING TRAINING DATA ==="
if [ -f "data/brick_training/train.jsonl" ]; then
    train_lines=$(wc -l < data/brick_training/train.jsonl)
    log "  Training data already exists ($train_lines examples). Skipping."
else
    log "  Downloading StableText2Brick from HuggingFace..."
    log "  This downloads ~44MB and processes ~47k structures into colored JSONL."
    log "  Expected time: 5-15 minutes."
    echo ""

    python3 -u -m backend.data_pipeline.prepare_brick_dataset 2>&1 | while IFS= read -r line; do
        log "  $line"
    done

    if [ ! -f "data/brick_training/train.jsonl" ]; then
        log "  ERROR: Training data not generated!"
        exit 1
    fi
fi

train_lines=$(wc -l < data/brick_training/train.jsonl)
test_lines=$(wc -l < data/brick_training/test.jsonl)
log "  Train: $train_lines examples"
log "  Test:  $test_lines examples"
echo ""

# ── Step 2: Download base model ────────────────────────────────────────
log "=== STEP 2/3: DOWNLOADING BASE MODEL ==="
log "  Model: Qwen/Qwen3.5-4B"
log "  This will download ~8GB on first run."

python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('  Downloading tokenizer...')
AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B', trust_remote_code=True)
print('  Downloading model weights...')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-4B', torch_dtype='auto', trust_remote_code=True)
print('  Model cached successfully.')
"
echo ""

# ── Step 3: Train ──────────────────────────────────────────────────────
log "=== STEP 3/3: TRAINING ==="
log "  Config:"
log "    Model:       Qwen3.5-4B + LoRA (r=32, alpha=64)"
log "    Targets:     q_proj, v_proj"
log "    LR:          2e-3 (cosine schedule)"
log "    Batch:       4/GPU × 4 accum × 4 GPUs = effective 64"
log "    Epochs:      3"
log "    Max seq len: 8192"
log "    Precision:   bf16"
log "    Eval:        every 500 steps"
log "    Save:        every 500 steps (keep last 2)"
log ""
log "  Estimated time: 12-16 hours on 4x RTX 5090"
log "  Training log follows..."
log "  ─────────────────────────────────────────"
echo ""

train_start=$(date +%s)

# Use accelerate for multi-GPU if available, else fallback to plain python
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")

if [ "$NUM_GPUS" -gt 1 ]; then
    log "  Using accelerate with $NUM_GPUS GPUs"
    pip install -q accelerate 2>/dev/null

    accelerate launch \
        --num_processes=$NUM_GPUS \
        --mixed_precision=bf16 \
        -m backend.training.train_brick 2>&1 | while IFS= read -r line; do
        # Add timestamp to every line for monitoring
        echo "[$(date '+%H:%M:%S')] $line"
    done
else
    log "  Using single GPU"
    python3 -u -m backend.training.train_brick 2>&1 | while IFS= read -r line; do
        echo "[$(date '+%H:%M:%S')] $line"
    done
fi

train_end=$(date +%s)
train_elapsed=$(( train_end - train_start ))
train_hours=$(( train_elapsed / 3600 ))
train_mins=$(( (train_elapsed % 3600) / 60 ))

echo ""
log "  ─────────────────────────────────────────"
log "  Training complete in ${train_hours}h ${train_mins}m"
echo ""

# ── Verify output ──────────────────────────────────────────────────────
log "=== VERIFICATION ==="
CKPT_DIR="backend/models/checkpoints/qwen35-4b-brick-lora"

if [ -d "$CKPT_DIR" ]; then
    ckpt_size=$(du -sh "$CKPT_DIR" | cut -f1)
    ckpt_files=$(ls "$CKPT_DIR" | wc -l)
    log "  Checkpoint: $CKPT_DIR"
    log "  Size: $ckpt_size ($ckpt_files files)"
    log "  Contents:"
    ls -lh "$CKPT_DIR" | head -10 | while IFS= read -r line; do
        log "    $line"
    done
else
    log "  WARNING: Checkpoint directory not found at $CKPT_DIR"
    log "  Check training logs above for errors."
fi
echo ""

# ── Summary ────────────────────────────────────────────────────────────
end_time=$(date +%s)
total_elapsed=$(( end_time - start_time ))
total_hours=$(( total_elapsed / 3600 ))
total_mins=$(( (total_elapsed % 3600) / 60 ))

log "============================================"
log "  TRAINING COMPLETE"
log "  Total wall time: ${total_hours}h ${total_mins}m"
log "  Training time:   ${train_hours}h ${train_mins}m"
log "  Checkpoint:      $CKPT_DIR"
log "============================================"
log ""
log "Next steps:"
log "  1. Download the checkpoint: scp -r runpod:$(pwd)/$CKPT_DIR ./"
log "  2. Or push to HuggingFace:"
log "     python3 -c \"from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='$CKPT_DIR', repo_id='YOUR_USER/qwen35-4b-brick-lora')\""
log "  3. Place checkpoint in your local repo at $CKPT_DIR"
log "  4. Start the server: python -m backend.app"
log "  5. Generate: curl -X POST http://localhost:8000/api/generate-bricks -F 'prompt=a red chair'"
