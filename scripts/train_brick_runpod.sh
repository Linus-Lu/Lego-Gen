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
#   3. Run:  git clone https://github.com/Linus-Lu/Lego-Gen.git && cd Lego-Gen && bash scripts/train_brick_runpod.sh
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
log "HF_HOME: $HF_HOME"
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

# Detect GPU arch to pick correct PyTorch build
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
log "  Detected GPU: $GPU_NAME"

if echo "$GPU_NAME" | grep -qi "5090\|5080\|5070\|blackwell\|RTX 50"; then
    log "  Blackwell GPU detected — installing PyTorch nightly with CUDA 12.8 (sm_120 support)"
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 2>&1 | tail -5
else
    log "  Installing PyTorch stable with CUDA 12.4"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -5
fi

log "  Installing training libraries..."
pip install transformers accelerate peft trl datasets bitsandbytes sentencepiece protobuf huggingface_hub 2>&1 | tail -5
log "  Done."

# Show installed versions
python3 -c "
import torch, transformers, peft, trl
print(f'  torch={torch.__version__}  transformers={transformers.__version__}  peft={peft.__version__}  trl={trl.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}  CUDA version: {torch.version.cuda}')
"
echo ""

# ── Verify GPU access ──────────────────────────────────────────────────
log "=== VERIFYING GPU ACCESS ==="
python3 -c "
import torch
n = torch.cuda.device_count()
print(f'  PyTorch sees {n} GPU(s)')
for i in range(n):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f'    GPU {i}: {name} ({mem:.1f} GB)')
if n == 0:
    print('  ERROR: No GPUs detected!')
    exit(1)
print(f'  CUDA arch list: {torch.cuda.get_arch_list()}')
"
echo ""

# ── Step 1: Prepare training data ──────────────────────────────────────
log "=== STEP 1/3: PREPARING TRAINING DATA ==="
if [ -f "data/brick_training/train.jsonl" ]; then
    train_lines=$(wc -l < data/brick_training/train.jsonl)
    log "  Training data already exists ($train_lines examples). Skipping download."
else
    log "  Downloading StableText2Brick from HuggingFace..."
    log "  This downloads ~44MB and processes ~47k structures into colored JSONL."
    log "  Expected time: 5-15 minutes."
    echo ""

    # Run with unbuffered output so progress shows in real time
    PYTHONUNBUFFERED=1 python3 -m backend.data_pipeline.prepare_brick_dataset 2>&1 | while IFS= read -r line; do
        log "  $line"
    done

    if [ ! -f "data/brick_training/train.jsonl" ]; then
        log "  ERROR: Training data not generated!"
        exit 1
    fi
fi

train_lines=$(wc -l < data/brick_training/train.jsonl)
train_size=$(du -sh data/brick_training/train.jsonl | cut -f1)
log "  Train: $train_lines examples ($train_size)"

# Create test split if missing (hold out 10%)
if [ ! -f "data/brick_training/test.jsonl" ]; then
    log "  No test split found — creating one (10% holdout)..."
    test_count=$((train_lines / 10))
    tail -n "$test_count" data/brick_training/train.jsonl > data/brick_training/test.jsonl
    head -n $((train_lines - test_count)) data/brick_training/train.jsonl > data/brick_training/train_tmp.jsonl
    mv data/brick_training/train_tmp.jsonl data/brick_training/train.jsonl
    train_lines=$((train_lines - test_count))
    log "  Split: $train_lines train, $test_count test"
fi

test_lines=$(wc -l < data/brick_training/test.jsonl)
log "  Test:  $test_lines examples"

# Preview one example
log "  Sample entry:"
python3 -c "
import json
with open('data/brick_training/train.jsonl') as f:
    ex = json.loads(f.readline())
caption = ex['messages'][1]['content'].split('### Input:\n')[1]
bricks = ex['messages'][2]['content'].split('\n')
print(f'    Caption: {caption[:80]}')
print(f'    Bricks:  {len(bricks)} total')
print(f'    First:   {bricks[0]}')
print(f'    Last:    {bricks[-1]}')
"
echo ""

# ── Step 2: Download base model ────────────────────────────────────────
log "=== STEP 2/3: DOWNLOADING BASE MODEL ==="
log "  Model: Qwen/Qwen3.5-4B → $HF_HOME"
log "  This will download ~8GB on first run."

python3 -c "
import sys, os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'  # show progress
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
logging.set_verbosity_info()

print('  Downloading tokenizer...', flush=True)
AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B', trust_remote_code=True)
print('  Tokenizer cached.', flush=True)

print('  Downloading model weights (~8GB)...', flush=True)
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-4B', torch_dtype='auto', trust_remote_code=True)
print('  Model cached successfully.', flush=True)
" 2>&1 | grep -v "^$" | while IFS= read -r line; do
    log "  $line"
done
echo ""

# ── Step 3: Train ──────────────────────────────────────────────────────
log "=== STEP 3/3: TRAINING ==="
log "  Config:"
log "    Model:       Qwen3.5-4B + LoRA (r=32, alpha=64)"
log "    Targets:     q_proj, v_proj"
log "    LR:          2e-3 (cosine schedule)"
log "    Epochs:      3"
log "    Max seq len: 8192"
log "    Precision:   bf16"
log "    Eval:        every 500 steps"
log "    Save:        every 500 steps (keep last 2)"

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
EFFECTIVE_BATCH=$((4 * 4 * NUM_GPUS))
log "    GPUs:        $NUM_GPUS"
log "    Batch:       4/GPU × 4 accum × $NUM_GPUS GPUs = effective $EFFECTIVE_BATCH"

# Estimate steps
TRAIN_LINES=$(wc -l < data/brick_training/train.jsonl)
STEPS_PER_EPOCH=$((TRAIN_LINES / EFFECTIVE_BATCH))
TOTAL_STEPS=$((STEPS_PER_EPOCH * 3))
log "    Est. steps:  ~$TOTAL_STEPS total ($STEPS_PER_EPOCH/epoch × 3 epochs)"
log ""
log "  Training log follows..."
log "  ─────────────────────────────────────────"
echo ""

train_start=$(date +%s)

if [ "$NUM_GPUS" -gt 1 ]; then
    log "  Launching accelerate with $NUM_GPUS GPUs..."

    PYTHONUNBUFFERED=1 accelerate launch \
        --num_processes=$NUM_GPUS \
        --mixed_precision=bf16 \
        -m backend.training.train_brick 2>&1 | while IFS= read -r line; do
        echo "[$(date '+%H:%M:%S')] $line"
    done
else
    log "  Using single GPU"
    PYTHONUNBUFFERED=1 python3 -m backend.training.train_brick 2>&1 | while IFS= read -r line; do
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
log "  1. Download the checkpoint:"
log "     scp -r runpod:$(pwd)/$CKPT_DIR ./"
log "  2. Or push to HuggingFace:"
log "     python3 -c \"from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='$CKPT_DIR', repo_id='YOUR_USER/qwen35-4b-brick-lora')\""
log "  3. Place checkpoint in your local repo at $CKPT_DIR"
log "  4. Start the server: python -m backend.app"
log "  5. Generate: curl -X POST localhost:8000/api/generate-bricks -F 'prompt=a red chair'"
