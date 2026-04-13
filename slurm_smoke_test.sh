#!/bin/bash
# =============================================================================
# StepForge — Smoke test (1 GPU, smallgpu partition)
#
# Runs 20 SFT steps then 3 RL steps to validate all new code paths:
#   - sft_multigpu.py: model load, data load, label masking, loss CSV, checkpoint
#   - rl_train.py: SFT checkpoint load, RL dataset build, GRPO forward pass,
#                  reward functions, namespaced output dir
#
# Submit: sbatch slurm_smoke_test.sh
# Log:    tail -f $SCRATCH/stepforge/runs/smoke_<JOBID>/slurm.out
# =============================================================================
#SBATCH --job-name=stepforge_smoke
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --gres=gpu:2
#SBATCH --partition=smallgpu
#SBATCH --account=lilly-agentic-gpu

if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi

module purge
module load anaconda/2024.10-py312
module load cuda/12.6.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stepforge

export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:?Set HUGGINGFACE_TOKEN before submitting}"
export HF_TOKEN="$HUGGINGFACE_TOKEN"
export HF_HOME="$SCRATCH/.hf-cache"
export PYTHONPATH="${HOME}/StepForge:${PYTHONPATH:-}"
export KMP_DUPLICATE_LIB_OK=TRUE
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker

# ── Namespaced run directory ──────────────────────────────────────────────────
RUN_DIR="$SCRATCH/stepforge/runs/smoke_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
exec > >(tee -a "${RUN_DIR}/slurm.out") 2>&1

echo "========================================"
echo " StepForge Smoke Test"
echo "========================================"
echo " Job ID  : $SLURM_JOB_ID"
echo " Run dir : $RUN_DIR"
echo " Node    : $(hostname)"
echo " GPU     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo " Started : $(date)"
echo "========================================"

cd "${HOME}/StepForge"

# ── Dependency pins ───────────────────────────────────────────────────────────
pip install -q "trl==0.14.0" "transformers==4.51.3"
pip uninstall -q torchao -y 2>/dev/null || true
python - <<'PATCH'
import os, trl
trl_dir = os.path.dirname(trl.__file__)
p1 = os.path.join(trl_dir, "models", "utils.py")
txt = open(p1).read()
if "from torch.distributed.fsdp import FSDPModule" in txt and "except ImportError" not in txt:
    txt = txt.replace(
        "from torch.distributed.fsdp import FSDPModule",
        "try:\n    from torch.distributed.fsdp import FSDPModule\nexcept ImportError:\n    FSDPModule = None"
    )
    open(p1, "w").write(txt)
    print("Patched FSDPModule in trl/models/utils.py")
p2 = os.path.join(trl_dir, "import_utils.py")
txt2 = open(p2).read()
if "_LazyModule" not in txt2:
    txt2 += "\ntry:\n    from transformers.utils.import_utils import _LazyModule\nexcept ImportError:\n    _LazyModule = type('_LazyModule', (), {})\n"
    open(p2, "w").write(txt2)
    print("Patched _LazyModule into trl/import_utils.py")
PATCH

# ── Phase 1a: SFT smoke test (20 steps, 1 GPU) ───────────────────────────────
echo ""
echo "========================================"
echo " Phase 1a: SFT — 20 steps, 1 GPU"
echo "========================================"

SFT_DIR="${RUN_DIR}/sft_smoke"
torchrun \
    --standalone \
    --nproc_per_node=1 \
    training/sft_multigpu.py \
        --config configs/config_gautschi.yaml \
        --output-dir "$SFT_DIR" \
        --per-device-batch 1 \
        --max-steps 20

SFT_EXIT=$?
echo " SFT 1-GPU finished: $(date)  (exit=$SFT_EXIT)"

if [ $SFT_EXIT -ne 0 ]; then
    echo "SMOKE TEST FAILED at SFT 1-GPU phase (exit=$SFT_EXIT)"
    exit $SFT_EXIT
fi

# ── Phase 1b: SFT DDP smoke test (10 steps, 2 GPU) ───────────────────────────
# Tests: NCCL init, rank-0 dataset barrier, DDP gradient sync, multi-rank logging
echo ""
echo "========================================"
echo " Phase 1b: SFT DDP — 10 steps, 2 GPUs"
echo "========================================"

SFT_DDP_DIR="${RUN_DIR}/sft_ddp_smoke"
# Clear any leftover dataset-ready flag from phase 1a so rank 1 waits properly
rm -f "${SFT_DDP_DIR}/.dataset_ready"
torchrun \
    --standalone \
    --nproc_per_node=2 \
    training/sft_multigpu.py \
        --config configs/config_gautschi.yaml \
        --output-dir "$SFT_DDP_DIR" \
        --per-device-batch 1 \
        --max-steps 10

SFT_DDP_EXIT=$?
echo " SFT 2-GPU DDP finished: $(date)  (exit=$SFT_DDP_EXIT)"

if [ $SFT_DDP_EXIT -ne 0 ]; then
    echo "SMOKE TEST FAILED at SFT DDP phase (exit=$SFT_DDP_EXIT)"
    exit $SFT_DDP_EXIT
fi

# Find the saved checkpoint
SFT_CKPT=$(ls -d "${SFT_DIR}/checkpoint-"* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -z "$SFT_CKPT" ]; then
    echo "SMOKE TEST FAILED: no SFT checkpoint found in $SFT_DIR"
    exit 1
fi
echo " SFT checkpoint: $SFT_CKPT"

# Verify loss CSV was written
if [ -f "${SFT_DIR}/sft_loss.csv" ]; then
    echo " Loss CSV: OK ($(wc -l < "${SFT_DIR}/sft_loss.csv") rows)"
else
    echo " WARNING: sft_loss.csv not found"
fi

# ── Phase 2: RL smoke test (3 steps, 1 GPU, quantized) ───────────────────────
echo ""
echo "========================================"
echo " Phase 2: RL — 3 steps"
echo "========================================"

RL_DIR="${RUN_DIR}/rl_smoke"
torchrun \
    --standalone \
    --nproc_per_node=1 \
    training/rl_train.py \
        --config configs/config_gautschi.yaml \
        --sft-checkpoint "$SFT_CKPT" \
        --output-dir "$RL_DIR" \
        --max-steps 3 \
        --num-generations 2

RL_EXIT=$?
echo " RL smoke finished: $(date)  (exit=$RL_EXIT)"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Smoke Test Summary"
echo "========================================"
echo " SFT phase : exit=$SFT_EXIT"
echo " RL phase  : exit=$RL_EXIT"
echo " Run dir   : $RUN_DIR"
echo " Finished  : $(date)"
echo "========================================"

if [ $SFT_EXIT -ne 0 ] || [ $RL_EXIT -ne 0 ]; then
    echo " SMOKE TEST FAILED"
    exit 1
else
    echo " SMOKE TEST PASSED"
fi
