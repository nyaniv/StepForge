#!/bin/bash
# =============================================================================
# StepForge — Multi-GPU SFT job for Gautschi (Purdue RCAC)
# Hardware:  8× NVIDIA H100 80GB  (full Gautschi-H node)
# DDP:       torchrun with 8 processes, 1 GPU each
#
# Paper match (Chen et al., 2026):
#   Paper ran 4×H100, effective batch=16, 10 epochs, lr=2e-4
#   Here:    8×H100 × 2 per_device × 1 grad_accum = 16 effective batch  ✓
#
# Each run is fully namespaced under its own directory:
#   $SCRATCH/stepforge/runs/sft_<JOBID>/
#
# Submit:  sbatch slurm_sft_multigpu_gautschi.sh
# Resume:  sbatch slurm_sft_multigpu_gautschi.sh /path/to/existing/run/dir
# Log:     tail -f $SCRATCH/stepforge/runs/sft_<JOBID>/slurm.out
# =============================================================================
#SBATCH --job-name=stepforge_sft_multigpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=14
#SBATCH --mem=800G
#SBATCH --gres=gpu:8
#SBATCH --partition=ai
#SBATCH --account=lilly-agentic-gpu
#SBATCH --requeue
#SBATCH --qos=preemptible
#SBATCH --signal=B:SIGUSR1@120

# ── Source the module system ─────────────────────────────────────────────────
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi

# ── Load Gautschi modules ────────────────────────────────────────────────────
module purge
module load anaconda/2024.10-py312
module load cuda/12.6.0

# ── Activate conda environment ───────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stepforge

# ── Environment variables ────────────────────────────────────────────────────
export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:?Set HUGGINGFACE_TOKEN before submitting}"
export HF_HOME="$SCRATCH/.hf-cache"
export PYTHONPATH="${HOME}/StepForge:${PYTHONPATH:-}"
export KMP_DUPLICATE_LIB_OK=TRUE
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker

# ── Namespaced run directory ──────────────────────────────────────────────────
# $1 is set on resume (resubmit passes the original run dir so we continue
# into the same directory instead of creating a new one).
if [ -n "${1:-}" ] && [ -d "$1" ]; then
    RUN_DIR="$1"
    echo "[$(date)] Resuming existing run: $RUN_DIR"
else
    RUN_DIR="$SCRATCH/stepforge/runs/sft_${SLURM_JOB_ID}"
    mkdir -p "$RUN_DIR"
    echo "[$(date)] New run directory: $RUN_DIR"
fi

# All output lives under RUN_DIR
exec > >(tee -a "${RUN_DIR}/slurm.out") 2>&1

# ── Dependency pins (self-healing) ───────────────────────────────────────────
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

# ── Signal handler: resubmit on approaching time limit ───────────────────────
_resubmit() {
    echo ""
    echo "[$(date)] Time limit approaching — resubmitting into same run dir: $RUN_DIR"
    sbatch "${HOME}/StepForge/slurm_sft_multigpu_gautschi.sh" "$RUN_DIR"
    echo "[$(date)] Resubmit issued. Exiting current job gracefully."
    exit 0
}
trap _resubmit SIGUSR1

# ── Job info ─────────────────────────────────────────────────────────────────
echo "========================================"
echo " StepForge SFT Multi-GPU — Gautschi"
echo "========================================"
echo " Job ID   : $SLURM_JOB_ID"
echo " Run dir  : $RUN_DIR"
echo " Node     : $(hostname)"
echo " GPUs     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo " World    : 8 processes (1 per GPU)"
echo " Eff batch: 2 × 1 × 8 = 16  (paper spec)"
echo " Started  : $(date)"
echo " Config   : configs/config_gautschi.yaml"
echo "========================================"

# ── Preflight check ──────────────────────────────────────────────────────────
cd "${HOME}/StepForge"

echo "Running preflight environment check..."
python training/preflight_check.py || { echo "PREFLIGHT FAILED — aborting job"; exit 1; }

# ── Run SFT via torchrun (8-GPU DDP) ─────────────────────────────────────────
torchrun \
    --standalone \
    --nproc_per_node=8 \
    training/sft_multigpu.py \
        --config configs/config_gautschi.yaml \
        --output-dir "$RUN_DIR" &

wait $!
SFT_EXIT=$?

echo "========================================"
echo " SFT finished : $(date)  (exit=$SFT_EXIT)"
echo "========================================"

# ── Timestamped weight snapshot on successful completion ─────────────────────
if [ $SFT_EXIT -eq 0 ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    FINAL_DIR="${RUN_DIR}/final"
    SNAPSHOT_DIR="${RUN_DIR}/sft_weights_${TIMESTAMP}"
    if [ -d "$FINAL_DIR" ]; then
        echo "Saving timestamped weight snapshot to $SNAPSHOT_DIR ..."
        cp -r "$FINAL_DIR" "$SNAPSHOT_DIR"
        echo "Snapshot saved: $SNAPSHOT_DIR"
    else
        echo "WARNING: $FINAL_DIR not found — no snapshot saved."
    fi
fi
