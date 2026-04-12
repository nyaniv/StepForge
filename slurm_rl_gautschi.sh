#!/bin/bash
# =============================================================================
# StepForge — RL (GRPO) job for Gautschi (Purdue RCAC)
# Hardware:  8× NVIDIA H100 80GB  (full Gautschi-H node)
# DDP:       torchrun with 8 processes, 1 GPU each
#
# Paper match (Chen et al., 2026):
#   Paper ran 4×H100 × 2 prompts/GPU × 8 gen/prompt = 64 sequences/step
#   Here:    8×H100 × 1 prompt/GPU  × 8 gen/prompt = 64 sequences/step  ✓
#   max_steps=80, lr=3e-6, kl_coef=0.02, entropy_coef=0.005
#
# Submit: sbatch slurm_rl_gautschi.sh [/path/to/sft/run]
# Resume: sbatch slurm_rl_gautschi.sh [sft_ckpt] /path/to/existing/rl/run
# Log:    tail -f $SCRATCH/stepforge/runs/rl_<JOBID>/slurm.out
#
# Automatic resubmission: SLURM sends SIGUSR1 120s before time limit;
# the handler re-queues the job and GRPOTrainer resumes from the latest
# checkpoint automatically on restart.
# =============================================================================
#SBATCH --job-name=stepforge_rl
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8                  # 1 task per GPU (torchrun model)
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=14          # 112 CPUs / 8 GPUs = 14 CPUs per task
#SBATCH --mem=800G                  # ~100 GB per GPU process; well within 1 TB
#SBATCH --gres=gpu:8                # full node: 8× H100 80GB
#SBATCH --partition=ai
#SBATCH --account=lilly-agentic-gpu
#SBATCH --requeue                   # allow requeue on preemption or time limit
#SBATCH --qos=preemptible
#SBATCH --signal=B:SIGUSR1@120      # warn 120 s before wall-time so we can resubmit
# #SBATCH --account=YOUR_ACCOUNT   # uncomment and set if your allocation requires it

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

# NCCL tuning for H100 NVLink interconnect
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0            # keep InfiniBand enabled if available
export NCCL_SOCKET_IFNAME=^lo,docker

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

# ── SFT checkpoint override ($1) and optional RL resume dir ($2) ─────────────
SFT_CKPT_ARG=""
if [ -n "${1:-}" ]; then
    SFT_CKPT_ARG="--sft-checkpoint $1"
    echo "Using SFT checkpoint: $1"
fi

# ── Namespaced run directory ──────────────────────────────────────────────────
if [ -n "${2:-}" ] && [ -d "$2" ]; then
    RUN_DIR="$2"
    echo "[$(date)] Resuming existing RL run: $RUN_DIR"
else
    RUN_DIR="$SCRATCH/stepforge/runs/rl_${SLURM_JOB_ID}"
    mkdir -p "$RUN_DIR"
    echo "[$(date)] New RL run directory: $RUN_DIR"
fi

exec > >(tee -a "${RUN_DIR}/slurm.out") 2>&1

# ── Signal handler ────────────────────────────────────────────────────────────
_resubmit() {
    echo ""
    echo "[$(date)] Time limit approaching — resubmitting into same run dir: $RUN_DIR"
    sbatch "${HOME}/StepForge/slurm_rl_gautschi.sh" "${1:-}" "$RUN_DIR"
    echo "[$(date)] Resubmit issued. Exiting current job gracefully."
    exit 0
}
trap _resubmit SIGUSR1

echo "========================================"
echo " StepForge RL (GRPO) — Gautschi"
echo "========================================"
echo " Job ID   : $SLURM_JOB_ID"
echo " Run dir  : $RUN_DIR"
echo " Node     : $(hostname)"
echo " GPUs     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo " World    : 8 processes (1 per GPU)"
echo " Started  : $(date)"
echo " Config   : configs/config_gautschi.yaml"
echo "========================================"

# ── Preflight check ──────────────────────────────────────────────────────────
cd "${HOME}/StepForge"

echo "Running preflight environment check..."
python training/preflight_check.py || { echo "PREFLIGHT FAILED — aborting job"; exit 1; }

torchrun \
    --standalone \
    --nproc_per_node=8 \
    training/rl_train.py \
        --config configs/config_gautschi.yaml \
        $SFT_CKPT_ARG &

wait $!
RL_EXIT=$?

echo "========================================"
echo " RL finished : $(date)  (exit=$RL_EXIT)"
echo "========================================"
