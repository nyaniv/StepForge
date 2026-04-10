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
# Submit: sbatch slurm_rl_gautschi.sh
# Check:  squeue -u $USER
# Log:    tail -f $SCRATCH/stepforge/logs/rl_<JOBID>.out
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
#SBATCH --partition=gpu             # check available partitions: slist
#SBATCH --requeue                   # allow requeue on preemption or time limit
#SBATCH --signal=B:SIGUSR1@120      # warn 120 s before wall-time so we can resubmit
# #SBATCH --account=YOUR_ACCOUNT   # uncomment and set if your allocation requires it

# ── Source the module system ─────────────────────────────────────────────────
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi

# ── Load Gautschi modules ────────────────────────────────────────────────────
module purge
module load anaconda        # check exact name: module spider anaconda
module load cuda/12.2       # H100 requires CUDA >= 11.8; check: module spider cuda

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

# ── WandB ─────────────────────────────────────────────────────────────────────
export WANDB_PROJECT="stepforge"
export WANDB_RUN_NAME="rl-${SLURM_JOB_ID}"    # unique name per job for side-by-side comparison
# WANDB_API_KEY must be set in your environment before submitting, or run `wandb login` first
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY or run: wandb login}"

# NCCL tuning for H100 NVLink interconnect
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0            # keep InfiniBand enabled if available
export NCCL_SOCKET_IFNAME=^lo,docker

# ── Ensure output directories exist ──────────────────────────────────────────
mkdir -p "$SCRATCH/stepforge/logs"
mkdir -p "$SCRATCH/stepforge/checkpoints/rl"

# Redirect SLURM logs to scratch
LOG_DIR="$SCRATCH/stepforge/logs"
exec > >(tee -a "${LOG_DIR}/rl_${SLURM_JOB_ID}.out") 2>&1

# ── Signal handler: resubmit on approaching time limit ───────────────────────
_resubmit() {
    echo ""
    echo "[$(date)] Time limit approaching — resubmitting job for checkpoint resume..."
    # GRPOTrainer saves at save_steps=20; trainer.train(resume_from_checkpoint=...)
    # will automatically pick up the latest checkpoint on the next run.
    sbatch "${HOME}/StepForge/slurm_rl_gautschi.sh"
    echo "[$(date)] Resubmit issued. Exiting current job gracefully."
    exit 0
}
trap _resubmit SIGUSR1

# ── Job info ─────────────────────────────────────────────────────────────────
echo "========================================"
echo " StepForge RL (GRPO) — Gautschi"
echo "========================================"
echo " Job ID   : $SLURM_JOB_ID"
echo " Node     : $(hostname)"
echo " GPUs     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo " World    : 8 processes (1 per GPU)"
echo " Started  : $(date)"
echo " Config   : configs/config_gautschi.yaml"
echo "========================================"

# ── Run RL via torchrun (8-GPU DDP) ─────────────────────────────────────────
cd "${HOME}/StepForge"

torchrun \
    --standalone \
    --nproc_per_node=8 \
    training/rl_train.py \
        --config configs/config_gautschi.yaml &

# Wait in the background so the SIGUSR1 trap can fire
wait $!

echo "========================================"
echo " RL finished : $(date)"
echo "========================================"
