#!/bin/bash
#SBATCH --job-name=step_llm_rl
#SBATCH --output=/home/nyaniv/StepLLM/logs/rl_%j.out
#SBATCH --error=/home/nyaniv/StepLLM/logs/rl_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --partition=scholar-gpu
#SBATCH --account=gpu
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@120

# ── Source module system ─────────────────────────────────────────────────────
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi

# ── Load Scholar modules ─────────────────────────────────────────────────────
module load anaconda
module load cuda/11.8

# ── Initialize conda ─────────────────────────────────────────────────────────
CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

# ── Activate environment ─────────────────────────────────────────────────────
conda activate /scratch/scholar/nyaniv/envs/step_llm

# ── Environment variables ────────────────────────────────────────────────────
# Set your HuggingFace token: export HUGGINGFACE_TOKEN="hf_..."
export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:?HF token not set}"
export PYTHONPATH=/home/nyaniv/StepLLM
export KMP_DUPLICATE_LIB_OK=TRUE
export TOKENIZERS_PARALLELISM=false

# ── Create output dirs ───────────────────────────────────────────────────────
mkdir -p /home/nyaniv/StepLLM/logs
mkdir -p /scratch/scholar/nyaniv/checkpoints/rl

# ── Signal handler for resubmission ─────────────────────────────────────────
_resubmit() {
    echo "Time limit approaching, resubmitting..."
    sbatch ~/StepLLM/slurm_rl.sh
    exit 0
}
trap _resubmit SIGUSR1

# ── Run RL (cold-starts from SFT checkpoint) ─────────────────────────────────
cd /home/nyaniv/StepLLM
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Starting RL at $(date)"

python training/rl_train.py --config configs/config_scholar.yaml &
wait $!

echo "RL finished at $(date)"
