#!/bin/bash
#SBATCH --job-name=step_llm_sft
#SBATCH --output=/home/SCHOLAR_USERNAME/StepLLM/logs/sft_%j.out
#SBATCH --error=/home/SCHOLAR_USERNAME/StepLLM/logs/sft_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ── Source module system ─────────────────────────────────────────────────────
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi

# ── Load Scholar modules ─────────────────────────────────────────────────────
module load anaconda/2020.11-py38
module load cuda/11.8

# ── Initialize conda ─────────────────────────────────────────────────────────
CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

# ── Activate environment ─────────────────────────────────────────────────────
conda activate step_llm || source activate step_llm

# ── Environment variables ────────────────────────────────────────────────────
# Set your HuggingFace token: export HUGGINGFACE_TOKEN="hf_..."
export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:?HF token not set}"
export PYTHONPATH=/home/SCHOLAR_USERNAME/StepLLM
export KMP_DUPLICATE_LIB_OK=TRUE
export TOKENIZERS_PARALLELISM=false

# ── Create output dirs ───────────────────────────────────────────────────────
mkdir -p /home/SCHOLAR_USERNAME/StepLLM/logs
mkdir -p /home/SCHOLAR_USERNAME/StepLLM/checkpoints/sft

# ── Run SFT ──────────────────────────────────────────────────────────────────
cd /home/SCHOLAR_USERNAME/StepLLM
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Starting SFT at $(date)"

python training/sft_train.py --config configs/config_scholar.yaml

echo "SFT finished at $(date)"
