#!/bin/bash
# ── Run this ONCE on Scholar after uploading files ──────────────────────────
# Usage: bash ~/StepLLM/scholar_setup.sh <your-purdue-username> <hf-token>
#
# Example:
#   bash ~/StepLLM/scholar_setup.sh nyaniv hf_XXXXXXXXXXXXXXXX

set -e  # exit on first error

USERNAME=${1:?'Usage: bash scholar_setup.sh <purdue-username> <hf-token>'}
HF_TOKEN=${2:?'Usage: bash scholar_setup.sh <purdue-username> <hf-token>'}

STEPLLM_DIR="/home/$USERNAME/StepLLM"

echo "==> Patching SCHOLAR_USERNAME -> $USERNAME in all config/SLURM files..."
sed -i "s|SCHOLAR_USERNAME|$USERNAME|g" \
    "$STEPLLM_DIR/configs/config_scholar.yaml" \
    "$STEPLLM_DIR/slurm_sft.sh" \
    "$STEPLLM_DIR/slurm_rl.sh"

echo "==> Patching HF token..."
sed -i "s|PASTE_YOUR_HF_TOKEN_HERE|$HF_TOKEN|g" \
    "$STEPLLM_DIR/slurm_sft.sh" \
    "$STEPLLM_DIR/slurm_rl.sh"

echo "==> Making SLURM scripts executable..."
chmod +x "$STEPLLM_DIR/slurm_sft.sh" "$STEPLLM_DIR/slurm_rl.sh"

echo "==> Creating output directories..."
mkdir -p "$STEPLLM_DIR/logs" \
         "$STEPLLM_DIR/checkpoints/sft" \
         "$STEPLLM_DIR/checkpoints/rl"

echo ""
echo "==> Loading modules..."
# Source modules init if not already available (common on Scholar)
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi

module load anaconda/2020.11-py38 2>/dev/null || \
    module load anaconda 2>/dev/null || \
    echo "WARNING: Could not load anaconda module — trying system conda"

module load cuda/11.8 2>/dev/null || \
    echo "WARNING: Could not load cuda/11.8 — check 'module avail cuda' for options"

# Initialize conda for this shell session
# Scholar may need conda init before 'conda env create' works
CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
if [ -z "$CONDA_BASE" ]; then
    # Try common Purdue Scholar conda locations
    for path in \
        /apps/spack/scholar/apps/anaconda/2020.11-py38-gcc-4.8.5-kzrqhxy/x86_64/bin \
        /opt/conda/bin \
        ~/miniconda3/bin; do
        if [ -f "$path/conda" ]; then
            CONDA_BASE=$(dirname "$path")
            break
        fi
    done
fi

if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    echo "==> conda initialized from $CONDA_BASE"
fi

echo "==> Creating conda environment from environment.yml..."
conda env create -f "$STEPLLM_DIR/environment.yml" || echo "(env may already exist — proceeding)"

echo "==> Activating step_llm environment..."
conda activate step_llm || source activate step_llm

echo "==> Installing GPU-specific packages..."
# Match CUDA 11.8 on Scholar A100/V100 nodes (Ampere or Turing arch)
pip install "unsloth[cu118-ampere-torch221]" 2>/dev/null || \
pip install "unsloth[cu118]" 2>/dev/null || \
pip install unsloth

pip install open3d trimesh plyfile joblib rich matplotlib pyyaml datasets

echo ""
echo "===================================================================="
echo "Setup complete for user: $USERNAME"
echo ""
echo "NEXT STEPS:"
echo ""
echo "  1. Check Scholar's GPU partition name:"
echo "     sinfo -s"
echo "     sinfo -o \"%P %G %l %C\" | grep -i gpu"
echo ""
echo "  2. If partition is not 'gpu', edit the SLURM scripts:"
echo "     nano $STEPLLM_DIR/slurm_sft.sh    # change --partition=gpu"
echo "     nano $STEPLLM_DIR/slurm_rl.sh     # change --partition=gpu"
echo ""
echo "  3. Verify CUDA works (in an interactive job or after module load):"
echo "     python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\""
echo ""
echo "  4. Submit SFT training (runs ~48 hours on A100):"
echo "     mkdir -p $STEPLLM_DIR/logs"
echo "     sbatch $STEPLLM_DIR/slurm_sft.sh"
echo ""
echo "  5. Monitor:"
echo "     squeue -u $USERNAME"
echo "     tail -f $STEPLLM_DIR/logs/sft_*.out"
echo ""
echo "  6. After SFT completes, submit RL:"
echo "     sbatch $STEPLLM_DIR/slurm_rl.sh"
echo "===================================================================="
