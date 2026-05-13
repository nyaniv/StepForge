#!/bin/bash
# =============================================================================
# StepForge — in-distribution eval ONLY (for resubmission after 8h wall hits)
#
# Submit: sbatch slurm_eval_indist_gautschi.sh <checkpoint-path> [num-examples]
# =============================================================================
#SBATCH --job-name=stepforge_eval_indist
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --partition=ai
#SBATCH --account=lilly-agentic-gpu

CKPT="${1:?Usage: sbatch slurm_eval_indist_gautschi.sh <checkpoint-path> [num-examples]}"
N_EXAMPLES="${2:-100}"

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
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore::DeprecationWarning"

OUT_DIR="$SCRATCH/stepforge/eval_indist_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"
exec > >(tee -a "${OUT_DIR}/eval.log") 2>&1

echo "========================================"
echo " Job ID    : $SLURM_JOB_ID"
echo " Node      : $(hostname)"
echo " Checkpoint: $CKPT"
echo " N examples: $N_EXAMPLES"
echo " Out dir   : $OUT_DIR"
echo " Started   : $(date)"
echo "========================================"

cd "${HOME}/StepForge"

python evaluation/evaluate_in_distribution.py \
    --checkpoint "$CKPT" \
    --config configs/config_gautschi.yaml \
    --max-examples "$N_EXAMPLES" \
    --out-json "$OUT_DIR/eval_in_dist.json"

echo ""
echo "========================================"
echo " Finished : $(date)"
echo "========================================"
