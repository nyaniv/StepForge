#!/bin/bash
# =============================================================================
# StepForge — evaluation job for Gautschi (Purdue RCAC)
# Hardware:  1× NVIDIA H100 80GB (single GPU is enough for inference)
#
# Runs both eval scripts back-to-back on the same checkpoint:
#   1. evaluation/evaluate.py             (original, no truncation match)
#   2. evaluation/evaluate_in_distribution.py (truncation matched to training)
#
# Submit: sbatch slurm_eval_gautschi.sh <checkpoint-path> [num-examples]
#   e.g. sbatch slurm_eval_gautschi.sh $SCRATCH/stepforge/checkpoints/rl/final 30
#
# Outputs JSON to $SCRATCH/stepforge/eval_<jobid>/ for both scripts plus
# a summary log file.
# =============================================================================
#SBATCH --job-name=stepforge_eval
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --partition=ai
#SBATCH --account=lilly-agentic-gpu

# ── Args ─────────────────────────────────────────────────────────────────────
CKPT="${1:?Usage: sbatch slurm_eval_gautschi.sh <checkpoint-path> [num-examples]}"
N_EXAMPLES="${2:-30}"

# ── Modules + conda ──────────────────────────────────────────────────────────
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi
module purge
module load anaconda/2024.10-py312
module load cuda/12.6.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stepforge

# ── Env ──────────────────────────────────────────────────────────────────────
export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:?Set HUGGINGFACE_TOKEN before submitting}"
export HF_TOKEN="$HUGGINGFACE_TOKEN"
export HF_HOME="$SCRATCH/.hf-cache"
export PYTHONPATH="${HOME}/StepForge:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore::DeprecationWarning"

# ── Output dir ───────────────────────────────────────────────────────────────
OUT_DIR="$SCRATCH/stepforge/eval_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"
exec > >(tee -a "${OUT_DIR}/eval.log") 2>&1

echo "========================================"
echo " StepForge eval"
echo "========================================"
echo " Job ID    : $SLURM_JOB_ID"
echo " Node      : $(hostname)"
echo " GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo " Checkpoint: $CKPT"
echo " N examples: $N_EXAMPLES"
echo " Out dir   : $OUT_DIR"
echo " Started   : $(date)"
echo "========================================"

cd "${HOME}/StepForge"

# ── Eval 1: original (no truncation match) ───────────────────────────────────
echo ""
echo "=== Eval 1: evaluate.py (no truncation match) ==="
python evaluation/evaluate.py \
    --checkpoint "$CKPT" \
    --config configs/config_gautschi.yaml \
    --max-examples "$N_EXAMPLES" \
    --out-json "$OUT_DIR/eval_original.json"
EVAL1_EXIT=$?

# ── Eval 2: in-distribution (training truncation mirrored) ───────────────────
echo ""
echo "=== Eval 2: evaluate_in_distribution.py (training truncation mirrored) ==="
python evaluation/evaluate_in_distribution.py \
    --checkpoint "$CKPT" \
    --config configs/config_gautschi.yaml \
    --max-examples "$N_EXAMPLES" \
    --out-json "$OUT_DIR/eval_in_dist.json"
EVAL2_EXIT=$?

echo ""
echo "========================================"
echo " Finished : $(date)"
echo " Eval 1 exit: $EVAL1_EXIT"
echo " Eval 2 exit: $EVAL2_EXIT"
echo " Outputs in : $OUT_DIR"
echo "========================================"
