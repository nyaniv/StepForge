#!/bin/bash
# =============================================================================
# StepForge — Data pipeline job for Gautschi (Purdue RCAC)
# Hardware:  CPU only (Gautschi-A node)
#
# Runs the full data preprocessing pipeline:
#   1. export_steps.py    — .pth → STEP files (176k files, ~2 hrs)
#   2. pair_captions.py   — pair STEP files with text captions
#   3. dfs_reserializer.py — reserialize CAD sequences
#   4. filter_dataset.py  — filter invalid samples
#   5. build_index.py     — build FAISS retrieval index
#   6. precompute_rag.py  — precompute RAG context
#
# Submit: sbatch slurm_data_gautschi.sh
# Check:  squeue -u $USER
# Log:    tail -f $SCRATCH/stepforge/logs/data_<JOBID>.out
# =============================================================================
#SBATCH --job-name=stepforge_data
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --partition=lilly-agentic-cpu
#SBATCH --account=lilly-agentic-cpu

# ── Source the module system ─────────────────────────────────────────────────
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi

# ── Load modules ─────────────────────────────────────────────────────────────
module purge
module load anaconda/2024.10-py312

# ── Activate conda environment ───────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stepforge

# ── Environment variables ────────────────────────────────────────────────────
export PYTHONPATH="${HOME}/StepForge:${PYTHONPATH:-}"
export KMP_DUPLICATE_LIB_OK=TRUE
export TOKENIZERS_PARALLELISM=false

# ── Ensure output directories exist ──────────────────────────────────────────
mkdir -p "$SCRATCH/stepforge/logs"
mkdir -p "$SCRATCH/stepforge/processed/step_files"
mkdir -p "$SCRATCH/stepforge/retrieval"

# Redirect logs to scratch
LOG_DIR="$SCRATCH/stepforge/logs"
exec > >(tee -a "${LOG_DIR}/data_${SLURM_JOB_ID}.out") 2>&1

# ── Job info ─────────────────────────────────────────────────────────────────
echo "========================================"
echo " StepForge Data Pipeline — Gautschi"
echo "========================================"
echo " Job ID  : $SLURM_JOB_ID"
echo " Node    : $(hostname)"
echo " CPUs    : $SLURM_CPUS_PER_TASK"
echo " Started : $(date)"
echo "========================================"

cd "${HOME}/StepForge"

echo "[1/6] Exporting STEP files..."
python data/export_steps.py --config configs/config_gautschi.yaml --workers 64

echo "[2/6] Pairing captions..."
python data/pair_captions.py --config configs/config_gautschi.yaml

echo "[3/6] DFS reserializer..."
python data/dfs_reserializer.py --config configs/config_gautschi.yaml

echo "[4/6] Filtering dataset..."
python data/filter_dataset.py --config configs/config_gautschi.yaml

echo "[5/6] Building FAISS index..."
python retrieval/build_index.py --config configs/config_gautschi.yaml

echo "[6/6] Precomputing RAG..."
python data/precompute_rag.py --config configs/config_gautschi.yaml

echo "========================================"
echo " Data pipeline finished : $(date)"
echo "========================================"
