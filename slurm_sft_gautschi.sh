#!/bin/bash
# =============================================================================
# StepForge — SFT job for Gautschi (Purdue RCAC)
# Hardware:  1× NVIDIA H100 80GB  (Unsloth is single-GPU only)
# Node type: Gautschi-H (8× H100 per node — we only request 1)
#
# Submit: sbatch slurm_sft_gautschi.sh
# Check:  squeue -u $USER
# Log:    tail -f $SCRATCH/stepforge/logs/sft_<JOBID>.out
# =============================================================================
#SBATCH --job-name=stepforge_sft
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14          # AI partition requires 14 CPUs per GPU
#SBATCH --mem=128G                  # well within the 1 TB node limit
#SBATCH --gres=gpu:1                # 1× H100 80GB — Unsloth does not support multi-GPU
#SBATCH --partition=ai
#SBATCH --account=lilly-agentic-gpu

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
export HF_HOME="$SCRATCH/.hf-cache"         # cache model weights on scratch, not home
export PYTHONPATH="${HOME}/StepForge:${PYTHONPATH:-}"
export KMP_DUPLICATE_LIB_OK=TRUE
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Dependency pins (self-healing) ───────────────────────────────────────────
pip install -q "trl==0.14.0" "transformers==4.51.3"
pip uninstall -q torchao -y 2>/dev/null || true
# Apply both TRL 0.14.0 compatibility patches
python - <<'PATCH'
import os, trl
trl_dir = os.path.dirname(trl.__file__)

# Patch 1: FSDPModule — added in torch 2.6, not in 2.5.1
p1 = os.path.join(trl_dir, "models", "utils.py")
txt = open(p1).read()
if "from torch.distributed.fsdp import FSDPModule" in txt and "except ImportError" not in txt:
    txt = txt.replace(
        "from torch.distributed.fsdp import FSDPModule",
        "try:\n    from torch.distributed.fsdp import FSDPModule\nexcept ImportError:\n    FSDPModule = None"
    )
    open(p1, "w").write(txt)
    print("Patched FSDPModule in trl/models/utils.py")

# Patch 2: _LazyModule — removed from trl.import_utils in 0.14.0, unsloth still needs it
p2 = os.path.join(trl_dir, "import_utils.py")
txt2 = open(p2).read()
if "_LazyModule" not in txt2:
    txt2 += "\n# Compatibility shim for unsloth which imports _LazyModule from here\ntry:\n    from transformers.utils.import_utils import _LazyModule\nexcept ImportError:\n    _LazyModule = type('_LazyModule', (), {})\n"
    open(p2, "w").write(txt2)
    print("Patched _LazyModule into trl/import_utils.py")
PATCH

# ── Ensure output directories exist ──────────────────────────────────────────
mkdir -p "$SCRATCH/stepforge/logs"
mkdir -p "$SCRATCH/stepforge/checkpoints/sft"

# Redirect SLURM logs to scratch
LOG_DIR="$SCRATCH/stepforge/logs"
exec > >(tee -a "${LOG_DIR}/sft_${SLURM_JOB_ID}.out") 2>&1

# ── Job info ─────────────────────────────────────────────────────────────────
echo "========================================"
echo " StepForge SFT — Gautschi"
echo "========================================"
echo " Job ID   : $SLURM_JOB_ID"
echo " Node     : $(hostname)"
echo " GPUs     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo " Started  : $(date)"
echo " Config   : configs/config_gautschi.yaml"
echo "========================================"

# ── Run SFT ──────────────────────────────────────────────────────────────────
cd "${HOME}/StepForge"

python training/llama3_SFT_response.py \
    --config configs/config_gautschi.yaml

echo "========================================"
echo " SFT finished : $(date)"
echo "========================================"
