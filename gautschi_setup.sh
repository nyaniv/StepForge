#!/bin/bash
# =============================================================================
# StepForge — One-time environment setup for Gautschi (Purdue RCAC)
#
# Run ONCE from a login node or an interactive session:
#   bash gautschi_setup.sh
#
# What this script does:
#   1. Creates the conda environment with pythonocc-core (conda-forge only)
#   2. Installs all pip packages with pinned compatible versions
#   3. Downloads the Text2CAD dataset to $SCRATCH
#   4. Creates all required output directories on $SCRATCH
#
# Key version constraints:
#   - torch==2.5.1+cu121  (cu121 wheel = CUDA 12.x compatible)
#   - trl==0.13.1         (has GRPOTrainer; >=0.14 imports FSDPModule→torch>=2.6)
#   - transformers==4.51.3 (unsloth_zoo compatible)
#   - torchao must be ABSENT (pulls in torch.int1 which doesn't exist in 2.5.1)
#   - pythonocc-core==7.7.2 must come from conda-forge (not pip)
#   - unsloth_zoo requires trl<=0.24.0 — 0.13.1 satisfies
# =============================================================================

set -euo pipefail

# ── User settings — edit these ────────────────────────────────────────────────
CONDA_ENV_NAME="stepforge"
PROJECT_DIR="${HOME}/StepForge"          # where you cloned/copied this repo
HF_TOKEN="${HUGGINGFACE_TOKEN:-}"        # or paste directly: HF_TOKEN="hf_..."
# ─────────────────────────────────────────────────────────────────────────────

echo "============================================================"
echo " StepForge Gautschi Setup"
echo " Project : $PROJECT_DIR"
echo " Scratch  : $SCRATCH"
echo " Env     : $CONDA_ENV_NAME"
echo "============================================================"

if [ -z "$SCRATCH" ]; then
    echo "ERROR: \$SCRATCH is not set. Source your environment or log in again."
    exit 1
fi

# ── 1. Load modules ──────────────────────────────────────────────────────────
echo "[1/5] Loading modules..."
module purge
module load anaconda/2024.10-py312
module load cuda/12.6.0

# ── 2. Create conda environment ──────────────────────────────────────────────
echo "[2/5] Creating conda environment '$CONDA_ENV_NAME'..."
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "  Environment '$CONDA_ENV_NAME' already exists — skipping creation."
else
    conda create -y -n "$CONDA_ENV_NAME" \
        python=3.10 \
        -c conda-forge
    echo "  Created base env with Python 3.10."
fi

# Activate the env for subsequent installs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

# Install pythonocc-core from conda-forge FIRST (before pip installs)
# Must be 7.7.2 to match the Text2CAD CadSeqProc build.
echo "  Installing pythonocc-core 7.7.2 from conda-forge..."
conda install -y -c conda-forge pythonocc-core=7.7.2

# ── 3. Install pip packages ──────────────────────────────────────────────────
echo "[3/5] Installing pip packages..."

# PyTorch 2.5.1 — CUDA 12.1 wheel works with CUDA 12.6 runtime
pip install \
    "torch==2.5.1" \
    "torchvision==0.20.1" \
    "torchaudio==2.5.1" \
    --index-url https://download.pytorch.org/whl/cu121

# Unsloth (SFT only — single-GPU, requires torch 2.5.x)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Flash Attention 2 — must build against installed torch/CUDA
# Use pre-built wheel to avoid cross-device link error on Gautschi scratch
mkdir -p "$SCRATCH/tmp"
FLASH_WHL="flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
if [ ! -f "$SCRATCH/tmp/$FLASH_WHL" ]; then
    wget -q "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/$FLASH_WHL" \
        -P "$SCRATCH/tmp/"
fi
pip install "$SCRATCH/tmp/$FLASH_WHL"

# Core training stack — pinned for torch 2.5.1 compatibility
# trl==0.13.1: has GRPOTrainer; >=0.14 imports FSDPModule which needs torch>=2.6
# unsloth_zoo requires trl<=0.24.0 — 0.13.1 satisfies
pip install \
    "trl==0.13.1" \
    "transformers==4.51.3" \
    "peft>=0.10" \
    "accelerate>=0.30" \
    "datasets" \
    "bitsandbytes"

# torchao gets pulled in by newer transformers but requires torch.int1 (torch>=2.6)
# Must uninstall AFTER transformers to ensure it's gone
pip uninstall torchao -y 2>/dev/null || true

# Text2CAD CadSeqProc dependencies
pip install \
    "trimesh==4.1.8" \
    "plyfile==0.9" \
    "pyvista" \
    "prettytable" \
    "nltk" \
    "python-dotenv"

# Retrieval, reward, and utility
pip install \
    "open3d" \
    "sentence-transformers" \
    "faiss-cpu" \
    "scipy" \
    "pandas" \
    "loguru" \
    "omegaconf" \
    "tqdm" \
    "gradio" \
    "rich" \
    "pillow" \
    "huggingface_hub"

echo "  All pip packages installed."

# ── 4. Download Text2CAD dataset ─────────────────────────────────────────────
echo "[4/5] Downloading Text2CAD dataset to \$SCRATCH..."

DATA_DIR="$SCRATCH/data"
mkdir -p "$DATA_DIR"

if [ -n "$HF_TOKEN" ]; then
    export HUGGINGFACE_TOKEN="$HF_TOKEN"
    export HF_TOKEN="$HF_TOKEN"
fi

python - <<PYEOF
import os, shutil
from huggingface_hub import hf_hub_download

DATA_DIR = "$DATA_DIR"
TOKEN = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
tmp_dir = f"{DATA_DIR}/.hf_tmp"
os.makedirs(tmp_dir, exist_ok=True)

files = [
    ("text2cad_v1.1/text2cad_v1.1.csv",  f"{DATA_DIR}/text2cad_v1.1.csv"),
    ("text2cad_v1.1/train_test_val.json", f"{DATA_DIR}/train_test_val.json"),
    ("cad_seq.zip",                        f"{DATA_DIR}/cad_seq.zip"),
]
for hf_fname, dest in files:
    if os.path.exists(dest):
        print(f"  already present: {dest}")
        continue
    if hf_fname == "cad_seq.zip" and os.path.isdir(f"{DATA_DIR}/cad_seq"):
        print(f"  cad_seq/ already extracted — skipping zip download")
        continue
    print(f"  downloading {hf_fname} ...")
    downloaded = hf_hub_download(
        repo_id="SadilKhan/Text2CAD",
        repo_type="dataset",
        filename=hf_fname,
        local_dir=tmp_dir,
        token=TOKEN,
    )
    shutil.move(downloaded, dest)
    print(f"  saved to {dest}")

shutil.rmtree(tmp_dir, ignore_errors=True)
PYEOF

if [ -f "$DATA_DIR/cad_seq.zip" ] && [ ! -d "$DATA_DIR/cad_seq" ]; then
    echo "  Extracting cad_seq.zip..."
    unzip -qo "$DATA_DIR/cad_seq.zip" -d "$DATA_DIR/"
    echo "  Extraction complete."
fi

# Clone Text2CAD source (for CadSeqProc)
if [ ! -d "$DATA_DIR/Text2CAD" ]; then
    echo "  Cloning Text2CAD source..."
    git clone --depth=1 https://github.com/SadilKhan/Text2CAD.git "$DATA_DIR/Text2CAD"
else
    echo "  Text2CAD source already present — skipping."
fi

# ── 5. Create output directories on scratch ──────────────────────────────────
echo "[5/5] Creating output directories on \$SCRATCH..."

STEPFORGE_SCRATCH="$SCRATCH/stepforge"
mkdir -p \
    "$STEPFORGE_SCRATCH/processed/step_files" \
    "$STEPFORGE_SCRATCH/retrieval" \
    "$STEPFORGE_SCRATCH/checkpoints/sft" \
    "$STEPFORGE_SCRATCH/checkpoints/rl" \
    "$STEPFORGE_SCRATCH/logs" \
    "$SCRATCH/.hf-cache" \
    "$SCRATCH/tmp"

echo "  Directories created under $STEPFORGE_SCRATCH"

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Setup complete!"
echo ""
echo " NEXT STEPS:"
echo "   1. Run the data pipeline:"
echo "        sbatch slurm_data_gautschi.sh"
echo ""
echo "   2. Submit SFT job:"
echo "        sbatch slurm_sft_gautschi.sh"
echo ""
echo "   3. After SFT completes, submit RL job:"
echo "        sbatch slurm_rl_gautschi.sh"
echo ""
echo "   Check queue status: squeue -u \$USER"
echo "   Check quota:        myquota"
echo "============================================================"
