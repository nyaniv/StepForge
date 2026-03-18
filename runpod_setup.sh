#!/usr/bin/env bash
# runpod_setup.sh — one-time (or idempotent) setup for RunPod pods.
#
# What it does:
#   1. Loads secrets from .env (optional — falls back to pod env vars)
#   2. Installs Miniforge if absent
#   3. Creates 'stepforge' conda env (Python 3.11) with conda-forge packages
#   4. Installs pip-only packages (transformers, trl, unsloth, etc.) into env
#   5. Downloads Text2CAD dataset from HuggingFace (skips if already present)
#   6. Clones Text2CAD source code (for export_steps.py)
#   7. Creates all required output directories
#   8. Validates everything is in place
#
# Usage:
#   bash runpod_setup.sh
#   (optional) cp .env.example .env  # fill in HUGGINGFACE_TOKEN if not set as pod env var
#
# Run this once after cloning the repo. Safe to re-run — skips completed steps.

set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
VOLUME="${VOLUME:-/runpod-volume}"

# ── Load secrets ───────────────────────────────────────────────────────────────
if [ -f "$REPO/.env" ]; then
    echo "==> Loading secrets from .env..."
    set -a; source "$REPO/.env"; set +a
else
    echo "==> No .env found — using pod environment variables"
fi

: "${HUGGINGFACE_TOKEN:?HUGGINGFACE_TOKEN must be set (in .env or as a pod env var)}"
export VOLUME REPO

echo "==> REPO   = $REPO"
echo "==> VOLUME = $VOLUME"

# ── Miniforge ──────────────────────────────────────────────────────────────────
# Install onto the network volume to avoid filling the container disk (~10GB limit)
MINIFORGE="${VOLUME}/miniforge"
if [ ! -d "$MINIFORGE" ]; then
    echo "==> Installing Miniforge..."
    curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -o /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$MINIFORGE"
    rm /tmp/miniforge.sh
else
    echo "==> Miniforge already installed"
fi

# Make conda available in this shell
source "$MINIFORGE/etc/profile.d/conda.sh"

# Persist conda init for interactive sessions
grep -qF "miniforge/etc/profile.d/conda.sh" ~/.bashrc \
    || echo "source \$VOLUME/miniforge/etc/profile.d/conda.sh" >> ~/.bashrc

# ── stepforge conda env ────────────────────────────────────────────────────────
if conda env list | grep -q "^stepforge "; then
    echo "==> conda env 'stepforge' already exists"
else
    echo "==> Creating stepforge env (Python 3.11 + conda-forge packages)..."
    conda create -n stepforge python=3.11 \
        pythonocc-core=7.7.2 \
        open3d \
        -c conda-forge -y
fi

conda activate stepforge

# ── pip packages (into stepforge) ─────────────────────────────────────────────
echo "==> Installing pip packages into stepforge..."
pip install --quiet \
    "transformers>=4.40" \
    "trl>=0.8.6" \
    "peft>=0.10" \
    "datasets" \
    "sentence-transformers" \
    "faiss-cpu" \
    "scipy" \
    "bitsandbytes" \
    "pandas" \
    "loguru" \
    "omegaconf" \
    "tqdm" \
    "huggingface_hub"

# Install correct unsloth CUDA variant
CUDA_VER=$(python -c "import torch; v=torch.version.cuda; print(v.replace('.','')[:3])" 2>/dev/null || echo "124")
echo "==> Installing unsloth[cu${CUDA_VER}]..."
pip install --quiet "unsloth[cu${CUDA_VER}]" --upgrade

# ── Output directories ─────────────────────────────────────────────────────────
echo "==> Creating output directories..."
mkdir -p "$VOLUME/data" \
         "$VOLUME/processed/step_files" \
         "$VOLUME/retrieval" \
         "$VOLUME/checkpoints/sft" \
         "$VOLUME/checkpoints/rl" \
         "$REPO/logs"

# ── Download Text2CAD dataset from HuggingFace (skips if already present) ─────
echo "==> Checking Text2CAD dataset..."
python - <<PYEOF
import os, sys
from huggingface_hub import hf_hub_download

VOLUME = os.environ["VOLUME"]
TOKEN  = os.environ["HUGGINGFACE_TOKEN"]

files = [
    ("cad_seq.zip",          f"{VOLUME}/data/cad_seq.zip"),
    ("text2cad_v1.1.csv",    f"{VOLUME}/data/text2cad_v1.1.csv"),
    ("train_test_val.json",  f"{VOLUME}/data/train_test_val.json"),
]
for fname, dest in files:
    if os.path.exists(dest):
        print(f"  already present: {dest}")
        continue
    print(f"  downloading {fname} ...")
    hf_hub_download(
        repo_id="SadilKhan/Text2CAD",
        repo_type="dataset",
        filename=fname,
        local_dir=f"{VOLUME}/data",
        token=TOKEN,
    )
    print(f"  saved to {dest}")
PYEOF

# Unzip cad_seq if needed
if [ ! -d "$VOLUME/data/cad_seq" ]; then
    echo "==> Unzipping cad_seq.zip..."
    unzip -q "$VOLUME/data/cad_seq.zip" -d "$VOLUME/data/"
else
    echo "==> cad_seq/ already unzipped"
fi

# ── Clone Text2CAD source (for export_steps.py) ────────────────────────────────
if [ ! -d "$VOLUME/data/Text2CAD" ]; then
    echo "==> Cloning Text2CAD source..."
    git clone --depth=1 https://github.com/SadilKhan/Text2CAD.git "$VOLUME/data/Text2CAD"
else
    echo "==> Text2CAD source already present"
fi

# ── Validate ───────────────────────────────────────────────────────────────────
echo "==> Validating..."
REQUIRED=(
    "$VOLUME/data/cad_seq"
    "$VOLUME/data/text2cad_v1.1.csv"
    "$VOLUME/data/train_test_val.json"
    "$VOLUME/data/Text2CAD/CadSeqProc"
)
ALL_OK=1
for f in "${REQUIRED[@]}"; do
    if [ -e "$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f"
        ALL_OK=0
    fi
done

if [ "$ALL_OK" -eq 0 ]; then
    echo "ERROR: Some required files are missing. Check the output above."
    exit 1
fi

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo "Setup complete. Activate the env and run training with:"
echo ""
echo "  conda activate stepforge"
echo "  cd $REPO"
echo ""
echo "  # Full pipeline (data prep — only needed once; persists on network volume)"
echo "  python data/build_dataset.py    --config configs/config_runpod.yaml"
echo "  python retrieval/build_index.py --config configs/config_runpod.yaml"
echo "  python data/precompute_rag.py   --config configs/config_runpod.yaml"
echo ""
echo "  # Training"
echo "  python training/sft_train.py --config configs/config_runpod.yaml"
echo "  python training/rl_train.py  --config configs/config_runpod.yaml"
