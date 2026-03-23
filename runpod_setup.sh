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

# ── System dependencies ────────────────────────────────────────────────────────
apt-get update -qq && apt-get install -y unzip curl > /dev/null 2>&1

# ── Miniforge ──────────────────────────────────────────────────────────────────
# Install on the container disk (fast local SSD, 100GB) — NOT the network volume.
# Only data + checkpoints go on the network volume (persistent).
MINIFORGE="/opt/miniforge"
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

# Redirect pip cache and HuggingFace cache to network volume.
# HF_HOME covers model weights, tokenizers, datasets — easily 10-20 GB.
# NOTE: conda packages stay on local disk — putting CONDA_PKGS_DIRS on the network
# volume causes cross-filesystem hardlink failures → silent hang during env creation.
export PIP_CACHE_DIR="$VOLUME/.pip-cache"
export HF_HOME="$VOLUME/.hf-cache"

# Persist for interactive sessions
grep -qF "miniforge/etc/profile.d/conda.sh" ~/.bashrc \
    || echo "source $MINIFORGE/etc/profile.d/conda.sh" >> ~/.bashrc
grep -qF "PIP_CACHE_DIR" ~/.bashrc \
    || echo "export PIP_CACHE_DIR=$VOLUME/.pip-cache" >> ~/.bashrc
grep -qF "HF_HOME" ~/.bashrc \
    || echo "export HF_HOME=$VOLUME/.hf-cache" >> ~/.bashrc
grep -qF "PYTORCH_ALLOC_CONF" ~/.bashrc \
    || echo "export PYTORCH_ALLOC_CONF=expandable_segments:True" >> ~/.bashrc

# ── stepforge conda env ────────────────────────────────────────────────────────
# open3d is installed via pip below (same package, avoids conda-forge solver overhead)
if conda env list | grep -q "^stepforge "; then
    echo "==> conda env 'stepforge' already exists"
else
    echo "==> Creating stepforge env (Python 3.11 + pythonocc-core from conda-forge)..."
    conda create -n stepforge python=3.11 \
        pythonocc-core=7.7.2 \
        -c conda-forge -y
fi

conda activate stepforge

# ── pip packages (into stepforge) ─────────────────────────────────────────────
echo "==> Installing pip packages into stepforge..."
pip install --quiet --no-cache-dir --root-user-action=ignore \
    "open3d" \
    "trimesh==4.1.8" \
    "plyfile==0.9" \
    "pyvista" \
    "rich" \
    "prettytable" \
    "nltk" \
    "python-dotenv" \
    "pillow" \
    "accelerate" \
    "gradio" \
    "transformers>=4.51.3" \
    "trl==0.29.0" \
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

# Install unsloth — newer versions dropped the cuXXX extras, just use base package
echo "==> Installing unsloth..."
pip install --quiet --no-cache-dir --root-user-action=ignore "unsloth" --upgrade

# ── Output directories ─────────────────────────────────────────────────────────
echo "==> Creating output directories..."
mkdir -p "$VOLUME/data" \
         "$VOLUME/processed/step_files" \
         "$VOLUME/processed/dfs_step" \
         "$VOLUME/retrieval" \
         "$VOLUME/checkpoints/sft" \
         "$VOLUME/checkpoints/rl" \
         "$REPO/logs"

# ── Download Text2CAD dataset from HuggingFace (skips if already present) ─────
echo "==> Checking Text2CAD dataset..."
python - <<PYEOF
import os, shutil
from huggingface_hub import hf_hub_download

VOLUME = os.environ["VOLUME"]
TOKEN  = os.environ["HUGGINGFACE_TOKEN"]

# Map: (HuggingFace filename in repo, local destination path)
files = [
    ("cad_seq.zip",                       f"{VOLUME}/data/cad_seq.zip"),
    ("text2cad_v1.1/text2cad_v1.1.csv",   f"{VOLUME}/data/text2cad_v1.1.csv"),
    ("text2cad_v1.1/train_test_val.json",  f"{VOLUME}/data/train_test_val.json"),
]
tmp_dir = f"{VOLUME}/data/.hf_tmp"
for hf_fname, dest in files:
    if os.path.exists(dest):
        print(f"  already present: {dest}")
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
# Clean up temp dir
shutil.rmtree(tmp_dir, ignore_errors=True)
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
echo "  # Step 1 — DFS-restructure raw STEP files (~2h, skips already-done files)"
echo "  python data/batch_restructure.py    --config configs/config_runpod.yaml"
echo ""
echo "  # Step 2 — Build RAG dataset (pairs each STEP with nearest-neighbour retrieval)"
echo "  python data/dataset_construct_rag.py --config configs/config_runpod.yaml"
echo ""
echo "  # Step 3 — Split into train / val / test"
echo "  python data/data_split.py           --config configs/config_runpod.yaml"
echo ""
echo "  # Step 4 — SFT training (~40 epochs on A100)"
echo "  python training/llama3_SFT_response.py"
echo ""
echo "  # Step 5 — RL training (cold-starts from SFT checkpoint)"
echo "  python training/rl_train.py  --config configs/config_runpod.yaml"
