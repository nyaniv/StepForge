#!/usr/bin/env bash
# setup_pod.sh — one-shot environment setup for a fresh RunPod container.
# Usage: bash setup_pod.sh
# Requires: HUGGINGFACE_TOKEN and VOLUME env vars to be set beforehand.

set -e

echo "=== Installing system packages ==="
apt-get update -qq && apt-get install -y tmux libxrender1

echo "=== Installing PyTorch 2.6.0 (CUDA 12.4) ==="
pip install --ignore-installed \
    "torch==2.6.0" "torchvision==0.21.0" \
    --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing Python packages ==="
pip install --ignore-installed \
    "trl==0.29.0" \
    transformers peft accelerate \
    sentence-transformers faiss-cpu \
    open3d cadquery \
    scipy bitsandbytes \
    datasets pandas loguru omegaconf tqdm

echo "=== Verifying OCP ==="
python -c "from OCP.STEPControl import STEPControl_Reader; print('OCP ok')"

echo "=== Verifying torch ==="
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())"

echo "=== Done. Now run: ==="
echo "  export HUGGINGFACE_TOKEN=hf_..."
echo "  export VOLUME=/runpod-volume"
echo "  tmux new -s rl"
echo "  python training/rl_train.py --config configs/config_runpod.yaml"
