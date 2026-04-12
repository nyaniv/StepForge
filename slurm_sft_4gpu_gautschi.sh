#!/bin/bash
# =============================================================================
# StepForge — 4-GPU SFT job for Gautschi (main branch)
# Hardware:  4× NVIDIA H100 80GB
# Effective batch: 4 per_device × 1 grad_accum × 4 GPUs = 16  (paper spec)
#
# Queues faster than 8-GPU (half the resources).
# Writes to its own namespaced run dir — no collision with 8-GPU runs.
#
# Submit:  sbatch slurm_sft_4gpu_gautschi.sh
# Resume:  sbatch slurm_sft_4gpu_gautschi.sh /path/to/existing/run/dir
# Log:     tail -f $SCRATCH/stepforge/runs/sft_4gpu_<JOBID>/slurm.out
# =============================================================================
#SBATCH --job-name=stepforge_sft_4gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=14
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --partition=ai
#SBATCH --account=lilly-agentic-gpu
#SBATCH --requeue
#SBATCH --qos=preemptible
#SBATCH --signal=B:SIGUSR1@120

if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi

module purge
module load anaconda/2024.10-py312
module load cuda/12.6.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stepforge

export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:?Set HUGGINGFACE_TOKEN before submitting}"
export HF_HOME="$SCRATCH/.hf-cache"
export PYTHONPATH="${HOME}/StepForge:${PYTHONPATH:-}"
export KMP_DUPLICATE_LIB_OK=TRUE
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker

# ── Namespaced run directory ──────────────────────────────────────────────────
if [ -n "${1:-}" ] && [ -d "$1" ]; then
    RUN_DIR="$1"
    echo "[$(date)] Resuming existing run: $RUN_DIR"
else
    RUN_DIR="$SCRATCH/stepforge/runs/sft_4gpu_${SLURM_JOB_ID}"
    mkdir -p "$RUN_DIR"
    echo "[$(date)] New run directory: $RUN_DIR"
fi

exec > >(tee -a "${RUN_DIR}/slurm.out") 2>&1

# ── Dependency pins ───────────────────────────────────────────────────────────
pip install -q "trl==0.14.0" "transformers==4.51.3"
pip uninstall -q torchao -y 2>/dev/null || true
python - <<'PATCH'
import os, trl
trl_dir = os.path.dirname(trl.__file__)
p1 = os.path.join(trl_dir, "models", "utils.py")
txt = open(p1).read()
if "from torch.distributed.fsdp import FSDPModule" in txt and "except ImportError" not in txt:
    txt = txt.replace(
        "from torch.distributed.fsdp import FSDPModule",
        "try:\n    from torch.distributed.fsdp import FSDPModule\nexcept ImportError:\n    FSDPModule = None"
    )
    open(p1, "w").write(txt)
    print("Patched FSDPModule in trl/models/utils.py")
p2 = os.path.join(trl_dir, "import_utils.py")
txt2 = open(p2).read()
if "_LazyModule" not in txt2:
    txt2 += "\ntry:\n    from transformers.utils.import_utils import _LazyModule\nexcept ImportError:\n    _LazyModule = type('_LazyModule', (), {})\n"
    open(p2, "w").write(txt2)
    print("Patched _LazyModule into trl/import_utils.py")
PATCH

# ── Signal handler ────────────────────────────────────────────────────────────
_resubmit() {
    echo ""
    echo "[$(date)] Time limit approaching — resubmitting into same run dir: $RUN_DIR"
    sbatch "${HOME}/StepForge/slurm_sft_4gpu_gautschi.sh" "$RUN_DIR"
    echo "[$(date)] Resubmit issued. Exiting current job gracefully."
    exit 0
}
trap _resubmit SIGUSR1

echo "========================================"
echo " StepForge SFT 4-GPU — Gautschi"
echo "========================================"
echo " Job ID   : $SLURM_JOB_ID"
echo " Run dir  : $RUN_DIR"
echo " Node     : $(hostname)"
echo " GPUs     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo " World    : 4 processes (1 per GPU)"
echo " Eff batch: 4 × 1 × 4 = 16  (paper spec)"
echo " Started  : $(date)"
echo " Config   : configs/config_gautschi.yaml"
echo "========================================"

cd "${HOME}/StepForge"

echo "Running preflight environment check..."
python training/preflight_check.py || { echo "PREFLIGHT FAILED — aborting job"; exit 1; }

torchrun \
    --standalone \
    --nproc_per_node=4 \
    training/sft_multigpu.py \
        --config configs/config_gautschi.yaml \
        --output-dir "$RUN_DIR" \
        --per-device-batch 4 &

wait $!
SFT_EXIT=$?

echo "========================================"
echo " SFT 4-GPU finished : $(date)  (exit=$SFT_EXIT)"
echo "========================================"

if [ $SFT_EXIT -eq 0 ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    FINAL_DIR="${RUN_DIR}/final"
    SNAPSHOT_DIR="${RUN_DIR}/sft_weights_${TIMESTAMP}"
    if [ -d "$FINAL_DIR" ]; then
        echo "Saving timestamped weight snapshot to $SNAPSHOT_DIR ..."
        cp -r "$FINAL_DIR" "$SNAPSHOT_DIR"
        echo "Snapshot saved: $SNAPSHOT_DIR"
    fi
fi
