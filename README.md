# StepForge

An independent reimplementation of [STEP-LLM](https://arxiv.org/abs/2601.12641)
(Chen et al., 2026), developed as part of an undergraduate research project at
Purdue University with Prof. Gomez.

The goal: fine-tune open-source LLMs (starting with Llama-3.2-3B-Instruct) to
generate raw STEP files (ISO 10303) from natural language descriptions of 3D parts,
using supervised fine-tuning and GRPO reinforcement learning with geometric reward
signals.

This is a from-scratch implementation — not a fork of the
[official code](https://github.com/JasonShiii/STEP-LLM) — built to support
ongoing extensions including agentic text → CAD → FEA orchestration, RAG-based
retrieval of design specifications, and experimentation with alternative base
models and reward functions.

---

## Status

Reimplementation of the core SFT + GRPO pipeline is complete.
Reproduction of published metrics (see below) is in progress.

| Phase | Status |
|-------|--------|
| Data pipeline (export, parse, reserialize, pair, split) | ✅ Done |
| RAG index build + pre-computation | ✅ Done |
| SFT training (10 epochs, Llama-3.2-3B) | ✅ Done — checkpoint saved |
| RL training (GRPO, 80 steps) | ⏳ In progress — RunPod A100 SXM 80 GB |
| Evaluation | ⏳ Pending — runs after RL |

---

## Comparison to prior work

The STEP-LLM paper reports the following results against the
[Text2CAD](https://github.com/SadilKhan/Text2CAD) baseline
(Khan et al., NeurIPS 2024):

| Method | CR (%) | RR (%) | MSCD | AEC |
|--------|--------|--------|------|-----|
| Text2CAD | — | 98.38 | 3.99 | 390.41 |
| STEP-LLM (SFT) | 97.00 | 95.18 | 0.53 | 240.99 |
| STEP-LLM (GRPO) | 99.00 | 92.00 | 0.098 | — |

*These are the paper's reported numbers, not yet independently reproduced here.*

---

## Attribution

The dataset (`cad_seq.zip`, ~170K `.pth` CAD vectors and `captions.csv`) and one
export script (`data/export_steps.py`) are adapted from
[Text2CAD](https://github.com/SadilKhan/Text2CAD) (Apache 2.0). Everything else
was written from scratch. See [ATTRIBUTION.md](ATTRIBUTION.md) for a file-by-file
breakdown.

---

## Setup

### RunPod (A100 SXM 80 GB — primary)

```bash
# One-time setup on a new pod — idempotent, safe to re-run
export HUGGINGFACE_TOKEN=your_token_here
bash runpod_setup.sh
conda activate stepforge
```

### Local / other

```bash
conda env create -f environment.yml
conda activate stepforge
export HUGGINGFACE_TOKEN=your_token_here   # required for Llama gated model
```

---

## Running

Steps 1–4 are complete. Step 5 (RL) is running on RunPod.
Use `configs/config_runpod.yaml` on the pod (paths read from `$VOLUME`).

```bash
# Step 1: Build dataset (export STEP files, reserialize, pair captions, split)
python data/build_dataset.py --config configs/config_runpod.yaml

# Step 2: Build FAISS retrieval index
python retrieval/build_index.py --config configs/config_runpod.yaml

# Step 3: Pre-compute RAG for training data
python data/precompute_rag.py --config configs/config_runpod.yaml

# Step 4: SFT (~10 epochs, ~2 days on A100 SXM 80 GB)
python training/sft_train.py --config configs/config_runpod.yaml

# Step 5: RL with GRPO (80 steps, cold-starts from SFT checkpoint)
python training/rl_train.py --config configs/config_runpod.yaml

# Step 6: Evaluate
python evaluation/evaluate.py --checkpoint checkpoints/rl/final --config configs/config_runpod.yaml
```

### Quick inference (after training)

```bash
python inference/generate.py \
    --caption "a hollow cylinder" \
    --output /tmp/cylinder.step \
    --checkpoint checkpoints/rl/final
```

### Verification

After each phase, run a quick sanity check:

```bash
# After Step 1: verify DFS reserialization produces valid geometry
python -c "
from data.step_parser import parse_step
from data.dfs_reserializer import reserialize
from reward.step_to_pointcloud import step_to_pointcloud
import glob, sys
files = glob.glob('data/step_files/*.step')
if not files: sys.exit('No step files found')
header, entities, ref_by = parse_step(files[0])
reser = reserialize(header, entities, ref_by)
pc = step_to_pointcloud(reser)
print('Point cloud shape:', pc.shape if pc is not None else 'FAILED')
"

# After Step 2: verify retriever
python -c "
from retrieval.retriever import Retriever
r = Retriever('retrieval/caption_index.faiss', 'retrieval/metadata.pkl')
result = r.retrieve('a hollow cylinder')
print('Retrieved UID:', result['uid'])
print('Caption:', result['caption'][:80])
"
```

---

## Project structure

```
StepForge/
├── configs/
│   ├── config.yaml              # All hyperparameters and local paths
│   ├── config_runpod.yaml       # RunPod A100 SXM 80 GB (primary)
│   └── config_scholar.yaml      # Purdue Scholar Cluster (archived)
├── data/
│   ├── export_steps.py          # [adapted from Text2CAD] .pth → STEP files
│   ├── step_parser.py           # Parse STEP entity DAG
│   ├── dfs_reserializer.py      # DFS traversal + CoT annotations
│   ├── pair_captions.py         # Pair STEP with abstract captions
│   ├── filter_dataset.py        # Filter + split
│   ├── precompute_rag.py        # Pre-retrieve top-1 STEP per training example
│   └── build_dataset.py         # Orchestrator
├── retrieval/
│   ├── build_index.py           # Build FAISS caption index
│   └── retriever.py             # Live RAG retrieval
├── training/
│   ├── sft_train.py             # SFT with LoRA via Unsloth
│   └── rl_train.py              # GRPO reinforcement learning
├── reward/
│   ├── step_to_pointcloud.py    # STEP → 3D point cloud
│   ├── alignment.py             # FPFH+RANSAC+ICP alignment
│   └── scd_reward.py            # Scaled Chamfer Distance reward
├── inference/generate.py        # Generate STEP from caption
├── evaluation/evaluate.py       # CR, RR, MSCD, AEC metrics
├── app.py                       # Gradio demo
├── ATTRIBUTION.md               # File-by-file code origin breakdown
├── LICENSE                      # Apache 2.0
├── runpod_setup.sh              # One-time pod setup (installs env + downloads data)
└── slurm_sft.sh / slurm_rl.sh  # Archived — Purdue Scholar SLURM scripts
```

---

## References

- **STEP-LLM**: Chen et al., 2026. *STEP-LLM: Generating CAD STEP Models from Natural Language with Large Language Models.* arXiv:2601.12641. [[Paper]](https://arxiv.org/abs/2601.12641) · [[Official code]](https://github.com/JasonShiii/STEP-LLM)

- **Text2CAD**: Khan et al., NeurIPS 2024 Spotlight. *Text2CAD: Generating Sequential CAD Designs from Beginner-to-Expert Level Text Prompts.* [[Paper]](https://arxiv.org/abs/2409.17106) · [[Code]](https://github.com/SadilKhan/Text2CAD) · [[Project page]](https://sadilkhan.github.io/text2cad-project/)
