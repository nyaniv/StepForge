# StepForge

An independent, from-scratch reimplementation of
[STEP-LLM](https://arxiv.org/abs/2601.12641) (Chen et al., 2026), developed
as an undergraduate research project at Purdue University with Prof. Gomez.

**This is not a fork of the [official STEP-LLM code](https://github.com/JasonShiii/STEP-LLM).**
Every component — data pipeline, retrieval system, training infrastructure,
reward functions, evaluation, and inference — was written independently.

---

## What this project is

The goal: fine-tune open-source LLMs to generate raw STEP files (ISO 10303)
from natural language descriptions of 3D parts, using supervised fine-tuning
followed by GRPO reinforcement learning with a geometric reward signal.

The approach follows the STEP-LLM paper. The dataset comes from Text2CAD.
Everything else — the code, the infrastructure, the engineering — is original work.

---

## My contributions

The following was built from scratch for this project:

- **Multi-GPU SFT training** (`training/sft_multigpu.py`) — HuggingFace Trainer
  + PEFT/LoRA with DDP across 4× H100 80GB on Purdue's Gautschi HPC cluster,
  including checkpoint resumption, gradient checkpointing, memory diagnostics,
  and automatic SLURM resubmission chains
- **GRPO RL training** (`training/rl_train.py`) — TRL GRPOTrainer with live
  RAG retrieval, three reward functions (format, parse, geometry), and
  distributed training across 4× H100
- **Data pipeline** — STEP file parsing, DFS reserializer with chain-of-thought
  annotations, caption pairing, dataset filtering and splitting
- **RAG system** — FAISS caption index, SentenceTransformer embeddings,
  pre-computed retrieval for SFT and live retrieval for RL
- **Reward functions** — STEP → 3D point cloud conversion, FPFH+RANSAC+ICP
  geometric alignment, Scaled Chamfer Distance reward implementation
- **Refined variant** — an experimental data variant with alternative field
  structure, run in parallel with the main pipeline for comparison
- **HPC infrastructure** — Gautschi cluster setup, SLURM job scripts, memory
  fragmentation debugging, OOM diagnosis and resolution across multiple failure
  modes (eval batch size, expandable segments, torch.load compatibility)
- **Evaluation and inference** — CR, RR, MSCD, AEC metric implementation,
  Gradio demo, single-file inference script

---

## Attribution to prior work

**STEP-LLM** (Chen et al., 2026) introduced the approach this project
reimplements: using LLMs to generate STEP files via SFT + GRPO with a Scaled
Chamfer Distance reward. The paper's hyperparameters, model selection
(Llama-3.2-3B-Instruct), and training methodology are followed as closely as
possible. The paper's reported results are the benchmark target below.

**Text2CAD** (Khan et al., NeurIPS 2024) provides the dataset: ~170K CAD
models paired with natural language captions. The dataset and one export script
(`data/export_steps.py`, adapted from their repository under Apache 2.0) are
used here. See [ATTRIBUTION.md](ATTRIBUTION.md) for a file-by-file breakdown.

---

## Status

| Phase | Status |
|-------|--------|
| Data pipeline (export, parse, reserialize, pair, split) | Done |
| RAG index build + pre-computation | Done |
| SFT training — main variant (10 epochs, Llama-3.2-3B, 4× H100) | Done |
| SFT training — refined variant (10 epochs, 4× H100) | Done |
| RL training — GRPO, 80 steps | In progress |
| Evaluation | Pending |

---

## Results (target)

The STEP-LLM paper reports the following against the Text2CAD baseline:

| Method | CR (%) | RR (%) | MSCD | AEC |
|--------|--------|--------|------|-----|
| Text2CAD | — | 98.38 | 3.99 | 390.41 |
| STEP-LLM (SFT) | 97.00 | 95.18 | 0.53 | 240.99 |
| STEP-LLM (GRPO) | 99.00 | 92.00 | 0.098 | — |

*These are the paper's reported numbers. Independent reproduction is in progress.*

---

## Setup

### Gautschi HPC (4× H100 80GB — primary)

```bash
bash gautschi_setup.sh
conda activate stepforge
export HUGGINGFACE_TOKEN=your_token_here
sbatch slurm_sft_4gpu_gautschi.sh
sbatch slurm_rl_gautschi.sh
```

### Local / other

```bash
conda env create -f environment.yml
conda activate stepforge
export HUGGINGFACE_TOKEN=your_token_here
```

---

## Running

```bash
# Step 1: Build dataset
python data/build_dataset.py --config configs/config_gautschi.yaml

# Step 2: Build FAISS retrieval index
python retrieval/build_index.py --config configs/config_gautschi.yaml

# Step 3: Pre-compute RAG for SFT
python data/precompute_rag.py --config configs/config_gautschi.yaml

# Step 4: SFT (10 epochs)
sbatch slurm_sft_4gpu_gautschi.sh

# Step 5: RL with GRPO (80 steps)
sbatch slurm_rl_gautschi.sh

# Step 6: Evaluate
python evaluation/evaluate.py --checkpoint checkpoints/rl/final \
    --config configs/config_gautschi.yaml
```

### Quick inference

```bash
python inference/generate.py \
    --caption "a hollow cylinder" \
    --output /tmp/cylinder.step \
    --checkpoint checkpoints/rl/final
```

---

## Project structure

```
StepForge/
├── configs/
│   ├── config_gautschi.yaml         # Main variant — Gautschi H100 cluster
│   └── config_gautschi_refined.yaml # Refined variant — alternative data format
├── data/
│   ├── export_steps.py              # [adapted from Text2CAD] .pth → STEP files
│   ├── step_parser.py               # Parse STEP entity DAG
│   ├── dfs_reserializer.py          # DFS traversal + CoT annotations
│   ├── pair_captions.py             # Pair STEP with abstract captions
│   ├── filter_dataset.py            # Filter + split
│   ├── precompute_rag.py            # Pre-retrieve top-1 STEP per training example
│   └── build_dataset.py             # Orchestrator
├── retrieval/
│   ├── build_index.py               # Build FAISS caption index
│   └── retriever.py                 # Live RAG retrieval
├── training/
│   ├── sft_multigpu.py              # Multi-GPU SFT (DDP, 4× H100)
│   ├── rl_train.py                  # GRPO reinforcement learning
│   └── preflight_check.py           # Environment validation
├── reward/
│   ├── step_to_pointcloud.py        # STEP → 3D point cloud
│   ├── alignment.py                 # FPFH+RANSAC+ICP alignment
│   └── scd_reward.py                # Scaled Chamfer Distance reward
├── inference/generate.py            # Generate STEP from caption
├── evaluation/evaluate.py           # CR, RR, MSCD, AEC metrics
├── app.py                           # Gradio demo
├── ATTRIBUTION.md                   # File-by-file code origin breakdown
└── LICENSE                          # Apache 2.0
```

---

## References

- **STEP-LLM**: Chen et al., 2026. *STEP-LLM: Generating CAD STEP Models from
  Natural Language with Large Language Models.* arXiv:2601.12641.
  [[Paper]](https://arxiv.org/abs/2601.12641) ·
  [[Official code]](https://github.com/JasonShiii/STEP-LLM)

- **Text2CAD**: Khan et al., NeurIPS 2024 Spotlight. *Text2CAD: Generating
  Sequential CAD Designs from Beginner-to-Expert Level Text Prompts.*
  [[Paper]](https://arxiv.org/abs/2409.17106) ·
  [[Code]](https://github.com/SadilKhan/Text2CAD) ·
  [[Project page]](https://sadilkhan.github.io/text2cad-project/)
