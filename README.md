# STEP_LLM — Reimplementation and Extension

An independent reimplementation of [STEP-LLM](https://arxiv.org/abs/2601.12641)
(Chen et al., 2026), developed as part of an undergraduate research project at
Purdue University with Prof. Gomez.

The goal: fine-tune open-source LLMs (starting with Llama-3.2-3B-Instruct) to
generate raw STEP files (ISO 10303) from natural language descriptions of 3D parts,
using supervised fine-tuning and GRPO reinforcement learning with geometric reward
signals.

This repo is a from-scratch implementation — not a fork of the
[official code](https://github.com/JasonShiii/STEP-LLM) — built to support
ongoing extensions including:
- Agentic text → CAD → FEA pipeline orchestration
- RAG-based retrieval of design specifications and reference geometry
- Experimentation with alternative base models and reward functions

## Status

Reimplementation of the core SFT + GRPO pipeline is complete.
Reproduction of published metrics (see below) is in progress.

| Phase | Status |
|-------|--------|
| Data pipeline (export, parse, reserialize, pair, split) | ✅ Done |
| RAG index build + pre-computation | ✅ Done |
| SFT training (10 epochs, Llama-3.2-3B) | ✅ Done — checkpoint saved |
| RL training (GRPO, 80 steps) | ⏳ Pending — queued on Purdue Scholar Cluster |
| Evaluation | ⏳ Pending — runs after RL |

## Comparison to prior work

The STEP-LLM paper reports the following results against the
[Text2CAD](https://github.com/SadilKhan/Text2CAD) baseline
(Khan et al., NeurIPS 2024):

| Method | CR (%) | RR (%) | MSCD | AEC |
|---|---|---|---|---|
| Text2CAD | — | 98.38 | 3.99 | 390.41 |
| STEP-LLM (SFT) | 97.00 | 95.18 | 0.53 | 240.99 |
| STEP-LLM (GRPO) | 99.00 | 92.00 | 0.098 | — |

*These are the paper's reported numbers, not yet independently reproduced here.*

---

## What came from Text2CAD vs what's new

### From Text2CAD
**Text2CAD** ([SadilKhan/Text2CAD](https://github.com/SadilKhan/Text2CAD), NeurIPS 2024 Spotlight)
provided two things we built on:

- **Dataset**: `cad_seq.zip` — ~170K quantized CAD vector files (`.pth`) and caption annotations
  (`captions.csv`). We used their train/val/test splits.
- **CAD export code** (`data/export_steps.py`): Wraps Text2CAD's `CADSequence` pipeline
  (`from_vec → create_cad_model → save_stp`) to convert their `.pth` vectors into STEP geometry
  files. This is the only file adapted from their codebase.

### New code (written for this project)

Everything else was written from scratch to implement the STEP-LLM paper:

**Data processing**
- `data/step_parser.py` — Parses flat STEP entity lists into a reference DAG
- `data/dfs_reserializer.py` — DFS traversal that linearizes cross-references, renumbers
  entities sequentially, normalizes floats, and adds CoT branch annotations (paper §3.1)
- `data/pair_captions.py` — Pairs exported STEP files with Text2CAD's abstract captions
- `data/filter_dataset.py` — Filters by entity count (≤500) and applies DFS reserialization
  before writing train/val/test splits
- `data/precompute_rag.py` — Pre-computes top-1 FAISS retrieval for all training examples
- `data/build_dataset.py` — Orchestrates the full pipeline in order

**Retrieval**
- `retrieval/build_index.py` — Encodes captions with SentenceTransformer → FAISS IndexFlatIP
- `retrieval/retriever.py` — Live RAG retrieval with self-exclusion (top-20 search, skips self)

**Reward**
- `reward/step_to_pointcloud.py` — STEP string → sampled 3D point cloud (via pythonOCC + BRepMesh)
- `reward/alignment.py` — Multi-stage alignment: center → FPFH+RANSAC → ICP (paper §3.3)
- `reward/scd_reward.py` — Scaled Chamfer Distance reward function (paper Eq. 1–3)

**Training**
- `training/sft_train.py` — SFT with Unsloth + LoRA; loss masked to STEP tokens only
- `training/rl_train.py` — GRPO trainer (TRL); cold-starts from SFT checkpoint; live RAG

**Evaluation & inference**
- `evaluation/evaluate.py` — Computes CR, RR, MSCD, AEC on test set
- `inference/generate.py` — Live RAG → prompt → generate → extract STEP
- `app.py` — Gradio demo: text input → STEP generation → STL preview + download

---

## Setup

```bash
conda env create -f environment.yml
conda activate step_llm
export HUGGINGFACE_TOKEN=your_token_here   # required for Llama gated model
```

---

## Project structure

```
StepLLM/
├── configs/
│   ├── config.yaml              # All hyperparameters and local paths
│   └── config_scholar.yaml      # Purdue Scholar Cluster paths (template)
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
└── slurm_sft.sh / slurm_rl.sh  # Purdue Scholar SLURM scripts
```

---

## References

- **STEP-LLM**: Chen et al., 2026. *STEP-LLM: Generating CAD STEP Models from Natural Language with Large Language Models.* arXiv:2601.12641. [[Paper]](https://arxiv.org/abs/2601.12641) · [[Official code]](https://github.com/JasonShiii/STEP-LLM)

- **Text2CAD**: Khan et al., NeurIPS 2024 Spotlight. *Text2CAD: Generating Sequential CAD Designs from Beginner-to-Expert Level Text Prompts.* [[Paper]](https://arxiv.org/abs/2409.17106) · [[Code]](https://github.com/SadilKhan/Text2CAD) · [[Project page]](https://sadilkhan.github.io/text2cad-project/)
