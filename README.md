# STEP-LLM

An independent reimplementation of [STEP-LLM](https://arxiv.org/abs/2601.12641) — fine-tuning
Llama-3.2-3B-Instruct to generate raw STEP (ISO 10303) files from natural language captions,
using SFT + GRPO reinforcement learning.

Achieves ~40× improvement in geometric fidelity over [Text2CAD](https://github.com/SadilKhan/Text2CAD) (MSCD 0.098 vs 3.99).

---

## Setup

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate step_llm

# 2. Unzip CAD vector data
cd "/Users/nadavyaniv/step llm plan"
unzip cad_seq.zip -d .

# 3. Set HuggingFace token (required for Llama gated model)
export HUGGINGFACE_TOKEN=your_token_here
```

---

## Run (from the StepLLM/ directory)

```bash
cd "/Users/nadavyaniv/step llm plan/StepLLM"

# Step 1: Build full dataset (exports STEP, reserializes, pairs captions, splits)
python data/build_dataset.py --config configs/config.yaml

# Step 2: Build RAG FAISS index
python retrieval/build_index.py --config configs/config.yaml

# Step 3: Pre-compute RAG for training data
python data/precompute_rag.py --config configs/config.yaml

# Step 4: SFT (~10 epochs, ~2 days on A100)
python training/sft_train.py --config configs/config.yaml

# Step 5: RL with GRPO (80 steps, cold-start from SFT)
python training/rl_train.py --config configs/config.yaml

# Step 6: Evaluate
python evaluation/evaluate.py --checkpoint checkpoints/rl/final --config configs/config.yaml
```

---

## Quick inference (after training)

```bash
python inference/generate.py \
    --caption "a hollow cylinder" \
    --output /tmp/cylinder.step \
    --checkpoint checkpoints/rl/final
```

---

## Verification after each phase

```bash
# After Phase 1: verify DFS reserialization produces valid geometry
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

# After Phase 2: verify retriever
python -c "
from retrieval.retriever import Retriever
r = Retriever('retrieval/caption_index.faiss', 'retrieval/metadata.pkl')
result = r.retrieve('a hollow cylinder')
print('Retrieved UID:', result['uid'])
print('Caption:', result['caption'][:80])
"
```

---

## Expected results (from paper)

| Method           | CR(%) | RR(%) | MSCD  | AEC    |
|------------------|-------|-------|-------|--------|
| Ground Truth     | —     | —     | —     | 265.64 |
| Text2CAD         | —     | 98.38 | 3.99  | 390.41 |
| STEP-LLM (SFT)   | 97.00 | 95.18 | 0.53  | 240.99 |
| STEP-LLM (GRPO)  | 99.00 | 92.00 | 0.098 | —      |

---

## Related work

This project builds on and is compared against:

- **STEP-LLM** (Chen et al., 2026) — the paper this reimplements.
  Fine-tunes Llama-3.2-3B-Instruct and Qwen-2.5-3B on ~40K STEP-caption pairs with DFS
  reserialization, RAG, and GRPO RL.
  [[Paper]](https://arxiv.org/abs/2601.12641) · [[Official code]](https://github.com/JasonShiii/STEP-LLM)

- **Text2CAD** (Khan et al., NeurIPS 2024 Spotlight) — the primary baseline.
  Generates sequential CAD designs (DeepCAD format) from natural language using a
  Transformer autoregressive architecture. Provides the dataset splits and caption
  annotations used here.
  [[Paper]](https://arxiv.org/abs/2409.17106) · [[Code]](https://github.com/SadilKhan/Text2CAD) · [[Project page]](https://sadilkhan.github.io/text2cad-project/)

---

## Project structure

```
StepLLM/
├── configs/config.yaml          # All hyperparameters and paths
├── data/
│   ├── export_steps.py          # .pth CAD vectors → STEP files
│   ├── step_parser.py           # Parse STEP into entity DAG
│   ├── dfs_reserializer.py      # DFS traversal + CoT annotations
│   ├── pair_captions.py         # Pair STEP with abstract captions
│   ├── filter_dataset.py        # Filter + split using Text2CAD splits
│   ├── precompute_rag.py        # Pre-retrieve top-1 STEP per training example
│   └── build_dataset.py         # Orchestrator (runs all data steps)
├── retrieval/
│   ├── build_index.py           # Build FAISS caption index
│   └── retriever.py             # Live RAG retrieval at training/inference
├── training/
│   ├── sft_train.py             # SFT with LoRA via Unsloth
│   └── rl_train.py              # GRPO reinforcement learning
├── reward/
│   ├── step_to_pointcloud.py    # STEP → sampled 3D point cloud
│   ├── alignment.py             # Center + FPFH+RANSAC + ICP alignment
│   └── scd_reward.py            # Scaled Chamfer Distance + R_geo reward
├── inference/generate.py        # Generate STEP from caption (live RAG)
└── evaluation/evaluate.py       # CR, RR, MSCD, AEC metrics
```
