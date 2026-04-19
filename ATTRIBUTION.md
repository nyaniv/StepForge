# Attribution

This project builds on two external sources. Everything not listed here was written
from scratch for StepForge.

---

## From Text2CAD (Apache 2.0)

**Repository**: [SadilKhan/Text2CAD](https://github.com/SadilKhan/Text2CAD)
**License**: Apache 2.0
**Paper**: Khan et al., NeurIPS 2024 Spotlight.

### Dataset
`cad_seq.zip` (~170K quantized CAD vector `.pth` files) and `captions.csv` (abstract
text annotations) are the Text2CAD dataset. We use their train/val/test splits
without modification.

### Adapted code
`data/export_steps.py` — wraps Text2CAD's `CADSequence` pipeline
(`from_vec → create_cad_model → save_stp`) to convert their `.pth` vectors into
raw STEP geometry files via pythonOCC. This is the only source file adapted from
their codebase.

---

## From Unsloth (Apache 2.0)

**Repository**: [unslothai/unsloth](https://github.com/unslothai/unsloth)
**License**: Apache 2.0

`training/llama3_SFT_response.py` — the original single-GPU SFT script — was
structured around the Unsloth SFT fine-tuning template. The training loop
structure, LoRA configuration pattern, and chat-template formatting approach
are derived from their template. This script is archived; the current multi-GPU
training (`training/sft_multigpu.py`) was rewritten from scratch using standard
HuggingFace Trainer + PEFT without Unsloth.

---

## From STEP-LLM paper (Chen et al., 2026)

**Paper**: arXiv:2601.12641
**Official code**: [JasonShiii/STEP-LLM](https://github.com/JasonShiii/STEP-LLM)

This repo is a from-scratch reimplementation of the methods described in that paper.
No code was copied from the official implementation. The following files implement
paper-specific algorithms:

- `data/dfs_reserializer.py` — DFS reserialization (§3.1)
- `reward/alignment.py` — point cloud alignment pipeline (§3.3)
- `reward/scd_reward.py` — Scaled Chamfer Distance reward, Equations 1–3

---

## Original code (written for this project)

All remaining files are original:

**Data processing**
- `data/step_parser.py` — parses flat STEP entity lists into a reference DAG
- `data/pair_captions.py` — pairs exported STEP files with Text2CAD abstract captions
- `data/filter_dataset.py` — entity-count filtering and train/val/test splitting
- `data/precompute_rag.py` — pre-computes top-1 FAISS retrieval for all training examples
- `data/build_dataset.py` — orchestrates the full data pipeline

**Retrieval**
- `retrieval/build_index.py` — SentenceTransformer embeddings → FAISS IndexFlatIP
- `retrieval/retriever.py` — live RAG retrieval with self-exclusion

**Reward**
- `reward/step_to_pointcloud.py` — STEP string → sampled 3D point cloud (pythonOCC + BRepMesh)

**Training**
- `training/sft_train.py` — SFT with Unsloth + LoRA; loss masked to STEP tokens only
- `training/rl_train.py` — GRPO trainer (TRL); cold-starts from SFT checkpoint; live RAG

**Evaluation & inference**
- `evaluation/evaluate.py` — computes CR, RR, MSCD, AEC on test set
- `inference/generate.py` — live RAG → prompt → generate → extract STEP
- `app.py` — Gradio demo

**Configuration & cluster**
- `configs/config.yaml`
- `configs/config_scholar.yaml`
- `slurm_sft.sh`, `slurm_rl.sh`
- `scholar_setup.sh`
