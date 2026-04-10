"""
Build a FAISS index over training captions for RAG retrieval.

From paper Section 3.2: captions are embedded with SentenceTransformer
(all-MiniLM-L6-v2) and indexed with FAISS IndexFlatIP (cosine similarity
on L2-normalized vectors).

Output:
  - caption_index.faiss  — FAISS index
  - metadata.pkl         — list of {"uid", "caption", "step"} dicts

Usage:
    python retrieval/build_index.py --config configs/config.yaml
"""

import argparse
import json
import os
import sys
import pickle
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
import numpy as np
from loguru import logger
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def build_index(train_jsonl: str, index_path: str, metadata_path: str,
                model_name: str = "all-MiniLM-L6-v2"):
    """
    Embed training captions and write FAISS index + metadata pickle.

    Uses IndexFlatIP on L2-normalized embeddings = exact cosine similarity search.
    """
    # M2: Active pipeline (dataset_construct_rag.py → data_split.py) writes
    # train.json (a JSON array). Only the inactive filter_dataset.py writes
    # JSONL. Support both so this script doesn't FileNotFoundError.
    logger.info(f"Loading training data from {train_jsonl}")
    with open(train_jsonl) as f:
        head = f.read(1); f.seek(0)
        if head == "[":
            records = json.load(f)
        else:
            records = [json.loads(l) for l in f if l.strip()]
    captions = [r["caption"] for r in records]
    logger.info(f"Embedding {len(captions)} captions with {model_name}")

    # S3: cuBLAS determinism for bitwise-reproducible embeddings across runs.
    # Without this, near-tie captions can flip top-1 retrieval between runs.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        captions,
        batch_size=256,
        normalize_embeddings=True,   # L2-normalize → cosine sim via dot product
        show_progress_bar=True,
    ).astype(np.float32)

    # IndexFlatIP on normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as fh:
        pickle.dump(records, fh)

    logger.info(f"Index built: {len(records)} entries")
    logger.info(f"  FAISS index → {index_path}")
    logger.info(f"  Metadata    → {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS caption index for RAG")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing index. Without this, refuses if "
                             "an index already exists (D2: dataset_construct_rag.py "
                             "writes a FILTERED index to the same path).")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # D2: dataset_construct_rag.py writes a filtered train-only index to the
    # same cfg.paths.faiss_index_path. Running this script afterwards would
    # silently overwrite it with an UNFILTERED index. Refuse without --force.
    if os.path.exists(cfg.paths.faiss_index_path) and not args.force:
        sys.exit(
            f"Index already exists at {cfg.paths.faiss_index_path}.\n"
            f"This was likely written by dataset_construct_rag.py (the active pipeline).\n"
            f"Rerun with --force to overwrite, or skip this script."
        )

    # M2: prefer active-pipeline output (train.json), fall back to JSONL.
    train_jsonl = os.path.join(cfg.paths.processed_dir, "train.json")
    if not os.path.exists(train_jsonl):
        train_jsonl = os.path.join(cfg.paths.processed_dir, "train.jsonl")

    build_index(
        train_jsonl=train_jsonl,
        index_path=cfg.paths.faiss_index_path,
        metadata_path=cfg.paths.faiss_metadata_path,
        model_name=cfg.retrieval.model,
    )


if __name__ == "__main__":
    main()
