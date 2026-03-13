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
    logger.info(f"Loading training data from {train_jsonl}")
    records = [json.loads(l) for l in open(train_jsonl)]
    captions = [r["caption"] for r in records]
    logger.info(f"Embedding {len(captions)} captions with {model_name}")

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
    pickle.dump(records, open(metadata_path, "wb"))

    logger.info(f"Index built: {len(records)} entries")
    logger.info(f"  FAISS index → {index_path}")
    logger.info(f"  Metadata    → {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS caption index for RAG")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    train_jsonl = f"{cfg.paths.processed_dir}/train.jsonl"

    build_index(
        train_jsonl=train_jsonl,
        index_path=cfg.paths.faiss_index_path,
        metadata_path=cfg.paths.faiss_metadata_path,
        model_name=cfg.retrieval.model,
    )


if __name__ == "__main__":
    main()
