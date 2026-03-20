"""
Pre-compute RAG retrievals for all training examples.

Enriches each training record with the top-1 retrieved STEP file so no live
retrieval is needed during SFT training (avoids GPU → CPU bottleneck).

Output: train_with_rag.jsonl — same as train.jsonl but with two extra fields:
  "retrieved_step":    str  — the STEP content of the retrieved example
  "retrieved_caption": str  — its caption (for debugging)

Usage:
    python data/precompute_rag.py --config configs/config.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

from retrieval.retriever import Retriever


def precompute_rag(train_jsonl: str, retriever: Retriever, output_jsonl: str):
    """
    Enrich every training record with its pre-retrieved STEP.
    Uses exclude_uid to prevent self-retrieval.
    """
    with open(train_jsonl) as f:
        records = [json.loads(l) for l in f]
    logger.info(f"Pre-computing RAG for {len(records)} training examples...")

    for record in tqdm(records, desc="Pre-computing RAG"):
        retrieved = retriever.retrieve(
            record["caption"],
            exclude_uid=record["uid"],
        )
        record["retrieved_step"]    = retrieved["step"]
        record["retrieved_caption"] = retrieved["caption"]

    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    logger.info(f"Saved {len(records)} enriched records to {output_jsonl}")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute RAG for training data")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    retriever = Retriever(
        index_path=cfg.paths.faiss_index_path,
        metadata_path=cfg.paths.faiss_metadata_path,
        model_name=cfg.retrieval.model,
    )

    train_jsonl = os.path.join(cfg.paths.processed_dir, "train.jsonl")
    output_jsonl = os.path.join(cfg.paths.processed_dir, "train_with_rag.jsonl")

    precompute_rag(train_jsonl, retriever, output_jsonl)


if __name__ == "__main__":
    main()
