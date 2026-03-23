"""
Split the RAG dataset into train / val / test using the Text2CAD predefined splits.

Using the predefined splits (rather than a random shuffle) ensures our evaluation
results are comparable to the paper, which reports numbers on the same test partition.

Input:  data/processed/rag_dataset.json
Output: data/processed/train.json
        data/processed/val.json
        data/processed/test.json

Usage:
    python data/data_split.py --config configs/config.yaml
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description="Split RAG dataset using Text2CAD splits")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    rag_path   = os.path.join(cfg.paths.processed_dir, "rag_dataset.json")
    split_json = cfg.paths.split_json          # Text2CAD train_test_val.json
    out_dir    = cfg.paths.processed_dir

    # ── Load RAG dataset ──────────────────────────────────────────────────────
    logger.info(f"Loading {rag_path}")
    with open(rag_path) as f:
        data = json.load(f)
    logger.info(f"Total records: {len(data)}")

    # ── Load Text2CAD splits ──────────────────────────────────────────────────
    logger.info(f"Loading splits from {split_json}")
    with open(split_json) as f:
        splits = json.load(f)

    # Build lookup sets — Text2CAD UIDs can be in various formats
    # Normalise to the uid format used in our dataset (e.g. "0000/00000007")
    def norm(uid: str) -> str:
        return uid.replace("_", "/")

    train_uids = {norm(u) for u in splits.get("train", [])}
    val_uids   = {norm(u) for u in splits.get("val",   [])}
    test_uids  = {norm(u) for u in splits.get("test",  [])}

    logger.info(f"Split sizes — train: {len(train_uids)}, val: {len(val_uids)}, test: {len(test_uids)}")

    train, val, test, unmatched = [], [], [], []
    for rec in data:
        uid = norm(rec["id_original"])
        if uid in train_uids:
            train.append(rec)
        elif uid in val_uids:
            val.append(rec)
        elif uid in test_uids:
            test.append(rec)
        else:
            unmatched.append(rec)

    # Fall back: if predefined splits don't cover our UIDs (different version of dataset),
    # do a deterministic 70/20/10 split so the script always produces output.
    if not train and not val and not test:
        logger.warning("No UIDs matched predefined splits — falling back to 70/20/10 random split")
        import random
        random.seed(42)
        random.shuffle(data)
        n = len(data)
        train = data[:int(n * 0.7)]
        test  = data[int(n * 0.7):int(n * 0.9)]
        val   = data[int(n * 0.9):]
        unmatched = []

    logger.info(f"Split result — train: {len(train)}, val: {len(val)}, test: {len(test)}, unmatched: {len(unmatched)}")

    for name, subset in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(out_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(subset, f, indent=2)
        logger.info(f"Saved {len(subset)} records → {path}")


if __name__ == "__main__":
    main()
