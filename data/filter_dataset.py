"""
Filter paired STEP-caption examples and split into train/val/test sets.

Reuses the existing Text2CAD train/validation/test split JSON so results are
directly comparable to the paper (same data partitioning).

Filter: exclude files with >= 500 entities (too long for 8192-token context window).
The plan DFS-reserializes STEP files before writing to the split files.

Output: three JSONL files in processed_dir/
  train.jsonl, val.jsonl, test.jsonl

NOTE: This script is NOT part of the current training pipeline. It produces JSONL
files which are incompatible with rl_train.py, evaluate.py, and diagnose_sft.py
(all of which expect JSON arrays from data/data_split.py). Use this script for
exploratory data work only. For the full training pipeline, run:
  data/dataset_construct_rag.py → data/data_split.py

Usage:
    python data/filter_dataset.py --config configs/config.yaml
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from omegaconf import OmegaConf

from data.step_parser import parse_step_from_string
from data.dfs_reserializer import reserialize


def count_entities(step_content: str) -> int:
    """Count STEP entities (lines matching '#N = ...')."""
    return len(re.findall(r"^#\d+\s*=", step_content, re.MULTILINE))


def reserialize_step(step_content: str, uid: str) -> str | None:
    """
    DFS-reserialize a STEP string.  Returns None if parsing fails.
    """
    try:
        header, entities, referenced_by = parse_step_from_string(step_content)
        if not entities:
            return None
        return reserialize(header, entities, referenced_by)
    except Exception as e:
        logger.warning(f"Reserialization failed for {uid}: {e}")
        return None


def filter_and_split(
    all_jsonl: str,
    split_json: str,
    output_dir: str,
    max_entities: int = 500,
):
    """
    Read all_jsonl, DFS-reserialize each STEP, filter by entity count, split.

    DFS reserialization is applied here so training data is already in the
    locality-preserving format the LLM will be trained on.
    """
    with open(split_json) as f:
        split = json.load(f)
    train_uids = set(split["train"])
    val_uids   = set(split["validation"])
    test_uids  = set(split["test"])

    splits = {"train": [], "val": [], "test": []}
    skipped_size = 0
    skipped_reser = 0
    unmatched = 0

    with open(all_jsonl) as f:
        lines = f.readlines()

    logger.info(f"Processing {len(lines)} examples...")

    for line in lines:
        record = json.loads(line)
        uid = record["uid"]

        # DFS reserialize
        reser = reserialize_step(record["step"], uid)
        if reser is None:
            skipped_reser += 1
            continue

        n_entities = count_entities(reser)
        if n_entities >= max_entities:
            skipped_size += 1
            continue

        record["step"] = reser
        record["entity_count"] = n_entities

        if uid in train_uids:
            splits["train"].append(record)
        elif uid in val_uids:
            splits["val"].append(record)
        elif uid in test_uids:
            splits["test"].append(record)
        else:
            unmatched += 1

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for split_name, records in splits.items():
        path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        logger.info(f"{split_name}: {len(records)} examples → {path}")

    logger.info(f"Filtered: {skipped_size} too large (>={max_entities} entities), "
                f"{skipped_reser} reserialization failures, {unmatched} unmatched UIDs")


def main():
    parser = argparse.ArgumentParser(description="Filter and split dataset")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    all_jsonl = os.path.join(cfg.paths.processed_dir, "all_pairs.jsonl")
    filter_and_split(
        all_jsonl=all_jsonl,
        split_json=cfg.paths.split_json,
        output_dir=cfg.paths.processed_dir,
        max_entities=cfg.data.max_entities,
    )


if __name__ == "__main__":
    main()
