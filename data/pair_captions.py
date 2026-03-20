"""
Pair exported STEP files with their abstract (L0) captions from the Text2CAD CSV.

CSV columns: uid, abstract, beginner, intermediate, expert
  - uid format: "root_id/chunk_id"  e.g. "0000/00001234"
  - abstract = L0 single-sentence description (closest to STEP-LLM paper's GPT-4o captions)

STEP files are named: {root_id}_{chunk_id}.step  e.g. "0000_00001234.step"

Output: JSONL file with one record per example:
  {"uid": str, "caption": str, "step": str}

Usage:
    python data/pair_captions.py --config configs/config.yaml
"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf


def pair_captions(caption_csv: str, step_output_dir: str, output_jsonl: str):
    """
    Read the captions CSV, find matching STEP files, write paired JSONL.
    Only the 'abstract' (L0) caption is used — matches paper's single-sentence style.
    """
    df = pd.read_csv(caption_csv)
    logger.info(f"Loaded {len(df)} rows from {caption_csv}")

    # uid "0000/00001234" → filename "0000_00001234.step"
    df["step_filename"] = df["uid"].str.replace("/", "_") + ".step"
    df["step_path"] = df["step_filename"].apply(
        lambda f: os.path.join(step_output_dir, f)
    )

    records = []
    missing_step = 0
    missing_caption = 0

    for _, row in df.iterrows():
        if not os.path.exists(row["step_path"]):
            missing_step += 1
            continue
        if pd.isna(row["abstract"]) or str(row["abstract"]).strip() == "":
            missing_caption += 1
            continue
        with open(row["step_path"], errors="replace") as fh:
            step_content = fh.read()
        records.append({
            "uid": row["uid"],
            "caption": str(row["abstract"]).strip(),
            "step": step_content,
        })

    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    logger.info(f"Paired {len(records)} caption-STEP examples")
    logger.info(f"Skipped: {missing_step} missing STEP files, {missing_caption} empty captions")
    return len(records)


def main():
    parser = argparse.ArgumentParser(description="Pair captions with STEP files")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    output_jsonl = os.path.join(cfg.paths.processed_dir, "all_pairs.jsonl")
    pair_captions(
        caption_csv=cfg.paths.caption_csv,
        step_output_dir=cfg.paths.step_output_dir,
        output_jsonl=output_jsonl,
    )


if __name__ == "__main__":
    main()
