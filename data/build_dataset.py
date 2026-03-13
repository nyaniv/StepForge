"""
Orchestrator: run the full dataset construction pipeline.

Steps (in order):
  1. Export STEP files from .pth CAD vectors (export_steps.py)
  2. Pair STEP files with abstract captions (pair_captions.py)
  3. DFS-reserialize, filter, and split (filter_dataset.py)

After this script completes, run:
  python retrieval/build_index.py --config configs/config.yaml
  python data/precompute_rag.py   --config configs/config.yaml

Usage:
    python data/build_dataset.py --config configs/config.yaml
    python data/build_dataset.py --config configs/config.yaml --skip-export  # if STEP files already exist
"""

import argparse
import os
import sys

# Ensure project root is on sys.path regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from omegaconf import OmegaConf

from data.export_steps import export_all_steps
from data.pair_captions import pair_captions
from data.filter_dataset import filter_and_split


def main():
    parser = argparse.ArgumentParser(description="Build full STEP-LLM dataset")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip STEP export if files already exist",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # ── Phase 1: Export STEP from .pth vectors ─────────────────────────────
    if not args.skip_export:
        logger.info("=== Phase 1: Exporting STEP files from .pth vectors ===")
        export_all_steps(
            cad_seq_dir=cfg.paths.cad_seq_dir,
            step_output_dir=cfg.paths.step_output_dir,
            text2cad_src=cfg.paths.text2cad_src,
        )
    else:
        logger.info("Skipping STEP export (--skip-export flag set)")

    # ── Phase 2: Pair captions ──────────────────────────────────────────────
    logger.info("=== Phase 2: Pairing captions with STEP files ===")
    all_jsonl = os.path.join(cfg.paths.processed_dir, "all_pairs.jsonl")
    pair_captions(
        caption_csv=cfg.paths.caption_csv,
        step_output_dir=cfg.paths.step_output_dir,
        output_jsonl=all_jsonl,
    )

    # ── Phase 3: Reserialize, filter, split ────────────────────────────────
    logger.info("=== Phase 3: DFS reserialization, filtering, and splitting ===")
    filter_and_split(
        all_jsonl=all_jsonl,
        split_json=cfg.paths.split_json,
        output_dir=cfg.paths.processed_dir,
        max_entities=cfg.data.max_entities,
    )

    logger.info("=== Dataset construction complete ===")
    logger.info(f"Output directory: {cfg.paths.processed_dir}")
    logger.info("Next steps:")
    logger.info("  python retrieval/build_index.py --config configs/config.yaml")
    logger.info("  python data/precompute_rag.py   --config configs/config.yaml")


if __name__ == "__main__":
    main()
