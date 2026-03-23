"""
Batch-restructure all raw STEP files using step_restructurer.py.

Adapted from the original batch_restructure.sh to handle our flat directory
structure: data/step_files/*.step  →  data/dfs_step/*.step

Usage:
    python data/batch_restructure.py --config configs/config.yaml
    python data/batch_restructure.py --config configs/config.yaml --workers 8
    python data/batch_restructure.py --config configs/config.yaml --limit 100  # quick test
"""

import argparse
import os
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from omegaconf import OmegaConf

from data.step_restructurer import StepRestructurer


def process_one(args):
    """Process a single STEP file. Returns (input_path, output_path, success, error)."""
    input_path, output_path = args
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        restructurer = StepRestructurer()
        restructurer.restructure_step_file(str(input_path), str(output_path))
        return str(input_path), str(output_path), True, None
    except Exception as e:
        return str(input_path), str(output_path), False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Batch DFS-restructure STEP files")
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers (default 8)")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Process only first N files (for testing)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    src_dir  = Path(cfg.paths.step_output_dir)      # data/step_files/
    dest_dir = Path(cfg.paths.step_output_dir).parent / "dfs_step"

    step_files = sorted(src_dir.glob("*.step"))
    if not step_files:
        logger.error(f"No .step files found in {src_dir}")
        sys.exit(1)

    if args.limit:
        step_files = step_files[:args.limit]

    logger.info(f"Found {len(step_files)} STEP files in {src_dir}")
    logger.info(f"Output directory: {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Build (input, output) pairs — output keeps the same filename
    tasks = [
        (p, dest_dir / p.name)
        for p in step_files
        if not (dest_dir / p.name).exists()  # skip already processed
    ]

    already_done = len(step_files) - len(tasks)
    if already_done:
        logger.info(f"Skipping {already_done} already-processed files")
    logger.info(f"Processing {len(tasks)} files with {args.workers} workers...")

    success = 0
    failure = 0

    with Pool(args.workers) as pool:
        for i, (inp, out, ok, err) in enumerate(pool.imap_unordered(process_one, tasks), 1):
            if ok:
                success += 1
            else:
                failure += 1
                logger.warning(f"Failed: {inp} — {err}")
            if i % 1000 == 0:
                logger.info(f"  Progress: {i}/{len(tasks)} (ok={success}, fail={failure})")

    logger.info(f"Done. Success: {success}, Failed: {failure}")
    logger.info(f"Restructured files saved to {dest_dir}")


if __name__ == "__main__":
    main()
