"""
Batch-restructure all raw STEP files using one of two CoT annotation formats.

  --cot-format semantic   (default) — step_restructurer.py
                          Semantic section markers like /* BEGIN STYLED_ITEM 1 */
                          KNOWN_DEVIATIONS.md #2: chosen over paper format based on
                          early SFT experiments (undocumented).

  --cot-format paper      — dfs_reserializer.py
                          Paper-faithful /* [BRANCH] depth=D children=C */ from §3.1.
                          Use this for paper-fidelity ablations.

Adapted from the original batch_restructure.sh to handle our flat directory
structure: data/step_files/*.step  →  data/dfs_step/*.step

Usage:
    python data/batch_restructure.py --config configs/config.yaml
    python data/batch_restructure.py --config configs/config.yaml --workers 8
    python data/batch_restructure.py --config configs/config.yaml --cot-format paper
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
from data.step_parser import parse_step
from data.dfs_reserializer import reserialize


def _restructure_semantic(input_path: str, output_path: str) -> None:
    StepRestructurer().restructure_step_file(input_path, output_path)


def _restructure_paper(input_path: str, output_path: str) -> None:
    header, entities, referenced_by = parse_step(input_path)
    out = reserialize(header, entities, referenced_by)
    with open(output_path, "w") as f:
        f.write(out)


_FORMATS = {
    "semantic": _restructure_semantic,
    "paper":    _restructure_paper,
}


def process_one(args):
    """Process a single STEP file. Returns (input_path, output_path, success, error)."""
    input_path, output_path, cot_format = args
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _FORMATS[cot_format](str(input_path), str(tmp_path))
        os.replace(tmp_path, output_path)  # atomic on POSIX
        return str(input_path), str(output_path), True, None
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        return str(input_path), str(output_path), False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Batch DFS-restructure STEP files")
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers (default 8)")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Process only first N files (for testing)")
    parser.add_argument("--cot-format", choices=("semantic", "paper"), default="semantic",
                        help="CoT annotation format. 'semantic' = step_restructurer.py "
                             "(default, KNOWN_DEVIATIONS #2). 'paper' = dfs_reserializer.py "
                             "(paper §3.1 branch statistics, for fidelity ablations).")
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
    logger.info(f"CoT annotation format: {args.cot_format} "
                f"({'step_restructurer.py' if args.cot_format == 'semantic' else 'dfs_reserializer.py'})")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # D1: skip-if-exists below ignores --cot-format, so re-running with a
    # different format would silently mix annotation styles in dest_dir.
    # Record the format on first run and refuse on mismatch.
    marker = dest_dir / ".cot_format"
    if marker.exists():
        existing = marker.read_text().strip()
        if existing != args.cot_format:
            logger.error(
                f"dfs_step/ was built with --cot-format={existing!r} but you asked "
                f"for {args.cot_format!r}. Delete {dest_dir} (or use a fresh "
                f"output dir) before switching formats."
            )
            sys.exit(1)
    else:
        marker.write_text(args.cot_format)

    # Build (input, output) pairs — output keeps the same filename
    tasks = [
        (p, dest_dir / p.name, args.cot_format)
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
