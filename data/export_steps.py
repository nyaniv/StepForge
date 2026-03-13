"""
Export STEP files from Text2CAD .pth CAD vector files.

Reuses Text2CAD's CADSequence pipeline:
  CADSequence.from_vec() → create_cad_model() → save_stp()

Each .pth file contains pre-processed quantized CAD tokens that OpenCASCADE
can reconstruct into a B-rep solid.

Usage:
    python data/export_steps.py --config configs/config.yaml
"""

import sys
import os
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from omegaconf import OmegaConf


def _ensure_text2cad_on_path(text2cad_src: str):
    """
    Add both Text2CAD/CadSeqProc/ and Text2CAD/ to sys.path.

    cad_sequence.py itself imports via 'from CadSeqProc.sequence...' (absolute),
    so the *parent* of CadSeqProc must also be on the path.
    """
    parent = os.path.dirname(text2cad_src)  # Text2CAD/
    for p in [text2cad_src, parent]:
        if p not in sys.path:
            sys.path.insert(0, p)


def export_step_from_pth(pth_path: str, output_path: str, text2cad_src: str) -> bool:
    """
    Load a .pth CAD vector file, reconstruct via OpenCASCADE, save as STEP.

    Returns True on success, False if geometry is invalid or an exception occurs.
    (~5-10% failure rate is normal for the dataset.)
    """
    _ensure_text2cad_on_path(text2cad_src)
    from cad_sequence import CADSequence

    try:
        data = torch.load(pth_path, weights_only=False)
        vec = data["vec"]
        # Only cad_vec is the input to from_vec().
        # flag_vec and index_vec are outputs used by the neural net decoder — NOT inputs here.
        cad_seq = CADSequence.from_vec(vec["cad_vec"])
        cad_seq.create_cad_model()
        # save_stp takes filename (stem only, no extension) + output_folder separately
        out_dir = os.path.dirname(output_path)
        out_stem = os.path.splitext(os.path.basename(output_path))[0]
        os.makedirs(out_dir, exist_ok=True)
        cad_seq.save_stp(filename=out_stem, output_folder=out_dir)
        return True
    except Exception as e:
        logger.warning(f"Failed to export {pth_path}: {e}")
        return False


def export_all_steps(cad_seq_dir: str, step_output_dir: str, text2cad_src: str):
    """
    Iterate all .pth files under cad_seq_dir and export to .step files.

    Output filename convention: {root_id}_{chunk_id}.step
    e.g. 0000/00001234/seq/00001234.pth → 0000_00001234.step
    """
    _ensure_text2cad_on_path(text2cad_src)

    Path(step_output_dir).mkdir(parents=True, exist_ok=True)
    pth_files = list(Path(cad_seq_dir).rglob("*.pth"))
    logger.info(f"Found {len(pth_files)} .pth files to export")

    success, fail = 0, 0
    for pth_path in tqdm(pth_files, desc="Exporting STEP"):
        parts = pth_path.parts
        chunk_id = pth_path.stem          # e.g. "00001234"
        root_id = parts[-4]               # e.g. "0000"
        uid = f"{root_id}/{chunk_id}"
        out_path = os.path.join(step_output_dir, f"{uid.replace('/', '_')}.step")

        if os.path.exists(out_path):      # skip already exported
            success += 1
            continue

        if export_step_from_pth(str(pth_path), out_path, text2cad_src):
            success += 1
        else:
            fail += 1

    logger.info(f"Done. Exported {success}, failed {fail} (~5-10% failure rate is normal)")


def main():
    parser = argparse.ArgumentParser(description="Export STEP files from .pth CAD vectors")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    export_all_steps(
        cad_seq_dir=cfg.paths.cad_seq_dir,
        step_output_dir=cfg.paths.step_output_dir,
        text2cad_src=cfg.paths.text2cad_src,
    )


if __name__ == "__main__":
    main()
