"""
Recompute eval metrics from a JSON of generated outputs.

Uses the spawn-pool isolation from reward/scd_reward.py so OCC segfaults
on pathological STEP files don't kill the whole script (the original
evaluate.py / evaluate_in_distribution.py call step_to_pointcloud
directly in the parent process, so one bad STEP crashes everything).

Usage:
    python scripts/recompute_eval_metrics.py \\
        --json $SCRATCH/stepforge/eval_9965567/eval_original.json \\
        --config configs/config_gautschi.yaml \\
        [--label "Eval 1 (no truncation)"]
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

# Use the segfault-isolated worker from scd_reward.py
from reward.scd_reward import _safe_step_to_pointcloud, scaled_chamfer_distance


def completion_rate(outputs):
    return sum("END-ISO-10303-21;" in o for o in outputs) / len(outputs)


def average_entity_count(outputs):
    return float(np.mean([
        len(re.findall(r"^#\d+\s*=", o, re.MULTILINE)) for o in outputs
    ]))


def renderability_rate(outputs, text2cad_src, n_points=2048):
    """Like evaluate.py's renderability_rate but via spawn pool (segfault-safe)."""
    renderable = 0
    for o in tqdm(outputs, desc="RR"):
        pc, _ = _safe_step_to_pointcloud(o, n_points=n_points,
                                          text2cad_src=text2cad_src,
                                          deflection=None)
        if pc is not None:
            renderable += 1
    return renderable / len(outputs)


def mscd(pred_steps, gt_steps, text2cad_src, n_points=2048):
    """Median scaled chamfer distance over parseable (pred, gt) pairs."""
    scds = []
    for pred, gt in tqdm(zip(pred_steps, gt_steps), desc="MSCD",
                         total=len(pred_steps)):
        pred_pc, _ = _safe_step_to_pointcloud(pred, n_points=n_points,
                                              text2cad_src=text2cad_src,
                                              deflection=None)
        gt_pc,   _ = _safe_step_to_pointcloud(gt,   n_points=n_points,
                                              text2cad_src=text2cad_src,
                                              deflection=None)
        if pred_pc is None or gt_pc is None:
            continue
        try:
            s = scaled_chamfer_distance(pred_pc, gt_pc)
            if np.isfinite(s):
                scds.append(s)
        except Exception:
            continue
    if not scds:
        return float("inf"), 0
    return float(np.median(scds)), len(scds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to eval output JSON")
    ap.add_argument("--config", default="configs/config_gautschi.yaml")
    ap.add_argument("--label", default=None, help="Display label")
    ap.add_argument("--in-dist-only", action="store_true",
                    help="If JSON has 'in_dist' flags, also compute metrics on that subset")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    text2cad_src = cfg.paths.text2cad_src
    n_points = int(cfg.rl.reward.n_sample_points)

    data = json.load(open(args.json))
    label = args.label or os.path.basename(args.json)

    def compute_one(records, name):
        gen = [d["generated"] for d in records]
        gt  = [d["gt"]        for d in records]
        cr  = completion_rate(gen)
        aec = average_entity_count(gen)
        rr  = renderability_rate(gen, text2cad_src, n_points=n_points)
        m, n_scd = mscd(gen, gt, text2cad_src, n_points=n_points)
        print(f"\n=== {name}  (N={len(records)}) ===")
        print(f"  CR    = {cr*100:6.2f}%")
        print(f"  RR    = {rr*100:6.2f}%")
        print(f"  MSCD  = {m:.4f}  (over {n_scd} parseable pairs)")
        print(f"  AEC   = {aec:.2f}")

    compute_one(data, f"{label} — all {len(data)} samples")

    if args.in_dist_only:
        in_dist = [d for d in data if d.get("in_dist")]
        if in_dist:
            compute_one(in_dist, f"{label} — in-distribution subset")
        else:
            print("\nNo records flagged in_dist=True in this JSON.")


if __name__ == "__main__":
    main()
