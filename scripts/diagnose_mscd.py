"""
Diagnose where MSCD comes from on the 30-sample test eval.

Reads two eval JSONs (same 30 test examples, different models) and computes
per-example scaled chamfer distance for both. Reports:

  - Side-by-side per-example SCD for SFT vs RL on identical inputs
  - Distribution stats (mean, median, p25, p75, max, # NaN)
  - Where RL improved vs hurt vs no-change
  - Whether the median is being dragged by a few outliers

CPU-only (OCC tessellation + numpy), runs on login node.

Usage:
    python scripts/diagnose_mscd.py \\
        --sft  $SCRATCH/stepforge/refined_rl_eval_30.json \\
        --rl   $SCRATCH/stepforge/rl_v2_eval_30.json \\
        --config configs/config_gautschi.yaml
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

from reward.step_to_pointcloud import step_to_pointcloud
from reward.scd_reward import scaled_chamfer_distance


def per_example_scd(generated, gt, text2cad_src, n_points):
    """Return list of (scd, status) — scd is None if either side failed to parse."""
    out = []
    for i, (pred, ground) in enumerate(zip(generated, gt)):
        pred_pc = step_to_pointcloud(pred,  n_points=n_points, text2cad_src=text2cad_src)
        gt_pc   = step_to_pointcloud(ground, n_points=n_points, text2cad_src=text2cad_src)
        if pred_pc is None:
            out.append((None, "pred_parse_fail"))
            continue
        if gt_pc is None:
            out.append((None, "gt_parse_fail"))
            continue
        try:
            s = scaled_chamfer_distance(pred_pc, gt_pc)
            if not np.isfinite(s):
                out.append((None, "nan_scd"))
            else:
                out.append((float(s), "ok"))
        except Exception as e:
            out.append((None, f"err:{type(e).__name__}"))
    return out


def summarize(scds, label):
    vals = [s for s, _ in scds if s is not None]
    print(f"\n  {label}:")
    if not vals:
        print(f"    no valid SCDs")
        return
    arr = np.array(vals)
    statuses = [st for _, st in scds]
    n_ok = statuses.count("ok")
    print(f"    N total:       {len(scds)}")
    print(f"    N parseable:   {n_ok}")
    print(f"    N failed:      {len(scds) - n_ok}")
    failures = [s for s in statuses if s != "ok"]
    if failures:
        from collections import Counter
        for status, count in Counter(failures).most_common():
            print(f"      {status:<24} {count}")
    print(f"    median (MSCD): {np.median(arr):.4f}")
    print(f"    mean:          {arr.mean():.4f}")
    print(f"    min:           {arr.min():.4f}")
    print(f"    p25:           {np.percentile(arr, 25):.4f}")
    print(f"    p75:           {np.percentile(arr, 75):.4f}")
    print(f"    p90:           {np.percentile(arr, 90):.4f}")
    print(f"    max:           {arr.max():.4f}")


def compare(sft_scds, rl_scds):
    n = min(len(sft_scds), len(rl_scds))
    print(f"\n  Per-example comparison (first {n}):")
    print(f"    {'idx':<5} {'SFT SCD':>10} {'RL SCD':>10} {'Δ':>10} {'verdict':<15}")
    print(f"    " + "-"*55)
    improved, hurt, samestat, both_fail = 0, 0, 0, 0
    for i in range(n):
        s_sft, st_sft = sft_scds[i]
        s_rl,  st_rl  = rl_scds[i]
        if s_sft is None and s_rl is None:
            verdict = "both failed"
            both_fail += 1
            s_sft_s, s_rl_s, d_s = f"({st_sft})", f"({st_rl})", "—"
        elif s_sft is None:
            verdict = "RL only"
            s_sft_s = f"({st_sft})"
            s_rl_s  = f"{s_rl:.4f}"
            d_s     = "—"
        elif s_rl is None:
            verdict = "SFT only"
            s_sft_s = f"{s_sft:.4f}"
            s_rl_s  = f"({st_rl})"
            d_s     = "—"
        else:
            d = s_rl - s_sft
            if abs(d) < 1e-4:
                verdict = "same"
                samestat += 1
            elif d < 0:
                verdict = "RL better"
                improved += 1
            else:
                verdict = "RL worse"
                hurt += 1
            s_sft_s = f"{s_sft:.4f}"
            s_rl_s  = f"{s_rl:.4f}"
            d_s     = f"{d:+.4f}"
        print(f"    {i:<5} {s_sft_s:>10} {s_rl_s:>10} {d_s:>10} {verdict:<15}")

    print(f"\n  Summary across {n} examples:")
    print(f"    RL better than SFT:  {improved:>3}")
    print(f"    RL worse than SFT:   {hurt:>3}")
    print(f"    Same (Δ<1e-4):       {samestat:>3}")
    print(f"    Both failed parse:   {both_fail:>3}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True, help="Path to SFT eval JSON")
    ap.add_argument("--rl",  required=True, help="Path to RL eval JSON")
    ap.add_argument("--config", default="configs/config_gautschi.yaml")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    text2cad_src = cfg.paths.text2cad_src
    n_points = int(cfg.rl.reward.n_sample_points)

    print(f"Loading SFT eval:  {args.sft}")
    sft_data = json.load(open(args.sft))
    print(f"Loading RL eval:   {args.rl}")
    rl_data  = json.load(open(args.rl))
    print(f"  SFT: {len(sft_data)} examples")
    print(f"  RL:  {len(rl_data)} examples")

    n = min(len(sft_data), len(rl_data))
    print(f"\nComputing per-example SCD on first {n}...")

    print("  -> SFT...")
    sft_scds = per_example_scd(
        [d["generated"] for d in sft_data[:n]],
        [d["gt"]        for d in sft_data[:n]],
        text2cad_src=text2cad_src, n_points=n_points,
    )
    print("  -> RL...")
    rl_scds = per_example_scd(
        [d["generated"] for d in rl_data[:n]],
        [d["gt"]        for d in rl_data[:n]],
        text2cad_src=text2cad_src, n_points=n_points,
    )

    print("\nDistribution stats:")
    summarize(sft_scds, "SFT")
    summarize(rl_scds,  "RL")

    compare(sft_scds, rl_scds)


if __name__ == "__main__":
    main()
