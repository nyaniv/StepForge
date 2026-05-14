"""
Convergence plot: running CR, RR, MSCD as the number of test samples grows.

Takes an eval JSON of N generated outputs, then for each k = 1 .. N computes
the metric on the first k samples and plots the trajectory. Shows that the
point estimate is stable at the reported N (not still oscillating).

CR  is a simple string check — fast, no tessellation needed.
RR  and MSCD require OCC tessellation; each STEP file is tessellated ONCE
    upfront (via spawn-pool isolation) and cached, then the running metrics
    are computed from the cache. Total runtime is ~1 minute per 100 samples.

Usage:
    python scripts/plot_metric_convergence.py \\
        --json $SCRATCH/stepforge/eval_9965567/eval_original.json \\
        --out  $SCRATCH/stepforge/plots/metric_convergence.png
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from reward.scd_reward import _safe_step_to_pointcloud, scaled_chamfer_distance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default="metric_convergence.png")
    ap.add_argument("--config", default="configs/config_gautschi.yaml")
    ap.add_argument("--n-points", type=int, default=2000)
    ap.add_argument("--title",
                    default="Metric convergence as sample size grows")
    ap.add_argument("--in-dist-only", action="store_true",
                    help="Restrict the running statistic to within-budget "
                         "examples (in_dist=True in the JSON).")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    text2cad_src = cfg.paths.text2cad_src

    data = json.load(open(args.json))
    if args.in_dist_only:
        before = len(data)
        data = [d for d in data if d.get("in_dist")]
        print(f"Loaded {before} samples; filtered to {len(data)} within-budget")
    n = len(data)
    print(f"Computing running metrics over {n} samples")

    # Pre-compute per-sample primitives (CR is just a string check; RR/MSCD
    # need OCC). Doing it once up front, then running metrics are O(n).
    has_end = np.array(["END-ISO-10303-21;" in d["generated"] for d in data])

    pred_pcs = []
    gt_pcs = []
    print("Tessellating pred + gt STEP files (this is the slow part)...")
    for d in tqdm(data, desc="tessellate"):
        pred_pc, _ = _safe_step_to_pointcloud(d["generated"],
                                              n_points=args.n_points,
                                              text2cad_src=text2cad_src,
                                              deflection=None)
        gt_pc, _ = _safe_step_to_pointcloud(d["gt"],
                                            n_points=args.n_points,
                                            text2cad_src=text2cad_src,
                                            deflection=None)
        pred_pcs.append(pred_pc)
        gt_pcs.append(gt_pc)

    renderable = np.array([pc is not None for pc in pred_pcs])

    # Per-sample SCDs (None when either side fails to parse)
    scds = []
    print("Computing per-sample scaled chamfer distance...")
    for pc_p, pc_g in zip(pred_pcs, gt_pcs):
        if pc_p is None or pc_g is None:
            scds.append(None)
            continue
        try:
            s = scaled_chamfer_distance(pc_p, pc_g)
            scds.append(float(s) if np.isfinite(s) else None)
        except Exception:
            scds.append(None)

    # Running metrics
    Ns = np.arange(1, n + 1)
    cr_running = np.cumsum(has_end) / Ns
    rr_running = np.cumsum(renderable) / Ns
    mscd_running = np.array([
        (lambda subset=[s for s in scds[:k] if s is not None]:
            (np.median(subset) if subset else np.nan))()
        for k in Ns
    ])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, vals, title, ylabel, yfmt, final_label in [
        (axes[0], cr_running * 100,
         "Completion Rate (CR)",
         "CR (%)",
         "{:.2f}%",
         f"final = {cr_running[-1]*100:.2f}%"),
        (axes[1], rr_running * 100,
         "Renderability Rate (RR)",
         "RR (%)",
         "{:.2f}%",
         f"final = {rr_running[-1]*100:.2f}%"),
        (axes[2], mscd_running,
         "MSCD",
         "median scaled chamfer (lower = better)",
         "{:.4f}",
         f"final = {mscd_running[-1]:.4f}"),
    ]:
        ax.plot(Ns, vals, color="#1c4587", lw=1.8)
        # Horizontal line for the final value
        final = vals[-1]
        ax.axhline(final, color="#0d7a4e", lw=1.2, ls="--", alpha=0.85,
                   label=final_label)
        ax.set_xlabel("number of test samples (N)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=10)

    fig.suptitle(args.title, fontsize=13, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nSaved {args.out}")
    print(f"  CR(N=1..{n}) range: [{cr_running.min()*100:.2f}, {cr_running.max()*100:.2f}]")
    print(f"  RR(N=1..{n}) range: [{rr_running.min()*100:.2f}, {rr_running.max()*100:.2f}]")
    finite_mscd = mscd_running[np.isfinite(mscd_running)]
    if len(finite_mscd):
        print(f"  MSCD running range: [{finite_mscd.min():.4f}, {finite_mscd.max():.4f}]")


if __name__ == "__main__":
    main()
