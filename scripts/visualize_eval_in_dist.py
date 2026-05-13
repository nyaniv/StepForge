"""
Visualize predicted vs ground-truth point clouds — IN-DISTRIBUTION ONLY.

Filters the eval JSON to records with in_dist=True (GT ≤ max_completion_length),
then renders the first N of them as a multi-row side-by-side comparison PNG.

These are the examples where eval conditions match RL training conditions,
so the visualizations should show the model's true geometric fidelity
without the long-prompt truncation artifact.

Usage:
    python scripts/visualize_eval_in_dist.py \\
        --json $SCRATCH/stepforge/eval_9965567/eval_in_dist.json \\
        --num 6 \\
        --out $SCRATCH/stepforge/plots/eval_pointclouds_in_dist.png
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from reward.scd_reward import _safe_step_to_pointcloud, scaled_chamfer_distance


def render_pair(ax_pred, ax_gt, pred_pc, gt_pc):
    for ax, pc, label in [(ax_pred, pred_pc, "predicted"),
                          (ax_gt,   gt_pc,   "ground truth")]:
        if pc is None or len(pc) == 0:
            ax.text(0.5, 0.5, 0.5, "parse failed", ha="center", va="center",
                    transform=ax.transAxes, color="red", fontsize=11)
            ax.set_title(label, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            continue
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=0.5, alpha=0.4,
                   c="C0" if label == "predicted" else "C2")
        ax.set_title(label, fontsize=10)
        ax.set_box_aspect((1, 1, 1))
        ranges = pc.max(axis=0) - pc.min(axis=0)
        max_range = max(ranges.max(), 1e-6)
        center = (pc.max(axis=0) + pc.min(axis=0)) / 2
        for setter, c in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
            setter(c - max_range/2, c + max_range/2)
        ax.tick_params(axis='both', labelsize=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--out", default="eval_pointclouds_in_dist.png")
    ap.add_argument("--config", default="configs/config_gautschi.yaml")
    ap.add_argument("--n-points", type=int, default=2000)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    text2cad_src = cfg.paths.text2cad_src

    data = json.load(open(args.json))
    in_dist = [(i, d) for i, d in enumerate(data) if d.get("in_dist")]
    print(f"Loaded {len(data)} examples; {len(in_dist)} in-distribution")
    selected = in_dist[:args.num]
    print(f"Rendering {len(selected)} in-dist examples (indices: "
          f"{[i for i, _ in selected]})")

    n = len(selected)
    fig = plt.figure(figsize=(9, 3.5 * n))
    fig.suptitle("In-distribution test examples — predicted vs ground truth",
                 fontsize=12, y=0.998)

    for row, (idx, d) in enumerate(selected):
        pred = d["generated"]
        gt   = d["gt"]
        cap  = d["caption"]
        print(f"  rendering idx={idx}: {cap[:60]}...")

        pred_pc, _ = _safe_step_to_pointcloud(pred, n_points=args.n_points,
                                               text2cad_src=text2cad_src,
                                               deflection=None)
        gt_pc, _   = _safe_step_to_pointcloud(gt,   n_points=args.n_points,
                                               text2cad_src=text2cad_src,
                                               deflection=None)
        scd_val = None
        if pred_pc is not None and gt_pc is not None:
            try:
                s = scaled_chamfer_distance(pred_pc, gt_pc)
                scd_val = s if np.isfinite(s) else None
            except Exception:
                pass

        ax_pred = fig.add_subplot(n, 2, 2*row + 1, projection="3d")
        ax_gt   = fig.add_subplot(n, 2, 2*row + 2, projection="3d")
        render_pair(ax_pred, ax_gt, pred_pc, gt_pc)

        scd_str = f"SCD={scd_val:.4f}" if scd_val is not None else "SCD=N/A"
        cap_short = cap if len(cap) <= 110 else cap[:107] + "..."
        fig.text(0.5, 1 - (row + 0.05) / n - 0.005,
                 f"idx={idx}  |  {scd_str}\n{cap_short}",
                 ha="center", fontsize=9)

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
