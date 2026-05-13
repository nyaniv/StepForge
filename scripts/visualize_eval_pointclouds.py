"""
Visualize predicted vs ground-truth point clouds for eval outputs.

For each selected example, tessellates both the predicted and the GT STEP
file into 3D point clouds and renders them side by side in a multi-panel
matplotlib figure. The caption is shown above each pair.

This is the strongest sanity check that the model is producing
geometrically faithful CAD models, not just syntactically valid files
that happen to score well by accident.

Uses the spawn-pool isolation from reward/scd_reward.py so a pathological
STEP file segfault on one example doesn't kill the whole script.

Usage:
    python scripts/visualize_eval_pointclouds.py \\
        --json $SCRATCH/stepforge/eval_9965567/eval_in_dist.json \\
        --indices 0 10 30 50 80 \\
        --out  $SCRATCH/stepforge/plots/eval_pointclouds.png
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


def render_pair(ax_pred, ax_gt, pred_pc, gt_pc, title, scd):
    """Render predicted and GT point clouds on two given Axes3D."""
    for ax, pc, label in [(ax_pred, pred_pc, "predicted"),
                          (ax_gt,   gt_pc,   "ground truth")]:
        if pc is None or len(pc) == 0:
            ax.text(0.5, 0.5, 0.5, "parse failed", ha="center", va="center",
                    transform=ax.transAxes, color="red", fontsize=12)
            ax.set_title(label, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            continue
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=0.5, alpha=0.4,
                   c="C0" if label == "predicted" else "C2")
        ax.set_title(label, fontsize=10)
        ax.set_box_aspect((1, 1, 1))
        # Normalize axes to same scale for visual comparability
        all_pts = pc
        ranges = all_pts.max(axis=0) - all_pts.min(axis=0)
        max_range = max(ranges.max(), 1e-6)
        center = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
        for setter, c, r in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim],
                                 center, [max_range] * 3):
            setter(c - r/2, c + r/2)
        ax.tick_params(axis='both', labelsize=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Eval output JSON")
    ap.add_argument("--indices", type=int, nargs="+", default=[0, 10, 30, 50, 80],
                    help="Sample indices to visualize")
    ap.add_argument("--out", default="eval_pointclouds.png")
    ap.add_argument("--config", default="configs/config_gautschi.yaml")
    ap.add_argument("--n-points", type=int, default=2000)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    text2cad_src = cfg.paths.text2cad_src

    data = json.load(open(args.json))
    n = len(args.indices)
    print(f"Loaded {len(data)} examples; rendering indices {args.indices}")

    fig = plt.figure(figsize=(8, 3.5 * n))
    for row, idx in enumerate(args.indices):
        if idx >= len(data):
            print(f"  skip idx={idx} (out of range)")
            continue
        d = data[idx]
        cap = d["caption"]
        pred = d["generated"]
        gt   = d["gt"]

        print(f"  rendering idx={idx} (caption: {cap[:60]}...)")
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
        render_pair(ax_pred, ax_gt, pred_pc, gt_pc, cap, scd_val)

        in_dist = d.get("in_dist")
        in_dist_str = f"  in_dist={in_dist}" if in_dist is not None else ""
        scd_str = f"SCD={scd_val:.4f}" if scd_val is not None else "SCD=N/A"
        cap_short = cap if len(cap) <= 110 else cap[:107] + "..."
        fig.text(0.5, 1 - (row + 0.05) / n,
                 f"idx={idx}{in_dist_str}  |  {scd_str}\n{cap_short}",
                 ha="center", fontsize=9, wrap=True)

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
