"""
Visualize predicted vs ground-truth point clouds — IN-DISTRIBUTION ONLY.

Filters the eval JSON to records with in_dist=True, then renders the first
N of them with proper subfigure layout (no caption-into-axes collisions).

Usage:
    python scripts/visualize_eval_in_dist.py \\
        --json $SCRATCH/stepforge/eval_9965567/eval_in_dist.json \\
        --num 6 \\
        --out  $SCRATCH/stepforge/plots/eval_pointclouds_in_dist.png
"""

import argparse
import json
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from reward.scd_reward import _safe_step_to_pointcloud, scaled_chamfer_distance


def render_axes(ax, pc, label):
    if pc is None or len(pc) == 0:
        ax.text2D(0.5, 0.5, "parse failed",
                  ha="center", va="center",
                  transform=ax.transAxes,
                  color="#c0392b", fontsize=12, fontweight="bold")
        ax.set_title(label, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        return
    color = "#1f77b4" if label == "predicted" else "#2ca02c"
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=0.6, alpha=0.45, c=color)
    ax.set_title(label, fontsize=10)
    ax.set_box_aspect((1, 1, 1))
    ranges = pc.max(axis=0) - pc.min(axis=0)
    r = max(ranges.max(), 1e-6)
    c = (pc.max(axis=0) + pc.min(axis=0)) / 2
    ax.set_xlim(c[0] - r/2, c[0] + r/2)
    ax.set_ylim(c[1] - r/2, c[1] + r/2)
    ax.set_zlim(c[2] - r/2, c[2] + r/2)
    ax.tick_params(axis="both", labelsize=6)


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
    print(f"Filtering to examples where both pred and GT parse cleanly...")

    # Pre-screen: pick the first `num` in-dist examples where BOTH pred and GT
    # tessellate successfully (skip parse failures — they don't belong in a
    # results-showcase visualization, and the headline RR already quantifies them)
    selected = []
    for (i, d) in in_dist:
        if len(selected) >= args.num:
            break
        pred_pc, _ = _safe_step_to_pointcloud(d["generated"],
                                               n_points=args.n_points,
                                               text2cad_src=text2cad_src,
                                               deflection=None)
        gt_pc, _   = _safe_step_to_pointcloud(d["gt"],
                                               n_points=args.n_points,
                                               text2cad_src=text2cad_src,
                                               deflection=None)
        if pred_pc is None or gt_pc is None:
            print(f"  skip idx={i} (parse failed)")
            continue
        selected.append((i, d, pred_pc, gt_pc))

    n = len(selected)
    print(f"Rendering {n} parseable in-distribution examples "
          f"(indices: {[i for i, _, _, _ in selected]})")

    fig = plt.figure(figsize=(10, 4.3 * n + 0.6), constrained_layout=True)
    fig.suptitle("In-distribution test examples — predicted vs ground truth",
                 fontsize=13, fontweight="bold")

    subfigs = fig.subfigures(n, 1, hspace=0.05) if n > 1 else [fig.subfigures(1, 1)]

    for sf, (idx, d, pred_pc, gt_pc) in zip(subfigs, selected):
        cap = d["caption"]
        print(f"  rendering idx={idx}: {cap[:60]}...")
        scd_val = None
        try:
            s = scaled_chamfer_distance(pred_pc, gt_pc)
            scd_val = s if np.isfinite(s) else None
        except Exception:
            pass

        scd_str = (f"SCD = {scd_val:.4f}" if scd_val is not None else "SCD = N/A")
        cap_wrapped = "\n".join(textwrap.wrap(cap, width=110))
        sf.suptitle(f"idx = {idx}  ·  {scd_str}\n{cap_wrapped}",
                    fontsize=10, ha="center")

        axes = sf.subplots(1, 2, subplot_kw={"projection": "3d"})
        render_axes(axes[0], pred_pc, "predicted")
        render_axes(axes[1], gt_pc,   "ground truth")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
