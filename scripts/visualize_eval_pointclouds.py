"""
Visualize predicted vs ground-truth point clouds for eval outputs.

Picks the first N examples where both predicted and ground-truth STEP
files tessellate successfully (skips parse failures — they don't belong
in a results-showcase visualization since the headline RR metric already
quantifies the failure rate).

Optional: restrict to in-distribution or out-of-distribution examples.

Usage:
    # Default: first 5 parseable from the full eval (mix of in/out of dist)
    python scripts/visualize_eval_pointclouds.py \\
        --json $SCRATCH/stepforge/eval_9965567/eval_in_dist.json \\
        --num 5 \\
        --out  $SCRATCH/stepforge/plots/eval_pointclouds.png

    # Force the displayed examples to be out-of-distribution only
    python scripts/visualize_eval_pointclouds.py \\
        --json $SCRATCH/stepforge/eval_9965567/eval_in_dist.json \\
        --num 5 --out-of-dist-only \\
        --out  $SCRATCH/stepforge/plots/eval_pointclouds_ood.png
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
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--out", default="eval_pointclouds.png")
    ap.add_argument("--config", default="configs/config_gautschi.yaml")
    ap.add_argument("--n-points", type=int, default=2000)
    ap.add_argument("--title", default="Test examples — predicted vs ground truth")
    ap.add_argument("--in-dist-only", action="store_true",
                    help="Restrict to in-distribution examples")
    ap.add_argument("--out-of-dist-only", action="store_true",
                    help="Restrict to out-of-distribution examples")
    ap.add_argument("--pred-fail-only", action="store_true",
                    help="Show only examples where prediction failed to parse but "
                         "GT tessellates cleanly")
    ap.add_argument("--include-fail", action="store_true",
                    help="Render (num-1) parseable examples plus 1 example where "
                         "the prediction fails to parse but GT tessellates cleanly "
                         "(mixes working and failure modes in one figure)")
    ap.add_argument("--start", type=int, default=0,
                    help="Start scanning from this index (default 0)")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    text2cad_src = cfg.paths.text2cad_src

    data = json.load(open(args.json))
    print(f"Loaded {len(data)} examples")

    candidates = list(enumerate(data))[args.start:]
    if args.in_dist_only:
        candidates = [(i, d) for i, d in candidates if d.get("in_dist")]
    elif args.out_of_dist_only:
        candidates = [(i, d) for i, d in candidates if not d.get("in_dist")]

    parseable_target = args.num
    fail_target = 0
    if args.pred_fail_only:
        parseable_target = 0
        fail_target = args.num
    elif args.include_fail:
        parseable_target = max(args.num - 1, 0)
        fail_target = 1

    print(f"Scanning {len(candidates)} candidates: need "
          f"{parseable_target} parseable + {fail_target} pred-fail")

    parseable = []
    failed = []
    for (i, d) in candidates:
        if len(parseable) >= parseable_target and len(failed) >= fail_target:
            break
        pred_pc, _ = _safe_step_to_pointcloud(d["generated"],
                                               n_points=args.n_points,
                                               text2cad_src=text2cad_src,
                                               deflection=None)
        gt_pc, _   = _safe_step_to_pointcloud(d["gt"],
                                               n_points=args.n_points,
                                               text2cad_src=text2cad_src,
                                               deflection=None)
        if pred_pc is None and gt_pc is not None:
            if len(failed) < fail_target:
                failed.append((i, d, pred_pc, gt_pc))
        elif pred_pc is not None and gt_pc is not None:
            if len(parseable) < parseable_target:
                parseable.append((i, d, pred_pc, gt_pc))
        # else: both failed or only gt failed — skip

    # Render parseable first, then the failure(s) so the failure case is
    # the visual "anchor" at the bottom of the figure.
    selected = parseable + failed

    n = len(selected)
    if n == 0:
        sys.exit("No parseable examples found matching the filter.")
    print(f"Rendering {n} examples (indices: {[i for i, _, _, _ in selected]})")

    fig = plt.figure(figsize=(10, 4.3 * n + 0.6), constrained_layout=True)
    fig.suptitle(args.title, fontsize=13, fontweight="bold")

    subfigs = fig.subfigures(n, 1, hspace=0.05) if n > 1 else [fig.subfigures(1, 1)]

    for sf, (idx, d, pred_pc, gt_pc) in zip(subfigs, selected):
        cap = d["caption"]
        in_dist = d.get("in_dist")

        scd_val = None
        try:
            s = scaled_chamfer_distance(pred_pc, gt_pc)
            scd_val = s if np.isfinite(s) else None
        except Exception:
            pass

        in_dist_str = (f"in-dist · " if in_dist else "out-of-dist · ") if in_dist is not None else ""
        scd_str = (f"SCD = {scd_val:.4f}" if scd_val is not None else "SCD = N/A")
        cap_wrapped = "\n".join(textwrap.wrap(cap, width=110))
        sf.suptitle(f"idx = {idx}  ·  {in_dist_str}{scd_str}\n{cap_wrapped}",
                    fontsize=10, ha="center")

        axes = sf.subplots(1, 2, subplot_kw={"projection": "3d"})
        render_axes(axes[0], pred_pc, "predicted")
        render_axes(axes[1], gt_pc,   "ground truth")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
