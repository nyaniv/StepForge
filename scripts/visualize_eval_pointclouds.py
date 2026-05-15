"""
Visualize predicted vs ground-truth tessellated CAD meshes.

Picks examples where both predicted and ground-truth STEP tessellate cleanly
(skipping parse failures by default — they don't belong in a results-showcase
figure). Each row shows the two shaded meshes side by side.

Flags:
    --in-dist-only        restrict to within-budget examples
    --out-of-dist-only    restrict to out-of-budget examples
    --pred-fail-only      show only rows where pred fails / GT succeeds
    --include-fail        render (num-1) parseable + 1 pred-fail row

Usage:
    python scripts/visualize_eval_pointclouds.py \\
        --json $SCRATCH/stepforge/eval_9965567/eval_in_dist.json \\
        --num 5 --include-fail --out-of-dist-only \\
        --out  $SCRATCH/stepforge/plots/eval_pointclouds.png
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from omegaconf import OmegaConf

from reward.scd_reward import (_safe_step_to_pointcloud, _safe_step_to_mesh,
                                scaled_chamfer_distance)


PRED_COLOR = "#a8c8f0"
GT_COLOR   = "#a8d5b9"
EDGE_COLOR = (0.15, 0.15, 0.15, 0.35)


def render_mesh(ax, tris, label):
    if tris is None or len(tris) == 0:
        ax.text2D(0.5, 0.5, "parse failed",
                  ha="center", va="center",
                  transform=ax.transAxes,
                  color="#c0392b", fontsize=12, fontweight="bold")
        ax.set_title(label, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        return
    color = PRED_COLOR if label == "predicted" else GT_COLOR
    poly = Poly3DCollection(tris, alpha=0.92, linewidth=0.15,
                            edgecolor=EDGE_COLOR, facecolor=color)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    safe_norms = np.where(norms < 1e-12, 1.0, norms)
    nz = (normals / safe_norms)[:, 2]
    shade = 0.55 + 0.225 * (nz + 1.0)
    base = np.array(matplotlib.colors.to_rgb(color))
    face_colors = base[None, :] * shade[:, None]
    face_colors = np.clip(face_colors, 0, 1)
    poly.set_facecolor(face_colors)
    ax.add_collection3d(poly)
    ax.set_title(label, fontsize=10)
    ax.set_box_aspect((1, 1, 1))
    pts = tris.reshape(-1, 3)
    ranges = pts.max(axis=0) - pts.min(axis=0)
    r = max(ranges.max(), 1e-6)
    c = (pts.max(axis=0) + pts.min(axis=0)) / 2
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
    ap.add_argument("--title", default="Test examples — predicted vs ground truth")
    ap.add_argument("--in-dist-only", action="store_true")
    ap.add_argument("--out-of-dist-only", action="store_true")
    ap.add_argument("--pred-fail-only", action="store_true",
                    help="Only rows where pred fails to parse but GT parses")
    ap.add_argument("--include-fail", action="store_true",
                    help="Render (num-1) parseable rows + 1 pred-fail row")
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    text2cad_src = cfg.paths.text2cad_src
    n_points_for_scd = int(cfg.rl.reward.n_sample_points)

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
        pred_mesh, _ = _safe_step_to_mesh(d["generated"],
                                           text2cad_src=text2cad_src,
                                           deflection=None)
        gt_mesh, _   = _safe_step_to_mesh(d["gt"],
                                           text2cad_src=text2cad_src,
                                           deflection=None)
        if pred_mesh is None and gt_mesh is not None:
            if len(failed) < fail_target:
                # Score not meaningful — pred didn't parse
                failed.append((i, d, None, gt_mesh, None))
        elif pred_mesh is not None and gt_mesh is not None:
            if len(parseable) < parseable_target:
                # Compute SCD on point clouds for the row caption
                pred_pc, _ = _safe_step_to_pointcloud(d["generated"],
                                                      n_points=n_points_for_scd,
                                                      text2cad_src=text2cad_src,
                                                      deflection=None)
                gt_pc, _   = _safe_step_to_pointcloud(d["gt"],
                                                      n_points=n_points_for_scd,
                                                      text2cad_src=text2cad_src,
                                                      deflection=None)
                scd_val = None
                if pred_pc is not None and gt_pc is not None:
                    try:
                        s = scaled_chamfer_distance(pred_pc, gt_pc)
                        scd_val = float(s) if np.isfinite(s) else None
                    except Exception:
                        pass
                parseable.append((i, d, pred_mesh, gt_mesh, scd_val))

    selected = parseable + failed
    n = len(selected)
    if n == 0:
        sys.exit("No examples found matching the filter.")
    print(f"Rendering {n} examples (indices: {[i for i, _, _, _, _ in selected]})")

    fig = plt.figure(figsize=(10, 4.3 * n + 0.6), constrained_layout=True)
    fig.suptitle(args.title, fontsize=13, fontweight="bold")

    subfigs = fig.subfigures(n, 1, hspace=0.05) if n > 1 else [fig.subfigures(1, 1)]

    for sf, (idx, d, pred_mesh, gt_mesh, scd_val) in zip(subfigs, selected):
        cap = d["caption"]
        in_dist = d.get("in_dist")
        in_dist_str = (f"in-budget · " if in_dist else "out-of-budget · ") \
                      if in_dist is not None else ""
        scd_str = (f"SCD = {scd_val:.4f}" if scd_val is not None else "SCD = N/A")
        cap_wrapped = "\n".join(textwrap.wrap(cap, width=110))
        sf.suptitle(f"idx = {idx}  ·  {in_dist_str}{scd_str}\n{cap_wrapped}",
                    fontsize=10, ha="center")
        axes = sf.subplots(1, 2, subplot_kw={"projection": "3d"})
        render_mesh(axes[0], pred_mesh, "predicted")
        render_mesh(axes[1], gt_mesh,   "ground truth")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
