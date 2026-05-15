"""
Visualize predicted vs ground-truth tessellated CAD meshes — within-budget only.

Filters the eval JSON to records with in_dist=True, then renders the first
N that tessellate cleanly. Each example is shown side-by-side as a shaded
3D mesh (predicted | ground truth) rather than as a point cloud.

Uses the spawn-pool isolation from reward/scd_reward.py so pathological
STEP files don't kill the whole script.

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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from omegaconf import OmegaConf

from reward.scd_reward import (_safe_step_to_pointcloud, _safe_step_to_mesh,
                                scaled_chamfer_distance)


PRED_COLOR = "#a8c8f0"   # soft blue
GT_COLOR   = "#a8d5b9"   # soft green
EDGE_COLOR = (0.15, 0.15, 0.15, 0.35)


def render_mesh(ax, tris, label):
    """Render a triangulated mesh on a 3D axes with shading + slight edges."""
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
    # Light shading: vary face color with surface normal z-component
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    safe_norms = np.where(norms < 1e-12, 1.0, norms)
    nz = (normals / safe_norms)[:, 2]
    # Map z-component [-1, 1] → shade factor [0.55, 1.0]
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
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--out", default="eval_pointclouds_in_dist.png")
    ap.add_argument("--config", default="configs/config_gautschi.yaml")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    text2cad_src = cfg.paths.text2cad_src
    n_points_for_scd = int(cfg.rl.reward.n_sample_points)

    data = json.load(open(args.json))
    in_dist = [(i, d) for i, d in enumerate(data) if d.get("in_dist")]
    print(f"Loaded {len(data)} examples; {len(in_dist)} in-distribution")
    print(f"Filtering to examples where both pred and GT tessellate cleanly...")

    selected = []
    for (i, d) in in_dist:
        if len(selected) >= args.num:
            break
        pred_mesh, _ = _safe_step_to_mesh(d["generated"],
                                           text2cad_src=text2cad_src,
                                           deflection=None)
        gt_mesh, _   = _safe_step_to_mesh(d["gt"],
                                           text2cad_src=text2cad_src,
                                           deflection=None)
        if pred_mesh is None or gt_mesh is None:
            print(f"  skip idx={i} (parse failed)")
            continue
        # Also compute SCD on point clouds (separate metric, well-tested)
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
        selected.append((i, d, pred_mesh, gt_mesh, scd_val))

    n = len(selected)
    print(f"Rendering {n} parseable within-budget examples "
          f"(indices: {[i for i, _, _, _, _ in selected]})")

    fig = plt.figure(figsize=(10, 4.3 * n + 0.6), constrained_layout=True)
    fig.suptitle("Within-budget test examples — predicted vs ground truth",
                 fontsize=13, fontweight="bold")

    subfigs = fig.subfigures(n, 1, hspace=0.05) if n > 1 else [fig.subfigures(1, 1)]

    for sf, (idx, d, pred_mesh, gt_mesh, scd_val) in zip(subfigs, selected):
        cap = d["caption"]
        print(f"  rendering idx={idx}: {cap[:60]}...")
        scd_str = (f"SCD = {scd_val:.4f}" if scd_val is not None else "SCD = N/A")
        cap_wrapped = "\n".join(textwrap.wrap(cap, width=110))
        sf.suptitle(f"idx = {idx}  ·  {scd_str}\n{cap_wrapped}",
                    fontsize=10, ha="center")
        axes = sf.subplots(1, 2, subplot_kw={"projection": "3d"})
        render_mesh(axes[0], pred_mesh, "predicted")
        render_mesh(axes[1], gt_mesh,   "ground truth")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
