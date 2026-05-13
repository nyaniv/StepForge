"""
Bar chart comparing CR / RR / MSCD across:
  - Your model: full eval (100 samples)
  - Your model: in-distribution subset (33 samples)
  - Paper STEP-LLM (SFT)
  - Paper STEP-LLM (GRPO)

Hardcoded paper numbers; your numbers passed in as args (or use defaults
matching the rl_v2 evaluation).

Usage:
    python scripts/plot_comparison_bars.py \\
        --out $SCRATCH/stepforge/plots/comparison_bars.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-cr",   type=float, default=87.00)
    ap.add_argument("--full-rr",   type=float, default=85.00)
    ap.add_argument("--full-mscd", type=float, default=0.0790)
    ap.add_argument("--in-dist-cr",   type=float, default=96.97)
    ap.add_argument("--in-dist-rr",   type=float, default=93.94)
    ap.add_argument("--in-dist-mscd", type=float, default=0.0563)
    ap.add_argument("--out", default="comparison_bars.png")
    args = ap.parse_args()

    methods = [
        "Mine (full eval, N=100)",
        "Mine (in-distribution, N=33)",
        "Paper SFT",
        "Paper GRPO",
    ]
    cr   = [args.full_cr,   args.in_dist_cr,   97.00, 99.00]
    rr   = [args.full_rr,   args.in_dist_rr,   95.18, 92.00]
    mscd = [args.full_mscd, args.in_dist_mscd, 0.53,  0.098]

    colors = ["#1f77b4", "#2ca02c", "#888888", "#444444"]
    x = np.arange(len(methods))
    width = 0.6

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, vals, title, ylabel, fmt in [
        (axes[0], cr,   "Completion Rate (CR)",   "% completion (higher = better)", "{:.2f}%"),
        (axes[1], rr,   "Renderability Rate (RR)", "% renderable (higher = better)", "{:.2f}%"),
        (axes[2], mscd, "MSCD",                    "Median Scaled Chamfer (lower = better)", "{:.4f}"),
    ]:
        bars = ax.bar(x, vals, width=width, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=18, ha="right", fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        # Annotate bars
        ymax = max(vals) * 1.15
        ax.set_ylim(0, ymax)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + ymax * 0.01,
                    fmt.format(v), ha="center", va="bottom", fontsize=9)

    fig.suptitle("StepForge: my model performance vs paper baselines",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
