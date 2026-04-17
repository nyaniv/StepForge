"""
Comprehensive training plots for Gautschi 4-GPU SFT runs.

Reads sft_loss.csv and sft_train.log directly from the run directory.
Generates 4 figures:
  fig1_loss_curve.png        — loss vs step + epoch, with smoothing
  fig2_grad_norm.png         — gradient norm over training
  fig3_lr_schedule.png       — learning rate schedule
  fig4_epoch_summary.png     — per-epoch avg/min/max loss bars
  fig5_main_vs_refined.png   — comparison (only if --refined-dir given)

Usage (on Gautschi or locally after scp):
    python scripts/plot_gautschi_run.py \
        --run-dir $SCRATCH/stepforge/runs/sft_4gpu_9281837 \
        [--refined-dir $SCRATCH/stepforge/runs/sft_4gpu_refined_9277807] \
        [--out-dir plots/gautschi]
"""

import argparse
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "blue":   "#2563EB",
    "red":    "#DC2626",
    "green":  "#16A34A",
    "orange": "#D97706",
    "gray":   "#6B7280",
    "purple": "#7C3AED",
    "teal":   "#0891B2",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_csv(path: str) -> dict:
    steps, epochs, losses, lrs, gnorms = [], [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                steps.append(int(row["step"]))
                epochs.append(float(row["epoch"]))
                losses.append(float(row["loss"]))
                lrs.append(float(row.get("learning_rate", 0)))
                gnorms.append(float(row.get("grad_norm", 0)))
            except (ValueError, KeyError):
                continue
    return {"steps": steps, "epochs": epochs, "losses": losses,
            "lrs": lrs, "gnorms": gnorms}


def load_epoch_summaries(log_path: str) -> list[dict]:
    """Parse VerboseEpochCallback lines from sft_train.log."""
    summaries = []
    if not os.path.exists(log_path):
        return summaries
    # e.g. "EPOCH 3 COMPLETE  |  154.6 min  |  loss avg=0.0101  min=0.0075  max=0.0169"
    pattern = re.compile(
        r"EPOCH\s+(\d+)\s+COMPLETE\s*\|.*?(\d+\.\d+)\s*min.*?"
        r"loss avg=([\d.]+)\s+min=([\d.]+)\s+max=([\d.]+)"
    )
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                summaries.append({
                    "epoch":    int(m.group(1)),
                    "time_min": float(m.group(2)),
                    "avg":      float(m.group(3)),
                    "min":      float(m.group(4)),
                    "max":      float(m.group(5)),
                })
    return summaries


def smooth(values: list, window: int = 100) -> list:
    out = []
    for i, v in enumerate(values):
        s = max(0, i - window + 1)
        out.append(sum(values[s:i + 1]) / (i - s + 1))
    return out


# ── Figures ───────────────────────────────────────────────────────────────────

def fig1_loss_curve(data: dict, label: str, out_path: str):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False)
    fig.suptitle(f"SFT Training Loss — {label}", fontsize=14, fontweight="bold")

    for ax, x_vals, x_label, x_title in [
        (axes[0], data["steps"],  "Step",  "Loss vs Training Step"),
        (axes[1], data["epochs"], "Epoch", "Loss vs Epoch"),
    ]:
        ax.plot(x_vals, data["losses"], alpha=0.15, linewidth=0.6,
                color=COLORS["blue"], label="_raw")
        if len(data["losses"]) > 100:
            ax.plot(x_vals, smooth(data["losses"], 100),
                    color=COLORS["blue"], linewidth=2.0, label="Smoothed (w=100)")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title(x_title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))

        # annotate final value
        if data["losses"]:
            ax.annotate(
                f"Final: {data['losses'][-1]:.4f}",
                xy=(x_vals[-1], data["losses"][-1]),
                xytext=(-80, 15), textcoords="offset points",
                fontsize=9, color=COLORS["blue"],
                arrowprops=dict(arrowstyle="->", color=COLORS["blue"], lw=1.2),
            )

    if data["steps"]:
        axes[0].text(0.01, 0.97,
                     f"Steps logged: {len(data['steps']):,}\n"
                     f"Loss range: {min(data['losses']):.4f} – {max(data['losses']):.4f}\n"
                     f"Epoch reached: {data['epochs'][-1]:.2f}",
                     transform=axes[0].transAxes, ha="left", va="top", fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def fig2_grad_norm(data: dict, label: str, out_path: str):
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle(f"Gradient Norm — {label}", fontsize=14, fontweight="bold")

    ax.plot(data["steps"], data["gnorms"], alpha=0.2, linewidth=0.6,
            color=COLORS["orange"])
    if len(data["gnorms"]) > 100:
        ax.plot(data["steps"], smooth(data["gnorms"], 100),
                color=COLORS["orange"], linewidth=2.0, label="Smoothed (w=100)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if data["gnorms"]:
        p95 = np.percentile(data["gnorms"], 95)
        ax.axhline(p95, color=COLORS["red"], linestyle="--", linewidth=1.2,
                   label=f"P95 = {p95:.4f}")
        ax.legend()
        ax.text(0.01, 0.97,
                f"Max: {max(data['gnorms']):.4f}\n"
                f"P95: {p95:.4f}\n"
                f"Avg: {np.mean(data['gnorms']):.4f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def fig3_lr_schedule(data: dict, label: str, out_path: str):
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle(f"Learning Rate Schedule — {label}", fontsize=14, fontweight="bold")

    ax.plot(data["steps"], data["lrs"], color=COLORS["green"], linewidth=1.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate over Training (linear warmup + decay)")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2e"))

    if data["lrs"]:
        peak_idx = np.argmax(data["lrs"])
        ax.annotate(
            f"Peak: {max(data['lrs']):.2e}\n(step {data['steps'][peak_idx]:,})",
            xy=(data["steps"][peak_idx], data["lrs"][peak_idx]),
            xytext=(40, -20), textcoords="offset points",
            fontsize=9, color=COLORS["green"],
            arrowprops=dict(arrowstyle="->", color=COLORS["green"], lw=1.2),
        )
        ax.text(0.99, 0.97,
                f"Final LR: {data['lrs'][-1]:.2e}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def fig4_epoch_summary(summaries: list, label: str, out_path: str):
    if not summaries:
        print(f"  No epoch summaries found for {label}, skipping fig4.")
        return

    epochs   = [s["epoch"]    for s in summaries]
    avgs     = [s["avg"]      for s in summaries]
    mins     = [s["min"]      for s in summaries]
    maxs     = [s["max"]      for s in summaries]
    times    = [s["time_min"] for s in summaries]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Per-Epoch Summary — {label}", fontsize=14, fontweight="bold")

    # Left: avg loss per epoch with min/max error bars
    ax = axes[0]
    yerr_lo = [a - m for a, m in zip(avgs, mins)]
    yerr_hi = [m - a for m, a in zip(maxs, avgs)]
    bars = ax.bar(epochs, avgs, color=COLORS["blue"], alpha=0.85,
                  edgecolor="white", linewidth=1.5)
    ax.errorbar(epochs, avgs, yerr=[yerr_lo, yerr_hi],
                fmt="none", color=COLORS["gray"], capsize=5, linewidth=1.5)
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(yerr_hi) * 0.05,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Avg Loss per Epoch\n(error bars = min/max)")
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3, axis="y")

    # Right: epoch wall time
    ax2 = axes[1]
    bars2 = ax2.bar(epochs, times, color=COLORS["teal"], alpha=0.85,
                    edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars2, times):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.0f}m", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Wall Time (min)")
    ax2.set_title("Epoch Wall Time\n(decreasing = model converging faster)")
    ax2.set_xticks(epochs)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def fig5_comparison(main_data: dict, ref_data: dict, main_summ: list, ref_summ: list,
                    out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Main vs Refined Variant — Training Comparison",
                 fontsize=14, fontweight="bold")

    # Left: loss curves
    ax = axes[0]
    for data, label, color in [
        (main_data, "main",    COLORS["blue"]),
        (ref_data,  "refined", COLORS["orange"]),
    ]:
        ax.plot(data["epochs"], data["losses"], alpha=0.15, linewidth=0.6, color=color)
        if len(data["losses"]) > 100:
            ax.plot(data["epochs"], smooth(data["losses"], 100),
                    color=color, linewidth=2.0, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: per-epoch avg comparison
    ax2 = axes[1]
    if main_summ and ref_summ:
        common = sorted(set(s["epoch"] for s in main_summ) &
                        set(s["epoch"] for s in ref_summ))
        m_avgs = {s["epoch"]: s["avg"] for s in main_summ}
        r_avgs = {s["epoch"]: s["avg"] for s in ref_summ}
        x = np.arange(len(common))
        w = 0.35
        ax2.bar(x - w/2, [m_avgs[e] for e in common], w,
                label="main",    color=COLORS["blue"],   alpha=0.85, edgecolor="white")
        ax2.bar(x + w/2, [r_avgs[e] for e in common], w,
                label="refined", color=COLORS["orange"], alpha=0.85, edgecolor="white")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"Epoch {e}" for e in common])
        ax2.set_ylabel("Avg Loss")
        ax2.set_title("Per-Epoch Avg Loss Comparison")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")
    else:
        ax2.text(0.5, 0.5, "Insufficient epoch data\nfor comparison",
                 transform=ax2.transAxes, ha="center", va="center", fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir",     required=True,
                        help="Path to main run directory (e.g. .../sft_4gpu_9281837)")
    parser.add_argument("--refined-dir", default=None,
                        help="Path to refined variant run directory (optional)")
    parser.add_argument("--out-dir",     default="plots/gautschi",
                        help="Output directory for PNG files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    csv_path = os.path.join(args.run_dir, "sft_loss.csv")
    log_path = os.path.join(args.run_dir, "sft_train.log")

    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.")
        return

    print(f"Loading main run: {args.run_dir}")
    data     = load_csv(csv_path)
    summaries = load_epoch_summaries(log_path)
    label    = os.path.basename(args.run_dir)

    print(f"  {len(data['steps']):,} steps logged, "
          f"epoch {data['epochs'][-1]:.2f}, "
          f"{len(summaries)} epochs complete")

    fig1_loss_curve(data,     label, os.path.join(args.out_dir, "fig1_loss_curve.png"))
    fig2_grad_norm(data,      label, os.path.join(args.out_dir, "fig2_grad_norm.png"))
    fig3_lr_schedule(data,    label, os.path.join(args.out_dir, "fig3_lr_schedule.png"))
    fig4_epoch_summary(summaries, label, os.path.join(args.out_dir, "fig4_epoch_summary.png"))

    if args.refined_dir:
        ref_csv  = os.path.join(args.refined_dir, "sft_loss.csv")
        ref_log  = os.path.join(args.refined_dir, "sft_train.log")
        if os.path.exists(ref_csv):
            print(f"Loading refined run: {args.refined_dir}")
            ref_data = load_csv(ref_csv)
            ref_summ = load_epoch_summaries(ref_log)
            print(f"  {len(ref_data['steps']):,} steps logged, "
                  f"epoch {ref_data['epochs'][-1]:.2f}, "
                  f"{len(ref_summ)} epochs complete")
            fig5_comparison(data, ref_data, summaries, ref_summ,
                            os.path.join(args.out_dir, "fig5_main_vs_refined.png"))
        else:
            print(f"  Refined CSV not found: {ref_csv}")

    print(f"\nAll plots saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
