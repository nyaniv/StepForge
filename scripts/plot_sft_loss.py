"""
Plot SFT training loss from sft_loss.csv.

Usage:
    python scripts/plot_sft_loss.py --csv $SCRATCH/stepforge/checkpoints/sft/sft_loss.csv
    python scripts/plot_sft_loss.py --csv .../sft_loss.csv --out loss_plot.png
    python scripts/plot_sft_loss.py \
        --csv .../sft/sft_loss.csv \
        --csv .../sft-refined/sft_loss.csv \
        --labels main refined \
        --out comparison.png
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")  # no display needed — saves to file
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_csv(path: str) -> dict:
    steps, epochs, losses, lrs = [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                steps.append(int(row["step"]))
                epochs.append(float(row["epoch"]))
                losses.append(float(row["loss"]))
                lrs.append(float(row.get("learning_rate", 0)))
            except (ValueError, KeyError):
                continue
    return {"steps": steps, "epochs": epochs, "losses": losses, "lrs": lrs}


def smooth(values: list, window: int = 50) -> list:
    out = []
    for i, v in enumerate(values):
        start = max(0, i - window + 1)
        out.append(sum(values[start:i+1]) / (i - start + 1))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    action="append", required=True,
                        help="Path(s) to sft_loss.csv — pass multiple times for comparison")
    parser.add_argument("--labels", action="append", default=None,
                        help="Legend labels — one per --csv (defaults to filename)")
    parser.add_argument("--out",    default="sft_loss.png",
                        help="Output PNG path (default: sft_loss.png)")
    parser.add_argument("--smooth", type=int, default=50,
                        help="Smoothing window in steps (default: 50, 0 = off)")
    args = parser.parse_args()

    labels = args.labels or [os.path.basename(os.path.dirname(p)) for p in args.csv]
    if len(labels) < len(args.csv):
        labels += [os.path.basename(os.path.dirname(p)) for p in args.csv[len(labels):]]

    data = [load_csv(p) for p in args.csv]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    fig.suptitle("SFT Training Loss", fontsize=14, fontweight="bold")

    # ── Top: loss vs step ────────────────────────────────────────────────────
    ax = axes[0]
    for d, label in zip(data, labels):
        raw = ax.plot(d["steps"], d["losses"], alpha=0.2, linewidth=0.8)
        color = raw[0].get_color()
        if args.smooth > 0 and len(d["losses"]) > args.smooth:
            ax.plot(d["steps"], smooth(d["losses"], args.smooth),
                    color=color, linewidth=1.8, label=label)
        else:
            raw[0].set_alpha(1.0)
            raw[0].set_label(label)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    # ── Bottom: loss vs epoch ────────────────────────────────────────────────
    ax2 = axes[1]
    for d, label in zip(data, labels):
        ax2.plot(d["epochs"], d["losses"], alpha=0.2, linewidth=0.8)
        color = ax2.lines[-1].get_color()
        if args.smooth > 0 and len(d["losses"]) > args.smooth:
            ax2.plot(d["epochs"], smooth(d["losses"], args.smooth),
                     color=color, linewidth=1.8, label=label)
        else:
            ax2.lines[-1].set_alpha(1.0)
            ax2.lines[-1].set_label(label)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss vs Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")

    # Print summary stats
    for d, label in zip(data, labels):
        if d["losses"]:
            print(f"\n{label}:")
            print(f"  Steps logged : {len(d['steps'])}")
            print(f"  Loss range   : {min(d['losses']):.4f} – {max(d['losses']):.4f}")
            print(f"  Final loss   : {d['losses'][-1]:.4f}")
            print(f"  Epoch reached: {d['epochs'][-1]:.2f}")


if __name__ == "__main__":
    main()
