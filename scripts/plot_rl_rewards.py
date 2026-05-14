"""
Plot RL training curves from a GRPOTrainer trainer_state.json.

Single-panel figure showing the four reward signals across all GRPO steps.
The previous 4-panel layout (loss / KL / completion length / grad norm)
was dropped: with only 80 steps, those subplots are noisy and not
informative — the reward components plot is the single load-bearing chart.

Usage:
    python scripts/plot_rl_rewards.py \\
        --state $SCRATCH/stepforge/checkpoints/rl/checkpoint-80/trainer_state.json \\
        --out   $SCRATCH/stepforge/plots/rl_v2_rewards.png
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True,
                    help="Path to trainer_state.json")
    ap.add_argument("--out", default="rl_rewards.png",
                    help="Output PNG path")
    ap.add_argument("--title",
                    default="GRPO reward signals across 80 RL optimization steps")
    args = ap.parse_args()

    with open(args.state) as f:
        state = json.load(f)
    hist = state["log_history"]

    steps = np.array([e["step"] for e in hist])
    total = np.array([e.get("reward", np.nan) for e in hist])
    fmt   = np.array([e.get("rewards/format_reward_fn", np.nan) for e in hist])
    parse = np.array([e.get("rewards/parse_reward_fn",  np.nan) for e in hist])
    scd   = np.array([e.get("rewards/reward_fn",        np.nan) for e in hist])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    ax.plot(steps, total, "k-",  lw=2.4, label="total reward")
    ax.plot(steps, scd,   color="#2ca02c", lw=1.6, alpha=0.9, label="SCD (chamfer)")
    ax.plot(steps, parse, color="#ff7f0e", lw=1.4, alpha=0.85, label="parse")
    ax.plot(steps, fmt,   color="#1f77b4", lw=1.4, alpha=0.85, label="format")
    ax.set_xlabel("optimization step", fontsize=11)
    ax.set_ylabel("reward", fontsize=11)
    ax.set_title(args.title, fontsize=13, fontweight="bold", pad=14)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, steps.max() + 1)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")

    print("\nFinal-step summary:")
    last = hist[-1]
    for k in ("step", "reward", "rewards/format_reward_fn",
              "rewards/parse_reward_fn", "rewards/reward_fn"):
        v = last.get(k)
        if isinstance(v, (int, float)):
            print(f"  {k:<32s} {v:>10.4f}")
        else:
            print(f"  {k:<32s} {v}")


if __name__ == "__main__":
    main()
