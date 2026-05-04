"""
Plot RL training curves from a GRPOTrainer trainer_state.json.

Reads `log_history` (one entry per step) and produces a 4-panel figure:
  fig1: total reward + format / parse / scd reward components
  fig2: loss + KL
  fig3: completion length
  fig4: gradient norm

Usage:
    python scripts/plot_rl_rewards.py \
        --state $SCRATCH/stepforge/checkpoints/rl-refined/checkpoint-80/trainer_state.json \
        --out plots/rl_refined_rewards.png
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
    ap.add_argument("--title", default="StepForge RL (refined SFT) — 80 GRPO steps")
    args = ap.parse_args()

    with open(args.state) as f:
        state = json.load(f)
    hist = state["log_history"]

    steps = np.array([e["step"] for e in hist])
    total = np.array([e.get("reward", np.nan) for e in hist])
    fmt   = np.array([e.get("rewards/format_reward_fn", np.nan) for e in hist])
    parse = np.array([e.get("rewards/parse_reward_fn",  np.nan) for e in hist])
    scd   = np.array([e.get("rewards/reward_fn",        np.nan) for e in hist])
    loss  = np.array([e.get("loss", np.nan) for e in hist])
    kl    = np.array([e.get("kl",   np.nan) for e in hist])
    comp  = np.array([e.get("completion_length", np.nan) for e in hist])
    grad  = np.array([e.get("grad_norm", np.nan) for e in hist])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(args.title, fontsize=14, y=0.995)

    ax = axes[0, 0]
    ax.plot(steps, total, "k-",  lw=2.2, label="total reward")
    ax.plot(steps, fmt,   "C0-", lw=1.4, alpha=0.8, label="format")
    ax.plot(steps, parse, "C1-", lw=1.4, alpha=0.8, label="parse")
    ax.plot(steps, scd,   "C2-", lw=1.4, alpha=0.8, label="SCD (chamfer)")
    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    ax.set_title("Reward components")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps, loss, "C3-", lw=1.6, label="loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss", color="C3")
    ax.tick_params(axis="y", labelcolor="C3")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(steps, kl, "C4-", lw=1.4, alpha=0.8, label="KL")
    ax2.set_ylabel("KL", color="C4")
    ax2.tick_params(axis="y", labelcolor="C4")
    ax.set_title("Loss & KL divergence")

    ax = axes[1, 0]
    ax.plot(steps, comp, "C5-", lw=1.6)
    ax.set_xlabel("step")
    ax.set_ylabel("avg completion length (tokens)")
    ax.set_title("Completion length")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps, grad, "C6-", lw=1.6)
    ax.set_xlabel("step")
    ax.set_ylabel("grad norm")
    ax.set_title("Gradient norm")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")

    print("\nFinal-step summary:")
    last = hist[-1]
    for k in ("step", "reward", "rewards/format_reward_fn",
              "rewards/parse_reward_fn", "rewards/reward_fn",
              "loss", "kl", "completion_length", "grad_norm"):
        print(f"  {k:<32s} {last.get(k):>10.4f}" if isinstance(last.get(k), (int, float))
              else f"  {k:<32s} {last.get(k)}")


if __name__ == "__main__":
    main()
