"""
Plot training summary figures from saved log files.

Usage:
    python scripts/plot_training_summary.py

Outputs plots/ directory with PNG files.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Hard-coded data extracted from saved logs ──────────────────────────────────

# From logs/sft_train_epoch1.log — VerboseEpochCallback epoch 1 summary
SFT = {
    "epochs_run": 1,
    "total_epochs": 10,  # planned
    "steps": 7299,
    "wall_time_min": 194.2,
    "loss_avg": 0.0769,
    "loss_min": 0.0509,
    "loss_max": 0.3449,
    "loss_p10": 0.0636,
    "loss_p90": 0.0892,
    "grad_norm_avg": 0.0684,
    "grad_norm_max": 0.3065,
    "vram_alloc_gb": 6.66,
    "vram_reserved_gb": 12.57,
    "samples_per_sec": 10.021,
    "label_unmasked_pct": [44.8, 44.7, 43.2, 42.8, 44.4, 44.6, 44.6, 42.4],  # first 8 examples
}

# Dataset stats from sft_train_epoch1.log
DATASET = {
    "train_total": 116771,
    "test_total": 7970,
    "max_seq_length": 8192,
    # Train sequence length percentiles (tokens)
    "train_p25": 7591,
    "train_p50": 12161,
    "train_p75": 21709,
    "train_p90": 37052,
    "train_max": 892687,
    "train_over_limit": 83833,
    "train_over_limit_pct": 71.8,
    # Test
    "test_p25": 6171,
    "test_p50": 9914,
    "test_p75": 17904,
    "test_p90": 32071,
    "test_over_limit": 4531,
    "test_over_limit_pct": 56.9,
    # Retrieved step truncation
    "train_truncated_retrieved": 116743,
    "train_truncated_retrieved_pct": 100.0,
}

# From logs/diagnose_sft_epoch1_4096tokens.log — 5-sample generation diagnosis
DIAGNOSE = {
    "n_samples": 5,
    "max_new_tokens": 4096,
    "hit_terminator": 0,
    "hit_eos": 0,
    "has_data_section": 5,
    "truncated_mid_entity": 4,
    "has_dangling": 5,
    # Per-sample (token ratio, entity ratio, GT tokens)
    "samples": [
        {"uid": "00351230", "caption": "Rect. plate w/ rounded corners + central hole",
         "gt_tokens": 6937,  "gen_tokens": 4096, "token_ratio": 0.59, "entity_ratio": 0.10},
        {"uid": "00358303", "caption": "Cylindrical object, uniform diameter",
         "gt_tokens": 3521,  "gen_tokens": 4096, "token_ratio": 1.16, "entity_ratio": 1.36},
        {"uid": "00351935", "caption": "Rectangular box, flat bottom, open top",
         "gt_tokens": 17325, "gen_tokens": 4096, "token_ratio": 0.24, "entity_ratio": 0.12},
        {"uid": "00356893", "caption": "Cylindrical base + smaller circular top on rod",
         "gt_tokens": 6057,  "gen_tokens": 4096, "token_ratio": 0.68, "entity_ratio": 0.87},
        {"uid": "00356004", "caption": "Cylinder w/ small hole at top center",
         "gt_tokens": 3699,  "gen_tokens": 4096, "token_ratio": 1.11, "entity_ratio": 1.37},
    ],
    "avg_token_ratio": 0.75,
    "avg_entity_ratio": 0.77,
}

# From logs/rl_train_40step_smoke.log — 2 steps completed
RL = {
    "steps": [1, 2],
    "entropy": [1.775, 1.569],
    "step_time_s": [368.5, 371.3],
    "reward": [0.0, 0.0],
    "clipped_ratio": [1.0, 1.0],
    "kl": [0.0, 0.0],
}

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"blue": "#2563EB", "red": "#DC2626", "green": "#16A34A",
          "orange": "#D97706", "gray": "#6B7280", "purple": "#7C3AED"}


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — SFT Training Loss Summary
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("SFT Training — Epoch 1 of 10\n"
             "Llama-3.2-3B-Instruct + LoRA (r=16)  |  8,192 seq len  |  116k examples",
             fontsize=13, fontweight="bold")

# Left: loss distribution
ax = axes[0]
stats = [SFT["loss_min"], SFT["loss_p10"], SFT["loss_avg"], SFT["loss_p90"], SFT["loss_max"]]
labels = ["Min\n(best step)", "P10", "Avg", "P90", "Max\n(worst step)"]
bar_colors = [COLORS["green"], COLORS["blue"], COLORS["blue"], COLORS["orange"], COLORS["red"]]
bars = ax.bar(labels, stats, color=bar_colors, edgecolor="white", linewidth=1.5, width=0.6)
for bar, val in zip(bars, stats):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.axhline(SFT["loss_avg"], color=COLORS["blue"], linestyle="--", alpha=0.5, linewidth=1)
ax.set_title("Cross-Entropy Loss Distribution\n(7,299 training steps)", fontsize=11)
ax.set_ylabel("Loss")
ax.set_ylim(0, SFT["loss_max"] * 1.2)
ax.text(0.97, 0.97, f"7,299 steps\n{SFT['wall_time_min']:.0f} min\n{SFT['samples_per_sec']:.1f} samples/s",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

# Right: label masking distribution
ax = axes[1]
pcts = SFT["label_unmasked_pct"]
x = range(len(pcts))
ax.bar(x, pcts, color=COLORS["purple"], edgecolor="white", linewidth=1.5, alpha=0.85)
ax.axhline(np.mean(pcts), color=COLORS["red"], linestyle="--", linewidth=1.5,
           label=f"Mean = {np.mean(pcts):.1f}%")
ax.set_title("Response Token Masking Check\n(first 8 training examples)", fontsize=11)
ax.set_xlabel("Example index")
ax.set_ylabel("% of tokens that are response (unmasked)")
ax.set_xticks(list(x))
ax.set_ylim(0, 60)
ax.legend(fontsize=10)
ax.text(0.03, 0.97,
        "Only response tokens after\n'### output:' contribute to loss.\n~44% confirms masking is correct.",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
out = os.path.join(PLOTS_DIR, "fig1_sft_loss_summary.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Dataset Statistics
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Dataset Statistics — Text2CAD STEP Files\n116,771 train  |  7,970 test",
             fontsize=13, fontweight="bold")

# Left: sequence length distribution (percentiles)
ax = axes[0]
percentiles = ["P25", "P50\n(median)", "P75", "P90"]
train_vals = [DATASET["train_p25"], DATASET["train_p50"], DATASET["train_p75"], DATASET["train_p90"]]
test_vals  = [DATASET["test_p25"],  DATASET["test_p50"],  DATASET["test_p75"],  DATASET["test_p90"]]
x = np.arange(len(percentiles))
w = 0.35
b1 = ax.bar(x - w/2, train_vals, w, label="Train", color=COLORS["blue"], alpha=0.85, edgecolor="white")
b2 = ax.bar(x + w/2, test_vals,  w, label="Test",  color=COLORS["orange"], alpha=0.85, edgecolor="white")
ax.axhline(DATASET["max_seq_length"], color=COLORS["red"], linestyle="--", linewidth=2,
           label=f"max_seq_length = {DATASET['max_seq_length']:,}")
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
            f"{bar.get_height():,.0f}", ha="center", va="bottom", fontsize=8)
ax.set_title("Full Sequence Length Distribution\n(prompt + retrieved + GT STEP)", fontsize=11)
ax.set_ylabel("Tokens")
ax.set_xticks(x)
ax.set_xticklabels(percentiles)
ax.set_ylim(0, 45000)
ax.legend(fontsize=10)

# Right: truncation rates
ax = axes[1]
categories = ["Train sequences\ntruncated", "Test sequences\ntruncated", "Retrieved steps\ntruncated (train)"]
pcts = [DATASET["train_over_limit_pct"], DATASET["test_over_limit_pct"], DATASET["train_truncated_retrieved_pct"]]
counts = [DATASET["train_over_limit"], DATASET["test_over_limit"], DATASET["train_truncated_retrieved"]]
colors = [COLORS["red"], COLORS["orange"], COLORS["purple"]]
bars = ax.bar(categories, pcts, color=colors, edgecolor="white", linewidth=1.5, alpha=0.85, width=0.5)
for bar, pct, cnt in zip(bars, pcts, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{pct:.1f}%\n({cnt:,})", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Truncation Rates\n(data clipped to fit max_seq_length = 8,192)", fontsize=11)
ax.set_ylabel("% of examples truncated")
ax.set_ylim(0, 115)
ax.text(0.03, 0.03,
        "71.8% of training sequences exceed\nmax_seq_length → only partial GT\nSTEP files seen during training.\n"
        "Increasing seq length would improve\ncoverage but requires more VRAM.",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
out = os.path.join(PLOTS_DIR, "fig2_dataset_stats.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — SFT Output Quality (Diagnosis)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("SFT Output Quality — 5-Sample Diagnosis (epoch 1, max_new_tokens=4,096)\n"
             "Greedy decoding on training set examples",
             fontsize=13, fontweight="bold")

samples = DIAGNOSE["samples"]
captions = [f"S{i+1}\n{s['uid']}" for i, s in enumerate(samples)]
gt_tokens = [s["gt_tokens"] for s in samples]
gen_tokens = [s["gen_tokens"] for s in samples]
token_ratios = [s["token_ratio"] for s in samples]
entity_ratios = [s["entity_ratio"] for s in samples]

# Left: GT vs Generated token counts
ax = axes[0]
x = np.arange(len(samples))
w = 0.35
ax.bar(x - w/2, gt_tokens,  w, label="GT STEP",    color=COLORS["green"],  alpha=0.85, edgecolor="white")
ax.bar(x + w/2, gen_tokens, w, label="Generated",  color=COLORS["blue"],   alpha=0.85, edgecolor="white")
ax.axhline(4096, color=COLORS["red"], linestyle="--", linewidth=1.5, label="max_new_tokens limit")
ax.set_title("Token Count: GT vs Generated", fontsize=11)
ax.set_ylabel("Tokens")
ax.set_xticks(x)
ax.set_xticklabels(captions, fontsize=9)
ax.legend(fontsize=9)
ax.text(0.5, 0.98, "All 5 completions hit the\ntoken limit — model never stops",
        transform=ax.transAxes, ha="center", va="top", fontsize=9, color=COLORS["red"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEE2E2", alpha=0.9))

# Middle: token ratio
ax = axes[1]
bar_cols = [COLORS["green"] if r >= 0.9 else COLORS["orange"] if r >= 0.5 else COLORS["red"]
            for r in token_ratios]
bars = ax.bar(captions, token_ratios, color=bar_cols, edgecolor="white", linewidth=1.5, alpha=0.85)
ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Perfect ratio = 1.0")
for bar, val in zip(bars, token_ratios):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.2f}x", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Token Ratio (Generated / GT)\n<1 = cut off, >1 = over-generating", fontsize=11)
ax.set_ylabel("Ratio")
ax.set_ylim(0, 1.7)
ax.legend(fontsize=9)
ax.text(0.03, 0.03, f"Avg: {DIAGNOSE['avg_token_ratio']:.2f}x",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=11, fontweight="bold",
        color=COLORS["orange"])

# Right: entity ratio
ax = axes[2]
bar_cols = [COLORS["green"] if 0.7 <= r <= 1.3 else COLORS["orange"] if 0.4 <= r <= 1.6 else COLORS["red"]
            for r in entity_ratios]
bars = ax.bar(captions, entity_ratios, color=bar_cols, edgecolor="white", linewidth=1.5, alpha=0.85)
ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Perfect ratio = 1.0")
for bar, val in zip(bars, entity_ratios):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.2f}x", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Entity Count Ratio (Generated / GT)\n<1 = incomplete, >1 = hallucinating entities", fontsize=11)
ax.set_ylabel("Ratio")
ax.set_ylim(0, 1.9)
ax.legend(fontsize=9)
ax.text(0.03, 0.03, f"Avg: {DIAGNOSE['avg_entity_ratio']:.2f}x",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=11, fontweight="bold",
        color=COLORS["orange"])

plt.tight_layout()
out = os.path.join(PLOTS_DIR, "fig3_sft_output_quality.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — RL Smoke Test (2 steps)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle("RL (GRPO) Smoke Test — 2 of 40 Steps\nCold-start from SFT epoch 1 checkpoint",
             fontsize=13, fontweight="bold")

steps = RL["steps"]

ax = axes[0]
ax.plot(steps, RL["reward"], "o-", color=COLORS["red"], linewidth=2, markersize=10)
ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.set_title("Total Reward per Step", fontsize=11)
ax.set_ylabel("Reward")
ax.set_xlabel("GRPO Step")
ax.set_ylim(-0.1, 0.5)
ax.set_xticks(steps)
ax.text(0.5, 0.5,
        "All rewards = 0\nModel never generates\nEND-ISO-10303-21;\nwithin 4,096 tokens\n\n"
        "→ No gradient signal\n→ RL cannot learn",
        transform=ax.transAxes, ha="center", va="center", fontsize=10,
        color=COLORS["red"], fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEE2E2", alpha=0.9))

ax = axes[1]
ax.plot(steps, RL["entropy"], "s-", color=COLORS["purple"], linewidth=2, markersize=10)
for s, e in zip(steps, RL["entropy"]):
    ax.annotate(f"{e:.3f}", (s, e), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=11, fontweight="bold")
ax.set_title("Policy Entropy\n(higher = more diverse outputs)", fontsize=11)
ax.set_ylabel("Entropy (nats)")
ax.set_xlabel("GRPO Step")
ax.set_ylim(0, 2.5)
ax.set_xticks(steps)
ax.text(0.03, 0.07,
        "Entropy falling: model collapsing\ntoward repetitive output\nwithout useful reward signal.",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

ax = axes[2]
# Summary table showing why RL fails and what fix is needed
ax.axis("off")
table_data = [
    ["Metric", "Value", "Expected (converged)"],
    ["Reward", "0.0", "> 0.2"],
    ["Format reward", "0.0", "> 0.1"],
    ["Parse reward", "0.0", "> 0.0"],
    ["SCD reward", "0.0", "> 0.0"],
    ["Clipped ratio", "1.0 (100%)", "< 0.5"],
    ["Terminated length", "0", "> 0"],
    ["KL divergence", "0.0", "> 0.0"],
    ["Entropy (step 2)", "1.569", "stable ~1.5+"],
]
col_colors = [["#DBEAFE"] * 3] + [["white"] * 3] * (len(table_data) - 1)
row_colors_list = [["#DBEAFE"] * 3]
for row in table_data[1:]:
    good = row[1] == row[2] or (row[2].startswith(">") and row[1] != "0.0" and row[1] != "1.0 (100%)")
    row_colors_list.append(["#FEE2E2", "#FEE2E2", "#DCFCE7"] if not good else ["#DCFCE7"] * 3)
tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
               cellLoc="center", loc="center",
               cellColours=row_colors_list[1:])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.1, 1.6)
ax.set_title("RL Metrics vs Converged Target\n(red = problem, green = target)", fontsize=11)

plt.tight_layout()
out = os.path.join(PLOTS_DIR, "fig4_rl_smoke_test.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Training Pipeline Overview (status summary)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle("StepLLM Training Pipeline — Current Status\nLlama-3.2-3B-Instruct → STEP File Generation",
             fontsize=14, fontweight="bold")
ax.axis("off")

stages = [
    ("Data Pipeline",       "[DONE]",     "116,771 train / 7,970 test\nRAG pairs built (FAISS retrieval)",  COLORS["green"]),
    ("SFT — Epoch 1/10",    "[DONE]",     "Loss avg=0.0769  |  3.24h on A100\nModel learns STEP syntax, not yet termination", COLORS["orange"]),
    ("SFT — Epochs 2–10",   "[PENDING]",  "~29h remaining\nRequired for model to learn END-ISO-10303-21;", COLORS["gray"]),
    ("RL (GRPO) Training",  "[BLOCKED]",  "Requires converged SFT (epochs 2–10)\nAll rewards=0 on 1-epoch SFT cold-start",  COLORS["red"]),
    ("Evaluation & App",    "[PENDING]",  "SCD metric on test set\nGradio demo app", COLORS["gray"]),
]

y_positions = [0.82, 0.62, 0.42, 0.22, 0.02]
for (stage, status, detail, color), y in zip(stages, y_positions):
    # Stage box
    fancy = mpatches.FancyBboxPatch((0.01, y), 0.97, 0.16,
                                     boxstyle="round,pad=0.01",
                                     facecolor=color + "22", edgecolor=color, linewidth=2,
                                     transform=ax.transAxes)
    ax.add_patch(fancy)
    ax.text(0.04, y + 0.11, stage, transform=ax.transAxes,
            fontsize=12, fontweight="bold", color=color, va="center")
    ax.text(0.35, y + 0.11, status, transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=color, va="center")
    ax.text(0.55, y + 0.08, detail, transform=ax.transAxes,
            fontsize=9, color="#374151", va="center")

plt.tight_layout()
out = os.path.join(PLOTS_DIR, "fig5_pipeline_status.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

print(f"\nAll plots saved to: {PLOTS_DIR}/")
