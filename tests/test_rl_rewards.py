"""
test_rl_rewards.py — RL reward pipeline smoke test (refined-variant).

Pulls N examples from train.json, runs all three reward functions (format, parse, SCD),
and prints a distribution summary.

This version uses the refined-variant API:
  compute_reward() returns (reward, raw_scd, fail_stage, n_tris)

What to look for:
  - format_reward: should be ~1.0 on GT STEP files (they contain END-ISO-10303-21;)
  - parse_reward: GT files should parse — confirms OCP/OCC is installed correctly
  - scd_reward: GT files should earn ~1.0 (SCD ≈ 0 when comparing GT to itself)
  - fail_stage: should be "ok" for GT files; any other value flags a pipeline problem
  - In corrupt mode: all rewards should be 0, confirming the reward discriminates

Usage:
    python tests/test_rl_rewards.py --config configs/config.yaml --n 10
    python tests/test_rl_rewards.py --config configs/config_gautschi.yaml --n 20 --mode corrupt
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from omegaconf import OmegaConf

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="RL reward pipeline smoke test (refined-variant)")
parser.add_argument("--config", default="configs/config.yaml")
parser.add_argument("--n", type=int, default=10, help="Number of examples to test")
parser.add_argument("--mode", choices=["gt", "corrupt"], default="gt",
                    help=(
                        "gt: run rewards on ground-truth STEP (should earn ~1.0). "
                        "corrupt: deliberately truncate completions (should earn 0.0)."
                    ))
parser.add_argument("--verbose", action="store_true", help="Show per-step reward subprocess output")
args = parser.parse_args()

cfg = OmegaConf.load(args.config)

train_json = os.path.join(cfg.paths.processed_dir, "train.json")
if not os.path.exists(train_json):
    print(f"[ERROR] train.json not found at {train_json}")
    print("  Run data/build_dataset.py first.")
    sys.exit(1)

from reward.scd_reward import RewardConfig, compute_reward, compute_parse_reward

rcfg = RewardConfig(
    n_points=cfg.rl.reward.n_sample_points,
    delta_low=cfg.rl.reward.delta_low,
    delta_high=cfg.rl.reward.delta_high,
    bidirectional=cfg.rl.reward.get("chamfer_bidirectional", True),
    scale_prenorm=cfg.rl.reward.get("scale_prenorm", True),
    deflection=cfg.rl.reward.get("mesh_deflection", None),
)
text2cad_src = cfg.paths.text2cad_src

# ── Load examples ─────────────────────────────────────────────────────────────
print(f"Loading {args.n} examples from {train_json}...")
with open(train_json) as f:
    records = json.load(f)
records = records[:args.n]

# ── Reward helpers ────────────────────────────────────────────────────────────
def format_reward(completion: str) -> float:
    return 0.2 if "END-ISO-10303-21;" in completion else 0.0

def corrupt_step(step: str) -> str:
    """Simulate a bad model output: truncate the STEP file mid-body."""
    lines = step.splitlines()
    return "\n".join(lines[:max(5, len(lines) // 3)])

# ── Run rewards ───────────────────────────────────────────────────────────────
print(f"\nRunning rewards on {len(records)} examples (mode={args.mode})...\n")
print(f"{'#':>4}  {'format':>7}  {'parse':>7}  {'scd_rwd':>7}  {'raw_scd':>8}  {'stage':>15}  {'tris':>6}  {'time':>7}")
print("-" * 75)

fmt_rewards   = []
parse_rewards = []
scd_rewards   = []
raw_scds      = []
fail_stages   = []

for i, record in enumerate(records):
    gt_step = record.get("output") or record.get("step") or ""
    if not gt_step:
        print(f"  [{i+1:3d}]  SKIP — no output field in record")
        continue

    completion = gt_step if args.mode == "gt" else corrupt_step(gt_step)
    label = "GT" if args.mode == "gt" else "CORRUPT"

    t0 = time.time()

    fmt = format_reward(completion)
    prs = compute_parse_reward(completion, text2cad_src=text2cad_src)
    result = compute_reward(
        completion, gt_step,
        rcfg=rcfg,
        text2cad_src=text2cad_src,
        verbose=args.verbose,
    )
    elapsed = time.time() - t0

    # Unpack 4-tuple: (reward, raw_scd, fail_stage, n_triangles)
    scd, raw_scd, stage, n_tris = result

    fmt_rewards.append(fmt)
    parse_rewards.append(prs)
    scd_rewards.append(scd if not (scd != scd) else 0.0)  # nan → 0 for stats
    raw_scds.append(raw_scd)
    fail_stages.append(stage)

    scd_str = f"{scd:.3f}" if scd == scd else " nan"
    raw_str = f"{raw_scd:.4f}" if raw_scd == raw_scd else "     nan"
    print(f"  [{i+1:3d}]  {fmt:>7.3f}  {prs:>7.3f}  {scd_str:>7}  {raw_str:>8}  {stage:>15}  {n_tris:>6}  {elapsed:>6.1f}s  {label}")

# ── Summary ───────────────────────────────────────────────────────────────────
def _stats(vals):
    finite = [v for v in vals if v == v]  # exclude nan
    if not finite:
        return "all nan"
    s = sorted(finite)
    n = len(s)
    return (
        f"mean={sum(s)/n:.3f}  min={s[0]:.3f}  max={s[-1]:.3f}  "
        f"non-zero={sum(1 for v in s if v>0)}/{len(vals)}"
    )

from collections import Counter
stage_counts = Counter(fail_stages)

print("\n" + "=" * 70)
print(f"  RL Reward Pipeline Report — refined-variant (mode={args.mode}, n={len(fmt_rewards)})")
print("=" * 70)
print(f"  format_reward  : {_stats(fmt_rewards)}")
print(f"  parse_reward   : {_stats(parse_rewards)}")
print(f"  scd_reward     : {_stats(scd_rewards)}")
print(f"  fail_stage dist: {dict(stage_counts)}")
print("=" * 70)

if args.mode == "gt":
    ok_count = stage_counts.get("ok", 0)
    fmt_nz = sum(1 for v in fmt_rewards if v > 0)
    if fmt_nz == 0:
        print("  VERDICT: FAIL — GT STEP files missing END-ISO-10303-21;")
        print("           Check that train.json 'output' field has complete STEP files.")
    elif ok_count == 0:
        print("  VERDICT: PARTIAL — format/parse may be OK but SCD pipeline not reaching 'ok'.")
        print(f"           Stages seen: {dict(stage_counts)}")
        print(f"           text2cad_src={text2cad_src}")
    else:
        print(f"  VERDICT: PASS — {ok_count}/{len(scd_rewards)} examples reached 'ok' stage.")
        print("           Reward pipeline is functional. RL will receive geometry signal.")
elif args.mode == "corrupt":
    scd_nz = sum(1 for v in scd_rewards if v > 0)
    if scd_nz == 0:
        print("  VERDICT: PASS — corrupted completions earn 0 reward (discriminative).")
    else:
        print(f"  VERDICT: WARNING — {scd_nz} corrupted completions earned non-zero reward.")
print("=" * 70)
print("\nTip: run --mode gt first to confirm pipeline fires, then --mode corrupt to confirm discrimination.")
