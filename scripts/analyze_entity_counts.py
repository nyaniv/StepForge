"""
Distribution of entity counts in training-set GT STEP files.

Used to pick the format_reward entity threshold so that legitimate
training-distribution outputs qualify while footer-only fragments (which
have ~10-15 entities) don't.

Reports the distribution two ways:
  1. All training examples
  2. Examples whose GT STEP fits in max_completion_length (the in-distribution
     subset RL actually trains on)

Usage:
    python scripts/analyze_entity_counts.py --config configs/config_gautschi.yaml
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from omegaconf import OmegaConf


def count_entities(step: str) -> int:
    return len(re.findall(r"^#\d+\s*=", step, re.MULTILINE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_gautschi.yaml")
    ap.add_argument("--char-per-token", type=float, default=3.5,
                    help="Heuristic char/token ratio for filtering "
                         "(avoids loading the tokenizer just for an analysis run)")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    train_json = os.path.join(cfg.paths.processed_dir, "train.json")
    max_comp = int(cfg.rl.max_completion_length)
    char_limit = int(max_comp * args.char_per_token)

    print(f"Loading {train_json}...")
    with open(train_json) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} training examples.\n")

    counts_all = []
    counts_in_dist = []
    for r in records:
        gt = r.get("output", "")
        n = count_entities(gt)
        counts_all.append(n)
        if len(gt) <= char_limit:
            counts_in_dist.append(n)

    def report(name, arr):
        if not arr:
            print(f"{name}: empty\n")
            return
        a = np.array(arr)
        print(f"=== {name} (N={len(a)}) ===")
        for p in (1, 5, 10, 25, 50, 75, 90, 99):
            print(f"  p{p:>2}: {np.percentile(a, p):>6.0f}")
        print(f"  mean: {a.mean():>6.1f}")
        print(f"  min/max: {a.min()} / {a.max()}")
        print()

    report("All training GT", counts_all)
    report(f"In-distribution GT (chars ≤ {char_limit} ≈ {max_comp} tokens)",
           counts_in_dist)

    print("Recommended thresholds:")
    a = np.array(counts_in_dist or counts_all)
    p5  = int(np.percentile(a, 5))
    p10 = int(np.percentile(a, 10))
    print(f"  Conservative (p5):  {p5}  — 95% of training outputs qualify")
    print(f"  Moderate     (p10): {p10}  — 90% of training outputs qualify")


if __name__ == "__main__":
    main()
