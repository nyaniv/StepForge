"""
Local reward pipeline debugger — runs entirely on CPU (Mac).

Tests:
  1. GT identity test: compute_reward(gt, gt) on real STEP files → expect ~1.0
  2. Truncated STEP: simulate model output by cutting GT at various token counts
  3. Terminated-but-invalid: GT with shuffled entity references

Usage (in step_llm conda env):
    conda run -n step_llm python scripts/debug_reward.py
    conda run -n step_llm python scripts/debug_reward.py --test identity
    conda run -n step_llm python scripts/debug_reward.py --test truncated
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glob import glob
from reward.step_to_pointcloud import step_to_pointcloud
from reward.scd_reward import compute_reward


STEP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "step_files")


def test_identity(n=10):
    """Test that compute_reward(gt, gt) returns ~1.0 for real STEP files."""
    print("\n" + "="*60)
    print("TEST 1: GT Identity (same file as pred and GT)")
    print("="*60)

    files = sorted(glob(os.path.join(STEP_DIR, "*.step")))[:n]
    if not files:
        print(f"ERROR: No STEP files found in {STEP_DIR}")
        return

    passed = 0
    for path in files:
        with open(path) as f:
            content = f.read()
        name = os.path.basename(path)

        print(f"\n--- {name} ---")
        pc = step_to_pointcloud(content, n_points=64, verbose=True)
        if pc is None:
            print(f"  FAIL: step_to_pointcloud returned None")
            continue

        reward = compute_reward(content, content, n_points=256, verbose=True)
        status = "PASS" if reward > 0.5 else "LOW"
        print(f"  Identity reward: {reward:.4f} [{status}]")
        if reward > 0.5:
            passed += 1

    print(f"\nIdentity test: {passed}/{len(files)} files returned reward > 0.5")


def test_truncated(n=3):
    """Test pipeline on truncated STEP content (simulates model output)."""
    print("\n" + "="*60)
    print("TEST 2: Truncated STEP (simulates model output without terminator)")
    print("="*60)

    files = sorted(glob(os.path.join(STEP_DIR, "*.step")))[:n]
    for path in files:
        with open(path) as f:
            content = f.read()
        name = os.path.basename(path)
        chars = len(content)

        print(f"\n--- {name} ({chars} chars) ---")
        for frac, label in [(0.25, "25%"), (0.50, "50%"), (0.75, "75%")]:
            truncated = content[:int(chars * frac)]
            # Add terminator so it passes the fast-path check
            truncated_with_end = truncated + "\nEND-ISO-10303-21;"
            pc = step_to_pointcloud(truncated_with_end, n_points=64, verbose=True)
            status = "parsed" if pc is not None else "FAILED"
            print(f"  Truncated at {label}: step_to_pointcloud → {status}")


def test_pointcloud_stats(n=20):
    """Report point cloud stats for GT files to check for degenerate cases."""
    print("\n" + "="*60)
    print("TEST 3: Point cloud stats for GT files")
    print("="*60)

    files = sorted(glob(os.path.join(STEP_DIR, "*.step")))[:n]
    results = []
    for path in files:
        with open(path) as f:
            content = f.read()
        pc = step_to_pointcloud(content, n_points=2048, verbose=False)
        if pc is None:
            results.append((os.path.basename(path), None, None))
        else:
            import numpy as np
            unique = len(np.unique(pc, axis=0))
            scale = float(np.sqrt(np.mean(np.sum((pc - pc.mean(0))**2, axis=1))))
            results.append((os.path.basename(path), unique, scale))

    parsed = [(n, u, s) for n, u, s in results if u is not None]
    failed = [n for n, u, s in results if u is None]

    print(f"\nParsed: {len(parsed)}/{len(results)}")
    if failed:
        print(f"Failed: {failed}")
    if parsed:
        import numpy as np
        uniques = [u for _, u, _ in parsed]
        scales  = [s for _, _, s in parsed]
        print(f"Unique points — min: {min(uniques)}, median: {int(np.median(uniques))}, max: {max(uniques)}")
        print(f"Scale factor  — min: {min(scales):.4f}, median: {np.median(scales):.4f}, max: {max(scales):.4f}")
        low_unique = [(n, u) for n, u, _ in parsed if u < 50]
        if low_unique:
            print(f"WARNING: {len(low_unique)} files have < 50 unique points: {low_unique}")
        else:
            print("All parsed files have >= 50 unique points ✓")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["identity", "truncated", "stats", "all"],
                        default="all")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    if args.test in ("identity", "all"):
        test_identity(n=args.n)
    if args.test in ("truncated", "all"):
        test_truncated(n=3)
    if args.test in ("stats", "all"):
        test_pointcloud_stats(n=args.n)


if __name__ == "__main__":
    main()
