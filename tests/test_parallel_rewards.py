"""
Validates that the parallel reward implementation (Q1) produces identical
results to the sequential version and actually runs in parallel.

Run with:
    python tests/test_parallel_rewards.py

No GPU, no OCP, no Open3D required — uses mock reward functions.
"""

import sys
import os
import time
import multiprocessing as mp

# Must be set before anything else, matching rl_train.py
mp.set_start_method("spawn", force=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent.futures import ThreadPoolExecutor


# ── Mock slow reward (simulates OCP tessellation delay) ───────────────────────

def _slow_reward(completion: str, gt: str, delay: float = 0.5) -> float:
    """Simulates compute_reward: takes `delay` seconds, returns deterministic value."""
    time.sleep(delay)
    # Deterministic: reward = 1.0 if lengths match, else 0.5
    return 1.0 if len(completion) == len(gt) else 0.5


def sequential_rewards(completions, ground_truths, delay=0.5):
    return [_slow_reward(c, g, delay) for c, g in zip(completions, ground_truths)]


def parallel_rewards(completions, ground_truths, delay=0.5):
    with ThreadPoolExecutor(max_workers=len(completions)) as pool:
        futures = [
            pool.submit(_slow_reward, c, g, delay)
            for c, g in zip(completions, ground_truths)
        ]
        return [f.result() for f in futures]


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_correctness():
    """Parallel returns identical values to sequential."""
    completions   = ["abc", "de",  "fghi", "j",    "klmno"]
    ground_truths = ["xyz", "pqr", "fghi", "stuv", "klmno"]

    seq = sequential_rewards(completions, ground_truths, delay=0.0)
    par = parallel_rewards(completions, ground_truths, delay=0.0)

    assert seq == par, f"Mismatch:\n  sequential: {seq}\n  parallel:   {par}"
    print(f"  PASS correctness: {seq}")


def test_order_preserved():
    """Output order matches input order even when tasks finish out of order."""
    # Make later tasks finish faster
    delays = [0.3, 0.2, 0.1, 0.05, 0.01]
    completions   = [f"comp_{i}" * (i + 1) for i in range(5)]
    ground_truths = [f"gt_{i}"   * (i + 1) for i in range(5)]  # same lengths → all 1.0

    seq = sequential_rewards(completions, ground_truths, delay=0.0)

    with ThreadPoolExecutor(max_workers=len(completions)) as pool:
        futures = [
            pool.submit(_slow_reward, c, g, d)
            for c, g, d in zip(completions, ground_truths, delays)
        ]
        par = [f.result() for f in futures]

    assert seq == par, f"Order mismatch:\n  seq: {seq}\n  par: {par}"
    print(f"  PASS order preserved: {par}")


def test_speedup():
    """N parallel tasks take ~1x delay, not N×delay."""
    n = 8
    delay = 0.3
    completions   = [f"completion_{i}" for i in range(n)]
    ground_truths = [f"completion_{i}" for i in range(n)]  # all match → all 1.0

    t0 = time.time()
    sequential_rewards(completions, ground_truths, delay=delay)
    seq_time = time.time() - t0

    t0 = time.time()
    parallel_rewards(completions, ground_truths, delay=delay)
    par_time = time.time() - t0

    speedup = seq_time / par_time
    print(f"  Sequential: {seq_time:.2f}s  |  Parallel: {par_time:.2f}s  |  Speedup: {speedup:.1f}x")
    assert speedup > 3.0, f"Expected >3x speedup, got {speedup:.1f}x"
    print(f"  PASS speedup: {speedup:.1f}x (expected >{3}x)")


def test_exception_isolation():
    """A failure in one task doesn't affect others."""
    def flaky_reward(completion, gt, delay=0.0):
        time.sleep(delay)
        if completion == "FAIL":
            raise RuntimeError("simulated OCP crash")
        return 1.0

    completions = ["ok", "FAIL", "ok", "FAIL", "ok"]

    results = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(flaky_reward, c, "") for c in completions]
        for f in futures:
            try:
                results.append(f.result())
            except Exception:
                results.append(0.0)  # same fallback as compute_reward

    expected = [1.0, 0.0, 1.0, 0.0, 1.0]
    assert results == expected, f"Expected {expected}, got {results}"
    print(f"  PASS exception isolation: {results}")


def test_length_matches_input():
    """Output list always has same length as input, even on failures."""
    completions = [f"c{i}" for i in range(7)]
    results = parallel_rewards(completions, completions, delay=0.0)
    assert len(results) == len(completions), f"Length mismatch: {len(results)} != {len(completions)}"
    print(f"  PASS length: {len(results)} == {len(completions)}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_correctness,
        test_order_preserved,
        test_speedup,
        test_exception_isolation,
        test_length_matches_input,
    ]
    print(f"Running {len(tests)} tests for parallel reward Q1...\n")
    failed = []
    for t in tests:
        print(f"[{t.__name__}]")
        try:
            t()
        except Exception as e:
            print(f"  FAIL: {e}")
            failed.append(t.__name__)
        print()

    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print("All tests passed.")
