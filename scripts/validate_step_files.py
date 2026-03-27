"""
Validate all raw STEP files in data/step_files/ without OCC.

Uses step_parser.py (regex-based) to check structural validity and entity
counts for all files. Reports:
  - Parse success / failure rate
  - Entity count distribution
  - How many will survive the >=500-entity filter
  - List of parse failures (if any)

Runs in parallel — ~2-3 minutes for 174k files.

Usage:
    python scripts/validate_step_files.py
    python scripts/validate_step_files.py --workers 4
    python scripts/validate_step_files.py --limit 1000   # quick sample
"""

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.step_parser import parse_step


def check_one(path: str) -> dict:
    """Parse one STEP file, return stats dict."""
    try:
        _, entities, _ = parse_step(path)
        n = len(entities)
        return {"path": path, "ok": True, "entities": n, "error": None}
    except Exception as e:
        return {"path": path, "ok": False, "entities": 0, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Validate raw STEP files without OCC")
    parser.add_argument("--dir",     default="data/step_files",
                        help="Directory of raw STEP files (default: data/step_files)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit",   type=int, default=None,
                        help="Process only first N files (for quick testing)")
    parser.add_argument("--max-entities", type=int, default=500,
                        help="Filter threshold — files >= this are excluded (default 500)")
    parser.add_argument("--show-failures", action="store_true",
                        help="Print paths of all failed files")
    args = parser.parse_args()

    step_dir = Path(args.dir)
    if not step_dir.exists():
        print(f"ERROR: {step_dir} does not exist")
        sys.exit(1)

    files = sorted(step_dir.glob("*.step"))
    if not files:
        print(f"ERROR: no .step files found in {step_dir}")
        sys.exit(1)

    if args.limit:
        files = files[:args.limit]

    total = len(files)
    print(f"Validating {total:,} STEP files with {args.workers} workers...")
    print(f"Filter threshold: {args.max_entities} entities\n")

    paths = [str(f) for f in files]
    results = []

    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(check_one, paths, chunksize=100), 1):
            results.append(r)
            if i % 10000 == 0:
                print(f"  {i:>7,} / {total:,} processed...")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    ok       = [r for r in results if r["ok"]]
    failed   = [r for r in results if not r["ok"]]
    counts   = sorted(r["entities"] for r in ok)

    will_pass   = [r for r in ok if r["entities"] < args.max_entities]
    will_filter = [r for r in ok if r["entities"] >= args.max_entities]
    empty       = [r for r in ok if r["entities"] == 0]

    def pct(n): return f"{n/total*100:.1f}%"

    print(f"\n{'='*60}")
    print(f"  STEP FILE VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"  Total files           : {total:>10,}")
    print(f"  Parse success         : {len(ok):>10,}  ({pct(len(ok))})")
    print(f"  Parse failure         : {len(failed):>10,}  ({pct(len(failed))})")
    print(f"  Empty (0 entities)    : {len(empty):>10,}  ({pct(len(empty))})")
    print()
    print(f"  Entity filter (<{args.max_entities})")
    print(f"  Will pass filter      : {len(will_pass):>10,}  ({pct(len(will_pass))})")
    print(f"  Will be filtered out  : {len(will_filter):>10,}  ({pct(len(will_filter))})")
    print()

    if counts:
        n = len(counts)
        print(f"  Entity count distribution (parsed files):")
        print(f"    min        : {counts[0]}")
        print(f"    p25        : {counts[n // 4]}")
        print(f"    median     : {counts[n // 2]}")
        print(f"    p75        : {counts[3 * n // 4]}")
        print(f"    p90        : {counts[int(n * 0.9)]}")
        print(f"    p99        : {counts[int(n * 0.99)]}")
        print(f"    max        : {counts[-1]}")

    print(f"{'='*60}\n")

    if failed and args.show_failures:
        print("PARSE FAILURES:")
        for r in sorted(failed, key=lambda x: x["path"]):
            print(f"  {r['path']}  →  {r['error']}")
        print()

    if failed and not args.show_failures:
        print(f"  (Run with --show-failures to list all {len(failed)} failed files)")

    # Summary verdict
    if len(failed) == 0:
        print("✓ All files parsed successfully.")
    elif len(failed) / total < 0.01:
        print(f"~ {len(failed)} parse failures ({pct(len(failed))}) — negligible, pipeline will skip these.")
    else:
        print(f"✗ {len(failed)} parse failures ({pct(len(failed))}) — investigate before full run.")

    print(f"\n  Estimated dataset size after filter: ~{len(will_pass):,} examples")


if __name__ == "__main__":
    main()
