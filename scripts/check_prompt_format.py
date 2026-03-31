"""
Debug check: Training Prompt box (Stage 2, Step 1)

Run this after data/data_split.py has produced train.json.
Verifies that every training example will be masked correctly by
train_on_responses_only before touching the model.

Checks:
  1. Required fields exist in every record
  2. Response marker "### output:\n" is present in every formatted prompt
     (if missing, train_on_responses_only silently trains on the full sequence)
  3. Output (GT STEP) is non-empty
  4. Prints one full formatted prompt so you can eyeball it

Usage:
    python scripts/check_prompt_format.py --config configs/config_runpod.yaml
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from omegaconf import OmegaConf

INSTRUCTION_PART = "### caption:\n"
RESPONSE_PART    = "### output:\n"

ABC_PROMPT_RAG = (
    "You are a CAD model generation assistant trained to produce STEP (.step) files "
    "based on textual descriptions. Given the following object description and relevant "
    "retrieved CAD data, generate a STEP file that accurately represents the described object."
    "\n\n\n### caption:\n{}\n\n### retrieved relevant step file:\n{}\n\n### output:\n"
)


def format_prompt(rec: dict) -> str:
    return ABC_PROMPT_RAG.format(
        rec["caption"],
        rec.get("relavant_step_file", ""),
        rec["output"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg       = OmegaConf.load(args.config)
    train_path = os.path.join(cfg.paths.processed_dir, "train.json")

    if not os.path.exists(train_path):
        logger.error(f"train.json not found at {train_path} — run data/data_split.py first")
        sys.exit(1)

    with open(train_path) as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} training records from {train_path}")

    # ── Check 1: required fields ──────────────────────────────────────────────
    required = {"caption", "output", "relavant_step_file"}
    missing_fields = [i for i, r in enumerate(data) if not required.issubset(r.keys())]
    if missing_fields:
        logger.error(f"FAIL: {len(missing_fields)} records missing required fields: {missing_fields[:5]}")
    else:
        logger.info(f"PASS: All records have required fields (caption, output, relavant_step_file)")

    # ── Check 2: response marker present in every formatted prompt ────────────
    missing_marker = []
    for i, rec in enumerate(data):
        text = format_prompt(rec)
        if RESPONSE_PART not in text:
            missing_marker.append(i)

    if missing_marker:
        logger.error(f"FAIL: {len(missing_marker)} records missing '{RESPONSE_PART}' marker")
        logger.error(f"      train_on_responses_only will NOT mask these correctly")
        logger.error(f"      First bad indices: {missing_marker[:5]}")
    else:
        logger.info(f"PASS: All {len(data)} records contain the response marker")

    # ── Check 3: non-empty GT STEP output ────────────────────────────────────
    empty_output = [i for i, r in enumerate(data) if not r.get("output", "").strip()]
    if empty_output:
        logger.warning(f"WARN: {len(empty_output)} records have empty GT STEP output")
    else:
        logger.info(f"PASS: All records have non-empty GT STEP output")

    # ── Check 4: non-empty retrieved STEP ────────────────────────────────────
    empty_retrieved = [i for i, r in enumerate(data) if not r.get("relavant_step_file", "").strip()]
    if empty_retrieved:
        logger.warning(f"WARN: {len(empty_retrieved)} records have empty retrieved STEP "
                       f"({len(empty_retrieved)/len(data):.1%}) — these will train without RAG context")
    else:
        logger.info(f"PASS: All records have non-empty retrieved STEP")

    # ── Check 5: print one full formatted prompt to eyeball ──────────────────
    rec   = data[0]
    text  = format_prompt(rec)
    lines = text.splitlines()

    print(f"\n{'='*70}")
    print("SAMPLE FORMATTED PROMPT (first record)")
    print(f"{'='*70}")
    # Print first 30 lines and last 10 lines
    for line in lines[:30]:
        print(line)
    if len(lines) > 40:
        print(f"\n  ... ({len(lines) - 40} lines omitted) ...\n")
        for line in lines[-10:]:
            print(line)
    print(f"{'='*70}")

    # Show where the mask boundary is
    marker_pos = text.find(RESPONSE_PART)
    prompt_only = text[:marker_pos + len(RESPONSE_PART)]
    prompt_tokens_approx = len(prompt_only.split())
    output_tokens_approx = len(text[marker_pos + len(RESPONSE_PART):].split())
    print(f"\nMask boundary at char {marker_pos} ('{RESPONSE_PART.strip()}')")
    print(f"Approx prompt tokens : {prompt_tokens_approx} (masked — no loss)")
    print(f"Approx output tokens : {output_tokens_approx} (loss computed here)")
    print(f"{'='*70}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    all_passed = not missing_fields and not missing_marker and not empty_output
    if all_passed:
        logger.info("All checks passed — prompt format is correct, safe to start SFT")
    else:
        logger.error("One or more checks failed — fix before starting SFT")
        sys.exit(1)


if __name__ == "__main__":
    main()
