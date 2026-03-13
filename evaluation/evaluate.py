"""
Evaluate a trained STEP-LLM checkpoint on the test set.

Computes the four metrics from the paper:
  CR   — Completion Rate:    % of generated files containing END-ISO-10303-21;
  RR   — Renderability Rate: % of files OpenCASCADE can parse into a valid mesh
  MSCD — Median Scaled Chamfer Distance (primary geometric fidelity metric)
  AEC  — Average Entity Count

Prints a results table matching the paper format (Tables 1, 2, 4).

Expected baseline results from the paper:
  Text2CAD:        RR=98.38, MSCD=3.99,  AEC=390.41
  STEP-LLM (SFT):  CR=97.00, RR=95.18, MSCD=0.53,  AEC=240.99
  STEP-LLM (GRPO): CR=99.00, RR=92.00, MSCD=0.098

Usage:
    python evaluation/evaluate.py \
        --checkpoint checkpoints/rl/final \
        --config configs/config.yaml

    python evaluation/evaluate.py \
        --checkpoint checkpoints/sft/final \
        --config configs/config.yaml \
        --max-examples 200   # quick eval on subset
"""

import argparse
import json
import os
import re
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm


# ── Metric functions ───────────────────────────────────────────────────────────

def completion_rate(outputs: list[str]) -> float:
    """CR: % of generated files that contain END-ISO-10303-21;"""
    return sum("END-ISO-10303-21;" in o for o in outputs) / len(outputs)


def renderability_rate(outputs: list[str], text2cad_src: str) -> float:
    """RR: % of files OpenCASCADE can parse into a non-null meshable shape."""
    from reward.step_to_pointcloud import step_to_pointcloud
    renderable = sum(
        step_to_pointcloud(o, text2cad_src=text2cad_src) is not None
        for o in tqdm(outputs, desc="RR")
    )
    return renderable / len(outputs)


def mscd(
    pred_steps: list[str],
    gt_steps: list[str],
    text2cad_src: str,
    n_points: int = 2048,
) -> float:
    """MSCD: Median Scaled Chamfer Distance (paper's primary metric)."""
    from reward.step_to_pointcloud import step_to_pointcloud
    from reward.scd_reward import scaled_chamfer_distance

    scds = []
    for pred, gt in tqdm(zip(pred_steps, gt_steps), desc="MSCD", total=len(pred_steps)):
        pred_pc = step_to_pointcloud(pred, n_points=n_points, text2cad_src=text2cad_src)
        gt_pc   = step_to_pointcloud(gt,   n_points=n_points, text2cad_src=text2cad_src)
        if pred_pc is None or gt_pc is None:
            continue
        try:
            s = scaled_chamfer_distance(pred_pc, gt_pc)
            if np.isfinite(s):
                scds.append(s)
        except Exception:
            continue

    if not scds:
        return float("inf")
    return float(np.median(scds))


def average_entity_count(outputs: list[str]) -> float:
    """AEC: mean number of entities in generated files (paper Table 2)."""
    counts = [
        len(re.findall(r"^#\d+\s*=", o, re.MULTILINE))
        for o in outputs
    ]
    return float(np.mean(counts))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate STEP-LLM on test set")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit evaluation to first N test examples")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # Load test set
    test_jsonl = os.path.join(cfg.paths.processed_dir, "test.jsonl")
    logger.info(f"Loading test data from {test_jsonl}")
    test_records = [json.loads(l) for l in open(test_jsonl)]
    if args.max_examples:
        test_records = test_records[:args.max_examples]
        logger.info(f"Evaluating on {len(test_records)} examples (--max-examples)")
    else:
        logger.info(f"Evaluating on {len(test_records)} test examples")

    # Load model
    logger.info(f"Loading checkpoint from {args.checkpoint}...")
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.checkpoint,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load retriever
    from retrieval.retriever import Retriever
    retriever = Retriever(
        index_path=cfg.paths.faiss_index_path,
        metadata_path=cfg.paths.faiss_metadata_path,
        model_name=cfg.retrieval.model,
    )

    from inference.generate import generate_step

    # Generate all outputs
    logger.info("Generating STEP files for test set...")
    generated = []
    gt_steps = []
    for record in tqdm(test_records, desc="Generating"):
        step_out = generate_step(
            caption=record["caption"],
            model=model,
            tokenizer=tokenizer,
            retriever=retriever,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        generated.append(step_out)
        gt_steps.append(record["step"])

    # Compute metrics
    logger.info("Computing metrics...")
    cr   = completion_rate(generated)
    rr   = renderability_rate(generated, cfg.paths.text2cad_src)
    m    = mscd(generated, gt_steps, cfg.paths.text2cad_src,
                n_points=cfg.rl.reward.n_sample_points)
    aec  = average_entity_count(generated)

    # Print results table
    print("\n" + "="*65)
    print(f"{'Method':<25} {'CR(%)':>8} {'RR(%)':>8} {'MSCD':>8} {'AEC':>8}")
    print("-"*65)
    print(f"{'Ground Truth':<25} {'—':>8} {'—':>8} {'—':>8} {'265.64':>8}")
    print(f"{'Text2CAD':<25} {'—':>8} {'98.38':>8} {'3.99':>8} {'390.41':>8}")
    print(f"{'STEP-LLM (SFT)':<25} {'97.00':>8} {'95.18':>8} {'0.53':>8} {'240.99':>8}")
    print(f"{'STEP-LLM (GRPO)':<25} {'99.00':>8} {'92.00':>8} {'0.098':>8} {'—':>8}")
    print("-"*65)
    print(f"{'Our result':<25} {cr*100:>8.2f} {rr*100:>8.2f} {m:>8.4f} {aec:>8.2f}")
    print("="*65 + "\n")

    logger.info(f"CR={cr:.4f}  RR={rr:.4f}  MSCD={m:.6f}  AEC={aec:.2f}")


if __name__ == "__main__":
    main()
