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
    # API-4: n_points=64 — cheap parse-check, matches _parse_worker.
    renderable = sum(
        step_to_pointcloud(o, n_points=64, text2cad_src=text2cad_src) is not None
        for o in tqdm(outputs, desc="RR")
    )
    return renderable / len(outputs)


def mscd(
    pred_steps: list[str],
    gt_steps: list[str],
    text2cad_src: str,
    rcfg: "RewardConfig | None" = None,
) -> dict[str, float]:
    """
    MSCD: Median Scaled Chamfer Distance (paper's primary metric).

    Reuses compute_reward's subprocess isolation so an OCP/Open3D segfault on
    a single sample doesn't kill the entire evaluation sweep.

    Returns both the paper-faithful MSCD (median over renderable pairs only)
    and an MSCD_penalized variant where parse failures count as delta_high
    (the worst possible score). The training reward penalizes parse failures
    with reward=0; the paper-faithful MSCD silently drops them. Reporting
    both makes the optimism gap explicit.
    """
    from reward.scd_reward import RewardConfig, compute_reward

    if rcfg is None:
        rcfg = RewardConfig()
    delta_high = rcfg.delta_high

    scds = []
    drops = {"pred_parse": 0, "gt": 0, "other": 0}
    for pred, gt in tqdm(zip(pred_steps, gt_steps), desc="MSCD", total=len(pred_steps)):
        _, raw_scd, stage, _ = compute_reward(
            pred, gt, rcfg=rcfg, text2cad_src=text2cad_src,
        )
        if stage == "ok":
            scds.append(raw_scd)
        elif stage in ("gt_parse", "gt_degenerate"):
            drops["gt"] += 1
        elif stage in ("no_terminator", "pred_parse", "pred_degenerate"):
            drops["pred_parse"] += 1
        else:
            drops["other"] += 1

    n_dropped = sum(drops.values())
    logger.info(f"MSCD computed over {len(scds)}/{len(pred_steps)} renderable pairs; "
                f"dropped: pred_parse={drops['pred_parse']} gt={drops['gt']} other={drops['other']}")

    mscd_paper = float(np.median(scds)) if scds else float("inf")
    penalized = scds + [delta_high] * drops["pred_parse"]
    mscd_penalized = float(np.median(penalized)) if penalized else float("inf")

    return {
        "mscd": mscd_paper,
        "mscd_penalized": mscd_penalized,
        "n_renderable": len(scds),
        "n_total": len(pred_steps),
        "n_dropped_pred": drops["pred_parse"],
        "n_dropped_gt": drops["gt"],
        "n_dropped_other": drops["other"],
    }


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
    test_json = os.path.join(cfg.paths.processed_dir, "test.json")
    logger.info(f"Loading test data from {test_json}")
    with open(test_json) as f:
        test_records = json.load(f)
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
            exclude_uid=record.get("id_original") or record.get("uid"),
        )
        generated.append(step_out)
        gt_steps.append(record.get("output") or record.get("step", ""))

    # Compute metrics
    logger.info("Computing metrics...")
    cr   = completion_rate(generated)
    rr   = renderability_rate(generated, cfg.paths.text2cad_src)
    # API-3: build the reward-shape config once from cfg.rl.reward.
    from reward.scd_reward import RewardConfig
    _defl = cfg.rl.reward.get("mesh_deflection", None)
    rcfg = RewardConfig(
        n_points=int(cfg.rl.reward.n_sample_points),
        delta_low=float(cfg.rl.reward.delta_low),
        delta_high=float(cfg.rl.reward.delta_high),
        bidirectional=bool(cfg.rl.reward.get("chamfer_bidirectional", True)),
        scale_prenorm=bool(cfg.rl.reward.get("scale_prenorm", True)),
        deflection=float(_defl) if _defl is not None else None,
    )
    m    = mscd(generated, gt_steps, cfg.paths.text2cad_src, rcfg=rcfg)
    aec  = average_entity_count(generated)

    mscd_val = m["mscd"]
    mscd_pen = m["mscd_penalized"]

    # Print results table
    print("\n" + "="*80)
    print(f"{'Method':<25} {'CR(%)':>8} {'RR(%)':>8} {'MSCD':>8} {'MSCD_pen':>10} {'AEC':>8}")
    print("-"*80)
    print(f"{'Ground Truth':<25} {'—':>8} {'—':>8} {'—':>8} {'—':>10} {'265.64':>8}")
    print(f"{'Text2CAD':<25} {'—':>8} {'98.38':>8} {'3.99':>8} {'—':>10} {'390.41':>8}")
    print(f"{'STEP-LLM (SFT)':<25} {'97.00':>8} {'95.18':>8} {'0.53':>8} {'—':>10} {'240.99':>8}")
    print(f"{'STEP-LLM (GRPO)':<25} {'99.00':>8} {'92.00':>8} {'0.098':>8} {'—':>10} {'—':>8}")
    print("-"*80)
    print(f"{'Our result':<25} {cr*100:>8.2f} {rr*100:>8.2f} {mscd_val:>8.4f} {mscd_pen:>10.4f} {aec:>8.2f}")
    print("="*80)
    print(f"  MSCD computed over {m['n_renderable']}/{m['n_total']} renderable pairs; "
          f"MSCD_pen treats {m['n_dropped_pred']} parse failures as worst-case (δ_high)\n")

    logger.info(f"CR={cr:.4f}  RR={rr:.4f}  MSCD={mscd_val:.6f}  "
                f"MSCD_penalized={mscd_pen:.6f}  AEC={aec:.2f}")


if __name__ == "__main__":
    main()
