"""
In-distribution evaluation of a trained STEP-LLM checkpoint.

Mirrors the truncation/filter rules used at training time so the eval
matches the conditions the model was actually trained on:

  1. Retrieval is truncated to MAX_RETRIEVED_TOKENS (same formula as
     rl_train.py: max_seq_length - max_completion_length - 300).
  2. (Optional, default ON) test examples whose GT STEP exceeds
     max_completion_length are flagged as out-of-distribution.

Reports TWO metric sets in one run:
  - "all (truncated)"      — every test example, retrieval truncated
  - "in_dist"              — only examples whose GT fits training cap

The pair of numbers exposes the long-tail effect cleanly: the gap between
the two columns is the cost of out-of-distribution test examples.

The original evaluate.py is left untouched as the "no truncation applied"
baseline.

Usage:
    python evaluation/evaluate_in_distribution.py \\
        --checkpoint $SCRATCH/stepforge/checkpoints/rl-refined/final \\
        --config configs/config_gautschi.yaml \\
        --max-examples 100 \\
        --out-json $SCRATCH/stepforge/eval_in_dist.json
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "You are a CAD model generation assistant trained to produce STEP (.step) files "
    "based on textual descriptions. Given the following object description and relevant "
    "retrieved CAD data, generate a STEP file that accurately represents the described object."
)


def load_model(checkpoint: str):
    scratch = os.environ.get("SCRATCH", "")
    hf_cache = os.path.join(
        scratch, ".hf-cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots")
    snapshots = os.listdir(hf_cache)
    base = os.path.join(hf_cache, snapshots[0])

    logger.info(f"Base model : {base}")
    logger.info(f"Checkpoint : {checkpoint}")

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, checkpoint)
    model.eval()
    return model, tok


def generate_step(model, tok, caption: str, retrieved: str,
                  max_seq_length: int, max_retrieved_tokens: int) -> str:
    """
    Truncates retrieval to max_retrieved_tokens (same logic as rl_train.format_prompt)
    before building the prompt, so eval-time conditions match training-time.
    """
    # Match training-time retrieval truncation
    ret_ids = tok(retrieved, add_special_tokens=False)["input_ids"]
    if len(ret_ids) > max_retrieved_tokens:
        retrieved = tok.decode(ret_ids[:max_retrieved_tokens],
                               skip_special_tokens=True)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"### caption:\n{caption}\n\n"
        f"### retrieved relevant step file:\n{retrieved}"
    )
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    prompt_len = ids.input_ids.shape[1]
    max_new_tokens = max(256, max_seq_length - prompt_len - 64)

    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )
    return tok.decode(out[0][prompt_len:], skip_special_tokens=True)


def completion_rate(outputs):
    return sum("END-ISO-10303-21;" in o for o in outputs) / len(outputs)


def renderability_rate(outputs, text2cad_src):
    from reward.step_to_pointcloud import step_to_pointcloud
    return sum(
        step_to_pointcloud(o, text2cad_src=text2cad_src) is not None
        for o in tqdm(outputs, desc="RR")
    ) / len(outputs)


def mscd(pred_steps, gt_steps, text2cad_src, n_points=2048):
    from reward.step_to_pointcloud import step_to_pointcloud
    from reward.scd_reward import scaled_chamfer_distance
    scds = []
    for pred, gt in tqdm(zip(pred_steps, gt_steps), desc="MSCD",
                         total=len(pred_steps)):
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
    return float(np.median(scds)) if scds else float("inf")


def average_entity_count(outputs):
    return float(np.mean([
        len(re.findall(r"^#\d+\s*=", o, re.MULTILINE)) for o in outputs
    ]))


def compute_metrics(generated, gt_steps, text2cad_src, n_points):
    return {
        "CR":   completion_rate(generated),
        "RR":   renderability_rate(generated, text2cad_src),
        "MSCD": mscd(generated, gt_steps, text2cad_src, n_points=n_points),
        "AEC":  average_entity_count(generated),
        "N":    len(generated),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate STEP-LLM on test set with training-time truncation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/config_gautschi.yaml")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--out-json", default=None,
                        help="Save raw generated outputs + per-example metadata")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    max_seq  = int(cfg.model.max_seq_length)
    max_comp = int(cfg.rl.max_completion_length)
    overhead = 300
    max_retrieved_tokens = max_seq - max_comp - overhead

    logger.info(f"max_seq_length={max_seq}, max_completion_length={max_comp}, "
                f"max_retrieved_tokens={max_retrieved_tokens}")

    # Load test set
    test_json = os.path.join(cfg.paths.processed_dir, "test.json")
    with open(test_json) as f:
        test_records = json.load(f)
    if args.max_examples:
        test_records = test_records[:args.max_examples]
    logger.info(f"Evaluating on {len(test_records)} test examples")

    # Load model + tokenizer
    model, tok = load_model(args.checkpoint)

    # Precompute GT token lengths to flag out-of-distribution examples
    logger.info("Tokenizing GT to flag in-distribution subset...")
    is_in_dist = []
    for r in tqdm(test_records, desc="GT-tok"):
        gt = r.get("output", "")
        n_tok = len(tok(gt, add_special_tokens=False)["input_ids"])
        is_in_dist.append(n_tok <= max_comp)
    n_in = sum(is_in_dist)
    logger.info(f"In-distribution: {n_in}/{len(test_records)} "
                f"(GT ≤ {max_comp} tokens)")

    # Generate
    generated, gt_steps = [], []
    for record in tqdm(test_records, desc="Generating"):
        output = generate_step(
            model, tok,
            caption=record["caption"],
            retrieved=record["relavant_step_file"],
            max_seq_length=max_seq,
            max_retrieved_tokens=max_retrieved_tokens,
        )
        stripped = output.lstrip()
        if not stripped.startswith("DATA;") and not stripped.startswith("ISO-10303-21;"):
            output = "DATA;\n" + stripped
        generated.append(output)
        gt_steps.append(record.get("output", ""))

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump([
                {"caption": r["caption"], "generated": g, "gt": gt,
                 "in_dist": bool(in_d)}
                for r, g, gt, in_d in zip(test_records, generated, gt_steps, is_in_dist)
            ], f)
        logger.info(f"Raw outputs saved to {args.out_json}")

    # Compute metrics — overall + in-distribution subset
    logger.info("\nComputing metrics on ALL examples (retrieval truncated)...")
    all_metrics = compute_metrics(generated, gt_steps, cfg.paths.text2cad_src,
                                  n_points=cfg.rl.reward.n_sample_points)

    in_gen = [g for g, in_d in zip(generated, is_in_dist) if in_d]
    in_gt  = [g for g, in_d in zip(gt_steps,  is_in_dist) if in_d]
    logger.info(f"\nComputing metrics on IN-DISTRIBUTION subset ({len(in_gen)})...")
    in_metrics = compute_metrics(in_gen, in_gt, cfg.paths.text2cad_src,
                                 n_points=cfg.rl.reward.n_sample_points) \
                 if in_gen else {"CR": 0, "RR": 0, "MSCD": float("inf"),
                                 "AEC": 0, "N": 0}

    # Print results table
    print("\n" + "="*78)
    print(f"{'Subset':<30} {'N':>5} {'CR(%)':>8} {'RR(%)':>8} {'MSCD':>10} {'AEC':>8}")
    print("-"*78)
    print(f"{'Paper STEP-LLM (SFT)':<30} {'—':>5} {'97.00':>8} {'95.18':>8} {'0.5300':>10} {'240.99':>8}")
    print(f"{'Paper STEP-LLM (GRPO)':<30} {'—':>5} {'99.00':>8} {'92.00':>8} {'0.0980':>10} {'—':>8}")
    print("-"*78)
    print(f"{'Yours: all (truncated)':<30} {all_metrics['N']:>5} "
          f"{all_metrics['CR']*100:>8.2f} {all_metrics['RR']*100:>8.2f} "
          f"{all_metrics['MSCD']:>10.4f} {all_metrics['AEC']:>8.2f}")
    print(f"{'Yours: in-distribution only':<30} {in_metrics['N']:>5} "
          f"{in_metrics['CR']*100:>8.2f} {in_metrics['RR']*100:>8.2f} "
          f"{in_metrics['MSCD']:>10.4f} {in_metrics['AEC']:>8.2f}")
    print("="*78 + "\n")

    logger.info(f"all:     CR={all_metrics['CR']:.4f} RR={all_metrics['RR']:.4f} "
                f"MSCD={all_metrics['MSCD']:.6f} AEC={all_metrics['AEC']:.2f}")
    logger.info(f"in_dist: CR={in_metrics['CR']:.4f} RR={in_metrics['RR']:.4f} "
                f"MSCD={in_metrics['MSCD']:.6f} AEC={in_metrics['AEC']:.2f}")


if __name__ == "__main__":
    main()
