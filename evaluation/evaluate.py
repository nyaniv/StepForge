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
        --checkpoint $SCRATCH/stepforge/runs/sft_4gpu_9281837/checkpoint-24650 \
        --config configs/config_gautschi.yaml

    python evaluation/evaluate.py \
        --checkpoint $SCRATCH/stepforge/runs/sft_4gpu_9281837/checkpoint-24650 \
        --config configs/config_gautschi.yaml \
        --max-examples 100   # quick eval on subset
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


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(checkpoint: str, cfg):
    scratch = os.environ.get("SCRATCH", "")
    hf_cache = os.path.join(scratch, ".hf-cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots")
    snapshots = os.listdir(hf_cache)
    base = os.path.join(hf_cache, snapshots[0])

    logger.info(f"Base model : {base}")
    logger.info(f"Checkpoint : {checkpoint}")

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, checkpoint)
    model.eval()
    return model, tok


# ── Generation ─────────────────────────────────────────────────────────────────

def generate_step(model, tok, caption: str, retrieved: str,
                  max_seq_length: int = 14336) -> str:
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


# ── Metric functions ───────────────────────────────────────────────────────────

def completion_rate(outputs: list) -> float:
    return sum("END-ISO-10303-21;" in o for o in outputs) / len(outputs)


def renderability_rate(outputs: list, text2cad_src: str) -> float:
    from reward.step_to_pointcloud import step_to_pointcloud
    renderable = sum(
        step_to_pointcloud(o, text2cad_src=text2cad_src) is not None
        for o in tqdm(outputs, desc="RR")
    )
    return renderable / len(outputs)


def mscd(pred_steps: list, gt_steps: list, text2cad_src: str,
         n_points: int = 2048) -> float:
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


def average_entity_count(outputs: list) -> float:
    counts = [
        len(re.findall(r"^#\d+\s*=", o, re.MULTILINE))
        for o in outputs
    ]
    return float(np.mean(counts))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate STEP-LLM on test set")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to HF/PEFT checkpoint directory")
    parser.add_argument("--config", default="configs/config_gautschi.yaml")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit evaluation to first N test examples (default: full test set)")
    parser.add_argument("--out-json", default=None,
                        help="Save raw generated outputs to this JSON file")
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
    model, tok = load_model(args.checkpoint, cfg)

    # Generate all outputs
    logger.info("Generating STEP files for test set...")
    generated = []
    gt_steps = []
    for record in tqdm(test_records, desc="Generating"):
        output = generate_step(
            model, tok,
            caption=record["caption"],
            retrieved=record["relavant_step_file"],
            max_seq_length=cfg.model.max_seq_length,
        )
        generated.append(output)
        gt_steps.append(record.get("output", ""))

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump([{"caption": r["caption"], "generated": g, "gt": gt}
                       for r, g, gt in zip(test_records, generated, gt_steps)], f)
        logger.info(f"Raw outputs saved to {args.out_json}")

    # Compute metrics
    logger.info("Computing CR and AEC...")
    cr  = completion_rate(generated)
    aec = average_entity_count(generated)

    logger.info("Computing RR (requires OpenCASCADE rendering)...")
    rr = renderability_rate(generated, cfg.paths.text2cad_src)

    logger.info("Computing MSCD (point cloud sampling)...")
    m = mscd(generated, gt_steps, cfg.paths.text2cad_src,
             n_points=cfg.rl.reward.n_sample_points)

    # Print results table
    print("\n" + "="*65)
    print(f"{'Method':<25} {'CR(%)':>8} {'RR(%)':>8} {'MSCD':>8} {'AEC':>8}")
    print("-"*65)
    print(f"{'Ground Truth':<25} {'—':>8} {'—':>8} {'—':>8} {'265.64':>8}")
    print(f"{'Text2CAD':<25} {'—':>8} {'98.38':>8} {'3.99':>8} {'390.41':>8}")
    print(f"{'STEP-LLM (SFT)':<25} {'97.00':>8} {'95.18':>8} {'0.53':>8} {'240.99':>8}")
    print(f"{'STEP-LLM (GRPO)':<25} {'99.00':>8} {'92.00':>8} {'0.098':>8} {'—':>8}")
    print("-"*65)
    print(f"{'Our SFT result':<25} {cr*100:>8.2f} {rr*100:>8.2f} {m:>8.4f} {aec:>8.2f}")
    print("="*65 + "\n")

    logger.info(f"CR={cr:.4f}  RR={rr:.4f}  MSCD={m:.6f}  AEC={aec:.2f}")


if __name__ == "__main__":
    main()
