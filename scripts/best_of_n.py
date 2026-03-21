"""
Best-of-N rejection sampling for StepForge.

For each prompt, generate N completions at temperature T, score each with the
existing parse + SCD reward pipeline, and keep the best-scoring completion.

Output JSONL can be used directly for a rejection-sampling SFT round.

Usage:
    python scripts/best_of_n.py --config configs/config_runpod.yaml
    python scripts/best_of_n.py --config configs/config_runpod.yaml --n 2 --max-examples 3  # quick test
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "HF_HOME" not in os.environ:
    _vol = os.environ.get("VOLUME", "/runpod-volume")
    os.environ["HF_HOME"] = os.path.join(_vol, ".hf-cache")

import torch
from loguru import logger
from omegaconf import OmegaConf
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from retrieval.retriever import Retriever
from reward.scd_reward import compute_parse_reward, compute_reward

SYSTEM_MSG = (
    "Given the object description and relevant CAD data, "
    "generate the corresponding STEP file."
)


def format_prompt(caption: str, retrieved_step: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_MSG}\n"
        f"<|user|>\n"
        f"caption: {caption}\n"
        f"retrieved step file:\n{retrieved_step}\n"
        f"<|assistant|>\n"
    )


def generate_completions(prompt: str, model, tokenizer,
                         n: int, temperature: float,
                         max_new_tokens: int = 4096,
                         batch_size: int = 4) -> list[str]:
    """Generate n completions in batches to avoid OOM."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    completions = []

    remaining = n
    while remaining > 0:
        bs = min(batch_size, remaining)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=bs,
                pad_token_id=tokenizer.eos_token_id,
            )
        for seq in output_ids:
            completion_ids = seq[prompt_len:]
            completions.append(
                tokenizer.decode(completion_ids, skip_special_tokens=True)
            )
        remaining -= bs

    return completions


def score_completions(completions: list[str], gt_step: str,
                      text2cad_src: str) -> list[float]:
    scores = []
    for c in completions:
        parse = compute_parse_reward(c, text2cad_src=text2cad_src)
        scd = compute_reward(c, gt_step, text2cad_src=text2cad_src) if parse > 0 else 0.0
        scores.append(parse + scd)
    return scores


def load_done_uids(output_path: str) -> set:
    done = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["uid"])
                except Exception:
                    pass
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_runpod.yaml")
    parser.add_argument("--n", type=int, default=16, help="completions per prompt")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-examples", type=int, default=None,
                        help="cap number of examples (for quick tests)")
    parser.add_argument("--output", default=None,
                        help="output JSONL path (default: processed/best_of_n.jsonl)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="generation batch size per prompt")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    hf_token = os.environ.get("HF_TOKEN")

    output_path = args.output or os.path.join(cfg.paths.processed_dir, "best_of_n.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────────
    sft_checkpoint = os.path.join(cfg.paths.sft_checkpoint_dir, "final")
    logger.info(f"Loading SFT checkpoint from {sft_checkpoint}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base_model, sft_checkpoint, is_trainable=False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── Load retriever ──────────────────────────────────────────────────────────
    retriever = Retriever(
        index_path=cfg.paths.faiss_index_path,
        metadata_path=cfg.paths.faiss_metadata_path,
        model_name=cfg.retrieval.model,
    )

    # ── Load dataset (filter to GT ≤ 4096 tokens) ───────────────────────────────
    train_jsonl = os.path.join(cfg.paths.processed_dir, "train.jsonl")
    with open(train_jsonl) as f:
        records = [json.loads(l) for l in f]

    filtered = []
    for r in records:
        ids = tokenizer(r["step"], add_special_tokens=False)["input_ids"]
        if len(ids) <= 4096:
            filtered.append(r)
    logger.info(f"Dataset: {len(filtered)} examples with GT STEP ≤ 4096 tokens")

    if args.max_examples:
        filtered = filtered[: args.max_examples]
        logger.info(f"Capped to {len(filtered)} examples (--max-examples)")

    done_uids = load_done_uids(output_path)
    if done_uids:
        logger.info(f"Resuming: {len(done_uids)} already done, skipping")

    # ── Main loop ───────────────────────────────────────────────────────────────
    total_parsed = 0
    total_scd = 0

    with open(output_path, "a") as out_f:
        for i, record in enumerate(filtered):
            uid = record["uid"]
            if uid in done_uids:
                continue

            retrieved = retriever.retrieve(record["caption"], exclude_uid=uid)
            retrieved_ids = tokenizer(
                retrieved["step"], add_special_tokens=False
            )["input_ids"]
            if len(retrieved_ids) > 500:
                retrieved_step = tokenizer.decode(
                    retrieved_ids[:500], skip_special_tokens=True
                )
            else:
                retrieved_step = retrieved["step"]

            prompt = format_prompt(record["caption"], retrieved_step)

            completions = generate_completions(
                prompt, model, tokenizer,
                n=args.n,
                temperature=args.temperature,
                batch_size=args.batch_size,
            )

            scores = score_completions(
                completions, record["step"], cfg.paths.text2cad_src
            )

            best_idx = max(range(len(scores)), key=lambda j: scores[j])
            best_score = scores[best_idx]
            n_terminated = sum(1 for c in completions if "END-ISO-10303-21;" in c)
            n_parsed = sum(1 for s in scores if s >= 0.3)
            has_scd = sum(1 for s in scores if s > 0.3)

            if n_parsed > 0:
                total_parsed += 1
            if has_scd > 0:
                total_scd += 1

            result = {
                "uid": uid,
                "caption": record["caption"],
                "prompt": prompt,
                "best_completion": completions[best_idx],
                "best_score": best_score,
                "scores": scores,
                "n_terminated": n_terminated,
                "n_parsed": n_parsed,
                "n": args.n,
            }
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

            logger.info(
                f"[{i+1}/{len(filtered)}] uid={uid} "
                f"best={best_score:.3f} "
                f"terminated={n_terminated}/{args.n} "
                f"parsed={n_parsed}/{args.n}"
            )

    processed = len(filtered) - len(done_uids)
    logger.info(
        f"\nDone. {processed} examples processed.\n"
        f"  Parse success (≥1 winner): {total_parsed}/{processed} "
        f"({100*total_parsed/max(processed,1):.1f}%)\n"
        f"  SCD > 0:                   {total_scd}/{processed} "
        f"({100*total_scd/max(processed,1):.1f}%)\n"
        f"  Output: {output_path}"
    )


if __name__ == "__main__":
    main()
