"""
Diagnostic: probe what the SFT checkpoint actually generates.

Loads the SFT model, runs greedy generation on 5 samples from the training
set, and prints the raw decoded outputs (with special tokens visible) so we
can see whether the model ever produces END-ISO-10303-21; or EOS.

Usage:
    python scripts/diagnose_sft.py --config configs/config_runpod.yaml
    python scripts/diagnose_sft.py --config configs/config_runpod.yaml --max-new-tokens 1024
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


ABC_PROMPT_RAG = (
    "You are a CAD model generation assistant trained to produce STEP (.step) files "
    "based on textual descriptions. Given the following object description and relevant "
    "retrieved CAD data, generate a STEP file that accurately represents the described object."
    "\n\n\n### caption:\n{}\n\n### retrieved relevant step file:\n{}\n\n### output:\n"
)

MAX_RETRIEVED_TOKENS = 500  # must match training/llama3_SFT_response.py


def format_prompt(caption: str, retrieved_step: str, tokenizer) -> str:
    ids = tokenizer(retrieved_step, add_special_tokens=False)["input_ids"]
    truncated = tokenizer.decode(ids[:MAX_RETRIEVED_TOKENS])
    return ABC_PROMPT_RAG.format(caption, truncated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_runpod.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Override checkpoint path (default: sft_checkpoint_dir/final)")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")

    sft_checkpoint = args.checkpoint or os.path.join(cfg.paths.sft_checkpoint_dir, "final")
    logger.info(f"Loading SFT checkpoint from {sft_checkpoint}")

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
    model = PeftModel.from_pretrained(base_model, sft_checkpoint)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load samples from train.jsonl
    train_jsonl = os.path.join(cfg.paths.processed_dir, "train.jsonl")
    with open(train_jsonl) as f:
        records = [json.loads(l) for l in f]

    samples = records[:args.n_samples]

    logger.info(f"\nRunning generation on {len(samples)} samples "
                f"(max_new_tokens={args.max_new_tokens})\n")

    for i, rec in enumerate(samples):
        retrieved = rec.get("relavant_step_file", rec.get("retrieved_step", ""))
        gt_step   = rec.get("output", rec.get("step", ""))
        prompt = format_prompt(rec["caption"], retrieved, tokenizer)
        prompt_ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(model.device)

        gt_ids = tokenizer(gt_step, add_special_tokens=False).input_ids

        print(f"\n{'='*70}")
        print(f"SAMPLE {i+1}  uid={rec.get('id_original', rec.get('uid', i))}")
        print(f"Caption: {rec['caption'][:100]}")
        print(f"Prompt tokens    : {prompt_ids.shape[1]}")
        print(f"GT STEP tokens   : {len(gt_ids)}")
        print(f"GT has terminator: {'END-ISO-10303-21;' in gt_step}")

        with torch.no_grad():
            output_ids = model.generate(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        completion_ids = output_ids[0, prompt_ids.shape[1]:]
        raw = tokenizer.decode(completion_ids, skip_special_tokens=False)
        clean = tokenizer.decode(completion_ids, skip_special_tokens=True)

        hit_eos        = tokenizer.eos_token in raw
        hit_terminator = "END-ISO-10303-21;" in raw

        print(f"Generated tokens : {len(completion_ids)}")
        print(f"Hit EOS token    : {hit_eos}")
        print(f"Hit terminator   : {hit_terminator}")
        print(f"\n--- First 400 chars (raw) ---")
        print(raw[:400])
        print(f"\n--- Last 200 chars (raw) ---")
        print(raw[-200:])

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"If 'Hit terminator: False' for all samples → model never learned EOS.")
    print(f"Fix: redo SFT with truncated retrieved_step + EOS appended.")


if __name__ == "__main__":
    main()
