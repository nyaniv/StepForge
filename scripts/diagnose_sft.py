"""
Diagnostic: probe what the SFT checkpoint actually generates.

Loads the SFT model, runs greedy generation on N samples and prints a
comprehensive per-sample report plus an aggregate summary.

Usage:
    python scripts/diagnose_sft.py --config configs/config_runpod.yaml
    python scripts/diagnose_sft.py --config configs/config_runpod.yaml --max-new-tokens 4096 --n-samples 10
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

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

MAX_RETRIEVED_TOKENS = 16384  # no truncation — full retrieved STEP up to context window


def format_prompt(caption: str, retrieved_step: str, tokenizer) -> str:
    ids = tokenizer(retrieved_step, add_special_tokens=False)["input_ids"]
    truncated = tokenizer.decode(ids[:MAX_RETRIEVED_TOKENS])
    return ABC_PROMPT_RAG.format(caption, truncated)


def analyze_step(step_str: str) -> dict:
    """Structural analysis of a generated STEP string."""
    entity_pattern  = re.compile(r"^#(\d+)\s*=\s*(\w+)\s*\(", re.MULTILINE)
    complex_pattern = re.compile(r"^#(\d+)\s*=\s*\(", re.MULTILINE)
    ref_pattern     = re.compile(r"#(\d+)")

    defined_ids = set()
    entity_types = Counter()

    for m in entity_pattern.finditer(step_str):
        defined_ids.add(int(m.group(1)))
        entity_types[m.group(2)] += 1

    # all #N references in the file
    all_refs = set(int(m.group(1)) for m in ref_pattern.finditer(step_str))
    dangling = all_refs - defined_ids

    complex_count = len(complex_pattern.findall(step_str))

    has_data_section   = "DATA;" in step_str
    has_terminator     = "END-ISO-10303-21;" in step_str
    has_iso_header     = "ISO-10303-21;" in step_str
    has_header_section = "HEADER;" in step_str

    # Check if file starts correctly
    stripped = step_str.strip()
    starts_correctly = stripped.startswith("ISO-10303-21;") or stripped.startswith("DATA;")

    # Detect truncation mid-entity (last line doesn't end with ;)
    lines = [l.strip() for l in step_str.strip().splitlines() if l.strip()]
    last_line = lines[-1] if lines else ""
    truncated_mid_entity = bool(lines) and not last_line.endswith(";") and not last_line.endswith("DATA;")

    # Entity count ratio vs expected (rough proxy)
    entity_count = len(defined_ids)

    return {
        "has_iso_header":        has_iso_header,
        "has_header_section":    has_header_section,
        "has_data_section":      has_data_section,
        "has_terminator":        has_terminator,
        "starts_correctly":      starts_correctly,
        "truncated_mid_entity":  truncated_mid_entity,
        "entity_count":          entity_count,
        "dangling_refs":         len(dangling),
        "dangling_ids":          sorted(dangling)[:10],  # first 10 for display
        "complex_entities":      complex_count,
        "entity_types":          entity_types,
        "last_line":             last_line[:120],
    }


def compare_entity_types(gt_types: Counter, gen_types: Counter) -> str:
    """Show which entity types are missing or extra compared to GT."""
    all_types = set(gt_types) | set(gen_types)
    lines = []
    for t in sorted(all_types):
        gt_n  = gt_types.get(t, 0)
        gen_n = gen_types.get(t, 0)
        if gt_n == 0:
            lines.append(f"  EXTRA    {t}: gen={gen_n}")
        elif gen_n == 0:
            lines.append(f"  MISSING  {t}: gt={gt_n}")
        elif abs(gt_n - gen_n) > max(2, int(gt_n * 0.3)):
            lines.append(f"  MISMATCH {t}: gt={gt_n} gen={gen_n}")
    return "\n".join(lines) if lines else "  (all major types match)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_runpod.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--split", choices=["train", "test"], default="train")
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

    json_path = os.path.join(
        cfg.paths.processed_dir,
        "train.json" if args.split == "train" else "test.json"
    )
    logger.info(f"Loading {args.split} data from {json_path}")
    with open(json_path) as f:
        records = json.load(f)

    samples = records[:args.n_samples]
    logger.info(f"Running generation on {len(samples)} samples (max_new_tokens={args.max_new_tokens})\n")

    # ── Aggregate counters ──────────────────────────────────────────────────────
    agg = {
        "hit_terminator": 0,
        "hit_eos": 0,
        "has_data_section": 0,
        "truncated_mid_entity": 0,
        "has_dangling": 0,
        "entity_count_ratios": [],
        "token_count_ratios": [],
    }

    for i, rec in enumerate(samples):
        retrieved = rec.get("relavant_step_file", rec.get("retrieved_step", ""))
        gt_step   = rec.get("output", rec.get("step", ""))
        caption   = rec["caption"]
        uid       = rec.get("id_original", rec.get("uid", i))

        prompt = format_prompt(caption, retrieved, tokenizer)
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        gt_ids = tokenizer(gt_step, add_special_tokens=False).input_ids

        print(f"\n{'='*72}")
        print(f"SAMPLE {i+1}  uid={uid}")
        print(f"Caption: {caption[:120]}")
        print(f"{'─'*72}")
        print(f"Prompt tokens      : {prompt_ids.shape[1]}")
        print(f"GT tokens          : {len(gt_ids)}")
        print(f"GT has terminator  : {'END-ISO-10303-21;' in gt_step}")

        with torch.no_grad():
            output_ids = model.generate(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        completion_ids = output_ids[0, prompt_ids.shape[1]:]
        raw   = tokenizer.decode(completion_ids, skip_special_tokens=False)
        clean = tokenizer.decode(completion_ids, skip_special_tokens=True)

        hit_eos        = tokenizer.eos_token in raw
        hit_terminator = "END-ISO-10303-21;" in raw
        hit_limit      = len(completion_ids) >= args.max_new_tokens

        # Structural analysis
        gen_analysis = analyze_step(clean)
        gt_analysis  = analyze_step(gt_step)

        token_ratio = len(completion_ids) / max(len(gt_ids), 1)
        entity_ratio = gen_analysis["entity_count"] / max(gt_analysis["entity_count"], 1)

        print(f"{'─'*72}")
        print(f"Generated tokens   : {len(completion_ids)} / {args.max_new_tokens}  ({'HIT LIMIT' if hit_limit else 'stopped early'})")
        print(f"Token ratio        : {token_ratio:.2f}x  (gen/gt)")
        print(f"Hit EOS            : {hit_eos}")
        print(f"Hit terminator     : {hit_terminator}")
        print(f"{'─'*72}")
        print(f"STRUCTURAL CHECKS (generated)")
        print(f"  Has DATA section     : {gen_analysis['has_data_section']}")
        print(f"  Has ISO header       : {gen_analysis['has_iso_header']}")
        print(f"  Has terminator       : {gen_analysis['has_terminator']}")
        print(f"  Truncated mid-entity : {gen_analysis['truncated_mid_entity']}  (last line: {gen_analysis['last_line']!r})")
        print(f"  Entity count         : {gen_analysis['entity_count']}  (GT: {gt_analysis['entity_count']}, ratio: {entity_ratio:.2f}x)")
        print(f"  Dangling refs        : {gen_analysis['dangling_refs']}" +
              (f"  (e.g. #{gen_analysis['dangling_ids'][:3]})" if gen_analysis['dangling_refs'] else ""))
        print(f"  Complex entities     : {gen_analysis['complex_entities']}")
        print(f"{'─'*72}")
        print(f"ENTITY TYPE COMPARISON")
        print(compare_entity_types(gt_analysis["entity_types"], gen_analysis["entity_types"]))
        print(f"{'─'*72}")
        print(f"FIRST 300 chars of output:")
        print(clean[:300])
        print(f"LAST 200 chars of output:")
        print(clean[-200:])

        # Update aggregates
        if hit_terminator: agg["hit_terminator"] += 1
        if hit_eos:        agg["hit_eos"] += 1
        if gen_analysis["has_data_section"]: agg["has_data_section"] += 1
        if gen_analysis["truncated_mid_entity"]: agg["truncated_mid_entity"] += 1
        if gen_analysis["dangling_refs"] > 0: agg["has_dangling"] += 1
        agg["entity_count_ratios"].append(entity_ratio)
        agg["token_count_ratios"].append(token_ratio)

    # ── Aggregate summary ────────────────────────────────────────────────────────
    n = len(samples)
    avg_entity_ratio = sum(agg["entity_count_ratios"]) / n
    avg_token_ratio  = sum(agg["token_count_ratios"]) / n

    print(f"\n{'='*72}")
    print(f"  AGGREGATE SUMMARY  ({n} samples, max_new_tokens={args.max_new_tokens})")
    print(f"{'='*72}")
    print(f"  Hit terminator       : {agg['hit_terminator']}/{n}  ({100*agg['hit_terminator']//n}%)")
    print(f"  Hit EOS              : {agg['hit_eos']}/{n}  ({100*agg['hit_eos']//n}%)")
    print(f"  Has DATA section     : {agg['has_data_section']}/{n}")
    print(f"  Truncated mid-entity : {agg['truncated_mid_entity']}/{n}  (cut off inside an entity line)")
    print(f"  Has dangling refs    : {agg['has_dangling']}/{n}")
    print(f"  Avg entity ratio     : {avg_entity_ratio:.2f}x  (>1 = model generates more entities than GT)")
    print(f"  Avg token ratio      : {avg_token_ratio:.2f}x  (>1 = model generates more tokens than GT)")
    print(f"{'─'*72}")
    if agg["hit_terminator"] == 0:
        if avg_token_ratio >= 0.95:
            print("  ⚠ No terminations AND token ratio ≥1 → model is generating too many entities (not converged)")
        else:
            print("  ⚠ No terminations AND token ratio <1 → likely hitting max_new_tokens limit, try increasing it")
    else:
        print(f"  ✓ Model terminates on {agg['hit_terminator']}/{n} samples")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
