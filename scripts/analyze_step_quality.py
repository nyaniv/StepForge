"""
Structural quality analysis for STEP files — GT training data vs SFT outputs.

Two modes:

  1. GT-only (no model needed, runs on CPU/Mac):
       python scripts/analyze_step_quality.py --config configs/config.yaml

  2. GT + SFT comparison (generates from checkpoint):
       python scripts/analyze_step_quality.py \
           --config configs/config.yaml \
           --checkpoint checkpoints/sft/final \
           --n 200

  3. GT + pre-generated SFT outputs (if you already have a JSONL):
       python scripts/analyze_step_quality.py \
           --config configs/config.yaml \
           --sft-outputs /path/to/sft_outputs.jsonl

Structural checks (independent of any reward signal):

  - has_terminator     : "END-ISO-10303-21;" present
  - entity_count       : number of #N = TYPE(...); lines
  - dangling_refs      : references to #IDs not defined in the file
  - dropped_complex    : complex STEP entities (#N = (TYPE1() TYPE2()...)) silently
                         skipped by the parser — the root cause of dangling refs in GT
  - empty_output       : no entities at all

NOTE on DFS ordering: the reserializer uses pre-order DFS (parents before children),
so every entity references IDs defined *later* in the file. Forward references are
100% expected and correct — they are NOT a bug.

dangling_refs is the key hard structural check. If GT has high dangling rates,
the data pipeline has a bug. If GT is clean but SFT outputs have high rates,
the model failed to learn the format.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger


# ── Core structural analysis ──────────────────────────────────────────────────

def analyze_step(step_str: str) -> dict:
    """
    Run structural checks on a single STEP string.

    Returns a dict with:
        has_terminator   (bool)
        entity_count     (int)
        dangling_refs    (int)   — refs to undefined IDs
        dropped_complex  (int)   — complex STEP entities skipped by parser
                                   (root cause of dangling refs in GT data)
        empty            (bool)  — no entities parsed at all
        entity_types     (Counter) — distribution of entity type names

    NOTE: forward references are 100% expected in this format.
    The reserializer uses pre-order DFS (parents before children), so every
    entity references IDs defined later. This is correct, not a bug.
    """
    has_terminator = "END-ISO-10303-21;" in step_str

    entity_pattern  = re.compile(r"^#(\d+)\s*=\s*(\w+)\s*\((.+)\)\s*;$")
    complex_pattern = re.compile(r"^#(\d+)\s*=\s*\(")
    ref_pattern     = re.compile(r"#(\d+)")

    defined_set    = set()
    entity_refs    = {}
    entity_types   = Counter()
    dropped_complex = 0

    in_data = False
    pending = ""

    for raw_line in step_str.splitlines():
        line = raw_line.strip()

        if not in_data:
            if line == "DATA;":
                in_data = True
            continue

        if line in ("ENDSEC;", "END-ISO-10303-21;"):
            break

        if line.startswith("/*"):
            continue

        pending = (pending + " " + line) if pending else line

        if not pending.endswith(";"):
            continue

        entity_line = pending
        pending = ""

        m = entity_pattern.match(entity_line)
        if m:
            eid   = int(m.group(1))
            etype = m.group(2)
            eargs = m.group(3)
            refs  = [int(r) for r in ref_pattern.findall(eargs)]
            defined_set.add(eid)
            entity_refs[eid] = refs
            entity_types[etype] += 1
        else:
            cm = complex_pattern.match(entity_line)
            if cm:
                # Complex STEP entity: #N = (TYPE1() TYPE2()...);
                # The ID IS defined in the file — add to defined_set so references
                # to it are not counted as dangling. Track count separately.
                eid = int(cm.group(1))
                defined_set.add(eid)
                dropped_complex += 1

    entity_count  = len(defined_set)
    dangling_refs = sum(
        1 for refs in entity_refs.values() for r in refs if r not in defined_set
    )

    return {
        "has_terminator":  has_terminator,
        "entity_count":    entity_count,
        "dangling_refs":   dangling_refs,
        "dropped_complex": dropped_complex,
        "empty":           entity_count == 0,
        "entity_types":    entity_types,
    }


# ── Dataset-level aggregation ─────────────────────────────────────────────────

def analyze_dataset(records: list[dict], step_key: str, label: str) -> dict:
    """
    Run analyze_step over a list of records and aggregate results.

    Args:
        records  : list of dicts, each must have step_key
        step_key : key in each record that holds the STEP string
        label    : display name for this dataset

    Returns aggregate stats dict.
    """
    n = len(records)
    if n == 0:
        logger.warning(f"{label}: no records to analyze")
        return {}

    logger.info(f"Analyzing {n} records [{label}]...")

    terminator_count    = 0
    empty_count         = 0
    files_with_dangling = 0
    files_with_complex  = 0
    total_dangling      = 0
    total_complex       = 0
    entity_counts       = []
    all_types           = Counter()

    for i, rec in enumerate(records):
        step_str = rec.get(step_key, "")
        result   = analyze_step(step_str)

        if result["has_terminator"]:
            terminator_count += 1
        if result["empty"]:
            empty_count += 1
        if result["dangling_refs"] > 0:
            files_with_dangling += 1
        if result["dropped_complex"] > 0:
            files_with_complex += 1

        total_dangling += result["dangling_refs"]
        total_complex  += result["dropped_complex"]
        entity_counts.append(result["entity_count"])
        all_types.update(result["entity_types"])

        if (i + 1) % 500 == 0:
            logger.info(f"  {label}: {i+1}/{n}")

    entity_counts.sort()
    median_entities = entity_counts[len(entity_counts) // 2] if entity_counts else 0

    return {
        "label":                   label,
        "n":                       n,
        "terminator_rate":         terminator_count / n,
        "empty_rate":              empty_count / n,
        "files_with_dangling_pct": files_with_dangling / n * 100,
        "files_with_complex_pct":  files_with_complex  / n * 100,
        "total_dangling_refs":     total_dangling,
        "total_dropped_complex":   total_complex,
        "median_entity_count":     median_entities,
        "top_entity_types":        all_types.most_common(10),
    }


# ── Report printing ───────────────────────────────────────────────────────────

def print_report(gt_stats: dict, sft_stats: dict | None):
    col = 22

    def row(label, gt_val, sft_val=None):
        line = f"  {label:<{col}} {str(gt_val):<20}"
        if sft_val is not None:
            line += f" {str(sft_val):<20}"
        print(line)

    header = f"\n{'='*70}"
    print(header)
    print(f"  STEP QUALITY ANALYSIS")
    print(f"{'='*70}")

    header_line = f"  {'Metric':<{col}} {'GT (train.jsonl)':<20}"
    if sft_stats:
        header_line += f" {'SFT outputs':<20}"
    print(header_line)
    print(f"  {'-'*col} {'-'*20}" + (f" {'-'*20}" if sft_stats else ""))

    row("N files analyzed",
        gt_stats["n"],
        sft_stats["n"] if sft_stats else None)

    row("Terminator rate",
        f"{gt_stats['terminator_rate']:.1%}",
        f"{sft_stats['terminator_rate']:.1%}" if sft_stats else None)

    row("Empty outputs",
        f"{gt_stats['empty_rate']:.1%}",
        f"{sft_stats['empty_rate']:.1%}" if sft_stats else None)

    row("Files w/ dangling refs",
        f"{gt_stats['files_with_dangling_pct']:.1f}%",
        f"{sft_stats['files_with_dangling_pct']:.1f}%" if sft_stats else None)

    row("Files w/ dropped complex",
        f"{gt_stats['files_with_complex_pct']:.1f}%",
        f"{sft_stats['files_with_complex_pct']:.1f}%" if sft_stats else None)

    row("Total dangling refs",
        gt_stats["total_dangling_refs"],
        sft_stats["total_dangling_refs"] if sft_stats else None)

    row("Total dropped complex",
        gt_stats["total_dropped_complex"],
        sft_stats["total_dropped_complex"] if sft_stats else None)

    row("Median entity count",
        gt_stats["median_entity_count"],
        sft_stats["median_entity_count"] if sft_stats else None)

    print(f"\n  Top entity types (GT):")
    for etype, count in gt_stats["top_entity_types"][:5]:
        print(f"    {etype:<35} {count}")

    if sft_stats:
        print(f"\n  Top entity types (SFT):")
        for etype, count in sft_stats["top_entity_types"][:5]:
            print(f"    {etype:<35} {count}")

    print(f"\n  INTERPRETATION")
    print(f"  {'─'*60}")

    gt_d = gt_stats["files_with_dangling_pct"]
    gt_c = gt_stats["files_with_complex_pct"]

    if gt_d > 5.0:
        print(f"  ✗ GT data has dangling refs in {gt_d:.1f}% of files")
        print(f"    Complex entities present in {gt_c:.1f}% of files")
        print(f"    → Dangling refs despite complex entity IDs being added to defined_set.")
        print(f"      This is a real data pipeline bug — check step_restructurer.py.")
    else:
        print(f"  ✓ GT data is structurally clean (dangling: {gt_d:.1f}%)")
        if gt_c > 0:
            print(f"    Complex entities present in {gt_c:.1f}% of files (normal — unit/context entities)")

    if sft_stats:
        sft_d = sft_stats["files_with_dangling_pct"]
        sft_c = sft_stats["files_with_complex_pct"]
        print()
        if sft_d > gt_d + 10.0:
            print(f"  ✗ SFT outputs have MORE dangling refs than GT ({sft_d:.1f}% vs {gt_d:.1f}%)")
            print(f"    → Model is introducing new dangling refs beyond the pipeline bug.")
            print(f"      Training failure: model did not learn valid entity references.")
        elif sft_c > 1.0:
            print(f"  ~ SFT outputs contain complex entities ({sft_c:.1f}% of files)")
            print(f"    → Unexpected: SFT model is generating complex entity syntax it was never trained on.")
        else:
            print(f"  ✓ SFT outputs have similar dangling rate to GT ({sft_d:.1f}% vs {gt_d:.1f}%)")
            print(f"    → Structural format learned correctly. Quality issues are semantic, not structural.")
    else:
        print(f"\n  Run with --checkpoint to compare against SFT outputs.")

    print(f"{'='*70}\n")


# ── SFT generation ────────────────────────────────────────────────────────────

def generate_sft_outputs(cfg, checkpoint: str, records: list[dict], n: int) -> list[dict]:
    """
    Generate STEP outputs from the SFT checkpoint for the first N records.
    Returns list of dicts with 'caption', 'uid', 'step' (generated), 'gt_step'.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from retrieval.retriever import Retriever

    hf_token = os.environ.get("HUGGINGFACE_TOKEN")

    logger.info(f"Loading SFT checkpoint from {checkpoint}...")
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
    model = PeftModel.from_pretrained(base_model, checkpoint)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    retriever = Retriever(
        index_path=cfg.paths.faiss_index_path,
        metadata_path=cfg.paths.faiss_metadata_path,
        model_name=cfg.retrieval.model,
    )

    MAX_RETRIEVED_TOKENS = 500  # must match training/llama3_SFT_response.py

    ABC_PROMPT_RAG = (
        "You are a CAD model generation assistant trained to produce STEP (.step) files "
        "based on textual descriptions. Given the following object description and relevant "
        "retrieved CAD data, generate a STEP file that accurately represents the described object."
        "\n\n\n### caption:\n{}\n\n### retrieved relevant step file:\n{}\n\n### output:\n"
    )

    def format_prompt(caption, retrieved_step):
        ids = tokenizer(retrieved_step, add_special_tokens=False)["input_ids"]
        truncated = tokenizer.decode(ids[:MAX_RETRIEVED_TOKENS])
        return ABC_PROMPT_RAG.format(caption, truncated)

    def extract_step(text):
        m = re.search(r"(DATA;.*?END-ISO-10303-21;)", text, re.DOTALL)
        if m:
            return m.group(1)
        m = re.search(r"(ISO-10303-21;.*?END-ISO-10303-21;)", text, re.DOTALL)
        return m.group(1) if m else text

    samples = records[:n]
    results = []

    logger.info(f"Generating {len(samples)} SFT outputs...")
    for i, rec in enumerate(samples):
        uid       = rec.get("id_original") or rec.get("uid", str(i))
        retrieved = retriever.retrieve(rec["caption"], exclude_uid=uid)
        retrieved_step = retrieved.get("output") or retrieved.get("step") or ""
        prompt    = format_prompt(rec["caption"], retrieved_step)

        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        completion_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        text           = tokenizer.decode(completion_ids, skip_special_tokens=True)
        step_content   = extract_step(text)

        results.append({
            "uid":     uid,
            "caption": rec["caption"],
            "step":    step_content,
            "gt_step": rec.get("output") or rec.get("step", ""),
        })

        if (i + 1) % 10 == 0:
            logger.info(f"  Generated {i+1}/{len(samples)}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Structural STEP quality analysis")
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--checkpoint",  default=None,
                        help="SFT checkpoint dir — triggers SFT generation")
    parser.add_argument("--sft-outputs", default=None,
                        help="Pre-generated SFT outputs JSONL (step key = 'step')")
    parser.add_argument("--n",           type=int, default=200,
                        help="Number of examples to analyze (default 200)")
    parser.add_argument("--split",       default="train",
                        choices=["train", "val", "test"],
                        help="Which split to load for GT analysis")
    parser.add_argument("--save-sft",    default=None,
                        help="If generating, save outputs to this JSONL path")
    args = parser.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)

    # ── Load GT data ──────────────────────────────────────────────────────────
    split_file = os.path.join(cfg.paths.processed_dir, f"{args.split}.json")
    logger.info(f"Loading GT data from {split_file}")
    with open(split_file) as f:
        gt_records = json.load(f)

    gt_records = gt_records[:args.n]
    gt_stats   = analyze_dataset(gt_records, step_key="output", label="GT")

    # ── Load or generate SFT outputs ─────────────────────────────────────────
    sft_stats = None

    if args.sft_outputs:
        logger.info(f"Loading SFT outputs from {args.sft_outputs}")
        with open(args.sft_outputs) as f:
            sft_records = [json.loads(l) for l in f]
        sft_records = sft_records[:args.n]
        sft_stats   = analyze_dataset(sft_records, step_key="step", label="SFT")

    elif args.checkpoint:
        sft_records = generate_sft_outputs(cfg, args.checkpoint, gt_records, args.n)

        if args.save_sft:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_sft)), exist_ok=True)
            with open(args.save_sft, "w") as f:
                for r in sft_records:
                    f.write(json.dumps({k: v for k, v in r.items() if k != "gt_step"}) + "\n")
            logger.info(f"SFT outputs saved to {args.save_sft}")

        sft_stats = analyze_dataset(sft_records, step_key="step", label="SFT")

    # ── Print report ──────────────────────────────────────────────────────────
    print_report(gt_stats, sft_stats)


if __name__ == "__main__":
    main()
