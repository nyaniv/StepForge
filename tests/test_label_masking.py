"""
Test that manual label masking in formatting_prompts_func works end-to-end.

This script exercises the EXACT codepath that runs during training:
  raw records → formatting_prompts_func → dataset.map → DataCollatorForSeq2Seq → batch

It does NOT need a GPU, unsloth, or the full model — just the tokenizer.

Usage:
    python tests/test_label_masking.py --config configs/config_gautschi.yaml
    python tests/test_label_masking.py --config configs/config.yaml
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config_gautschi.yaml")
parser.add_argument("--n", type=int, default=8, help="Number of examples to test")
parser.add_argument("--tokenizer_path", default=None,
                    help="Local path to tokenizer (overrides config; use when HF auth unavailable)")
args = parser.parse_args()

from omegaconf import OmegaConf
cfg = OmegaConf.load(args.config)

from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from datasets import Dataset
import torch

_tok_path = args.tokenizer_path or cfg.model.base_model
print(f"Loading tokenizer from: {_tok_path}")
tokenizer = AutoTokenizer.from_pretrained(
    _tok_path,
    local_files_only=args.tokenizer_path is not None,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"  Vocab size: {tokenizer.vocab_size}  |  pad_token_id: {tokenizer.pad_token_id}")

# ── Replicate the exact constants and functions from llama3_SFT_response.py ──
max_seq_length = cfg.model.max_seq_length
MAX_RETRIEVED_TOKENS = int(
    cfg.sft.get("max_retrieved_tokens", None)
    or getattr(cfg.model, "max_retrieved_tokens", None)
    or 4500
)

_ASSISTANT_HEADER_IDS = [128006, 78191, 128007, 271]
_ASSISTANT_HEADER_LEN = len(_ASSISTANT_HEADER_IDS)

FAILURES = []


def _find_response_start(input_ids):
    for i in range(len(input_ids) - _ASSISTANT_HEADER_LEN + 1):
        if input_ids[i:i + _ASSISTANT_HEADER_LEN] == _ASSISTANT_HEADER_IDS:
            return i + _ASSISTANT_HEADER_LEN
    return -1


def _build_user_message(instruction, retrieved, use_rag):
    if use_rag:
        return (
            "You are a CAD model generation assistant trained to produce STEP (.step) files "
            "based on textual descriptions. Given the following object description and relevant "
            "retrieved CAD data, generate a STEP file that accurately represents the described object.\n\n"
            f"### caption:\n{instruction}\n\n"
            f"### retrieved relevant step file:\n{retrieved}"
        )
    return (
        "You are a CAD model generation assistant trained to produce STEP (.step) files "
        "based on textual descriptions. Given the following object description, generate a "
        "STEP file that accurately represents the described object.\n\n"
        f"### caption:\n{instruction}"
    )


def formatting_prompts_func(examples):
    # Detect field names: main branch uses step/retrieved_step, refined-variant uses output/relavant_step_file
    if "step" in examples:
        outputs = examples["step"]
        retrieved_key = "retrieved_step"
    else:
        outputs = examples["output"]
        retrieved_key = "relavant_step_file"

    instructions = examples["caption"]
    inputs = examples.get(retrieved_key, [""] * len(instructions))

    all_input_ids, all_attention_masks, all_labels, all_texts = [], [], [], []

    for instruction, input_, output in zip(instructions, inputs, outputs):
        ids = tokenizer(input_ or "", add_special_tokens=False)["input_ids"]
        if len(ids) > MAX_RETRIEVED_TOKENS:
            input_ = tokenizer.decode(ids[:MAX_RETRIEVED_TOKENS])

        messages = [
            {"role": "user",      "content": _build_user_message(instruction or "", input_ or "", True)},
            {"role": "assistant", "content": output or ""},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        encoded = tokenizer(text, truncation=True, max_length=max_seq_length, add_special_tokens=False)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        response_start = _find_response_start(input_ids)
        if response_start == -1:
            labels = [-100] * len(input_ids)
        else:
            labels = [-100] * response_start + input_ids[response_start:]

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)
        all_texts.append(text)

    return {"input_ids": all_input_ids, "attention_mask": all_attention_masks,
            "labels": all_labels, "text": all_texts}


# ── Load real data ────────────────────────────────────────────────────────────
processed_dir = cfg.paths.processed_dir

# Try both data formats
for fname in ("train_with_rag.jsonl", "train.json"):
    train_path = os.path.join(processed_dir, fname)
    if os.path.exists(train_path):
        break
else:
    print(f"ERROR: No train data found in {processed_dir}")
    sys.exit(1)

print(f"\nLoading {args.n} records from {train_path} ...")
with open(train_path) as f:
    head = f.read(1); f.seek(0)
    if head == "[":
        all_records = json.load(f)
    else:
        all_records = [json.loads(l) for l in f if l.strip()]

records = all_records[:args.n]
print(f"  Loaded {len(records)} records")

# ── Step 1: Run formatting_prompts_func (the map step) ────────────────────────
print("\n[Step 1] Running formatting_prompts_func ...")
dataset = Dataset.from_list(records)
dataset = dataset.map(formatting_prompts_func, batched=True)

print(f"  Dataset columns: {dataset.column_names}")
assert "input_ids" in dataset.column_names, "FAIL: input_ids not in dataset"
assert "labels" in dataset.column_names,    "FAIL: labels not in dataset"
assert "attention_mask" in dataset.column_names, "FAIL: attention_mask not in dataset"

# ── Step 2: Check labels per example ─────────────────────────────────────────
print("\n[Step 2] Checking labels per example ...")
n_all_masked = 0
for i in range(len(dataset)):
    ex = dataset[i]
    input_ids = ex["input_ids"]
    labels = ex["labels"]

    assert len(input_ids) == len(labels), \
        f"FAIL example {i}: len(input_ids)={len(input_ids)} != len(labels)={len(labels)}"

    unmasked = sum(1 for l in labels if l != -100)
    total = len(labels)
    response_start = next((j for j, l in enumerate(labels) if l != -100), -1)

    if unmasked == 0:
        n_all_masked += 1
        print(f"  FAIL  example {i}: ALL {total} labels are -100 — header not found!")
        FAILURES.append(f"Example {i}: all labels masked")
    else:
        pct = 100.0 * unmasked / total
        # Sanity: verify the boundary is at the assistant header
        if response_start >= _ASSISTANT_HEADER_LEN:
            header_check = input_ids[response_start - _ASSISTANT_HEADER_LEN : response_start]
            header_ok = header_check == _ASSISTANT_HEADER_IDS
        else:
            header_ok = False
        status = "PASS" if header_ok else "WARN(header mismatch)"
        print(f"  {status}  example {i}: {unmasked}/{total} label tokens ({pct:.1f}%)  "
              f"response_start={response_start}  seq_len={total}")
        if not header_ok:
            FAILURES.append(f"Example {i}: header mismatch at response_start={response_start}")

# ── Step 3: Pass through DataCollatorForSeq2Seq and check batch labels ────────
print("\n[Step 3] Running DataCollatorForSeq2Seq (pad + batch) ...")
collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")

# Use dataset.select to get a small batch
batch_records = [dataset[i] for i in range(min(4, len(dataset)))]
# DataCollatorForSeq2Seq expects dicts with lists/tensors
batch = collator(batch_records)

assert "input_ids" in batch, "FAIL: input_ids not in batch"
assert "labels" in batch,    "FAIL: labels not in batch"

labels_batch = batch["labels"]
print(f"  Batch labels shape: {labels_batch.shape}")
for i in range(labels_batch.shape[0]):
    row = labels_batch[i]
    unmasked = (row != -100).sum().item()
    total = row.shape[0]
    print(f"  Batch example {i}: {unmasked}/{total} label tokens unmasked after collation")
    if unmasked == 0:
        FAILURES.append(f"Batch example {i}: all labels masked after DataCollatorForSeq2Seq")

# ── Step 4: Verify token IDs for _ASSISTANT_HEADER_IDS ───────────────────────
print("\n[Step 4] Verifying _ASSISTANT_HEADER_IDS are correct for this tokenizer ...")
header_str = "<|start_header_id|>assistant<|end_header_id|>\n\n"
header_ids = tokenizer.encode(header_str, add_special_tokens=False)
print(f"  tokenizer.encode('{header_str!r}') = {header_ids}")
if header_ids == _ASSISTANT_HEADER_IDS:
    print(f"  PASS: matches _ASSISTANT_HEADER_IDS = {_ASSISTANT_HEADER_IDS}")
else:
    msg = f"FAIL: expected {_ASSISTANT_HEADER_IDS}, got {header_ids}"
    print(f"  {msg}")
    FAILURES.append(msg)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
if FAILURES:
    print(f"FAILED — {len(FAILURES)} issue(s):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print(f"ALL CHECKS PASSED ({args.n} examples)")
    print("  Label masking works. Ready to submit training job.")
print("="*60)
