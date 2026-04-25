"""
test_sft_eos.py — SFT go/no-go check.

Loads the SFT checkpoint, generates completions on N val examples, and reports
what fraction hit EOS (END-ISO-10303-21;) vs. truncated at max_length.

A model that has learned to terminate is ready for RL.  Target: >50% EOS rate.
If most completions are truncated, SFT has not converged enough to advance.

Usage (local):
    python tests/test_sft_eos.py --config configs/config.yaml --n 20

Usage (Gautschi — after SFT job completes):
    python tests/test_sft_eos.py --config configs/config_gautschi.yaml --n 50
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Route HF cache before transformers import
if "HF_HOME" not in os.environ:
    _scratch = os.environ.get("SCRATCH", "")
    os.environ["HF_HOME"] = (
        os.path.join(_scratch, ".hf-cache") if _scratch
        else os.path.join(os.environ.get("VOLUME", "/runpod-volume"), ".hf-cache")
    )

import torch
from omegaconf import OmegaConf
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="SFT EOS termination check")
parser.add_argument("--config", default="configs/config.yaml")
parser.add_argument("--n", type=int, default=20, help="Number of val examples to generate on")
parser.add_argument("--max-new-tokens", type=int, default=512,
                    help="Max tokens to generate per example (keep short for speed)")
parser.add_argument("--sft-checkpoint", default=None,
                    help="Path to SFT checkpoint (defaults to config sft_checkpoint_dir/final)")
args = parser.parse_args()

cfg = OmegaConf.load(args.config)

sft_checkpoint = args.sft_checkpoint or os.path.join(cfg.paths.sft_checkpoint_dir, "final")
if not os.path.exists(sft_checkpoint):
    print(f"[ERROR] SFT checkpoint not found: {sft_checkpoint}")
    print("  Run training/llama3_SFT_response.py first.")
    sys.exit(1)

val_json = os.path.join(cfg.paths.processed_dir, "val.json")
if not os.path.exists(val_json):
    # Fallback: use test set
    val_json = os.path.join(cfg.paths.processed_dir, "test.json")
if not os.path.exists(val_json):
    print(f"[ERROR] Val/test JSON not found at {cfg.paths.processed_dir}")
    sys.exit(1)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading tokenizer from {sft_checkpoint}...")
tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print(f"Loading base model {cfg.model.base_model}...")
hf_token = os.environ.get("HUGGINGFACE_TOKEN")
use_quantization = bool(getattr(cfg.model, "use_quantization", True))

if not use_quantization:
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )
else:
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model,
        quantization_config=bnb,
        device_map="auto",
        token=hf_token,
    )

print(f"Attaching LoRA from {sft_checkpoint}...")
model = PeftModel.from_pretrained(base, sft_checkpoint)
model.eval()

# ── Prompt template (must match training exactly) ─────────────────────────────
ABC_PROMPT_RAG = (
    "You are a CAD model generation assistant trained to produce STEP (.step) files "
    "based on textual descriptions. Given the following object description and relevant "
    "retrieved CAD data, generate a STEP file that accurately represents the described object."
    "\n\n\n### caption:\n{}\n\n### retrieved relevant step file:\n{}\n\n### output:\n"
)

MAX_RETRIEVED_TOKENS = int(getattr(cfg.model, "max_retrieved_tokens", 500))

def build_prompt(record: dict) -> str:
    caption = record.get("caption", "")
    retrieved = record.get("relavant_step_file", record.get("retrieved_step", ""))
    ids = tokenizer(retrieved, add_special_tokens=False)["input_ids"]
    truncated = tokenizer.decode(ids[:MAX_RETRIEVED_TOKENS])
    return ABC_PROMPT_RAG.format(caption, truncated)

# ── Load val examples ─────────────────────────────────────────────────────────
print(f"Loading {args.n} examples from {val_json}...")
with open(val_json) as f:
    records = json.load(f)
records = records[:args.n]

# ── Generate ──────────────────────────────────────────────────────────────────
STEP_TERMINATOR = "END-ISO-10303-21;"
n_eos = 0
n_truncated = 0
n_total = 0

print(f"\nGenerating {len(records)} completions (max_new_tokens={args.max_new_tokens})...\n")

for i, record in enumerate(records):
    prompt = build_prompt(record)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=cfg.model.max_seq_length - args.max_new_tokens)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,           # greedy — deterministic, faster
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated portion (not the prompt)
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_len:]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

    hit_eos = STEP_TERMINATOR in completion
    hit_max = len(generated_ids) >= args.max_new_tokens

    n_total += 1
    if hit_eos:
        n_eos += 1
    elif hit_max:
        n_truncated += 1

    status = "EOS" if hit_eos else ("TRUNC" if hit_max else "OTHER")
    preview = completion[:80].replace("\n", " ")
    print(f"  [{i+1:3d}/{len(records)}] {status:5s}  |  {preview!r}")

# ── Summary ───────────────────────────────────────────────────────────────────
eos_pct = 100.0 * n_eos / n_total if n_total else 0
trunc_pct = 100.0 * n_truncated / n_total if n_total else 0

print("\n" + "=" * 60)
print(f"  SFT EOS Termination Report ({n_total} examples)")
print("=" * 60)
print(f"  Hit EOS (END-ISO-10303-21;)  : {n_eos:3d} / {n_total}  ({eos_pct:.1f}%)")
print(f"  Truncated (hit max_new_tokens): {n_truncated:3d} / {n_total}  ({trunc_pct:.1f}%)")
print(f"  Other (no EOS, not truncated) : {n_total - n_eos - n_truncated:3d} / {n_total}")
print("=" * 60)

if eos_pct >= 50:
    print("  VERDICT: PASS — model terminates reliably. Ready to advance to RL.")
elif eos_pct >= 20:
    print("  VERDICT: MARGINAL — model sometimes terminates. Consider more SFT epochs.")
else:
    print("  VERDICT: FAIL — model rarely terminates. Do not advance to RL yet.")
print("=" * 60)
