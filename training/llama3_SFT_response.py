"""
STEP-LLM Training Script
========================
Fine-tunes a Llama-3.2 or Qwen-2.5 model on the STEP-CAD dataset using
Unsloth's efficient LoRA implementation.

This script is based on the Unsloth SFT template:
  https://github.com/unslothai/unsloth

For a notebook version see: llama3_SFT_response.ipynb

Quick start
-----------
1. Set the configuration variables in the "── Configuration ──" section below.
2. Run: python llama3_SFT_response.py
3. The LoRA adapter will be saved to LORA_SAVE_PATH.

The adapter can then be used directly in generate_step.py, or merged into a
full model with: python scripts/merge_lora_adapter.py
"""

# ── Configuration ──────────────────────────────────────────────────────────────
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Route HF downloads to the network volume on RunPod
if "HF_HOME" not in os.environ:
    _vol = os.environ.get("VOLUME", "/runpod-volume")
    os.environ["HF_HOME"] = os.path.join(_vol, ".hf-cache")

import argparse as _argparse
from omegaconf import OmegaConf

_ap = _argparse.ArgumentParser(add_help=False)
_ap.add_argument("--config", default=None, help="Path to config YAML (default: configs/config_runpod.yaml)")
_args, _ = _ap.parse_known_args()
_default_config = _args.config or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "config_runpod.yaml",
)
_cfg = OmegaConf.load(_default_config)

BASE_MODEL_PATH = _cfg.model.base_model                           # meta-llama/Llama-3.2-3B-Instruct
TRAIN_JSON      = os.path.join(_cfg.paths.processed_dir, "train_with_rag.jsonl")
TEST_JSON       = os.path.join(_cfg.paths.processed_dir, "test.jsonl")
LORA_SAVE_PATH  = os.path.join(_cfg.paths.sft_checkpoint_dir, "final")
OUTPUT_DIR      = _cfg.paths.sft_checkpoint_dir
USE_RAG         = True
# ── End configuration ──────────────────────────────────────────────────────────


# ── Logging setup ───────────────────────────────────────────────────────────────
import time
import math
import collections
from loguru import logger

os.makedirs(OUTPUT_DIR, exist_ok=True)
_log_path = os.path.join(OUTPUT_DIR, "sft_train.log")
logger.add(_log_path, level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", enqueue=True)
logger.add(sys.stdout, level="INFO",  format="{time:HH:mm:ss} | {level} | {message}")
logger.info(f"Logging to {_log_path}")
# ── End logging setup ──────────────────────────────────────────────────────────


from unsloth import FastLanguageModel
import torch

max_seq_length = _cfg.model.max_seq_length  # RoPE Scaling is handled automatically by Unsloth
# Token budget for the retrieved STEP context.
# A typical retrieved STEP file (~265 entities at ~14 tok/entity) is ~3,700–4,200 tokens.
# DFS ordering front-loads root structure (shell, faces, axes) so the first ~35 entities
# (~500 tokens) carry the bulk of useful structural signal; trailing entities are mostly
# repeated CARTESIAN_POINT coordinates specific to the retrieved shape.
# 500 tokens is a deliberate cost-saving choice. The paper does not truncate retrieved
# context — increasing MAX_RETRIEVED_TOKENS improves RAG fidelity at the cost of
# proportionally less room for the GT STEP output within max_seq_length.
MAX_RETRIEVED_TOKENS = int(getattr(_cfg.model, "max_retrieved_tokens", 500))  # configurable; Gautschi sets 4096 (paper-faithful)
dtype = None            # None = auto-detect (bfloat16 on Ampere+, float16 on older GPUs)
load_in_4bit = False    # Set True to use 4-bit quantisation (reduces VRAM, slight quality loss)

logger.info(f"Loading base model: {BASE_MODEL_PATH}")
logger.info(f"max_seq_length={max_seq_length}, load_in_4bit={load_in_4bit}, USE_RAG={USE_RAG}")

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_PATH,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token="hf_...",   # needed for gated models (e.g. Llama)
)

# Attach LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=_cfg.model.lora_r,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=_cfg.model.lora_alpha,
    lora_dropout=0,                     # must be 0 — Unsloth kernel fusions break with dropout
    bias="none",                        # "none" is optimised in Unsloth
    use_gradient_checkpointing="unsloth",  # saves 30% VRAM; also fits 2× larger batches
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
logger.info(f"LoRA config: r={_cfg.model.lora_r}, alpha={_cfg.model.lora_alpha}")
model.print_trainable_parameters()

# ── Prompt templates ────────────────────────────────────────────────────────────
# These templates MUST be used consistently at both training and inference time.

ABC_PROMPT_RAG = """You are a CAD model generation assistant trained to produce STEP (.step) files based on textual descriptions. Given the following object description and relevant retrieved CAD data, generate a STEP file that accurately represents the described object.


### caption:
{}

### retrieved relevant step file:
{}

### output:
{}"""

ABC_PROMPT_NO_RAG = """You are a CAD model generation assistant trained to produce STEP (.step) files based on textual descriptions. Given the following object description, generate a STEP file that accurately represents the described object.

### caption:
{}

### output:
{}"""

EOS_TOKEN = tokenizer.eos_token  # must be appended to every training example

# ── Formatting stats (populated during dataset.map) ────────────────────────────
_fmt_stats = {
    "total": 0,
    "truncated": 0,
    "truncation_lengths": [],   # original token lengths of truncated retrieved files
    "seq_lengths": [],
    "missing_caption": 0,
    "missing_output": 0,
    "missing_retrieved": 0,
}


def formatting_prompts_func(examples):
    """Format dataset examples into the training prompt."""
    instructions = examples["caption"]
    outputs = examples["step"]
    texts = []

    if USE_RAG:
        inputs = examples.get("retrieved_step", [""] * len(instructions))
        for instruction, input_, output in zip(instructions, inputs, outputs):
            _fmt_stats["total"] += 1

            # Track missing fields
            if not instruction:
                _fmt_stats["missing_caption"] += 1
                logger.warning(f"[fmt] Missing caption at index {_fmt_stats['total']}")
            if not output:
                _fmt_stats["missing_output"] += 1
                logger.warning(f"[fmt] Missing output at index {_fmt_stats['total']}")
            if not input_:
                _fmt_stats["missing_retrieved"] += 1

            # Truncate retrieved STEP to MAX_RETRIEVED_TOKENS.
            # See comment above MAX_RETRIEVED_TOKENS for rationale.
            ids = tokenizer(input_ or "", add_special_tokens=False)["input_ids"]
            if len(ids) > MAX_RETRIEVED_TOKENS:
                _fmt_stats["truncated"] += 1
                _fmt_stats["truncation_lengths"].append(len(ids))
                truncated = tokenizer.decode(ids[:MAX_RETRIEVED_TOKENS])
            else:
                truncated = input_ or ""

            text = ABC_PROMPT_RAG.format(instruction or "", truncated, output or "") + EOS_TOKEN
            texts.append(text)

            # Track sequence length
            seq_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            _fmt_stats["seq_lengths"].append(len(seq_ids))

    else:
        for instruction, output in zip(instructions, outputs):
            _fmt_stats["total"] += 1
            text = ABC_PROMPT_NO_RAG.format(instruction, output) + EOS_TOKEN
            texts.append(text)
            seq_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            _fmt_stats["seq_lengths"].append(len(seq_ids))

    return {"text": texts}


def _log_fmt_stats(tag: str):
    """Log a summary of the formatting stats collected so far."""
    s = _fmt_stats
    if not s["seq_lengths"]:
        return
    lengths = sorted(s["seq_lengths"])
    n = len(lengths)
    over_limit = sum(1 for l in lengths if l > max_seq_length)
    trunc_pct = 100.0 * s["truncated"] / max(s["total"], 1)
    over_pct  = 100.0 * over_limit / max(n, 1)

    logger.info(f"[{tag}] Dataset formatting stats ({n} examples):")
    logger.info(f"  truncated retrieved_step : {s['truncated']} / {s['total']}  ({trunc_pct:.1f}%)")
    if s["truncation_lengths"]:
        tl = sorted(s["truncation_lengths"])
        logger.info(f"    original len (p50/p90/max): {tl[len(tl)//2]} / {tl[int(len(tl)*0.9)]} / {tl[-1]} tokens")
    logger.info(f"  seq length  p25={lengths[n//4]}  p50={lengths[n//2]}  p75={lengths[3*n//4]}  p90={lengths[int(n*0.9)]}  max={lengths[-1]}")
    logger.info(f"  over max_seq_length ({max_seq_length}): {over_limit} ({over_pct:.1f}%) — will be truncated by trainer")
    if s["missing_caption"]:
        logger.error(f"  MISSING captions: {s['missing_caption']}")
    if s["missing_output"]:
        logger.error(f"  MISSING outputs: {s['missing_output']}")
    if s["missing_retrieved"] and USE_RAG:
        logger.error(f"  MISSING retrieved_step: {s['missing_retrieved']}")


# ── Dataset ─────────────────────────────────────────────────────────────────────
import json
from datasets import Dataset

_FORMATTED_TRAIN = os.path.join(OUTPUT_DIR, "formatted_train")
_FORMATTED_TEST  = os.path.join(OUTPUT_DIR, "formatted_test")

if os.path.exists(_FORMATTED_TRAIN) and os.path.exists(_FORMATTED_TEST):
    logger.info(f"Loading pre-formatted datasets from disk (skipping map)...")
    dataset      = Dataset.load_from_disk(_FORMATTED_TRAIN)
    test_dataset = Dataset.load_from_disk(_FORMATTED_TEST)
    logger.info(f"  Train: {len(dataset)}  |  Test: {len(test_dataset)}")
else:
    logger.info(f"Loading train data from {TRAIN_JSON}")
    with open(TRAIN_JSON) as f:
        train_records = [json.loads(line) for line in f if line.strip()]
    dataset = Dataset.from_list(train_records)
    del train_records
    logger.info(f"  Raw train records: {len(dataset)}")

    logger.info(f"Loading test data from {TEST_JSON}")
    with open(TEST_JSON) as f:
        test_records = [json.loads(line) for line in f if line.strip()]
    test_dataset = Dataset.from_list(test_records)
    del test_records
    logger.info(f"  Raw test records:  {len(test_dataset)}")

    logger.info("Formatting train dataset...")
    dataset = dataset.map(formatting_prompts_func, batched=True)
    _log_fmt_stats("train")

    # Reset counters for test set
    for k in ("total", "truncated", "missing_caption", "missing_output", "missing_retrieved"):
        _fmt_stats[k] = 0
    _fmt_stats["truncation_lengths"].clear()
    _fmt_stats["seq_lengths"].clear()

    logger.info("Formatting test dataset...")
    test_dataset = test_dataset.map(formatting_prompts_func, batched=True)
    _log_fmt_stats("test")

    logger.info("Saving formatted datasets to disk for future runs...")
    dataset.save_to_disk(_FORMATTED_TRAIN)
    test_dataset.save_to_disk(_FORMATTED_TEST)
    logger.info(f"  Saved to {_FORMATTED_TRAIN} and {_FORMATTED_TEST}")

logger.info(f"Final train size: {len(dataset)}  |  test size: {len(test_dataset)}")


# ── Per-epoch callback ───────────────────────────────────────────────────────────
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments as HFTrainingArguments

class VerboseEpochCallback(TrainerCallback):
    """Logs a comprehensive per-epoch summary and detects silent masking failures."""

    def __init__(self):
        self._epoch_start = None
        self._step_losses = []
        self._step_grad_norms = []
        self._all_masked_count = 0

    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self._epoch_start = time.time()
        self._step_losses = []
        self._step_grad_norms = []
        self._all_masked_count = 0
        epoch = int(state.epoch or 0) + 1
        logger.info(f"{'='*60}")
        logger.info(f"  EPOCH {epoch} / {args.num_train_epochs} — starting")
        logger.info(f"  Steps this epoch: {state.max_steps // args.num_train_epochs}")
        logger.info(f"{'='*60}")

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return
        loss = logs.get("loss")
        grad_norm = logs.get("grad_norm")
        lr = logs.get("learning_rate")
        if loss is not None:
            self._step_losses.append(loss)
            if math.isnan(loss) or math.isinf(loss):
                logger.error(f"[step {state.global_step}] NaN/Inf loss detected! loss={loss}  lr={lr}  grad_norm={grad_norm}")
            elif loss == 0.0:
                self._all_masked_count += 1
                logger.warning(
                    f"[step {state.global_step}] Zero loss — possible all-masked labels "
                    f"(train_on_responses_only found no response tokens). "
                    f"Consecutive zero-loss steps: {self._all_masked_count}"
                )
            else:
                self._all_masked_count = 0  # reset streak
        if grad_norm is not None:
            self._step_grad_norms.append(grad_norm)
            if math.isnan(grad_norm) or math.isinf(grad_norm):
                logger.error(f"[step {state.global_step}] NaN/Inf grad_norm detected! grad_norm={grad_norm}")

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        elapsed = time.time() - (self._epoch_start or time.time())
        epoch = int(state.epoch or 0)

        if self._step_losses:
            avg_loss = sum(self._step_losses) / len(self._step_losses)
            min_loss = min(self._step_losses)
            max_loss = max(self._step_losses)
            sorted_losses = sorted(self._step_losses)
            n = len(sorted_losses)
            p10 = sorted_losses[max(0, int(n * 0.10))]
            p90 = sorted_losses[min(n - 1, int(n * 0.90))]
        else:
            avg_loss = min_loss = max_loss = p10 = p90 = float("nan")

        if self._step_grad_norms:
            avg_gn = sum(self._step_grad_norms) / len(self._step_grad_norms)
            max_gn = max(self._step_grad_norms)
        else:
            avg_gn = max_gn = float("nan")

        steps_done = len(self._step_losses)
        zero_loss_steps = sum(1 for l in self._step_losses if l == 0.0)

        logger.info(f"{'='*60}")
        logger.info(f"  EPOCH {epoch} / {args.num_train_epochs} — COMPLETE")
        logger.info(f"  Wall time      : {elapsed/60:.1f} min")
        logger.info(f"  Steps logged   : {steps_done}")
        logger.info(f"  Loss  avg={avg_loss:.4f}  min={min_loss:.4f}  max={max_loss:.4f}  p10={p10:.4f}  p90={p90:.4f}")
        logger.info(f"  Grad norm  avg={avg_gn:.4f}  max={max_gn:.4f}")
        logger.info(f"  Zero-loss steps: {zero_loss_steps} / {steps_done}"
                    + (" ← WARNING: response masking may be broken" if zero_loss_steps > steps_done * 0.05 else ""))
        logger.info(f"  Global step    : {state.global_step}")
        logger.info(f"  Best metric    : {state.best_metric}")
        # Memory snapshot
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"  VRAM  alloc={alloc:.2f}GB  reserved={reserved:.2f}GB")
        logger.info(f"{'='*60}")

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        logger.info("Training finished.")
        logger.info(f"  Total steps    : {state.global_step}")
        logger.info(f"  Best checkpoint: {state.best_model_checkpoint}")


# ── Training ─────────────────────────────────────────────────────────────────────
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# Use TRL's DataCollatorForCompletionOnlyLM to train on response tokens only.
# This replicates the paper's setup: loss is computed only on the STEP output,
# not on the system prompt or caption. It works with custom prompt formats
# (unlike Unsloth's train_on_responses_only which uses token-ID matching).
_response_template = "\n### output:\n"
_collator = DataCollatorForCompletionOnlyLM(
    response_template=_response_template,
    tokenizer=tokenizer,
)
logger.info(f"Using DataCollatorForCompletionOnlyLM with response_template={_response_template!r}")

logger.info("Building SFTTrainer...")
logger.info(f"  epochs={_cfg.sft.num_epochs}  lr={_cfg.sft.learning_rate}  "
            f"batch={_cfg.sft.per_device_train_batch_size}  grad_accum={_cfg.sft.gradient_accumulation_steps}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=test_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=_collator,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=_cfg.sft.per_device_train_batch_size,
        gradient_accumulation_steps=_cfg.sft.gradient_accumulation_steps,
        warmup_ratio=_cfg.sft.warmup_ratio,
        num_train_epochs=_cfg.sft.num_epochs,
        learning_rate=_cfg.sft.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim=_cfg.sft.optim,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
        save_strategy="steps",
        save_steps=300,
    ),
    callbacks=[VerboseEpochCallback()],
)

logger.info("Starting SFT training...")
_train_start = time.time()
trainer_stats = trainer.train()
_train_total = time.time() - _train_start

logger.info(f"Training complete in {_train_total/3600:.2f}h")
logger.info(f"Trainer stats: {trainer_stats}")

# Save LoRA adapter
logger.info(f"Saving LoRA adapter to {LORA_SAVE_PATH} ...")
model.save_pretrained(LORA_SAVE_PATH)
tokenizer.save_pretrained(LORA_SAVE_PATH)
logger.info("Done!")
