"""
STEP-LLM SFT — Multi-GPU (8× H100, torchrun DDP)
==================================================
Matches paper Section 4.1: 4×H100, 10 epochs, lr=2e-4, batch=16.
On Gautschi 8×H100 we use the same effective batch size (batch=2 per GPU ×
grad_accum=1 × 8 GPUs = 16 total) for identical training dynamics.

Key differences from llama3_SFT_response.py (Unsloth single-GPU):
  - Standard HuggingFace Trainer + PEFT (no Unsloth)
  - FlashAttention-2 enabled (H100 native)
  - torchrun DDP: 8 processes, 1 GPU each
  - Only rank 0 formats/saves datasets and writes logs to file
  - Same manual label masking logic (chat template, _ASSISTANT_HEADER_IDS)

Launch:
    torchrun --standalone --nproc_per_node=8 training/sft_multigpu.py \\
        --config configs/config_gautschi.yaml

Or via SLURM:
    sbatch slurm_sft_multigpu_gautschi.sh
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "HF_HOME" not in os.environ:
    _scratch = os.environ.get("SCRATCH", "")
    if _scratch:
        os.environ["HF_HOME"] = os.path.join(_scratch, ".hf-cache")
    else:
        os.environ["HF_HOME"] = os.path.join(os.environ.get("VOLUME", "/runpod-volume"), ".hf-cache")

import argparse
import csv
import glob as _glob
import hashlib
import json
import math
import time

import torch
from datasets import Dataset
from loguru import logger
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

# ── Distributed context ───────────────────────────────────────────────────────
local_rank  = int(os.environ.get("LOCAL_RANK", 0))
world_size  = int(os.environ.get("WORLD_SIZE", 1))
is_rank0    = local_rank == 0

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="STEP-LLM SFT (multi-GPU)")
parser.add_argument("--config", default="configs/config_gautschi.yaml")
parser.add_argument("--output-dir", default=None,
                    help="Override cfg.paths.sft_checkpoint_dir (used by SLURM for job-namespaced runs)")
parser.add_argument("--per-device-batch", type=int, default=None,
                    help="Override cfg.sft.per_device_train_batch_size (e.g. 4 for 4-GPU runs)")
parser.add_argument("--max-steps", type=int, default=None,
                    help="Override num_train_epochs with a fixed step count (e.g. 20 for smoke tests)")
args, _ = parser.parse_known_args()

cfg_path = args.config if os.path.isabs(args.config) else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.config
)
cfg = OmegaConf.load(cfg_path)

BASE_MODEL_PATH = cfg.model.base_model
OUTPUT_DIR      = args.output_dir or cfg.paths.sft_checkpoint_dir
LORA_SAVE_PATH  = os.path.join(OUTPUT_DIR, "final")

# Auto-detect data format:
#   main branch:            train_with_rag.jsonl  (fields: step, retrieved_step)
#   refined-variant branch: train.json            (fields: output, relavant_step_file)
_train_jsonl = os.path.join(cfg.paths.processed_dir, "train_with_rag.jsonl")
_train_json  = os.path.join(cfg.paths.processed_dir, "train.json")
if os.path.exists(_train_jsonl):
    TRAIN_JSON   = _train_jsonl
    TEST_JSON    = os.path.join(cfg.paths.processed_dir, "test.jsonl")
    _STEP_FIELD  = "step"
    _RET_FIELD   = "retrieved_step"
    _LOAD_JSON   = False  # JSONL format
else:
    TRAIN_JSON   = _train_json
    TEST_JSON    = os.path.join(cfg.paths.processed_dir, "test.json")
    _STEP_FIELD  = "output"
    _RET_FIELD   = "relavant_step_file"
    _LOAD_JSON   = True   # JSON array format

max_seq_length = cfg.model.max_seq_length
MAX_RETRIEVED_TOKENS = int(
    cfg.sft.get("max_retrieved_tokens", None)
    or getattr(cfg.model, "max_retrieved_tokens", None)
    or max_seq_length  # no truncation beyond context window (paper spec)
)

# ── Logging — only rank 0 writes to file ─────────────────────────────────────
logger.remove()
if is_rank0:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _log_path = os.path.join(OUTPUT_DIR, "sft_train.log")
    logger.add(_log_path, level="DEBUG",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
               enqueue=True)
    logger.info(f"Logging to {_log_path}")
logger.add(sys.stdout, level="INFO",
           format=f"{{time:HH:mm:ss}} | rank{local_rank} | {{level}} | {{message}}")

if is_rank0:
    logger.info(f"World size: {world_size}  |  Local rank: {local_rank}")
    logger.info(f"Config: {cfg_path}")
    logger.info(f"Base model: {BASE_MODEL_PATH}")
    logger.info(f"Data format: {'JSON array' if _LOAD_JSON else 'JSONL'}  "
                f"step_field='{_STEP_FIELD}'  ret_field='{_RET_FIELD}'")
    logger.info(f"Train: {TRAIN_JSON}")
    logger.info(f"max_seq_length={max_seq_length}  MAX_RETRIEVED_TOKENS={MAX_RETRIEVED_TOKENS}")

# ── Load tokenizer ────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # SFT uses right-padding (left-padding is for generation)

# ── Validate assistant header token IDs ──────────────────────────────────────
_ASSISTANT_HEADER_IDS = [128006, 78191, 128007, 271]
_ASSISTANT_HEADER_LEN = len(_ASSISTANT_HEADER_IDS)
_HEADER_STR = "<|start_header_id|>assistant<|end_header_id|>\n\n"
_actual = tokenizer.encode(_HEADER_STR, add_special_tokens=False)
if _actual != _ASSISTANT_HEADER_IDS:
    raise RuntimeError(
        f"_ASSISTANT_HEADER_IDS mismatch!\n"
        f"  Expected : {_ASSISTANT_HEADER_IDS}\n"
        f"  Got      : {_actual}\n"
        f"Update _ASSISTANT_HEADER_IDS in this script to match."
    )
if is_rank0:
    logger.info(f"Tokenizer header IDs validated: {_ASSISTANT_HEADER_IDS}")


def _find_response_start(input_ids: list) -> int:
    for i in range(len(input_ids) - _ASSISTANT_HEADER_LEN + 1):
        if input_ids[i:i + _ASSISTANT_HEADER_LEN] == _ASSISTANT_HEADER_IDS:
            return i + _ASSISTANT_HEADER_LEN
    return -1


def _build_user_message(caption: str, retrieved: str) -> str:
    return (
        "You are a CAD model generation assistant trained to produce STEP (.step) files "
        "based on textual descriptions. Given the following object description and relevant "
        "retrieved CAD data, generate a STEP file that accurately represents the described object.\n\n"
        f"### caption:\n{caption}\n\n"
        f"### retrieved relevant step file:\n{retrieved}"
    )


def formatting_prompts_func(examples):
    instructions = examples["caption"]
    outputs      = examples[_STEP_FIELD]
    inputs       = examples.get(_RET_FIELD, [""] * len(instructions))

    all_input_ids, all_attention_masks, all_labels, all_texts = [], [], [], []

    for caption, retrieved, output in zip(instructions, inputs, outputs):
        # Truncate retrieved STEP if needed (no-op when MAX_RETRIEVED_TOKENS == max_seq_length)
        ret_ids = tokenizer(retrieved or "", add_special_tokens=False)["input_ids"]
        if len(ret_ids) > MAX_RETRIEVED_TOKENS:
            retrieved = tokenizer.decode(ret_ids[:MAX_RETRIEVED_TOKENS])

        messages = [
            {"role": "user",      "content": _build_user_message(caption or "", retrieved or "")},
            {"role": "assistant", "content": output or ""},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        encoded = tokenizer(text, truncation=True, max_length=max_seq_length,
                            add_special_tokens=False)
        input_ids      = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        response_start = _find_response_start(input_ids)
        if response_start == -1:
            raise RuntimeError(
                f"Assistant header {_ASSISTANT_HEADER_IDS} not found in tokenized sequence "
                f"(len={len(input_ids)}). Aborting to avoid training on all-masked examples."
            )
        labels = [-100] * response_start + input_ids[response_start:]

        _fmt_stats["seq_lengths"].append(len(input_ids))
        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)
        all_texts.append(text)

    return {"input_ids": all_input_ids, "attention_mask": all_attention_masks,
            "labels": all_labels, "text": all_texts}


def _log_and_check_fmt_stats(tag: str, ds, is_train: bool):
    """Derive stats from the mapped dataset directly (avoids cross-process mutation issues)."""
    if len(ds) == 0:
        raise RuntimeError(f"[{tag}] No examples formatted — dataset is empty.")
    lengths = sorted(len(ids) for ids in ds["input_ids"])
    n = len(lengths)
    over = sum(1 for l in lengths if l > max_seq_length)
    logger.info(f"[{tag}] {n} examples  |  "
                f"p50_len={lengths[n//2]}  p90_len={lengths[int(n*0.9)]}  max_len={lengths[-1]}  "
                f"over_limit={over}")


# ── Dataset — only rank 0 formats; others load from disk ─────────────────────
PROMPT_VERSION    = "multigpu_v1"
_FORMATTED_TRAIN  = os.path.join(OUTPUT_DIR, "formatted_train")
_FORMATTED_TEST   = os.path.join(OUTPUT_DIR, "formatted_test")
_CACHE_KEY_PATH   = os.path.join(OUTPUT_DIR, ".formatted_cache_key")
_DATASET_DONE_FLAG = os.path.join(OUTPUT_DIR, ".dataset_ready")


def _compute_cache_key() -> str:
    parts = []
    for p in (TRAIN_JSON, TEST_JSON):
        try:
            parts.append(f"{p}:{os.path.getmtime(p)}:{os.path.getsize(p)}")
        except OSError:
            parts.append(f"{p}:missing")
    parts += [f"max_seq={max_seq_length}", f"max_ret={MAX_RETRIEVED_TOKENS}",
              f"model={BASE_MODEL_PATH}", f"prompt_version={PROMPT_VERSION}"]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


_current_key = _compute_cache_key()
_stored_key  = open(_CACHE_KEY_PATH).read().strip() if os.path.exists(_CACHE_KEY_PATH) else ""
_cache_valid = (os.path.exists(_FORMATTED_TRAIN) and os.path.exists(_FORMATTED_TEST)
                and _stored_key == _current_key)

if is_rank0:
    if _cache_valid:
        logger.info("Loading pre-formatted datasets from disk (cache valid)...")
        dataset      = Dataset.load_from_disk(_FORMATTED_TRAIN)
        test_dataset = Dataset.load_from_disk(_FORMATTED_TEST)
        logger.info(f"  Train: {len(dataset)}  |  Test: {len(test_dataset)}")
    else:
        logger.info(f"Formatting datasets (cache miss)...")
        logger.info(f"  Loading train from {TRAIN_JSON}  (format={'JSON array' if _LOAD_JSON else 'JSONL'})")
        logger.info(f"  Fields: step='{_STEP_FIELD}'  retrieved='{_RET_FIELD}'")
        with open(TRAIN_JSON) as f:
            train_records = json.load(f) if _LOAD_JSON else [json.loads(l) for l in f if l.strip()]
        dataset = Dataset.from_list(train_records); del train_records
        logger.info(f"  Raw train: {len(dataset)}")

        logger.info(f"  Loading test from {TEST_JSON}")
        with open(TEST_JSON) as f:
            test_records = json.load(f) if _LOAD_JSON else [json.loads(l) for l in f if l.strip()]
        test_dataset = Dataset.from_list(test_records); del test_records
        logger.info(f"  Raw test: {len(test_dataset)}")

        dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=8)
        _log_and_check_fmt_stats("train", dataset, is_train=True)

        test_dataset = test_dataset.map(formatting_prompts_func, batched=True, num_proc=8)
        _log_and_check_fmt_stats("test", test_dataset, is_train=False)

        dataset.save_to_disk(_FORMATTED_TRAIN)
        test_dataset.save_to_disk(_FORMATTED_TEST)
        with open(_CACHE_KEY_PATH, "w") as f:
            f.write(_current_key)
        logger.info(f"  Saved formatted datasets to disk.")

    if len(dataset) == 0:
        raise RuntimeError(f"Train dataset is empty. Check {TRAIN_JSON}.")
    if len(test_dataset) == 0:
        raise RuntimeError(f"Test dataset is empty. Check {TEST_JSON}.")

    logger.info(f"Train: {len(dataset)}  |  Test: {len(test_dataset)}")

    # Label masking sanity check
    logger.info("Checking manual label masking on first 8 examples...")
    n_bad = 0
    for i in range(min(8, len(dataset))):
        labels   = dataset[i]["labels"]
        unmasked = sum(1 for l in labels if l != -100)
        if unmasked == 0:
            n_bad += 1
            logger.error(f"  Example {i}: ALL {len(labels)} labels are -100!")
        else:
            logger.info(f"  Example {i}: {unmasked}/{len(labels)} unmasked ({100*unmasked/len(labels):.1f}%)")
    if n_bad > 0:
        raise RuntimeError(f"Label masking broken: {n_bad}/8 examples fully masked.")
    logger.info("Label masking check passed.")

    # Signal other ranks
    open(_DATASET_DONE_FLAG, "w").close()
else:
    # Non-rank-0: wait for rank 0 to finish formatting
    logger.info(f"Rank {local_rank}: waiting for rank 0 to prepare dataset...")
    while not os.path.exists(_DATASET_DONE_FLAG):
        time.sleep(5)
    dataset      = Dataset.load_from_disk(_FORMATTED_TRAIN)
    test_dataset = Dataset.load_from_disk(_FORMATTED_TEST)
    logger.info(f"Rank {local_rank}: loaded dataset ({len(dataset)} train, {len(test_dataset)} test)")


# ── Load model ────────────────────────────────────────────────────────────────
if is_rank0:
    logger.info(f"Loading base model: {BASE_MODEL_PATH}")

attn_impl = "sdpa"
try:
    import flash_attn  # noqa: F401
    attn_impl = "flash_attention_2"
    if is_rank0:
        logger.info("FlashAttention-2 enabled")
except ImportError:
    if is_rank0:
        logger.warning("flash-attn not installed — using sdpa")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map={"": local_rank},
    attn_implementation=attn_impl,
)
model.config.use_cache = False  # required for gradient checkpointing

# Attach LoRA
lora_config = LoraConfig(
    r=cfg.model.lora_r,
    lora_alpha=cfg.model.lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
if is_rank0:
    model.print_trainable_parameters()
    logger.info(f"LoRA: r={cfg.model.lora_r}  alpha={cfg.model.lora_alpha}")


# ── Per-epoch callback ────────────────────────────────────────────────────────
class VerboseEpochCallback(TrainerCallback):
    def __init__(self):
        self._epoch_start = None
        self._step_losses = []
        self._zero_streak = 0

    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self._epoch_start = time.time()
        self._step_losses = []
        self._zero_streak = 0
        epoch = int(state.epoch or 0) + 1
        logger.info(f"{'='*60}")
        logger.info(f"  EPOCH {epoch} / {args.num_train_epochs} — starting")
        logger.info(f"{'='*60}")

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not logs:
            return
        loss = logs.get("loss")
        if loss is None:
            return
        self._step_losses.append(loss)
        if math.isnan(loss) or math.isinf(loss):
            logger.error(f"[step {state.global_step}] NaN/Inf loss — halting.")
            control.should_training_stop = True
        elif loss == 0.0:
            self._zero_streak += 1
            logger.warning(f"[step {state.global_step}] Zero loss. Streak: {self._zero_streak}")
            if self._zero_streak >= 20:
                logger.error("ABORTING: 20 consecutive zero-loss steps — masking broken.")
                control.should_training_stop = True
        else:
            self._zero_streak = 0

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        elapsed = time.time() - (self._epoch_start or time.time())
        epoch   = int(state.epoch or 0)
        losses  = self._step_losses
        if losses:
            avg = sum(losses) / len(losses)
            logger.info(f"{'='*60}")
            logger.info(f"  EPOCH {epoch} COMPLETE  |  {elapsed/60:.1f} min  |  "
                        f"loss avg={avg:.4f}  min={min(losses):.4f}  max={max(losses):.4f}")
            logger.info(f"{'='*60}")


class LossLoggerCallback(TrainerCallback):
    """Writes every logged step to a CSV for easy plotting later."""

    def __init__(self, csv_path: str):
        self._csv_path = csv_path
        self._file = None
        self._writer = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._file = open(self._csv_path, "a", newline="")
        self._writer = csv.writer(self._file)
        # Write header only if file is empty
        if self._file.tell() == 0:
            self._writer.writerow(["timestamp", "step", "epoch", "loss", "learning_rate", "grad_norm"])
        self._file.flush()

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not logs or self._writer is None:
            return
        loss = logs.get("loss")
        if loss is None:
            return
        self._writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            state.global_step,
            round(state.epoch or 0, 4),
            round(loss, 6),
            round(logs.get("learning_rate", 0), 8),
            round(logs.get("grad_norm", 0), 6),
        ])
        self._file.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self._file:
            self._file.close()


# ── Training ──────────────────────────────────────────────────────────────────
# Effective batch = per_device × grad_accum × world_size
# Paper: batch=16. With 8 GPUs: per_device=2, grad_accum=1 → 2×1×8=16 ✓
per_device_batch = args.per_device_batch or cfg.sft.per_device_train_batch_size
grad_accum       = cfg.sft.gradient_accumulation_steps
effective_batch  = per_device_batch * grad_accum * world_size

if is_rank0:
    logger.info(f"Effective batch size: {per_device_batch} × {grad_accum} × {world_size} = {effective_batch}")
    logger.info(f"epochs={cfg.sft.num_epochs}  lr={cfg.sft.learning_rate}  optim={cfg.sft.optim}")

_smoke = args.max_steps is not None
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=per_device_batch,
    gradient_accumulation_steps=grad_accum,
    gradient_checkpointing=True,
    num_train_epochs=cfg.sft.num_epochs,
    max_steps=args.max_steps if _smoke else -1,  # >0 overrides num_train_epochs
    learning_rate=cfg.sft.learning_rate,
    warmup_ratio=cfg.sft.warmup_ratio,
    lr_scheduler_type="linear",
    bf16=True,
    optim=cfg.sft.optim,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="steps" if _smoke else "epoch",
    save_steps=10 if _smoke else 500,
    eval_strategy="no" if _smoke else "epoch",
    report_to="none",
    seed=3407,
    ddp_find_unused_parameters=False,
    dataloader_num_workers=4,
    remove_unused_columns=False,
)

_loss_csv_path = os.path.join(OUTPUT_DIR, "sft_loss.csv")
callbacks = [VerboseEpochCallback(), LossLoggerCallback(_loss_csv_path)] if is_rank0 else []

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=test_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=callbacks,
)

# Auto-resume from latest checkpoint
_ckpts = sorted(
    _glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")),
    key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1,
)
_resume_from = _ckpts[-1] if _ckpts else None
if is_rank0:
    logger.info(f"Resuming from: {_resume_from}" if _resume_from else "Starting from scratch.")

if is_rank0:
    logger.info("Starting SFT training...")
_train_start = time.time()
trainer.train(resume_from_checkpoint=_resume_from)
_train_total = time.time() - _train_start

if is_rank0:
    logger.info(f"Training complete in {_train_total/3600:.2f}h")
    logger.info(f"Saving LoRA adapter to {LORA_SAVE_PATH} ...")
    model.save_pretrained(LORA_SAVE_PATH)
    tokenizer.save_pretrained(LORA_SAVE_PATH)
    logger.info("Done!")
