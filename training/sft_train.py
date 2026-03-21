"""
Supervised Fine-Tuning (SFT) with LoRA via Unsloth.

From paper Sections 3.2 and 4.1.

Model: meta-llama/Llama-3.2-3B-Instruct (best per paper Table 3)
LoRA: r=16, alpha=32, dropout=0.05, all attention + MLP projections
Training: 10 epochs, batch=16 (2×8 grad accum), lr=2e-4, linear schedule

Prompt format (must match paper exactly — same for SFT, RL, and inference):
    <|system|>
    Given the object description and relevant CAD data, generate the corresponding STEP file.
    <|user|>
    caption: {caption}
    retrieved step file:
    {retrieved_step}
    <|assistant|>
    {ground_truth_step}

Loss masking: cross-entropy only on {ground_truth_step} tokens.
All prompt tokens (system + user + retrieved STEP) are masked to -100.

Prerequisites:
    export HUGGINGFACE_TOKEN=<your_token>
    python data/build_dataset.py --config configs/config.yaml
    python retrieval/build_index.py --config configs/config.yaml
    python data/precompute_rag.py --config configs/config.yaml

Usage:
    python training/sft_train.py --config configs/config.yaml
"""

import argparse
import glob
import json
import os
import sys

# Route HF downloads to the network volume — the container disk is too small.
# Must be set before any transformers/huggingface_hub imports.
if "HF_HOME" not in os.environ:
    _vol = os.environ.get("VOLUME", "/runpod-volume")
    os.environ["HF_HOME"] = os.path.join(_vol, ".hf-cache")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import Dataset
from loguru import logger
from omegaconf import OmegaConf
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


# Truncate retrieved STEP to this many tokens to leave room for GT STEP + EOS.
# With max_seq_length=8192: prompt ≈ 1100 tokens, leaving ~7090 tokens for GT.
MAX_RETRIEVED_TOKENS = 500

# ── Prompt helpers ─────────────────────────────────────────────────────────────

SYSTEM_MSG = (
    "Given the object description and relevant CAD data, "
    "generate the corresponding STEP file."
)


def build_prompt(caption: str, retrieved_step: str, ground_truth_step: str = "") -> str:
    """Build the full prompt string (paper format)."""
    prompt = (
        f"<|system|>\n{SYSTEM_MSG}\n"
        f"<|user|>\n"
        f"caption: {caption}\n"
        f"retrieved step file:\n{retrieved_step}\n"
        f"<|assistant|>\n"
    )
    if ground_truth_step:
        prompt += ground_truth_step
    return prompt


def tokenize_and_mask(example: dict, tokenizer, max_seq_length: int) -> dict | None:
    """
    Tokenize and apply loss masking.

    Two key fixes vs. the original:
      1. Truncates retrieved_step to MAX_RETRIEVED_TOKENS so the GT STEP fits
         within max_seq_length (original was truncating the GT STEP instead).
      2. Appends EOS after the GT STEP so the model learns to terminate.

    Skips examples where the full sequence (with EOS) still exceeds max_seq_length
    after retrieved_step truncation, so every training example ends with EOS and
    the model sees a complete STEP file.
    """
    # Truncate retrieved_step to leave room for the full GT STEP + EOS
    retrieved_ids = tokenizer(
        example["retrieved_step"], add_special_tokens=False
    )["input_ids"]
    if len(retrieved_ids) > MAX_RETRIEVED_TOKENS:
        retrieved_step = tokenizer.decode(
            retrieved_ids[:MAX_RETRIEVED_TOKENS], skip_special_tokens=True
        )
    else:
        retrieved_step = example["retrieved_step"]

    prompt = build_prompt(example["caption"], retrieved_step)
    full   = build_prompt(example["caption"], retrieved_step, example["step"])

    prompt_ids   = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids_raw = tokenizer(full,   add_special_tokens=False)["input_ids"]

    # Always append EOS so the model learns to terminate after the STEP file
    full_ids = full_ids_raw + [tokenizer.eos_token_id]

    # Skip if the complete sequence (with EOS) doesn't fit — avoids training on
    # truncated outputs where the model would never see the terminator
    if len(full_ids) > max_seq_length:
        logger.warning(
            f"Skipping example (too long even after retrieved_step truncation): "
            f"uid={example.get('uid', '?')}, len={len(full_ids)}, max={max_seq_length}"
        )
        return None

    prompt_len = min(len(prompt_ids), len(full_ids))

    if prompt_len >= len(full_ids) - 1:
        logger.warning(
            f"Skipping example (prompt fills context): uid={example.get('uid', '?')}, "
            f"prompt_len={len(prompt_ids)}, max_seq_length={max_seq_length}"
        )
        return None

    labels = [-100] * prompt_len + full_ids[prompt_len:]

    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SFT training for STEP-LLM")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override num_train_epochs from config (e.g. 3 for quick fix SFT)",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HUGGINGFACE_TOKEN environment variable not set. "
            "Required for meta-llama/Llama-3.2-3B-Instruct gated access."
        )

    # ── Load model (standard transformers + peft, no Unsloth) ───────────────
    logger.info(f"Loading {cfg.model.base_model}...")
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
    base_model.config.use_cache = False
    base_model.enable_input_require_grads()  # required for gradient checkpointing with PEFT

    # Load from existing final checkpoint if available (continued training),
    # otherwise create a new LoRA adapter from scratch.
    final_path = os.path.join(cfg.paths.sft_checkpoint_dir, "final")
    if os.path.exists(final_path):
        logger.info(f"Loading existing SFT adapter from {final_path} for continued training...")
        model = PeftModel.from_pretrained(base_model, final_path, is_trainable=True)
    else:
        logger.info("No existing SFT checkpoint — creating new LoRA adapter from base model...")
        lora_config = LoraConfig(
            r=cfg.model.lora_r,
            lora_alpha=cfg.model.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=cfg.model.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load training data ───────────────────────────────────────────────────
    train_jsonl = os.path.join(cfg.paths.processed_dir, "train_with_rag.jsonl")
    logger.info(f"Loading training data from {train_jsonl}")
    with open(train_jsonl) as f:
        records = [json.loads(l) for l in f]

    def gen():
        for r in records:
            result = tokenize_and_mask(r, tokenizer, cfg.model.max_seq_length)
            if result is not None:
                yield result

    train_dataset = Dataset.from_generator(gen)
    logger.info(f"Training examples: {len(train_dataset)}")

    # ── Training arguments (paper Section 4.1) ───────────────────────────────
    os.makedirs(cfg.paths.sft_checkpoint_dir, exist_ok=True)
    num_epochs = args.epochs if args.epochs is not None else cfg.sft.num_epochs
    logger.info(f"Training for {num_epochs} epochs")
    training_args = TrainingArguments(
        output_dir=cfg.paths.sft_checkpoint_dir,
        per_device_train_batch_size=cfg.sft.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.sft.gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        learning_rate=cfg.sft.learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=cfg.sft.warmup_ratio,
        optim=cfg.sft.optim,
        bf16=True,
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        logging_steps=10,
        dataloader_num_workers=4,
        report_to="none",
    )

    # Use standard Trainer with DataCollatorForSeq2Seq for reliable pre-tokenized
    # data handling with loss masking. SFTTrainer's behavior with pre-tokenized
    # data varies across TRL versions; this approach is stable.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,      # align to 8 for tensor core efficiency
        label_pad_token_id=-100,   # masked tokens are excluded from loss
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=data_collator,
    )

    logger.info("Starting SFT training...")
    checkpoints = sorted(glob.glob(os.path.join(cfg.paths.sft_checkpoint_dir, "checkpoint-*")),
                         key=lambda x: int(x.rsplit("-", 1)[-1]))
    resume_from = checkpoints[-1] if checkpoints else None
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
    trainer.train(resume_from_checkpoint=resume_from)

    final_path = os.path.join(cfg.paths.sft_checkpoint_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"SFT complete. Model saved to {final_path}")


if __name__ == "__main__":
    main()
