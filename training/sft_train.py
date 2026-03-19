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

import unsloth  # must be first to patch transformers before import
import torch
from datasets import Dataset
from loguru import logger
from omegaconf import OmegaConf
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from unsloth import FastLanguageModel


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


def tokenize_and_mask(example: dict, tokenizer, max_seq_length: int) -> dict:
    """
    Tokenize and apply loss masking.
    Labels for prompt tokens are set to -100 (excluded from loss).
    """
    prompt = build_prompt(example["caption"], example["retrieved_step"])
    full   = build_prompt(example["caption"], example["retrieved_step"],
                          example["step"])

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids   = tokenizer(
        full,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False,
    )["input_ids"]

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

    # Pad/truncate labels to same length as input_ids
    labels = labels[:max_seq_length]

    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SFT training for STEP-LLM")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HUGGINGFACE_TOKEN environment variable not set. "
            "Required for meta-llama/Llama-3.2-3B-Instruct gated access."
        )

    # ── Load model with Unsloth ──────────────────────────────────────────────
    logger.info(f"Loading {cfg.model.base_model} with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.base_model,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=True,
        token=hf_token,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=cfg.model.lora_dropout,
        bias="none",
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load training data ───────────────────────────────────────────────────
    train_jsonl = os.path.join(cfg.paths.processed_dir, "train_with_rag.jsonl")
    logger.info(f"Loading training data from {train_jsonl}")
    records = [json.loads(l) for l in open(train_jsonl)]

    def gen():
        for r in records:
            yield tokenize_and_mask(r, tokenizer, cfg.model.max_seq_length)

    train_dataset = Dataset.from_generator(gen)
    logger.info(f"Training examples: {len(train_dataset)}")

    # ── Training arguments (paper Section 4.1) ───────────────────────────────
    os.makedirs(cfg.paths.sft_checkpoint_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=cfg.paths.sft_checkpoint_dir,
        per_device_train_batch_size=cfg.sft.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.sft.gradient_accumulation_steps,
        num_train_epochs=cfg.sft.num_epochs,
        learning_rate=cfg.sft.learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=cfg.sft.warmup_ratio,
        optim=cfg.sft.optim,
        bf16=True,
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
