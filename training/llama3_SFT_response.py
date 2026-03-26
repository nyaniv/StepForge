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

from omegaconf import OmegaConf
_cfg = OmegaConf.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "config_runpod.yaml"))

BASE_MODEL_PATH = _cfg.model.base_model                           # meta-llama/Llama-3.2-3B-Instruct
TRAIN_JSON      = os.path.join(_cfg.paths.processed_dir, "train.json")
TEST_JSON       = os.path.join(_cfg.paths.processed_dir, "test.json")
LORA_SAVE_PATH  = os.path.join(_cfg.paths.sft_checkpoint_dir, "final")
OUTPUT_DIR      = _cfg.paths.sft_checkpoint_dir
USE_RAG         = True
# ── End configuration ──────────────────────────────────────────────────────────


from unsloth import FastLanguageModel
import torch

max_seq_length = _cfg.model.max_seq_length  # RoPE Scaling is handled automatically by Unsloth
# Token budget for the retrieved STEP context. A full retrieved STEP file is
# ~22k tokens — combined with the target output (~7k) that blows the context
# window and causes train_on_responses_only to mask everything.
# 500 tokens keeps the RAG context meaningful while leaving ~9k tokens for
# the target output within the 16384 limit.
MAX_RETRIEVED_TOKENS = 500
dtype = None            # None = auto-detect (bfloat16 on Ampere+, float16 on older GPUs)
load_in_4bit = False    # Set True to use 4-bit quantisation (reduces VRAM, slight quality loss)

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


def formatting_prompts_func(examples):
    """Format dataset examples into the training prompt."""
    instructions = examples["caption"]
    outputs = examples["output"]
    texts = []

    if USE_RAG:
        inputs = examples["relavant_step_file"]
        for instruction, input_, output in zip(instructions, inputs, outputs):
            # Truncate retrieved STEP to MAX_RETRIEVED_TOKENS.
            # Full retrieved STEP is ~22k tokens; combined with ~7k output tokens
            # it far exceeds the context window and masks all response labels.
            ids = tokenizer(input_, add_special_tokens=False)["input_ids"]
            truncated = tokenizer.decode(ids[:MAX_RETRIEVED_TOKENS])
            text = ABC_PROMPT_RAG.format(instruction, truncated, output) + EOS_TOKEN
            texts.append(text)
    else:
        for instruction, output in zip(instructions, outputs):
            text = ABC_PROMPT_NO_RAG.format(instruction, output) + EOS_TOKEN
            texts.append(text)

    return {"text": texts}


# ── Dataset ─────────────────────────────────────────────────────────────────────
from datasets import load_dataset

# Load from our JSON files produced by data/data_split.py
dataset      = load_dataset("json", data_files=TRAIN_JSON, split="train")
dataset      = dataset.map(formatting_prompts_func, batched=True)

test_dataset = load_dataset("json", data_files=TEST_JSON, split="train")
test_dataset = test_dataset.map(formatting_prompts_func, batched=True)

# ── Training ─────────────────────────────────────────────────────────────────────
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=test_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
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
        report_to="none",       # set to "wandb" for experiment tracking
        save_strategy="steps",
        save_steps=300,
    ),
)

# Train only on model outputs (mask the prompt so loss is only on the STEP data)
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part="### caption:\n",
    response_part="### output:\n",
)

print("\nTraining started...\n")
trainer_stats = trainer.train()
print("\nTraining completed!")
print(trainer_stats)

# Save LoRA adapter
print(f"\nSaving LoRA adapter to {LORA_SAVE_PATH} ...")
model.save_pretrained(LORA_SAVE_PATH)
tokenizer.save_pretrained(LORA_SAVE_PATH)
print("Done!")
