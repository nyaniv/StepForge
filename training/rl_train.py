"""
Reinforcement Learning refinement with GRPO.

From paper Section 3.3 and Section 4.4.

Cold-starts from the SFT checkpoint.  Uses the Scaled Chamfer Distance reward
(reward/scd_reward.py) to explicitly optimize geometric fidelity.

Hyperparameters (from paper):
  - num_generations: 8  (reduce to 4 on single GPU)
  - kl_coef: 0.02
  - entropy_coef: 0.005
  - learning_rate: 3e-6
  - max_steps: 80  (increase to 160 on single GPU)

The RL prompt is identical to SFT but uses LIVE RAG retrieval (not pre-computed)
so the retriever is used dynamically when building each batch.

Prerequisites:
    python training/sft_train.py --config configs/config.yaml  (must complete first)

Usage:
    python training/rl_train.py --config configs/config.yaml
    python training/rl_train.py --config configs/config.yaml --sft-checkpoint path/to/sft
"""

import argparse
import glob
import inspect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Route HF downloads to the network volume — the container disk is too small.
# Must be set before any transformers/huggingface_hub imports.
if "HF_HOME" not in os.environ:
    _vol = os.environ.get("VOLUME", "/runpod-volume")
    os.environ["HF_HOME"] = os.path.join(_vol, ".hf-cache")

import torch
from datasets import Dataset
from loguru import logger
from omegaconf import OmegaConf
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from retrieval.retriever import Retriever
from reward.scd_reward import compute_reward, compute_parse_reward


# ── Prompt format (identical to SFT) ──────────────────────────────────────────

SYSTEM_MSG = (
    "Given the object description and relevant CAD data, "
    "generate the corresponding STEP file."
)


def format_prompt(caption: str, retrieved_step: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_MSG}\n"
        f"<|user|>\n"
        f"caption: {caption}\n"
        f"retrieved step file:\n{retrieved_step}\n"
        f"<|assistant|>\n"
    )


# ── Format reward (completion bonus) ──────────────────────────────────────────

def format_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """
    Small credit (0.2) for generating a syntactically complete STEP file.
    Gives GRPO gradient signal even when geometry is wrong, encouraging the
    model to learn to terminate with END-ISO-10303-21; before the geometric
    reward can kick in.
    """
    return [0.2 if "END-ISO-10303-21;" in c else 0.0 for c in completions]


# ── Geometry reward ────────────────────────────────────────────────────────────

def make_reward_fn(text2cad_src: str, delta_low: float, delta_high: float,
                   n_points: int):
    """Return a GRPO-compatible reward function with closed-over config."""

    def reward_fn(completions: list[str], ground_truth_step: list[str],
                  **kwargs) -> list[float]:
        return [
            compute_reward(
                gen, gt,
                n_points=n_points,
                delta_low=delta_low,
                delta_high=delta_high,
                text2cad_src=text2cad_src,
            )
            for gen, gt in zip(completions, ground_truth_step)
        ]

    return reward_fn


# ── Parse reward (OCP parse success) ──────────────────────────────────────────

def make_parse_reward_fn(text2cad_src: str):
    """Return a GRPO-compatible parse reward function."""

    def parse_reward_fn(completions: list[str], **kwargs) -> list[float]:
        return [
            compute_parse_reward(gen, text2cad_src=text2cad_src)
            for gen in completions
        ]

    return parse_reward_fn


# ── Build RL dataset with live RAG ────────────────────────────────────────────

def build_rl_dataset(train_jsonl: str, retriever: Retriever,
                     tokenizer, max_completion_length: int) -> Dataset:
    """
    Build the RL training dataset.
    Only includes examples where the GT STEP fits within max_completion_length
    tokens — these are the only examples the model can possibly complete and
    earn non-zero reward on.
    """
    with open(train_jsonl) as f:
        records = [json.loads(l) for l in f]
    logger.info(f"Building RL dataset from {len(records)} examples (live RAG)...")

    data = []
    skipped = 0
    for record in records:
        step_ids = tokenizer(record["step"], add_special_tokens=False)["input_ids"]
        if len(step_ids) > max_completion_length:
            skipped += 1
            continue
        retrieved = retriever.retrieve(record["caption"], exclude_uid=record["uid"])
        # Truncate retrieved STEP to 500 tokens — same as SFT — so the model
        # cannot copy the context and must generate original geometry.
        retrieved_ids = tokenizer(retrieved["step"], add_special_tokens=False)["input_ids"]
        if len(retrieved_ids) > 500:
            retrieved_step = tokenizer.decode(retrieved_ids[:500], skip_special_tokens=True)
        else:
            retrieved_step = retrieved["step"]
        prompt = format_prompt(record["caption"], retrieved_step)
        data.append({
            "prompt": prompt,
            "ground_truth_step": record["step"],
        })

    logger.info(
        f"RL dataset: {len(data)} examples kept, {skipped} skipped "
        f"(GT step > {max_completion_length} tokens)"
    )
    return Dataset.from_list(data)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RL (GRPO) training for STEP-LLM")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--sft-checkpoint",
        default=None,
        help="Path to SFT checkpoint (defaults to config.paths.sft_checkpoint_dir/final)",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise EnvironmentError("HUGGINGFACE_TOKEN environment variable not set.")

    sft_checkpoint = args.sft_checkpoint or os.path.join(
        cfg.paths.sft_checkpoint_dir, "final"
    )
    if not os.path.exists(sft_checkpoint):
        raise FileNotFoundError(
            f"SFT checkpoint not found at {sft_checkpoint}. "
            "Run training/sft_train.py first."
        )

    # ── Load SFT model (cold-start for RL) ──────────────────────────────────
    # Use standard transformers + bitsandbytes instead of Unsloth to avoid
    # Unsloth's fast_forward_inference shape mismatch during GRPO generation.
    logger.info(f"Loading SFT checkpoint from {sft_checkpoint}...")
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
        attn_implementation="sdpa",   # eager materializes O(seq²) attn matrix → OOM
    )
    model = PeftModel.from_pretrained(base_model, sft_checkpoint, is_trainable=True)
    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # GRPO requires left-padding

    # ── Load retriever for live RAG ──────────────────────────────────────────
    retriever = Retriever(
        index_path=cfg.paths.faiss_index_path,
        metadata_path=cfg.paths.faiss_metadata_path,
        model_name=cfg.retrieval.model,
    )

    # ── Build RL dataset ─────────────────────────────────────────────────────
    train_jsonl = os.path.join(cfg.paths.processed_dir, "train.jsonl")
    rl_dataset = build_rl_dataset(
        train_jsonl, retriever, tokenizer, max_completion_length=4096
    )
    rl_dataset = rl_dataset.shuffle(seed=42)

    # ── Reward functions ─────────────────────────────────────────────────────
    parse_reward_fn = make_parse_reward_fn(text2cad_src=cfg.paths.text2cad_src)
    reward_fn = make_reward_fn(
        text2cad_src=cfg.paths.text2cad_src,
        delta_low=cfg.rl.reward.delta_low,
        delta_high=cfg.rl.reward.delta_high,
        n_points=cfg.rl.reward.n_sample_points,
    )

    # ── GRPO config (paper Section 4.4) ─────────────────────────────────────
    # Note: TRL >=0.9 renamed kl_coef → beta; pass both for compatibility.
    os.makedirs(cfg.paths.rl_checkpoint_dir, exist_ok=True)

    grpo_params = inspect.signature(GRPOConfig.__init__).parameters
    kl_kwarg = "beta" if "beta" in grpo_params else "kl_coef"
    optional_kwargs = {}
    if "entropy_coef" in grpo_params:
        optional_kwargs["entropy_coef"] = cfg.rl.entropy_coef

    grpo_config = GRPOConfig(
        output_dir=cfg.paths.rl_checkpoint_dir,
        num_generations=cfg.rl.num_generations,
        **{kl_kwarg: cfg.rl.kl_coef},
        **optional_kwargs,
        learning_rate=cfg.rl.learning_rate,
        per_device_train_batch_size=cfg.rl.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.rl.gradient_accumulation_steps,
        max_steps=cfg.rl.max_steps,
        max_completion_length=4096,   # covers 40.9% of examples; sdpa fits 80GB
        bf16=True,
        logging_steps=1,
        save_steps=20,
        report_to="none",
    )

    model.config.use_cache = False
    model.base_model.model.config.use_cache = False

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward_fn, parse_reward_fn, reward_fn],
        args=grpo_config,
        train_dataset=rl_dataset,
    )

    logger.info("Starting GRPO RL training...")
    rl_checkpoints = sorted(
        glob.glob(os.path.join(cfg.paths.rl_checkpoint_dir, "checkpoint-*")),
        key=lambda x: int(x.rsplit("-", 1)[-1]),
    )
    rl_resume_from = rl_checkpoints[-1] if rl_checkpoints else None
    if rl_resume_from:
        logger.info(f"Resuming RL from checkpoint: {rl_resume_from}")
    trainer.train(resume_from_checkpoint=rl_resume_from)

    final_path = os.path.join(cfg.paths.rl_checkpoint_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"RL training complete. Model saved to {final_path}")


if __name__ == "__main__":
    main()
