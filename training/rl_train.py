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

import multiprocessing as mp
# Must be set before any CUDA or torch import.  compute_reward() spawns
# subprocesses for OCP tessellation; forking after CUDA init corrupts GPU
# handles in the child and causes non-deterministic hangs.
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # already set (torchrun spawns worker processes before script entry)

import argparse
import glob
import inspect
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Route HF downloads to the network volume — the container disk is too small.
# Must be set before any transformers/huggingface_hub imports.
if "HF_HOME" not in os.environ:
    _vol = os.environ.get("VOLUME", "/runpod-volume")
    os.environ["HF_HOME"] = os.path.join(_vol, ".hf-cache")

import time
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

ABC_PROMPT_RAG = (
    "You are a CAD model generation assistant trained to produce STEP (.step) files "
    "based on textual descriptions. Given the following object description and relevant "
    "retrieved CAD data, generate a STEP file that accurately represents the described object."
    "\n\n\n### caption:\n{}\n\n### retrieved relevant step file:\n{}\n\n### output:\n"
)

MAX_RETRIEVED_TOKENS: int = 500  # overridden per-config below in main(); default matches RunPod/local configs


def format_prompt(caption: str, retrieved_step: str, tokenizer) -> str:
    ids = tokenizer(retrieved_step, add_special_tokens=False)["input_ids"]
    truncated = tokenizer.decode(ids[:MAX_RETRIEVED_TOKENS])
    return ABC_PROMPT_RAG.format(caption, truncated)


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
        with ThreadPoolExecutor(max_workers=len(completions)) as pool:
            futures = [
                pool.submit(
                    compute_reward, gen, gt,
                    n_points=n_points,
                    delta_low=delta_low,
                    delta_high=delta_high,
                    text2cad_src=text2cad_src,
                )
                for gen, gt in zip(completions, ground_truth_step)
            ]
            return [f.result() for f in futures]

    return reward_fn


# ── Parse reward (OCP parse success) ──────────────────────────────────────────

def make_parse_reward_fn(text2cad_src: str):
    """Return a GRPO-compatible parse reward function."""

    def parse_reward_fn(completions: list[str], **kwargs) -> list[float]:
        with ThreadPoolExecutor(max_workers=len(completions)) as pool:
            futures = [
                pool.submit(compute_parse_reward, gen, text2cad_src=text2cad_src)
                for gen in completions
            ]
            return [f.result() for f in futures]

    return parse_reward_fn


# ── Build RL dataset with live RAG ────────────────────────────────────────────

def build_rl_dataset(train_json: str, retriever: Retriever,
                     tokenizer, max_completion_length: int) -> Dataset:
    """
    Build the RL training dataset.
    Only includes examples where the GT STEP fits within max_completion_length
    tokens — these are the only examples the model can possibly complete and
    earn non-zero reward on.
    """
    with open(train_json) as f:
        records = json.load(f)   # JSON array, not JSONL
    logger.info(f"Building RL dataset from {len(records)} examples (live RAG)...")

    data = []
    skipped = 0
    # ~4 chars per token is a safe upper bound for STEP files
    char_limit = max_completion_length * 4
    for i, record in enumerate(records):
        if i % 5000 == 0:
            logger.info(f"  Building RL dataset: {i}/{len(records)} (kept={len(data)}, skipped={skipped})")
        gt_step = record.get("output") or record.get("step") or ""
        # Pre-filter by char length to avoid tokenizing huge files
        if len(gt_step) > char_limit:
            skipped += 1
            continue
        step_ids = tokenizer(gt_step, add_special_tokens=False)["input_ids"]
        if len(step_ids) > max_completion_length:
            skipped += 1
            continue
        uid = record.get("uid") or record.get("id_original") or ""
        retrieved = retriever.retrieve(record["caption"], exclude_uid=uid)
        retrieved_step = retrieved.get("output") or retrieved.get("step") or ""
        prompt = format_prompt(record["caption"], retrieved_step, tokenizer)
        data.append({
            "prompt": prompt,
            "ground_truth_step": gt_step,
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

    # ── Apply config-driven globals ──────────────────────────────────────────
    global MAX_RETRIEVED_TOKENS
    MAX_RETRIEVED_TOKENS = int(getattr(cfg.model, "max_retrieved_tokens", 500))

    # ── Distributed context (set by torchrun on Gautschi) ───────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    use_quantization = bool(getattr(cfg.model, "use_quantization", True))

    # ── File logging ─────────────────────────────────────────────────────────
    os.makedirs(cfg.paths.rl_checkpoint_dir, exist_ok=True)
    _log_path = os.path.join(cfg.paths.rl_checkpoint_dir, "rl_train.log")
    logger.add(_log_path, level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", enqueue=True)
    logger.add(sys.stdout, level="INFO",  format="{time:HH:mm:ss} | {level} | {message}")
    logger.info(f"Logging to {_log_path}")
    _train_start = time.time()
    # ── End file logging ─────────────────────────────────────────────────────

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
    # Two paths:
    #   A) Gautschi / multi-GPU / no-quantization:
    #      Load in bf16 with device_map per local rank. 3B model = ~6 GB in bf16,
    #      well within H100 80 GB. Use FlashAttention-2 for long-context efficiency.
    #   B) Single-GPU / quantized (RunPod A100, local):
    #      4-bit BnB quantization + device_map="auto" (original path).
    logger.info(f"Loading SFT checkpoint from {sft_checkpoint}...")
    logger.info(f"  use_quantization={use_quantization}  is_distributed={is_distributed}  "
                f"local_rank={local_rank}  world_size={world_size}")

    if not use_quantization:
        # ── Path A: bf16, no quantization (Gautschi H100, multi-GPU) ────────
        # device_map={"": local_rank} places the entire model on this process's GPU.
        # DDP handles gradient sync across ranks via GRPOTrainer + accelerate.
        attn_impl = "flash_attention_2"  # H100 + flash-attn package required
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            logger.warning("flash-attn not installed — falling back to sdpa")
            attn_impl = "sdpa"

        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model.base_model,
            torch_dtype=torch.bfloat16,
            device_map={"": local_rank},
            token=hf_token,
            attn_implementation=attn_impl,
        )
    else:
        # ── Path B: 4-bit quantization (single GPU, RunPod / local) ─────────
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
    train_json = os.path.join(cfg.paths.processed_dir, "train.json")
    rl_dataset = build_rl_dataset(
        train_json, retriever, tokenizer, max_completion_length=4096
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
        report_to="tensorboard",
        seed=42,
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

    _train_elapsed = time.time() - _train_start
    logger.info(f"RL training complete in {_train_elapsed/3600:.2f}h")

    final_path = os.path.join(cfg.paths.rl_checkpoint_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
