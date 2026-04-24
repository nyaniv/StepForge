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

# Route HF downloads to scratch/network volume — home quota is too small for
# multi-GB model weights.  Must be set before any transformers/huggingface_hub
# imports.  Priority: existing env var → $SCRATCH (Gautschi) → VOLUME (RunPod).
if "HF_HOME" not in os.environ:
    _scratch = os.environ.get("SCRATCH", "")
    if _scratch:
        os.environ["HF_HOME"] = os.path.join(_scratch, ".hf-cache")
    else:
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


# ── Prompt format (must match SFT chat template format) ───────────────────────
# SFT trains with apply_chat_template — RL must use the same format so the
# model sees the same prompt structure it was trained on.

MAX_RETRIEVED_TOKENS: int = 16384  # overridden per-config in main(); no truncation by default (paper spec)


def _build_user_message(caption: str, retrieved_step: str) -> str:
    return (
        "You are a CAD model generation assistant trained to produce STEP (.step) files "
        "based on textual descriptions. Given the following object description and relevant "
        "retrieved CAD data, generate a STEP file that accurately represents the described object.\n\n"
        f"### caption:\n{caption}\n\n"
        f"### retrieved relevant step file:\n{retrieved_step}"
    )


def format_prompt(caption: str, retrieved_step: str, tokenizer) -> str:
    ids = tokenizer(retrieved_step, add_special_tokens=False)["input_ids"]
    truncated = tokenizer.decode(ids[:MAX_RETRIEVED_TOKENS])
    messages = [
        {"role": "user", "content": _build_user_message(caption, truncated)},
    ]
    # add_generation_prompt=True so the model continues from the assistant turn
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ── Format reward (completion bonus) ──────────────────────────────────────────

def format_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """
    Small credit (0.2) for generating a syntactically complete STEP file.
    Gives GRPO gradient signal even when geometry is wrong, encouraging the
    model to learn to terminate with END-ISO-10303-21; before the geometric
    reward can kick in.
    """
    rewards = [0.2 if "END-ISO-10303-21;" in c else 0.0 for c in completions]
    # Debug: log first completion so we can verify what the model is generating
    if completions:
        logger.debug(f"[format_reward] completion[0][:200]: {repr(completions[0][:200])}")
        logger.debug(f"[format_reward] rewards: {rewards}")
    return rewards


# ── Geometry reward ────────────────────────────────────────────────────────────

def make_reward_fn(text2cad_src: str, delta_low: float, delta_high: float,
                   n_points: int):
    """Return a GRPO-compatible reward function with closed-over config."""

    def reward_fn(completions: list[str], ground_truth_step: list[str],
                  **kwargs) -> list[float]:
        # Model omits the "DATA;\n" prefix because it already appears in the
        # retrieved STEP context. Prepend it so step_to_pointcloud can parse.
        def _fix(s: str) -> str:
            s = s.lstrip()
            if not s.startswith("DATA;") and not s.startswith("ISO-10303-21;"):
                s = "DATA;\n" + s
            return s
        import os as _os
        if _os.environ.get("LOCAL_RANK", "0") == "0":
            sample = completions[0] if completions else ""
            fixed = _fix(sample)
            logger.info(f"[reward_fn] completion[0] first 200 chars: {repr(fixed[:200])}")
        with ThreadPoolExecutor(max_workers=len(completions)) as pool:
            futures = [
                pool.submit(
                    compute_reward, _fix(gen), gt,
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
        def _fix(s: str) -> str:
            s = s.lstrip()
            if not s.startswith("DATA;") and not s.startswith("ISO-10303-21;"):
                s = "DATA;\n" + s
            return s
        with ThreadPoolExecutor(max_workers=len(completions)) as pool:
            futures = [
                pool.submit(compute_parse_reward, _fix(gen), text2cad_src=text2cad_src)
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
        head = f.read(1); f.seek(0)
        if head == "[":
            records = json.load(f)
        else:
            records = [json.loads(l) for l in f if l.strip()]
    logger.info(f"Building RL dataset from {len(records)} examples (live RAG)...")

    data = []
    skipped = 0
    # ~4 chars per token is a safe upper bound for STEP files
    char_limit = max_completion_length * 4
    for i, record in enumerate(records):
        if i % 5000 == 0:
            logger.info(f"  Building RL dataset: {i}/{len(records)} (kept={len(data)}, skipped={skipped})")
        gt_step = record.get("output") or record.get("step") or ""
        if not gt_step:
            raise RuntimeError(
                f"Record {i} has empty ground truth step (both 'output' and 'step' are missing/empty). "
                f"Record uid: {record.get('uid') or record.get('id_original', '?')}. "
                f"Fix the data pipeline before running RL."
            )
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
        if not retrieved_step:
            raise RuntimeError(
                f"Retriever returned empty step for record {i} (uid={uid!r}, caption={record['caption'][:80]!r}). "
                f"Check that the FAISS index is built and cfg.paths.faiss_index_path is correct."
            )
        prompt = format_prompt(record["caption"], retrieved_step, tokenizer)
        data.append({
            "prompt": prompt,
            "ground_truth_step": gt_step,
        })

    logger.info(
        f"RL dataset: {len(data)} examples kept, {skipped} skipped "
        f"(GT step > {max_completion_length} tokens)"
    )
    if len(data) == 0:
        raise RuntimeError(
            f"RL dataset is empty after filtering. All {skipped} records had GT steps "
            f"exceeding max_completion_length={max_completion_length} tokens. "
            f"Increase cfg.rl.max_completion_length or reduce the entity cap."
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
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override cfg.rl.max_steps (e.g. 3 for smoke tests)")
    parser.add_argument("--num-generations", type=int, default=None,
                        help="Override cfg.rl.num_generations (e.g. 2 for smoke tests)")
    parser.add_argument("--use-quantization", action="store_true", default=None,
                        help="Force 4-bit quantization (smoke test on small GPUs)")
    parser.add_argument("--output-dir", default=None,
                        help="Override cfg.paths.rl_checkpoint_dir (e.g. smoke test run dir)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # ── Apply config-driven globals ──────────────────────────────────────────
    global MAX_RETRIEVED_TOKENS
    # Paper does not truncate retrieved context. Fall back to max_seq_length so
    # the only hard limit is the model's context window.
    MAX_RETRIEVED_TOKENS = int(getattr(cfg.model, "max_retrieved_tokens",
                                       getattr(cfg.model, "max_seq_length", 16384)))

    # ── CLI overrides (smoke test / debugging) ───────────────────────────────
    if args.max_steps is not None:
        cfg.rl.max_steps = args.max_steps
    if args.num_generations is not None:
        cfg.rl.num_generations = args.num_generations
    if args.output_dir is not None:
        cfg.paths.rl_checkpoint_dir = args.output_dir

    # ── Distributed context (set by torchrun on Gautschi) ───────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    use_quantization = args.use_quantization if args.use_quantization is not None \
        else bool(getattr(cfg.model, "use_quantization", True))

    # ── File logging ─────────────────────────────────────────────────────────
    # Remove ALL existing handlers first (loguru adds a default stderr sink at
    # import time; repeated restarts via SLURM requeue would stack handlers).
    logger.remove()
    os.makedirs(cfg.paths.rl_checkpoint_dir, exist_ok=True)
    # Only rank 0 writes to the shared log file; all ranks log to stdout with
    # their rank prefix so per-rank output is distinguishable.
    if local_rank == 0:
        _log_path = os.path.join(cfg.paths.rl_checkpoint_dir, "rl_train.log")
        logger.add(_log_path, level="DEBUG",
                   format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                   enqueue=True)
        logger.info(f"Logging to {_log_path}")
    logger.add(sys.stdout, level="INFO",
               format=f"{{time:HH:mm:ss}} | rank{local_rank} | {{level}} | {{message}}")
    _train_start = time.time()
    # ── End file logging ─────────────────────────────────────────────────────

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN environment variable not set.")
    os.environ["HF_TOKEN"] = hf_token  # ensure HF library picks it up

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

    # ── Build RL dataset — rank 0 only, then file-barrier sync ──────────────
    # BUG PREVENTION:
    #   - Only rank 0 loads the Retriever so that the SentenceTransformer
    #     (which auto-selects the first available GPU) is not instantiated by
    #     all 8 processes simultaneously, all targeting cuda:0.
    #   - We use a file-based barrier instead of dist.init_process_group so
    #     we don't race with accelerate's own distributed initialisation inside
    #     GRPOTrainer.
    max_completion_length = int(getattr(cfg.rl, "max_completion_length", 4096))
    # Support both main (train_with_rag.jsonl) and refined-variant (train.json) data formats
    _train_json_jsonl = os.path.join(cfg.paths.processed_dir, "train_with_rag.jsonl")
    _train_json_json  = os.path.join(cfg.paths.processed_dir, "train.json")
    train_json = _train_json_jsonl if os.path.exists(_train_json_jsonl) else _train_json_json
    dataset_cache_path = os.path.join(cfg.paths.rl_checkpoint_dir, "rl_dataset_cache")
    dataset_done_flag  = dataset_cache_path + ".done"

    if local_rank == 0:
        logger.info("Rank 0: building RL dataset with live RAG...")
        retriever = Retriever(
            index_path=cfg.paths.faiss_index_path,
            metadata_path=cfg.paths.faiss_metadata_path,
            model_name=cfg.retrieval.model,
            device="cpu",   # keep SentenceTransformer off GPU 0 — training model is already there
        )
        rl_dataset = build_rl_dataset(
            train_json, retriever, tokenizer,
            max_completion_length=max_completion_length,
        )
        rl_dataset = rl_dataset.shuffle(seed=42)
        rl_dataset.save_to_disk(dataset_cache_path)
        # Signal other ranks that the dataset is ready
        open(dataset_done_flag, "w").close()
        logger.info(f"Rank 0: dataset saved to {dataset_cache_path}")
    else:
        # Wait for rank 0's file-barrier signal (shared filesystem is visible to all nodes)
        logger.info(f"Rank {local_rank}: waiting for rank 0 to build dataset...")
        while not os.path.exists(dataset_done_flag):
            time.sleep(5)
        from datasets import load_from_disk
        rl_dataset = load_from_disk(dataset_cache_path)
        logger.info(f"Rank {local_rank}: dataset loaded ({len(rl_dataset)} examples)")

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
        max_completion_length=max_completion_length,  # from config; matches build_rl_dataset filter
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_steps=20,
        report_to="none",
        seed=42,
        # remove_unused_columns=False is CRITICAL: GRPOTrainer passes reward
        # function kwargs from dataset columns.  The default (True) would silently
        # strip ground_truth_step before it ever reaches the geometry reward fn,
        # making the entire reward signal zero while training appears to proceed.
        remove_unused_columns=False,
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

    # Only rank 0 saves — all 8 DDP ranks hitting save_pretrained on the same
    # path simultaneously causes a race condition and corrupts the checkpoint.
    if local_rank == 0:
        final_path = os.path.join(cfg.paths.rl_checkpoint_dir, "final")
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        logger.info(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
