"""
Reinforcement Learning refinement with GRPO.

From paper Section 3.3 and Section 4.4.

Cold-starts from the SFT checkpoint.  Uses the Scaled Chamfer Distance reward
(reward/scd_reward.py) to explicitly optimize geometric fidelity.

Hyperparameters (from paper §4.4):
  - num_generations: 8
  - kl_coef: 0.02
  - entropy_coef: 0.005   (NB: TRL GRPOConfig has no such param — see W17 warning at runtime)
  - learning_rate: 3e-6
  - max_steps: 80

The RL prompt is identical to SFT but uses LIVE RAG retrieval (not pre-computed)
so the retriever is used dynamically when building each batch.

Prerequisites:
    python training/llama3_SFT_response.py --config configs/config.yaml  (must complete first)

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
import numpy as np
import torch
from datasets import Dataset
from loguru import logger
from omegaconf import OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from retrieval.retriever import Retriever
from reward.scd_reward import RewardConfig, compute_reward


class RewardTelemetryCallback(TrainerCallback):
    """
    Drains the per-step telemetry buffer that scd_geometry_reward populates.
    Writes to tensorboard (via trainer.log) and to metrics.jsonl so
    plot_training_summary.py and CI assertions can read structured data
    instead of hand-transcribing from log text.
    """

    def __init__(self, jsonl_path: str, kl_halt_threshold: float | None = None,
                 resuming: bool = False, is_main: bool = True):
        self.jsonl_path = jsonl_path
        self.kl_halt_threshold = kl_halt_threshold
        # DS-2/F4: only rank 0 truncates the file. Other ranks no-op in on_log.
        if is_main:
            os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
            if not resuming:
                open(jsonl_path, "w").close()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # DS-2c: KL-halt runs on EVERY rank so all set should_training_stop
        # together — TrainerControl is per-rank and a rank-0-only stop would
        # leave other ranks at the next all-reduce until NCCL times out.
        if self.kl_halt_threshold is not None:
            kl = logs.get("kl") or logs.get("objective/kl") or logs.get("kl_div")
            if kl is not None and kl > self.kl_halt_threshold:
                if state.is_world_process_zero:
                    logger.error(f"KL divergence {kl:.4f} > halt threshold {self.kl_halt_threshold} — stopping")
                control.should_training_stop = True
        # DS-2/F4: only rank 0 writes the JSONL.
        if not state.is_world_process_zero:
            return
        merged = dict(logs)
        merged["step"] = state.global_step
        while _telemetry_buf:
            merged.update(_telemetry_buf.pop(0))
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(merged) + "\n")

    def on_step_end(self, args, state, control, **kwargs):
        # SIG-2: HF's CallbackHandler.on_step_begin resets should_save=False each
        # step. Re-assert here so a SIGTERM that lands in the inter-step gap is
        # honoured at the very next save check (runs after this callback).
        if _sigterm_requested:
            control.should_save = True
            control.should_training_stop = True


# ── Prompt format (identical to SFT) ──────────────────────────────────────────

ABC_PROMPT_RAG = (
    "You are a CAD model generation assistant trained to produce STEP (.step) files "
    "based on textual descriptions. Given the following object description and relevant "
    "retrieved CAD data, generate a STEP file that accurately represents the described object."
    "\n\n\n### caption:\n{}\n\n### retrieved relevant step file:\n{}\n\n### output:\n"
)

# W1: Paper does NOT truncate (§3.2). Default 4500 covers typical retrievals;
# override via cfg.rl.max_retrieved_tokens for cost-saving runs.
MAX_RETRIEVED_TOKENS = 4500  # overwritten in main() from config


# Heuristic char-per-token bound for STEP text — skips tokenizer round-trip
# in the common case where the retrieved STEP is well under the cap.
_CHARS_PER_TOKEN_UB = 5

def format_prompt(caption: str, retrieved_step: str, tokenizer) -> str:
    if len(retrieved_step) > MAX_RETRIEVED_TOKENS * _CHARS_PER_TOKEN_UB:
        ids = tokenizer(retrieved_step, add_special_tokens=False)["input_ids"]
        if len(ids) > MAX_RETRIEVED_TOKENS:
            retrieved_step = tokenizer.decode(ids[:MAX_RETRIEVED_TOKENS])
    return ABC_PROMPT_RAG.format(caption, retrieved_step)


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

# Per-step telemetry buffer. Populated by scd_geometry_reward, drained by RewardTelemetryCallback.
_telemetry_buf: list[dict] = []

# SIG-2: set by the SIGTERM handler; re-asserted by RewardTelemetryCallback.on_step_end
# because HF's CallbackHandler.on_step_begin resets control.should_save every step.
_sigterm_requested: bool = False


def make_reward_fn(text2cad_src: str, rcfg: RewardConfig, verbose: bool = False):
    """Return a GRPO-compatible reward function with closed-over config."""

    import math
    from collections import Counter

    _max_workers = min(os.cpu_count() or 4, 16)

    from reward.scd_reward import _GT_PC_CACHE, _GT_PC_CACHE_LOCK, _GT_PC_CACHE_MAX, _gt_cache_key
    from reward.step_to_pointcloud import step_to_pointcloud

    def scd_geometry_reward(completions: list[str], ground_truth_step: list[str],
                            **kwargs) -> list[float]:
        # PERF-1: GRPO sends one prompt's N rollouts here, all sharing the same
        # GT. Pre-tessellate the unique GT(s) in the parent so all N
        # compute_reward calls hit _GT_PC_CACHE instead of cold-missing in
        # parallel (which yielded N+1 redundant tessellations per step).
        # Note: this runs in-parent (no subprocess isolation). GTs are curated
        # Text2CAD exports so OCC C-segfaults are unlikely; Python-level
        # failures are caught below and the GT falls through to compute_reward's
        # subprocess-isolated path. A C-segfault here would crash the rank.
        for _gt in set(ground_truth_step):
            _ck = _gt_cache_key(_gt, rcfg)
            if _ck in _GT_PC_CACHE:
                continue
            try:
                _pc, _tris = step_to_pointcloud(
                    _gt, n_points=rcfg.n_points, text2cad_src=text2cad_src,
                    return_triangle_count=True, deflection=rcfg.deflection,
                )
            except Exception:
                continue
            if _pc is not None:
                with _GT_PC_CACHE_LOCK:
                    if len(_GT_PC_CACHE) >= _GT_PC_CACHE_MAX:
                        _GT_PC_CACHE.pop(next(iter(_GT_PC_CACHE), None), None)
                    _GT_PC_CACHE[_ck] = (_pc, _tris)
        with ThreadPoolExecutor(max_workers=min(len(completions), _max_workers)) as pool:
            futures = [
                pool.submit(
                    compute_reward, gen, gt,
                    rcfg=rcfg,
                    text2cad_src=text2cad_src,
                    verbose=verbose,
                )
                for gen, gt in zip(completions, ground_truth_step)
            ]
            results = [f.result() for f in futures]

        rewards    = [r for r, _, _, _ in results]
        raw_scds   = [s for _, s, _, _ in results]
        stages     = [st for _, _, st, _ in results]
        n_tris     = [t for _, _, _, t in results]

        # Share parse status with the zero-weight parse_reward_fn so it
        # doesn't redo OCP work. Any stage past pred_parse means OCP accepted it.
        _parsed_stages = {"ok", "scd_nonfinite", "pred_degenerate",
                          "gt_parse", "gt_degenerate", "segfault_gt"}
        _parse_status_buf.clear()
        _parse_status_buf.extend(0.3 if st in _parsed_stages else 0.0 for st in stages)

        # Telemetry — queued for the callback so failure modes are diagnosable
        # from tensorboard during a live H100 run instead of only post-mortem.
        stage_counts = Counter(stages)
        finite_scds = [s for s in raw_scds if not math.isnan(s)]
        n_zero = sum(1 for r in rewards if r == 0.0 and not math.isnan(r))
        _telemetry_buf.append({
            "scd/frac_ok":           stage_counts.get("ok", 0) / len(results),
            "scd/frac_no_terminator": stage_counts.get("no_terminator", 0) / len(results),
            "scd/frac_pred_parse_fail": stage_counts.get("pred_parse", 0) / len(results),
            "scd/frac_pred_degenerate": stage_counts.get("pred_degenerate", 0) / len(results),
            "scd/frac_gt_fail":      (stage_counts.get("gt_parse", 0) + stage_counts.get("gt_degenerate", 0)) / len(results),
            "scd/frac_segfault":     stage_counts.get("segfault", 0) / len(results),
            "scd/frac_timeout":      stage_counts.get("timeout", 0) / len(results),
            "scd/frac_zero_reward":  n_zero / len(results),
            "scd/raw_p50":           float(np.median(finite_scds)) if finite_scds else float("nan"),
            "scd/raw_min":           float(min(finite_scds)) if finite_scds else float("nan"),
            "scd/raw_p90":           float(np.percentile(finite_scds, 90)) if finite_scds else float("nan"),
            "scd/n_triangles_mean":  float(np.mean([t for t in n_tris if t > 0])) if any(t > 0 for t in n_tris) else 0.0,
            # OB-2: NaN-masked stages — distinguishes "model getting worse"
            # (frac_ok↓) from "reward pipeline breaking" (frac_infra_fail↑).
            "scd/frac_infra_fail":   sum(
                1 for s in stages
                if s in ("segfault_gt", "scd_nonfinite", "spawn_fail", "exception")
            ) / len(results),
            "scd/completion_len_mean": float(np.mean([len(c) for c in completions])),
        })

        # GT-side failures (NaN reward): replace with batch mean so they
        # contribute zero advantage instead of a spurious negative signal.
        # NB: this is the BATCH mean (TRL passes the flat batch of all
        # prompts × generations), not the per-prompt group mean — TRL's
        # internal advantage normalization handles per-group centering.
        valid = [r for r in rewards if not math.isnan(r)]
        n_masked = len(rewards) - len(valid)
        if not valid:
            logger.warning(f"[scd_geometry_reward] All {len(rewards)} rewards masked (GT failures) — dead batch")
            return [0.0] * len(rewards)
        fill = sum(valid) / len(valid)
        out = [fill if math.isnan(r) else r for r in rewards]
        if n_masked:
            logger.info(f"[scd_geometry_reward] Masked {n_masked}/{len(rewards)} GT-side failures → batch mean {fill:.4f}")
        if max(out) == 0.0:
            logger.warning(f"[scd_geometry_reward] All-zero reward batch ({len(out)} completions) — zero gradient")
        elif n_zero > 0:
            logger.info(f"[scd_geometry_reward] {n_zero}/{len(out)} zero rewards — partial gradient")
        return out

    return scd_geometry_reward


# ── Parse reward (OCP parse success) ──────────────────────────────────────────

# Per-batch parse status, populated by scd_geometry_reward and read by the
# zero-weight parse_reward_fn so it doesn't have to re-spawn subprocesses.
_parse_status_buf: list[float] = []

def make_parse_reward_fn(text2cad_src: str):
    """Return a GRPO-compatible parse reward function.

    When registered alongside scd_geometry_reward (the normal config), this
    reads parse status from _parse_status_buf instead of spawning its own
    subprocess per completion. compute_reward already determines parse success
    internally — re-running OCP just to log a 0/1 doubles subprocess overhead.
    """

    def parse_reward_fn(completions: list[str], **kwargs) -> list[float]:
        if len(_parse_status_buf) == len(completions):
            out = list(_parse_status_buf)
            _parse_status_buf.clear()
            return out
        # Fallback (parse-only mode without scd reward): cheap terminator check.
        return [0.3 if "END-ISO-10303-21;" in c else 0.0 for c in completions]

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

    # First pass: filter by token length so we only retrieve for kept records.
    char_limit = max_completion_length * 4
    kept_records = []
    skipped = 0
    for record in records:
        gt_step = record.get("output") or record.get("step") or ""
        if len(gt_step) > char_limit:
            skipped += 1
            continue
        step_ids = tokenizer(gt_step, add_special_tokens=False)["input_ids"]
        if len(step_ids) > max_completion_length:
            skipped += 1
            continue
        kept_records.append(record)
    logger.info(f"  Filtered to {len(kept_records)} records (skipped {skipped} oversized)")

    # Batched retrieval — single SentenceTransformer.encode call instead of N×.
    captions     = [r["caption"] for r in kept_records]
    exclude_uids = [r.get("uid") or r.get("id_original") or "" for r in kept_records]
    logger.info(f"  Batch-retrieving {len(captions)} captions ...")
    retrieved_recs = retriever.retrieve_batch(captions, exclude_uids=exclude_uids)

    data = []
    n_empty_retrieval = 0
    for record, retrieved in zip(kept_records, retrieved_recs):
        gt_step = record.get("output") or record.get("step") or ""
        retrieved_step = retrieved.get("output") or retrieved.get("step") or ""
        if not retrieved_step:
            n_empty_retrieval += 1
        prompt = format_prompt(record["caption"], retrieved_step, tokenizer)
        data.append({
            "prompt": prompt,
            "ground_truth_step": gt_step,
        })
    if n_empty_retrieval:
        logger.warning(f"  {n_empty_retrieval}/{len(data)} records got an empty retrieved STEP "
                       f"— RAG context will be blank for those prompts")

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
    MAX_RETRIEVED_TOKENS = int(
        cfg.rl.get("max_retrieved_tokens", None)
        or getattr(cfg.model, "max_retrieved_tokens", None)
        or 4500
    )

    # ── Distributed context (set by torchrun on Gautschi) ───────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    use_quantization = bool(getattr(cfg.model, "use_quantization", True))


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

    # ── WandB (rank 0 only — other ranks set to disabled) ───────────────────
    import wandb
    if local_rank == 0:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "stepforge"),
            name=os.environ.get("WANDB_RUN_NAME", "rl"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    else:
        os.environ["WANDB_DISABLED"] = "true"
    # ── End WandB ────────────────────────────────────────────────────────────

    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise EnvironmentError("HUGGINGFACE_TOKEN environment variable not set.")

    sft_checkpoint = args.sft_checkpoint or os.path.join(
        cfg.paths.sft_checkpoint_dir, "final"
    )
    if not os.path.exists(sft_checkpoint):
        raise FileNotFoundError(
            f"SFT checkpoint not found at {sft_checkpoint}. "
            "Run training/llama3_SFT_response.py first."  # E2E-4
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

    # C12: TRL GRPO with PEFT and no explicit ref_model uses disable_adapter()
    # to expose the BASE model as the KL reference. If we pass a single PEFT
    # model with the SFT adapter, the reference becomes raw Llama — the KL term
    # then pulls the policy BACK toward pre-SFT, inverting the regularizer.
    # Fix: merge the SFT adapter into the base weights first, so disable_adapter()
    # exposes SFT (the correct anchor), then attach a fresh trainable LoRA.
    # Merging quantized weights is unsupported, so load fp16 → merge → save.
    merged_dir = os.path.join(cfg.paths.rl_checkpoint_dir, "_sft_merged")
    _merge_stamp = os.path.join(merged_dir, ".sft_source")
    _sft_mtime = max(
        (os.path.getmtime(os.path.join(sft_checkpoint, f))
         for f in os.listdir(sft_checkpoint)
         if os.path.isfile(os.path.join(sft_checkpoint, f))),
        default=0.0,
    )

    if local_rank == 0:
        # DS-3: exists()→open() race + partial-write tolerance.
        try:
            _stamp_mtime = (float(open(_merge_stamp).read().strip())
                            if os.path.exists(_merge_stamp) else -1.0)
        except (FileNotFoundError, ValueError):
            _stamp_mtime = -1.0
        if os.path.isdir(merged_dir) and _stamp_mtime >= _sft_mtime:
            logger.info(f"Reusing existing merged SFT at {merged_dir} (source unchanged)")
        else:
            logger.info("Loading SFT adapter into fp16 base for merge (KL reference fix)...")
            fp16_base = AutoModelForCausalLM.from_pretrained(
                cfg.model.base_model,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                token=hf_token,
            )
            sft_peft = PeftModel.from_pretrained(fp16_base, sft_checkpoint)
            merged   = sft_peft.merge_and_unload()
            merged.save_pretrained(merged_dir, safe_serialization=True)
            with open(_merge_stamp, "w") as f:
                f.write(str(_sft_mtime))
            del merged, sft_peft, fp16_base
            torch.cuda.empty_cache()
    elif is_distributed:
        # Upstream Bug 2: file-barrier sync — non-zero ranks wait for the merge.
        # DS-3: tolerate exists()→open() race and partial writes.
        def _read_stamp() -> float:
            try:
                return float(open(_merge_stamp).read().strip() or "-1")
            except (FileNotFoundError, ValueError):
                return -1.0
        # FR-1: bound the wait so a rank-0 failure surfaces clearly instead of
        # being buried under indefinite "waiting..." lines.
        _waited = 0
        while not os.path.exists(_merge_stamp) or _read_stamp() < _sft_mtime:
            time.sleep(5)
            _waited += 5
            if _waited >= 1800:
                raise RuntimeError(
                    f"Rank {local_rank}: waited 30min for rank 0's merge stamp at "
                    f"{_merge_stamp} — rank 0 likely failed; check its traceback above."
                )

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
            merged_dir,
            torch_dtype=torch.bfloat16,
            device_map={"": local_rank},
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
            merged_dir,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="sdpa",   # eager materializes O(seq²) attn matrix → OOM
        )

    rl_lora = LoraConfig(
        r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, rl_lora)
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
    train_json = os.path.join(cfg.paths.processed_dir, "train.json")
    dataset_cache_path = os.path.join(cfg.paths.rl_checkpoint_dir, "rl_dataset_cache")
    dataset_done_flag  = dataset_cache_path + ".done"

    if local_rank == 0:
        # CR-3: skip rebuild when the cache is already fresh for this SFT
        # checkpoint — same mtime-stamp pattern as merged_dir reuse at :412.
        try:
            _existing_flag = float(open(dataset_done_flag).read().strip())
        except (FileNotFoundError, ValueError):
            _existing_flag = -1.0
        if _existing_flag >= _sft_mtime and os.path.isdir(dataset_cache_path):
            logger.info("Rank 0: reusing cached RL dataset (flag matches SFT mtime)")
            from datasets import load_from_disk
            rl_dataset = load_from_disk(dataset_cache_path)
        else:
            # DS-1: clear stale flag+cache so a 2nd run's non-rank-0 can't load
            # the previous run's dataset while rank 0 is rebuilding.
            if os.path.exists(dataset_done_flag):
                os.remove(dataset_done_flag)
            import shutil as _shutil
            _shutil.rmtree(dataset_cache_path, ignore_errors=True)
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
            # DS-1b: write _sft_mtime into the flag (mirroring the merge-stamp at :424)
            # so a stale flag from a previous run fails the content check below.
            with open(dataset_done_flag, "w") as _f:
                _f.write(str(_sft_mtime))
            logger.info(f"Rank 0: dataset saved to {dataset_cache_path}")
    else:
        # Wait for rank 0's file-barrier signal (shared filesystem is visible to all nodes)
        logger.info(f"Rank {local_rank}: waiting for rank 0 to build dataset...")
        def _read_ds_flag() -> float:
            try:
                return float(open(dataset_done_flag).read().strip() or "-1")
            except (FileNotFoundError, ValueError):
                return -1.0
        # FR-1: bound the wait so a rank-0 dataset-build failure surfaces clearly.
        _waited = 0
        while _read_ds_flag() < _sft_mtime:
            time.sleep(5)
            _waited += 5
            if _waited >= 3600:
                raise RuntimeError(
                    f"Rank {local_rank}: waited 60min for rank 0's dataset flag at "
                    f"{dataset_done_flag} — rank 0 likely failed; check its traceback above."
                )
        from datasets import load_from_disk
        rl_dataset = load_from_disk(dataset_cache_path)
        logger.info(f"Rank {local_rank}: dataset loaded ({len(rl_dataset)} examples)")

    # ── Reward functions ─────────────────────────────────────────────────────
    parse_reward_fn = make_parse_reward_fn(text2cad_src=cfg.paths.text2cad_src)
    reward_verbose = bool(cfg.rl.get("verbose_reward", False))
    # A1: chamfer_bidirectional=True matches paper Eq.(1); False matches the
    # official eval code (chamferdist defaults to forward-only). Paper tables
    # were produced with the official code — use False to compare numbers,
    # True to compare method.
    # API-3: build the reward-shape config once from cfg.rl.reward.
    _mesh_defl = cfg.rl.reward.get("mesh_deflection", None)
    rcfg = RewardConfig(
        n_points=int(cfg.rl.reward.n_sample_points),
        delta_low=float(cfg.rl.reward.delta_low),
        delta_high=float(cfg.rl.reward.delta_high),
        bidirectional=bool(cfg.rl.reward.get("chamfer_bidirectional", True)),
        scale_prenorm=bool(cfg.rl.reward.get("scale_prenorm", True)),
        deflection=float(_mesh_defl) if _mesh_defl is not None else None,
    )
    logger.info(f"RewardConfig: {rcfg}")
    reward_fn = make_reward_fn(
        text2cad_src=cfg.paths.text2cad_src,
        rcfg=rcfg,
        verbose=reward_verbose,
    )

    # ── GRPO config (paper Section 4.4) ─────────────────────────────────────
    # Note: TRL >=0.9 renamed kl_coef → beta; pass both for compatibility.
    grpo_params = inspect.signature(GRPOConfig.__init__).parameters
    kl_kwarg = "beta" if "beta" in grpo_params else "kl_coef"
    optional_kwargs = {}
    # W17: Paper §4.4 specifies entropy_coef=0.005 but TRL GRPOConfig (DeepSeekMath
    # GRPO) is KL-only and has no such parameter. Log this divergence loudly.
    if "entropy_coef" in grpo_params:
        optional_kwargs["entropy_coef"] = cfg.rl.entropy_coef
    elif cfg.rl.get("entropy_coef") is not None:
        logger.warning(
            f"Paper §4.4 specifies entropy_coef={cfg.rl.entropy_coef} but the installed "
            f"TRL GRPOConfig has no such parameter (GRPO is KL-only per DeepSeekMath). "
            f"Training proceeds WITHOUT entropy regularization — this is a known "
            f"paper-fidelity gap."
        )

    grpo_config_kwargs = dict(
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
        logging_steps=1,
        save_steps=int(cfg.rl.get("save_steps", 10)),  # CR-4
        report_to="wandb",
        seed=42,
        # remove_unused_columns=False is CRITICAL: GRPOTrainer passes reward
        # function kwargs from dataset columns.  The default (True) would silently
        # strip ground_truth_step before it ever reaches the geometry reward fn,
        # making the entire reward signal zero while training appears to proceed.
        remove_unused_columns=False,
    )
    # Persist completions for visual reward-hacking inspection (model emitting
    # tiny spheres that align trivially). Only if this TRL version supports it.
    if "log_completions" in inspect.signature(GRPOConfig.__init__).parameters:
        grpo_config_kwargs["log_completions"] = True
    grpo_config = GRPOConfig(**grpo_config_kwargs)

    model.config.use_cache = False
    model.base_model.model.config.use_cache = False

    # W2: Paper §3.3 specifies a single reward (SCD piecewise linear, Eq. 3).
    # The format/parse bonuses are reproduction shaping not in the paper.
    # S10: When shaping IS used, pass weights explicitly so the effective
    # ratio (format=13%, parse=20%, geom=67% at max) is auditable.
    use_shaping = bool(cfg.rl.get("use_reward_shaping", False))
    # C3: reward_fn (scd) populates _parse_status_buf; parse_reward_fn reads it.
    # TRL invokes reward_funcs in list order — scd must precede parse or parse
    # reads the previous batch's status.
    if use_shaping:
        logger.warning("rl.use_reward_shaping=true — adding format+parse bonuses (NOT in paper §3.3)")
        reward_funcs   = [format_reward_fn, reward_fn, parse_reward_fn]
        reward_weights = [1.0, 1.0, 1.0]
    else:
        logger.info("Paper-faithful reward: SCD only (rl.use_reward_shaping=false)")
        # Register format/parse with weight=0.0 so TRL still logs them as
        # separate tensorboard scalars without contributing to the gradient.
        # Otherwise the four failure modes that all collapse to scd_geometry=0.0
        # are indistinguishable on a live run.
        reward_funcs   = [format_reward_fn, reward_fn, parse_reward_fn]
        reward_weights = [0.0, 1.0, 0.0]

    trainer_kwargs = dict(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=rl_dataset,
    )
    if "reward_weights" in inspect.signature(GRPOTrainer.__init__).parameters:
        trainer_kwargs["reward_weights"] = reward_weights
    elif not use_shaping:
        # TRL too old for reward_weights — fall back to true single-func mode
        # so format/parse don't leak into the gradient.
        logger.warning("TRL version lacks reward_weights — disabling zero-weight diagnostics")
        trainer_kwargs["reward_funcs"] = [reward_fn]
    trainer = GRPOTrainer(**trainer_kwargs)

    metrics_jsonl_path = os.path.join(cfg.paths.rl_checkpoint_dir, "metrics.jsonl")
    rl_checkpoints = sorted(
        glob.glob(os.path.join(cfg.paths.rl_checkpoint_dir, "checkpoint-*")),
        key=lambda x: int(x.rsplit("-", 1)[-1]),
    )
    rl_resume_from = rl_checkpoints[-1] if rl_checkpoints else None
    trainer.add_callback(RewardTelemetryCallback(
        metrics_jsonl_path,
        kl_halt_threshold=cfg.rl.get("kl_halt_threshold", None),
        resuming=rl_resume_from is not None,
        is_main=(local_rank == 0),
    ))
    logger.info(f"Per-step telemetry → {metrics_jsonl_path}")

    # Reward pipeline health check — fail fast if OCC/OCP isn't importable
    # in spawned subprocesses, instead of discovering it as 100% pred_parse_fail
    # at step 80 after burning the H100 budget.
    # DS-4: rank 0 only — avoids 8× redundant OCC spawn at startup and an
    # asymmetric crash on a non-zero rank tearing down DDP mid-init.
    if local_rank == 0:
        logger.info("Reward pipeline health check ...")
        _hc_gt = rl_dataset[0]["ground_truth_step"]
        _hc_reward, _hc_scd, _hc_stage, _hc_tris = compute_reward(
            _hc_gt, _hc_gt,
            rcfg=rcfg,
            text2cad_src=cfg.paths.text2cad_src,
            verbose=True,
        )
        if _hc_stage != "ok":
            raise RuntimeError(
                f"Reward health check FAILED: GT-vs-GT returned stage={_hc_stage!r} "
                f"(reward={_hc_reward}, scd={_hc_scd}, tris={_hc_tris}). "
                f"Expected stage='ok' with reward≈1.0. Check OCC/OCP installation "
                f"in the spawned subprocess environment."
            )
        if _hc_reward < 0.99:
            logger.warning(
                f"Reward health check: GT-vs-GT reward={_hc_reward:.4f} (scd={_hc_scd:.6f}). "
                f"Expected ≈1.0. Alignment may be unstable on this geometry."
            )
        else:
            logger.info(f"Reward health check OK: GT-vs-GT reward={_hc_reward:.4f} scd={_hc_scd:.6f}")
    else:
        logger.info(f"Rank {local_rank}: skipping reward health check (rank 0 runs it)")

    # CR-2: HF Trainer has no built-in SIGTERM handler. Register one on every
    # rank so torchrun's SIGTERM (forwarded from the SLURM trap) requests a
    # save+stop at the next step boundary instead of immediate exit.
    import signal as _signal

    def _on_sigterm(_signum, _frame):
        # SIG-1: set the flags FIRST. logger.warning() can raise (loguru's
        # _protected_lock is non-reentrant) if SIGTERM lands while the main
        # thread is inside another logger.* call — that would otherwise abort
        # the handler before flags were set.
        global _sigterm_requested
        _sigterm_requested = True  # SIG-2: re-asserted by on_step_end
        trainer.control.should_save = True
        trainer.control.should_training_stop = True
        try:
            logger.warning("SIGTERM received — requesting save+stop at next step boundary")
        except Exception:
            pass

    _signal.signal(_signal.SIGTERM, _on_sigterm)

    logger.info("Starting GRPO RL training...")
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
