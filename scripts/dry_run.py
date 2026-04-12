"""
Dry-run validation — catches as many errors as possible without a GPU.

Runs on a login node. Tests everything up to the point of model loading.

Usage:
    python scripts/dry_run.py --config configs/config_gautschi.yaml
    python scripts/dry_run.py --config configs/config_gautschi_refined.yaml
"""

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = []
FAIL = []


def check(name, fn):
    try:
        result = fn()
        PASS.append(name)
        msg = f"  OK    {name}"
        if result:
            msg += f"  ({result})"
        print(msg)
        return True
    except Exception as e:
        FAIL.append(name)
        print(f"  FAIL  {name}")
        print(f"        {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config_gautschi.yaml")
parser.add_argument("--n", type=int, default=16, help="Examples to format/test")
parser.add_argument("--tokenizer-path", default=None,
                    help="Local tokenizer path (skip HF download on login node)")
args = parser.parse_args()

print("=" * 60)
print(f"  StepForge Dry Run")
print(f"  Config : {args.config}")
print("=" * 60)

# ── 1. Imports ─────────────────────────────────────────────────────────────────
section("1. Imports")
check("omegaconf",           lambda: __import__("omegaconf"))
check("torch (CPU)",         lambda: __import__("torch").__version__)
check("transformers",        lambda: __import__("transformers").__version__)
check("datasets",            lambda: __import__("datasets").__version__)
check("peft",                lambda: __import__("peft").__version__)
check("trl",                 lambda: __import__("trl").__version__)
check("sentence_transformers",lambda: __import__("sentence_transformers").__version__)
check("faiss",               lambda: __import__("faiss").__version__)
check("loguru",              lambda: __import__("loguru"))
check("tqdm",                lambda: __import__("tqdm"))

# ── 2. Config ──────────────────────────────────────────────────────────────────
section("2. Config")
cfg = None

def load_config():
    from omegaconf import OmegaConf
    global cfg
    cfg = OmegaConf.load(args.config)
    return f"{len(cfg)} top-level keys"

check("config loads", load_config)

if cfg is not None:
    check("cfg.paths.processed_dir",    lambda: cfg.paths.processed_dir)
    check("cfg.paths.faiss_index_path", lambda: cfg.paths.faiss_index_path)
    check("cfg.model.base_model",       lambda: cfg.model.base_model)
    check("cfg.model.max_seq_length",   lambda: str(cfg.model.max_seq_length))
    check("cfg.sft.num_epochs",         lambda: str(cfg.sft.num_epochs))
    check("cfg.sft.learning_rate",      lambda: str(cfg.sft.learning_rate))
    check("cfg.sft.per_device_train_batch_size", lambda: str(cfg.sft.per_device_train_batch_size))
    check("cfg.sft.gradient_accumulation_steps", lambda: str(cfg.sft.gradient_accumulation_steps))
    check("cfg.rl.num_generations",     lambda: str(cfg.rl.num_generations))
    check("cfg.rl.max_steps",           lambda: str(cfg.rl.max_steps))

# ── 3. Data files ──────────────────────────────────────────────────────────────
section("3. Data files")

if cfg is not None:
    processed_dir = cfg.paths.processed_dir

    # Detect format
    train_jsonl = os.path.join(processed_dir, "train_with_rag.jsonl")
    train_json  = os.path.join(processed_dir, "train.json")
    test_jsonl  = os.path.join(processed_dir, "test.jsonl")
    test_json   = os.path.join(processed_dir, "test.json")

    use_jsonl = os.path.exists(train_jsonl)
    train_path = train_jsonl if use_jsonl else train_json
    test_path  = test_jsonl  if use_jsonl else test_json
    step_field = "step" if use_jsonl else "output"
    ret_field  = "retrieved_step" if use_jsonl else "relavant_step_file"

    check("train file exists",  lambda: f"{train_path} ({os.path.getsize(train_path)/1e9:.2f} GB)")
    check("test file exists",   lambda: f"{test_path} ({os.path.getsize(test_path)/1e6:.1f} MB)")

    def load_train():
        with open(train_path) as f:
            if use_jsonl:
                records = [json.loads(l) for l in f if l.strip()]
            else:
                records = json.load(f)
        return f"{len(records)} records"
    check("train file loads", load_train)

    def check_train_fields():
        with open(train_path) as f:
            r = json.loads(f.readline()) if use_jsonl else json.load(f)[0]
        keys = list(r.keys())
        missing = [k for k in ["caption", step_field, ret_field] if k not in r]
        if missing:
            raise KeyError(f"Missing fields: {missing}  (found: {keys})")
        return f"fields OK: {keys}"
    check("train fields (caption, step, retrieved)", check_train_fields)

    def load_test():
        with open(test_path) as f:
            if use_jsonl:
                records = [json.loads(l) for l in f if l.strip()]
            else:
                records = json.load(f)
        return f"{len(records)} records"
    check("test file loads", load_test)

    check("faiss index exists",    lambda: f"{cfg.paths.faiss_index_path} ({os.path.getsize(cfg.paths.faiss_index_path)/1e6:.1f} MB)")
    check("faiss metadata exists", lambda: f"{cfg.paths.faiss_metadata_path}")

# ── 4. FAISS / Retriever ───────────────────────────────────────────────────────
section("4. FAISS index + Retriever")

if cfg is not None:
    def load_faiss():
        import faiss, pickle
        idx = faiss.read_index(cfg.paths.faiss_index_path)
        with open(cfg.paths.faiss_metadata_path, "rb") as f:
            meta = pickle.load(f)
        return f"{idx.ntotal} vectors, {len(meta)} metadata records"
    check("faiss index loads", load_faiss)

    def test_retriever():
        from retrieval.retriever import Retriever
        r = Retriever(
            index_path=cfg.paths.faiss_index_path,
            metadata_path=cfg.paths.faiss_metadata_path,
            model_name=cfg.retrieval.model,
            device="cpu",
        )
        result = r.retrieve("a hollow cylinder", exclude_uid="0000/00000001")
        step = result.get("output") or result.get("step") or ""
        if not step:
            raise RuntimeError("Retriever returned empty step")
        return f"retrieved {len(step)} chars"
    check("retriever query works", test_retriever)

# ── 5. Tokenizer + label masking ──────────────────────────────────────────────
section("5. Tokenizer + label masking")

tokenizer = None
if cfg is not None:
    tok_path = args.tokenizer_path or cfg.model.base_model

    def load_tokenizer():
        global tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tok_path,
            local_files_only=args.tokenizer_path is not None,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return f"vocab_size={tokenizer.vocab_size}"
    check("tokenizer loads", load_tokenizer)

    if tokenizer is not None:
        def check_header_ids():
            HEADER_IDS = [128006, 78191, 128007, 271]
            HEADER_STR = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            actual = tokenizer.encode(HEADER_STR, add_special_tokens=False)
            if actual != HEADER_IDS:
                raise ValueError(f"Expected {HEADER_IDS}, got {actual}")
            return f"{HEADER_IDS}"
        check("assistant header IDs match", check_header_ids)

        def run_label_masking():
            from datasets import Dataset
            from transformers import DataCollatorForSeq2Seq

            max_seq = cfg.model.max_seq_length
            HEADER_IDS = [128006, 78191, 128007, 271]
            HEADER_LEN = len(HEADER_IDS)

            def find_response_start(ids):
                for i in range(len(ids) - HEADER_LEN + 1):
                    if ids[i:i+HEADER_LEN] == HEADER_IDS:
                        return i + HEADER_LEN
                return -1

            # Load N real examples
            with open(train_path) as f:
                if use_jsonl:
                    raw = [json.loads(l) for l, _ in zip(f, range(args.n)) if l.strip()]
                else:
                    raw = json.load(f)[:args.n]

            results = []
            for rec in raw:
                caption   = rec.get("caption", "")
                output    = rec.get(step_field, "")
                retrieved = rec.get(ret_field, "")
                messages = [
                    {"role": "user",      "content": f"Caption: {caption}\n\nRetrieved:\n{retrieved[:500]}"},
                    {"role": "assistant", "content": output},
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                enc  = tokenizer(text, truncation=True, max_length=max_seq, add_special_tokens=False)
                ids  = enc["input_ids"]
                rs   = find_response_start(ids)
                if rs == -1:
                    raise RuntimeError(f"Header not found in sequence of len {len(ids)}")
                labels   = [-100] * rs + ids[rs:]
                unmasked = sum(1 for l in labels if l != -100)
                if unmasked == 0:
                    raise RuntimeError(f"All labels masked — response_start={rs}, len={len(ids)}")
                results.append(unmasked / len(labels))

            avg_pct = sum(results) / len(results) * 100
            # Test collator
            _COLS = {"input_ids", "attention_mask", "labels"}
            ds = Dataset.from_list([
                {"input_ids": tokenizer(text, truncation=True, max_length=512,
                                        add_special_tokens=False)["input_ids"],
                 "attention_mask": [1]*min(512, len(tokenizer.encode(text))),
                 "labels": [1]*min(512, len(tokenizer.encode(text)))}
                for text in ["hello world"] * 2
            ])
            collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
            batch = [{k: v for k, v in ds[i].items() if k in _COLS} for i in range(2)]
            collator(batch)
            return f"{len(results)} examples, avg {avg_pct:.1f}% unmasked"

        check("label masking end-to-end", run_label_masking)

# ── 6. RL dataset builder (no model) ──────────────────────────────────────────
section("6. RL dataset builder (no model)")

if cfg is not None and tokenizer is not None:
    def test_rl_dataset_builder():
        from retrieval.retriever import Retriever
        from training.rl_train import build_rl_dataset

        retriever = Retriever(
            index_path=cfg.paths.faiss_index_path,
            metadata_path=cfg.paths.faiss_metadata_path,
            model_name=cfg.retrieval.model,
            device="cpu",
        )
        # Use a tiny slice of train data
        with open(train_path) as f:
            if use_jsonl:
                records = [json.loads(l) for l, _ in zip(f, range(20)) if l.strip()]
            else:
                records = json.load(f)[:20]

        import tempfile, json as _json
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            for r in records:
                tmp.write(_json.dumps(r) + "\n")
            tmp_path = tmp.name

        ds = build_rl_dataset(
            tmp_path, retriever, tokenizer,
            max_completion_length=cfg.rl.max_completion_length,
        )
        os.unlink(tmp_path)
        return f"{len(ds)} RL examples built from 20 records"

    check("RL dataset builder", test_rl_dataset_builder)

# ── 7. SLURM script syntax ────────────────────────────────────────────────────
section("7. SLURM script syntax (bash -n)")

slurm_scripts = [
    "slurm_sft_multigpu_gautschi.sh",
    "slurm_sft_multigpu_refined_gautschi.sh",
    "slurm_sft_4gpu_gautschi.sh",
    "slurm_sft_4gpu_refined_gautschi.sh",
    "slurm_rl_gautschi.sh",
]
for script in slurm_scripts:
    def make_check(s):
        def fn():
            import subprocess
            r = subprocess.run(["bash", "-n", s], capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(r.stderr.strip())
            return "syntax OK"
        return fn
    check(f"bash -n {script}", make_check(script))

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  PASSED: {len(PASS)}   FAILED: {len(FAIL)}")
if FAIL:
    print(f"\n  Failed checks:")
    for f in FAIL:
        print(f"    ✗ {f}")
    print("=" * 60)
    sys.exit(1)
else:
    print("  All checks passed — ready to run.")
    print("=" * 60)
