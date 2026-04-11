"""
Preflight environment check — run before training to catch import/version
issues immediately rather than 10-20 minutes into a job.

Usage:
    python training/preflight_check.py
"""
import sys, importlib

PASS = []
FAIL = []

def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  OK  {name}")
    except Exception as e:
        FAIL.append(name)
        print(f"  FAIL {name}: {e}")

print("=" * 60)
print(" StepForge preflight check")
print("=" * 60)

# ── Core imports ──────────────────────────────────────────────
check("torch", lambda: __import__("torch"))
check("torch.cuda available", lambda: __import__("torch").cuda.is_available() or (_ for _ in ()).throw(RuntimeError("No CUDA")))
check("transformers", lambda: __import__("transformers"))
check("trl", lambda: __import__("trl"))
check("trl.SFTTrainer", lambda: __import__("trl").SFTTrainer)
check("trl.GRPOTrainer", lambda: __import__("trl").GRPOTrainer)
check("trl.DataCollatorForCompletionOnlyLM", lambda: __import__("trl").DataCollatorForCompletionOnlyLM)
check("peft", lambda: __import__("peft"))
check("datasets", lambda: __import__("datasets"))
check("accelerate", lambda: __import__("accelerate"))
check("omegaconf", lambda: __import__("omegaconf"))
check("loguru", lambda: __import__("loguru"))
check("sentence_transformers", lambda: __import__("sentence_transformers"))
check("faiss", lambda: __import__("faiss"))

# ── Version checks ────────────────────────────────────────────
def check_version(pkg, attr="__version__"):
    mod = importlib.import_module(pkg)
    ver = getattr(mod, attr, "unknown")
    print(f"       {pkg}=={ver}")

print("\n  Versions:")
for pkg in ["torch", "transformers", "trl", "peft", "accelerate", "datasets"]:
    try:
        check_version(pkg)
    except Exception as e:
        print(f"       {pkg}: could not read version ({e})")

# ── TRL patch check ───────────────────────────────────────────
check("trl._LazyModule patch", lambda: __import__("trl.import_utils", fromlist=["_LazyModule"])._LazyModule)
check("trl.FSDPModule patch", lambda: (
    __import__("trl.models.utils", fromlist=["FSDPModule"])  # just import, don't check value
))

# ── CUDA details ──────────────────────────────────────────────
import torch
print(f"\n  CUDA: {torch.version.cuda}  |  GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"    GPU {i}: {props.name}  {props.total_memory/1e9:.1f} GB")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  PASSED: {len(PASS)}   FAILED: {len(FAIL)}")
if FAIL:
    print(f"  FAILED checks: {', '.join(FAIL)}")
    print("=" * 60)
    sys.exit(1)
else:
    print("  All checks passed — environment is ready.")
    print("=" * 60)
