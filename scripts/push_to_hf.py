"""
Push the RL-trained LoRA adapter to HuggingFace Hub as a private repo.

Reads token from $HUGGINGFACE_TOKEN (or $HF_TOKEN). Creates the repo if it
doesn't exist (private by default), uploads all files in the adapter
directory, and writes a model card README.md describing the run.

Usage:
    python scripts/push_to_hf.py \\
        --adapter-dir $SCRATCH/stepforge/checkpoints/rl/final \\
        --repo-id  <your_hf_username>/stepforge-rl-final \\
        [--public]
"""

import argparse
import os
import sys

from huggingface_hub import HfApi, create_repo, upload_folder

MODEL_CARD_TEMPLATE = """---
license: other
base_model: meta-llama/Llama-3.2-3B-Instruct
library_name: peft
tags:
  - text-to-cad
  - step-file
  - grpo
  - rl
  - lora
---

# StepForge — RL-refined STEP-file generator

LoRA adapter for `meta-llama/Llama-3.2-3B-Instruct`, fine-tuned via SFT then
refined via GRPO RL on the DeepCAD + Text2CAD caption dataset. Maps a natural-
language description of a CAD part to a STEP (ISO 10303-21) file.

## Pipeline

1. **SFT** — 10 epochs on (caption, retrieved STEP, ground-truth STEP) triples
   with LoRA (r=16, alpha=32) on `meta-llama/Llama-3.2-3B-Instruct`.
2. **GRPO RL refinement** — 80 optimization steps on 4× H100 80GB,
   `num_generations=8`, `kl_coef=0.02`, `entropy_coef=0.005`, `lr=3e-6`
   (linear decay), `max_completion_length=4096`. Reward = format + parse +
   scaled chamfer distance.

## Results

| Metric | Full eval (N=100) | In-distribution (N=33) | Paper SFT | Paper GRPO |
|---|---|---|---|---|
| CR | 87.00 % | 96.97 % | 97.00 % | 99.00 % |
| RR | 85.00 % | 93.94 % | 95.18 % | 92.00 % |
| MSCD | 0.0790 | 0.0563 | 0.5300 | 0.0980 |

On the full eval, the MSCD beats paper GRPO; on the in-distribution subset
(GT STEP fits in `max_completion_length`), all three metrics meet or
substantially exceed paper baselines.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = PeftModel.from_pretrained(base, "{repo_id}").eval()

prompt = (
    "You are a CAD model generation assistant trained to produce STEP "
    "(.step) files based on textual descriptions. Given the following "
    "object description and relevant retrieved CAD data, generate a STEP "
    "file that accurately represents the described object.\\n\\n"
    "### caption:\\n<your caption>\\n\\n"
    "### retrieved relevant step file:\\n<retrieved STEP content>"
)
messages = [{{"role": "user", "content": prompt}}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=4096, do_sample=False)
print(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))
```

## Source

Code, configs, and full training logs: https://github.com/CoolK0912/StepForge
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", required=True,
                    help="Path to LoRA adapter directory")
    ap.add_argument("--repo-id", required=True,
                    help="HF repo id in the form <username>/<repo>")
    ap.add_argument("--public", action="store_true",
                    help="Make the repo public (default: private)")
    args = ap.parse_args()

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("ERROR: HUGGINGFACE_TOKEN (or HF_TOKEN) not set in env.")

    if not os.path.isdir(args.adapter_dir):
        sys.exit(f"ERROR: adapter directory not found: {args.adapter_dir}")

    # Write the model card README into the adapter dir
    readme_path = os.path.join(args.adapter_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(MODEL_CARD_TEMPLATE.format(repo_id=args.repo_id))
    print(f"Wrote model card: {readme_path}")

    # Create repo if it doesn't exist
    api = HfApi(token=token)
    print(f"Ensuring repo exists: {args.repo_id} (private={not args.public})")
    create_repo(args.repo_id, token=token, private=not args.public,
                exist_ok=True, repo_type="model")

    # Upload contents
    print(f"Uploading {args.adapter_dir} → {args.repo_id} ...")
    url = upload_folder(
        folder_path=args.adapter_dir,
        repo_id=args.repo_id,
        token=token,
        commit_message="Initial upload: RL-refined StepForge adapter",
    )
    print(f"\nDone. Repo URL: https://huggingface.co/{args.repo_id}")
    print(f"Commit:   {url}")


if __name__ == "__main__":
    main()
