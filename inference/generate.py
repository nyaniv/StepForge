"""
Generate a STEP file from a natural language caption.

RAG is active at inference time — the retriever queries the training FAISS
index live to find the most similar STEP as a structural template.

The prompt format is identical to SFT training.

Usage:
    python inference/generate.py \
        --caption "a hollow cylinder" \
        --output /tmp/cylinder.step \
        --checkpoint checkpoints/rl/final \
        --config configs/config.yaml

    python inference/generate.py \
        --caption "a 90-degree pipe elbow connector" \
        --output /tmp/elbow.step \
        --checkpoint checkpoints/sft/final
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from omegaconf import OmegaConf


# ── Prompt helpers (identical to training) ────────────────────────────────────

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


def extract_step(text: str) -> str:
    """Extract STEP content between ISO-10303-21; and END-ISO-10303-21;"""
    m = re.search(r"(ISO-10303-21;.*?END-ISO-10303-21;)", text, re.DOTALL)
    return m.group(1) if m else text


# ── Core generation function ───────────────────────────────────────────────────

def generate_step(
    caption: str,
    model,
    tokenizer,
    retriever,
    max_new_tokens: int = 2048,
    temperature: float = 0.3,
) -> str:
    """
    Generate a STEP file for the given caption using live RAG retrieval.

    Returns the extracted STEP content string.
    """
    retrieved = retriever.retrieve(caption)  # live retrieval at inference
    prompt = format_prompt(caption, retrieved["step"])

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return extract_step(text)


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate STEP file from caption")
    parser.add_argument("--caption",    required=True, help="Text description of the CAD model")
    parser.add_argument("--output",     required=True, help="Output .step file path")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # Load model
    logger.info(f"Loading model from {args.checkpoint}...")
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.checkpoint,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load retriever
    from retrieval.retriever import Retriever
    retriever = Retriever(
        index_path=cfg.paths.faiss_index_path,
        metadata_path=cfg.paths.faiss_metadata_path,
        model_name=cfg.retrieval.model,
    )

    # Generate
    logger.info(f"Generating STEP for: {args.caption}")
    step_content = generate_step(
        caption=args.caption,
        model=model,
        tokenizer=tokenizer,
        retriever=retriever,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(step_content)

    valid = "END-ISO-10303-21;" in step_content
    logger.info(f"Saved to {args.output} ({'complete' if valid else 'TRUNCATED — may not render'})")


if __name__ == "__main__":
    main()
