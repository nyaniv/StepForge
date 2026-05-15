"""
Run the trained StepForge model on free-form natural-language captions
(without going through the held-out test set).

For each caption, this script:
  1. Embeds the caption with the same SentenceTransformer used at training.
  2. Retrieves the top-1 most similar training STEP via FAISS.
  3. Builds the chat-template prompt (caption + retrieved STEP) the same way
     evaluate.py does.
  4. Generates a STEP file with greedy decoding.
  5. Tessellates the predicted STEP into a mesh.
  6. Saves the generated STEP, retrieved STEP, and a mesh visualization.

Designed for one-off probe queries — e.g. the captions in Fig. 4 of
the STEP-LLM paper ("a 90-degree pipe elbow connector", etc.).

Usage:
    python scripts/generate_from_captions.py \\
        --checkpoint $SCRATCH/stepforge/checkpoints/rl/final \\
        --captions "a cube" "a round flat plate" "a hollow cylinder" \\
                   "a 90-degree pipe elbow connector" \\
                   "a bolt with a hexagonal socket head and a cylindrical shaft" \\
        --out-dir $SCRATCH/stepforge/probe_paper_captions
"""

import argparse
import json
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from reward.scd_reward import _safe_step_to_mesh


SYSTEM_PROMPT = (
    "You are a CAD model generation assistant trained to produce STEP (.step) files "
    "based on textual descriptions. Given the following object description and relevant "
    "retrieved CAD data, generate a STEP file that accurately represents the described object."
)


def load_model(checkpoint: str):
    scratch = os.environ.get("SCRATCH", "")
    hf_cache = os.path.join(
        scratch, ".hf-cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots")
    snapshots = os.listdir(hf_cache)
    base = os.path.join(hf_cache, snapshots[0])
    logger.info(f"Base model : {base}")
    logger.info(f"Checkpoint : {checkpoint}")
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, checkpoint)
    model.eval()
    return model, tok


def generate(model, tok, caption, retrieved, max_seq_length):
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"### caption:\n{caption}\n\n"
        f"### retrieved relevant step file:\n{retrieved}"
    )
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    prompt_len = ids.input_ids.shape[1]
    max_new_tokens = max(256, max_seq_length - prompt_len - 64)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )
    return tok.decode(out[0][prompt_len:], skip_special_tokens=True)


def render_mesh(ax, tris, label):
    if tris is None or len(tris) == 0:
        ax.text2D(0.5, 0.5, "parse failed", ha="center", va="center",
                  transform=ax.transAxes, color="#c0392b",
                  fontsize=12, fontweight="bold")
        ax.set_title(label, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        return
    color = "#a8c8f0" if label.startswith("pred") else "#cccccc"
    poly = Poly3DCollection(tris, alpha=0.92, linewidth=0.15,
                            edgecolor=(0.15, 0.15, 0.15, 0.35), facecolor=color)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    safe = np.where(norms < 1e-12, 1.0, norms)
    nz = (normals / safe)[:, 2]
    shade = 0.55 + 0.225 * (nz + 1.0)
    base = np.array(matplotlib.colors.to_rgb(color))
    face_colors = base[None, :] * shade[:, None]
    poly.set_facecolor(np.clip(face_colors, 0, 1))
    ax.add_collection3d(poly)
    ax.set_title(label, fontsize=10)
    ax.set_box_aspect((1, 1, 1))
    pts = tris.reshape(-1, 3)
    ranges = pts.max(0) - pts.min(0)
    r = max(ranges.max(), 1e-6)
    c = (pts.max(0) + pts.min(0)) / 2
    ax.set_xlim(c[0] - r/2, c[0] + r/2)
    ax.set_ylim(c[1] - r/2, c[1] + r/2)
    ax.set_zlim(c[2] - r/2, c[2] + r/2)
    ax.tick_params(axis="both", labelsize=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--captions", nargs="+", required=True,
                    help="One or more natural-language captions to generate from.")
    ap.add_argument("--config", default="configs/config_gautschi.yaml")
    ap.add_argument("--out-dir", required=True,
                    help="Directory to write generated STEP files + figure")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Retriever ────────────────────────────────────────────────────────────
    from retrieval.retriever import Retriever
    retriever = Retriever(
        index_path=cfg.paths.faiss_index_path,
        metadata_path=cfg.paths.faiss_metadata_path,
        model_name=cfg.retrieval.model,
        device="cpu",
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model, tok = load_model(args.checkpoint)
    max_seq = int(cfg.model.max_seq_length)

    # ── Generate ─────────────────────────────────────────────────────────────
    results = []
    for cap in args.captions:
        logger.info(f"\n=== caption: {cap!r} ===")
        retrieved = retriever.retrieve(cap)
        retrieved_step = retrieved.get("output", "")
        logger.info(f"  retrieved uid: {retrieved.get('uid', retrieved.get('id_original', '?'))}")
        logger.info(f"  retrieved caption: {retrieved.get('caption', '')[:120]}")

        gen = generate(model, tok, cap, retrieved_step, max_seq_length=max_seq)
        # The model often skips "DATA;" because the retrieved file already has it
        stripped = gen.lstrip()
        if not stripped.startswith("DATA;") and not stripped.startswith("ISO-10303-21;"):
            gen = "DATA;\n" + stripped

        has_end = "END-ISO-10303-21;" in gen
        logger.info(f"  generated len={len(gen)}  has_END={has_end}")

        # Tessellate predicted STEP
        pred_mesh, n_tris = _safe_step_to_mesh(
            gen, text2cad_src=cfg.paths.text2cad_src, deflection=None)
        renderable = pred_mesh is not None
        logger.info(f"  renderable={renderable}  n_triangles={n_tris}")

        results.append({
            "caption": cap,
            "retrieved_caption": retrieved.get("caption", ""),
            "retrieved_uid": retrieved.get("uid", retrieved.get("id_original", "")),
            "generated": gen,
            "has_END": has_end,
            "renderable": renderable,
            "n_triangles": int(n_tris),
            "pred_mesh": pred_mesh,
        })

    # ── Save JSON (without the mesh array, which doesn't serialize) ─────────
    json_path = os.path.join(args.out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "pred_mesh"} for r in results], f, indent=2)
    logger.info(f"\nSaved JSON: {json_path}")

    # ── Render a single figure with one row per caption ─────────────────────
    n = len(results)
    fig = plt.figure(figsize=(7, 4.3 * n + 0.6), constrained_layout=True)
    fig.suptitle(f"StepForge RL — free-form caption probe ({n} prompts)",
                 fontsize=13, fontweight="bold")
    subfigs = fig.subfigures(n, 1, hspace=0.05) if n > 1 else [fig.subfigures(1, 1)]
    for sf, r in zip(subfigs, results):
        cap_wrapped = "\n".join(textwrap.wrap(r["caption"], width=80))
        status = "renderable" if r["renderable"] else "parse failed"
        sf.suptitle(f"{cap_wrapped}\n[{status}  ·  n_triangles={r['n_triangles']}]",
                    fontsize=10, ha="center")
        ax = sf.subplots(1, 1, subplot_kw={"projection": "3d"})
        render_mesh(ax, r["pred_mesh"], "predicted")

    png_path = os.path.join(args.out_dir, "probe.png")
    plt.savefig(png_path, dpi=140, bbox_inches="tight")
    logger.info(f"Saved figure: {png_path}")


if __name__ == "__main__":
    main()
