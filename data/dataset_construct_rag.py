"""
Build the RAG training dataset from DFS-restructured STEP files + captions.

Adapted from the original dataset_construct_rag.py to work with:
  - Our CSV format: uid + abstract columns (Text2CAD v1.1)
  - Our flat STEP structure: data/dfs_step/<uid_flat>.step
    where uid_flat = uid.replace('/', '_')  e.g. "0000/00000007" → "0000_00000007"
  - Our predefined train/val/test splits from Text2CAD JSON

Output: data/processed/rag_dataset.json
  Fields per record:
    id_original       — uid from CSV
    caption           — abstract caption
    id_retrieve       — uid of the retrieved nearest-neighbour
    relavant_step_file — DATA section of the retrieved STEP (note: original typo preserved)
    output            — DATA section of the target STEP

Usage:
    python data/dataset_construct_rag.py --config configs/config.yaml
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer


# ── Helpers ───────────────────────────────────────────────────────────────────

def uid_to_step_path(uid: str, dfs_step_dir: str) -> str | None:
    """Convert a uid like '0000/00000007' to a STEP file path."""
    uid_flat = uid.replace("/", "_")
    path = os.path.join(dfs_step_dir, f"{uid_flat}.step")
    return path if os.path.exists(path) else None


def load_step_data_section(step_path: str) -> str:
    """Extract the DATA section from a STEP file (everything from DATA; onwards)."""
    try:
        with open(step_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return ""

    data_lines = []
    collecting = False
    for line in lines:
        if "DATA;" in line:
            collecting = True
        if collecting:
            data_lines.append(line.strip())
    return "\n".join(data_lines)


def build_faiss_index(embeddings: np.ndarray):
    """Build a FAISS L2 index over the embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def search_faiss(index, query_emb: np.ndarray, uids: list, exclude_uid: str, top_k: int = 1):
    """Search FAISS, excluding the query's own uid. Returns list of (uid, similarity)."""
    distances, indices = index.search(query_emb, top_k + 1)
    results = []
    for i, idx in enumerate(indices[0]):
        uid = uids[idx]
        if uid != exclude_uid:
            sim = 1 / (1 + distances[0][i])
            results.append((uid, sim))
        if len(results) == top_k:
            break
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build RAG training dataset")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    dfs_step_dir = str(Path(cfg.paths.step_output_dir).parent / "dfs_step")
    output_path  = os.path.join(cfg.paths.processed_dir, "rag_dataset.json")
    os.makedirs(cfg.paths.processed_dir, exist_ok=True)

    # ── Load CSV ──────────────────────────────────────────────────────────────
    logger.info(f"Loading captions from {cfg.paths.caption_csv}")
    df = pd.read_csv(cfg.paths.caption_csv, dtype={"uid": str})

    caption_col = cfg.data.caption_column  # "abstract"
    df = df[["uid", caption_col]].dropna()
    df = df.rename(columns={caption_col: "description"})
    df = df.reset_index(drop=True)
    logger.info(f"Loaded {len(df)} caption records")

    # ── Filter to UIDs that have a DFS-restructured STEP file ─────────────────
    logger.info(f"Filtering to UIDs with restructured STEP files in {dfs_step_dir} ...")
    df["step_path"] = df["uid"].apply(lambda uid: uid_to_step_path(uid, dfs_step_dir))
    df = df[df["step_path"].notna()].reset_index(drop=True)
    logger.info(f"Kept {len(df)} UIDs with STEP files")

    # ── Encode captions ───────────────────────────────────────────────────────
    logger.info("Encoding captions with all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["description"].tolist(), convert_to_tensor=False,
                               show_progress_bar=True)
    embeddings = np.array(embeddings)

    # ── Build FAISS index ─────────────────────────────────────────────────────
    logger.info("Building FAISS index ...")
    index = build_faiss_index(embeddings)
    uids  = df["uid"].tolist()

    # ── Build dataset ─────────────────────────────────────────────────────────
    logger.info("Building RAG dataset ...")
    dataset = []
    skipped = 0

    for i, row in df.iterrows():
        uid     = row["uid"]
        caption = row["description"]

        # Retrieve nearest neighbour (excluding self)
        query_emb = model.encode([caption], convert_to_tensor=False)
        results   = search_faiss(index, query_emb, uids, exclude_uid=uid, top_k=1)

        retrieved_uid  = results[0][0] if results else None
        retrieved_path = uid_to_step_path(retrieved_uid, dfs_step_dir) if retrieved_uid else None

        output_step   = load_step_data_section(row["step_path"])
        retrieved_step = load_step_data_section(retrieved_path) if retrieved_path else ""

        if not output_step:
            skipped += 1
            continue

        dataset.append({
            "id_original":       uid,
            "caption":           caption,
            "id_retrieve":       retrieved_uid or "",
            "relavant_step_file": retrieved_step,   # typo preserved for SFT script compatibility
            "output":            output_step,
        })

        if (i + 1) % 1000 == 0:
            logger.info(f"  {i+1}/{len(df)} processed (skipped={skipped})")

    logger.info(f"Built {len(dataset)} records (skipped {skipped} with empty STEP output)")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    from pathlib import Path
    main()
