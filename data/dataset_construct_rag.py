"""
Build the RAG training dataset from already-filtered JSONL files.

Reads from train.jsonl / val.jsonl / test.jsonl (produced by filter_dataset.py),
which already contain DFS-reserialised STEP content in the 'step' field.

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
import pickle
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Determinism for SentenceTransformer GPU encode (matches retrieval/build_index.py).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import faiss
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_data_section(step_str: str) -> str:
    """Extract the DATA section from a STEP string (everything from DATA; onwards)."""
    lines = step_str.splitlines()
    data_lines = []
    collecting = False
    for line in lines:
        if "DATA;" in line:
            collecting = True
        if collecting:
            data_lines.append(line.strip())
    return "\n".join(data_lines)


_ENTITY_RE = re.compile(r"^#\d+\s*=", re.MULTILINE)


def count_entities(step_text: str) -> int:
    """Count STEP entity definitions in a DATA section string."""
    return len(_ENTITY_RE.findall(step_text))


def build_faiss_index(embeddings: np.ndarray):
    """Build a FAISS IndexFlatIP (cosine sim on normalized vectors, paper §3.2)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def search_faiss(index, query_emb: np.ndarray, uids: list, exclude_uid: str, top_k: int = 1):
    """Search FAISS, excluding the query's own uid. Returns list of (uid, similarity)."""
    scores, indices = index.search(query_emb, top_k + 1)
    results = []
    for i, idx in enumerate(indices[0]):
        uid = uids[idx]
        if uid != exclude_uid:
            results.append((uid, float(scores[0][i])))
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

    output_path = os.path.join(cfg.paths.processed_dir, "rag_dataset.json")
    os.makedirs(cfg.paths.processed_dir, exist_ok=True)
    max_entities = cfg.data.max_entities

    # ── Load from already-filtered JSONL files ────────────────────────────────
    # filter_dataset.py (step 4) already did DFS reserialization + entity filtering.
    # Records have fields: uid, caption, step (full STEP string), entity_count.
    logger.info("Loading records from train/val/test JSONL files ...")
    all_records = []
    split_membership = {}  # uid -> "train" | "val" | "test"
    for split_name in ["train", "val", "test"]:
        path = os.path.join(cfg.paths.processed_dir, f"{split_name}.jsonl")
        if not os.path.exists(path):
            logger.warning(f"  {path} not found, skipping")
            continue
        with open(path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    all_records.append(rec)
                    split_membership[rec["uid"]] = split_name

    if not all_records:
        raise RuntimeError(
            f"No records found in train/val/test JSONL files under {cfg.paths.processed_dir}. "
            "Ensure filter_dataset.py (step 4) ran successfully first."
        )
    logger.info(f"Loaded {len(all_records)} total records "
                f"({sum(1 for v in split_membership.values() if v == 'train')} train, "
                f"{sum(1 for v in split_membership.values() if v == 'val')} val, "
                f"{sum(1 for v in split_membership.values() if v == 'test')} test)")

    # ── Build dataframe ───────────────────────────────────────────────────────
    rows = []
    for rec in all_records:
        output_step = extract_data_section(rec["step"])
        rows.append({
            "uid":          rec["uid"],
            "description":  rec["caption"],
            "output_step":  output_step,
            "entity_count": rec.get("entity_count", count_entities(output_step)),
        })
    df = pd.DataFrame(rows).reset_index(drop=True)

    df["is_train"] = df["uid"].apply(lambda u: split_membership.get(u) == "train")
    n_train = int(df["is_train"].sum())
    logger.info(f"Train UIDs: {n_train} of {len(df)} total")
    if n_train == 0:
        raise RuntimeError("No train records found — check split_membership logic.")

    # ── Fast lookup dicts ─────────────────────────────────────────────────────
    uid_to_output_step   = dict(zip(df["uid"], df["output_step"]))
    uid_to_entity_count  = dict(zip(df["uid"], df["entity_count"]))

    # ── Encode captions ───────────────────────────────────────────────────────
    logger.info("Encoding captions with all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        df["description"].tolist(),
        convert_to_tensor=False,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=256,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # ── Build train-only FAISS index ──────────────────────────────────────────
    train_mask = df["is_train"].to_numpy()
    train_embs = embeddings[train_mask]
    train_uids = df.loc[train_mask, "uid"].tolist()
    train_df_indices = np.flatnonzero(train_mask)

    logger.info(f"Building train-only FAISS index ({len(train_uids)} records) ...")
    index = build_faiss_index(train_embs)

    # ── Build persisted metadata (filtered train records only) ────────────────
    metadata: list = []
    keep_in_index = np.zeros(len(train_uids), dtype=bool)
    for j, uid in enumerate(train_uids):
        s  = uid_to_output_step[uid]
        ec = uid_to_entity_count[uid]
        if s and ec < max_entities:
            keep_in_index[j] = True
            metadata.append({
                "id_original": uid,
                "caption":     df.at[train_df_indices[j], "description"],
                "output":      s,
            })
    persist_embs = train_embs[keep_in_index]
    logger.info(f"Persisted index: {len(metadata)} records pass entity filter "
                f"(dropped {len(train_uids) - len(metadata)} train records)")

    # ── Build dataset (all splits) ────────────────────────────────────────────
    logger.info("Building RAG dataset ...")
    dataset = []
    skipped = 0

    for i, row in df.iterrows():
        uid         = row["uid"]
        caption     = row["description"]
        output_step = row["output_step"]
        ec          = row["entity_count"]

        if not output_step or ec >= max_entities:
            skipped += 1
            continue

        query_emb = embeddings[i:i+1]
        results   = search_faiss(index, query_emb, train_uids, exclude_uid=uid, top_k=1)

        retrieved_uid  = results[0][0] if results else None
        retrieved_step = ""
        if retrieved_uid:
            rs = uid_to_output_step.get(retrieved_uid, "")
            rec = uid_to_entity_count.get(retrieved_uid, max_entities)
            retrieved_step = rs if (rs and rec < max_entities) else ""

        dataset.append({
            "id_original":        uid,
            "caption":            caption,
            "id_retrieve":        retrieved_uid or "",
            "relavant_step_file": retrieved_step,   # typo preserved for SFT script compatibility
            "output":             output_step,
        })

        if (i + 1) % 1000 == 0:
            logger.info(f"  {i+1}/{len(df)} processed (skipped={skipped})")

    logger.info(f"Built {len(dataset)} records (skipped {skipped} empty/oversized)")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    logger.info(f"Saved to {output_path}")

    # ── Persist filtered train-only index (consumed by rl_train.py / evaluate.py) ──
    persist_index = build_faiss_index(persist_embs)
    faiss_index_path    = cfg.paths.faiss_index_path
    faiss_metadata_path = cfg.paths.faiss_metadata_path
    os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
    faiss.write_index(persist_index, faiss_index_path)
    with open(faiss_metadata_path, "wb") as fh:
        pickle.dump(metadata, fh)
    logger.info(f"FAISS index saved to {faiss_index_path} ({len(metadata)} filtered train records)")


if __name__ == "__main__":
    main()
