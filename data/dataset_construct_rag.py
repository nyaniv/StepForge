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

def uid_to_step_path(uid: str, dfs_step_dir: str) -> str | None:
    """Convert a uid like '0000/00000007' to a STEP file path."""
    uid_flat = uid.replace("/", "_")
    path = os.path.join(dfs_step_dir, f"{uid_flat}.step")
    return path if os.path.exists(path) else None


def load_step_data_section(step_path: str) -> str:
    """Extract the DATA section from a STEP file (everything from DATA; onwards)."""
    try:
        # W12: errors="replace" leaves a visible � marker; "ignore" silently
        # drops bytes mid-coordinate, producing valid-looking but wrong numbers.
        with open(step_path, "r", encoding="utf-8", errors="replace") as f:
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
    text = "\n".join(data_lines)
    # Pipeline step 3 (round_step_numbers.py) was never wired between
    # step_restructurer.py output and this consumer. Inline it so training
    # labels carry 6-decimal floats instead of raw 15-digit OCC precision.
    from data.round_step_numbers import round_float_numbers
    return round_float_numbers(text)


_ENTITY_RE = re.compile(r"^#\d+\s*=", re.MULTILINE)


def count_entities(step_text: str) -> int:
    """Count STEP entity definitions in a DATA section string."""
    return len(_ENTITY_RE.findall(step_text))


def build_faiss_index(embeddings: np.ndarray):
    """Build a FAISS IndexFlatIP (cosine sim on normalized vectors, paper §3.2)."""
    dim = embeddings.shape[1]
    # W7: Was IndexFlatL2. On L2-normalized vectors L2 and IP give identical
    # rankings, but the implicit equivalence breaks if anyone disables
    # normalization. Match build_index.py and the paper's "cosine similarity".
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def search_faiss(index, query_emb: np.ndarray, uids: list, exclude_uid: str, top_k: int = 1):
    """Search FAISS, excluding the query's own uid. Returns list of (uid, similarity)."""
    # IndexFlatIP on normalized vectors returns cosine similarity directly (range [-1, 1]).
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

    # ── C4: Restrict the persisted FAISS index to TRAIN UIDs only ─────────────
    # The index built here is saved to cfg.paths.faiss_index_path and consumed by
    # rl_train.py and evaluate.py. Indexing all splits leaks val/test as RAG
    # templates (paper §3.2: index is over training captions). Retrieval for
    # building rag_dataset.json (below) uses ALL captions so val/test rows still
    # get a relevant template; only the persisted index is train-restricted.
    with open(cfg.paths.split_json) as f:
        split_data = json.load(f)
    def _norm(u: str) -> str:
        return u.replace("_", "/")
    train_uid_set = {_norm(u) for u in split_data.get("train", [])}
    df["is_train"] = df["uid"].apply(lambda u: _norm(u) in train_uid_set)
    n_train_idx = int(df["is_train"].sum())
    logger.info(f"Train-only index: {n_train_idx} of {len(df)} UIDs are in the train split")
    if n_train_idx == 0:
        raise RuntimeError(
            f"No UIDs from {cfg.paths.split_json} 'train' split match the loaded "
            f"STEP files. Check UID normalization or split file format."
        )

    # ── Encode captions ───────────────────────────────────────────────────────
    logger.info("Encoding captions with all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # W8: explicit float32 (FAISS requirement) + batch_size=256 to match build_index.py
    embeddings = model.encode(df["description"].tolist(), convert_to_tensor=False,
                               normalize_embeddings=True, show_progress_bar=True,
                               batch_size=256)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # ── Build FAISS index (train-only, both for retrieval AND persistence) ──
    # The previous all-splits in-memory index leaked test geometry into training
    # prompts: a train record's nearest neighbour could be a test record, baked
    # into relavant_step_file. Retrieve from train-only here too — val/test rows
    # will retrieve a train template, which is exactly what evaluate.py does.
    train_mask = df["is_train"].to_numpy()
    train_embs = embeddings[train_mask]
    train_uids = df.loc[train_mask, "uid"].tolist()
    train_paths = df.loc[train_mask, "step_path"].tolist()
    train_row_to_df_idx = np.flatnonzero(train_mask)

    logger.info(f"Building train-only FAISS index ({len(train_uids)} records) ...")
    index = build_faiss_index(train_embs)

    # ── Preload all STEP outputs (we need them for both metadata filtering and dataset) ──
    logger.info("Loading STEP DATA sections ...")
    max_entities = cfg.data.max_entities
    output_steps: dict[str, str] = {}
    for i, row in df.iterrows():
        s = load_step_data_section(row["step_path"])
        output_steps[row["uid"]] = s
        if (i + 1) % 5000 == 0:
            logger.info(f"  {i+1}/{len(df)} STEP files loaded")

    # ── Build persisted metadata aligned to the train-only index ──
    # Only include records that pass the same filters as the training set, so
    # rl_train.py / evaluate.py never retrieve an empty or >max_entities template.
    metadata: list[dict] = []
    keep_in_index = np.zeros(len(train_uids), dtype=bool)
    for j, uid in enumerate(train_uids):
        s = output_steps[uid]
        if s and count_entities(s) < max_entities:
            keep_in_index[j] = True
            metadata.append({
                "id_original": uid,
                "caption":     df.at[train_row_to_df_idx[j], "description"],
                "output":      s,
            })
    persist_embs = train_embs[keep_in_index]
    logger.info(f"Persisted index: {len(metadata)} records pass entity/non-empty filter "
                f"(dropped {len(train_uids) - len(metadata)} train records)")

    # ── Build dataset (all splits — data_split.py partitions afterwards) ──
    logger.info("Building RAG dataset ...")
    dataset  = []
    skipped  = 0
    for i, row in df.iterrows():
        uid     = row["uid"]
        caption = row["description"]

        output_step = output_steps[uid]
        if not output_step:
            skipped += 1
            continue
        if count_entities(output_step) >= max_entities:
            skipped += 1
            continue

        # Retrieve from train-only index. exclude_uid still needed: a train
        # row's nearest neighbour can be itself.
        query_emb = embeddings[i:i+1]
        results   = search_faiss(index, query_emb, train_uids, exclude_uid=uid, top_k=1)

        retrieved_uid  = results[0][0] if results else None
        retrieved_step = output_steps.get(retrieved_uid, "") if retrieved_uid else ""
        if retrieved_step and count_entities(retrieved_step) >= max_entities:
            retrieved_step = ""

        dataset.append({
            "id_original":       uid,
            "caption":           caption,
            "id_retrieve":       retrieved_uid or "",
            "relavant_step_file": retrieved_step,   # typo preserved for SFT script compatibility
            "output":            output_step,
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
    from pathlib import Path
    main()
