"""
RAG retriever: find the most semantically similar STEP file in the training set.

Used at both training time (pre-computed via precompute_rag.py) and inference
time (live retrieval in generate.py).

The retriever embeds the query caption with the same SentenceTransformer model
used to build the index, then does a cosine similarity search over the FAISS index.
"""

import pickle
from typing import Optional

import faiss
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, index_path: str, metadata_path: str,
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Load the FAISS index and metadata from disk.

        Args:
            index_path: path to the .faiss index file
            metadata_path: path to the metadata .pkl file
            model_name: SentenceTransformer model (must match build_index.py)
        """
        logger.info(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as fh:
            self.records = pickle.load(fh)
        self.model = SentenceTransformer(model_name)
        logger.info(f"Retriever ready: {len(self.records)} training examples indexed")

    def retrieve(self, query_caption: str, exclude_uid: Optional[str] = None) -> dict:
        """
        Retrieve the top-1 most semantically similar training example.

        Args:
            query_caption: the natural language description to query with
            exclude_uid: UID to exclude (prevents self-retrieval during training)

        Returns:
            dict with keys "uid", "caption", "step"
        """
        emb = self.model.encode(
            [query_caption],
            normalize_embeddings=True,
        ).astype(np.float32)

        # Search top-20 to allow filtering out self
        _, indices = self.index.search(emb, 20)

        for idx in indices[0]:
            rec = self.records[int(idx)]
            rec_uid = rec.get("uid") or rec.get("id_original") or ""
            if exclude_uid is None or rec_uid != exclude_uid:
                return rec

        # Fallback: return top-1 even if it's self (shouldn't happen in practice)
        return self.records[int(indices[0][0])]

    def retrieve_batch(self, captions: list[str],
                       exclude_uids: Optional[list[str]] = None) -> list[dict]:
        """
        Retrieve top-1 for a batch of captions (more efficient for pre-computation).
        """
        embeddings = self.model.encode(
            captions,
            batch_size=256,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        _, indices = self.index.search(embeddings, 20)
        results = []
        for i, query_indices in enumerate(indices):
            exclude = exclude_uids[i] if exclude_uids and i < len(exclude_uids) else None
            for idx in query_indices:
                rec = self.records[int(idx)]
                rec_uid = rec.get("uid") or rec.get("id_original") or ""
                if exclude is None or rec_uid != exclude:
                    results.append(rec)
                    break
            else:
                results.append(self.records[int(query_indices[0])])
        return results
