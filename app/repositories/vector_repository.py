"""FAISS-backed vector store, one index per conversation_id.

Uses raw ``faiss`` + ``numpy`` (no langchain): this keeps the import graph tiny,
which is what lets the app boot inside small RAM budgets (Render free = 512MB).
The repository layer owns *storage mechanics only*: it receives an embeddings
object (provided by the embedding service at wiring time) and never decides
what or when to embed.

Vectors are L2-normalized and stored in an inner-product index, so the returned
scores are cosine similarities in [-1, 1] (higher = more relevant).
"""

import threading
from typing import Any

import faiss
import numpy as np

from ..services.embedding_service import Embeddings


def _normalize_inplace(vectors: np.ndarray) -> None:
    """L2-normalize rows in place (zero vectors become unit, harmless)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors /= norms


class VectorRepository:
    def __init__(self, embeddings: Embeddings) -> None:
        self._embeddings = embeddings
        self._texts: dict[str, list[str]] = {}
        self._indexes: dict[str, faiss.Index] = {}
        self._lock = threading.Lock()

    def store_chunks(self, conversation_id: str, chunks: list[str]) -> int:
        """Embed and index chunks for a conversation.

        Appends to the conversation's index if one already exists; otherwise a
        fresh index is created. Returns the number of chunks stored.
        """
        if not chunks:
            return 0
        vectors = np.asarray(
            self._embeddings.embed_documents(chunks), dtype=np.float32
        )
        _normalize_inplace(vectors)
        with self._lock:
            texts = self._texts.setdefault(conversation_id, [])
            index = self._indexes.get(conversation_id)
            if index is None:
                index = faiss.IndexFlatIP(vectors.shape[1])
                self._indexes[conversation_id] = index
            index.add(vectors)
            texts.extend(chunks)
        return len(chunks)

    def search(
        self, conversation_id: str, query: str, k: int
    ) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(chunk_text, relevance_score)`` pairs, best first.

        Scores are cosine similarities in [-1, 1]. Returns an empty list when
        the conversation has no index yet.
        """
        with self._lock:
            texts = self._texts.get(conversation_id)
            index = self._indexes.get(conversation_id)
        if not texts or index is None or index.ntotal == 0:
            return []

        query_vector = np.asarray(
            self._embeddings.embed_query(query), dtype=np.float32
        ).reshape(1, -1)
        _normalize_inplace(query_vector)

        top_k = min(k, index.ntotal)
        if top_k < 1:
            return []
        scores, indices = index.search(query_vector, top_k)
        return [
            (texts[int(i)], float(score))
            for score, i in zip(scores[0], indices[0])
            if i >= 0
        ]

    def delete(self, conversation_id: str) -> bool:
        with self._lock:
            removed_texts = self._texts.pop(conversation_id, None)
            removed_index = self._indexes.pop(conversation_id, None)
        return removed_index is not None or removed_texts is not None
