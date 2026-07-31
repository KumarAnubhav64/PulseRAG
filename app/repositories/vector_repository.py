"""FAISS-backed vector store, one index per conversation_id.

The repository layer owns *storage mechanics only*: it receives an embeddings
object (provided by the embedding service at wiring time) and never decides
what or when to embed.
"""

import threading

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


class VectorRepository:
    def __init__(self, embeddings: Embeddings) -> None:
        self._embeddings = embeddings
        self._indexes: dict[str, FAISS] = {}
        self._lock = threading.Lock()

    def store_chunks(self, conversation_id: str, chunks: list[str]) -> int:
        """Embed and index chunks for a conversation.

        Appends to the conversation's index if one already exists; otherwise a
        fresh index is created. Returns the number of chunks stored.
        """
        with self._lock:
            existing = self._indexes.get(conversation_id)
            if existing is None:
                self._indexes[conversation_id] = FAISS.from_texts(chunks, self._embeddings)
            else:
                existing.add_texts(chunks)
            return len(chunks)

    def search(self, conversation_id: str, query: str, k: int) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(chunk_text, relevance_score)`` pairs, best first.

        Returns an empty list when the conversation has no index yet.
        """
        with self._lock:
            index = self._indexes.get(conversation_id)
        if index is None:
            return []
        results = index.similarity_search_with_relevance_scores(query, k=k)
        return [(doc.page_content, float(score)) for doc, score in results]

    def delete(self, conversation_id: str) -> bool:
        with self._lock:
            return self._indexes.pop(conversation_id, None) is not None
