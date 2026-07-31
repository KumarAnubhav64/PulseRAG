"""Pluggable local embedding backend, loaded lazily on first use.

- ``sentence_transformers`` (torch, ~350–450MB) — default; your machine.
- ``fastembed`` (ONNX, ~150–250MB) — for Render's 512MB free tier.

Both run the same model (``all-MiniLM-L6-v2``), so switching backends does not
meaningfully change retrieval quality. Lazy loading keeps ``/health`` instant
and defers the ~90MB one-time model download until the first ingest/ask.
"""

from langchain_core.embeddings import Embeddings


class EmbeddingService:
    def __init__(self, backend: str, model_name: str) -> None:
        self._backend = backend
        self._model_name = model_name
        self._embeddings: Embeddings | None = None

    @property
    def backend(self) -> str:
        return self._backend

    def get_embeddings(self) -> Embeddings:
        if self._embeddings is None:
            if self._backend == "fastembed":
                from langchain_community.embeddings import FastEmbedEmbeddings

                self._embeddings = FastEmbedEmbeddings(model_name=self._model_name)
            else:
                from langchain_huggingface import HuggingFaceEmbeddings

                self._embeddings = HuggingFaceEmbeddings(model_name=self._model_name)
        return self._embeddings
