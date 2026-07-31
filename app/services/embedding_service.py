"""Pluggable local embedding backend, loaded lazily on first use.

- ``sentence_transformers`` (torch, ~350–450MB) — default; your machine.
- ``fastembed`` (ONNX, ~150–250MB) — for Render's 512MB free tier.

Both run the same model (``all-MiniLM-L6-v2``), so switching backends does not
meaningfully change retrieval quality. Lazy loading keeps ``/health`` instant
and defers the ~90MB one-time model download until the first ingest/ask.
"""

import threading

from langchain_core.embeddings import Embeddings


# fastembed's registry uses full HF-style names (e.g.
# "sentence-transformers/all-MiniLM-L6-v2"), not the bare short name that
# HuggingFaceEmbeddings accepts. Map short → registry name so the same
# EMBEDDING_MODEL setting works for both backends.
_FASTEMBED_MODEL_ALIASES = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}


class EmbeddingService:
    def __init__(self, backend: str, model_name: str) -> None:
        self._backend = backend
        self._model_name = model_name
        self._embeddings: Embeddings | None = None
        # Guards lazy init so a boot-time warmup thread and a first request can't
        # both load the model (double load = double RAM on a small instance).
        self._load_lock = threading.Lock()

    @property
    def backend(self) -> str:
        return self._backend

    def get_embeddings(self) -> Embeddings:
        if self._embeddings is not None:
            return self._embeddings
        with self._load_lock:
            if self._embeddings is None:
                if self._backend == "fastembed":
                    from langchain_community.embeddings import FastEmbedEmbeddings

                    model = _FASTEMBED_MODEL_ALIASES.get(
                        self._model_name, self._model_name
                    )
                    self._embeddings = FastEmbedEmbeddings(model_name=model)
                else:
                    from langchain_huggingface import HuggingFaceEmbeddings

                    self._embeddings = HuggingFaceEmbeddings(model_name=self._model_name)
        return self._embeddings
