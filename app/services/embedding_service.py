"""Pluggable embedding backends, loaded lazily on first use.

- ``sentence_transformers`` (torch, ~760MB) — local dev default.
- ``fastembed`` (ONNX, ~220MB) — low-RAM local/offline option.
- ``remote`` (Mistral API, ~0MB) — for tiny-RAM hosts like Render free (512MB):
  no model is loaded into RAM at all; embeddings come from the Mistral API.

All backends expose the same minimal duck-typed interface
(``embed_documents`` / ``embed_query``), so callers never care which is active.
Everything is imported lazily inside the adapters so the app's import graph
stays tiny (a big langchain import alone used to eat ~700MB of RAM at boot).

The remote backend uses only the standard library (``urllib``), so it adds no
runtime dependencies.
"""

import json
import threading
import urllib.error
import urllib.request
from typing import Protocol

# Mistral accepts arrays of inputs; batch to keep each request small.
_REMOTE_BATCH = 32

# fastembed's registry uses full HF-style names (e.g.
# "sentence-transformers/all-MiniLM-L6-v2"), not the bare short name used by
# the sentence_transformers backend. Map short → registry name so the same
# EMBEDDING_MODEL setting works for both backends.
_FASTEMBED_MODEL_ALIASES = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}


class Embeddings(Protocol):
    """Minimal embedding interface (duck-typed; no langchain)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedAdapter:
    """ONNX embeddings via fastembed (low RAM, fully offline)."""

    def __init__(self, model_name: str, threads: int = 1) -> None:
        from fastembed import TextEmbedding

        model = _FASTEMBED_MODEL_ALIASES.get(model_name, model_name)
        self._model = TextEmbedding(model_name=model, threads=threads)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class SentenceTransformersAdapter:
    """torch embeddings via sentence-transformers (local dev / offline)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()


class RemoteEmbeddings:
    """Mistral embeddings API client — zero local model RAM."""

    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        if not api_key:
            raise ValueError(
                "EMBEDDING_BACKEND=remote requires a Mistral API key "
                "(set MISTRAL_API_KEY)."
            )
        self._api_key = api_key
        self._model = model
        self._base_url = base_url

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _REMOTE_BATCH):
            vectors.extend(self._embed_batch(texts[start : start + _REMOTE_BATCH]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self._model, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            self._base_url,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"Embeddings API error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Embeddings API unreachable: {exc.reason}") from exc

        items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
        if len(items) != len(texts):
            raise RuntimeError(
                f"Embeddings API returned {len(items)} vectors for "
                f"{len(texts)} inputs — unexpected response."
            )
        return [list(item["embedding"]) for item in items]


class EmbeddingService:
    """Lazily builds and caches the selected embedding backend."""

    def __init__(
        self,
        backend: str,
        model_name: str,
        threads: int = 1,
        api_key: str | None = None,
        remote_model: str = "mistral-embed",
        remote_base_url: str = "https://api.mistral.ai/v1/embeddings",
    ) -> None:
        self._backend = backend
        self._model_name = model_name
        self._threads = threads
        self._api_key = api_key
        self._remote_model = remote_model
        self._remote_base_url = remote_base_url
        self._embeddings: Embeddings | None = None
        # Guards lazy init so a boot-time warmup thread and a first request can't
        # both load the model (double load = double RAM on a small instance).
        self._load_lock = threading.Lock()

    @property
    def backend(self) -> str:
        return self._backend

    def get_embeddings(self) -> Embeddings:
        if self._embeddings is None:
            with self._load_lock:
                if self._embeddings is None:
                    if self._backend == "fastembed":
                        self._embeddings = FastEmbedAdapter(
                            self._model_name, threads=self._threads
                        )
                    elif self._backend == "remote":
                        self._embeddings = RemoteEmbeddings(
                            api_key=self._api_key or "",
                            model=self._remote_model,
                            base_url=self._remote_base_url,
                        )
                    else:
                        self._embeddings = SentenceTransformersAdapter(self._model_name)
        return self._embeddings
