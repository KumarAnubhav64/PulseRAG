"""Shared test configuration.

- Forces demo mode for the *entire* session, set *before* any app module is
  imported, so no test ever makes a network call to Groq.
- Provides a deterministic offline ``stub_embeddings`` fixture so unit tests
  never trigger the ~90MB ``all-MiniLM-L6-v2`` download (only the end-to-end
  flow in ``test_basic.py`` exercises the real model).
- Provides a ``no_redis`` fixture that makes the cache repository fall back to
  its in-memory store instantly, with zero network attempts.
- Clears the lru_cached ``get_settings()`` after every test so env-var changes
  made by one test can never leak into another.
"""

import hashlib
import os

os.environ["DEMO_MODE"] = "on"
os.environ.pop("GROQ_API_KEY", None)
# Keep the app-lifespan embedding preload out of tests (offline, no model load).
os.environ["PRELOAD_EMBEDDINGS"] = "false"

import pytest


class StubEmbeddings:
    """Deterministic, offline embeddings (no model download, no network).

    Plain duck-typed class matching the app's minimal Embeddings interface
    (embed_documents / embed_query) — no framework base class needed.
    """

    def __init__(self, dim: int = 32) -> None:
        self._dim = dim

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [float(digest[i % len(digest)]) / 255.0 for i in range(self._dim)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def stub_embeddings() -> StubEmbeddings:
    return StubEmbeddings()


@pytest.fixture
def no_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make CacheRepository use its in-memory fallback without any network."""

    class _NoRedis:
        def __init__(self, *args, **kwargs) -> None:
            raise ConnectionError("redis unavailable")

        @classmethod
        def from_url(cls, *args, **kwargs):
            raise ConnectionError("redis unavailable")

    import redis

    monkeypatch.setattr(redis, "Redis", _NoRedis)


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    """Ensure the cached Settings object never leaks between tests."""
    from app.config import get_settings

    yield
    get_settings.cache_clear()
