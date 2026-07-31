"""Unit tests for the three repositories.

- ``TranscriptRepository`` — in-memory CRUD.
- ``VectorRepository`` — FAISS store/search/delete using offline stub embeddings
  (no ~90MB model download).
- ``CacheRepository`` — in-memory TTL fallback when Redis is unavailable.
"""

from app.repositories.cache_repository import CacheRepository
from app.repositories.transcript_repository import TranscriptRepository
from app.repositories.vector_repository import VectorRepository


# --- TranscriptRepository ------------------------------------------------


def test_transcript_save_and_get() -> None:
    repo = TranscriptRepository()
    repo.save("c1", "hello transcript")
    assert repo.get("c1") == "hello transcript"


def test_transcript_get_missing_returns_none() -> None:
    assert TranscriptRepository().get("nope") is None


def test_transcript_delete() -> None:
    repo = TranscriptRepository()
    repo.save("c1", "x")
    assert repo.delete("c1") is True
    assert repo.delete("c1") is False
    assert repo.get("c1") is None


def test_transcript_len() -> None:
    repo = TranscriptRepository()
    repo.save("a", "1")
    repo.save("b", "2")
    assert len(repo) == 2
    repo.delete("a")
    assert len(repo) == 1


# --- VectorRepository ----------------------------------------------------


def test_vector_store_chunks_returns_count(stub_embeddings) -> None:
    repo = VectorRepository(stub_embeddings)
    n = repo.store_chunks("c1", ["chunk one", "chunk two"])
    assert n == 2


def test_vector_search_returns_top_k(stub_embeddings) -> None:
    repo = VectorRepository(stub_embeddings)
    repo.store_chunks("c1", ["alpha alpha alpha", "beta beta beta", "gamma gamma gamma"])
    results = repo.search("c1", "alpha", k=2)
    assert len(results) == 2
    assert all(isinstance(text, str) and isinstance(score, float) for text, score in results)


def test_vector_append_chunks_to_existing_index(stub_embeddings) -> None:
    repo = VectorRepository(stub_embeddings)
    repo.store_chunks("c1", ["one"])
    repo.store_chunks("c1", ["two", "three"])
    assert len(repo.search("c1", "anything", k=5)) == 3


def test_vector_search_unknown_conversation(stub_embeddings) -> None:
    repo = VectorRepository(stub_embeddings)
    assert repo.search("missing", "q", k=3) == []


def test_vector_delete(stub_embeddings) -> None:
    repo = VectorRepository(stub_embeddings)
    repo.store_chunks("c1", ["one"])
    assert repo.delete("c1") is True
    assert repo.delete("c1") is False
    assert repo.search("c1", "q", k=1) == []


# --- CacheRepository -----------------------------------------------------


def test_cache_memory_fallback(no_redis) -> None:
    cache = CacheRepository("redis://localhost:1/0", ttl_seconds=60)
    assert cache.connected is False
    assert cache.get("missing") is None
    cache.set("k", "v")
    assert cache.get("k") == "v"


def test_cache_ttl_expiry(no_redis, monkeypatch) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr("app.repositories.cache_repository.time.monotonic", lambda: clock["now"])
    cache = CacheRepository("redis://localhost:1/0", ttl_seconds=10)
    cache.set("k", "v")
    assert cache.get("k") == "v"
    clock["now"] = 10.5
    assert cache.get("k") is None


def test_cache_delete(no_redis) -> None:
    cache = CacheRepository("redis://localhost:1/0", ttl_seconds=60)
    cache.set("k", "v")
    cache.delete("k")
    assert cache.get("k") is None
