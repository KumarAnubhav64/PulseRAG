"""Unit tests for the four services.

- ``TranscriptionService`` — demo-mode canned transcript; real path with a fake
  Groq client (no network).
- ``EmbeddingService`` — lazy, pluggable backend selection with fakes.
- ``LLMService`` — demo-mode mock answers; real path with a fake ChatGroq.
- ``RAGService`` — ingest → ask → cache pipeline with stub embeddings and the
  in-memory cache fallback (no model download, no network).
"""

from types import SimpleNamespace

import pytest

from app.config import Settings
from app.models.schemas import AskRequest
from app.repositories.cache_repository import CacheRepository
from app.repositories.transcript_repository import TranscriptRepository
from app.repositories.vector_repository import VectorRepository
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.transcription_service import TranscriptionService


# --- TranscriptionService ------------------------------------------------


def test_transcription_demo_returns_canned_transcript() -> None:
    svc = TranscriptionService(Settings(demo_mode="on"))
    text, demo = svc.transcribe(b"audio", "demo.wav")
    assert demo is True
    assert "PulseRAG" in text


def test_transcription_real_path_uses_groq(monkeypatch) -> None:
    class FakeResult:
        text = "Hello from groq"

    class FakeTranscriptions:
        def create(self, **kwargs):
            return FakeResult()

    class FakeAudio:
        transcriptions = FakeTranscriptions()

    class FakeGroq:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        @property
        def audio(self):
            return FakeAudio()

    monkeypatch.setattr("app.services.transcription_service.Groq", FakeGroq)
    settings = Settings(groq_api_key="sk-test", demo_mode="off")
    svc = TranscriptionService(settings)
    text, demo = svc.transcribe(b"audio", "demo.wav")
    assert demo is False
    assert text == "Hello from groq"


# --- EmbeddingService ----------------------------------------------------


def test_embedding_backend_property() -> None:
    assert EmbeddingService("fastembed", "model").backend == "fastembed"
    assert EmbeddingService("sentence_transformers", "model").backend == "sentence_transformers"


def test_embedding_loads_fastembed(monkeypatch) -> None:
    # Patch sys.modules with a stub so we never import the (torch-heavy)
    # langchain_community.embeddings package — the service imports lazily
    # inside get_embeddings(), so the stub is what it resolves.
    import sys

    fake = object()
    monkeypatch.setitem(
        sys.modules,
        "langchain_community.embeddings",
        SimpleNamespace(FastEmbedEmbeddings=lambda **kwargs: fake),
    )
    svc = EmbeddingService("fastembed", "all-MiniLM-L6-v2")
    assert svc.get_embeddings() is fake


def test_embedding_loads_sentence_transformers(monkeypatch) -> None:
    import sys

    fake = object()
    monkeypatch.setitem(
        sys.modules,
        "langchain_huggingface",
        SimpleNamespace(HuggingFaceEmbeddings=lambda **kwargs: fake),
    )
    svc = EmbeddingService("sentence_transformers", "all-MiniLM-L6-v2")
    assert svc.get_embeddings() is fake


def test_embedding_is_lazy_and_cached(monkeypatch) -> None:
    import sys

    fake = object()
    monkeypatch.setitem(
        sys.modules,
        "langchain_huggingface",
        SimpleNamespace(HuggingFaceEmbeddings=lambda **kwargs: fake),
    )
    svc = EmbeddingService("sentence_transformers", "all-MiniLM-L6-v2")
    assert svc._embeddings is None  # lazy: not loaded until first use
    assert svc.get_embeddings() is fake
    assert svc.get_embeddings() is fake  # cached: same instance, no rebuild


# --- LLMService ----------------------------------------------------------


def test_llm_demo_answer_with_context() -> None:
    svc = LLMService(Settings(demo_mode="on"))
    # "price" (not just "pricing") in the question hits the pricing branch.
    answer, demo = svc.generate("What did the customer say about the price?", ["chunk..."])
    assert demo is True
    assert "price" in answer.lower()
    assert "$49" in answer


def test_llm_demo_answer_without_context() -> None:
    svc = LLMService(Settings(demo_mode="on"))
    answer, demo = svc.generate("hi", [])
    assert demo is True
    assert "no transcript" in answer.lower()


def test_llm_demo_answer_general_question() -> None:
    svc = LLMService(Settings(demo_mode="on"))
    answer, _ = svc.generate("What about onboarding?", ["chunk..."])
    assert "relevant part" in answer


def test_llm_real_path_builds_prompt_and_invokes(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        content = "Grounded answer"

    class FakeChatGroq:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def invoke(self, prompt):
            captured["prompt"] = prompt
            return FakeResponse()

    monkeypatch.setattr("app.services.llm_service.ChatGroq", FakeChatGroq)
    svc = LLMService(Settings(groq_api_key="sk-test", demo_mode="off"))
    answer, demo = svc.generate("What about pricing?", ["chunk one", "chunk two"])
    assert demo is False
    assert answer == "Grounded answer"
    assert "[1] chunk one" in captured["prompt"]
    assert "[2] chunk two" in captured["prompt"]
    assert "QUESTION: What about pricing?" in captured["prompt"]
    assert captured["kwargs"]["temperature"] == 0


# --- RAGService ----------------------------------------------------------


def _make_rag(settings: Settings, embeddings) -> RAGService:
    # NOTE: callers must include the `no_redis` fixture in their signature so
    # CacheRepository uses the in-memory fallback instead of attempting a real
    # Redis connection (1s timeout) at construction.
    return RAGService(
        settings=settings,
        transcript_repo=TranscriptRepository(),
        vector_repo=VectorRepository(embeddings),
        cache_repo=CacheRepository("redis://localhost:1/0", ttl_seconds=3600),
        transcription=TranscriptionService(settings),
        llm=LLMService(settings),
    )


def test_rag_ingest_indexes_transcript(stub_embeddings, no_redis) -> None:
    settings = Settings(demo_mode="on")
    rag = _make_rag(settings, stub_embeddings)
    resp = rag.ingest(b"audio", "demo.wav")
    assert resp.conversation_id
    assert resp.chunk_count >= 1
    assert resp.transcript_length > 0
    assert resp.demo is True


def test_rag_ask_returns_sources_and_caches(stub_embeddings, no_redis) -> None:
    settings = Settings(demo_mode="on")
    rag = _make_rag(settings, stub_embeddings)
    conv = rag.ingest(b"audio", "demo.wav").conversation_id
    req = AskRequest(
        conversation_id=conv, question="What did the customer say about pricing?"
    )
    first = rag.ask(req)
    assert first.cached is False
    assert first.demo is True
    assert len(first.sources) > 0
    assert "price" in first.answer.lower()

    second = rag.ask(req)
    assert second.cached is True
    assert second.answer == first.answer


def test_rag_ask_unknown_conversation(stub_embeddings, no_redis) -> None:
    settings = Settings(demo_mode="on")
    rag = _make_rag(settings, stub_embeddings)
    out = rag.ask(AskRequest(conversation_id="missing", question="hi"))
    assert out.sources == []
    assert "upload" in out.answer.lower()


def test_rag_get_transcript(stub_embeddings, no_redis) -> None:
    settings = Settings(demo_mode="on")
    rag = _make_rag(settings, stub_embeddings)
    conv = rag.ingest(b"audio", "demo.wav").conversation_id
    tr = rag.get_transcript(conv)
    assert tr is not None
    assert tr.conversation_id == conv
    assert len(tr.transcript) > 0
    assert rag.get_transcript("missing") is None


def test_rag_cache_key_normalizes_question() -> None:
    assert RAGService._cache_key("c", "Hi There") == RAGService._cache_key("c", "hi there")
    assert RAGService._cache_key("c", "a") != RAGService._cache_key("d", "a")
