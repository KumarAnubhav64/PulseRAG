"""Unit tests for the Pydantic contracts in ``app.models.schemas``."""

import pytest
from pydantic import ValidationError

from app.models.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    Source,
    TranscriptResponse,
    UploadResponse,
)


def test_upload_response_defaults_demo_false() -> None:
    r = UploadResponse(conversation_id="abc", chunk_count=3, transcript_length=100)
    assert r.demo is False


def test_ask_request_accepts_valid() -> None:
    r = AskRequest(conversation_id="abc", question="What about pricing?")
    assert r.question == "What about pricing?"


def test_ask_request_rejects_blank_question() -> None:
    with pytest.raises(ValidationError):
        AskRequest(conversation_id="abc", question="")


def test_ask_request_rejects_too_long_question() -> None:
    with pytest.raises(ValidationError):
        AskRequest(conversation_id="abc", question="x" * 2001)


def test_source_fields() -> None:
    s = Source(text="chunk", index=1, score=0.42)
    assert s.text == "chunk"
    assert s.index == 1
    assert s.score == 0.42


def test_ask_response_defaults() -> None:
    r = AskResponse(conversation_id="abc", question="q", answer="a")
    assert r.sources == []
    assert r.cached is False
    assert r.demo is False


def test_ask_response_roundtrip_json() -> None:
    r = AskResponse(
        conversation_id="abc",
        question="q",
        answer="a",
        sources=[Source(text="t", index=0, score=0.9)],
        cached=True,
        demo=True,
    )
    data = r.model_dump()
    assert data["cached"] is True
    assert data["demo"] is True
    assert data["sources"][0]["score"] == 0.9
    assert data["sources"][0]["text"] == "t"


def test_transcript_response() -> None:
    r = TranscriptResponse(conversation_id="abc", transcript="hello")
    assert r.transcript == "hello"


def test_health_response() -> None:
    r = HealthResponse(
        status="ok",
        demo_mode=True,
        groq_key_configured=False,
        redis_connected=False,
        embedding_backend="sentence_transformers",
        version="0.1.0",
    )
    assert r.status == "ok"
    assert r.embedding_backend == "sentence_transformers"
