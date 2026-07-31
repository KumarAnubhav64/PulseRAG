"""Unit tests for ``app.config`` — the env-safety boundary."""

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings


def test_defaults_without_env() -> None:
    # _env_file=None keeps the test hermetic: it must not pick up a real .env
    # the developer created locally (e.g. a pasted GROQ_API_KEY).
    s = Settings(_env_file=None)
    assert s.app_name == "PulseRAG"
    assert s.version == "0.1.0"
    assert s.groq_api_key is None
    assert s.groq_stt_model == "whisper-large-v3-turbo"
    assert s.groq_llm_model == "llama-3.3-70b-versatile"
    assert s.groq_request_timeout_seconds == 60.0
    assert s.redis_url == "redis://localhost:6379/0"
    assert s.embedding_backend == "sentence_transformers"
    assert s.embedding_model == "all-MiniLM-L6-v2"
    assert s.chunk_size == 500
    assert s.chunk_overlap == 50
    assert s.top_k == 4
    assert s.cache_ttl_seconds == 3600
    assert s.max_upload_mb == 25


def test_env_overrides_demo_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_MODE", "off")
    assert Settings().demo_mode == "off"


def test_demo_enabled_auto_without_key() -> None:
    assert Settings(demo_mode="auto", groq_api_key=None).demo_enabled is True


def test_demo_enabled_auto_with_key() -> None:
    assert Settings(demo_mode="auto", groq_api_key="sk-test").demo_enabled is False


def test_demo_enabled_forced_on() -> None:
    assert Settings(demo_mode="on", groq_api_key=None).demo_enabled is True
    assert Settings(demo_mode="on", groq_api_key="sk-test").demo_enabled is True


def test_demo_enabled_forced_off() -> None:
    assert Settings(demo_mode="off", groq_api_key=None).demo_enabled is False
    assert Settings(demo_mode="off", groq_api_key="sk-test").demo_enabled is False


def test_demo_mode_rejects_invalid_value() -> None:
    with pytest.raises(ValidationError):
        Settings(demo_mode="sometimes")  # type: ignore[arg-type]


def test_get_settings_is_lru_cached() -> None:
    assert get_settings() is get_settings()
