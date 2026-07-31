"""Application settings — the single boundary where environment variables are read.

Every other layer receives a typed ``Settings`` object; nothing else touches
``os.environ``. Secrets are never logged anywhere in the codebase.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Core ---
    app_name: str = "PulseRAG"
    version: str = "0.1.0"

    # --- Groq (optional — demo mode kicks in when the key is absent) ---
    groq_api_key: str | None = None
    groq_stt_model: str = "whisper-large-v3-turbo"
    groq_llm_model: str = "llama-3.3-70b-versatile"
    groq_request_timeout_seconds: float = 60.0

    # --- Storage ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Embeddings (local, free, offline after first download) ---
    embedding_backend: str = "sentence_transformers"  # or "fastembed"
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- RAG tuning ---
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4

    # --- Cache ---
    cache_ttl_seconds: int = 3600

    # --- Demo mode: "auto" | "on" | "off" ---
    # auto = demo only when GROQ_API_KEY is missing; on/off force the behaviour.
    demo_mode: Literal["auto", "on", "off"] = "auto"

    # --- Uploads ---
    max_upload_mb: int = 25

    @property
    def demo_enabled(self) -> bool:
        """Whether Groq-backed calls (STT + LLM) should return mock data."""
        if self.demo_mode == "on":
            return True
        if self.demo_mode == "off":
            return False
        # "auto": demo only when no API key is configured
        return not self.groq_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
