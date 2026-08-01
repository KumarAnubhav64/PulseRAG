"""Application settings — the single boundary where environment variables are read.

Every other layer receives a typed ``Settings`` object; nothing else touches
``os.environ``. Secrets are never logged anywhere in the codebase.
"""

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
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

    # --- Embeddings ---
    # "sentence_transformers" (torch, ~760MB) | "fastembed" (ONNX, ~220MB) |
    # "remote" (Mistral API, ~0MB — REQUIRED to fit Render free's 512MB).
    embedding_backend: str = "sentence_transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    # onnxruntime spawns one intra-op thread per *reported* CPU core by default;
    # small instances (Render free reports the host's cores while only 0.1 CPU is
    # allocated) can balloon RAM past 512MB just from thread workspace. Cap it.
    embedding_threads: int = 1

    # --- Remote embeddings (Mistral) — zero local model RAM ---
    # Reads the key from MISTRAL_API_KEY (or REMOTE_EMBEDDING_API_KEY).
    remote_embedding_api_key: str | None = Field(
        default=None,
        # Field name included so the value can be set either via env
        # (MISTRAL_API_KEY) or programmatically (remote_embedding_api_key=...).
        validation_alias=AliasChoices(
            "MISTRAL_API_KEY", "REMOTE_EMBEDDING_API_KEY", "remote_embedding_api_key"
        ),
    )
    remote_embedding_model: str = "mistral-embed"
    remote_embedding_base_url: str = "https://api.mistral.ai/v1/embeddings"

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

    # --- Startup ---
    # Load the embedding model at boot instead of on the first upload. Keeps
    # the ~200-300MB model-load spike out of the request path, so an upload
    # can't push a small instance (e.g. Render free's 512MB) past its memory
    # limit mid-request. Tests disable this to stay offline.
    preload_embeddings: bool = True

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
