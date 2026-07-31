"""Pydantic contracts shared between the API, services and repositories."""

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    conversation_id: str
    chunk_count: int
    transcript_length: int
    demo: bool = Field(default=False, description="True when a mock transcription was used")


class AskRequest(BaseModel):
    conversation_id: str
    question: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    text: str
    index: int
    score: float


class AskResponse(BaseModel):
    conversation_id: str
    question: str
    answer: str
    sources: list[Source] = Field(default_factory=list)
    cached: bool = False
    demo: bool = Field(default=False, description="True when a mock LLM answer was used")


class TranscriptResponse(BaseModel):
    conversation_id: str
    transcript: str


class HealthResponse(BaseModel):
    status: str
    demo_mode: bool
    groq_key_configured: bool
    redis_connected: bool
    embedding_backend: str
    version: str
