"""HTTP layer: thin handlers that validate input and delegate to services.

Dependency wiring for the service graph lives here (module-level, cached) so
the routes stay readable and the graph is built exactly once.
"""

from functools import lru_cache

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ..config import Settings, get_settings
from ..models.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    TranscriptResponse,
    UploadResponse,
)
from ..repositories.cache_repository import CacheRepository
from ..repositories.transcript_repository import TranscriptRepository
from ..repositories.vector_repository import VectorRepository
from ..services.embedding_service import EmbeddingService
from ..services.llm_service import LLMService
from ..services.rag_service import RAGService
from ..services.transcription_service import TranscriptionService

router = APIRouter()


# --- Dependency wiring -------------------------------------------------

@lru_cache
def _get_cache_repository() -> CacheRepository:
    settings = get_settings()
    return CacheRepository(settings.redis_url, settings.cache_ttl_seconds)


@lru_cache
def _get_rag_service() -> RAGService:
    settings = get_settings()
    embedding_service = EmbeddingService(
        settings.embedding_backend, settings.embedding_model
    )
    return RAGService(
        settings=settings,
        transcript_repo=TranscriptRepository(),
        vector_repo=VectorRepository(embedding_service.get_embeddings()),
        cache_repo=_get_cache_repository(),
        transcription=TranscriptionService(settings),
        llm=LLMService(settings),
    )


# --- Endpoints ---------------------------------------------------------

@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_audio(
    audio: UploadFile = File(...),
    rag: RAGService = Depends(_get_rag_service),
    settings: Settings = Depends(get_settings),
) -> UploadResponse:
    """Transcribe an audio file (<25MB), chunk + embed it, and index it."""
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        size_mb = len(data) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large: {size_mb:.1f} MB exceeds the "
                f"{settings.max_upload_mb} MB limit."
            ),
        )
    return rag.ingest(data, audio.filename or "audio.wav")


@router.post("/ask", response_model=AskResponse)
async def ask(
    question: AskRequest,
    rag: RAGService = Depends(_get_rag_service),
) -> AskResponse:
    """Ask a grounded question about an uploaded conversation."""
    if not question.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be blank.")
    return rag.ask(question)


@router.get("/transcript/{conversation_id}", response_model=TranscriptResponse)
async def get_transcript(
    conversation_id: str,
    rag: RAGService = Depends(_get_rag_service),
) -> TranscriptResponse:
    """Fetch the raw transcript for a conversation."""
    result = rag.get_transcript(conversation_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No transcript found for conversation '{conversation_id}'.",
        )
    return result


@router.get("/health", response_model=HealthResponse)
async def health(
    settings: Settings = Depends(get_settings),
    cache: CacheRepository = Depends(_get_cache_repository),
) -> HealthResponse:
    """Reveals *whether* things are configured — never the key itself."""
    return HealthResponse(
        status="ok",
        demo_mode=settings.demo_enabled,
        groq_key_configured=bool(settings.groq_api_key),
        redis_connected=cache.connected,
        embedding_backend=settings.embedding_backend,
        version=settings.version,
    )
