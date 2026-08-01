"""HTTP layer: thin handlers that validate input and delegate to services.

Dependency wiring for the service graph lives here (module-level, cached) so
the routes stay readable and the graph is built exactly once.

Uploads are handled asynchronously: the file is streamed to a temp file
(keeping it out of RAM), a job is queued, and a background task runs the
heavy transcription + embedding pipeline. The client polls ``GET /jobs/{id}``
for the result. This keeps the model-load spike out of the request path, so a
large upload can't push a small instance (Render free, 512MB) over its memory
limit mid-request.
"""

import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from ..config import Settings, get_settings
from ..models.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    JobResponse,
    TranscriptResponse,
)
from ..repositories.cache_repository import CacheRepository
from ..repositories.job_repository import JobRepository
from ..repositories.transcript_repository import TranscriptRepository
from ..repositories.vector_repository import VectorRepository
from ..services.embedding_service import EmbeddingService
from ..services.llm_service import LLMService
from ..services.rag_service import RAGService
from ..services.transcription_service import TranscriptionService

logger = logging.getLogger(__name__)

router = APIRouter()

# Stream the upload in 1MB chunks so only one chunk is ever in memory.
_STREAM_CHUNK_BYTES = 1024 * 1024


# --- Dependency wiring -------------------------------------------------

@lru_cache
def _get_cache_repository() -> CacheRepository:
    settings = get_settings()
    return CacheRepository(settings.redis_url, settings.cache_ttl_seconds)


@lru_cache
def _get_job_repository() -> JobRepository:
    return JobRepository()


@lru_cache
def _get_rag_service() -> RAGService:
    settings = get_settings()
    embedding_service = EmbeddingService(
        settings.embedding_backend,
        settings.embedding_model,
        threads=settings.embedding_threads,
        api_key=settings.remote_embedding_api_key,
        remote_model=settings.remote_embedding_model,
        remote_base_url=settings.remote_embedding_base_url,
    )
    return RAGService(
        settings=settings,
        transcript_repo=TranscriptRepository(),
        vector_repo=VectorRepository(embedding_service.get_embeddings()),
        cache_repo=_get_cache_repository(),
        transcription=TranscriptionService(settings),
        llm=LLMService(settings),
    )


# --- Upload helpers ------------------------------------------------------

def _unlink_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


async def _stream_upload_to_temp(audio: UploadFile, max_bytes: int) -> str:
    """Write the upload to a temp file, enforcing the size limit.

    Raises ``HTTPException`` (400/413) before any service is reached. Returns
    the temp file path; the caller owns its cleanup.
    """
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    size = 0
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        while True:
            chunk = await audio.read(_STREAM_CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"File too large: exceeds the "
                        f"{max_bytes // (1024 * 1024)} MB limit."
                    ),
                )
            tmp.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    except Exception:
        tmp.close()
        _unlink_quietly(tmp.name)
        raise
    tmp.close()
    return tmp.name


def _process_job(jobs: JobRepository, job_id: str) -> None:
    """Background worker: transcribe + index the queued audio file.

    The RAG service (and therefore the embedding model) is resolved *here*, in
    the worker thread, not in the upload endpoint — so validation-only requests
    (400/413/422) never touch the model, and the load spike stays out of the
    request path entirely.
    """
    record = jobs.get(job_id)
    if record is None:
        return
    try:
        jobs.mark_processing(job_id)
        upload = _get_rag_service().ingest(record.path, record.filename)
        jobs.complete(job_id, upload)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the client
        logger.exception("Job %s failed", job_id)
        jobs.fail(job_id, str(exc))
    finally:
        _unlink_quietly(record.path)


# --- Endpoints ---------------------------------------------------------

@router.post("/upload", response_model=JobResponse, status_code=202)
async def upload_audio(
    audio: UploadFile = File(...),
    background: BackgroundTasks = BackgroundTasks(),
    settings: Settings = Depends(get_settings),
    jobs: JobRepository = Depends(_get_job_repository),
) -> JobResponse:
    """Queue an audio file (<25MB) for transcription + indexing.

    Returns a job the client polls via ``GET /jobs/{job_id}``. No service (and
    therefore no embedding model) is touched until the background worker runs.
    """
    max_bytes = settings.max_upload_mb * 1024 * 1024
    try:
        tmp_path = await _stream_upload_to_temp(audio, max_bytes)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Could not read the uploaded file."
        ) from exc

    job = jobs.create(filename=audio.filename or "audio.wav", path=tmp_path)
    background.add_task(_process_job, jobs=jobs, job_id=job.job_id)
    return JobResponse(job_id=job.job_id, status=job.status)


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    jobs: JobRepository = Depends(_get_job_repository),
) -> JobResponse:
    """Poll the status/result of an upload job."""
    record = jobs.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No job found for '{job_id}'.")
    return JobResponse(
        job_id=record.job_id,
        status=record.status,
        upload=record.upload,
        error=record.error,
    )


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
