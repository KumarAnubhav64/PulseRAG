"""Orchestration: transcribe → chunk → embed → store; retrieve → answer → cache.

The only service that knows the full pipeline. It talks to repositories for
storage, to the transcription/LLM services for external AI calls, and to the
embedding service (via the vector repository) for local embeddings.
"""

import hashlib
import json
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import Settings
from ..models.schemas import AskRequest, AskResponse, Source, TranscriptResponse, UploadResponse
from ..repositories.cache_repository import CacheRepository
from ..repositories.transcript_repository import TranscriptRepository
from ..repositories.vector_repository import VectorRepository
from .llm_service import LLMService
from .transcription_service import TranscriptionService


class RAGService:
    def __init__(
        self,
        settings: Settings,
        transcript_repo: TranscriptRepository,
        vector_repo: VectorRepository,
        cache_repo: CacheRepository,
        transcription: TranscriptionService,
        llm: LLMService,
    ) -> None:
        self._settings = settings
        self._transcripts = transcript_repo
        self._vectors = vector_repo
        self._cache = cache_repo
        self._transcription = transcription
        self._llm = llm
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    # --- Ingest ---------------------------------------------------------

    def ingest(self, source: bytes | str, filename: str) -> UploadResponse:
        """Transcribe + index an audio source and return the result.

        ``source`` is either raw audio bytes (legacy callers) or a path to an
        on-disk audio file (streamed uploads). Streaming to disk keeps the
        whole file out of RAM during the request.
        """
        conversation_id = uuid.uuid4().hex
        transcript, demo = self._transcription.transcribe(source, filename)
        self._transcripts.save(conversation_id, transcript)
        chunks = self._splitter.split_text(transcript)
        chunk_count = self._vectors.store_chunks(conversation_id, chunks)
        return UploadResponse(
            conversation_id=conversation_id,
            chunk_count=chunk_count,
            transcript_length=len(transcript),
            demo=demo,
        )

    # --- Query ----------------------------------------------------------

    def ask(self, request: AskRequest) -> AskResponse:
        cache_key = self._cache_key(request.conversation_id, request.question)

        if (cached := self._cache.get(cache_key)) is not None:
            data = json.loads(cached)
            data["cached"] = True
            return AskResponse(**data)

        results = self._vectors.search(
            request.conversation_id, request.question, self._settings.top_k
        )
        sources = [
            Source(text=text, index=i, score=round(score, 4))
            for i, (text, score) in enumerate(results)
        ]
        answer, demo = self._llm.generate(request.question, [text for text, _ in results])

        response = AskResponse(
            conversation_id=request.conversation_id,
            question=request.question,
            answer=answer,
            sources=sources,
            demo=demo,
        )
        self._cache.set(cache_key, response.model_dump_json())
        return response

    def get_transcript(self, conversation_id: str) -> TranscriptResponse | None:
        transcript = self._transcripts.get(conversation_id)
        if transcript is None:
            return None
        return TranscriptResponse(conversation_id=conversation_id, transcript=transcript)

    @staticmethod
    def _cache_key(conversation_id: str, question: str) -> str:
        digest = hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()[:16]
        return f"rag:{conversation_id}:{digest}"
