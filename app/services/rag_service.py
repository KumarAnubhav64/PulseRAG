"""Orchestration: transcribe → chunk → embed → store; retrieve → answer → cache.

The only service that knows the full pipeline. It talks to repositories for
storage, to the transcription/LLM services for external AI calls, and to the
embedding service (via the vector repository) for local embeddings.
"""

import hashlib
import json
import re
import uuid

from ..config import Settings
from ..models.schemas import AskRequest, AskResponse, Source, TranscriptResponse, UploadResponse
from ..repositories.cache_repository import CacheRepository
from ..repositories.transcript_repository import TranscriptRepository
from ..repositories.vector_repository import VectorRepository
from .llm_service import LLMService
from .transcription_service import TranscriptionService


# Plain-Python chunker (langchain's text-splitter pulled in ~700MB of imports
# at boot — the reason the app OOM'd on Render's 512MB. This mirrors its
# recursive behavior with zero dependencies.)
_SEPARATORS = ("\n\n", "\n", ". ", " ", "")


def _recursive_split(text: str, chunk_size: int, out: list[str]) -> None:
    if len(text) <= chunk_size:
        out.append(text)
        return
    for sep in _SEPARATORS:
        if not sep:
            break
        # Split while KEEPING the separator as its own token (regex capture
        # group). This guarantees every part is strictly shorter than the input
        # — no part ends with the separator it was split on — so the recursion
        # always terminates. A naive "split + re-attach" approach can loop
        # forever on dense text (e.g. a long transcript with no paragraph
        # breaks), which surfaced as a RecursionError on real uploads.
        parts = re.split(f"({re.escape(sep)})", text)
        if len(parts) > 1:
            for part in parts:
                if part:
                    _recursive_split(part, chunk_size, out)
            return
    # No separator found: hard-slice at chunk_size.
    out.append(text[:chunk_size])
    rest = text[chunk_size:]
    if rest:
        _recursive_split(rest, chunk_size, out)


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split ``text`` into overlapping chunks (langchain-recursive style).

    - Recursively break on paragraph → sentence → word boundaries until every
      piece fits ``chunk_size``.
    - Greedily merge pieces into chunks and carry the last ``chunk_overlap``
      characters into the next chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")
    if not text:
        return []

    pieces: list[str] = []
    _recursive_split(text, chunk_size, pieces)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > chunk_size:
            chunks.append(current)
            current = (current[-chunk_overlap:] if chunk_overlap else "") + piece
        else:
            current += piece
    if current:
        chunks.append(current)
    return chunks


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
        self._chunk_size = settings.chunk_size
        self._chunk_overlap = settings.chunk_overlap

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
        chunks = split_text(transcript, self._chunk_size, self._chunk_overlap)
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
