"""In-memory transcript storage, keyed by conversation_id.

The single source of truth for transcripts in Phase 1. A later phase can swap
this for SQLite without touching any other layer.
"""

import threading


class TranscriptRepository:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._lock = threading.Lock()

    def save(self, conversation_id: str, transcript: str) -> None:
        with self._lock:
            self._store[conversation_id] = transcript

    def get(self, conversation_id: str) -> str | None:
        with self._lock:
            return self._store.get(conversation_id)

    def delete(self, conversation_id: str) -> bool:
        with self._lock:
            return self._store.pop(conversation_id, None) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
