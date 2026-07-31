"""In-memory job store for asynchronous audio processing.

A job holds the temp-file path of the uploaded audio plus its lifecycle
status. A worker (background task) flips ``pending -> processing ->
done/failed`` and deletes the temp file once it is no longer needed.

Like the other repositories this is process-local: on Render's free tier the
instance can be restarted at any time, which loses in-flight jobs. That is
acceptable for the current single-upload UX; a durable queue would be a later
phase.
"""

import threading
import uuid
from dataclasses import dataclass

from ..models.schemas import JobStatus, UploadResponse


@dataclass
class JobRecord:
    job_id: str
    filename: str
    path: str
    status: JobStatus = JobStatus.pending
    upload: UploadResponse | None = None
    error: str | None = None


class JobRepository:
    def __init__(self) -> None:
        self._store: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def create(self, filename: str, path: str) -> JobRecord:
        record = JobRecord(job_id=uuid.uuid4().hex, filename=filename, path=path)
        with self._lock:
            self._store[record.job_id] = record
        return record

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._store.get(job_id)

    def mark_processing(self, job_id: str) -> None:
        with self._lock:
            record = self._store.get(job_id)
            if record is not None and record.status == JobStatus.pending:
                record.status = JobStatus.processing

    def complete(self, job_id: str, upload: UploadResponse) -> None:
        with self._lock:
            record = self._store.get(job_id)
            if record is not None:
                record.status = JobStatus.done
                record.upload = upload

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            record = self._store.get(job_id)
            if record is not None:
                record.status = JobStatus.failed
                record.error = error

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
