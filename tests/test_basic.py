"""End-to-end smoke tests: /health plus the full demo-mode flow.

Demo mode is forced on so the tests never touch the network for STT/LLM.
Embeddings are real (all-MiniLM-L6-v2) and are downloaded from HuggingFace on
the very first run (~90MB); afterwards they are cached and offline.
"""

import os

os.environ["DEMO_MODE"] = "on"
os.environ.pop("GROQ_API_KEY", None)

import io
import math
import struct
import wave

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def _fake_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        frames = b"".join(
            struct.pack("<h", int(0.3 * 32767 * math.sin(2 * math.pi * 440 * i / 8000)))
            for i in range(8000)
        )
        wav.writeframes(frames)
    return buf.getvalue()


def test_health(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    # These are booleans whose values depend on the machine (e.g. a local .env
    # with a real key, or Redis running) — assert the type, not the value.
    assert isinstance(body["groq_key_configured"], bool)
    assert isinstance(body["redis_connected"], bool)
    assert isinstance(body["version"], str)
    assert body["demo_mode"] is True  # forced on at module import


def test_demo_flow(client: TestClient) -> None:
    # Upload a fake wav → 202 with a job id, then poll until done.
    up = client.post("/upload", files={"audio": ("demo.wav", _fake_wav(), "audio/wav")})
    assert up.status_code == 202, up.text
    job_id = up.json()["job_id"]
    assert job_id

    done = client.get(f"/jobs/{job_id}")
    assert done.status_code == 200
    body = done.json()
    assert body["status"] == "done", body
    upload = body["upload"]
    conversation_id = upload["conversation_id"]
    assert upload["chunk_count"] >= 1
    assert upload["demo"] is True

    # The transcript is retrievable.
    tr = client.get(f"/transcript/{conversation_id}")
    assert tr.status_code == 200
    assert len(tr.json()["transcript"]) > 0

    # Ask a question — first call uncached, second served from cache.
    first = client.post(
        "/ask",
        json={"conversation_id": conversation_id, "question": "What did the customer say about pricing?"},
    )
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["cached"] is False
    assert body["demo"] is True
    assert len(body["sources"]) > 0
    assert isinstance(body["sources"][0]["score"], float)
    assert "price" in body["answer"].lower()

    second = client.post(
        "/ask",
        json={"conversation_id": conversation_id, "question": "What did the customer say about pricing?"},
    )
    assert second.status_code == 200
    assert second.json()["cached"] is True


def test_ask_unknown_conversation(client: TestClient) -> None:
    res = client.post("/ask", json={"conversation_id": "does-not-exist", "question": "hi"})
    assert res.status_code == 200
    body = res.json()
    assert body["sources"] == []
    assert "indexed" in body["answer"].lower() or "upload" in body["answer"].lower()


def test_transcript_404(client: TestClient) -> None:
    res = client.get("/transcript/nope")
    assert res.status_code == 404


def test_upload_empty_file_400(client: TestClient) -> None:
    res = client.post("/upload", files={"audio": ("empty.wav", b"", "audio/wav")})
    assert res.status_code == 400


def test_upload_job_404(client: TestClient) -> None:
    res = client.get("/jobs/does-not-exist")
    assert res.status_code == 404
