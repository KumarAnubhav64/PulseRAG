"""Route-level tests: validation edge cases and the app factory.

The full demo flow (upload → transcript → ask → cache) is covered end-to-end in
``test_basic.py``. These tests cover only the *error* paths, which are validated
before any service (and therefore the embedding model) is reached.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app, create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def test_app_factory_registers_routes() -> None:
    # Read paths from the OpenAPI schema — robust across Starlette versions that
    # wrap included routers in _IncludedRouter objects (which have no .path).
    paths = set(create_app().openapi()["paths"].keys())
    assert "/upload" in paths
    assert "/ask" in paths
    assert "/transcript/{conversation_id}" in paths
    assert "/health" in paths


def test_health_shape(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["demo_mode"] is True  # forced on via conftest
    assert isinstance(body["groq_key_configured"], bool)
    assert isinstance(body["redis_connected"], bool)
    assert isinstance(body["embedding_backend"], str)
    assert body["version"] == "0.1.0"


def test_ask_blank_question_400(client: TestClient) -> None:
    res = client.post("/ask", json={"conversation_id": "c1", "question": "   "})
    assert res.status_code == 400


def test_ask_missing_question_422(client: TestClient) -> None:
    res = client.post("/ask", json={"conversation_id": "c1"})
    assert res.status_code == 422


def test_upload_missing_file_422(client: TestClient) -> None:
    res = client.post("/upload")
    assert res.status_code == 422


def test_upload_too_large_413(client: TestClient) -> None:
    big = b"x" * (26 * 1024 * 1024)  # 26 MB > 25 MB limit
    res = client.post("/upload", files={"audio": ("big.wav", big, "audio/wav")})
    assert res.status_code == 413


def test_frontend_index_served(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "PulseRAG" in res.text
    assert "MediaRecorder" in res.text  # mic recording UI present


def test_frontend_static_asset_served(client: TestClient) -> None:
    res = client.get("/static/index.html")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
