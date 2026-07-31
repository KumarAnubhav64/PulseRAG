"""FastAPI app factory: lifespan, router registration, static frontend, uvicorn entry.

Run with:  uv run uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import router
from .config import get_settings

# Phase 2: the mic/chat frontend lives here.
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Nothing to set up in Phase 1: repositories are in-memory and the cache
    # falls back to a process-local store when Redis is absent.
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=(
            "PulseRAG: transcribe audio with Groq Whisper, index it with local "
            "embeddings + FAISS, and answer questions with grounded RAG."
        ),
        lifespan=lifespan,
    )
    app.include_router(router)

    # Phase 2 frontend: serve static assets and the single-page app at /.
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
