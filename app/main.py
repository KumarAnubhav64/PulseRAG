"""FastAPI app factory: lifespan, router registration, static frontend, uvicorn entry.

Run with:  uv run uvicorn app.main:app --reload
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import _get_rag_service, router
from .config import get_settings

logger = logging.getLogger(__name__)

# Phase 2: the mic/chat frontend lives here.
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    if settings.preload_embeddings:
        try:
            # Force the (cached) RAG service into existence now so the embedding
            # model loads at boot — not on the first upload, where the ~200-300MB
            # load spike could push a small instance past its memory limit
            # mid-request (the cause of the Render free-tier 502s).
            _get_rag_service()
        except Exception as exc:  # keep the app bootable if the model can't load
            logger.warning("Embedding model preload failed at startup: %s", exc)
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
