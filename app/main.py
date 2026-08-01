"""FastAPI app factory: lifespan, router registration, static frontend, uvicorn entry.

Run with:  uv run uvicorn app.main:app --reload
"""

import logging
import threading
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


def _warmup_embeddings() -> None:
    """Load the embedding model; failures must not take the app down."""
    try:
        _get_rag_service()
    except Exception as exc:  # keep the app bootable if the model can't load
        logger.warning("Embedding model preload failed at startup: %s", exc)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    if settings.preload_embeddings:
        # Warm the model in a daemon thread: uvicorn won't serve until lifespan
        # startup returns, and blocking here could let Render time out a slow
        # 0.1-CPU boot. The model still loads before the first upload finishes
        # streaming, so the ~200-300MB spike stays out of the request path (the
        # cause of the Render free-tier 502s).
        threading.Thread(target=_warmup_embeddings, daemon=True).start()
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
