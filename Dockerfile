# syntax=docker/dockerfile:1
# PulseRAG — single-stage image.
#
# The embedding model is baked in at BUILD time (build = has internet) so the
# runtime is fully offline for embeddings — no first-request stall, no runtime
# download (see IMPLEMENTATION_PLAN.md §9.1).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/.cache/huggingface \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install locked dependencies first (cached layer, no dev group)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Bake the embedding model at build time (needs internet during `docker build`).
# Which model cache gets baked is driven by the build arg so the image always
# ships the backend it will actually use (e.g. fastembed for Render's 512MB).
ARG EMBEDDING_BACKEND=sentence_transformers
RUN if [ "$EMBEDDING_BACKEND" = "fastembed" ]; then \
      uv run python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='all-MiniLM-L6-v2')"; \
    else \
      uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"; \
    fi
ENV EMBEDDING_BACKEND=${EMBEDDING_BACKEND}

# Runtime: embeddings are fully offline
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Copy the application source
COPY app ./app
COPY scripts ./scripts

# Re-sync so the venv is consistent with the copied sources
RUN uv sync --frozen --no-dev

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
