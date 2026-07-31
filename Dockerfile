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
    # Cap onnxruntime/native thread pools to 1 (small instances over-report cores)
    OMP_NUM_THREADS=1 \
    # fastembed does NOT use HF_HOME — it caches to FASTEMBED_CACHE_PATH.
    # Pin it to the same /app/.cache tree so the model baked below is found
    # at runtime (the bake and runtime must share this exact path).
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install locked dependencies first (cached layer, no dev group)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Bake BOTH embedding models at build time (needs internet during `docker build`).
# Render's blueprint spec has no dockerBuildArgs field, so instead of an ARG-
# driven bake we ship both caches and let the runtime EMBEDDING_BACKEND env var
# select the backend. Extra image size (~90MB) is the tradeoff; RAM at runtime
# is unaffected (only the selected backend loads).
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
 && uv run python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')"

# Runtime: embeddings are fully offline
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Copy the application source
COPY app ./app
COPY scripts ./scripts

# Re-sync so the venv is consistent with the copied sources
RUN uv sync --frozen --no-dev

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
