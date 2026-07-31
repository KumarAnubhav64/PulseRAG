# PulseRAG

Transcribe audio with **Groq Whisper**, index it with **local embeddings + FAISS**, and answer questions with **grounded RAG** — all behind a clean, layered FastAPI backend.

```
Audio file ──► [Groq Whisper] ──► transcript ──► chunk ──► embed (local) ──► FAISS index
                                                                              │
User question ──► cache check ──► retrieve top-k chunks ──► [Groq LLM] ──► grounded answer ──► cache store
```

## Quickstart (no Docker, no API key needed)

```bash
# 1. Install uv (one-time) — https://docs.astral.sh/uv/
# 2. Install dependencies
uv sync

# 3. Copy the env template (placeholders only; add a real key later if you want)
cp .env.example .env

# 4. Run the API
uv run uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** (Swagger UI) or **http://localhost:8000/** (Phase 2
mic/chat frontend — record from your microphone, upload, then ask questions).
Without a `GROQ_API_KEY` the app runs in **demo mode**: transcription and answers
are mock, so the entire flow works out of the box.

```bash
# 5. Try it end to end
uv run python scripts/make_sample_audio.py     # writes sample_audio.wav
#    → POST sample_audio.wav to /upload, then ask /ask:
#      "What did the customer say about pricing?"
```

## Endpoints

| Method | Path | Body/Param | Returns |
|---|---|---|---|
| `POST` | `/upload` | multipart `audio` file (<25MB) | `{job_id, status}` — async; poll below |
| `GET` | `/jobs/{job_id}` | path param | `{job_id, status, upload?, error?}` — `status` is `pending` \| `processing` \| `done` \| `failed` |
| `POST` | `/ask` | `{conversation_id, question}` | `{answer, sources[], cached, demo}` |
| `GET` | `/transcript/{conversation_id}` | path param | `{conversation_id, transcript}` |
| `GET` | `/health` | — | `{status, demo_mode, groq_key_configured, redis_connected, embedding_backend, version}` |

Uploads are processed **asynchronously**: `POST /upload` streams the file to a
temp file (keeping it out of RAM), returns a job id, and transcribes + indexes
in the background. Poll `GET /jobs/{job_id}` until `status` is `done` (then
`upload` holds `{conversation_id, chunk_count, transcript_length, demo}`) or
`failed` (`error` explains why).

`/health` reveals *whether* a Groq key is configured — never the key itself.

## Configuration

All settings live in `.env` (see `.env.example`). `DEMO_MODE=auto` means: demo
mode only when `GROQ_API_KEY` is missing. Set `DEMO_MODE=off` to force real
Groq calls (and real failures without a key).

| Variable | Default | Notes |
|---|---|---|
| `GROQ_API_KEY` | — | Optional; enables real Whisper + LLM |
| `DEMO_MODE` | `auto` | `auto` \| `on` \| `off` |
| `REDIS_URL` | `redis://localhost:6379/0` | Falls back to an in-memory TTL cache if unreachable |
| `EMBEDDING_BACKEND` | `sentence_transformers` | `sentence_transformers` (torch) \| `fastembed` (ONNX, low-RAM) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | ~90MB, downloaded once on first use |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `50` | Chunking for retrieval |
| `TOP_K` | `4` | Chunks fed to the LLM |
| `CACHE_TTL_SECONDS` | `3600` | Answer cache TTL |
| `MAX_UPLOAD_MB` | `25` | Upload size limit |
| `PRELOAD_EMBEDDINGS` | `true` | Load the embedding model at startup instead of on the first upload — keeps the memory spike out of the request path on small instances |

## Project layout

```
app/
├── main.py                 # FastAPI app factory + lifespan
├── config.py               # pydantic-settings — the env safety boundary
├── api/routes.py           # HTTP handlers (/upload /ask /transcript /health)
├── models/schemas.py       # Pydantic contracts
├── repositories/           # transcript (in-mem), vector (FAISS), cache (Redis + fallback)
└── services/               # transcription, embedding, llm, rag (orchestration)
tests/test_basic.py         # /health + demo-mode flow
scripts/make_sample_audio.py
```

Dependency flow is one-directional: **api → service → repository**, with
**models** as shared contracts.

## Docker

```bash
docker compose up --build
# API on http://localhost:8000, Redis on 6379
```

The embedding model is **baked into the image at build time** and the runtime
is fully offline for embeddings (`HF_HUB_OFFLINE=1`), so the first request is
fast even with no internet. Set `GROQ_API_KEY` in your shell or the compose
file to leave demo mode.

## Tests

```bash
uv run pytest
```

> First run downloads the ~90MB embedding model from HuggingFace; after that it's cached.

## Known limitations (Phase 1)

- Transcripts, FAISS indexes, and in-flight jobs are **in-memory** — restarting the app clears them.
- No speaker diarization, no auth (fine locally).
- Demo mode is mock, not real.
- Groq free-tier rate limits apply (retry/backoff is a later step).

## Roadmap

- **Phase 2:** simple browser frontend (upload + chat).
- **Phase 3:** deployed to Render (free, no credit card). **Local dev uses
`sentence_transformers` (torch); the deployed Render service uses `fastembed`
(ONNX)** — the same model, but fastembed fits Render's 512MB free tier where
torch (~1-2GB) would OOM. `EMBEDDING_BACKEND` selects the backend per env.
