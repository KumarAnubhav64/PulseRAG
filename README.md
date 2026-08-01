# PulseRAG

Transcribe audio with **Groq Whisper**, index it with **embeddings + FAISS**, and answer questions with **grounded RAG** — all behind a clean, layered FastAPI backend that boots in ~80MB of RAM (fits Render's free 512MB tier).

```
Audio file ──► [Groq Whisper] ──► transcript ──► chunk ──► embed ──► FAISS index
                                                                   │
User question ──► cache check ──► retrieve top-k chunks ──► [Groq LLM] ──► grounded answer ──► cache store
```

Embeddings can run **locally** (`sentence_transformers` / `fastembed`) or **remotely** (Mistral API). The remote backend loads **zero model RAM** — it's what makes the app fit on Render's 512MB free tier. No langchain: the whole pipeline uses plain Python + raw `faiss`/`groq` SDKs (a langchain import alone used to cost ~700MB at boot).

## Quickstart (no Docker, no API key needed)

```bash
# 1. Install uv (one-time) — https://docs.astral.sh/uv/
# 2. Install dependencies
uv sync

# 3. Copy the env template (placeholders only; add real keys later if you want)
cp .env.example .env

# 4. Run the API
uv run uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** (Swagger UI) or **http://localhost:8000/** (mic/chat frontend — record from your microphone, upload, then ask questions; shadcn-style UI with a light/dark theme toggle). Without a `GROQ_API_KEY` the app runs in **demo mode**: transcription and answers are mock, so the entire flow works out of the box.

```bash
# 5. Try it end to end
uv run python scripts/make_sample_audio.py     # writes sample_audio.wav
#    → POST sample_audio.wav to /upload, then ask /ask:
#      "What did the customer say about pricing?"
```

Local dev defaults to `sentence_transformers` (torch) embeddings. On a small-RAM host, either switch to `fastembed` (`EMBEDDING_BACKEND=fastembed`, ~220MB) or to the zero-RAM remote backend (`EMBEDDING_BACKEND=remote` + `MISTRAL_API_KEY`).

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
| `EMBEDDING_BACKEND` | `sentence_transformers` | `sentence_transformers` (torch, ~760MB) \| `fastembed` (ONNX, ~220MB) \| `remote` (Mistral API, ~0MB — **use this on Render free's 512MB**) |
| `MISTRAL_API_KEY` | — | Required when `EMBEDDING_BACKEND=remote`; free tier, no credit card |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local backends only; ~90MB, downloaded once on first use |
| `REMOTE_EMBEDDING_MODEL` | `mistral-embed` | Model used by the remote backend |
| `EMBEDDING_THREADS` | `1` | onnxruntime intra-op threads (fastembed) — cap on small instances to stay under the RAM limit |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `50` | Chunking for retrieval |
| `TOP_K` | `4` | Chunks fed to the LLM |
| `CACHE_TTL_SECONDS` | `3600` | Answer cache TTL |
| `MAX_UPLOAD_MB` | `25` | Upload size limit |
| `PRELOAD_EMBEDDINGS` | `true` | Preload the local model at boot instead of on the first upload — keeps the memory spike out of the request path on small instances |

## Project layout

```
app/
├── main.py                 # FastAPI app factory + lifespan
├── config.py               # pydantic-settings — the env safety boundary
├── api/routes.py           # HTTP handlers (/upload /ask /transcript /health)
├── models/schemas.py       # Pydantic contracts
├── repositories/           # transcript (in-mem), vector (raw faiss), cache (Redis + fallback)
└── services/               # transcription (Groq), embedding (3 backends), llm (Groq), rag (orchestration)
tests/                      # 61 tests — demo flow, backends, repositories, routes
scripts/make_sample_audio.py
```

Dependency flow is one-directional: **api → service → repository**, with
**models** as shared contracts. No langchain anywhere in the import graph.

## Running on Render's 512MB free tier

The app is deliberately slim enough to boot inside 512MB:

- **No langchain imports.** The previous langchain stack measured ~755MB of RSS
  at boot — the app OOM-crash-looped before serving `/health`. Replaced with a
  plain-Python chunker, raw `faiss` + `numpy` (L2-normalized cosine search), and
  the raw `groq` SDK.
- **`EMBEDDING_BACKEND=remote`** (see `render.yaml`): embeddings come from the
  Mistral API, so no model is ever loaded into RAM. Measured: app boots at
  **~78MB**; even the `fastembed` fallback stays under 300MB.
- Uploads stream to disk, transcription/indexing runs in a background job, and
  `OMP_NUM_THREADS=1` caps native thread pools — all keeping memory spikes out
  of the request path.

To deploy: push to GitHub, create a Blueprint from `render.yaml`, and set the
`GROQ_API_KEY` and `MISTRAL_API_KEY` secrets in the Render dashboard. The image
still bakes the local embedding models at build time, so you can flip
`EMBEDDING_BACKEND` back to `fastembed` anytime with no rebuild.

## Docker

```bash
docker compose up --build
# API on http://localhost:8000, Redis on 6379
```

The local embedding models are **baked into the image at build time** and the
runtime is fully offline for embeddings (`HF_HUB_OFFLINE=1`). Set
`GROQ_API_KEY` (and `MISTRAL_API_KEY` if using `EMBEDDING_BACKEND=remote`) in
your shell or the compose file to leave demo mode.

## Tests

```bash
uv run pytest
```

> 61 tests: demo-mode end-to-end flow, all three embedding backends, FAISS
> search, chunking, and route validation. The first run downloads the ~90MB
> embedding model from HuggingFace for `test_basic`; after that it's cached.

## Known limitations

- Transcripts, FAISS indexes, and in-flight jobs are **in-memory** — restarting the app clears them.
- No speaker diarization, no auth (fine locally).
- Demo mode is mock, not real.
- Groq and Mistral free-tier rate limits apply (retry/backoff is a later step).

## Roadmap

- **Done:** browser frontend (shadcn-style UI, light/dark theme), Render
  deployment blueprint, and the 512MB memory fix (no langchain + remote
  embeddings).
- **Next:** persistent storage (SQLite/Postgres for transcripts + FAISS
  serialization), retry/backoff for Groq/Mistral rate limits, auth, speaker
  diarization.
