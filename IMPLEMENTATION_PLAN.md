# PulseRAG — Implementation Plan

> Phase 1 deliverable: **plan only** (no code yet). Review this, then I build.
> Goal: a clean, layered FastAPI backend (`api / model / repository / service`) that you can test **immediately via Swagger UI** — even before you have a Groq key or Docker running.

---

## 1. What we're building (recap)

```
Audio file ──► [Groq Whisper] ──► transcript ──► chunk ──► embed (local) ──► FAISS index
                                                                              │
User question ──► cache check ──► retrieve top-k chunks ──► [Groq LLM] ──► grounded answer ──► cache store
```

FastAPI exposes it as REST endpoints. Everything runs locally first; Docker packaging comes after it works.

---

## 2. Tech decisions (validated against your machine)

Your machine has **Python 3.14.4 installed**, but the project will be **pinned to Python 3.12** — the most stable version across the whole LangChain + torch + faiss stack (py3.14 is still not supported by LangChain).

| Concern | Decision | Why |
|---|---|---|
| Web framework | **FastAPI + uvicorn** | As you asked; free Swagger UI at `/docs` |
| STT | **`groq`** (official client) | Whisper `whisper-large-v3-turbo` |
| LLM | **`langchain-groq`** → `ChatGroq` | LLM `llama-3.3-70b-versatile`, RAG prompt building |
| Embeddings | **Pluggable backend** (`EMBEDDING_BACKEND`): locally `langchain-huggingface` → `HuggingFaceEmbeddings` (torch); on 512MB deploys `langchain-community` → `FastEmbedEmbeddings` (ONNX, no torch) | Groq has **no embeddings API**; both are free + offline |
| Vector store | **`langchain-community`** → `FAISS` + `faiss-cpu` | The canonical LangChain vector store |
| Chunking | **`langchain-text-splitters`** → `RecursiveCharacterTextSplitter` (500 tokens, 50 overlap) | Overlap prevents answers cut at chunk boundaries |
| Cache | **Redis** via Docker + **in-memory TTL fallback** | Swagger testing works with *zero* Docker; Redis takes over when available |
| Config / env safety | **`pydantic-settings`** + `.env` | Strict typing, single source of truth, secrets never logged |
| Uploads | `python-multipart` | Needed for `POST /upload` file handling |
| Project mgmt | **`uv`** | One tool for venv + deps + lockfile + running — replaces pip/venv entirely |

### 2.1 Hardware check — your machine (Ryzen 5 5600H · 14GB RAM · 67GB free · no CUDA)

- ✅ **CPU 6C/12T** — `all-MiniLM-L6-v2` embeds in ~100–300ms on CPU; FAISS search is trivial at demo scale.
- ✅ **RAM** — pipeline needs ~2–3GB once torch + sentence-transformers load (torch alone reserves ~1GB); ~5.7GB is available, close heavy apps if it gets tight. Tight? Switch `EMBEDDING_BACKEND=fastembed` (ONNX) → drops to ~200MB.
- ✅ **Disk** — 67GB free; deps incl. PyTorch-CPU ≈ 2–3GB, model ≈ 90MB.
- ⚠️ **GPU** — integrated Radeon, no CUDA. Doesn't matter: Whisper + LLM run on Groq's servers; only embeddings are local, and they're CPU-friendly.
- ✅ **OS** — Ubuntu x86_64, exactly the platform faiss-cpu / torch wheels target.

**Using the modern LangChain split** (each integration is its own small package — no heavyweight install):
`langchain` · `langchain-text-splitters` · `langchain-community` (FAISS) · `langchain-groq` · `langchain-huggingface` · `faiss-cpu` · `sentence-transformers` · `fastembed` (ONNX backend — **required for the 512MB Render deploy**). All pin cleanly on **Python 3.12**.

> ⚠️ **First-run note:** `all-MiniLM-L6-v2` (~90MB) is downloaded from HuggingFace on first use — you need internet the first time, even in demo mode. Cache it once and it's offline afterwards.
> ⚠️ **Hedge:** `langchain-community` is in a phased archival discussion; FAISS currently lives there. If it moves, we swap one import — the repository layer isolates this.

---

## 3. Clean layered architecture

Dependency flow is one-directional: **api → service → repository**, with **models** shared as contracts between layers.

```
pulserag/
├── app/
│   ├── main.py                 # FastAPI app factory + router registration + startup/shutdown
│   ├── config.py               # Settings (pydantic-settings) — THE env safety boundary
│   ├── api/
│   │   └── routes.py           # HTTP handlers: /upload /ask /transcript /health
│   ├── models/
│   │   └── schemas.py          # Pydantic contracts: request/response bodies
│   ├── repositories/           # DATA ACCESS — how things are stored/read
│   │   ├── transcript_repository.py  # transcripts by conversation_id (in-memory dict; sqlite later)
│   │   ├── vector_repository.py      # FAISS index + chunks per conversation
│   │   └── cache_repository.py       # Redis w/ in-memory TTL fallback
│   └── services/               # BUSINESS LOGIC — orchestration
│       ├── transcription_service.py  # Groq Whisper + demo-mode fallback
│       ├── embedding_service.py      # HuggingFaceEmbeddings wrapper (lazy model load)
│       ├── llm_service.py            # ChatGroq wrapper + demo-mode fallback
│       └── rag_service.py            # chunk → embed → store; retrieve → prompt → answer
├── tests/
│   └── test_basic.py           # /health + demo-mode flow
├── scripts/
│   └── make_sample_audio.py    # generates a tiny test .wav so you can try /upload
├── .env.example                # COMMITTED — placeholders only, no real secrets
├── .gitignore                  # excludes .env, caches, models
├── .python-version            # pins Python 3.12 (mise + uv both read it)
├── pyproject.toml              # deps — managed by uv
├── uv.lock                     # locked, reproducible environment
├── Dockerfile
├── docker-compose.yml          # api + redis
├── README.md
└── IMPLEMENTATION_PLAN.md      # this file
```

### Layer responsibilities

| Layer | Owns | Does NOT do |
|---|---|---|
| **api** (`routes.py`) | HTTP contracts, status codes, multipart parsing | No business logic — just calls services |
| **models** (`schemas.py`) | Pydantic schemas: `UploadResponse`, `AskRequest`, `AskResponse`, `TranscriptResponse`, `HealthResponse` | No I/O |
| **service** | Orchestration: transcribe→chunk→embed→store; retrieve→generate; demo-mode decisions | No direct DB/vector/cache calls — goes through repositories |
| **repository** | Concrete storage: in-memory/FAISS/Redis + fallbacks | No HTTP, no prompt-building |

---

## 4. API surface (Swagger-ready)

| Method | Path | Body/Param | Returns |
|---|---|---|---|
| `POST` | `/upload` | multipart `audio` file (<25MB) | `{conversation_id, chunk_count, transcript_length}` |
| `POST` | `/ask` | `{conversation_id, question}` | `{answer, sources[], cached}` |
| `GET` | `/transcript/{conversation_id}` | path param | `{conversation_id, transcript}` |
| `GET` | `/health` | — | `{status, groq_key_configured, redis_connected, version}` |

`/health` reveals *whether* a key is configured — **never the key itself**.

---

## 5. Env safety (your explicit requirement)

- **`.env.example` is committed** with placeholder values only:
  ```
  GROQ_API_KEY=your_groq_key_here
  REDIS_URL=redis://localhost:6379/0
  EMBEDDING_MODEL=all-MiniLM-L6-v2
  TOP_K=4
  CACHE_TTL_SECONDS=3600
  DEMO_MODE=auto
  ```
- **`.env` is gitignored** — never committed, never pushed. You paste your real key there locally.
- `config.py` (pydantic-settings) is the only place env vars are read; services receive a typed `Settings` object.
- **No logging of the key anywhere.** Errors that touch Groq log status codes, not the key.
- `DEMO_MODE=auto`: if `GROQ_API_KEY` is missing → transcription/LLM services return mock data so the **entire flow works on Swagger today**. Set `DEMO_MODE=off` to force real-call failures instead.

---

## 6. Step-by-step build order (after your approval)

| # | Step | Testable via | Done when |
|---|---|---|---|
| 1 | Scaffold: pin Python 3.12 (`.python-version`), `uv init` + `uv add` all deps, `.gitignore`, `.env.example` | `uv sync` | deps lock cleanly on py3.12 |
| 2 | `config.py` + `models/schemas.py` | import check | types load |
| 3 | Repositories (transcript in-mem, vector FAISS, cache w/ Redis→in-mem fallback) | unit smoke | all three answer reads/writes |
| 4 | Services (transcription, embedding, llm, rag) incl. demo mode | unit smoke | demo-mode transcribe+ask work |
| 5 | `api/routes.py` + `main.py` | **Swagger UI at `/docs`** | `/health`, `/upload` (demo wav), `/ask`, `/transcript` all pass |
| 6 | Dockerfile (`uv sync --frozen` inside, **model baked in at build time**) + docker-compose (api + redis) | `docker compose up` | runs with one command |
| 7 | README + `tests/test_basic.py` + optional CI | `uv run pytest` | green |
| 8 | **Phase 2:** simple frontend (HTML/JS page calling the API) | browser | upload + chat work in browser |
| 9 | **Phase 3: deploy to Render** (see §9) — no credit card needed | public URL | /health + /ask work online |

---

## 7. What YOU need to do now (right after I build Phase 1)

0. **Pin Python 3.12** (LangChain + torch + faiss are rock-solid here): `mise use python@3.12` — creates `.python-version`, which `uv` respects automatically.
1. **Install `uv`** (not installed yet): `mise use -g uv@latest` — or `curl -LsSf https://astral.sh/uv/install.sh | sh`.
2. **Get a free Groq API key** → `console.groq.com` → API Keys → create. (Free tier.)
3. `cp .env.example .env` and paste the key into `.env`.
4. `uv run uvicorn app.main:app --reload` → open **http://localhost:8000/docs**.
5. Run `uv run python scripts/make_sample_audio.py` (generates a tiny `.wav`), upload it via Swagger, then ask a question like *"what did the customer say about pricing?"*.
6. No key handy? Everything still works in demo mode — just skip step 2.

---

## 8. Known limitations (carried over from the doc, honest by design)

- No speaker diarization · no async ingestion queue · single-node in-memory vectors · demo mode is mock, not real · no auth (fine locally) · Groq free-tier rate limits (add retry/backoff later).
- FAISS index lives in memory — restarting the app loses indexes (acceptable for Phase 1; sqlite/persistence is a later step).

---

## 9. Deployment — make the model run online (Phase 3)

### 9.1 The critical trick: bake the embedding model into the image

The `all-MiniLM-L6-v2` model (~90MB) must **not** be downloaded at container runtime — first request would be slow or fail entirely if HuggingFace is unreachable. Instead:

```dockerfile
# In the Dockerfile (build time = has internet)
ENV HF_HOME=/app/.cache/huggingface
RUN uv sync --frozen \
  && uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Runtime = fully offline for embeddings
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

This guarantees **the model runs online** the moment the container starts. Gotchas: the bake step must come **after** `uv sync --frozen`; if the Dockerfile is multi-stage, `COPY` the HF cache dir into the runtime stage with the **same `HF_HOME`** in both stages.

For the **fastembed** backend (Render), bake the same way: `uv run python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='all-MiniLM-L6-v2')"` — it caches under the same HF dir, so the offline env vars still apply. The actual Dockerfile branches on `EMBEDDING_BACKEND` (or bakes the ONNX model) so the right bake command ships with the right backend.

### 9.2 Platform choice — RAM *and* the no-credit-card constraint

Two constraints decide this: **no billing card required** + **enough RAM for the model**:

| Platform | Free tier RAM | Credit card? | Verdict |
|---|---|---|---|
| **Render** | 512MB, spins down after 15min (~1min cold start) | **No** | ✅ **Recommended** — with fastembed (ONNX), not torch |
| Railway | 512MB permanent (1GB during 30-day trial) | No (trial) | ⚠️ Borderline; $1/mo credit |
| Google Cloud Run | Configurable 1–2GB free quota | **Yes, required** | ⚠️ Great fit, but needs a card |
| HF Spaces (Docker) | 16GB / 2 vCPU (but **Docker needs paid plan**) | No | ❌ Docker Spaces now need a **paid plan** (only static/Gradio-ZeroGPU are free) |

**The fix for 512MB: swap torch for ONNX.** Locally we use `sentence-transformers` (torch, ~350–450MB — fine on your laptop). For Render, the embedding service switches to **`fastembed`** (`langchain_community.embeddings.FastEmbedEmbeddings`, ONNX runtime, zero torch) — **~150–250MB total**, leaving headroom for FastAPI + uvicorn. Same `all-MiniLM-L6-v2` model, near-identical embeddings, toggled via the `EMBEDDING_BACKEND` env var. This is exactly why the embedding backend is abstracted behind `embedding_service.py`.

### 9.3 Render deploy — the card-free path (primary)

1. Push the repo to GitHub.
2. Render dashboard → **New → Web Service** → connect the repo.
3. Render auto-detects the `Dockerfile`; add env vars:
   - `GROQ_API_KEY` — set as a **secret** (never committed, never in the image)
   - `EMBEDDING_BACKEND=fastembed` — ONNX backend to fit 512MB
   - `DEMO_MODE=auto`
4. Set the health-check path to `/health`. Deploy.

- **Model runs online:** baked into the image at build time (§9.1), `HF_HUB_OFFLINE=1` at runtime — no download, no first-request stall.
- **Redis on prod:** Render free has no managed Redis → our **in-memory cache fallback** becomes the production cache automatically when `REDIS_URL` is absent. This is exactly why we built the fallback. Caveat: it's **per-instance** (fine for a demo — it's an optimization, not correctness).
- **Cold start:** free tier spins down after ~15min idle; first request after idle takes ~30–60s to wake. Mention this upfront if demoing live (or keep the tab open).
- **Quotas & ephemeral FS:** 750 free instance-hours/mo (one always-on service fits) and the filesystem is **ephemeral** — wiped on redeploy/restart/spin-down, so nothing is persisted on disk.
- **Note:** FAISS indexes are per-instance/in-memory → vectors lost on restart; acceptable for a demo, and the README will state it.

### 9.4 Alternate path — Cloud Run (only if you add a card)

Best *fit* (1–2GB RAM, always-free quota) and reinforces your GCP internship story — but requires a billing card:

```bash
gcloud run deploy pulserag --source . --allow-unauthenticated \
  --memory 1Gi --cpu 1 --port 8000 --region us-central1 --min-instances 0
```

- **`--port 8000` is critical** — Cloud Run injects `PORT` (default 8080) and only routes traffic to the port the container listens on. Pass `--port 8000` (matching uvicorn) or better, run uvicorn with `--port ${PORT:-8000}` so the platform can override it.
- **Free quota reality check:** 1Gi burns 360k GiB-s in **~100 active hours/month** (2Gi ≈ 50h) — fine with scale-to-zero + light traffic.
- **Cold start:** scale-to-zero → first request wakes a container (~1–3s + model load). Keep `--min-instances 0` for free tier.
- Same env-safety rules: `GROQ_API_KEY` in the console env vars, never in the repo/image.

### 9.5 Railway — viable backup

Free trial gives ~1GB (enough even for torch), then drops to 512MB + $1/mo credit — works with the same fastembed setting, but the ongoing credit math is worse than Render's always-free tier.

---

## 10. Embeddings explained — the part that makes RAG work (Phase 1 addition)

### 10.1 What an embedding is

An **embedding** turns text into a list of numbers (a **vector**) such that *semantically similar* texts end up close together in that numeric space. `all-MiniLM-L6-v2` maps any sentence to a 384-dim vector. "Similar" is measured by **cosine similarity** (the angle between vectors): closer angle → closer meaning.

### 10.2 Why we embed instead of keyword-match

Keyword search fails on paraphrase: *"what did the customer say about cost?"* vs. the transcript *"they said the price was too high."* Embeddings capture meaning, so retrieval finds the right chunk even when the wording differs — this is the core of RAG.

### 10.3 The two embedding backends (torch vs ONNX)

| Backend | Library | RAM at load | Where it runs |
|---|---|---|---|
| `sentence_transformers` | `langchain_huggingface.HuggingFaceEmbeddings` (torch) | ~350–450MB | Your laptop (Phase 1 default) |
| `fastembed` | `langchain_community.embeddings.FastEmbedEmbeddings` (ONNX) | ~150–250MB | Render's 512MB free tier |

Both run the *same* `all-MiniLM-L6-v2` model → near-identical vectors, so switching backends does not change retrieval quality meaningfully. The **`EMBEDDING_BACKEND`** env var selects it in `embedding_service.py` — this is the layer that makes the swap invisible to the rest of the app.

### 10.4 Chunking + overlap (why it matters)

Long transcripts are split into chunks (500 chars, 50 overlap). The **overlap** prevents an answer that spans a chunk boundary from being cut off — a classic RAG quality trap. Each chunk gets embedded and stored in FAISS with its text.

### 10.5 The retrieval flow (how an answer is grounded)

```
1. Transcript → chunk → embed each chunk → store vectors in FAISS (keyed by conversation_id)
2. User question → embed the question with the SAME model
3. FAISS cosine search → top-k most similar chunks
4. Chunks + question → ChatGroq prompt: "answer ONLY from context; say 'not mentioned' if absent"
5. Answer → cached by (conversation_id, question-hash) for TTL
```

### 10.6 Where to read these AI concepts (curated learning path)

| Concept | Where to learn it |
|---|---|
| Embeddings, visually | Jay Alammar — "How to explain embeddings": https://jalammar.github.io/ |
| RAG end-to-end | LangChain docs — "Retrieval" + "Vector stores": https://python.langchain.com/docs |
| RAG, the paper | Lewis et al. 2020, *Retrieval-Augmented Generation*: https://arxiv.org/abs/2005.11401 |
| Sentence-transformers (local embeddings) | https://www.sbert.net/ + the model card for `all-MiniLM-L6-v2` |
| FAISS internals | https://github.com/facebookresearch/faiss (wiki on index types) |
| Vector databases/embeddings book | Pinecone's free *Vector Embeddings* ebook: https://www.pinecone.io/learn/ |
| LLM prompting foundations | DeepLearning.AI short courses (free): https://www.deeplearning.ai/courses/ |
| Practical RAG with LangChain | LangChain official tutorials: https://python.langchain.com/docs/tutorials/ |
| Whisper STT | OpenAI Whisper GitHub: https://github.com/openai/whisper |
| Groq APIs | https://console.groq.com/docs (models + rate limits) |
