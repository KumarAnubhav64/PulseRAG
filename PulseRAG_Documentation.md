# PulseRAG — Real-Time Conversational Intelligence Pipeline

**Build documentation for a RAG + STT project targeting the Darwix AI stack**

---

## 1. What You're Building

A pipeline that takes a recorded audio conversation (sales call, meeting, interview) and lets you ask natural-language questions about it, grounded in the actual transcript.

```
Audio file
    │
    ▼
[Groq Whisper API] ──► Raw transcript
    │
    ▼
[LangChain: chunk + embed] ──► Vector embeddings
    │
    ▼
[FAISS] ──► Vector store (searchable)
    │
    ▼
User question ──► [Retrieve top-k chunks] ──► [Groq LLM] ──► Grounded answer
    │
    ▼
[Redis] ──► Cache repeated queries/embeddings
    │
    ▼
[FastAPI] ──► Exposes it all as REST endpoints
    │
    ▼
[Docker] ──► Containerized, reproducible
```

This directly mirrors Darwix AI's "agent-assist" product: capture conversation → transcribe → extract insight in real time.

---

## 2. Requirements

### 2.1 Accounts (free tier is enough)

| Service | Purpose | Free tier |
|---|---|---|
| Groq | Whisper STT + Llama LLM calls | Yes, generous free tier |
| GitHub | Repo hosting + Actions CI | Yes |
| Docker Hub (optional) | If you want to push an image | Yes |

Sign up at `console.groq.com`, generate an API key, store it as an environment variable (`GROQ_API_KEY`) — never commit it to the repo.

### 2.2 Local environment

- Python 3.10+
- Docker Desktop (for Redis container + final packaging)
- Git

### 2.3 Python packages

```
fastapi
uvicorn
groq
langchain
langchain-community
faiss-cpu
redis
python-multipart
python-dotenv
pydantic
```

Install with:
```bash
pip install fastapi uvicorn groq langchain langchain-community faiss-cpu redis python-multipart python-dotenv pydantic
```

---

## 3. Build Steps

### Step 1 — Transcription service

Create a function that sends an audio file to Groq's Whisper endpoint and returns raw text.

- Use model `whisper-large-v3-turbo` for speed, or `whisper-large-v3` for max accuracy.
- Input file must be under 25MB — if larger, split the audio into chunks first (e.g. with `pydub`) and transcribe sequentially, then merge.
- Store the raw transcript with a timestamp and a unique conversation ID.

### Step 2 — Chunking and embedding

- Split the transcript into overlapping chunks (e.g. 500 tokens per chunk, 50-token overlap) using LangChain's text splitter.
- Overlap matters — without it, an answer that spans a chunk boundary gets cut and retrieval quality drops. Be ready to explain this choice in an interview.
- Generate embeddings for each chunk (you can use a free HuggingFace sentence-transformer model locally, avoiding any embedding API cost) and store them in a FAISS index, keyed by conversation ID.

### Step 3 — Retrieval + generation (the RAG core)

- When a user asks a question, embed the question the same way, run a similarity search against FAISS for the top-k (e.g. 3-5) most relevant chunks.
- Construct a prompt: system instruction + retrieved chunks as context + the user's question.
- Send that prompt to a Groq-hosted LLM (e.g. `llama-3.3-70b-versatile`) and return the grounded answer.
- Always instruct the model to answer only from the provided context, and to say "not mentioned in this conversation" if the answer isn't there — this is the difference between a real RAG system and a model just hallucinating from general knowledge, and interviewers will probe this.

### Step 4 — Caching layer

- Run Redis locally via Docker: `docker run -p 6379:6379 redis`.
- Cache by a hash of (conversation ID + question) → answer, with a TTL (e.g. 1 hour).
- On each query, check Redis first before hitting the LLM — this is your "latency optimization" story for the interview.

### Step 5 — API layer

Expose these endpoints with FastAPI:

- `POST /upload` — accepts an audio file, runs transcription + chunking + embedding, returns a conversation ID
- `POST /ask` — accepts `{conversation_id, question}`, returns a grounded answer
- `GET /transcript/{conversation_id}` — returns the full raw transcript
- `GET /health` — basic liveness check

### Step 6 — Containerize

Write a `Dockerfile` that installs dependencies and runs `uvicorn main:app --host 0.0.0.0 --port 8000`. Add a `docker-compose.yml` that spins up your API container alongside a Redis container together, so the whole thing runs with one command.

### Step 7 — CI (optional but cheap to add)

A minimal `.github/workflows/ci.yml` that installs dependencies and runs a basic import/lint check on every push. You don't need real tests to get value here — it shows you understand the concept.

### Step 8 — Documentation

Write a clear `README.md`: architecture diagram, setup instructions, example request/response, and a short "Design Decisions" section explaining *why* you chose chunk size, top-k, caching strategy, and Groq specifically. This section is what actually gets read in an interview — invest real time here.

---

## 4. Limitations (be upfront about these — don't let an interviewer catch you first)

- **No speaker diarization** — the transcript doesn't distinguish "agent" vs "customer" speech. Real Darwix AI-style products need this; you're mentioning it as a scoped-out extension, not pretending it's solved.
- **No async ingestion queue (Kafka)** — uploads are processed synchronously in this version. In a production system you'd decouple upload from processing with a queue so the API isn't blocked during transcription. Say this explicitly rather than avoiding the topic.
- **Single-node FAISS, no persistence layer** — the vector index lives in memory/local disk. A production system would use a managed vector DB (Pinecone) or persist FAISS to disk with a proper reload strategy.
- **No multilingual testing** — Whisper supports multiple languages, but you likely only tested English. Say so rather than claiming multilingual support you didn't verify.
- **Groq free tier rate limits** — you may hit request-per-minute limits during heavy testing; this is fine for a demo but worth noting as a real constraint you worked around (e.g. by adding basic retry/backoff logic).
- **No authentication on the API** — fine for a demo project, but flag it as a known gap, since a real deployment would need this immediately.
- **Cost/latency tradeoff not benchmarked at scale** — you tested on a handful of files, not production-scale conversation volumes.

Listing limitations honestly is a strength in an interview, not a weakness — it shows you understand the difference between a demo and a production system, which is exactly the "techno-functional" judgment they're likely screening for.

---

## 5. What to Say About It in the Interview

- Why RAG instead of fine-tuning: cheaper, faster to iterate, answers stay grounded in the specific conversation rather than baked into model weights.
- Why Groq: real-time/low-latency inference matters for an agent-assist use case; same reasoning Darwix AI's own product likely applies.
- Why chunk overlap: prevents answers from being cut at chunk boundaries.
- Why caching: repeated questions in a live call (e.g. "what did the customer say about pricing") shouldn't re-hit the LLM every time.
- What you'd change for production: add Kafka for async ingestion, add diarization, move to a managed vector DB, add auth, add rate-limit handling.

---

## 6. Deployment

For a resume/interview project, prioritize **live, free, and easy to demo** over enterprise-grade infra.

### Option A — Render (recommended for simplicity)

- Connects directly to your GitHub repo and deploys from your `Dockerfile` automatically on every push.
- Free tier includes a web service plus a free managed Redis instance (limited memory, but enough for a demo).
- **Limitation to know:** free-tier services spin down after inactivity, so the first request after idle takes a few seconds to wake up — mention this upfront if demoing live so it doesn't look like a bug.
- Setup: push repo to GitHub → create a new "Web Service" on Render → point it at the repo → Render detects the Dockerfile → add `GROQ_API_KEY` as an environment variable in the Render dashboard → deploy. Add a Redis instance from Render's dashboard and wire the connection URL into the same environment variables.
- Good default choice if you want the least setup time.

### Option B — Railway

- Very similar workflow to Render: push to GitHub, connect the repo, auto-deploys from your Dockerfile.
- Free tier gives a small monthly usage credit rather than an always-free tier — don't leave it running continuously for weeks, only spin it up for demos/interviews.
- Slightly faster cold starts than Render in practice, but the setup experience is nearly identical.

### Option C — Google Cloud Run (recommended if you want to reinforce your GCP experience)

- You already have real Cloud Run experience from your DAaranya.ai internship, so deploying here lets you say *"I deployed this the same way I deploy production services at work"* — a stronger interview line than naming a host you've never used before.
- Genuinely generous free tier (2 million requests/month).
- Deploy directly from your Dockerized app:
  ```bash
  gcloud run deploy pulserag --source . --allow-unauthenticated
  ```
- **Catch:** managed Redis (Cloud Memorystore) isn't free. For the free-tier deployed version, either swap Redis for a simple in-memory cache in production and mention in your README that Redis is used locally via Docker Compose, or self-host Redis in a second container (more setup than it's worth for a demo).

### Recommendation

If you want the fastest path to a live demo link: **Render**. If you want a stronger, more consistent interview story that ties back to your existing resume experience: **Cloud Run**. Either way, skip Kubernetes, AWS, and Azure — they add no value here and burn hours you don't have.

---

## 7. Suggested Repo Structure

```
pulserag/
├── app/
│   ├── main.py              # FastAPI app + routes
│   ├── transcribe.py        # Groq Whisper integration
│   ├── rag.py                # Chunking, embedding, retrieval, generation
│   ├── cache.py              # Redis logic
│   └── config.py             # Env vars / settings
├── tests/
│   └── test_basic.py
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```
