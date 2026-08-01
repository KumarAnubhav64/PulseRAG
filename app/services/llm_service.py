"""Grounded answer generation via Groq (raw SDK client), with a mock fallback.

Uses the same ``groq`` package as the transcription service instead of
``langchain-groq`` — one less heavy import in the boot path.
"""

from groq import Groq

from ..config import Settings

_SYSTEM_PROMPT = (
    "You are a helpful assistant for a call-transcript Q&A system. "
    "Answer the user's question based on the provided transcript excerpts. "
    "Be generous when the excerpts are relevant: even if they only mention a "
    "topic or contain partial information, summarize what the transcript says "
    "about the question and quote the supporting excerpt (e.g. 'The transcript "
    "mentions...'). Do not add facts that are not present in the excerpts. "
    "Only if the excerpts contain nothing at all related to the question, "
    "reply exactly: \"That information is not mentioned in the transcript.\" "
    "Do not use outside knowledge."
)


class LLMService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._demo = settings.demo_enabled
        self._client: Groq | None = None

    def _get_client(self) -> Groq:
        if self._client is None:
            self._client = Groq(
                api_key=self._settings.groq_api_key,
                timeout=self._settings.groq_request_timeout_seconds,
            )
        return self._client

    def generate(self, question: str, context_chunks: list[str]) -> tuple[str, bool]:
        """Return ``(answer, used_demo)``."""
        if self._demo:
            return self._demo_answer(question, context_chunks), True

        context = "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(context_chunks))
        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"TRANSCRIPT EXCERPTS:\n{context or '(no excerpts available)'}\n\n"
            f"QUESTION: {question}\n\nANSWER:"
        )
        response = self._get_client().chat.completions.create(
            model=self._settings.groq_llm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return str(response.choices[0].message.content), False

    @staticmethod
    def _demo_answer(question: str, context_chunks: list[str]) -> str:
        """Extractive mock answer — retrieves the top chunk and cites it."""
        if not context_chunks:
            return (
                "Demo mode: no GROQ_API_KEY configured, and no transcript is "
                "indexed for this conversation yet. Upload an audio file first, "
                "then ask again."
            )

        best = context_chunks[0]
        question_lower = question.lower()
        if "price" in question_lower or "cost" in question_lower or "discount" in question_lower:
            snippet = (
                "The customer said the current price of $49 per month was too "
                "high compared to competitors at ~$35, and asked about an "
                "annual-plan discount (20% off for yearly billing was mentioned)."
            )
        else:
            snippet = "The most relevant part of the transcript is quoted below."

        return (
            "Demo mode: no GROQ_API_KEY configured, so this is a mock answer "
            "(set DEMO_MODE=off and add a real key for grounded LLM answers).\n\n"
            f"{snippet}\n\n"
            f'Source excerpt: "{best[:300]}"'
        )
