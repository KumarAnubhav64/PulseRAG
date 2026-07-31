"""Speech-to-text via Groq Whisper, with a mock fallback for demo mode."""

import io
import os

from groq import Groq

from ..config import Settings

# Canned transcript used in demo mode so the *whole* RAG flow is exercisable
# without a Groq key. It is written to be grounded: every claim appears
# verbatim in the text, so retrieval questions like "what did the customer say
# about pricing?" retrieve the right chunk and the mock answer can cite it.
_DEMO_TRANSCRIPT = (
    "Welcome to the PulseRAG demo transcript. The customer, Sarah, said the "
    "current price of $49 per month was too high compared to competitors who "
    "charge around $35. She asked if there was an annual plan discount and "
    "seemed happy when we mentioned 20% off for yearly billing. She said "
    "customer support quality matters most to her, and that she would switch "
    "if onboarding took more than a week. She also asked about a free trial, "
    "which we confirmed is available for 14 days."
)


def _open_audio_source(source: bytes | str | os.PathLike):
    """Return a file-like object for ``source``.

    ``source`` is either raw audio bytes (legacy callers) or a path to an
    on-disk audio file (streamed uploads). Passing a path avoids loading the
    whole file into RAM before the Groq request.
    """
    if isinstance(source, (str, os.PathLike)):
        return open(source, "rb")
    return io.BytesIO(source)


class TranscriptionService:
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

    def transcribe(self, source: bytes | str, filename: str) -> tuple[str, bool]:
        """Return ``(transcript, used_demo)``.

        ``source`` may be audio bytes or a path to an audio file on disk.
        """
        if self._demo:
            return _DEMO_TRANSCRIPT, True

        with _open_audio_source(source) as file_obj:
            # Groq's client expects a file-like object; the tuple form carries the
            # filename so the extension can drive content-type detection.
            result = self._get_client().audio.transcriptions.create(
                file=(filename or "audio.wav", file_obj),
                model=self._settings.groq_stt_model,
            )
        return result.text, False
