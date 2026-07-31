"""Generate a tiny test .wav so you can try ``POST /upload`` without a microphone.

Usage:  uv run python scripts/make_sample_audio.py   → writes sample_audio.wav
"""

import math
import struct
import wave
from pathlib import Path

SAMPLE_RATE = 16_000
DURATION_SECONDS = 2.0
FREQUENCY_HZ = 440.0
AMPLITUDE = 0.4
OUTPUT = Path(__file__).resolve().parent.parent / "sample_audio.wav"


def main() -> None:
    n_frames = int(SAMPLE_RATE * DURATION_SECONDS)
    frames = bytearray()
    for i in range(n_frames):
        sample = AMPLITUDE * math.sin(2 * math.pi * FREQUENCY_HZ * i / SAMPLE_RATE)
        frames += struct.pack("<h", int(sample * 32767))

    with wave.open(str(OUTPUT), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(bytes(frames))

    size = OUTPUT.stat().st_size
    print(f"Wrote {OUTPUT} ({size} bytes, {DURATION_SECONDS}s 440Hz mono).")
    print("Next: open http://localhost:8000/docs and POST it to /upload.")


if __name__ == "__main__":
    main()
