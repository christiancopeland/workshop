#!/usr/bin/env python3
"""
Test NVIDIA Canary ASR with live microphone input.

Records 5 seconds of audio and transcribes it.
Usage: python test_canary_mic.py
"""

import numpy as np
import sounddevice as sd
import time
from canary_wrapper import CanaryWrapper

SAMPLE_RATE = 16000
DURATION = 5  # seconds


def record_audio(duration: float = 5.0) -> np.ndarray:
    """Record audio from microphone."""
    print(f"\n🎤 Recording for {duration} seconds... SPEAK NOW!")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()  # Wait for recording to finish
    print("✅ Recording complete!")
    return audio.flatten()


def main():
    print("=" * 50)
    print("NVIDIA Canary ASR Test")
    print("=" * 50)

    # Initialize Canary (will load model)
    print("\n⏳ Loading Canary model...")
    asr = CanaryWrapper()
    asr._load_model()

    while True:
        print("\n" + "-" * 50)
        input("Press ENTER to record (Ctrl+C to quit)...")

        # Record
        audio = record_audio(DURATION)

        # Transcribe
        print("\n🧠 Transcribing with Canary...")
        start = time.time()
        text = asr.transcribe_array(audio, sample_rate=SAMPLE_RATE)
        elapsed = time.time() - start

        print(f"\n📝 Transcription: \"{text}\"")
        print(f"⏱️  Time: {elapsed:.2f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
