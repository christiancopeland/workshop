"""
NVIDIA Canary ASR Wrapper

High-quality local speech recognition using NVIDIA's Canary-1B model.
Provides same interface as FasterWhisperWrapper for drop-in replacement.

Installation:
    pip install nemo_toolkit[asr]

Usage:
    from canary_wrapper import CanaryWrapper
    asr = CanaryWrapper()
    text = asr.transcribe_array(audio_array, sample_rate=16000)
"""

import numpy as np
from typing import Optional
from logger import get_logger

log = get_logger("canary_asr")


class CanaryWrapper:
    """
    NVIDIA Canary-1B ASR wrapper with WhisperWrapper-compatible interface.

    Canary is a state-of-the-art multilingual ASR model that often
    outperforms Whisper on conversational speech.

    Model: nvidia/canary-1b (~1.5GB)
    VRAM: ~3-4GB for inference
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize Canary ASR.

        Args:
            device: "cuda" or "cpu"
        """
        self._model = None
        self._device = device
        self.model_name = "canary-1b"

        log.info(f"CanaryWrapper initialized (device={device})")

    def _load_model(self):
        """Lazy-load the Canary model."""
        if self._model is not None:
            return

        try:
            # Use EncDecMultiTaskModel for Canary (not generic ASRModel)
            from nemo.collections.asr.models import EncDecMultiTaskModel

            log.info("Loading NVIDIA Canary-1B model...")
            print("     Loading NVIDIA Canary-1B (first run downloads ~1.5GB)...")

            # Load the Canary model - MUST use EncDecMultiTaskModel
            self._model = EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b")

            # Move to device
            if self._device == "cuda":
                self._model = self._model.cuda()

            # Configure decoding strategy
            decode_cfg = self._model.cfg.decoding
            decode_cfg.beam.beam_size = 1
            self._model.change_decoding_strategy(decode_cfg)

            # Set to eval mode
            self._model.eval()

            log.info(f"Canary-1B loaded on {self._device}")
            print(f"     ✓ Canary-1B ready on {self._device}")

        except ImportError as e:
            log.error("NeMo toolkit not installed. Run: pip install nemo_toolkit[asr]")
            raise ImportError(
                "NeMo toolkit required for Canary ASR. "
                "Install with: pip install nemo_toolkit[asr]"
            ) from e
        except Exception as e:
            log.error(f"Failed to load Canary model: {e}")
            raise

    def transcribe_array(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe numpy audio array.

        Args:
            audio: Audio samples as float32 array
            sample_rate: Sample rate (16000 recommended)

        Returns:
            Transcribed text
        """
        self._load_model()

        # Ensure correct dtype
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize if needed (Canary expects -1 to 1 range)
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()

        try:
            import torch
            import tempfile
            import soundfile as sf
            import os
            import json

            # Create temp directory for audio and manifest
            tmp_dir = tempfile.mkdtemp()
            audio_path = os.path.join(tmp_dir, "audio.wav")
            manifest_path = os.path.join(tmp_dir, "manifest.json")

            try:
                # Write audio to temp file
                sf.write(audio_path, audio, sample_rate)

                # Calculate duration
                duration = len(audio) / sample_rate

                # Create manifest file (required by Canary's AggregateTokenizer)
                manifest_entry = {
                    "audio_filepath": audio_path,
                    "duration": duration,
                    "taskname": "asr",
                    "source_lang": "en",
                    "target_lang": "en",
                    "pnc": "yes",
                    "answer": "na"
                }

                with open(manifest_path, 'w') as f:
                    f.write(json.dumps(manifest_entry) + '\n')

                # Transcribe using manifest
                with torch.no_grad():
                    hypotheses = self._model.transcribe(
                        manifest_path,
                        batch_size=1,
                    )

                if hypotheses and len(hypotheses) > 0:
                    # Extract text from Hypothesis object
                    hyp = hypotheses[0]

                    # Try multiple ways to get the text
                    text = ""
                    if hasattr(hyp, 'text') and hyp.text:
                        text = hyp.text.strip()
                    elif isinstance(hyp, str):
                        text = hyp.strip()
                    else:
                        # Last resort: string conversion
                        text = str(hyp).strip()

                    log.debug(f"Canary transcribed: {text[:50] if text else '(empty)'}...")
                    return text
                else:
                    return ""

            finally:
                # Clean up temp files
                import shutil
                if os.path.exists(tmp_dir):
                    shutil.rmtree(tmp_dir)

        except Exception as e:
            log.error(f"Canary transcription error: {e}")
            return ""

    def transcribe_chunks(self, audio_chunks: list, sample_rate: int = 16000) -> str:
        """Transcribe multiple audio chunks as single utterance."""
        audio = np.concatenate(audio_chunks)
        return self.transcribe_array(audio, sample_rate)

    def get_stats(self) -> dict:
        """Get wrapper statistics."""
        return {
            "model": self.model_name,
            "device": self._device,
            "backend": "nvidia-nemo",
            "loaded": self._model is not None
        }


def test_canary():
    """Test Canary ASR with a simple audio sample."""
    print("Testing NVIDIA Canary ASR...")

    # Create 1 second of silence (should return empty or minimal text)
    silence = np.zeros(16000, dtype=np.float32)

    try:
        asr = CanaryWrapper()
        text = asr.transcribe_array(silence)
        print(f"Transcription of silence: '{text}'")
        print("✓ Canary ASR working!")
    except ImportError as e:
        print(f"✗ {e}")
        print("\nInstall NeMo with: pip install nemo_toolkit[asr]")
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    test_canary()
