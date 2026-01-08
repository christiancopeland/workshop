"""
Workshop Phase 2: Whisper.cpp Python Wrapper
Streaming speech-to-text transcription using whisper.cpp

Phase 5 enhancements:
- Post-processing to clean transcription artifacts
- Corrections dictionary for misheard technical terms
"""

import subprocess
import tempfile
import wave
import re
import numpy as np
from pathlib import Path
from typing import Optional, Generator
from logger import get_logger

log = get_logger("whisper_cpp")


# Technical terms correction dictionary (shared with voice.py)
CORRECTIONS = {
    # Workshop/Claude specific
    "claw to code": "Claude Code",
    "cloud code": "Claude Code",
    "claud code": "Claude Code",
    "claude coat": "Claude Code",
    "work shop": "Workshop",

    # Technical terms
    "g.p.t.": "GPT",
    "l.l.m.": "LLM",
    "a.p.i.": "API",
    "o llama": "Ollama",
    "olama": "Ollama",
    "pie torch": "PyTorch",
    "pi torch": "PyTorch",

    # Hardware/Electronics
    "ina 219": "INA219",
    "i.n.a. 219": "INA219",
    "esp 32": "ESP32",
    "e.s.p. 32": "ESP32",
    "raspberry pi": "Raspberry Pi",
    "ras pi": "Raspberry Pi",
    "arduino": "Arduino",
    "j.s.t.": "JST",

    # Add project-specific terms below
}


def clean_transcription(text: str) -> str:
    """
    Post-process transcription to clean artifacts and fix common errors.

    Args:
        text: Raw transcription text

    Returns:
        Cleaned transcription
    """
    if not text:
        return ""

    # Remove common filler words
    fillers = [
        r'\b(um+)\b',
        r'\b(uh+)\b',
        r'\b(er+)\b',
        r'\b(ah+)\b',
        r'\blike,\s*',
        r'\byou know,\s*',
        r'\bbasically,\s*',
        r'\bso,\s+(?=[a-z])',
        r'\bI mean,\s*',
    ]
    for filler in fillers:
        text = re.sub(filler, '', text, flags=re.IGNORECASE)

    # Apply corrections dictionary
    text_lower = text.lower()
    for wrong, right in CORRECTIONS.items():
        if wrong.lower() in text_lower:
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            text = pattern.sub(right, text)
            text_lower = text.lower()

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Fix common Whisper artifacts
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    text = re.sub(r'([.,!?])([A-Za-z])', r'\1 \2', text)

    # Remove repeated words
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)

    return text.strip()


class WhisperCpp:
    """
    Python wrapper for whisper.cpp with streaming support.
    
    Uses subprocess to call whisper-cli binary.
    Optimized for real-time transcription with chunked audio.
    
    Example:
        whisper = WhisperCpp(
            model_path="~/whisper.cpp/models/ggml-base.en.bin",
            whisper_bin="~/whisper.cpp/build/bin/whisper-cli"
        )
        
        # Transcribe audio file
        text = whisper.transcribe_file("audio.wav")
        
        # Transcribe numpy array
        audio = np.random.randn(16000)  # 1 second @ 16kHz
        text = whisper.transcribe_array(audio)
        
        # Streaming (for Phase 3)
        for partial_text in whisper.transcribe_stream(audio_chunks):
            print(partial_text)
    """
    
    def __init__(self,
                 model_path: str = "~/whisper.cpp/models/ggml-base.en.bin",
                 whisper_bin: str = "~/whisper.cpp/build/bin/whisper-cli",
                 language: str = "en",
                 threads: int = 4):
        """
        Initialize whisper.cpp wrapper.
        
        Args:
            model_path: Path to ggml model file
            whisper_bin: Path to whisper-cli binary
            language: Language code (en, es, fr, etc.)
            threads: Number of CPU threads to use
        """
        self.model_path = Path(model_path).expanduser()
        self.whisper_bin = Path(whisper_bin).expanduser()
        self.language = language
        self.threads = threads
        
        # Verify files exist
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.whisper_bin.exists():
            raise FileNotFoundError(f"Whisper binary not found: {self.whisper_bin}")
        
        log.info(f"WhisperCpp initialized: model={self.model_path.name}, threads={threads}")
    
    def transcribe_file(self, audio_path: str, no_timestamps: bool = True) -> str:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to WAV file (16kHz, mono, 16-bit)
            no_timestamps: If True, return only text without timestamps

        Returns:
            Transcribed text (post-processed)
        """
        cmd = [
            str(self.whisper_bin),
            "-m", str(self.model_path),
            "-f", str(audio_path),
            "-l", self.language,
            "-t", str(self.threads),
            "-nt",  # No timestamps in output
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # Parse output - whisper-cli prints transcription to stdout
            # Format: "[00:00:00.000 --> 00:00:11.000]   Text here"
            # With -nt flag, it's just "Text here"
            text = result.stdout.strip()

            # Remove any remaining timestamp markers
            lines = [line for line in text.split('\n') if line.strip() and not line.startswith('[')]
            raw_text = ' '.join(lines).strip()

            # Apply post-processing (Phase 5)
            text = clean_transcription(raw_text)

            if raw_text != text:
                log.debug(f"Cleaned: '{raw_text[:50]}...' → '{text[:50]}...'")

            log.debug(f"Transcribed: {text[:50]}...")
            return text

        except subprocess.CalledProcessError as e:
            log.error(f"Whisper transcription failed: {e.stderr}")
            return ""
    
    def transcribe_array(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe numpy audio array.
        
        Args:
            audio: Audio samples as float32 array
            sample_rate: Sample rate (must be 16000 for Whisper)
            
        Returns:
            Transcribed text
        """
        if sample_rate != 16000:
            raise ValueError("Whisper requires 16kHz audio")
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Write audio to WAV file
            self._write_wav(tmp_path, audio, sample_rate)
            
            # Transcribe
            text = self.transcribe_file(tmp_path)
            
            return text
            
        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)
    
    def transcribe_chunks(self, 
                         audio_chunks: list[np.ndarray],
                         sample_rate: int = 16000) -> str:
        """
        Transcribe multiple audio chunks as single utterance.
        
        Concatenates chunks and transcribes as one piece.
        Use this for complete utterances captured in chunks.
        
        Args:
            audio_chunks: List of audio arrays
            sample_rate: Sample rate
            
        Returns:
            Transcribed text
        """
        # Concatenate all chunks
        audio = np.concatenate(audio_chunks)
        
        # Transcribe combined audio
        return self.transcribe_array(audio, sample_rate)
    
    def _write_wav(self, path: str, audio: np.ndarray, sample_rate: int):
        """
        Write numpy array to WAV file.
        
        Args:
            path: Output file path
            audio: Audio samples (float32, -1.0 to 1.0)
            sample_rate: Sample rate
        """
        # Convert float32 to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Write WAV
        with wave.open(path, 'wb') as wav:
            wav.setnchannels(1)  # Mono
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(sample_rate)
            wav.writeframes(audio_int16.tobytes())
    
    def get_stats(self) -> dict:
        """Get wrapper statistics."""
        return {
            "model": self.model_path.name,
            "binary": str(self.whisper_bin),
            "language": self.language,
            "threads": self.threads,
            "model_exists": self.model_path.exists(),
            "binary_exists": self.whisper_bin.exists()
        }


# Streaming transcription (for Phase 3)
class StreamingWhisper:
    """
    Streaming transcription using overlapping windows.
    
    Accumulates audio in buffer, transcribes when buffer is full,
    uses overlapping windows to avoid cutting words.
    
    This is for Phase 3 (STT-002: Chunk-based transcription).
    """
    
    def __init__(self,
                 whisper: WhisperCpp,
                 window_size: float = 3.0,
                 overlap: float = 0.5,
                 sample_rate: int = 16000):
        """
        Initialize streaming transcriber.
        
        Args:
            whisper: WhisperCpp instance
            window_size: Seconds of audio per transcription
            overlap: Seconds of overlap between windows
            sample_rate: Audio sample rate
        """
        self.whisper = whisper
        self.window_size = window_size
        self.overlap = overlap
        self.sample_rate = sample_rate
        
        # Buffer for accumulating audio
        self.buffer = np.array([], dtype=np.float32)
        
        # Sizes in samples
        self.window_samples = int(window_size * sample_rate)
        self.overlap_samples = int(overlap * sample_rate)
        
        log.info(f"StreamingWhisper: {window_size}s window, {overlap}s overlap")
    
    def add_audio(self, audio: np.ndarray) -> Optional[str]:
        """
        Add audio chunk to buffer and transcribe if ready.
        
        Args:
            audio: Audio samples to add
            
        Returns:
            Transcribed text if window is full, None otherwise
        """
        # Add to buffer
        self.buffer = np.append(self.buffer, audio)
        
        # Check if we have enough for a window
        if len(self.buffer) >= self.window_samples:
            # Extract window
            window = self.buffer[:self.window_samples]
            
            # Transcribe
            text = self.whisper.transcribe_array(window, self.sample_rate)
            
            # Keep overlap for next window
            self.buffer = self.buffer[self.window_samples - self.overlap_samples:]
            
            return text
        
        return None
    
    def flush(self) -> str:
        """
        Transcribe remaining audio in buffer.
        
        Returns:
            Transcribed text of remaining audio
        """
        if len(self.buffer) > 0:
            text = self.whisper.transcribe_array(self.buffer, self.sample_rate)
            self.buffer = np.array([], dtype=np.float32)
            return text
        return ""
    
    def reset(self):
        """Clear buffer."""
        self.buffer = np.array([], dtype=np.float32)


class FasterWhisperWrapper:
    """
    Faster-whisper wrapper with WhisperCpp-compatible interface.

    Uses faster-whisper (CTranslate2) instead of whisper.cpp.
    Provides the same interface as WhisperCpp for drop-in replacement.
    Includes Phase 5 enhancements: auto model selection, post-processing.

    Example:
        whisper = FasterWhisperWrapper()  # Auto-selects model based on GPU
        text = whisper.transcribe_array(audio)
    """

    def __init__(self, model_name: str = "auto", device: str = "auto"):
        """
        Initialize faster-whisper wrapper.

        Args:
            model_name: Model name or "auto" for GPU-based selection
            device: "cuda", "cpu", or "auto"
        """
        self.model_name = model_name
        self._model = None
        self._device = device

        # Auto-select model based on GPU memory
        if model_name == "auto":
            self.model_name = self._select_optimal_model()

        log.info(f"FasterWhisperWrapper: model={self.model_name}, device={device}")

    def _select_optimal_model(self) -> str:
        """Select the best Whisper model that fits in GPU memory."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                available = gpu_mem - 3.0  # Reserve 3GB headroom

                log.info(f"GPU memory: {gpu_mem:.1f}GB, available for Whisper: {available:.1f}GB")

                if available >= 5.0:
                    model = "large-v3-turbo"  # Newer, better for conversational speech
                elif available >= 3.0:
                    model = "medium.en"
                elif available >= 1.0:
                    model = "small.en"
                else:
                    model = "base.en"

                log.info(f"Auto-selected Whisper model: {model}")
                return model
            else:
                log.info("No CUDA available, using small.en for CPU")
                return "small.en"
        except Exception as e:
            log.warning(f"GPU detection failed: {e}, defaulting to base.en")
            return "base.en"

    def _load_model(self):
        """Lazy-load the model."""
        if self._model is None:
            from faster_whisper import WhisperModel

            log.info(f"Loading faster-whisper model: {self.model_name}")

            # Determine device
            device = self._device
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            compute_type = "float16" if device == "cuda" else "int8"

            try:
                self._model = WhisperModel(
                    self.model_name,
                    device=device,
                    compute_type=compute_type
                )
                self._device = device
                log.info(f"Model loaded: {self.model_name} on {device} ({compute_type})")
            except Exception as e:
                log.error(f"Failed to load model: {e}")
                raise

    def transcribe_array(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe numpy audio array.

        Args:
            audio: Audio samples as float32 array
            sample_rate: Sample rate (must be 16000 for Whisper)

        Returns:
            Transcribed text (post-processed)
        """
        if sample_rate != 16000:
            raise ValueError("Whisper requires 16kHz audio")

        self._load_model()

        try:
            segments, info = self._model.transcribe(
                audio,
                language="en",
                task="transcribe",
                beam_size=5,
                best_of=5,
                patience=1.0,
                temperature=0.0,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 300,
                    "speech_pad_ms": 200
                },
                condition_on_previous_text=False,  # Prevents hallucination loops
                no_speech_threshold=0.6,
                hallucination_silence_threshold=0.5,  # Detect silence hallucinations
                repetition_penalty=1.1,  # Penalize repeated phrases
            )

            raw_text = " ".join(seg.text for seg in segments).strip()

            # Hallucination detection: if output is suspiciously long for input duration
            duration_s = len(audio) / sample_rate
            words = len(raw_text.split()) if raw_text else 0
            words_per_second = words / duration_s if duration_s > 0 else 0

            # Normal speech is ~2-3 words/sec. If >5 w/s on short audio, likely hallucination
            if duration_s < 2.0 and words_per_second > 5.0:
                log.warning(f"Possible hallucination: {words} words in {duration_s:.1f}s ({words_per_second:.1f} w/s)")
                # Return empty to trigger retry or rejection
                return ""

            # Apply post-processing
            text = clean_transcription(raw_text)

            if raw_text != text:
                log.debug(f"Cleaned: '{raw_text[:50]}...' → '{text[:50]}...'")

            log.debug(f"Transcribed: {text[:50] if text else '(empty)'}...")
            return text

        except Exception as e:
            log.error(f"Transcription error: {e}")
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
            "backend": "faster-whisper",
            "loaded": self._model is not None
        }


# Alias for backwards compatibility - use FasterWhisperWrapper if WhisperCpp fails
def WhisperWrapper(model_name: str = "auto"):
    """
    Factory function that returns the best available whisper wrapper.

    Tries WhisperCpp first, falls back to FasterWhisperWrapper.
    """
    try:
        return WhisperCpp()
    except FileNotFoundError:
        log.info("whisper.cpp not found, using faster-whisper")
        return FasterWhisperWrapper(model_name=model_name)


def test_whisper_cpp():
    """Test whisper.cpp wrapper."""
    import time

    print("Testing WhisperCpp wrapper...\n")
    
    # Initialize
    print("Test 1: Initialize wrapper")
    whisper = WhisperCpp(
        model_path="~/whisper.cpp/models/ggml-base.en.bin",
        whisper_bin="~/whisper.cpp/build/bin/whisper-cli"
    )
    print(f"✅ Initialized: {whisper.get_stats()}\n")
    
    # Test 2: Transcribe sample file
    print("Test 2: Transcribe sample file (jfk.wav)")
    start = time.time()
    text = whisper.transcribe_file("~/whisper.cpp/samples/jfk.wav")
    elapsed = time.time() - start
    
    print(f"  Text: {text}")
    print(f"  Time: {elapsed:.2f}s")
    assert "ask not what your country" in text.lower()
    print("✅ File transcription works\n")
    
    # Test 3: Transcribe numpy array
    print("Test 3: Transcribe numpy array (1 second of silence)")
    silence = np.zeros(16000, dtype=np.float32)
    text = whisper.transcribe_array(silence)
    print(f"  Text: '{text}'")
    print("✅ Array transcription works\n")
    
    # Test 4: Streaming mode
    print("Test 4: Streaming transcription")
    streamer = StreamingWhisper(whisper, window_size=3.0, overlap=0.5)
    
    # Simulate adding audio chunks
    chunk_size = 8000  # 0.5s chunks
    total_chunks = 8   # 4 seconds total
    
    for i in range(total_chunks):
        chunk = np.random.randn(chunk_size).astype(np.float32) * 0.1
        result = streamer.add_audio(chunk)
        if result:
            print(f"  Window {i}: '{result}'")
    
    # Flush remaining
    final = streamer.flush()
    print(f"  Final: '{final}'")
    print("✅ Streaming works\n")
    
    print("✅ All tests passed!")


if __name__ == "__main__":
    test_whisper_cpp()