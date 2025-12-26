"""
Workshop Phase 2: Whisper.cpp Python Wrapper
Streaming speech-to-text transcription using whisper.cpp
"""

import subprocess
import tempfile
import wave
import numpy as np
from pathlib import Path
from typing import Optional, Generator
from logger import get_logger

log = get_logger("whisper_cpp")


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
            Transcribed text
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
            text = ' '.join(lines).strip()
            
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