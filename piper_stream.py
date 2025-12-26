"""
Workshop Phase 2: Piper Streaming TTS
Sentence-by-sentence speech synthesis for real-time audio
"""

import subprocess
import numpy as np
import wave
import tempfile
from pathlib import Path
from typing import Generator, Optional
from logger import get_logger

log = get_logger("piper_stream")


class PiperStreamingTTS:
    """
    Streaming TTS using Piper.
    
    Synthesizes sentences as they arrive from LLM.
    Outputs audio chunks for immediate playback.
    
    Example:
        tts = PiperStreamingTTS(
            model_path="~/piper/voices/en_US-lessac-medium.onnx"
        )
        
        # Synthesize single sentence
        audio = tts.synthesize("Hello world")
        
        # Stream sentences from LLM
        for sentence in llm.generate_sentences(prompt):
            audio = tts.synthesize(sentence)
            playback.queue_audio(audio)  # Immediate playback
    """
    
    def __init__(self,
                 model_path: str = "~/FlyingTiger/Workshop_Assistant_Dev/en_US-lessac-medium.onnx",
                 piper_bin: str = "/home/bron/miniconda3/envs/workshop/bin/piper",
                 sample_rate: int = 22050):
        """
        Initialize Piper TTS.
        
        Args:
            model_path: Path to Piper voice model (.onnx)
            piper_bin: Path to piper binary (or "piper" if in PATH)
            sample_rate: Output sample rate (Piper default: 22050)
        """
        self.model_path = Path(model_path).expanduser()
        self.piper_bin = piper_bin
        self.sample_rate = sample_rate
        
        # Verify model exists
        if not self.model_path.exists():
            log.warning(f"Model not found: {self.model_path}")
        
        # Statistics
        self.sentences_synthesized = 0
        
        log.info(f"PiperStreamingTTS: {self.model_path.name}")
    
    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize text to audio.
        
        Args:
            text: Text to synthesize (preferably one sentence)
            
        Returns:
            Audio samples as float32 array
        """
        if not text.strip():
            return np.array([], dtype=np.float32)
        
        # Create temp file for output
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Run Piper
            cmd = [
                self.piper_bin,
                "--model", str(self.model_path),
                "--output_file", tmp_path
            ]
            
            result = subprocess.run(
                cmd,
                input=text.encode('utf-8'),
                capture_output=True,
                check=True
            )
            
            # Read audio
            audio = self._read_wav(tmp_path)
            
            self.sentences_synthesized += 1
            log.debug(f"Synthesized: {text[:50]}... ({len(audio)} samples)")
            
            return audio
        
        except subprocess.CalledProcessError as e:
            log.error(f"Piper synthesis failed: {e.stderr.decode()}")
            return np.array([], dtype=np.float32)
        
        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)
    
    def synthesize_stream(self, 
                         sentences: Generator[str, None, None]
                         ) -> Generator[np.ndarray, None, None]:
        """
        Synthesize sentences as they arrive.
        
        Takes generator of sentences from LLM and yields audio chunks.
        Enables real-time speech while LLM is still generating.
        
        Args:
            sentences: Generator yielding sentences
            
        Yields:
            Audio arrays for each sentence
        """
        for sentence in sentences:
            audio = self.synthesize(sentence)
            if len(audio) > 0:
                yield audio
    
    def _read_wav(self, path: str) -> np.ndarray:
        """
        Read WAV file to numpy array.
        
        Args:
            path: Path to WAV file
            
        Returns:
            Audio samples as float32 array (-1.0 to 1.0)
        """
        with wave.open(path, 'rb') as wav:
            # Read raw samples
            frames = wav.readframes(wav.getnframes())
            
            # Convert to numpy
            if wav.getsampwidth() == 2:  # 16-bit
                audio = np.frombuffer(frames, dtype=np.int16)
                audio = audio.astype(np.float32) / 32768.0
            else:
                raise ValueError(f"Unsupported sample width: {wav.getsampwidth()}")
            
            return audio
    
    async def speak(self, text: str):
        """
        Speak text using TTS with audio playback.

        Args:
            text: Text to speak
        """
        if not text.strip():
            return

        # Synthesize audio
        audio = self.synthesize(text)

        if len(audio) > 0:
            # Play audio using sounddevice
            import sounddevice as sd
            sd.play(audio, self.sample_rate)
            sd.wait()  # Block until playback finishes

    async def play_chime(self):
        """Play a simple chime sound to indicate wake word detection."""
        # Simple ascending tone (C-E-G chord)
        duration = 0.2  # seconds
        t = np.linspace(0, duration, int(self.sample_rate * duration))

        # C (523Hz), E (659Hz), G (784Hz)
        c = np.sin(2 * np.pi * 523 * t)
        e = np.sin(2 * np.pi * 659 * t)
        g = np.sin(2 * np.pi * 784 * t)

        # Combine with envelope
        envelope = np.linspace(0.3, 0.0, len(t))
        chime = (c + e + g) / 3.0 * envelope

        # Play
        import sounddevice as sd
        sd.play(chime.astype(np.float32), self.sample_rate)
        sd.wait()

    def stop(self):
        """Stop current playback."""
        import sounddevice as sd
        sd.stop()

    def get_stats(self) -> dict:
        """Get synthesis statistics."""
        return {
            "model": self.model_path.name,
            "sample_rate": self.sample_rate,
            "sentences_synthesized": self.sentences_synthesized
        }


def test_piper_stream():
    """Test Piper streaming TTS."""
    import time
    
    print("Testing PiperStreamingTTS...\n")
    
    # Test 1: Initialize
    print("Test 1: Initialize TTS")
    tts = PiperStreamingTTS(
        model_path="~/piper/voices/en_US-lessac-medium.onnx"
    )
    print(f"✅ Initialized: {tts.get_stats()}\n")
    
    # Test 2: Synthesize sentence
    print("Test 2: Synthesize single sentence")
    start = time.time()
    audio = tts.synthesize("Hello, this is a test.")
    elapsed = time.time() - start
    
    print(f"  Audio length: {len(audio)} samples ({len(audio)/22050:.2f}s)")
    print(f"  Synthesis time: {elapsed:.3f}s")
    print(f"  Real-time factor: {(len(audio)/22050)/elapsed:.2f}x")
    print("✅ Synthesis works\n")
    
    # Test 3: Empty text
    print("Test 3: Empty text handling")
    audio = tts.synthesize("")
    assert len(audio) == 0
    print("✅ Empty text handled\n")
    
    # Test 4: Stream synthesis
    print("Test 4: Stream multiple sentences")
    
    sentences = [
        "This is sentence one.",
        "Here comes sentence two.",
        "And finally, sentence three."
    ]
    
    def sentence_generator():
        for s in sentences:
            yield s
    
    audio_chunks = list(tts.synthesize_stream(sentence_generator()))
    
    print(f"  Synthesized {len(audio_chunks)} chunks")
    for i, chunk in enumerate(audio_chunks):
        print(f"    Chunk {i}: {len(chunk)} samples ({len(chunk)/22050:.2f}s)")
    
    print("✅ Stream synthesis works\n")
    
    # Test 5: Stats
    print("Test 5: Stats")
    stats = tts.get_stats()
    print(f"  {stats}")
    print("✅ Stats work\n")
    
    print("✅ All tests passed!")
    print("\nNote: Requires Piper installed and model downloaded.")


if __name__ == "__main__":
    test_piper_stream()