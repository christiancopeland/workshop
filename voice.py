"""
Workshop Voice Interface
Handles speech-to-text (Whisper) and text-to-speech (Piper)
"""

import asyncio
import numpy as np
import queue
import threading
import sys
from pathlib import Path
from typing import Optional
import wave
import io
import tempfile
import subprocess

from logger import get_logger

log = get_logger("voice")


class VoiceInterface:
    """Voice input/output interface using Whisper and Piper"""

    # Model VRAM estimates (GB) for selection
    MODEL_VRAM = {
        "tiny.en": 0.5, "tiny": 0.5,
        "base.en": 0.5, "base": 0.5,
        "small.en": 1.0, "small": 1.0,
        "medium.en": 3.0, "medium": 3.0,
        "large-v2": 5.0, "large-v3": 5.0, "large": 5.0,
    }

    # Technical terms correction dictionary
    # Add your commonly misheard terms here
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

        # Add your project-specific terms below
    }

    def __init__(
        self,
        whisper_model: str = "auto",
        piper_model: str = "en_US-lessac-medium",
        wake_word: str = "workshop",
        sample_rate: int = 16000,
        vad_confidence_threshold: float = 0.6
    ):
        """
        Initialize voice interface with auto model selection.

        Args:
            whisper_model: Model name or "auto" for GPU-based selection
            piper_model: Piper TTS model name
            wake_word: Wake word for activation
            sample_rate: Audio sample rate (16kHz for Whisper)
            vad_confidence_threshold: Minimum VAD confidence to transcribe (0.0-1.0)
        """
        self.piper_model = piper_model
        self.wake_word = wake_word.lower()
        self.sample_rate = sample_rate
        self.vad_confidence_threshold = vad_confidence_threshold

        # Auto-select model based on GPU memory
        if whisper_model == "auto":
            self.whisper_model_name = self._select_optimal_model()
        else:
            self.whisper_model_name = whisper_model

        self._whisper_model = None
        self._audio_queue = queue.Queue()
        self._is_listening = False

        # Try to import audio dependencies
        self._sounddevice = None
        self._has_audio = self._init_audio()

        log.info(f"VoiceInterface initialized: model={self.whisper_model_name}, wake_word={wake_word}, vad_threshold={vad_confidence_threshold}")

    def _select_optimal_model(self) -> str:
        """Select the best Whisper model that fits in GPU memory."""
        try:
            import torch
            if torch.cuda.is_available():
                # Get available GPU memory in GB
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                # Reserve 3GB for other models/overhead
                available = gpu_mem - 3.0

                log.info(f"GPU memory: {gpu_mem:.1f}GB, available for Whisper: {available:.1f}GB")

                # Select largest model that fits
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
    
    def _init_audio(self) -> bool:
        """Initialize audio dependencies"""
        try:
            import sounddevice as sd
            self._sounddevice = sd
            
            # Test that we can actually access a microphone
            devices = sd.query_devices()
            default_input = sd.query_devices(kind='input')
            log.info(f"Default input device: {default_input['name']}")
            
            return True
        except ImportError:
            print("⚠️  sounddevice not installed. Voice features disabled.")
            print("   Install with: pip install sounddevice")
            return False
        except Exception as e:
            print(f"⚠️  Audio initialization failed: {e}")
            log.error(f"Audio init failed: {e}")
            return False
    
    def _load_whisper(self):
        """Lazy-load Whisper model with optimal settings."""
        if self._whisper_model is None:
            try:
                # Try faster-whisper first (recommended)
                from faster_whisper import WhisperModel
                print(f"  → Loading Whisper model ({self.whisper_model_name})...")
                log.info(f"Loading faster-whisper model: {self.whisper_model_name}")

                # Try CUDA first, fall back to CPU
                try:
                    self._whisper_model = WhisperModel(
                        self.whisper_model_name,
                        device="cuda",
                        compute_type="float16"
                    )
                    self._device = "cuda"
                    log.info(f"Whisper {self.whisper_model_name} loaded (faster-whisper, CUDA float16)")
                    print(f"  ✓ Loaded {self.whisper_model_name} on CUDA")
                except Exception as cuda_err:
                    log.warning(f"CUDA failed: {cuda_err}, falling back to CPU")
                    print(f"  → CUDA unavailable, using CPU with int8...")
                    self._whisper_model = WhisperModel(
                        self.whisper_model_name,
                        device="cpu",
                        compute_type="int8"
                    )
                    self._device = "cpu"
                    log.info(f"Whisper {self.whisper_model_name} loaded (faster-whisper, CPU int8)")

                self._whisper_type = "faster"

            except ImportError:
                log.warning("faster-whisper not found, trying openai-whisper")
                # Fallback to openai-whisper
                try:
                    import whisper
                    print(f"  → Loading Whisper model ({self.whisper_model_name})...")
                    self._whisper_model = whisper.load_model(self.whisper_model_name)
                    self._whisper_type = "openai"
                    self._device = "cuda" if next(self._whisper_model.parameters()).is_cuda else "cpu"
                    log.info("Whisper loaded successfully (openai-whisper)")
                except ImportError:
                    print("⚠️  No Whisper library found.")
                    print("   Install with: pip install faster-whisper")
                    log.error("No Whisper library available")
                    self._whisper_model = None
                    self._whisper_type = None
            except Exception as e:
                log.error(f"Whisper load failed: {e}")
                print(f"⚠️  Whisper load failed: {e}")
                self._whisper_model = None
                self._whisper_type = None

        return self._whisper_model is not None

    def _clean_transcription(self, text: str) -> str:
        """
        Post-process transcription to clean artifacts and fix common errors.

        Args:
            text: Raw transcription text

        Returns:
            Cleaned transcription
        """
        import re

        if not text:
            return ""

        # Remove common filler words (but preserve meaning)
        fillers = [
            r'\b(um+)\b',
            r'\b(uh+)\b',
            r'\b(er+)\b',
            r'\b(ah+)\b',
            r'\blike,\s*',
            r'\byou know,\s*',
            r'\bbasically,\s*',
            r'\bso,\s+(?=[a-z])',  # "so," at start of clauses
            r'\bI mean,\s*',
        ]
        for filler in fillers:
            text = re.sub(filler, '', text, flags=re.IGNORECASE)

        # Apply corrections dictionary (case-insensitive matching)
        text_lower = text.lower()
        for wrong, right in self.CORRECTIONS.items():
            if wrong.lower() in text_lower:
                # Use case-insensitive replacement
                pattern = re.compile(re.escape(wrong), re.IGNORECASE)
                text = pattern.sub(right, text)
                text_lower = text.lower()

        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Fix common Whisper artifacts
        text = re.sub(r'\s+([.,!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,!?])([A-Za-z])', r'\1 \2', text)  # Add space after punctuation

        # Remove repeated words (Whisper sometimes stutters)
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)

        return text.strip()
    
    async def wait_for_activation(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        Wait for wake word or goodbye command.
        
        Returns:
            "wake" - wake word detected
            "goodbye" - goodbye command detected
            None - nothing detected (silence or unrecognized speech)
        """
        if not self._has_audio:
            return "wake"  # In text mode, just return wake
        
        if not self._load_whisper():
            log.error("Cannot wait for activation - Whisper not loaded")
            return None
        
        # Visual feedback - show we're listening
        sys.stdout.write(".")
        sys.stdout.flush()
        
        # Record a short clip and check for wake word
        log.debug("Recording for wake word detection...")
        audio = await self._record_audio(duration=2.0)
        
        if audio is None:
            log.warning("No audio captured")
            return None
        
        # Check amplitude - skip transcription if too quiet
        amplitude = np.abs(audio).mean()
        if amplitude < 0.005:  # Very quiet, probably silence
            log.debug(f"Audio too quiet ({amplitude:.6f}), skipping transcription")
            return None
        
        log.debug(f"Recorded {len(audio)} samples, transcribing...")
        text = await self._transcribe(audio)
        
        if not text:
            return None
        
        text_lower = text.lower()
        log.debug(f"Heard: '{text}'")
        
        # Check for goodbye first
        if any(word in text_lower for word in ["goodbye", "bye bye", "bye-bye", "good bye"]):
            log.info(f"Goodbye detected in: '{text}'")
            print(f"\n👋 Goodbye detected!")
            return "goodbye"
        
        # Check for wake word
        if self.wake_word in text_lower:
            log.info(f"Wake word detected in: '{text}'")
            print(f"\n✨ Wake word detected!")
            return "wake"
        
        return None
    
    async def listen(
        self,
        max_duration: float = 30.0,
        silence_duration: float = 1.5,
        silence_threshold: float = 0.01
    ) -> Optional[str]:
        """
        Listen for speech and transcribe.
        
        Stops when silence is detected or max duration reached.
        Returns transcribed text or None.
        """
        if not self._has_audio or not self._load_whisper():
            log.warning("Cannot listen - audio or Whisper not available")
            return None
        
        log.info("Listening for speech...")
        audio = await self._record_until_silence(
            max_duration=max_duration,
            silence_duration=silence_duration,
            silence_threshold=silence_threshold
        )
        
        if audio is None or len(audio) < self.sample_rate * 0.5:  # Less than 0.5s
            log.debug("Audio too short or empty")
            return None
        
        # Check if audio is mostly silence (avoid Whisper hallucinations)
        amplitude = np.abs(audio).mean()
        if amplitude < 0.008:
            log.debug(f"Audio too quiet ({amplitude:.6f}), likely silence")
            return None
        
        print("🧠 Processing...")
        result = await self._transcribe(audio)
        log.info(f"Listen result: '{result}'")
        return result
    
    async def _record_audio(self, duration: float) -> Optional[np.ndarray]:
        """Record audio for a fixed duration"""
        if not self._has_audio:
            return None
        
        try:
            log.debug(f"Recording {duration}s of audio...")
            loop = asyncio.get_event_loop()
            audio = await loop.run_in_executor(
                None,
                lambda: self._sounddevice.rec(
                    int(duration * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=2,
                    dtype=np.float32
                )
            )
            self._sounddevice.wait()
            audio = audio.flatten()
            
            amplitude = np.abs(audio).mean()
            log.debug(f"Recorded {len(audio)} samples, mean amplitude: {amplitude:.6f}")
            
            return audio
        except Exception as e:
            log.error(f"Recording error: {e}")
            print(f"Recording error: {e}")
            return None
    
    async def _record_until_silence(
        self,
        max_duration: float,
        silence_duration: float,
        silence_threshold: float
    ) -> Optional[np.ndarray]:
        """Record audio until silence is detected"""
        if not self._has_audio:
            return None
        
        chunk_samples = int(0.1 * self.sample_rate)  # 100ms chunks
        max_samples = int(max_duration * self.sample_rate)
        silence_samples = int(silence_duration * self.sample_rate)
        
        audio_chunks = []
        silent_samples = 0
        total_samples = 0
        
        log.debug(f"Recording until silence (max {max_duration}s, silence threshold {silence_threshold})...")
        print("🎤 Listening... (speak now)")
        
        try:
            with self._sounddevice.InputStream(
                samplerate=self.sample_rate,
                channels=2,
                dtype=np.float32
            ) as stream:
                while total_samples < max_samples:
                    chunk, _ = stream.read(chunk_samples)
                    chunk = chunk.flatten()
                    audio_chunks.append(chunk)
                    total_samples += len(chunk)
                    
                    # Check for silence
                    amplitude = np.abs(chunk).mean()
                    if amplitude < silence_threshold:
                        silent_samples += len(chunk)
                        if silent_samples >= silence_samples:
                            log.debug("Silence detected, stopping recording")
                            break
                    else:
                        silent_samples = 0
                    
                    # Yield control
                    await asyncio.sleep(0)
            
            if audio_chunks:
                audio = np.concatenate(audio_chunks)
                log.debug(f"Recorded {len(audio)} total samples")
                return audio
            return None
            
        except Exception as e:
            log.error(f"Recording error: {e}")
            print(f"Recording error: {e}")
            return None
    
    async def _transcribe(self, audio: np.ndarray) -> Optional[str]:
        """
        Transcribe audio to text using Whisper with anti-hallucination settings.

        Uses beam search and multiple candidates, but avoids initial_prompt
        and condition_on_previous_text which can cause hallucinations.
        """
        if self._whisper_model is None:
            log.error("Transcribe called but Whisper not loaded")
            return None

        try:
            loop = asyncio.get_event_loop()
            duration = len(audio) / self.sample_rate
            log.debug(f"Transcribing {len(audio)} samples ({duration:.2f}s)...")

            if self._whisper_type == "faster":
                # faster-whisper with anti-hallucination settings
                def do_transcribe():
                    segments, info = self._whisper_model.transcribe(
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
                        hallucination_silence_threshold=0.5,
                        repetition_penalty=1.1,
                    )
                    return list(segments), info

                segments, info = await loop.run_in_executor(None, do_transcribe)

                # Log transcription quality info
                if segments:
                    avg_logprob = sum(s.avg_logprob for s in segments) / len(segments)
                    log.debug(f"Transcription info: lang={info.language}, prob={info.language_probability:.2f}, avg_logprob={avg_logprob:.2f}")

                text = " ".join(seg.text for seg in segments).strip()
            else:
                # openai-whisper
                def do_transcribe():
                    return self._whisper_model.transcribe(
                        audio,
                        language="en",
                        task="transcribe",
                        beam_size=5,
                        best_of=5,
                        temperature=0,
                        condition_on_previous_text=False,
                    )

                result = await loop.run_in_executor(None, do_transcribe)
                text = result["text"].strip()

            # Hallucination detection: reject suspiciously verbose output
            words = len(text.split()) if text else 0
            words_per_second = words / duration if duration > 0 else 0
            if duration < 2.0 and words_per_second > 5.0:
                log.warning(f"Possible hallucination: {words} words in {duration:.1f}s ({words_per_second:.1f} w/s)")
                return None

            # Apply post-processing
            raw_text = text
            text = self._clean_transcription(text)

            if raw_text != text:
                log.debug(f"Cleaned: '{raw_text}' → '{text}'")

            log.debug(f"Transcription: '{text}'")
            return text if text else None

        except Exception as e:
            log.error(f"Transcription error: {e}", exc_info=True)
            print(f"Transcription error: {e}")
            return None
    
    async def speak(self, text: str):
        """Convert text to speech and play it"""
        if not text:
            return
        
        try:
            # Try Piper first
            if await self._speak_piper(text):
                return
            
            # Fallback to system TTS
            await self._speak_system(text)
            
        except Exception as e:
            print(f"TTS error: {e}")
    
    async def _speak_piper(self, text: str) -> bool:
        """Use Piper for TTS"""
        try:
            # Check if piper is installed
            result = subprocess.run(
                ["which", "piper"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                return False
            
            # Generate audio with Piper
            loop = asyncio.get_event_loop()
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            
            # Run Piper
            process = await asyncio.create_subprocess_exec(
                "piper",
                "--model", self.piper_model,
                "--output_file", temp_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.communicate(input=text.encode())
            
            # Play the audio
            if self._has_audio and Path(temp_path).exists():
                await self._play_audio_file(temp_path)
                Path(temp_path).unlink()
                return True
            
            return False
            
        except Exception as e:
            print(f"Piper TTS error: {e}")
            return False
    
    async def _speak_system(self, text: str):
        """Fallback to system TTS"""
        try:
            loop = asyncio.get_event_loop()
            
            # Try different system TTS options
            if await self._try_command(["say", text]):  # macOS
                return
            if await self._try_command(["espeak", text]):  # Linux
                return
            if await self._try_command(["spd-say", text]):  # Linux speech-dispatcher
                return
            
            # No TTS available, just print
            print(f"[TTS unavailable] {text}")
            
        except Exception as e:
            print(f"System TTS error: {e}")
    
    async def _try_command(self, cmd: list) -> bool:
        """Try to run a command, return True if successful"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.wait()
            return process.returncode == 0
        except:
            return False
    
    async def _play_audio_file(self, path: str):
        """Play an audio file"""
        try:
            import soundfile as sf
            data, samplerate = sf.read(path)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._sounddevice.play(data, samplerate)
            )
            self._sounddevice.wait()
        except ImportError:
            # Fallback to command line player
            await self._try_command(["aplay", path]) or \
            await self._try_command(["afplay", path])  # macOS
    
    async def play_activation_sound(self):
        """Play a short activation sound"""
        if not self._has_audio:
            print("🔔")
            return
        
        try:
            # Generate a simple beep
            duration = 0.1
            frequency = 880  # A5
            t = np.linspace(0, duration, int(self.sample_rate * duration))
            tone = 0.3 * np.sin(2 * np.pi * frequency * t)
            
            # Apply envelope
            envelope = np.exp(-3 * t / duration)
            tone = tone * envelope
            
            self._sounddevice.play(tone.astype(np.float32), self.sample_rate)
            self._sounddevice.wait()
        except Exception as e:
            print("🔔")
    
    def cleanup(self):
        """Clean up resources"""
        if self._has_audio:
            try:
                self._sounddevice.stop()
            except:
                pass
