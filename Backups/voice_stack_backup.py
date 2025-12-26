"""
Workshop Voice Stack
- OpenWakeWord: Always-on wake word detection (CPU, low power)
- faster-whisper: Fast transcription after wake (GPU)
- Piper: Text-to-speech responses (CPU)
"""

import asyncio
import numpy as np
import threading
import subprocess
import tempfile
import sys
import wave
from pathlib import Path
from typing import Optional, Callable, Awaitable
from collections import deque
from dataclasses import dataclass
from enum import Enum

from logger import get_logger

log = get_logger("voice_stack")


class VoiceState(Enum):
    IDLE = "idle"           # Waiting for wake word
    LISTENING = "listening"  # Recording user speech
    PROCESSING = "processing"  # Transcribing / thinking
    SPEAKING = "speaking"    # TTS output


@dataclass
class VoiceConfig:
    """Voice stack configuration"""
    # Wake word (only used if require_wake_word=True)
    wake_word_model: str = "hey_jarvis"  # or path to custom .onnx
    wake_threshold: float = 0.5
    require_wake_word: bool = False  # Set True for always-on background mode
    
    # Transcription
    whisper_model: str = "distil-large-v3"  # Best speed/accuracy
    whisper_device: str = "cuda"
    whisper_compute: str = "int8_float16"
    
    # TTS
    piper_model: str = "en_US-lessac-medium"
    piper_speaker: int = 0
    
    # Audio
    sample_rate: int = 44100    # Native rate
    device_id: int = 9  # Blue Microphones USB Audio
    
    # Behavior
    silence_threshold: float = 0.01
    silence_duration: float = 1.5  # seconds of silence to stop recording
    max_listen_duration: float = 30.0
    
    # Feedback
    play_activation_sound: bool = True
    print_transcript: bool = True


class VoiceStack:
    """
    Complete voice interface for Workshop.
    
    Usage:
        voice = VoiceStack()
        await voice.start()
        
        # Voice will call your handler when user speaks
        voice.on_command = async def handler(text): ...
        
        # Speak a response
        await voice.speak("Compiling now")
    """
    
    def __init__(self, config: Optional[VoiceConfig] = None):
        self.config = config or VoiceConfig()
        self.state = VoiceState.IDLE
        
        # Callbacks
        self.on_wake: Optional[Callable[[], Awaitable[None]]] = None
        self.on_command: Optional[Callable[[str], Awaitable[str]]] = None
        self.on_state_change: Optional[Callable[[VoiceState], None]] = None
        
        # Audio
        self._sounddevice = None
        self._stream = None
        self._audio_buffer = deque(maxlen=int(self.config.sample_rate * 2))  # 2s rolling buffer
        
        # Models (lazy loaded)
        self._oww_model = None
        self._whisper_model = None
        
        # Control
        self._running = False
        self._loop = None
        
        log.info(f"VoiceStack initialized: wake={self.config.wake_word_model}, whisper={self.config.whisper_model}")
    
    def _set_state(self, state: VoiceState):
        """Update state and notify"""
        if state != self.state:
            self.state = state
            log.debug(f"State: {state.value}")
            if self.on_state_change:
                self.on_state_change(state)
    
    # =========================================================================
    # Initialization
    # =========================================================================
    
    def _init_audio(self) -> bool:
        """Initialize audio input"""
        try:
            import sounddevice as sd
            self._sounddevice = sd
            
            # Find input device
            default = sd.query_devices(kind='input')
            log.info(f"Audio input: {default['name']}")
            return True
            
        except ImportError:
            log.error("sounddevice not installed: pip install sounddevice")
            return False
        except Exception as e:
            log.error(f"Audio init failed: {e}")
            return False
    
    def _init_wake_word(self) -> bool:
        """Initialize OpenWakeWord"""
        try:
            from openwakeword.model import Model
            
            print("  → Loading wake word model...")
            
            # Check if it's a built-in model or custom path
            model_path = self.config.wake_word_model
            if not Path(model_path).exists():
                # Use built-in model name
                self._oww_model = Model(
                    wakeword_models=[model_path],
                    inference_framework="onnx"
                )
            else:
                # Custom model file
                self._oww_model = Model(
                    wakeword_models=[model_path],
                    inference_framework="onnx"
                )
            
            log.info(f"OpenWakeWord loaded: {model_path}")
            return True
            
        except ImportError:
            log.error("openwakeword not installed: pip install openwakeword")
            print("⚠️  OpenWakeWord not installed. Install with:")
            print("   pip install openwakeword")
            return False
        except Exception as e:
            log.error(f"Wake word init failed: {e}")
            print(f"⚠️  Wake word init failed: {e}")
            return False
    
    def _init_whisper(self) -> bool:
        """Initialize faster-whisper"""
        try:
            from faster_whisper import WhisperModel
            
            print(f"  → Loading Whisper ({self.config.whisper_model})...")
            
            try:
                self._whisper_model = WhisperModel(
                    self.config.whisper_model,
                    device=self.config.whisper_device,
                    compute_type=self.config.whisper_compute
                )
                log.info(f"faster-whisper loaded: {self.config.whisper_model} ({self.config.whisper_device})")
            except Exception as e:
                log.warning(f"GPU failed ({e}), trying CPU...")
                print("  → GPU unavailable, using CPU...")
                self._whisper_model = WhisperModel(
                    self.config.whisper_model,
                    device="cpu",
                    compute_type="int8"
                )
                log.info(f"faster-whisper loaded: {self.config.whisper_model} (CPU)")
            
            return True
            
        except ImportError:
            log.error("faster-whisper not installed: pip install faster-whisper")
            print("⚠️  faster-whisper not installed. Install with:")
            print("   pip install faster-whisper")
            return False
        except Exception as e:
            log.error(f"Whisper init failed: {e}")
            print(f"⚠️  Whisper init failed: {e}")
            return False
    
    def _check_piper(self) -> bool:
        """Check if Piper is available (pip package or binary)"""
        # Check pip package first
        try:
            from piper import PiperVoice
            log.info("Piper TTS available (pip package)")
            return True
        except ImportError:
            pass
        
        # Check binary
        try:
            result = subprocess.run(["which", "piper"], capture_output=True)
            if result.returncode == 0:
                log.info("Piper TTS available (binary)")
                return True
        except Exception:
            pass
        
        log.warning("Piper not found")
        print("⚠️  Piper TTS not installed. Install with:")
        print("   pip install piper-tts")
        return False
    
    # =========================================================================
    # Main Loop
    # =========================================================================
    
    async def start(self) -> bool:
        """Initialize and start the voice stack"""
        print("🎤 Starting voice stack...")
        
        if not self._init_audio():
            return False
        
        # Only load wake word model if required
        if self.config.require_wake_word:
            if not self._init_wake_word():
                return False
        
        if not self._init_whisper():
            return False
        
        self._check_piper()  # Optional, don't fail if missing
        
        self._running = True
        self._loop = asyncio.get_event_loop()
        
        print("✅ Voice stack ready!")
        if self.config.require_wake_word:
            print(f"   Wake word: '{self.config.wake_word_model}'")
            print(f"   Say the wake word to activate\n")
        else:
            print("   Continuous mode - speak anytime\n")
        
        return True
    
    async def run(self):
        """Main voice loop - call after start()"""
        if not self._running:
            log.error("Voice stack not started")
            return
        
        log.info("Voice loop starting")
        
        # In continuous mode, start in LISTENING state
        if not self.config.require_wake_word:
            self._set_state(VoiceState.LISTENING)
            # Play initial beep to indicate ready
            if self.config.play_activation_sound:
                await self._play_beep()
        
        # Track if we just spoke (to play beep after response)
        just_spoke = False
        
        while self._running:
            try:
                if self.state == VoiceState.IDLE:
                    # Only used when wake word is required
                    activated = await self._wait_for_wake()
                    if activated:
                        self._set_state(VoiceState.LISTENING)
                        if self.config.play_activation_sound:
                            await self._play_beep()
                
                elif self.state == VoiceState.LISTENING:
                    # Play beep after speaking a response (feedback that we're listening again)
                    if just_spoke and self.config.play_activation_sound:
                        await self._play_beep()
                        just_spoke = False
                    
                    # Notify
                    if self.on_wake:
                        await self.on_wake()
                    
                    # Listen for command
                    text = await self._listen()
                    
                    if text:
                        self._set_state(VoiceState.PROCESSING)
                        
                        if self.config.print_transcript:
                            print(f"You: {text}")
                        
                        # Check for exit before processing
                        if any(word in text.lower() for word in ['goodbye', 'exit', 'quit', 'stop listening']):
                            if self.on_command:
                                await self.on_command(text)  # Let handler know
                            self._running = False
                            break
                        
                        # Process command
                        if self.on_command:
                            response = await self.on_command(text)
                            if response:
                                self._set_state(VoiceState.SPEAKING)
                                await self.speak(response)
                                just_spoke = True
                        
                        # Back to listening (continuous mode) or idle (wake word mode)
                        if self.config.require_wake_word:
                            self._set_state(VoiceState.IDLE)
                        else:
                            self._set_state(VoiceState.LISTENING)
                    else:
                        # No speech detected, keep listening
                        if not self.config.require_wake_word:
                            # Small delay before next listen attempt
                            await asyncio.sleep(0.1)
                
                else:
                    await asyncio.sleep(0.1)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Voice loop error: {e}", exc_info=True)
                self._set_state(VoiceState.LISTENING if not self.config.require_wake_word else VoiceState.IDLE)
                await asyncio.sleep(1)
        
        log.info("Voice loop ended")
    
    def stop(self):
        """Stop the voice stack"""
        self._running = False
        if self._stream:
            self._stream.stop()
        log.info("Voice stack stopped")
    
    # =========================================================================
    # Wake Word Detection
    # =========================================================================
    
    async def _wait_for_wake(self) -> bool:
        """Listen for wake word using OpenWakeWord"""
        if not self._oww_model:
            log.error("Wake word model not loaded")
            return False
        
        chunk_size = 1280  # ~80ms at 16kHz (OpenWakeWord requirement)
        
        try:
            with self._sounddevice.InputStream(
                samplerate=self.config.sample_rate,
                channels=2,
                dtype=np.int16,
                blocksize=chunk_size
            ) as stream:
                log.debug("Listening for wake word...")
                
                while self._running and self.state == VoiceState.IDLE:
                    # Read audio chunk
                    audio, _ = stream.read(chunk_size)
                    audio = audio.flatten()
                    
                    # Feed to OpenWakeWord
                    prediction = self._oww_model.predict(audio)
                    
                    # Check all wake word scores
                    for name, score in prediction.items():
                        if score > self.config.wake_threshold:
                            log.info(f"Wake word detected: {name} (score={score:.3f})")
                            print(f"\n✨ Wake word detected!")
                            
                            # Reset model state
                            self._oww_model.reset()
                            return True
                    
                    # Yield to event loop
                    await asyncio.sleep(0)
                
        except Exception as e:
            log.error(f"Wake word detection error: {e}")
        
        return False
    
    # =========================================================================
    # Speech Recognition
    # =========================================================================
    
    async def _listen(self) -> Optional[str]:
        """Record speech and transcribe"""
        if not self._whisper_model:
            log.error("Whisper not loaded")
            return None
        
        print("🎤 Listening...")
        
        # Record until silence
        audio = await self._record_until_silence()
        
        if audio is None or len(audio) < self.config.sample_rate * 0.3:
            log.debug("Audio too short")
            return None
        
        # Check if mostly silence
        amplitude = np.abs(audio).mean()
        if amplitude < 0.005:
            log.debug(f"Audio too quiet ({amplitude:.4f})")
            return None
        
        print("🧠 Processing...")
        
        # Transcribe
        return await self._transcribe(audio)
    
    async def _record_until_silence(self) -> Optional[np.ndarray]:
        chunk_ms = 100
        chunk_samples = int(self.config.sample_rate * chunk_ms / 1000)
        
        audio_data = []
        silence_counter = 0
        done = asyncio.Event()
        
        def callback(indata, frames, time, status):
            if status:
                log.warning(f"Status: {status}")
            
            # Stereo → mono conversion
            chunk = np.mean(indata[:, 0:1], axis=1).flatten()  # Use left channel or average
            audio_data.append(chunk)
            
            if np.abs(chunk).mean() < self.config.silence_threshold:
                silence_counter += 1
                if silence_counter >= 15:
                    done.set()
            else:
                silence_counter = 0
        
        try:
            with self._sounddevice.InputStream(
                device=self.config.device_id,
                samplerate=self.config.sample_rate,
                channels=2,  # ← CRITICAL: Blue Yeti is stereo only
                dtype=np.float32,
                blocksize=chunk_samples,
                callback=callback
            ) as stream:
                stream.start()
                await asyncio.wait_for(done.wait(), timeout=30.0)
                stream.stop()
            
            if audio_data:
                return np.concatenate(audio_data)
            return None
        except Exception as e:
            log.error(f"Recording failed: {e}")
            return None


    def _resample_for_whisper(self, audio: np.ndarray) -> np.ndarray:
        """Resample 44.1kHz → 16kHz for Whisper"""
        try:
            import librosa
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            return librosa.resample(audio, orig_sr=self.config.sample_rate, target_sr=16000)
        except ImportError:
            log.error("librosa not installed: pip install librosa")
            return audio  # Fallback (may hurt accuracy)


    
    async def _transcribe(self, audio: np.ndarray) -> Optional[str]:
        """Transcribe audio using faster-whisper"""
        audio_16k = self._resample_for_whisper(audio)  # 44.1k → 16k
        try:
            loop = asyncio.get_event_loop()
            
            # Run transcription in executor (it's blocking)
            segments, info = await loop.run_in_executor(
                None,
                lambda: self._whisper_model.transcribe(
                    audio,
                    language="en",
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=500
                    )
                )
            )
            
            # Combine segments
            text = " ".join(seg.text for seg in segments).strip()
            
            log.debug(f"Transcribed: '{text}'")
            return text if text else None
            
        except Exception as e:
            log.error(f"Transcription error: {e}")
            return None
    
    # =========================================================================
    # Text-to-Speech
    # =========================================================================
    
    async def speak(self, text: str):
        """Speak text using Piper TTS"""
        if not text:
            return
        
        log.info(f"Speaking: {text[:50]}...")
        
        # Try Piper first
        if await self._speak_piper(text):
            return
        
        # Fallback to espeak
        if await self._speak_espeak(text):
            return
        
        # No TTS available - just print
        print(f"Workshop: {text}")
    
    async def _speak_piper(self, text: str) -> bool:
        """Use Piper for high-quality TTS"""
        # Try pip package first (piper-tts)
        try:
            from piper import PiperVoice
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            
            # Load voice model
            voice = PiperVoice.load(self.config.piper_model)
            
            # Synthesize to file
            with wave.open(temp_path, 'wb') as wav_file:
                voice.synthesize(text, wav_file)
            
            # Play audio
            await self._play_wav(temp_path)
            Path(temp_path).unlink(missing_ok=True)
            return True
            
        except ImportError:
            pass  # Fall through to binary method
        except Exception as e:
            log.debug(f"Piper pip package failed: {e}, trying binary...")
        
        # Fallback to piper binary
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            
            # Run Piper binary
            process = await asyncio.create_subprocess_exec(
                "piper",
                "--model", self.config.piper_model,
                "--output_file", temp_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE
            )
            
            _, stderr = await process.communicate(input=text.encode())
            
            if process.returncode != 0:
                log.warning(f"Piper binary failed: {stderr.decode()}")
                return False
            
            # Play audio
            await self._play_wav(temp_path)
            
            # Cleanup
            Path(temp_path).unlink(missing_ok=True)
            return True
            
        except FileNotFoundError:
            return False
        except Exception as e:
            log.error(f"Piper error: {e}")
            return False
    
    async def _speak_espeak(self, text: str) -> bool:
        """Fallback to espeak"""
        try:
            process = await asyncio.create_subprocess_exec(
                "espeak", text,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.wait()
            return process.returncode == 0
        except:
            return False
    
    async def _play_wav(self, path: str):
        """Play a WAV file"""
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
            # Fallback to aplay
            process = await asyncio.create_subprocess_exec(
                "aplay", "-q", path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.wait()
    
    async def _play_beep(self):
        """Play activation beep"""
        try:
            duration = 0.1
            freq = 880
            t = np.linspace(0, duration, int(self.config.sample_rate * duration))
            tone = 0.3 * np.sin(2 * np.pi * freq * t)
            envelope = np.exp(-3 * t / duration)
            tone = (tone * envelope).astype(np.float32)
            
            self._sounddevice.play(tone, self.config.sample_rate)
            self._sounddevice.wait()
        except:
            print("🔔")


# =============================================================================
# Simple test
# =============================================================================

async def test_voice_stack():
    """Test the voice stack in continuous mode"""
    config = VoiceConfig(
        require_wake_word=False,  # Continuous mode
        whisper_model="base.en",  # Smaller model for testing
    )
    voice = VoiceStack(config)
    
    async def on_command(text: str) -> str:
        print(f"[Command received: {text}]")
        return f"I heard you say: {text}"
    
    voice.on_command = on_command
    
    if await voice.start():
        try:
            await voice.run()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            voice.stop()


if __name__ == "__main__":
    asyncio.run(test_voice_stack())
