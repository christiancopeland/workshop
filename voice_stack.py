# voice_stack.py - PHASE 2 INTEGRATION
"""
Phase 2 Complete: WakeWordPipeline → VAD → Whisper → Ollama → Piper
"""

from wake_pipeline import WakeWordPipeline
from whisper_wrapper import WhisperWrapper
from ollama_stream import OllamaStreamer  
from piper_stream import PiperStreamer
from logger import get_logger

log = get_logger("voice_stack")

class VoiceStack:
    def __init__(self, config=None):
        self.pipeline = WakeWordPipeline(wake_word="hey_jarvis", timeout_s=90.0)
        self.whisper = WhisperWrapper()
        self.ollama = OllamaStreamer()
        self.piper = PiperStreamer()
        
        # Wire Phase 2 callbacks
        self.pipeline.register_callbacks(
            on_wake=self._on_wake,
            on_speech=self._on_speech
        )
    
    async def _on_wake(self):
        log.info("🎤 Wake word 'workshop' detected")
        await self.piper.play_chime()
    
    async def _on_speech(self, segment, reason):
        log.info(f"📦 Speech captured ({reason}): {len(segment)/16000:.1f}s")
        
        # Phase 2 → Phase 1 handoff (STT → LLM → TTS)
        text = self.whisper.transcribe(segment)
        if text:
            response = await self.ollama.chat(text)
            await self.piper.speak(response)
    
    async def start(self):
        return True  # WakeWordPipeline handles its own init
    
    async def run(self):
        self.pipeline.run()  # Phase 2 blocking main loop
    
    def stop(self):
        self.pipeline.stop()
