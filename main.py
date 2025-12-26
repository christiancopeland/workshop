#!/usr/bin/env python3
"""
Workshop - Local Agentic Voice Assistant
Phase 2 INTEGRATED: WakeWordPipeline → VAD → Whisper → Agent → Piper
"""

import asyncio
import signal
import sys
from datetime import datetime
import argparse

from config import Config
from agent import Agent
from memory import MemorySystem
from tools import ToolRegistry, register_default_tools
from logger import get_logger, log as root_log

# PHASE 2 IMPORTS - NEW INTEGRATION
from wake_pipeline import WakeWordPipeline
from audio_pipeline import AudioFramePipeline
from vad import VoiceActivityDetector, SpeechEndDetector, InterruptionDetector
from whisper_wrapper import WhisperCpp
from ollama_stream import OllamaStreamingClient
from piper_stream import PiperStreamingTTS  # Assumes Phase 2 streaming version

log = get_logger("main")

class Workshop:
    """Main orchestrator with Phase 2 voice integration"""
    
    def __init__(self, enable_constructs: bool = True):
        self.config = Config()
        self.running = False
        self.construct_manager = None
        
        # Initialize subsystems (Phase 1 unchanged)
        print("🔧 Initializing Workshop...")
        
        print("  → Loading memory system...")
        self.memory = MemorySystem(
            chroma_path=self.config.CHROMA_PATH,
            sqlite_path=self.config.SQLITE_PATH
        )
        
        print("  → Registering tools...")
        self.tools = ToolRegistry()
        self.context_mgr = register_default_tools(self.tools, self.memory)

        # PHASE 3: Initialize context intelligence
        print("  → Initializing Phase 3 context intelligence...")
        from context_manager import get_context_manager_v3
        self.context_mgr_v3 = get_context_manager_v3(
            monitored_projects=self.config.MONITORED_PROJECTS,
            indexable_extensions=self.config.INDEXABLE_EXTENSIONS,
            memory_system=self.memory
        )
        print(f"     Monitoring {len(self.config.MONITORED_PROJECTS)} projects")

        # Register Phase 3 context tools
        from context_tools import register_context_tools
        register_context_tools(self.tools, self.context_mgr_v3, self.memory)
        print("     Registered 8 context retrieval tools")

        if enable_constructs:
            print("  → Initializing construct system...")
            from construct_manager import get_construct_manager
            from construct_tools import register_construct_tools
            
            self.construct_manager = get_construct_manager()
            register_construct_tools(self.tools, self.construct_manager)
        
        print("  → Registering project tools...")
        from project_tools import register_project_tools
        register_project_tools(self.tools, self.memory, self.construct_manager)
        
        print("  → Initializing agent...")
        self.agent = Agent(
            model=self.config.MODEL,
            tools=self.tools,
            memory=self.memory,
            ollama_url=self.config.OLLAMA_URL,
            construct_manager=self.construct_manager,
            context_manager=self.context_mgr_v3  # PHASE 3: Use new context manager
        )
        
        # PHASE 2 VOICE STACK - NEW INTEGRATION
        print("  → Initializing Phase 2 voice stack...")
        self.whisper = WhisperCpp()
        self.ollama = OllamaStreamingClient()
        self.piper = PiperStreamingTTS()
        
        
        # PHASE 2 CORE PIPELINE
        self.vad = VoiceActivityDetector()
        self.speech_detector = SpeechEndDetector(self.vad, timeout_s=30.0)
        self.interruption_detector = InterruptionDetector(self.vad)
        
        self.audio_pipeline = AudioFramePipeline(timeout_s=30.0)
        self.audio_pipeline.interruption_detector = self.interruption_detector
        
        # Wire interruption callback to stop TTS
        def on_interrupt():
            self.piper.stop()
            log.info("🛑 TTS interrupted by user speech")
        
        self.interruption_detector.on_interruption(on_interrupt)
        
        # Wake pipeline will be initialized with event loop in run_voice_mode_phase2
        self.wake_pipeline = None
        
        print("✅ Phase 2 voice stack ready! (Wake → VAD → Whisper → Agent → Piper)")
        print(f"📝 Log file: {root_log.log_file}\n")
    
    async def process_input(self, text: str) -> str:
        """Process user input through agent (unchanged)"""
        self.memory.add_message("user", text)
        response = await self.agent.chat(text)
        self.memory.add_message("assistant", response)
        
        if self.memory.message_count % 5 == 0:
            recent = self.memory.get_recent_messages(10)
            if recent:
                await self.agent.update_user_profile(recent)
        
        return response
    
    async def _on_wake(self):
        """Phase 2 callback: Wake word detected"""
        print("\n✨ WORKSHOP → Wake word detected! Listening...")
        await self.piper.play_chime()
    
    async def _on_speech(self, segment, reason):
        """Phase 2 callback: Speech segment captured"""
        duration = len(segment) / 16000
        print(f"📦 Speech captured ({reason}): {duration:.1f}s")

        # PHASE 2 → AGENT PIPELINE
        print("🧠 Transcribing...")
        text = self.whisper.transcribe_array(segment, sample_rate=16000)

        if not text or len(text.strip()) < 3:
            print("  → No speech detected")
            # Return to idle state
            self.wake_pipeline.set_state("idle")
            return

        print(f"🔊 You: {text}")

        # Check exit commands
        if self._is_exit_command(text):
            await self.piper.speak("Goodbye!")
            self.running = False
            self.wake_pipeline.stop()
            return

        # Agent processing
        print("🤔 Thinking...")
        response = await self.process_input(text)
        print(f"🤖 Workshop: {response}")

        # Phase 2 TTS with interruption support
        print("🗣️  Speaking...")
        self.wake_pipeline.set_state("speaking")
        self.audio_pipeline.set_assistant_speaking(True)
        await self.piper.speak(response)
        self.audio_pipeline.set_assistant_speaking(False)

        # Return to idle state to listen for next wake word
        print("👂 Ready for next wake word...")
        self.wake_pipeline.set_state("idle")
    
    async def run_voice_mode_phase2(self):
        """PHASE 2 PRODUCTION MODE - Full integration"""
        print("🚀 Phase 2 Voice Mode ACTIVE")
        print("   Say 'ALEXA' to activate")
        print("   Interrupt anytime by speaking")
        print("   Say 'goodbye' to exit\n")

        # Initialize wake pipeline with event loop
        loop = asyncio.get_event_loop()
        self.wake_pipeline = WakeWordPipeline(
            wake_word="alexa",
            timeout_s=30.0,
            workshop=self,
            event_loop=loop  # Pass event loop for async callbacks
        )

        # Wire Phase 2 callbacks
        self.wake_pipeline.register_callbacks(
            on_wake=self._on_wake,
            on_speech=self._on_speech
        )

        self.running = True

        try:
            # PHASE 2 MAIN LOOP - Run in executor to avoid blocking event loop
            await loop.run_in_executor(None, self.wake_pipeline.run)
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted")
        finally:
            self.cleanup()
    
    # Legacy modes (unchanged for compatibility)
    async def run_voice_mode_v2(self):
        """Legacy Phase 1 voice mode (DEPRECATED)"""
        print("⚠️  Using legacy Phase 1 voice stack")
        # ... existing implementation unchanged ...
        pass
    
    async def run_voice_mode(self):
        """Legacy Phase 1 voice mode (DEPRECATED)"""
        print("⚠️  Using legacy Phase 1 voice mode")
        # ... existing implementation unchanged ...
        pass
    
    async def run_text_mode(self):
        """Text mode (unchanged)"""
        self.running = True
        print("⌨️  Text mode. Type 'exit' to quit.\n")
        
        while self.running:
            try:
                text = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("You: ")
                )
                
                if not text.strip():
                    continue
                
                if self._is_exit_command(text):
                    print("👋 Goodbye!")
                    break
                
                response = await self.process_input(text)
                print(f"Workshop: {response}\n")
                
            except (KeyboardInterrupt, EOFError):
                break
    
    def _is_exit_command(self, text: str) -> bool:
        """Check exit commands (unchanged)"""
        exit_commands = {'exit', 'quit', 'goodbye', 'bye', 'stop', 'shut down'}
        return text.lower().strip() in exit_commands
    
    def cleanup(self):
        """Graceful Phase 2 cleanup - handles sounddevice properly"""
        print("\n⏹️  Cleaning up Phase 2 components...")
        
        try:
            # Stop audio FIRST (most important)
            if hasattr(self, 'wake_pipeline'):
                self.wake_pipeline.stop()
            if hasattr(self, 'audio_pipeline') and hasattr(self.audio_pipeline, 'stop_listening'):
                self.audio_pipeline.stop_listening()
            
            # Memory (safe)
            self.memory.save_session()
            
            # Construct server
            if self.construct_manager:
                asyncio.create_task(self.stop_construct_server())
            
            print("✅ Graceful cleanup complete")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")

    
    async def start_construct_server(self):
        if self.construct_manager:
            await self.construct_manager.start_server()
    
    async def stop_construct_server(self):
        if self.construct_manager:
            await self.construct_manager.stop_server()

async def run_with_constructs(workshop, mode: str):
    """Run with construct server (unchanged)"""
    if workshop.construct_manager:
        await workshop.start_construct_server()
        print("🖼️  Construct UI: ws://localhost:8765")
    
    try:
        if mode == 'voice':
            await workshop.run_voice_mode_phase2()  # NEW PHASE 2 DEFAULT
        elif mode == 'voice-v1':
            await workshop.run_voice_mode_v2()
        elif mode == 'voice-legacy':
            await workshop.run_voice_mode()
        else:
            await workshop.run_text_mode()
    finally:
        if workshop.construct_manager:
            await workshop.stop_construct_server()

def main():
    """Phase 2 Entry Point"""
    parser = argparse.ArgumentParser(description="Workshop Phase 2 - Real-time Voice")
    parser.add_argument(
        '--mode', 
        choices=['voice', 'text', 'voice-v1', 'voice-legacy'],
        default='voice',  # Phase 2 is now default!
        help='voice=Phase2, text, voice-v1=Phase1 new, voice-legacy=Phase1 old'
    )
    parser.add_argument('--model', help='Override Ollama model')
    parser.add_argument('--no-constructs', action='store_true', help='Disable constructs')
    args = parser.parse_args()
    
    workshop = Workshop(enable_constructs=not args.no_constructs)
    
    if args.model:
        workshop.agent.model = args.model
    
    def signal_handler(sig, frame):
        """Graceful shutdown - NO sys.exit()"""
        print("\n⚠️  CTRL-C detected - shutting down gracefully...")
        workshop.running = False  # Set global flag
    
    signal.signal(signal.SIGINT, signal_handler)

    try:
    
        if args.no_constructs:
            if args.mode == 'voice':
                asyncio.run(workshop.run_voice_mode_phase2())
            else:
                asyncio.run(run_with_constructs(workshop, args.mode))
        else:
            asyncio.run(run_with_constructs(workshop, args.mode))
    finally:
        workshop.cleanup()

if __name__ == "__main__":
    main()
