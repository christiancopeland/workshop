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

from pathlib import Path

from config import Config
from agent import Agent
from memory import MemorySystem
from logger import get_logger, log as root_log
from terminal_ui import get_terminal, configure_terminal, Verbosity
import aiohttp

log = get_logger("main")


def detect_project_from_cwd() -> tuple[str | None, str | None]:
    """
    Detect project from current working directory.

    Returns:
        Tuple of (project_name, project_path) or (None, None) if not a project.
    """
    cwd = Path.cwd()
    project_name = cwd.name
    project_path = str(cwd)

    # Check for common project indicators
    indicators = [
        '.git',
        'package.json',
        'pyproject.toml',
        'Cargo.toml',
        'go.mod',
        'Makefile',
        'requirements.txt',
        'setup.py',
        '.workshop',
    ]

    is_project = any((cwd / ind).exists() for ind in indicators)

    if is_project:
        return project_name, project_path
    return None, None


async def warm_model(model: str, ollama_url: str = "http://localhost:11434") -> bool:
    """
    Pre-warm a model by making a minimal request to load it into VRAM.

    This dramatically reduces first-request latency (can save 2-10+ seconds).
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": "hi",
                    "stream": False,
                    "options": {"num_predict": 1}  # Single token
                },
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status == 200:
                    return True
                else:
                    log.warning(f"Failed to warm model {model}: {resp.status}")
                    return False
    except Exception as e:
        log.warning(f"Failed to warm model {model}: {e}")
        return False


async def warm_models(models: list, ollama_url: str = "http://localhost:11434"):
    """Pre-warm multiple models concurrently."""
    term = get_terminal()
    term.init_step("Pre-warming models for fast inference...")
    tasks = [warm_model(m, ollama_url) for m in models]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for model, result in zip(models, results):
        if result is True:
            term.debug(f"  \u2713 {model} loaded")
        else:
            term.debug(f"  \u2717 {model} failed: {result}")

# Dashboard integration (optional)
_dashboard = None

class Workshop:
    """Main orchestrator with Phase 2 voice integration"""

    def __init__(
        self,
        enable_constructs: bool = True,
        enable_dashboard: bool = False,
        trace_mode: bool = False,
        trace_dir: str = None,
        trace_format: str = "both",
        mode: str = "text",  # "voice" or "text" - affects system prompt
        verbosity: str = "normal",  # "quiet", "normal", or "verbose"
    ):
        self.config = Config()
        self.running = False
        self.construct_manager = None
        self.dashboard = None
        self.enable_dashboard = enable_dashboard
        self.mode = mode  # Store mode for agent initialization
        self.verbosity = verbosity

        # Get terminal UI instance
        self.term = get_terminal()

        # Trace mode initialization
        self.trace_mode = trace_mode
        self.trace_output = None
        if trace_mode:
            from trace_output import TraceOutputManager
            from pathlib import Path
            self.trace_output = TraceOutputManager(
                output_dir=Path(trace_dir) if trace_dir else None,
                format=trace_format,
                print_ascii=True,
            )
            self.term.info(f"Trace mode: {self.trace_output.output_dir}")

        # Initialize subsystems
        self.term.init_start()

        self.term.init_step("Loading memory system...")
        self.memory = MemorySystem(
            chroma_path=self.config.CHROMA_PATH,
            sqlite_path=self.config.SQLITE_PATH
        )

        # Auto-detect project from current working directory
        self.term.init_step("Detecting project context...")
        project_name, project_path = detect_project_from_cwd()
        if project_name:
            self.memory.set_active_project(project_name, project_path)
            self.term.debug(f"  Active project: {project_name}")
        else:
            self.term.debug("  No project detected in current directory")

        # PHASE 3: Initialize context intelligence (automatic + personal)
        self.term.init_step("Initializing context intelligence...")
        from context_manager import get_context_manager_v3
        self.context_mgr_v3 = get_context_manager_v3(
            monitored_projects=self.config.MONITORED_PROJECTS,
            indexable_extensions=self.config.INDEXABLE_EXTENSIONS,
            memory_system=self.memory
        )
        self.term.debug(f"  Monitoring {len(self.config.MONITORED_PROJECTS)} projects")

        # PHASE 3 TELOS: Personal context system
        self.term.init_step("Loading Telos (personal context)...")
        from telos_manager import get_telos_manager
        self.telos_mgr = get_telos_manager()
        telos_stats = self.telos_mgr.get_stats()
        self.term.debug(f"  Profile: {telos_stats['has_profile']}, Goals: {telos_stats['has_goals']}, "
              f"Projects: {telos_stats['project_count']}")

        # Initialize construct manager if requested
        self.construct_manager = None
        if enable_constructs:
            self.term.init_step("Initializing construct system...")
            from construct_manager import get_construct_manager
            self.construct_manager = get_construct_manager()

        # Initialize SessionManager for session isolation
        self.term.init_step("Initializing session management...")
        from session_manager import get_session_manager
        self.session_mgr = get_session_manager()

        # Determine mode for session based on passed mode
        session_mode = "voice" if self.mode == "voice" else "text"
        self.current_session = self.session_mgr.start_session(mode=session_mode)
        self.term.debug(f"  Session: {self.current_session.session_id}")

        # Initialize TaskManager for tracking multi-step work
        self.term.init_step("Initializing task management...")
        from task_manager import get_task_manager
        self.task_mgr = get_task_manager()

        # Bind TaskManager to current session
        self.task_mgr.bind_to_session(self.current_session.session_id)

        task_stats = self.task_mgr.get_stats()
        if task_stats['has_tasks']:
            self.term.debug(f"  Resumed {task_stats['total']} tasks ({task_stats['completed']} completed)")

        # Initialize SkillRegistry with dependency injection
        self.term.init_step("Loading Skills...")
        from skill_registry import SkillRegistry
        from pathlib import Path

        self.tools = SkillRegistry(
            skills_dir=Path(__file__).parent / ".workshop" / "Skills",
            dependencies={
                "memory": self.memory,
                "config": self.config,
                "context_manager": self.context_mgr_v3,
                "construct_manager": self.construct_manager,
                "telos_manager": self.telos_mgr,  # Telos personal context
                "task_manager": self.task_mgr,  # Task tracking
            }
        )
        # Add skill_registry to dependencies for subagent tool access
        self.tools.dependencies["skill_registry"] = self.tools

        self.term.init_step("Initializing agent...")
        self.agent = Agent(
            model=self.config.MODEL,
            tools=self.tools,
            memory=self.memory,
            ollama_url=self.config.OLLAMA_URL,
            construct_manager=self.construct_manager,
            context_manager=self.context_mgr_v3,  # PHASE 3: Automatic context
            telos_manager=self.telos_mgr,  # PHASE 3 TELOS: Personal context
            task_manager=self.task_mgr,  # Task tracking
            voice_mode=(self.mode == "voice")  # Voice-optimized prompts only in voice mode
        )

        # Wire session manager to agent for auto-continue validation
        self.agent.set_session_manager(self.session_mgr)

        # Start session on memory system for session-scoped messages
        self.memory.start_session(self.current_session.session_id)

        # PHASE 2 VOICE STACK - Lazy initialization (only when voice mode is used)
        self.whisper = None
        self.ollama = None
        self.piper = None
        self.vad = None
        self.speech_detector = None
        self.interruption_detector = None
        self.audio_pipeline = None
        self.wake_pipeline = None

        # Show initialization complete
        self.term.init_done(
            skills=len(self.tools.skills),
            tools=len(self.tools.list_all_tools()),
            session_id=self.current_session.session_id,
            log_file=str(root_log.log_file) if hasattr(root_log, 'log_file') else ""
        )

    def _initialize_voice_stack(self):
        """Lazy initialization of voice components (Phase 2)"""
        if self.whisper is not None:
            return  # Already initialized

        self.term.init_step("Initializing voice stack...")

        # Import voice modules here to avoid dependency issues in text mode
        from wake_pipeline import WakeWordPipeline
        from audio_pipeline import AudioFramePipeline
        from vad import VoiceActivityDetector, SpeechEndDetector, InterruptionDetector
        from ollama_stream import OllamaStreamingClient
        from piper_stream import PiperStreamingTTS

        # Try NVIDIA Canary first (best quality), fall back to Whisper
        try:
            from canary_wrapper import CanaryWrapper
            self.term.debug("  Loading NVIDIA Canary ASR...")
            self.whisper = CanaryWrapper()
            self.whisper._load_model()
            self.term.debug(f"  Canary-1B ready on {self.whisper._device}")
        except ImportError:
            self.term.debug("  Canary not available, using Whisper...")
            from whisper_wrapper import FasterWhisperWrapper
            self.whisper = FasterWhisperWrapper()
            self.whisper._load_model()
            self.term.debug(f"  Whisper {self.whisper.model_name} ready")

        self.ollama = OllamaStreamingClient()
        self.piper = PiperStreamingTTS()

        self.vad = VoiceActivityDetector()
        self.speech_detector = SpeechEndDetector(self.vad, timeout_s=90.0)
        self.interruption_detector = InterruptionDetector(self.vad)

        self.audio_pipeline = AudioFramePipeline(timeout_s=90.0)
        self.audio_pipeline.interruption_detector = self.interruption_detector

        # Wire interruption callback to stop TTS
        def on_interrupt():
            self.piper.stop()
            log.info("TTS interrupted by user speech")

        self.interruption_detector.on_interruption(on_interrupt)

        # Enable voice progress updates via hooks
        try:
            from hooks import enable_voice_progress_updates, register_voice_progress_handler
            enable_voice_progress_updates(piper_tts_instance=self.piper)
            register_voice_progress_handler()
            self.term.debug("  Voice progress updates enabled")
        except Exception as e:
            log.warning(f"Could not enable voice progress updates: {e}")

        # Set TTS callback on agent for real-time Haiku progress updates
        # This enables contextual voice feedback during long operations
        if self.agent and self.piper:
            self.agent.tts_callback = self.piper.speak
            log.info("[VOICE] TTS callback set on agent for Haiku progress updates")

            # Also inject voice settings into skill registry dependencies
            # This allows subagents to receive progress updates via spawn_subagent/parallel_dispatch
            if self.tools and hasattr(self.tools, 'dependencies'):
                self.tools.dependencies['voice_mode'] = True
                self.tools.dependencies['tts_callback'] = self.piper.speak
                log.info("[VOICE] Voice mode and TTS callback injected into skill dependencies")

        self.term.init_voice_done()

    async def process_input(self, text: str) -> str:
        """Process user input through agent with dashboard events and optional tracing."""
        # Emit user input event
        if self.dashboard:
            await self.dashboard.emit_user_input(text)

        self.memory.add_message("user", text)

        # === TRACE MODE WRAPPER ===
        if self.trace_mode and self.trace_output:
            from trace_bridge import create_pipeline_tracer_from_input, merge_telemetry_into_pipeline_trace
            from tests.e2e.context_tracer import TraceStage
            import time

            # Create pipeline trace for this request
            pipeline_trace = create_pipeline_tracer_from_input(
                text,
                self.current_session.session_id if self.current_session else ""
            )

            # Inject tracer into agent for hook callbacks
            self.agent._pipeline_tracer = pipeline_trace

            start_time = time.time()
            try:
                response = await self.agent.chat(text)

                # Complete the trace
                pipeline_trace.complete(response)

                # Merge telemetry data if available
                if hasattr(self.agent, '_current_trace') and self.agent._current_trace:
                    merge_telemetry_into_pipeline_trace(pipeline_trace, self.agent._current_trace)

                # Calculate final timing
                pipeline_trace.duration_total_ms = int((time.time() - start_time) * 1000)

            except Exception as e:
                pipeline_trace.fail(TraceStage.ERROR, str(e))
                raise
            finally:
                # Save trace
                saved = self.trace_output.save_trace(pipeline_trace)
                if saved:
                    paths = [str(p) for p in saved.values()]
                    print(f"📄 Trace saved: {', '.join(paths)}")
        else:
            response = await self.agent.chat(text)

        # Condense large responses (like research documents) for memory storage
        # The full content is available in the research platform files
        memory_response = self._condense_for_memory(response)
        self.memory.add_message("assistant", memory_response)

        # Emit response event
        if self.dashboard:
            await self.dashboard.emit_response(response)

        if self.memory.message_count % 5 == 0:
            recent = self.memory.get_recent_messages(10)
            if recent:
                await self.agent.update_user_profile(recent)

        return response

    def _condense_for_memory(self, response: str, threshold: int = 3000) -> str:
        """
        Condense large responses for memory storage.

        Research platform documents and other large outputs are stored in full
        elsewhere (files, databases). For conversation memory, we provide a
        meaningful reference that maintains context without overwhelming the
        message history.

        Note: We never truncate arbitrarily - we extract structured summaries.
        """
        # If small enough, keep as-is
        if len(response) <= threshold:
            return response

        # Detect research platform output (starts with "# Research Platform:")
        if response.startswith("# Research Platform:"):
            # Extract header info and key metadata
            lines = response.split("\n")
            summary_lines = []

            # Get header section
            in_header = True
            source_count = 0
            for line in lines:
                if in_header:
                    summary_lines.append(line)
                    if line.startswith("---"):
                        in_header = False
                elif line.startswith("## ") or line.startswith("### "):
                    # Count sections
                    if "Sources" in line:
                        # Count sources
                        source_count = sum(1 for l in lines if l.startswith("### ") and l[4:5].isdigit())

            summary_lines.append("")
            summary_lines.append(f"*Research platform with {source_count} detailed sources.*")
            summary_lines.append("*Full content available via 'show research' or get_research_section()*")
            return "\n".join(summary_lines)

        # For other large responses, provide a structured summary reference
        # Extract first meaningful paragraph and word count
        paragraphs = response.split("\n\n")
        first_para = ""
        for p in paragraphs:
            if len(p.strip()) > 50:
                first_para = p.strip()
                break

        word_count = len(response.split())
        summary = f"{first_para[:500]}\n\n*[Full response: {word_count} words, {len(response)} chars]*"
        return summary

    async def _on_wake(self):
        """Phase 2 callback: Wake word detected"""
        self.term.voice_wake()
        await self.piper.play_chime()

    async def _on_speech(self, segment, reason):
        """Phase 2 callback: Speech segment captured"""
        duration = len(segment) / 16000
        self.term.voice_listening(duration)

        # PHASE 2 → AGENT PIPELINE
        self.term.voice_transcribing()
        text = self.whisper.transcribe_array(segment, sample_rate=16000)

        if not text or len(text.strip()) < 3:
            self.term.voice_no_speech()
            # Return to idle state
            self.wake_pipeline.set_state("idle")
            return

        self.term.voice_transcribed(text)

        # Check exit commands
        if self._is_exit_command(text):
            await self.piper.speak("Goodbye!")
            self.term.goodbye()
            self.running = False
            self.wake_pipeline.stop()
            return

        # Agent processing
        self.term.thinking()
        response = await self.process_input(text)
        self.term.assistant_response(response)

        # Phase 2 TTS with interruption support
        self.term.voice_speaking()
        self.wake_pipeline.set_state("speaking")
        self.audio_pipeline.set_assistant_speaking(True)
        await self.piper.speak(response)
        self.audio_pipeline.set_assistant_speaking(False)

        # Return to idle state to listen for next wake word
        self.term.voice_ready()
        self.wake_pipeline.set_state("idle")

    async def _text_input_loop(self):
        """
        Concurrent text input loop for voice mode.

        Allows users to type text commands while voice mode is active.
        Text input is triggered with '/text' prefix, e.g., '/text hello workshop'

        This runs in parallel with the wake word pipeline.
        """
        loop = asyncio.get_event_loop()

        while self.running:
            try:
                # Read stdin in executor to avoid blocking
                line = await loop.run_in_executor(
                    None,
                    lambda: sys.stdin.readline()
                )

                if not line:
                    # EOF - stdin closed
                    continue

                line = line.strip()
                if not line:
                    continue

                # Check for /text prefix
                if line.lower().startswith('/text '):
                    text = line[6:].strip()  # Remove '/text ' prefix
                    if text:
                        await self._process_text_in_voice_mode(text)
                elif line.lower() == '/text':
                    self.term.status("Usage: /text <your message>")
                elif self._is_exit_command(line):
                    self.term.goodbye()
                    self.running = False
                    if self.wake_pipeline:
                        self.wake_pipeline.stop()
                    break
                # Silently ignore other input (could be accidental keystrokes)

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Don't crash on input errors, just log and continue
                self.term.debug(f"Text input error: {e}")
                continue

    async def _process_text_in_voice_mode(self, text: str):
        """
        Process text input in voice mode (similar to _on_speech but without transcription).

        Args:
            text: The text message to process
        """
        self.term.user_input(text)

        # Pause wake word detection during processing
        if self.wake_pipeline:
            prev_state = self.wake_pipeline.state
            self.wake_pipeline.set_state("processing")

        try:
            # Agent processing
            self.term.thinking()
            response = await self.process_input(text)
            self.term.assistant_response(response)

            # TTS output (same as voice mode)
            if self.piper:
                self.term.voice_speaking()
                if self.wake_pipeline:
                    self.wake_pipeline.set_state("speaking")
                if hasattr(self, 'audio_pipeline') and self.audio_pipeline:
                    self.audio_pipeline.set_assistant_speaking(True)

                await self.piper.speak(response)

                if hasattr(self, 'audio_pipeline') and self.audio_pipeline:
                    self.audio_pipeline.set_assistant_speaking(False)
        finally:
            # Return to idle state
            if self.wake_pipeline:
                self.term.voice_ready()
                self.wake_pipeline.set_state("idle")

    async def run_voice_mode_phase2(self):
        """PHASE 2 PRODUCTION MODE - Full integration"""
        # Initialize voice stack on demand
        self._initialize_voice_stack()

        self.term.voice_mode_start(wake_word="alexa")

        # Import WakeWordPipeline here (already imported in _initialize_voice_stack)
        from wake_pipeline import WakeWordPipeline

        # Initialize wake pipeline with event loop
        loop = asyncio.get_event_loop()
        self.wake_pipeline = WakeWordPipeline(
            wake_word="alexa",
            timeout_s=90.0,
            workshop=self,
            event_loop=loop  # Pass event loop for async callbacks
        )

        # Wire Phase 2 callbacks
        self.wake_pipeline.register_callbacks(
            on_wake=self._on_wake,
            on_speech=self._on_speech
        )

        self.running = True

        # Show hint for text input
        self.term.status("Voice mode active. Type '/text <message>' for text input, or 'exit' to quit.")

        try:
            # Run voice pipeline and text input loop concurrently
            # Voice pipeline runs in executor (blocking), text loop is async
            voice_task = loop.run_in_executor(None, self.wake_pipeline.run)
            text_task = asyncio.create_task(self._text_input_loop())

            # Wait for either to complete (usually exit command or Ctrl+C)
            done, pending = await asyncio.wait(
                [voice_task, text_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel any pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        except KeyboardInterrupt:
            pass  # Handled by cleanup
        finally:
            self.cleanup()
    
    # Legacy modes (unchanged for compatibility)
    async def run_voice_mode_v2(self):
        """Legacy Phase 1 voice mode (DEPRECATED)"""
        self.term.warning("Using legacy Phase 1 voice stack")
        pass

    async def run_voice_mode(self):
        """Legacy Phase 1 voice mode (DEPRECATED)"""
        self.term.warning("Using legacy Phase 1 voice mode")
        pass

    async def run_text_mode(self):
        """Text mode with clean terminal UI"""
        self.running = True
        self.term.status("Text mode. Type 'exit' to quit.")

        while self.running:
            try:
                text = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("You: ")
                )

                if not text.strip():
                    continue

                if self._is_exit_command(text):
                    self.term.goodbye()
                    break

                self.term.thinking()
                response = await self.process_input(text)
                self.term.assistant_response(response)

            except (KeyboardInterrupt, EOFError):
                break

    def _is_exit_command(self, text: str) -> bool:
        """Check exit commands"""
        exit_commands = {'exit', 'quit', 'goodbye', 'bye', 'stop', 'shut down'}
        return text.lower().strip() in exit_commands

    def cleanup(self):
        """Graceful Phase 2 cleanup - handles sounddevice properly"""
        self.term.cleanup_start()

        try:
            # Stop audio FIRST (most important)
            if hasattr(self, 'wake_pipeline') and self.wake_pipeline:
                self.wake_pipeline.stop()
            if hasattr(self, 'audio_pipeline') and hasattr(self.audio_pipeline, 'stop_listening'):
                self.audio_pipeline.stop_listening()

            # Memory (safe)
            self.memory.save_session()

            # Construct server
            if self.construct_manager:
                asyncio.create_task(self.stop_construct_server())

            # Disable voice updates if they were enabled
            try:
                from hooks import disable_voice_progress_updates
                disable_voice_progress_updates()
            except ImportError:
                pass

            self.term.cleanup_done()
        except Exception as e:
            self.term.warning(f"Cleanup: {e}")

    
    async def start_construct_server(self):
        if self.construct_manager:
            await self.construct_manager.start_server()

    async def stop_construct_server(self):
        if self.construct_manager:
            await self.construct_manager.stop_server()

    async def start_dashboard(self):
        """Start the real-time dashboard server."""
        if self.enable_dashboard:
            from dashboard import start_dashboard
            import dashboard_integration
            self.dashboard = await start_dashboard()
            dashboard_integration.set_dashboard(self.dashboard)
            # Inject dashboard into tools for research events
            self.tools.dependencies['dashboard'] = self.dashboard
            # Connect task manager to dashboard for task events
            if self.task_mgr:
                self.task_mgr.set_dashboard(self.dashboard)
            # Connect session manager to dashboard for session events
            if self.session_mgr:
                self.dashboard.set_session_manager(self.session_mgr)
                self.session_mgr.set_dashboard(self.dashboard)
            # Register message handler for dashboard commands
            self.dashboard.set_message_handler(self._handle_dashboard_message)
            self.dashboard.set_stop_handler(self._handle_emergency_stop)

    async def _handle_dashboard_message(self, message: str):
        """Handle a message sent from the dashboard UI."""
        log.info(f"Dashboard message: {message}")
        response = await self.process_input(message)
        # Response is already emitted via process_input -> dashboard.emit_response
        return response

    async def _handle_emergency_stop(self):
        """Handle emergency stop request from dashboard."""
        log.warning("Emergency stop triggered from dashboard")
        # Cancel any running agent tasks
        if hasattr(self.agent, 'cancel_current'):
            await self.agent.cancel_current()
        # Reset state
        if self.dashboard:
            await self.dashboard.emit_warning("Emergency stop activated")

    async def stop_dashboard(self):
        """Stop the dashboard server."""
        if self.dashboard:
            await self.dashboard.stop()

async def run_with_constructs(workshop, mode: str):
    """Run with construct server and optional dashboard."""
    term = get_terminal()

    if workshop.construct_manager:
        await workshop.start_construct_server()
        term.debug("Construct UI: ws://localhost:8765")

    # Start dashboard if enabled
    if workshop.enable_dashboard:
        await workshop.start_dashboard()

    try:
        if mode == 'voice':
            await workshop.run_voice_mode_phase2()  # Phase 2 default
        elif mode == 'voice-v1':
            await workshop.run_voice_mode_v2()
        elif mode == 'voice-legacy':
            await workshop.run_voice_mode()
        else:
            await workshop.run_text_mode()
    finally:
        if workshop.construct_manager:
            await workshop.stop_construct_server()
        if workshop.dashboard:
            await workshop.stop_dashboard()


def main():
    """Workshop Entry Point with clean terminal UI"""
    parser = argparse.ArgumentParser(
        description="Workshop - AI Voice Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Verbosity levels:
  --quiet    Minimal output (errors and responses only)
  (default)  Clean status updates
  --verbose  Full debug output with hook traces

Examples:
  python main.py --mode voice          # Voice mode (default)
  python main.py --mode text --quiet   # Text mode, minimal output
  python main.py --verbose             # Show all debug info
"""
    )
    parser.add_argument(
        '--mode',
        choices=['voice', 'text', 'voice-v1', 'voice-legacy'],
        default='voice',
        help='voice=Phase2, text, voice-v1=Phase1 new, voice-legacy=Phase1 old'
    )
    parser.add_argument('--model', help='Override Ollama model')
    parser.add_argument('--no-constructs', action='store_true', help='Disable constructs')
    parser.add_argument('--dashboard', action='store_true', help='Enable real-time dashboard')

    # Verbosity control
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output (errors and responses only)'
    )
    verbosity_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Full debug output including hook traces'
    )

    # Trace mode arguments
    parser.add_argument(
        '--trace-mode',
        action='store_true',
        help='Enable comprehensive pipeline tracing'
    )
    parser.add_argument(
        '--trace-dir',
        type=str,
        default=None,
        help='Directory for trace output (default: ~/.workshop/traces/)'
    )
    parser.add_argument(
        '--trace-format',
        choices=['json', 'markdown', 'both', 'ascii'],
        default='both',
        help='Output format for traces (default: both)'
    )
    args = parser.parse_args()

    # Configure terminal UI based on verbosity
    if args.quiet:
        verbosity = "quiet"
    elif args.verbose:
        verbosity = "verbose"
    else:
        verbosity = "normal"

    configure_terminal(
        verbosity=verbosity,
        show_tool_calls=(verbosity != "quiet"),
        show_hook_traces=(verbosity == "verbose")
    )

    workshop = Workshop(
        enable_constructs=not args.no_constructs,
        enable_dashboard=args.dashboard,
        trace_mode=args.trace_mode,
        trace_dir=args.trace_dir,
        trace_format=args.trace_format,
        mode=args.mode,
        verbosity=verbosity,
    )

    if args.model:
        workshop.agent.model = args.model

    term = get_terminal()

    def signal_handler(sig, frame):
        """Graceful shutdown"""
        workshop.running = False

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
