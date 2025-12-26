"""
Wake Word Pipeline (WAKE-001) - FULLY FIXED VERSION

Integrates wake word detection with audio frame pipeline for hands-free activation.
Manages state machine: idle → wake → listening → processing → speaking → idle
"""

import numpy as np
import time
from typing import Optional, Callable
from logger import get_logger

# Import Phase 1 wake word detector
from wake_word import WakeWordDetector

# Import Phase 2 audio pipeline
from audio_pipeline import AudioFramePipeline

log = get_logger("wake_pipeline")

class WakeWordPipeline:
    """
    Integrates wake word detection with audio frame pipeline.
    """
    
    def __init__(self,
                 wake_word: str = "alexa",  # ← FIXED: Use built-in model
                 model_path: Optional[str] = None,
                 timeout_s: float = 30.0,
                 sample_rate: int = 44100,       # ← FIXED: Blue mic rate
                 workshop=None,                 # ← FIXED: Proper param
                 event_loop=None):              # ← NEW: Event loop for async callbacks
        """
        Initialize wake word pipeline.
        """
        self.wake_word = wake_word
        self.sample_rate = sample_rate
        self.workshop = workshop
        self.event_loop = event_loop  # Store event loop for async callbacks
        
        log.info(f"Initializing WakeWordPipeline (wake_word='{wake_word}', timeout={timeout_s}s)")

        # FIXED: Initialize wake word detector PROPERLY with correct model
        self.wake_detector = WakeWordDetector(model_name=wake_word)
        
        # FIXED: Match Blue mic sample rate
        self.audio_pipeline = AudioFramePipeline(
            sample_rate=sample_rate,
            timeout_s=timeout_s
        )
        
        # State machine
        self.state = "idle"
        self.state_start_time = time.time()
        
        # Callbacks
        self.on_wake = None
        self.on_speech = None
        
        # Statistics
        self.wake_count = 0
        self.speech_count = 0
        self.state_transitions = []
        
        # Control
        self.running = False
        
        log.info("✅ WakeWordPipeline initialized successfully")
    
    def register_callbacks(self,
                          on_wake: Optional[Callable] = None,
                          on_speech: Optional[Callable] = None):
        """Register callbacks for events."""
        self.on_wake = on_wake
        self.on_speech = on_speech
        log.info(f"Callbacks registered: on_wake={on_wake is not None}, on_speech={on_speech is not None}")
    
    def set_state(self, new_state: str):
        """Set pipeline state with transition logging."""
        if new_state == self.state:
            return
        
        duration = time.time() - self.state_start_time
        log.info(f"State transition: {self.state} → {new_state} (was {self.state} for {duration:.2f}s)")
        
        self.state_transitions.append({
            "from": self.state,
            "to": new_state,
            "duration": duration,
            "timestamp": time.time()
        })
        
        self.state = new_state
        self.state_start_time = time.time()
    
    def _process_idle_state(self, frame: np.ndarray):
        """Process frame in idle state (listening for wake word)."""
        # FIXED: Add debug logging + proper frame validation
        if frame is None or len(frame) == 0:
            return

        # Convert 44.1kHz frame to 16kHz for wake word detector
        frame_16k = self.audio_pipeline.to_vad_frame(frame)

        # DEBUG: Log wake word scores
        prediction = self.wake_detector.process_frame(frame_16k)
        if prediction:
            log.debug(f"🔍 Wake scores: {prediction}")
            
            # Check if our wake word triggered
            if self.wake_word in prediction and prediction[self.wake_word] > 0.5:
                self.wake_count += 1
                log.info(f"🎤 WAKE WORD DETECTED! '{self.wake_word}' (#{self.wake_count}, score={prediction[self.wake_word]:.3f})")

                # Fire callback (handle both sync and async)
                if self.on_wake:
                    try:
                        import asyncio
                        import inspect
                        if inspect.iscoroutinefunction(self.on_wake):
                            # Async callback - schedule on provided event loop
                            if self.event_loop:
                                # We have an event loop - schedule callback and wait
                                future = asyncio.run_coroutine_threadsafe(
                                    self.on_wake(),
                                    self.event_loop
                                )
                                # Block until callback completes (with 10s timeout)
                                log.debug("Waiting for async wake callback to complete...")
                                future.result(timeout=10.0)
                                log.debug("Async wake callback completed")
                            else:
                                # No event loop provided - create temporary one
                                log.warning("No event loop provided, creating temporary loop for wake callback")
                                temp_loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(temp_loop)
                                try:
                                    temp_loop.run_until_complete(self.on_wake())
                                finally:
                                    temp_loop.close()
                        else:
                            # Sync callback
                            self.on_wake()
                    except Exception as e:
                        log.error(f"Wake callback error: {e}")
                        import traceback
                        traceback.print_exc()

                # Transition to listening
                self.set_state("listening")

                # IMPORTANT: Allow audio queue to refill after wake word detection
                # During wake word detection, frames are consumed rapidly which can
                # drain the queue. Give it time to build up a buffer before capturing speech.
                queue_size = self.audio_pipeline.capture.frame_queue.qsize()
                log.info(f"Queue size before refill: {queue_size} frames")

                # Wait for queue to fill (target: at least 10 frames = 320ms of audio)
                refill_start = time.time()
                while self.audio_pipeline.capture.frame_queue.qsize() < 10 and (time.time() - refill_start) < 0.5:
                    time.sleep(0.05)

                queue_size_after = self.audio_pipeline.capture.frame_queue.qsize()
                refill_time = time.time() - refill_start
                log.info(f"Queue refilled: {queue_size_after} frames in {refill_time*1000:.0f}ms")

                # Start listening mode (capture stream already running, just reset VAD)
                self.audio_pipeline.start_listening(start_capture=False)
    
    def _process_listening_state(self):
        """Process listening state (capturing user command)."""
        try:
            # Capture speech segment (blocking)
            segment, reason = self.audio_pipeline.capture_speech_segment(
                max_wait_s=30.0,
                require_speech=False
            )
            
            if segment is not None:
                self.speech_count += 1
                duration = len(segment) / self.sample_rate
                log.info(f"📝 SPEECH CAPTURED! (#{self.speech_count}, {duration:.2f}s, {reason})")
                
                # Fire callback (handle both sync and async)
                if self.on_speech:
                    try:
                        import asyncio
                        import inspect
                        if inspect.iscoroutinefunction(self.on_speech):
                            # Async callback - schedule on provided event loop
                            if self.event_loop:
                                # We have an event loop - schedule callback and wait
                                future = asyncio.run_coroutine_threadsafe(
                                    self.on_speech(segment, reason),
                                    self.event_loop
                                )
                                # Block until callback completes (with 60s timeout)
                                log.debug("Waiting for async speech callback to complete...")
                                future.result(timeout=60.0)
                                log.debug("Async speech callback completed")
                            else:
                                # No event loop provided - create temporary one
                                log.warning("No event loop provided, creating temporary loop for callback")
                                temp_loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(temp_loop)
                                try:
                                    temp_loop.run_until_complete(self.on_speech(segment, reason))
                                finally:
                                    temp_loop.close()
                        else:
                            # Sync callback
                            self.on_speech(segment, reason)
                    except Exception as e:
                        log.error(f"Speech callback error: {e}")
                        import traceback
                        traceback.print_exc()
                
                self.set_state("processing")
            else:
                log.warning("No speech detected, returning to idle")
                self.set_state("idle")
                
        except TimeoutError:
            log.warning("Speech capture timeout, returning to idle")
            self.set_state("idle")
        except Exception as e:
            log.error(f"Error capturing speech: {e}")
            self.set_state("idle")
        finally:
            # Don't stop capture - it needs to keep running for next wake word
            # Just reset the listening flag
            self.audio_pipeline.listening = False
    
    def run(self):
        """Main loop (blocking)."""
        log.info("🚀 Starting wake word pipeline...")
        self.running = True

        # Start audio capture for wake word detection
        if not self.audio_pipeline.capture.start():
            log.error("Failed to start audio capture")
            return

        try:
            while self.running and (self.workshop is None or self.workshop.running):
                if self.state == "idle":
                    # Get audio frame for wake word detection with timeout
                    # Use timeout to prevent tight loop and allow queue to fill
                    frame = self.audio_pipeline.capture.get_frame(timeout=0.05)
                    if frame is not None:
                        self._process_idle_state(frame)
                    # No sleep needed - timeout handles timing
                
                elif self.state == "listening":
                    self._process_listening_state()
                
                elif self.state in ["processing", "speaking"]:
                    time.sleep(0.1)  # Wait for external processing
                
                else:
                    log.error(f"Unknown state: {self.state}")
                    self.set_state("idle")
        
        except KeyboardInterrupt:
            log.info("Keyboard interrupt received")
        except Exception as e:
            log.error(f"Pipeline error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
    
    def stop(self):
        """Stop pipeline gracefully."""
        log.info("🛑 Stopping pipeline...")
        self.running = False
        try:
            # Stop listening if active
            if self.audio_pipeline.listening:
                self.audio_pipeline.stop_listening()
            # Stop the audio capture stream
            if self.audio_pipeline.capture:
                self.audio_pipeline.capture.stop()
        except Exception as e:
            log.warning(f"Error during stop: {e}")
        log.info("✅ Pipeline stopped")
    
    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        current_state_duration = time.time() - self.state_start_time
        return {
            "current_state": self.state,
            "wake_count": self.wake_count,
            "speech_count": self.speech_count,
            "wake_word": self.wake_word
        }

def test_wake_pipeline():
    """Basic test."""
    print("🧪 Testing WakeWordPipeline...")
    pipeline = WakeWordPipeline(wake_word="hey_jarvis")
    
    def test_wake():
        print("✅ Wake callback fired!")
    
    def test_speech(segment, reason):
        print(f"✅ Speech callback: {len(segment)} samples ({reason})")
    
    pipeline.register_callbacks(on_wake=test_wake, on_speech=test_speech)
    print("✅ Test complete - run with main.py for full integration!")

if __name__ == "__main__":
    test_wake_pipeline()
