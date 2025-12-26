"""
Audio Frame Processing Pipeline (AUDIO-002)

Integrates microphone capture with VAD components for speech segment detection.
Orchestrates the flow from audio input through detection to speech capture.
"""

import numpy as np
import time
from typing import Optional, Tuple, Callable
from logger import get_logger, log as root_log

# Import Phase 1 components
from audio_realtime import AudioStream

# Import Phase 2 VAD components
from vad import (
    VoiceActivityDetector,
    SpeechEndDetector,
    InterruptionDetector
)

log = get_logger("audio_pipeline")


class AudioFramePipeline:
    """
    Unified audio frame processing pipeline.
    
    Orchestrates microphone input through VAD components to speech capture.
    Provides clean API: start listening → get speech segment → transcribe.
    
    Example:
        pipeline = AudioFramePipeline(timeout_s=30.0)
        pipeline.start_listening()
        
        segment, reason = pipeline.capture_speech_segment()
        print(f"Captured {len(segment)} samples, ended via {reason}")
    """
    
    def __init__(self,
                 sample_rate: int = 16000,
                 frame_size: int = 512,
                 timeout_s: float = 30.0,
                 vad_threshold: float = 0.5,
                 interruption_threshold: float = 0.6):
        """
        Initialize audio frame pipeline.
        
        Args:
            sample_rate: Audio sample rate (16kHz for Whisper)
            frame_size: Frame size in samples (512 = 32ms @ 16kHz)
            timeout_s: Maximum speech duration before forcing end
            vad_threshold: Speech detection threshold
            interruption_threshold: Interruption detection threshold (higher = less sensitive)
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.timeout_s = timeout_s
        
        log.info(f"Initializing AudioFramePipeline (rate={sample_rate}, frame={frame_size}, timeout={timeout_s}s)")
        
        # Initialize audio capture
        self.capture = AudioStream(sample_rate=44100, channels=2, frame_size=1470, device_id=None)
        
        # Initialize VAD components
        self.vad = VoiceActivityDetector(
            threshold=vad_threshold,
            min_speech_duration_ms=250,
            min_silence_duration_ms=300
        )
        
        self.speech_detector = SpeechEndDetector(
            self.vad,
            timeout_s=timeout_s
        )
        
        self.interruption_detector = InterruptionDetector(
            self.vad,
            interruption_threshold=interruption_threshold,
            confirmation_frames=3
        )
        
        # State
        self.listening = False
        self.assistant_speaking = False
        
        # Statistics
        self.segments_captured = 0
        self.interruptions_detected = 0
        
        log.info("AudioFramePipeline initialized successfully")
    
    def start_listening(self, start_capture: bool = True):
        """
        Start capturing and processing audio.

        Starts the microphone capture and resets VAD state.

        Args:
            start_capture: If True, start the audio capture stream.
                          Set to False if stream is already running.
        """
        if self.listening:
            log.warning("Already listening")
            return

        log.info("🎤 Starting audio capture...")

        # Start microphone capture (if not already started)
        if start_capture:
            self.capture.start()

        # Reset VAD components
        self.vad.reset()
        self.speech_detector.reset()
        self.interruption_detector.reset()

        self.listening = True
        log.info("✅ Listening active")
    
    def stop_listening(self):
        """
        Stop audio capture.
        
        Stops the microphone and cleans up resources.
        """
        if not self.listening:
            return
        
        log.info("Stopping audio capture...")
        
        # Stop microphone
        self.capture.stop()
        
        self.listening = False
        log.info("Listening stopped")
    
    def set_assistant_speaking(self, speaking: bool):
        """
        Set assistant speaking state for interruption detection.
        
        Args:
            speaking: True if assistant is currently speaking (TTS active)
        """
        self.assistant_speaking = speaking
        self.interruption_detector.set_assistant_speaking(speaking)
        
        if speaking:
            log.info("🔊 Assistant speaking - monitoring for interruptions")
        else:
            log.info("🔇 Assistant stopped speaking")
    
    def on_interruption(self, callback: Callable):
        """
        Register callback for interruption events.
        
        Callback will be called when user interrupts assistant speech.
        
        Args:
            callback: Function to call on interruption (typically stops TTS)
        """
        self.interruption_detector.on_interruption(callback)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Process single audio frame through appropriate detector.

        Routes frame based on current state:
        - If assistant speaking: check for interruption
        - If listening: detect speech end

        Args:
            frame: Audio frame (variable size from hardware, typically 1470 @ 44.1kHz)

        Returns:
            (segment, reason) tuple:
                - (None, None): No segment yet
                - (segment, "natural_pause"): Speech ended naturally
                - (segment, "timeout"): Speech exceeded timeout
                - (None, "interrupted"): User interrupted assistant
        """
        # Convert frame to 16kHz 512-sample format for VAD
        frame_16k = self.to_vad_frame(frame)

        if self.assistant_speaking:
            # Check for user interruption during assistant speech
            if self.interruption_detector.process_frame(frame_16k):
                self.interruptions_detected += 1
                log.info(f"🛑 User interruption #{self.interruptions_detected}")
                return None, "interrupted"

            return None, None

        else:
            # Normal speech detection and capture
            segment, reason = self.speech_detector.process_frame(frame_16k)

            if segment is not None:
                self.segments_captured += 1
                duration = len(segment) / self.sample_rate
                log.info(f"📦 Speech segment #{self.segments_captured}: {duration:.2f}s ({reason})")
                return segment, reason

            return None, None
    
    def capture_speech_segment(self, 
                               max_wait_s: float = 60.0,
                               require_speech: bool = True) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Capture a complete speech segment (blocking).
        
        Processes audio frames until a complete speech segment is captured
        or timeout is reached.
        
        Args:
            max_wait_s: Maximum time to wait for speech (default 60s)
            require_speech: If True, raise TimeoutError if no speech detected
        
        Returns:
            (segment, end_reason) tuple where:
                segment: Speech audio (np.ndarray) or None if timeout
                end_reason: "natural_pause", "timeout", or None
        
        Raises:
            TimeoutError: If no speech detected within max_wait_s and require_speech=True
        """
        if not self.listening:
            raise RuntimeError("Pipeline not listening - call start_listening() first")
        
        if self.assistant_speaking:
            raise RuntimeError("Cannot capture while assistant is speaking")
        
        log.info(f"Waiting for speech (max {max_wait_s:.0f}s)...")
        
        start_time = time.time()
        frames_processed = 0
        
        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > max_wait_s:
                if require_speech:
                    raise TimeoutError(f"No speech detected within {max_wait_s:.0f}s")
                else:
                    log.warning(f"Timeout waiting for speech ({elapsed:.1f}s)")
                    return None, None
            
            # Get frame from microphone with blocking timeout
            # This prevents tight loop and waits for frames to arrive
            frame = self.capture.get_frame(timeout=0.1)
            if frame is None:
                # Still no frame after timeout - log occasionally, not every time
                if frames_processed % 50 == 0:
                    log.debug(f"Waiting for audio frames... ({frames_processed} attempts, {elapsed:.1f}s)")
                continue
            
            frames_processed += 1
            
            # Process frame
            segment, reason = self.process_frame(frame)
            
            if segment is not None:
                log.info(f"✅ Speech captured after {elapsed:.1f}s ({frames_processed} frames)")
                return segment, reason
            
            # Periodic progress update
            if frames_processed % 100 == 0:
                log.info(f"Processed {frames_processed} frames ({elapsed:.1f}s elapsed)...")
    
    def get_stats(self) -> dict:
        """
        Get pipeline statistics.
        
        Aggregates statistics from all components.
        
        Returns:
            Dictionary with comprehensive pipeline statistics
        """
        stats = {
            "sample_rate": self.sample_rate,
            "frame_size": self.frame_size,
            "timeout_s": self.timeout_s,
            "listening": self.listening,
            "assistant_speaking": self.assistant_speaking,
            "segments_captured": self.segments_captured,
            "interruptions_detected": self.interruptions_detected,
            "vad": self.vad.get_stats(),
            "speech_detector": self.speech_detector.get_stats(),
            "interruption_detector": self.interruption_detector.get_stats()
        }
        
        return stats
    
    def reset(self):
        """Reset all pipeline components."""
        log.info("Resetting pipeline...")
        
        self.vad.reset()
        self.speech_detector.reset()
        self.interruption_detector.reset()
        
        self.segments_captured = 0
        self.interruptions_detected = 0
        
        log.info("Pipeline reset complete")


    def to_vad_frame(self, frame):
        """Convert 44.1kHz frame → 16kHz 512-sample for Silero VAD"""
        # Stereo → mono
        if frame.ndim > 1:
            frame = np.mean(frame, axis=1)
        
        # Resample 44100 → 16000
        ratio = 44100 / 16000
        idx = (np.arange(512) * ratio).astype(int)
        idx = np.clip(idx, 0, len(frame)-1)
        return frame[idx]

    
    def __enter__(self):
        """Context manager entry."""
        self.start_listening()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_listening()
        return False


def test_pipeline():
    """Basic pipeline test with simulated audio."""
    print("Testing AudioFramePipeline...\n")
    
    # Test 1: Initialize
    print("Test 1: Initialize pipeline")
    pipeline = AudioFramePipeline(timeout_s=5.0)
    print(f"✅ Initialized\n")
    
    # Test 2: Stats
    print("Test 2: Get statistics")
    stats = pipeline.get_stats()
    print(f"  Stats: {stats}")
    print(f"✅ Statistics work\n")
    
    # Test 3: Context manager
    print("Test 3: Context manager")
    with AudioFramePipeline() as p:
        assert p.listening
        print("  ✅ Start listening in context")
    print("  ✅ Stop listening on exit\n")
    
    print("✅ All basic tests passed!")
    print("\nNote: Full integration testing requires microphone input.")
    print("Run with actual audio in Phase 2 integration testing.")


if __name__ == "__main__":
    test_pipeline()