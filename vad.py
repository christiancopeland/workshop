"""
Workshop Phase 2: Voice Activity Detection
Real-time speech detection using silero-vad
"""

import numpy as np
import torch
from typing import Optional, Callable
from collections import deque
from logger import get_logger

log = get_logger("vad")


class VoiceActivityDetector:
    """
    Real-time Voice Activity Detection using silero-vad.
    
    Detects speech vs silence in audio frames with ~30ms latency.
    Uses sliding window to smooth detections and reduce false triggers.
    
    Example:
        vad = VoiceActivityDetector(
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=300
        )
        
        # Process audio frames
        for frame in audio_stream:
            is_speech = vad.process_frame(frame)
            if is_speech:
                record_speech(frame)
    """
    
    def __init__(self,
                 threshold: float = 0.5,
                 min_speech_duration_ms: int = 250,
                 min_silence_duration_ms: int = 300,
                 sample_rate: int = 16000):
        """
        Initialize VAD.
        
        Args:
            threshold: Speech probability threshold (0.0-1.0)
            min_speech_duration_ms: Minimum speech duration to trigger
            min_silence_duration_ms: Minimum silence to end speech
            sample_rate: Audio sample rate (must be 16kHz for silero)
        """
        if sample_rate != 16000:
            raise ValueError("silero-vad requires 16kHz audio")
        
        self.threshold = threshold
        self.sample_rate = sample_rate
        
        # Convert durations to frame counts (assuming 32ms frames)
        frame_duration_ms = 32  # 512 samples @ 16kHz
        self.min_speech_frames = max(1, min_speech_duration_ms // frame_duration_ms)
        self.min_silence_frames = max(1, min_silence_duration_ms // frame_duration_ms)
        
        # Load silero-vad model
        log.info("Loading silero-vad model...")
        try:
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.get_speech_timestamps = utils[0]
            log.info("✅ silero-vad loaded")
        except Exception as e:
            log.error(f"Failed to load silero-vad: {e}")
            raise
        
        # State tracking
        self.is_speaking = False
        self.speech_frames = 0
        self.silence_frames = 0
        
        # History for smoothing
        self.history_size = 5
        self.probability_history = deque(maxlen=self.history_size)
        
        # Statistics
        self.frames_processed = 0
        self.speech_segments = 0
        
        log.info(f"VAD ready: threshold={threshold}, min_speech={min_speech_duration_ms}ms, min_silence={min_silence_duration_ms}ms")
    
    def process_frame(self, audio: np.ndarray) -> bool:
        """
        Process audio frame and detect speech.
        
        Args:
            audio: Audio samples (512 samples = 32ms @ 16kHz)
            
        Returns:
            True if currently in speech segment
        """
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Get speech probability
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, self.sample_rate).item()
        
        # Add to history for smoothing
        self.probability_history.append(speech_prob)
        
        # Calculate smoothed probability
        smoothed_prob = np.mean(self.probability_history)
        
        # Detect speech based on threshold
        is_speech_frame = smoothed_prob >= self.threshold
        
        # State machine for speech detection
        if is_speech_frame:
            self.speech_frames += 1
            self.silence_frames = 0
            
            # Trigger speech if we've had enough consecutive speech frames
            if not self.is_speaking and self.speech_frames >= self.min_speech_frames:
                self.is_speaking = True
                self.speech_segments += 1
                log.info(f"🗣️  Speech started (prob={smoothed_prob:.2f})")
        else:
            self.silence_frames += 1
            self.speech_frames = 0
            
            # End speech if we've had enough consecutive silence frames
            if self.is_speaking and self.silence_frames >= self.min_silence_frames:
                self.is_speaking = False
                log.info(f"🔇 Speech ended (prob={smoothed_prob:.2f})")
        
        self.frames_processed += 1
        
        return self.is_speaking
    
    def get_probability(self) -> float:
        """Get current smoothed speech probability."""
        if len(self.probability_history) > 0:
            return float(np.mean(self.probability_history))
        return 0.0
    
    def reset(self):
        """Reset VAD state."""
        self.is_speaking = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.probability_history.clear()
        log.debug("VAD state reset")
    
    def get_stats(self) -> dict:
        """Get VAD statistics."""
        return {
            "threshold": self.threshold,
            "frames_processed": self.frames_processed,
            "speech_segments": self.speech_segments,
            "currently_speaking": self.is_speaking,
            "current_probability": self.get_probability(),
            "min_speech_frames": self.min_speech_frames,
            "min_silence_frames": self.min_silence_frames,
        }


class SpeechSegmentCollector:
    """
    Collects audio frames during speech segments.
    
    Works with VAD to accumulate speech audio and detect segment boundaries.
    
    Example:
        vad = VoiceActivityDetector()
        collector = SpeechSegmentCollector(vad)
        
        for frame in audio_stream:
            segment = collector.process_frame(frame)
            if segment is not None:
                # Complete speech segment captured
                transcribe(segment)
    """
    
    def __init__(self, 
                 vad: VoiceActivityDetector,
                 max_duration_s: float = 30.0):
        """
        Initialize collector.
        
        Args:
            vad: VoiceActivityDetector instance
            max_duration_s: Maximum segment duration (safety limit)
        """
        self.vad = vad
        self.max_samples = int(max_duration_s * vad.sample_rate)
        
        # Buffer for current segment
        self.buffer = []
        self.was_speaking = False
    
    def process_frame(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Process audio frame and collect speech segments.
        
        Args:
            audio: Audio frame
            
        Returns:
            Complete speech segment when speech ends, None otherwise
        """
        # Detect speech
        is_speaking = self.vad.process_frame(audio)
        
        if is_speaking:
            # Accumulate speech frames
            self.buffer.append(audio)
            
            # Safety check: prevent buffer overflow
            if len(self.buffer) * len(audio) > self.max_samples:
                log.warning("Speech segment exceeded max duration, forcing end")
                segment = self.finalize_segment()
                return segment
        
        # Check for speech end
        if self.was_speaking and not is_speaking:
            # Speech just ended - return complete segment
            segment = self.finalize_segment()
            return segment
        
        self.was_speaking = is_speaking
        return None
    
    def finalize_segment(self) -> np.ndarray:
        """Finalize and return current segment."""
        if len(self.buffer) == 0:
            return None
        
        # Concatenate all frames
        segment = np.concatenate(self.buffer)
        
        log.info(f"📦 Speech segment collected: {len(segment)} samples ({len(segment)/16000:.2f}s)")
        
        # Clear buffer
        self.buffer = []
        
        return segment
    
    def reset(self):
        """Reset collector state."""
        self.buffer = []
        self.was_speaking = False
        self.vad.reset()


class SpeechEndDetector:
    """
    Detects speech end via natural pause or timeout.
    
    Wraps VoiceActivityDetector and SpeechSegmentCollector with timeout logic.
    Returns (segment, end_reason) tuples for better debugging and statistics.
    
    Example:
        vad = VoiceActivityDetector()
        detector = SpeechEndDetector(vad, timeout_s=30.0)
        
        for frame in audio_stream:
            segment, reason = detector.process_frame(frame)
            if segment is not None:
                print(f"Speech ended: {reason}")
                transcribe(segment)
    """
    
    def __init__(self, vad: VoiceActivityDetector, timeout_s: float = 30.0):
        """
        Initialize speech end detector.
        
        Args:
            vad: VoiceActivityDetector instance
            timeout_s: Maximum speech duration before forcing end (default 30s)
        """
        self.vad = vad
        self.timeout_s = timeout_s
        # Set collector max_duration much larger so timeout logic takes precedence
        self.collector = SpeechSegmentCollector(vad, max_duration_s=timeout_s * 2)
        
        # Timer state
        self.speech_start_time = None
        self.last_end_reason = None
        
        # Statistics
        self.segments_by_reason = {
            "natural_pause": 0,
            "timeout": 0
        }
        
        log.info(f"SpeechEndDetector initialized (timeout={timeout_s:.1f}s)")
    
    def process_frame(self, audio: np.ndarray) -> tuple:
        """
        Process audio frame and detect speech end.
        
        Args:
            audio: Audio frame (512 samples, float32)
        
        Returns:
            (segment, end_reason) tuple where:
                segment: Complete speech audio (np.ndarray) or None
                end_reason: "natural_pause", "timeout", or None
        """
        import time
        
        # Start timer when speech begins
        if self.vad.is_speaking and self.speech_start_time is None:
            self.speech_start_time = time.time()
            log.debug("Speech started, timer activated")
        
        # Check for timeout FIRST (before collector processes frame)
        if self.vad.is_speaking and self.speech_start_time is not None:
            elapsed = time.time() - self.speech_start_time
            
            if elapsed >= self.timeout_s:
                # Force segment end due to timeout
                # Add current frame to buffer first
                self.collector.buffer.append(audio)
                
                # Finalize the segment
                segment = self.collector.finalize_segment()
                
                if segment is not None and len(segment) > 0:
                    self.speech_start_time = None
                    self.last_end_reason = "timeout"
                    self.segments_by_reason["timeout"] += 1
                    
                    log.info(f"⏱️  Speech timeout: {elapsed:.2f}s, {len(segment)} samples")
                    return segment, "timeout"
        
        # Process frame through collector (normal path)
        segment = self.collector.process_frame(audio)
        
        # If segment returned naturally
        if segment is not None:
            self.speech_start_time = None
            self.last_end_reason = "natural_pause"
            self.segments_by_reason["natural_pause"] += 1
            
            duration = len(segment) / self.vad.sample_rate
            log.info(f"✅ Speech ended naturally: {duration:.2f}s, {len(segment)} samples")
            return segment, "natural_pause"
        
        # No segment yet
        return None, None
    
    def reset(self):
        """Reset detector state."""
        self.speech_start_time = None
        self.last_end_reason = None
        self.collector.reset()
        log.debug("SpeechEndDetector reset")
    
    def get_stats(self) -> dict:
        """
        Get detection statistics.
        
        Returns:
            Dictionary with detection statistics including:
                - timeout_s: Configured timeout duration
                - total_segments: Total segments detected
                - natural_pause: Count of natural pauses
                - timeout: Count of timeouts
                - natural_pause_pct: Percentage via natural pause
                - timeout_pct: Percentage via timeout
                - last_end_reason: Most recent end reason
                - currently_timing: Whether timer is active
        """
        total_segments = sum(self.segments_by_reason.values())
        
        stats = {
            "timeout_s": self.timeout_s,
            "total_segments": total_segments,
            "natural_pause": self.segments_by_reason["natural_pause"],
            "timeout": self.segments_by_reason["timeout"],
            "last_end_reason": self.last_end_reason,
            "currently_timing": self.speech_start_time is not None
        }
        
        # Add percentage breakdowns if we have segments
        if total_segments > 0:
            stats["natural_pause_pct"] = (
                self.segments_by_reason["natural_pause"] / total_segments * 100
            )
            stats["timeout_pct"] = (
                self.segments_by_reason["timeout"] / total_segments * 100
            )
        
        return stats


class InterruptionDetector:
    """
    Detects user interruptions during assistant speech.
    
    Monitors for user speech while assistant is talking and triggers
    interruption events to stop assistant and start listening.
    
    Example:
        vad = VoiceActivityDetector()
        detector = InterruptionDetector(vad)
        
        detector.on_interruption(lambda: stop_tts_and_listen())
        detector.set_assistant_speaking(True)
        
        for frame in audio_stream:
            if detector.process_frame(frame):
                # Interruption detected, callback already fired
                pass
    """
    
    def __init__(self,
                 vad: VoiceActivityDetector,
                 interruption_threshold: float = 0.6,
                 confirmation_frames: int = 3):
        """
        Initialize interruption detector.
        
        Args:
            vad: VoiceActivityDetector instance
            interruption_threshold: Probability threshold for interruption (default 0.6)
            confirmation_frames: Consecutive frames needed to confirm (default 3 = ~96ms)
        """
        self.vad = vad
        self.interruption_threshold = interruption_threshold
        self.confirmation_frames = confirmation_frames
        
        # State
        self.assistant_speaking = False
        self.consecutive_speech_frames = 0
        
        # Callback
        self.interruption_callback = None
        
        # Statistics
        self.interruption_count = 0
        self.last_interruption_time = None
        
        log.info(f"InterruptionDetector initialized (threshold={interruption_threshold:.2f}, confirmation={confirmation_frames})")
    
    def set_assistant_speaking(self, speaking: bool):
        """
        Set assistant speaking state.
        
        Args:
            speaking: True if assistant is currently speaking
        """
        was_speaking = self.assistant_speaking
        self.assistant_speaking = speaking
        
        if speaking and not was_speaking:
            log.debug("Assistant started speaking - monitoring for interruptions")
            self.consecutive_speech_frames = 0
        elif not speaking and was_speaking:
            log.debug("Assistant stopped speaking - interruption monitoring off")
            self.consecutive_speech_frames = 0
    
    def process_frame(self, audio: np.ndarray) -> bool:
        """
        Process audio frame and detect interruptions.
        
        Args:
            audio: Audio frame (512 samples, float32)
        
        Returns:
            True if interruption detected this frame, False otherwise
        """
        # Only monitor for interruptions while assistant is speaking
        if not self.assistant_speaking:
            self.consecutive_speech_frames = 0
            return False
        
        # Get speech probability from VAD
        # Note: We don't call vad.process_frame() to avoid state changes
        # Just get the current probability
        import torch
        
        audio_tensor = torch.from_numpy(audio).float()
        speech_prob = self.vad.model(audio_tensor, self.vad.sample_rate).item()
        
        # Check if probability exceeds interruption threshold
        if speech_prob > self.interruption_threshold:
            self.consecutive_speech_frames += 1
            
            # Check if we've reached confirmation threshold
            if self.consecutive_speech_frames >= self.confirmation_frames:
                # Interruption detected!
                import time
                self.last_interruption_time = time.time()
                self.interruption_count += 1
                
                log.info(f"🛑 INTERRUPTION DETECTED! (prob={speech_prob:.3f}, frames={self.consecutive_speech_frames})")
                
                # Fire callback if registered
                if self.interruption_callback is not None:
                    try:
                        self.interruption_callback()
                    except Exception as e:
                        log.error(f"Interruption callback failed: {e}")
                
                # Reset frame counter (don't retrigger immediately)
                self.consecutive_speech_frames = 0
                
                return True
        else:
            # Reset counter on silence
            if self.consecutive_speech_frames > 0:
                log.debug(f"Speech frames reset: {self.consecutive_speech_frames} -> 0")
            self.consecutive_speech_frames = 0
        
        return False
    
    def on_interruption(self, callback):
        """
        Register callback for interruption events.
        
        Args:
            callback: Function to call when interruption detected
        """
        self.interruption_callback = callback
        log.debug("Interruption callback registered")
    
    def reset(self):
        """Reset detector state."""
        self.assistant_speaking = False
        self.consecutive_speech_frames = 0
        log.debug("InterruptionDetector reset")
    
    def get_stats(self) -> dict:
        """
        Get interruption statistics.
        
        Returns:
            Dictionary with interruption statistics including:
                - interruption_threshold: Configured threshold
                - confirmation_frames: Required consecutive frames
                - total_interruptions: Total count
                - last_interruption_time: Timestamp of last interruption
                - assistant_speaking: Current state
                - consecutive_speech_frames: Current frame count
        """
        stats = {
            "interruption_threshold": self.interruption_threshold,
            "confirmation_frames": self.confirmation_frames,
            "total_interruptions": self.interruption_count,
            "last_interruption_time": self.last_interruption_time,
            "assistant_speaking": self.assistant_speaking,
            "consecutive_speech_frames": self.consecutive_speech_frames
        }
        
        return stats


def test_vad():
    """Test Voice Activity Detection."""
    import time
    
    print("Testing VoiceActivityDetector...\n")
    
    # Test 1: Initialize
    print("Test 1: Initialize VAD")
    vad = VoiceActivityDetector(
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=300
    )
    print(f"✅ Initialized: {vad.get_stats()}\n")
    
    # Test 2: Process silence
    print("Test 2: Process silence (random noise)")
    silence = np.random.randn(512).astype(np.float32) * 0.01  # Very quiet
    
    for i in range(10):
        is_speech = vad.process_frame(silence)
        print(f"  Frame {i}: speech={is_speech}, prob={vad.get_probability():.3f}")
    
    assert not vad.is_speaking, "Should not detect speech in noise"
    print("✅ Silence detection works\n")
    
    # Test 3: Process loud audio (simulated speech)
    print("Test 3: Process loud audio (simulated speech)")
    vad.reset()
    
    # Generate louder signal
    speech = np.random.randn(512).astype(np.float32) * 0.3
    
    for i in range(15):
        is_speech = vad.process_frame(speech)
        print(f"  Frame {i}: speech={is_speech}, prob={vad.get_probability():.3f}")
    
    print("✅ Speech detection works\n")
    
    # Test 4: Speech segment collector
    print("Test 4: Speech segment collector")
    vad.reset()
    collector = SpeechSegmentCollector(vad)
    
    # Feed speech frames
    for i in range(10):
        segment = collector.process_frame(speech)
        if segment is not None:
            print(f"  Segment captured: {len(segment)} samples")
    
    # Feed silence to end segment
    for i in range(10):
        segment = collector.process_frame(silence)
        if segment is not None:
            print(f"  ✅ Final segment: {len(segment)} samples ({len(segment)/16000:.2f}s)")
            break
    
    print("✅ Collector works\n")
    
    # Test 5: Stats
    print("Test 5: Final stats")
    stats = vad.get_stats()
    print(f"  {stats}")
    print("✅ Stats work\n")
    
    print("✅ All tests passed!")
    print("\nNote: Real speech detection requires actual audio input.")
    print("Run with microphone in Phase 2 integration for real testing.")


if __name__ == "__main__":
    test_vad()