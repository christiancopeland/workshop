"""
Workshop Phase 2: Wake Word Detection - PRODUCTION VERSION

FIXES APPLIED:
1. OpenWakeWord requires int16 audio, not float32 - added conversion
2. Added warmup period to prevent false positives during model initialization
3. Added cooldown period to prevent repeated triggers
4. Added process_frame() method for wake_pipeline.py compatibility
5. Auto-reset of model state after detection and cooldown exit

Interface:
    detector = WakeWordDetector()
    
    # For wake_pipeline.py integration:
    scores = detector.process_frame(audio_frame)  # Returns {model_name: score}
    
    # For direct use with 1280-sample chunks:
    detected = detector.detect(audio_chunk)  # Returns bool
"""

import numpy as np
import time
from typing import Optional, Dict
from logger import get_logger

try:
    from openwakeword.model import Model
    OPENWAKEWORD_AVAILABLE = True
except ImportError:
    OPENWAKEWORD_AVAILABLE = False

log = get_logger("wake_word")


def float32_to_int16(audio_float: np.ndarray) -> np.ndarray:
    """
    Convert float32 audio [-1, 1] to int16 [-32768, 32767].
    
    CRITICAL: OpenWakeWord expects 16-bit signed integer PCM audio,
    not normalized float32. This conversion is required.
    
    Args:
        audio_float: Audio samples as float32 in [-1, 1] range
        
    Returns:
        Audio samples as int16 in [-32768, 32767] range
    """
    audio_clipped = np.clip(audio_float, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)
    return audio_int16


class WakeWordDetector:
    """
    Wake word detection using OpenWakeWord with proper state management.
    
    Key features:
    - Warmup period: Ignores detections for first N seconds (model stabilization)
    - Cooldown: Prevents repeated triggers after detection
    - Auto-reset: Clears model state after detection
    - Accepts float32 audio, converts to int16 internally
    - process_frame() for wake_pipeline.py integration
    - detect() for direct 1280-sample chunk processing
    
    Example (wake_pipeline.py integration):
        detector = WakeWordDetector()
        
        for frame in audio_stream:  # Variable size frames
            scores = detector.process_frame(frame)
            if scores and scores.get('hey_jarvis', 0) > 0.5:
                print("Wake word detected!")
    
    Example (direct use):
        detector = WakeWordDetector(model_name="alexa", threshold=0.5)
        buffer = WakeWordBuffer(detector)
        
        for frame in audio_stream:  # 512-sample frames
            if buffer.add_frame(frame):
                print("Wake word detected!")
    """
    
    def __init__(self,
                 model_name: str = "hey_jarvis",
                 threshold: float = 0.5,
                 sample_rate: int = 16000,
                 cooldown_seconds: float = 2.0,
                 warmup_seconds: float = 1.5):
        """
        Initialize wake word detector.
        
        Args:
            model_name: OpenWakeWord model ("alexa", "hey_jarvis", "hey_mycroft")
            threshold: Detection confidence threshold (0.0-1.0)
            sample_rate: Must be 16000 (OpenWakeWord requirement)
            cooldown_seconds: Minimum time between detections
            warmup_seconds: Ignore detections for this long after start/reset
        """
        if not OPENWAKEWORD_AVAILABLE:
            raise ImportError("openwakeword not installed. Run: pip install openwakeword")

        try:
            import openwakeword
            openwakeword.utils.download_models()
            log.info("Pre-trained models downloaded/verified")
        except Exception as e:
            log.warning(f"Could not download models: {e}")
        
        if sample_rate != 16000:
            raise ValueError("OpenWakeWord requires 16kHz audio")
        
        self.model_name = model_name
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.cooldown_seconds = cooldown_seconds
        self.warmup_seconds = warmup_seconds
        
        # OpenWakeWord processes 80ms chunks (1280 samples @ 16kHz)
        self.chunk_size = 1280
        
        # Internal buffer for process_frame() - bridges variable-size frames to 1280 chunks
        self._frame_buffer = np.array([], dtype=np.float32)
        
        # Initialize model
        log.info(f"Loading OpenWakeWord model: {model_name}")
        self.model = Model(wakeword_models=[model_name])
        
        # Discover prediction key (model name in predictions dict)
        test_audio = np.zeros(self.chunk_size, dtype=np.int16)
        test_prediction = self.model.predict(test_audio)
        
        if test_prediction:
            self.prediction_key = list(test_prediction.keys())[0]
            log.info(f"Discovered prediction key: '{self.prediction_key}'")
        else:
            self.prediction_key = model_name
            log.warning(f"Could not discover prediction key, using model_name: {model_name}")
        
        # State tracking
        self._start_time = time.time()
        self._last_detection_time = 0.0
        self._in_cooldown = False
        self._in_warmup = True
        self._last_score = 0.0
        self._last_prediction = {}
        
        # Statistics
        self.detections = 0
        self.frames_processed = 0
        self.max_score_seen = 0.0
        
        # Reset model to clear any state from test prediction
        self.model.reset()
        
        log.info(f"WakeWordDetector ready: model={model_name}, key={self.prediction_key}, "
                f"threshold={threshold}, cooldown={cooldown_seconds}s, warmup={warmup_seconds}s")
    
    def process_frame(self, audio: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Process audio frame and return prediction scores.
        
        This method is designed for wake_pipeline.py integration.
        It handles variable-size input frames by buffering internally.
        
        Args:
            audio: Audio samples (any size, typically 512 from audio pipeline)
                   Accepts float32 [-1, 1] or int16 [-32768, 32767]
            
        Returns:
            Dict of {model_name: score} if enough audio accumulated for prediction,
            None otherwise. Score respects warmup/cooldown (returns 0.0 during those periods).
        """
        if audio is None or len(audio) == 0:
            return None
        
        # Ensure float32 for buffer consistency
        if audio.dtype != np.float32:
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32767.0
            else:
                audio = audio.astype(np.float32)
        
        # Add to buffer
        self._frame_buffer = np.append(self._frame_buffer, audio)
        
        # Check if we have enough for prediction
        if len(self._frame_buffer) < self.chunk_size:
            return None
        
        # Extract chunk and process
        chunk = self._frame_buffer[:self.chunk_size]
        self._frame_buffer = self._frame_buffer[self.chunk_size:]
        
        # Get raw prediction (handles int16 conversion internally)
        score = self._process_chunk(chunk)
        
        # Build return dict
        self._last_prediction = {self.prediction_key: score}
        return self._last_prediction
    
    def _process_chunk(self, audio: np.ndarray) -> float:
        """
        Internal: Process a 1280-sample chunk and return score.
        Handles warmup/cooldown state and model resets.
        """
        current_time = time.time()
        
        # Convert to int16 (OpenWakeWord requirement)
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            audio_int16 = float32_to_int16(audio)
        elif audio.dtype == np.int16:
            audio_int16 = audio
        else:
            audio_int16 = float32_to_int16(audio.astype(np.float32))
        
        # Always feed audio to model (maintains continuity)
        prediction = self.model.predict(audio_int16)
        raw_score = prediction.get(self.prediction_key, 0.0)
        
        self.frames_processed += 1
        
        # Track max score for diagnostics
        if raw_score > self.max_score_seen:
            self.max_score_seen = raw_score
        
        # Check warmup period - return 0 score but still process audio
        if self._in_warmup:
            if current_time - self._start_time >= self.warmup_seconds:
                self._in_warmup = False
                log.info("Warmup complete, detection enabled")
            else:
                return 0.0  # Suppress score during warmup
        
        # Check cooldown period
        if self._in_cooldown:
            if current_time - self._last_detection_time >= self.cooldown_seconds:
                self._in_cooldown = False
                # Reset model when exiting cooldown to clear accumulated state
                self.model.reset()
                log.debug("Cooldown expired, model reset")
            else:
                return 0.0  # Suppress score during cooldown
        
        # Store score
        self._last_score = raw_score
        
        # Check for detection (for auto-reset)
        if raw_score >= self.threshold:
            self.detections += 1
            self._last_detection_time = current_time
            self._in_cooldown = True
            
            log.info(f"🎤 Wake word detected! (score={raw_score:.3f})")
            
            # Reset model state to prevent re-triggering
            self.model.reset()
        
        return raw_score
    
    def detect(self, audio: np.ndarray) -> bool:
        """
        Process audio chunk and check for wake word.
        
        This method is for direct use with exactly 1280-sample chunks.
        Use process_frame() for variable-size frames.
        
        Args:
            audio: Audio samples (must be 1280 samples = 80ms @ 16kHz)
                   Accepts float32 [-1, 1] or int16 [-32768, 32767]
            
        Returns:
            True if wake word detected above threshold
        """
        if len(audio) != self.chunk_size:
            log.warning(f"Audio chunk size {len(audio)} != expected {self.chunk_size}")
            return False
        
        score = self._process_chunk(audio)
        return score >= self.threshold
    
    def get_score(self) -> float:
        """Get the most recent prediction score."""
        return self._last_score
    
    def get_prediction(self) -> Dict[str, float]:
        """Get the most recent prediction dict."""
        return self._last_prediction
    
    def reset(self):
        """Full reset - clears model state, buffer, and restarts warmup."""
        self.model.reset()
        self._frame_buffer = np.array([], dtype=np.float32)
        self._start_time = time.time()
        self._in_warmup = True
        self._in_cooldown = False
        self._last_score = 0.0
        self._last_prediction = {}
        log.debug("Full detector reset")
    
    def get_stats(self) -> dict:
        """Get detection statistics."""
        return {
            "model": self.model_name,
            "prediction_key": self.prediction_key,
            "threshold": self.threshold,
            "in_warmup": self._in_warmup,
            "in_cooldown": self._in_cooldown,
            "last_score": self._last_score,
            "frames_processed": self.frames_processed,
            "detections": self.detections,
            "max_score_seen": self.max_score_seen,
            "buffer_size": len(self._frame_buffer)
        }


class WakeWordBuffer:
    """
    Accumulates audio frames for wake word detection.
    Bridges variable-size frames (e.g., 512 samples) to 1280-sample chunks.
    
    Example:
        detector = WakeWordDetector()
        buffer = WakeWordBuffer(detector)
        
        for frame in audio_stream:  # 512 samples, float32
            if buffer.add_frame(frame):
                print("Wake word detected!")
    """
    
    def __init__(self, detector: WakeWordDetector):
        """
        Initialize buffer.
        
        Args:
            detector: WakeWordDetector instance
        """
        self.detector = detector
        self.buffer = np.array([], dtype=np.float32)
        self.chunk_size = detector.chunk_size
    
    def add_frame(self, audio: np.ndarray) -> bool:
        """
        Add audio frame to buffer and detect when ready.
        
        Args:
            audio: Audio samples (any size, typically 512, float32)
            
        Returns:
            True if wake word detected
        """
        # Ensure float32 for consistent buffer
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Add to buffer
        self.buffer = np.append(self.buffer, audio)
        
        # Check if we have enough for detection
        if len(self.buffer) >= self.chunk_size:
            # Extract chunk
            chunk = self.buffer[:self.chunk_size]
            
            # Detect (handles int16 conversion internally)
            detected = self.detector.detect(chunk)
            
            # Keep remaining samples
            self.buffer = self.buffer[self.chunk_size:]
            
            return detected
        
        return False
    
    def get_score(self) -> float:
        """Get most recent score from detector."""
        return self.detector.get_score()
    
    def reset(self):
        """Clear buffer and reset detector."""
        self.buffer = np.array([], dtype=np.float32)
        self.detector.reset()


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_unit():
    """Unit tests for WakeWordDetector."""
    print("=" * 70)
    print("UNIT TESTS: WakeWordDetector")
    print("=" * 70)
    
    if not OPENWAKEWORD_AVAILABLE:
        print("❌ OpenWakeWord not installed")
        print("Run: pip install openwakeword")
        return False
    
    # Test 1: Initialize
    print("\n[Test 1] Initialize detector")
    try:
        detector = WakeWordDetector(
            model_name="alexa",
            threshold=0.5,
            warmup_seconds=0.5,
            cooldown_seconds=1.0
        )
        print(f"✅ Initialized: model={detector.model_name}, key={detector.prediction_key}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Test 2: Verify int16 conversion
    print("\n[Test 2] Verify int16 conversion")
    test_float = np.array([0.0, 0.5, 1.0, -0.5, -1.0], dtype=np.float32)
    test_int16 = float32_to_int16(test_float)
    expected = np.array([0, 16383, 32767, -16383, -32767], dtype=np.int16)
    
    print(f"  Input float32:  {test_float}")
    print(f"  Output int16:   {test_int16}")
    print(f"  Expected:       {expected}")
    
    assert test_int16.dtype == np.int16, "Wrong dtype!"
    assert np.allclose(test_int16, expected, atol=1), "Conversion mismatch!"
    print("✅ int16 conversion correct")
    
    # Test 3: process_frame() with variable-size input
    print("\n[Test 3] process_frame() with variable-size input")
    detector.reset()
    
    # Send 512-sample frames (should need 3 to get 1280)
    results = []
    for i in range(5):
        frame = np.zeros(512, dtype=np.float32)
        result = detector.process_frame(frame)
        results.append(result)
        print(f"  Frame {i}: result={result}")
    
    # Should have some None (buffering) and some dicts (predictions)
    none_count = sum(1 for r in results if r is None)
    dict_count = sum(1 for r in results if r is not None)
    print(f"  None results: {none_count}, Dict results: {dict_count}")
    assert dict_count > 0, "Should have at least one prediction!"
    print("✅ process_frame() buffering works")
    
    # Test 4: detect() with exact chunk size
    print("\n[Test 4] detect() with exact chunk size")
    detector.reset()
    time.sleep(0.6)  # Wait for warmup
    
    silence = np.zeros(1280, dtype=np.float32)
    detected = detector.detect(silence)
    print(f"  Silence detected: {detected} (should be False)")
    assert detected == False, "False positive on silence!"
    print("✅ detect() works with silence")
    
    # Test 5: Stats
    print("\n[Test 5] Stats")
    stats = detector.get_stats()
    print(f"  {stats}")
    assert "model" in stats, "Missing model in stats"
    assert "prediction_key" in stats, "Missing prediction_key in stats"
    print("✅ Stats working")
    
    # Test 6: WakeWordBuffer
    print("\n[Test 6] WakeWordBuffer")
    buffer = WakeWordBuffer(detector)
    
    for i in range(5):
        frame = np.random.randn(512).astype(np.float32) * 0.1
        detected = buffer.add_frame(frame)
        score = buffer.get_score()
        print(f"  Frame {i}: buffer_len={len(buffer.buffer)}, detected={detected}, score={score:.4f}")
    
    print("✅ WakeWordBuffer works")
    
    print("\n" + "=" * 70)
    print("✅ ALL UNIT TESTS PASSED!")
    print("=" * 70)
    return True


def test_live():
    """Live microphone test with real-time score visualization."""
    import sys
    
    print("=" * 70)
    print("LIVE Wake Word Detection Test")
    print("=" * 70)
    
    if not OPENWAKEWORD_AVAILABLE:
        print("❌ OpenWakeWord not installed")
        return
    
    try:
        from audio_realtime import AudioStream
    except ImportError:
        print("❌ audio_realtime.py not found")
        return
    
    # Parse threshold from command line
    threshold = 0.5
    if len(sys.argv) > 2:
        try:
            threshold = float(sys.argv[2])
        except ValueError:
            pass
    
    print("\nInitializing...")
    
    detector = WakeWordDetector(
        model_name="alexa",
        threshold=threshold,
        cooldown_seconds=2.0,
        warmup_seconds=1.5
    )
    buffer = WakeWordBuffer(detector)
    
    stream = AudioStream(
        sample_rate=16000,
        device_id=None,
        hardware_sample_rate=44100
    )
    
    if not stream.start():
        print("❌ Failed to start audio stream")
        return
    
    print(f"\n🎤 Model: {detector.model_name} (key: {detector.prediction_key})")
    print(f"   Threshold: {threshold}")
    print(f"   Warmup: {detector.warmup_seconds}s")
    print(f"   Cooldown: {detector.cooldown_seconds}s")
    print()
    print("   Say 'ALEXA' clearly after warmup.")
    print("   Press Ctrl+C to stop.")
    print()
    print("-" * 70)
    
    try:
        frame_count = 0
        display_interval = 5
        
        while True:
            frame = stream.get_frame(timeout=0.1)
            
            if frame is not None:
                frame_count += 1
                
                detected = buffer.add_frame(frame)
                score = buffer.get_score()
                
                if frame_count % display_interval == 0:
                    bar_filled = int(score * 50)
                    bar = "█" * bar_filled + "░" * (50 - bar_filled)
                    
                    if detector._in_warmup:
                        status = "WARMUP"
                    elif detector._in_cooldown:
                        status = "COOLDOWN"
                    else:
                        status = "listening"
                    
                    print(f"\r  [{bar}] {score:.3f} ({status:10s})", end="", flush=True)
                
                if detected:
                    print()
                    print()
                    print("  " + "🎉" * 20)
                    print("     WAKE WORD DETECTED!")
                    print("  " + "🎉" * 20)
                    print(f"     Score: {score:.3f}")
                    print()
                    
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        stream.stop()
    
    print("\nFinal stats:")
    print(f"  {detector.get_stats()}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--live":
            test_live()
        elif sys.argv[1] == "--test":
            test_unit()
        else:
            print("Usage:")
            print("  python wake_word.py --test           # Run unit tests")
            print("  python wake_word.py --live [thresh]  # Live microphone test")
    else:
        # Default: run unit tests
        test_unit()