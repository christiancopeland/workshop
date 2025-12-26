"""
Workshop Phase 2: Real-Time Audio Stream
Continuous audio capture with non-blocking frame access
"""

import numpy as np
import sounddevice as sd
from queue import Queue, Empty
from typing import Optional
from scipy import signal
from logger import get_logger

log = get_logger("audio")


class AudioStream:
    """
    Continuous audio capture from microphone with non-blocking frame access.
    
    Captures audio in 32ms frames (512 samples @ 16kHz) and provides
    thread-safe queue access for consumers (VAD, wake word, STT).
    
    Example:
        stream = AudioStream()
        stream.start()
        
        while running:
            frame = stream.get_frame(timeout=0.1)
            if frame is not None:
                # Process audio frame
                process(frame)
        
        stream.stop()
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 channels: int = 1,
                 frame_size: int = 512,
                 device_id: Optional[int] = None,
                 hardware_sample_rate: Optional[int] = None):
        """
        Initialize audio stream with automatic resampling.
        
        Args:
            sample_rate: Target sample rate for output (16kHz for Whisper/VAD)
            channels: 1 (mono) or 2 (stereo) - mono recommended
            frame_size: Samples per frame at TARGET rate (512 = 32ms @ 16kHz)
            device_id: Specific audio device index (None = default)
            hardware_sample_rate: Native mic sample rate (None = auto-detect, or 44100 for Blue USB)
        
        Note: If hardware_sample_rate differs from sample_rate, automatic resampling will convert hardware rate to target rate.
        """
        self.target_rate = sample_rate  # What we output (16kHz)
        self.hardware_rate = hardware_sample_rate or sample_rate  # What mic provides (44.1kHz)
        self.channels = channels
        self.frame_size = frame_size  # At target rate
        self.device_id = device_id
        
        # Calculate hardware frame size
        if self.hardware_rate != self.target_rate:
            # Need more samples from hardware to downsample to target
            # Use ceiling to ensure we have ENOUGH samples
            ratio = self.hardware_rate / self.target_rate
            self.hardware_frame_size = int(np.ceil(frame_size * ratio))
            self.needs_resampling = True
            log.info(f"Resampling enabled: {self.hardware_rate}Hz → {self.target_rate}Hz")
            log.info(f"  Hardware frames: {self.hardware_frame_size} samples → Output: {frame_size} samples")
        else:
            self.hardware_frame_size = frame_size
            self.needs_resampling = False
        
        # Queue for audio frames (6.4 seconds buffer = 200 frames)
        # Larger buffer prevents overflow when processing temporarily slows
        self.frame_queue = Queue(maxsize=200)
        
        # Stream state
        self.stream = None
        self._running = False
        
        log.info(f"AudioStream initialized: HW={self.hardware_rate}Hz, Output={self.target_rate}Hz, "
                f"{channels}ch, {frame_size} samples/frame, device={device_id}")
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """
        Callback fired by sounddevice for each audio frame.
        
        Runs in separate thread - must be non-blocking and fast.
        
        Note: Input overflow warnings are normal and expected when processing
        temporarily falls behind. The queue buffering handles this gracefully.
        """
        # Copy audio data (indata is reused by sounddevice)
        audio_frame = indata.copy()
        
        # Convert stereo to mono if needed
        if audio_frame.ndim > 1 and audio_frame.shape[1] > 1:
            audio_frame = np.mean(audio_frame, axis=1)
        else:
            audio_frame = audio_frame.flatten()
        
        # Resample if needed (hardware rate → target rate)
        if self.needs_resampling:
            audio_frame = self._resample(audio_frame)
        
        # Push to queue (non-blocking)
        try:
            self.frame_queue.put_nowait(audio_frame)
        except:
            # Queue full - drop oldest frames to make room
            # This prevents buffer overflow when processing is slow
            dropped = 0
            while dropped < 3:  # Drop up to 3 old frames
                try:
                    self.frame_queue.get_nowait()
                    dropped += 1
                except:
                    break
            
            # Try again to add new frame
            try:
                self.frame_queue.put_nowait(audio_frame)
            except:
                pass  # Still full, drop this frame
    
    def _resample(self, audio_hw: np.ndarray) -> np.ndarray:
        """
        Downsample audio from hardware rate to target rate with anti-aliasing.
        
        Uses polyphase filtering to prevent aliasing and preserve spectral content.
        Normalizes output to prevent filter overshoot/ringing.
        """
        if len(audio_hw) == 0:
            return np.array([], dtype=np.float32)
        
        from scipy import signal
        
        # Downsample: 44100 -> 16000 = 160/441 ratio
        up = 160
        down = 441
        
        resampled = signal.resample_poly(audio_hw, up, down)
        
        # FIX: Normalize to prevent clipping from filter overshoot
        # Polyphase filters can ring beyond [-1, 1] even if input is within range
        peak = np.abs(resampled).max()
        if peak > 0.95:  # If approaching clipping
            resampled = resampled * (0.95 / peak)  # Scale to 95% of range
        
        # Ensure exact output length
        if len(resampled) > self.frame_size:
            resampled = resampled[:self.frame_size]
        elif len(resampled) < self.frame_size:
            resampled = np.pad(resampled, (0, self.frame_size - len(resampled)))
        
        return resampled.astype(np.float32)
    
    def start(self) -> bool:
        """
        Start continuous audio capture.
        
        Returns:
            True if stream started successfully, False otherwise
        """
        if self._running:
            log.warning("Stream already running")
            return True
        
        try:
            # Create input stream with hardware settings
            self.stream = sd.InputStream(
                samplerate=self.hardware_rate,  # Use hardware's native rate
                channels=self.channels,
                blocksize=self.hardware_frame_size,  # Larger blocks for hardware
                callback=self._audio_callback,
                dtype=np.float32,
                device=self.device_id  # Specific device or default
            )
            
            # Start stream
            self.stream.start()
            self._running = True
            
            log.info(f"✅ Audio stream started (device={self.device_id}, "
                    f"HW={self.hardware_rate}Hz, Output={self.target_rate}Hz)")
            return True
            
        except Exception as e:
            log.error(f"Failed to start audio stream: {e}")
            return False
    
    def stop(self):
        """Stop audio capture and cleanup."""
        if not self._running:
            return
        
        self._running = False
        
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                log.error(f"Error stopping stream: {e}")
            finally:
                self.stream = None
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        log.info("Audio stream stopped")
    
    def get_frame(self, timeout: float = 0.0) -> Optional[np.ndarray]:
        """
        Get next audio frame from queue.
        
        Args:
            timeout: Max seconds to wait for frame (0 = non-blocking)
        
        Returns:
            Audio frame as numpy array shape (frame_size,) or None if no frame available
        """
        if not self._running:
            return None
        
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return frame
        except Empty:
            return None

    
    @property
    def is_running(self) -> bool:
        """Check if stream is actively capturing."""
        return self._running
    
    def get_stats(self) -> dict:
        """Get stream statistics for debugging."""
        return {
            "running": self._running,
            "hardware_rate": self.hardware_rate,
            "target_rate": self.target_rate,
            "resampling": self.needs_resampling,
            "channels": self.channels,
            "frame_size": self.frame_size,
            "hardware_frame_size": self.hardware_frame_size,
            "device_id": self.device_id,
            "frame_duration_ms": (self.frame_size / self.target_rate) * 1000,
            "queue_size": self.frame_queue.qsize(),
            "queue_capacity": self.frame_queue.maxsize
        }
    
    def __enter__(self):
        """Context manager support."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.stop()

def find_device_by_name(name_pattern: str) -> Optional[int]:
        """
        Find audio device by name pattern.
        
        Args:
            name_pattern: String to search for in device name (case-insensitive)
            
        Returns:
            Device ID if found, None otherwise
        """
        devices = sd.query_devices()
        
        for i, device in enumerate(devices):
            device_name = device['name'].lower()
            if name_pattern.lower() in device_name and device['max_input_channels'] > 0:
                log.info(f"Found device matching '{name_pattern}': [{i}] {device['name']}")
                return i
        
        log.warning(f"No device found matching '{name_pattern}'")
        return None


def test_audio_stream():
    """Quick test of audio stream functionality."""
    import time
    
    print("Testing AudioStream...")
    
    stream = AudioStream()
    
    # Test 1: Start stream
    print("  Testing start...")
    assert stream.start() == True
    assert stream.is_running == True
    print("  ✅ Stream started")
    
    # Test 2: Capture frames
    print("  Testing frame capture (1 second)...")
    frames_captured = 0
    start_time = time.time()
    
    while time.time() - start_time < 1.0:
        frame = stream.get_frame(timeout=0.1)
        if frame is not None:
            frames_captured += 1
            assert frame.shape == (512,)
            assert frame.dtype == np.float32
    
    print(f"  ✅ Captured {frames_captured} frames")
    assert frames_captured >= 25  # Should get ~31 frames/second
    
    # Test 3: Non-blocking read
    print("  Testing non-blocking read...")
    start = time.time()
    frame = stream.get_frame(timeout=0)
    elapsed = time.time() - start
    assert elapsed < 0.01  # Should return immediately
    print("  ✅ Non-blocking read works")
    
    # Test 4: Stop stream
    print("  Testing stop...")
    stream.stop()
    assert stream.is_running == False
    assert stream.get_frame() is None
    print("  ✅ Stream stopped cleanly")
    
    # Test 5: Stats
    print("  Testing stats...")
    stats = stream.get_stats()
    print(f"    Stats: {stats}")
    assert stats["running"] == False
    print("  ✅ Stats working")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_audio_stream()


# Alias for backward compatibility
RealtimeAudioCapture = AudioStream