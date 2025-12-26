"""
Workshop Phase 2: Real-Time Audio Playback
Non-blocking playback with immediate interruption support
"""

import numpy as np
import sounddevice as sd
from queue import Queue, Empty
from threading import Event
from typing import Optional
from logger import get_logger

log = get_logger("audio_playback")


class AudioPlayback:
    """
    Non-blocking audio playback with interruption support.
    
    Plays audio chunks through speakers with minimal latency (~32ms).
    Supports immediate stop for user interruptions.
    
    Example:
        playback = AudioPlayback()
        playback.start()
        
        # Queue TTS audio
        playback.queue_audio(audio_chunk_1)
        playback.queue_audio(audio_chunk_2)
        
        # Wait for completion or interrupt
        playback.wait_until_done()
        # OR
        playback.stop()  # Immediate interrupt
        
        playback.stop()  # Cleanup
    """
    
    def __init__(self,
                 sample_rate: int = 16000,
                 channels: int = 1,
                 block_size: int = 512):
        """
        Initialize audio playback.
        
        Args:
            sample_rate: Audio sample rate (16kHz for speech)
            channels: 1 (mono) or 2 (stereo)
            block_size: Samples per block (512 = 32ms @ 16kHz)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        
        # Playback queue and buffer
        self.audio_queue = Queue()
        self.current_chunk = None
        self.chunk_position = 0
        
        # Stream state
        self.stream = None
        self._running = False
        self._playing = False
        
        # Synchronization
        self.done_event = Event()
        self.done_event.set()  # Start as done
        
        log.info(f"AudioPlayback initialized: {sample_rate}Hz, {channels}ch, {block_size} samples/block")
    
    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status):
        """
        Callback fired by sounddevice for each audio block.
        
        Pulls audio from queue and writes to output buffer.
        Runs in separate thread - must be non-blocking and fast.
        """
        if status:
            log.warning(f"Audio callback status: {status}")
        
        if not self._running:
            # Fill with silence
            outdata.fill(0)
            return
        
        # Fill output buffer
        samples_needed = frames
        output_position = 0
        
        while samples_needed > 0:
            # Get next chunk if needed
            if self.current_chunk is None or self.chunk_position >= len(self.current_chunk):
                try:
                    self.current_chunk = self.audio_queue.get_nowait()
                    self.chunk_position = 0
                    self._playing = True
                except Empty:
                    # No more audio - fill rest with silence
                    outdata[output_position:] = 0
                    self._playing = False
                    self.done_event.set()
                    return
            
            # Copy samples from current chunk
            samples_available = len(self.current_chunk) - self.chunk_position
            samples_to_copy = min(samples_needed, samples_available)
            
            chunk_end = self.chunk_position + samples_to_copy
            outdata[output_position:output_position + samples_to_copy] = \
                self.current_chunk[self.chunk_position:chunk_end].reshape(-1, 1)
            
            self.chunk_position += samples_to_copy
            output_position += samples_to_copy
            samples_needed -= samples_to_copy
    
    def start(self) -> bool:
        """
        Start playback stream.
        
        Returns:
            True if stream started successfully, False otherwise
        """
        if self._running:
            log.warning("Stream already running")
            return True
        
        try:
            # Create output stream
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.block_size,
                callback=self._audio_callback,
                dtype=np.float32
            )
            
            # Start stream
            self.stream.start()
            self._running = True
            
            log.info("✅ Audio playback started")
            return True
            
        except Exception as e:
            log.error(f"Failed to start playback stream: {e}")
            return False
    
    def stop(self):
        """Stop playback immediately and clear queue."""
        if not self._running:
            return
        
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Empty:
                break
        
        # Clear current chunk
        self.current_chunk = None
        self.chunk_position = 0
        self._playing = False
        self.done_event.set()
        
        # Stop stream
        self._running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                log.error(f"Error stopping stream: {e}")
            finally:
                self.stream = None
        
        log.info("Audio playback stopped")
    
    def queue_audio(self, audio: np.ndarray):
        """
        Queue audio chunk for playback.
        
        Args:
            audio: Audio data as numpy array (any length, float32, mono)
        """
        if not self._running:
            log.warning("Cannot queue audio: stream not running")
            return
        
        # Ensure correct format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Flatten if needed
        if len(audio.shape) > 1:
            audio = audio.flatten()
        
        # Queue for playback
        self.audio_queue.put(audio)
        self.done_event.clear()
        
        log.debug(f"Queued {len(audio)} samples ({len(audio)/self.sample_rate:.3f}s)")
    
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._playing
    
    def is_running(self) -> bool:
        """Check if playback stream is active."""
        return self._running
    
    def wait_until_done(self, timeout: float = None) -> bool:
        """
        Block until all queued audio finishes playing.
        
        Args:
            timeout: Max seconds to wait (None = infinite)
            
        Returns:
            True if finished, False if timed out
        """
        return self.done_event.wait(timeout=timeout)
    
    def get_stats(self) -> dict:
        """Get playback statistics for debugging."""
        return {
            "running": self._running,
            "playing": self._playing,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "block_size": self.block_size,
            "block_duration_ms": (self.block_size / self.sample_rate) * 1000,
            "queue_size": self.audio_queue.qsize(),
            "current_chunk_samples": len(self.current_chunk) if self.current_chunk is not None else 0
        }
    
    def __enter__(self):
        """Context manager support."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.stop()


def generate_sine_wave(duration: float, frequency: float = 440.0, 
                       sample_rate: int = 16000) -> np.ndarray:
    """Generate sine wave for testing."""
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples, False)
    wave = np.sin(2 * np.pi * frequency * t)
    return wave.astype(np.float32) * 0.3  # 30% volume


def test_audio_playback():
    """Test audio playback functionality."""
    import time
    
    print("Testing AudioPlayback...")
    
    playback = AudioPlayback()
    
    # Test 1: Start playback
    print("  Testing start...")
    assert playback.start() == True
    assert playback.is_running() == True
    assert playback.is_playing() == False  # Nothing queued yet
    print("  ✅ Playback started")
    
    # Test 2: Queue and play audio
    print("  Testing playback (0.5s sine wave @ 440Hz)...")
    audio = generate_sine_wave(0.5, 440)
    playback.queue_audio(audio)
    
    time.sleep(0.05)  # Let playback start
    assert playback.is_playing() == True
    print("  ✅ Audio playing")
    
    # Test 3: Wait for completion
    print("  Testing wait_until_done...")
    assert playback.wait_until_done(timeout=2.0) == True
    assert playback.is_playing() == False
    print("  ✅ Playback completed")
    
    # Test 4: Multiple chunks
    print("  Testing multiple chunks (3 x 0.2s)...")
    playback.queue_audio(generate_sine_wave(0.2, 440))
    playback.queue_audio(generate_sine_wave(0.2, 554))  # C# note
    playback.queue_audio(generate_sine_wave(0.2, 659))  # E note
    
    assert playback.wait_until_done(timeout=2.0) == True
    print("  ✅ Multiple chunks played")
    
    # Test 5: Interruption
    print("  Testing interruption (play 2s, stop after 0.2s)...")
    playback.queue_audio(generate_sine_wave(2.0, 440))
    time.sleep(0.2)
    
    start = time.time()
    playback.stop()
    stop_time = time.time() - start
    
    assert stop_time < 0.1  # Should stop within 100ms
    assert playback.is_playing() == False
    print(f"  ✅ Interrupted in {stop_time*1000:.1f}ms")
    
    # Test 6: Restart after stop
    print("  Testing restart after stop...")
    assert playback.start() == True
    playback.queue_audio(generate_sine_wave(0.3, 880))  # High A
    assert playback.wait_until_done(timeout=1.0) == True
    print("  ✅ Restart works")
    
    # Test 7: Stats
    print("  Testing stats...")
    stats = playback.get_stats()
    print(f"    Stats: {stats}")
    assert stats["running"] == True
    print("  ✅ Stats working")
    
    # Cleanup
    playback.stop()
    assert playback.is_running() == False
    
    print("\n✅ All tests passed!")
    print("\nNote: You should have heard:")
    print("  - 0.5s tone (440Hz)")
    print("  - 3 short tones ascending (C-C#-E chord)")
    print("  - 0.2s tone (interrupted from 2s)")
    print("  - 0.3s high tone (880Hz)")


if __name__ == "__main__":
    test_audio_playback()