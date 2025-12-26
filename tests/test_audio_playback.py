"""
Unit tests for AudioPlayback that work without audio hardware
"""

import sys
from pathlib import Path
import numpy as np
import time
from unittest.mock import MagicMock
from queue import Queue
from threading import Event

# Mock sounddevice and logger before importing
sys.modules['sounddevice'] = MagicMock()
sys.modules['logger'] = MagicMock()

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))


def test_playback_init():
    """Test AudioPlayback initialization"""
    from audio_playback import AudioPlayback
    
    playback = AudioPlayback(sample_rate=16000, channels=1, block_size=512)
    
    assert playback.sample_rate == 16000
    assert playback.channels == 1
    assert playback.block_size == 512
    assert playback.is_running() == False
    assert playback.is_playing() == False
    assert isinstance(playback.audio_queue, Queue)
    assert isinstance(playback.done_event, Event)
    
    print("✅ Initialization test passed")


def test_playback_queue_audio():
    """Test queuing audio"""
    from audio_playback import AudioPlayback
    
    playback = AudioPlayback()
    playback._running = True  # Fake running state
    
    # Queue audio
    audio = np.random.randn(8000).astype(np.float32)  # 0.5s @ 16kHz
    playback.queue_audio(audio)
    
    assert playback.audio_queue.qsize() == 1
    
    # Get queued audio
    queued = playback.audio_queue.get()
    assert len(queued) == 8000
    assert queued.dtype == np.float32
    
    print("✅ Queue audio test passed")


def test_playback_callback():
    """Test audio callback logic"""
    from audio_playback import AudioPlayback
    
    playback = AudioPlayback(block_size=512)
    playback._running = True
    
    # Queue some audio
    audio = np.ones(1024, dtype=np.float32)  # 2 blocks worth
    playback.audio_queue.put(audio)
    
    # Simulate callback pulling first block
    outdata = np.zeros((512, 1), dtype=np.float32)
    playback._audio_callback(outdata, 512, None, None)
    
    # Should have pulled 512 samples
    assert playback.chunk_position == 512
    assert np.allclose(outdata.flatten()[:512], 1.0)
    assert playback.is_playing() == True
    
    print("✅ Callback test passed")


def test_playback_callback_underrun():
    """Test callback handles empty queue gracefully"""
    from audio_playback import AudioPlayback
    
    playback = AudioPlayback(block_size=512)
    playback._running = True
    playback._playing = True
    
    # No audio in queue
    assert playback.audio_queue.empty()
    
    # Simulate callback
    outdata = np.zeros((512, 1), dtype=np.float32)
    playback._audio_callback(outdata, 512, None, None)
    
    # Should fill with silence and mark as not playing
    assert np.all(outdata == 0)
    assert playback.is_playing() == False
    assert playback.done_event.is_set() == True
    
    print("✅ Callback underrun test passed")


def test_playback_stop_clears_queue():
    """Test stop() clears all queued audio"""
    from audio_playback import AudioPlayback
    
    playback = AudioPlayback()
    playback._running = True
    
    # Queue multiple chunks
    for i in range(5):
        audio = np.random.randn(1000).astype(np.float32)
        playback.audio_queue.put(audio)
    
    assert playback.audio_queue.qsize() == 5
    
    # Stop should clear queue
    playback.stop()
    assert playback.audio_queue.qsize() == 0
    assert playback.current_chunk is None
    assert playback.is_playing() == False
    
    print("✅ Stop clears queue test passed")


def test_playback_wait_until_done():
    """Test wait_until_done blocking"""
    from audio_playback import AudioPlayback
    
    playback = AudioPlayback()
    
    # Start as done
    assert playback.done_event.is_set() == True
    assert playback.wait_until_done(timeout=0.1) == True
    
    # Clear done event (simulate playback)
    playback.done_event.clear()
    
    # Should timeout waiting
    start = time.time()
    result = playback.wait_until_done(timeout=0.1)
    elapsed = time.time() - start
    
    assert result == False  # Timed out
    assert 0.09 < elapsed < 0.12  # ~0.1s
    
    # Set done event
    playback.done_event.set()
    assert playback.wait_until_done(timeout=0.1) == True
    
    print("✅ Wait until done test passed")


def test_playback_stats():
    """Test stats reporting"""
    from audio_playback import AudioPlayback
    
    playback = AudioPlayback(sample_rate=16000, block_size=512)
    
    stats = playback.get_stats()
    
    assert stats["running"] == False
    assert stats["playing"] == False
    assert stats["sample_rate"] == 16000
    assert stats["block_size"] == 512
    assert stats["block_duration_ms"] == 32.0
    assert stats["queue_size"] == 0
    
    print("✅ Stats test passed")


def test_generate_sine_wave():
    """Test sine wave generation helper"""
    from audio_playback import generate_sine_wave
    
    wave = generate_sine_wave(duration=1.0, frequency=440, sample_rate=16000)
    
    assert len(wave) == 16000  # 1 second @ 16kHz
    assert wave.dtype == np.float32
    assert np.max(np.abs(wave)) <= 0.31  # 30% volume + tolerance
    
    # Check it's actually a sine wave (zero crossings)
    zero_crossings = np.where(np.diff(np.sign(wave)))[0]
    expected_crossings = 2 * 440  # 2 per cycle
    assert abs(len(zero_crossings) - expected_crossings) < 10  # Allow some tolerance
    
    print("✅ Sine wave generation test passed")


if __name__ == "__main__":
    print("Running AudioPlayback unit tests...\n")
    
    test_playback_init()
    test_playback_queue_audio()
    test_playback_callback()
    test_playback_callback_underrun()
    test_playback_stop_clears_queue()
    test_playback_wait_until_done()
    test_playback_stats()
    test_generate_sine_wave()
    
    print("\n✅ All unit tests passed!")
    print("\nNote: Hardware audio test requires real speakers.")
    print("To test on your machine: python audio_playback.py")