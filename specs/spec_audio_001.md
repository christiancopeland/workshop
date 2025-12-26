# Feature Spec: AUDIO-001 - Continuous Audio Stream Capture

## User Story
As Workshop, I need to continuously capture audio from the microphone so that I can detect wake words, speech, and interruptions in real-time without batch delays.

## Implementation Contract

### Level 1: Plain English
Create a non-blocking audio stream that continuously captures microphone input in fixed-size frames (512 samples @ 16kHz = 32ms per frame). Audio frames feed into a queue that other components (VAD, wake word, STT) can consume without blocking the capture thread.

### Level 2: Logic Flow
**Input:** Microphone hardware
**Logic:**
1. Initialize sounddevice input stream (16kHz, mono, 512 frame size)
2. Register callback that fires every 32ms with audio frame
3. Callback converts numpy array to bytes, pushes to thread-safe queue
4. Stream runs in background thread, never blocks main loop
5. Provide `get_frame()` method that returns frames or None if empty
**Output:** Continuous stream of 32ms audio frames

### Level 3: Formal Interface
```python
class AudioStream:
    """Continuous audio capture with non-blocking frame access"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 channels: int = 1, 
                 frame_size: int = 512):
        """
        Args:
            sample_rate: Audio sample rate (16kHz for Whisper)
            channels: Mono (1) or stereo (2)
            frame_size: Samples per frame (512 = 32ms @ 16kHz)
        """
        
    def start(self) -> bool:
        """Start audio capture. Returns True if successful."""
        
    def stop(self):
        """Stop audio capture and cleanup."""
        
    def get_frame(self, timeout: float = 0.0) -> Optional[np.ndarray]:
        """
        Get next audio frame from queue.
        
        Args:
            timeout: Max seconds to wait (0 = non-blocking)
            
        Returns:
            numpy array of shape (frame_size,) or None if no frame available
        """
        
    @property
    def is_running(self) -> bool:
        """Check if stream is active."""
```

## Validation Contract

### Level 1: Scenarios
- ✅ Happy path: Stream starts, frames captured continuously
- ✅ Error: Microphone not available
- ✅ Edge: Stream stops cleanly, no memory leaks

### Level 2: Test Logic

**Scenario: Continuous capture**
- Given: AudioStream initialized
- When: start() called
- Then: is_running == True AND get_frame() returns frames every ~32ms

**Scenario: Non-blocking reads**
- Given: Stream running but no audio
- When: get_frame(timeout=0) called
- Then: Returns None immediately (doesn't block)

**Scenario: Clean shutdown**
- Given: Stream running
- When: stop() called
- Then: is_running == False AND no frames available

### Level 3: Formal Tests
```python
def test_audio_stream_starts():
    stream = AudioStream()
    assert stream.start() == True
    assert stream.is_running == True
    stream.stop()

def test_frames_captured():
    stream = AudioStream()
    stream.start()
    time.sleep(0.1)  # Capture 3 frames
    frames = []
    while frame := stream.get_frame():
        frames.append(frame)
    assert len(frames) >= 2
    stream.stop()

def test_non_blocking_read():
    stream = AudioStream()
    stream.start()
    start = time.time()
    frame = stream.get_frame(timeout=0)  # Non-blocking
    elapsed = time.time() - start
    assert elapsed < 0.01  # Returns immediately
    stream.stop()
```

## Dependencies
None (this is Phase 1, feature 1)

## Implementation Notes
- Use `sounddevice.InputStream` (already in requirements)
- Queue size: 100 frames = 3.2 seconds of audio buffer
- Frame size 512 chosen for balance: small enough for low latency, large enough to avoid overhead

---
