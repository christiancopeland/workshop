# Feature Spec: AUDIO-003 - Audio Playback Pipeline

## User Story
As Workshop, I need to play synthesized speech audio with low latency and support immediate interruption so that responses feel natural and users can interrupt mid-sentence.

## Implementation Contract

### Level 1: Plain English
Create a non-blocking audio playback system that accepts audio chunks from TTS, queues them, and plays through speakers. Support immediate abort when user interrupts. Playback runs in background thread, never blocks main loop.

### Level 2: Logic Flow
**Input:** Audio chunks (numpy arrays from TTS)
**Logic:**
1. Initialize sounddevice output stream (16kHz, mono)
2. Provide `queue_audio(chunk)` method that adds to playback queue
3. Stream callback pulls chunks from queue, plays to speakers
4. Provide `stop()` that immediately clears queue and stops playback
5. Track playback state (idle/playing)
**Output:** Audio played through speakers

### Level 3: Formal Interface
```python
class AudioPlayback:
    """Non-blocking audio playback with interruption support"""
    
    def __init__(self,
                 sample_rate: int = 16000,
                 channels: int = 1):
        """Initialize playback system"""
        
    def start(self) -> bool:
        """Start playback stream. Returns True if successful."""
        
    def stop(self):
        """Stop playback immediately and clear queue."""
        
    def queue_audio(self, audio: np.ndarray):
        """
        Queue audio chunk for playback.
        
        Args:
            audio: Audio data as numpy array (any length)
        """
        
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        
    def wait_until_done(self, timeout: float = None) -> bool:
        """
        Block until all queued audio finishes playing.
        
        Args:
            timeout: Max seconds to wait (None = infinite)
            
        Returns:
            True if finished, False if timed out
        """
```

## Validation Contract

### Level 1: Scenarios
- ✅ Happy path: Queue audio, plays through speakers
- ✅ Interruption: stop() clears queue immediately
- ✅ Multiple chunks: Plays sequentially without gaps
- ✅ Edge: Empty queue handled gracefully

### Level 2: Test Logic

**Scenario: Basic playback**
- Given: AudioPlayback initialized and started
- When: queue_audio(sine_wave) called
- Then: is_playing() == True AND audio plays

**Scenario: Immediate stop**
- Given: Audio playing
- When: stop() called
- Then: is_playing() == False within 50ms AND queue empty

**Scenario: Sequential chunks**
- Given: AudioPlayback started
- When: 3 audio chunks queued
- Then: All 3 play sequentially without gaps

### Level 3: Formal Tests
```python
def test_playback_basic():
    playback = AudioPlayback()
    playback.start()
    
    # Generate 0.5s sine wave
    audio = generate_sine(0.5)
    playback.queue_audio(audio)
    
    assert playback.is_playing() == True
    playback.wait_until_done()
    assert playback.is_playing() == False
    
    playback.stop()

def test_playback_interrupt():
    playback = AudioPlayback()
    playback.start()
    
    # Queue 5 seconds of audio
    audio = generate_sine(5.0)
    playback.queue_audio(audio)
    
    time.sleep(0.1)  # Let playback start
    assert playback.is_playing() == True
    
    # Interrupt
    start = time.time()
    playback.stop()
    elapsed = time.time() - start
    
    assert elapsed < 0.05  # Stops within 50ms
    assert playback.is_playing() == False
```

## Dependencies
- AUDIO-003 depends on AUDIO-003 (playback uses same output device)

## Implementation Notes
- Use `sounddevice.OutputStream` with callback
- Callback pulls from queue, plays to hardware
- stop() sets flag that callback checks, clears queue
- Block size: 512 samples = 32ms latency

---

