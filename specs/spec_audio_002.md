# Feature Specification: AUDIO-002

## User Story
As a developer, I want a unified frame processing pipeline that connects the microphone to VAD components so that I can capture speech segments without managing the low-level integration.

## Implementation Contract

### Level 1: Plain English
The frame processing pipeline orchestrates the flow from microphone input through VAD detection to speech segment capture. It:
1. Receives audio frames from RealtimeAudioCapture
2. Routes frames through appropriate VAD detector based on state
3. Detects speech end via natural pause or timeout
4. Handles interruptions during assistant speech
5. Returns complete speech segments for transcription

This provides a clean interface: start listening → get speech segment → transcribe.

### Level 2: Logic Flow

**Input:** 
- Microphone audio stream (via RealtimeAudioCapture)
- Listening state (listening vs assistant_speaking)

**Logic:**
1. Initialize components:
   - VoiceActivityDetector
   - SpeechEndDetector (with timeout)
   - InterruptionDetector
   - RealtimeAudioCapture

2. Frame processing loop:
   ```
   while listening:
       frame = capture.get_frame()
       
       if assistant_speaking:
           # Check for interruption
           if interruption_detector.process_frame(frame):
               stop_tts()
               switch_to_listening()
       else:
           # Normal speech detection
           segment, reason = speech_end_detector.process_frame(frame)
           if segment:
               return segment, reason
   ```

3. State transitions:
   - Idle → Listening (on wake word or button press)
   - Listening → Processing (speech segment captured)
   - Processing → Speaking (TTS starts)
   - Speaking → Listening (on interruption or TTS complete)

**Output:** 
- Speech segments (numpy arrays)
- End reason ("natural_pause" or "timeout")
- State change events

### Level 3: Formal Interface

```python
class AudioFramePipeline:
    """
    Unified audio frame processing pipeline.
    
    Orchestrates microphone input through VAD components to speech capture.
    """
    
    def __init__(self,
                 sample_rate: int = 16000,
                 frame_size: int = 512,
                 timeout_s: float = 30.0):
        """
        Initialize pipeline.
        
        Args:
            sample_rate: Audio sample rate (16kHz for Whisper)
            frame_size: Frame size in samples (512 = 32ms @ 16kHz)
            timeout_s: Maximum speech duration before forcing end
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        
        # Audio capture
        self.capture = RealtimeAudioCapture(
            sample_rate=sample_rate,
            frame_size=frame_size
        )
        
        # VAD components
        self.vad = VoiceActivityDetector()
        self.speech_detector = SpeechEndDetector(self.vad, timeout_s=timeout_s)
        self.interruption_detector = InterruptionDetector(self.vad)
        
        # State
        self.listening = False
        self.assistant_speaking = False
    
    def start_listening(self):
        """Start capturing and processing audio."""
        pass
    
    def stop_listening(self):
        """Stop audio capture."""
        pass
    
    def set_assistant_speaking(self, speaking: bool):
        """Set assistant speaking state for interruption detection."""
        pass
    
    def capture_speech_segment(self, timeout: float = None) -> tuple:
        """
        Capture a complete speech segment.
        
        Blocks until speech is detected and completed.
        
        Args:
            timeout: Optional timeout in seconds (uses default if None)
        
        Returns:
            (segment, end_reason) tuple where:
                segment: Speech audio (np.ndarray) or None on timeout
                end_reason: "natural_pause", "timeout", or "interrupted"
        
        Raises:
            TimeoutError: If no speech detected within timeout
        """
        pass
    
    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        pass
```

## Validation Contract

### Level 1: Scenarios

- [ ] Start/stop listening: Capture starts/stops correctly
- [ ] Speech capture: Returns complete segment on natural pause
- [ ] Timeout handling: Returns segment after timeout duration
- [ ] Interruption handling: Detects user speech during assistant output
- [ ] State management: Correctly tracks listening/speaking states
- [ ] Error handling: Gracefully handles audio device errors
- [ ] Statistics: Aggregates stats from all components

### Level 2: Test Logic

**Scenario: Capture speech segment with natural pause**
- Given: Pipeline started, listening active
- When: User speaks for 2 seconds, then pauses
- Then:
  - Segment captured after ~300ms silence
  - End reason is "natural_pause"
  - Segment length is ~2 seconds of audio

**Scenario: Timeout during long speech**
- Given: Pipeline with 5s timeout
- When: User speaks continuously for 6 seconds
- Then:
  - Segment captured at 5 seconds
  - End reason is "timeout"
  - Segment length is ~5 seconds

**Scenario: User interrupts assistant**
- Given: Pipeline with assistant_speaking=True
- When: User starts speaking
- Then:
  - Interruption detected within ~100ms
  - Assistant speech stops
  - New segment capture begins

**Scenario: No speech timeout**
- Given: Pipeline started
- When: No speech for 10 seconds
- Then:
  - Returns None or raises TimeoutError
  - No segment captured

### Level 3: Formal Tests

```python
def test_speech_capture_natural_pause():
    """Test capturing speech with natural pause."""
    pipeline = AudioFramePipeline()
    pipeline.start_listening()
    
    # Simulate speech frames
    # Assert: segment returned with "natural_pause"

def test_speech_capture_timeout():
    """Test capturing speech that exceeds timeout."""
    pipeline = AudioFramePipeline(timeout_s=5.0)
    pipeline.start_listening()
    
    # Simulate continuous speech > 5s
    # Assert: segment returned with "timeout"

def test_interruption_detection():
    """Test user interruption during assistant speech."""
    pipeline = AudioFramePipeline()
    pipeline.set_assistant_speaking(True)
    
    # Simulate user speech
    # Assert: interruption detected
```

## Dependencies

- @VAD-001: VoiceActivityDetector
- @VAD-002: SpeechEndDetector  
- @VAD-003: InterruptionDetector
- @AUDIO-001 (Phase 1): RealtimeAudioCapture
- Uses existing audio infrastructure from Phase 1

## Implementation Notes

**Component Initialization:**
```python
# Create VAD first
vad = VoiceActivityDetector(
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=300
)

# Create speech end detector with timeout
speech_detector = SpeechEndDetector(vad, timeout_s=30.0)

# Create interruption detector with higher threshold
interruption_detector = InterruptionDetector(
    vad, 
    interruption_threshold=0.6,
    confirmation_frames=3
)
```

**Frame Processing Flow:**
```python
def process_frame(self, frame: np.ndarray):
    """Process single audio frame through appropriate detector."""
    
    if self.assistant_speaking:
        # Check for interruption
        if self.interruption_detector.process_frame(frame):
            return "interrupted"
    else:
        # Normal speech detection
        segment, reason = self.speech_detector.process_frame(frame)
        if segment is not None:
            return segment, reason
    
    return None, None
```

**Blocking Capture Method:**
```python
def capture_speech_segment(self, timeout=None):
    """Block until speech segment captured."""
    
    self.start_listening()
    start_time = time.time()
    
    while True:
        # Get frame from capture
        frame = self.capture.get_frame()
        
        # Process through pipeline
        segment, reason = self.process_frame(frame)
        
        if segment is not None:
            return segment, reason
        
        # Check timeout
        if timeout and (time.time() - start_time) > timeout:
            raise TimeoutError("No speech detected")
```

**State Management:**
- `listening`: Audio capture active, processing frames
- `assistant_speaking`: TTS active, monitoring for interruptions
- State transitions logged for debugging

**Error Handling:**
- Audio device errors → log and retry
- VAD errors → log and continue (degrade gracefully)
- Buffer overflow → log warning, force segment end

**Integration Points:**
- Whisper: Segments go directly to transcription
- Piper: Set assistant_speaking during TTS
- Wake word: Triggers start_listening()

## Acceptance Criteria

✅ Pipeline initializes all components correctly
✅ Captures speech segments with natural pause
✅ Enforces timeout on long speech
✅ Detects interruptions during assistant speech
✅ Start/stop listening works correctly
✅ State management tracks listening/speaking
✅ Statistics aggregate from all components
✅ Error handling degrades gracefully
✅ All tests pass