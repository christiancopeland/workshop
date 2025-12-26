# Feature Specification: WAKE-001

## User Story
As a user, I want to activate the assistant by saying a wake word so that I can interact hands-free without pressing buttons.

## Implementation Contract

### Level 1: Plain English
Wake word integration connects the wake word detector (from Phase 1) to the audio frame pipeline. When the wake word is detected:
1. Pipeline switches from idle to listening mode
2. User's command is captured as a speech segment
3. Segment is sent to transcription
4. Normal workflow proceeds

This enables "Hey Workshop" → command → response interaction flow.

### Level 2: Logic Flow

**Input:**
- Audio frames from microphone
- Wake word model (from Phase 1)
- Audio pipeline (from AUDIO-002)

**Logic:**
```
State: IDLE
while True:
    frame = get_audio_frame()
    
    if state == IDLE:
        if wake_word_detector.process_frame(frame):
            log("Wake word detected!")
            state = LISTENING
            pipeline.start_listening()
            
    elif state == LISTENING:
        segment, reason = pipeline.capture_speech_segment()
        if segment:
            state = PROCESSING
            transcribe(segment)
            
    elif state == PROCESSING:
        # Wait for LLM response
        response = get_llm_response()
        state = SPEAKING
        
    elif state == SPEAKING:
        play_tts(response)
        state = IDLE  # or LISTENING if interrupted
```

**Output:**
- State transitions (idle → listening → processing → speaking → idle)
- Speech segments for transcription
- Wake word detection events

### Level 3: Formal Interface

```python
class WakeWordPipeline:
    """
    Integrates wake word detection with audio frame pipeline.
    
    Manages state machine: idle → wake → listening → processing → speaking → idle
    """
    
    def __init__(self,
                 wake_word: str = "workshop",
                 model_path: str = None,
                 timeout_s: float = 30.0):
        """
        Initialize wake word pipeline.
        
        Args:
            wake_word: Wake word to detect (default "workshop")
            model_path: Path to wake word model (optional)
            timeout_s: Speech timeout for audio pipeline
        """
        self.wake_word = wake_word
        
        # Initialize components
        self.wake_detector = WakeWordDetector(wake_word, model_path)
        self.audio_pipeline = AudioFramePipeline(timeout_s=timeout_s)
        
        # State machine
        self.state = "idle"  # idle, listening, processing, speaking
        
        # Callbacks
        self.on_wake = None
        self.on_speech = None
    
    def run(self):
        """
        Run main loop (blocking).
        
        Handles wake word detection and speech capture.
        """
        pass
    
    def set_state(self, state: str):
        """Set pipeline state."""
        pass
    
    def register_callbacks(self,
                          on_wake: Callable = None,
                          on_speech: Callable = None):
        """
        Register callbacks for events.
        
        Args:
            on_wake: Called when wake word detected
            on_speech: Called with speech segment when captured
        """
        pass
    
    def stop(self):
        """Stop pipeline."""
        pass
    
    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        pass
```

## Validation Contract

### Level 1: Scenarios

- [ ] Wake word detection: "workshop" triggers listening
- [ ] Speech capture after wake: Captures user command
- [ ] State transitions: Idle → listening → processing flow
- [ ] Callback firing: on_wake and on_speech called correctly
- [ ] False wake rejection: Non-wake-word speech doesn't trigger
- [ ] Statistics: Track wake events and speech captures

### Level 2: Test Logic

**Scenario: Wake word triggers listening**
- Given: WakeWordPipeline in idle state
- When: Wake word "workshop" detected in audio
- Then:
  - State transitions to "listening"
  - on_wake callback fires
  - Audio pipeline starts listening

**Scenario: Speech captured after wake**
- Given: Pipeline in listening state (after wake)
- When: User speaks command and pauses
- Then:
  - Speech segment captured
  - on_speech callback fires with segment
  - State transitions to "processing"

**Scenario: False wake rejection**
- Given: Pipeline in idle state
- When: Audio contains speech but not wake word
- Then:
  - No state transition
  - No callbacks fire
  - Remains in idle state

**Scenario: State machine**
- Given: Pipeline initialized
- When: Complete wake → speech → process cycle
- Then:
  - State: idle → listening → processing
  - All transitions logged
  - Stats track each state duration

### Level 3: Formal Tests

```python
def test_wake_word_detection():
    """Test wake word triggers listening."""
    pipeline = WakeWordPipeline()
    
    wake_fired = [False]
    pipeline.register_callbacks(on_wake=lambda: wake_fired.__setitem__(0, True))
    
    # Simulate wake word audio
    # Assert: state == "listening"
    # Assert: wake_fired[0] == True

def test_speech_capture_after_wake():
    """Test speech capture after wake."""
    pipeline = WakeWordPipeline()
    
    speech_captured = [None]
    pipeline.register_callbacks(on_speech=lambda seg: speech_captured.__setitem__(0, seg))
    
    # Trigger wake
    # Simulate speech
    # Assert: speech_captured[0] is not None
    # Assert: state == "processing"

def test_false_wake_rejection():
    """Test non-wake speech doesn't trigger."""
    pipeline = WakeWordPipeline()
    
    wake_fired = [False]
    pipeline.register_callbacks(on_wake=lambda: wake_fired.__setitem__(0, True))
    
    # Simulate non-wake speech
    # Assert: state == "idle"
    # Assert: wake_fired[0] == False
```

## Dependencies

- @WAKE-001 (Phase 1): WakeWordDetector for wake word detection
- @AUDIO-002: AudioFramePipeline for speech capture
- All VAD components (VAD-001, VAD-002, VAD-003)

## Implementation Notes

**State Machine:**
- **idle**: Listening for wake word only
- **listening**: Capturing user speech after wake
- **processing**: Transcribing and getting LLM response
- **speaking**: Playing TTS output (can be interrupted)

**State Transitions:**
- idle → listening: Wake word detected
- listening → processing: Speech segment captured
- processing → speaking: LLM response ready
- speaking → listening: User interruption
- speaking → idle: TTS complete

**Wake Word Integration:**
```python
# In idle state
while self.state == "idle":
    frame = capture.get_frame()
    
    if self.wake_detector.process_frame(frame):
        log.info("🎤 Wake word detected!")
        self.set_state("listening")
        
        if self.on_wake:
            self.on_wake()
```

**Speech Capture After Wake:**
```python
# In listening state
if self.state == "listening":
    segment, reason = self.audio_pipeline.capture_speech_segment(
        max_wait_s=30.0
    )
    
    if segment is not None:
        log.info(f"📝 Speech captured: {len(segment)} samples")
        self.set_state("processing")
        
        if self.on_speech:
            self.on_speech(segment, reason)
```

**Integration with Main Loop:**
```python
# main.py
pipeline = WakeWordPipeline()

def on_wake():
    print("Wake word detected!")

def on_speech(segment, reason):
    print(f"Transcribing {len(segment)} samples...")
    text = whisper.transcribe(segment)
    print(f"You said: {text}")
    # Send to LLM...

pipeline.register_callbacks(on_wake=on_wake, on_speech=on_speech)
pipeline.run()  # Blocking
```

**Error Handling:**
- Audio device errors → log and retry
- Wake detection errors → log and continue
- Speech capture timeout → return to idle
- Callback errors → log but don't crash

**Performance:**
- Wake word detection: ~30ms per frame
- State transitions: < 10ms
- Callback execution: Non-blocking (run in callbacks)

## Acceptance Criteria

✅ Wake word detection triggers listening mode
✅ Speech captured after wake word
✅ State machine transitions correctly
✅ Callbacks fire at appropriate times
✅ False wake words rejected
✅ Statistics track wake events
✅ Error handling degrades gracefully
✅ All tests pass

## Integration Notes

This is the final piece of Phase 2. Once complete, we have:
- ✅ VAD-001: Voice activity detection
- ✅ VAD-002: Speech end detection with timeout
- ✅ VAD-003: User interruption detection
- ✅ AUDIO-002: Frame processing pipeline
- ✅ WAKE-001: Wake word integration

**Full system flow:**
1. User says "workshop" → Wake detector triggers
2. User speaks command → AudioFramePipeline captures
3. Whisper transcribes → LLM processes
4. Piper speaks → InterruptionDetector monitors
5. User can interrupt → Back to listening

Phase 2 complete! 🎉