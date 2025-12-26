# Feature Specification: VAD-003

## User Story
As a user, I want to interrupt the assistant while it's speaking so that I can correct it, stop it, or give a new command without waiting for it to finish.

## Implementation Contract

### Level 1: Plain English
The system monitors for user speech WHILE the assistant is speaking (TTS output). If the user starts speaking during assistant output, the system:
1. Immediately stops assistant speech
2. Switches to listening mode
3. Captures the user's interruption as a new command

This creates natural, conversational interaction where users don't have to wait for the assistant to finish.

### Level 2: Logic Flow

**Input:** 
- Audio frames (512 samples @ 16kHz)
- Assistant speaking state (boolean)
- VAD detection results

**Logic:**
1. While assistant is speaking:
   - Monitor VAD for user speech
   - If VAD detects speech (probability > threshold):
     - Trigger interruption event
     - Stop TTS playback
     - Switch to listening mode
2. While listening:
   - Normal speech detection flow
   - Capture user's complete utterance

**Output:** 
- Interruption event (with timestamp)
- User's interrupting speech segment

### Level 3: Formal Interface

```python
class InterruptionDetector:
    """Detects user interruptions during assistant speech."""
    
    def __init__(self, vad: VoiceActivityDetector,
                 interruption_threshold: float = 0.6,
                 confirmation_frames: int = 3):
        """
        Initialize interruption detector.
        
        Args:
            vad: VoiceActivityDetector instance
            interruption_threshold: Probability threshold for interruption
            confirmation_frames: Frames of speech needed to confirm interruption
        """
        self.vad = vad
        self.interruption_threshold = interruption_threshold
        self.confirmation_frames = confirmation_frames
        self.assistant_speaking = False
        self.interruption_callback = None
        self.interruption_count = 0
    
    def set_assistant_speaking(self, speaking: bool):
        """Set assistant speaking state."""
        pass
    
    def process_frame(self, audio: np.ndarray) -> bool:
        """
        Process frame and detect interruptions.
        
        Args:
            audio: Audio frame (512 samples, float32)
        
        Returns:
            True if interruption detected, False otherwise
        """
        pass
    
    def on_interruption(self, callback: Callable):
        """Register callback for interruption events."""
        pass
    
    def reset(self):
        """Reset detector state."""
        pass
    
    def get_stats(self) -> dict:
        """Get interruption statistics."""
        pass
```

## Validation Contract

### Level 1: Scenarios

- [ ] Interruption detected: User speaks during assistant speech
- [ ] No false positives: Ambient noise during assistant speech doesn't trigger
- [ ] Confirmation required: Brief noise doesn't trigger, need N frames
- [ ] Callback fires: Registered callback called on interruption
- [ ] State reset: Detector resets after interruption handled
- [ ] Statistics: Track interruption count and timing

### Level 2: Test Logic

**Scenario: User interrupts assistant**
- Given: InterruptionDetector with assistant_speaking=True
- When: Process 3+ frames with high speech probability
- Then:
  - Interruption detected (returns True)
  - Callback fires if registered
  - interruption_count increments

**Scenario: Noise doesn't trigger interruption**
- Given: InterruptionDetector with assistant_speaking=True
- When: Process frames with low speech probability
- Then:
  - No interruption detected
  - Callback doesn't fire

**Scenario: Confirmation frames required**
- Given: InterruptionDetector with confirmation_frames=3
- When: Process 2 frames with high probability, then low
- Then:
  - No interruption (need 3 consecutive frames)

**Scenario: No interruption when assistant not speaking**
- Given: InterruptionDetector with assistant_speaking=False
- When: Process frames with high speech probability
- Then:
  - No interruption detected (normal speech, not interruption)

### Level 3: Formal Tests

```python
def test_interruption_detection():
    """Test interruption during assistant speech."""
    detector = InterruptionDetector(vad)
    detector.set_assistant_speaking(True)
    
    # Mock high probability speech
    # Process 3 frames
    # Assert: interruption detected
    
def test_no_false_positives():
    """Test noise doesn't trigger interruption."""
    detector = InterruptionDetector(vad)
    detector.set_assistant_speaking(True)
    
    # Mock low probability
    # Process frames
    # Assert: no interruption

def test_confirmation_frames():
    """Test confirmation frames requirement."""
    detector = InterruptionDetector(vad, confirmation_frames=3)
    detector.set_assistant_speaking(True)
    
    # Process 2 high-prob frames
    # Assert: no interruption yet
    # Process 1 more
    # Assert: interruption detected

def test_callback_firing():
    """Test interruption callback fires."""
    detector = InterruptionDetector(vad)
    
    callback_fired = [False]
    def callback():
        callback_fired[0] = True
    
    detector.on_interruption(callback)
    detector.set_assistant_speaking(True)
    
    # Trigger interruption
    # Assert: callback_fired[0] == True
```

## Dependencies

- @VAD-001: VoiceActivityDetector for speech detection
- Uses VAD probability and is_speaking state
- Independent of VAD-002 (doesn't need SpeechEndDetector)

## Implementation Notes

**Threshold Tuning:**
- Use higher threshold than normal VAD (default 0.6 vs 0.5)
- Reduces false positives from background noise
- Users speaking intentionally will exceed this easily

**Confirmation Frames:**
- Require N consecutive frames of speech to confirm interruption
- Default: 3 frames (~96ms)
- Prevents single-frame noise spikes from triggering

**State Management:**
- `assistant_speaking` flag set externally by TTS system
- When True: monitor for interruptions
- When False: ignore (normal listening mode)

**Callback Pattern:**
- Register callback with `on_interruption(callback_fn)`
- Callback fires when interruption detected
- Callback should stop TTS and switch to listening

**Integration Flow:**
```python
# Main loop
detector = InterruptionDetector(vad)
detector.on_interruption(lambda: stop_tts_and_listen())

# When assistant starts speaking
detector.set_assistant_speaking(True)
play_tts(response)

# Audio loop continues
for frame in audio_stream:
    if detector.process_frame(frame):
        # Interruption detected
        # Callback already fired, just continue to listening
        pass
```

**Edge Cases:**
- Assistant finishes speaking before interruption: Normal completion
- Multiple rapid interruptions: Each tracked separately
- Interruption at end of assistant speech: Still valid, restart listening

## Acceptance Criteria

✅ Interruption detected during assistant speech
✅ High threshold prevents false positives
✅ Confirmation frames prevent noise triggers
✅ Callback mechanism works correctly
✅ No interruption when assistant not speaking
✅ Statistics track interruption events
✅ Reset clears all state
✅ All tests pass