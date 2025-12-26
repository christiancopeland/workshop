# Feature Specification: VAD-002

## User Story
As a user, I want the system to automatically detect when I've finished speaking so that it processes my command without requiring me to press a button.

## Implementation Contract

### Level 1: Plain English
The system detects speech end by monitoring two conditions:
1. Natural pause: VAD detects silence after speech (already in VAD-001)
2. Timeout: If user speaks continuously beyond a timeout period, force segment end

This prevents unbounded speech segments and provides a fallback if the user doesn't naturally pause.

### Level 2: Logic Flow

**Input:** 
- Audio frames (512 samples @ 16kHz)
- Timeout duration (configurable, default 30 seconds)

**Logic:**
1. Start timer when speech begins
2. While processing frames:
   - If VAD detects natural speech end → return segment
   - If timer exceeds timeout → force segment end, return segment
3. Reset timer when returning segment

**Output:** 
- Complete speech segment (numpy array)
- End reason: "natural_pause" or "timeout"

### Level 3: Formal Interface

```python
class SpeechEndDetector:
    """Detects speech end via natural pause or timeout."""
    
    def __init__(self, vad: VoiceActivityDetector, 
                 timeout_s: float = 30.0):
        """
        Initialize speech end detector.
        
        Args:
            vad: VoiceActivityDetector instance
            timeout_s: Maximum speech duration before forcing end
        """
        self.vad = vad
        self.collector = SpeechSegmentCollector(vad, max_duration_s=timeout_s)
        self.timeout_s = timeout_s
        self.speech_start_time = None
    
    def process_frame(self, audio: np.ndarray) -> tuple[np.ndarray | None, str | None]:
        """
        Process audio frame and detect speech end.
        
        Args:
            audio: Audio frame (512 samples, float32)
        
        Returns:
            (segment, end_reason) where:
                segment: Complete speech audio or None
                end_reason: "natural_pause", "timeout", or None
        """
        pass
    
    def reset(self):
        """Reset detector state."""
        pass
    
    def get_stats(self) -> dict:
        """Get detection statistics."""
        pass
```

## Validation Contract

### Level 1: Scenarios

- [ ] Natural pause: Speech stops, detector returns segment with reason "natural_pause"
- [ ] Timeout: Speech exceeds timeout duration, returns segment with reason "timeout"
- [ ] No speech: Silence doesn't trigger detection
- [ ] Reset: Clears timer and state
- [ ] Statistics: Tracks segments by end reason

### Level 2: Test Logic

**Scenario: Natural pause detection**
- Given: SpeechEndDetector with 30s timeout
- When: Process 10 speech frames, then 12 silence frames
- Then: 
  - Segment returned after silence
  - End reason is "natural_pause"
  - Segment length is 10 * 512 samples

**Scenario: Timeout detection**
- Given: SpeechEndDetector with 1s timeout
- When: Process 40 continuous speech frames (~1.28s)
- Then:
  - Segment returned after ~32 frames (1.024s)
  - End reason is "timeout"
  - Timer reset after segment

**Scenario: No false triggers**
- Given: SpeechEndDetector
- When: Process only silence frames
- Then:
  - No segment returned
  - Timer remains None
  - No end reason

### Level 3: Formal Tests

```python
def test_natural_pause_detection():
    """Test detection via natural speech pause."""
    detector = SpeechEndDetector(vad, timeout_s=30.0)
    
    # Setup: Mock VAD to return speech, then silence
    # Process frames
    # Assert: segment returned with reason "natural_pause"
    
def test_timeout_detection():
    """Test detection via timeout."""
    detector = SpeechEndDetector(vad, timeout_s=1.0)
    
    # Setup: Mock VAD to return continuous speech
    # Process 40 frames
    # Assert: segment returned with reason "timeout" at ~32 frames
    
def test_no_false_triggers():
    """Test that silence doesn't trigger detection."""
    detector = SpeechEndDetector(vad)
    
    # Process silence frames
    # Assert: no segment, no timer, no reason
```

## Dependencies

- @VAD-001: VoiceActivityDetector and SpeechSegmentCollector
- Uses existing VAD logic for natural pause detection
- Adds timeout layer on top

## Implementation Notes

**Timer Management:**
- Start timer when `vad.is_speaking` transitions to True
- Check timer on every frame during speech
- Reset timer when segment is returned
- Timer should be None when not actively tracking speech

**End Reason Tracking:**
- Track reason for each segment end
- Useful for debugging and statistics
- Helps identify if users are hitting timeout (may need adjustment)

**Integration:**
- SpeechEndDetector wraps SpeechSegmentCollector
- Adds timeout logic without modifying VAD core
- Returns tuple (segment, reason) instead of just segment
- Maintains backward compatibility if reason is ignored

**Edge Cases:**
- Very short speech (< min_speech_duration): No segment returned
- Exactly at timeout boundary: Should return segment
- Multiple segments in sequence: Timer resets between segments

## Acceptance Criteria

✅ Natural pause detected correctly (returns "natural_pause")
✅ Timeout enforced correctly (returns "timeout")
✅ Timer starts only during speech
✅ Timer resets after segment returned
✅ No false triggers on silence
✅ Statistics track both end reasons
✅ Reset clears all state
✅ All tests pass