# Workshop Phase 2: Integration Testing Guide

**Manual testing procedures to validate the complete voice input system**

---

## 🎯 Testing Objectives

1. Verify all components work together in real conditions
2. Test with actual microphone input
3. Validate user experience flow
4. Identify any integration issues
5. Confirm system ready for Phase 3

---

## 📋 Pre-Test Checklist

### Hardware Setup
- [ ] Microphone connected and working
- [ ] Test microphone: `python -c "import sounddevice; print(sounddevice.query_devices())"`
- [ ] Verify default input device set correctly
- [ ] Quiet environment (minimal background noise)

### Software Dependencies
- [ ] All Phase 1 components working (tested previously)
- [ ] Phase 2 components installed (`vad.py`, `audio_pipeline.py`, `wake_pipeline.py`)
- [ ] Models downloaded (Silero VAD, OpenWakeWord)
- [ ] Python environment active

### Quick Sanity Check
```bash
cd /home/claude
python -c "import vad; import audio_pipeline; import wake_pipeline; print('✅ All imports work')"
```

---

## 🧪 Test Suite

### Test 1: Basic VAD Detection (5 minutes)
**Purpose:** Verify voice activity detection works with your microphone

**Procedure:**
1. Run: `python integration_test_1_vad.py`
2. Wait for "Listening for speech..."
3. **SPEAK** into microphone for 2-3 seconds
4. **PAUSE** for 1 second
5. Observe console output

**Expected Behavior:**
- ✅ Console shows "🎤 SPEECH DETECTED" when you speak
- ✅ Console shows "🔇 Silence" when you pause
- ✅ Speech probability shown (should be >0.5 when speaking)
- ✅ No false positives from ambient noise

**What to Check:**
- Does it detect your voice consistently?
- Does it ignore background noise?
- Is the threshold (0.5) appropriate or too sensitive?

**Troubleshooting:**
- If no detection: Check microphone input level
- If too sensitive: Increase threshold to 0.6
- If not sensitive enough: Decrease threshold to 0.4

---

### Test 2: Speech Segment Capture (5 minutes)
**Purpose:** Verify complete speech segments are captured

**Procedure:**
1. Run: `python integration_test_2_segment.py`
2. Wait for "Speak a command..."
3. **SPEAK**: "Hello Workshop, how are you today?"
4. **PAUSE** for 1 second
5. Observe segment capture

**Expected Behavior:**
- ✅ Captures speech after you pause
- ✅ Console shows segment duration (should match your speech length)
- ✅ End reason: "natural_pause"
- ✅ Segment contains your complete utterance

**What to Check:**
- Does it wait for you to finish speaking?
- Does it capture the complete sentence?
- Is the pause duration (300ms) appropriate?

**Repeat with variations:**
- Short utterance: "Hello"
- Long utterance: Speak for 10+ seconds continuously
- Multiple sentences: "This is sentence one. This is sentence two."

---

### Test 3: Timeout Detection (3 minutes)
**Purpose:** Verify timeout stops long speech

**Procedure:**
1. Run: `python integration_test_3_timeout.py` (5 second timeout)
2. Wait for "Speak continuously..."
3. **SPEAK** continuously for 10+ seconds without pausing
4. Observe timeout trigger

**Expected Behavior:**
- ✅ Segment captured after ~5 seconds
- ✅ End reason: "timeout"
- ✅ Segment duration ~5 seconds (not full 10 seconds)

**What to Check:**
- Does timeout enforce the limit?
- Is the timeout duration appropriate for commands?

---

### Test 4: Wake Word Detection (5 minutes)
**Purpose:** Verify "workshop" activates listening

**Procedure:**
1. Run: `python integration_test_4_wake.py`
2. System in IDLE, waiting for wake word
3. **SAY**: "workshop"
4. Observe wake detection
5. System switches to LISTENING
6. **SAY**: "what time is it"
7. Observe speech capture

**Expected Behavior:**
- ✅ Console shows "🎤 WAKE WORD DETECTED!" when you say "workshop"
- ✅ System transitions IDLE → LISTENING
- ✅ Captures your command after wake
- ✅ Returns speech segment

**What to Check:**
- Does "workshop" consistently trigger?
- Does it ignore similar words ("work", "workshop")?
- Does it work from different distances/volumes?

**Troubleshooting:**
- If wake word not detected: Try speaking louder/clearer
- If too many false triggers: Adjust wake word threshold
- If inconsistent: Try different wake word model

---

### Test 5: Interruption Handling (5 minutes)
**Purpose:** Verify you can interrupt during TTS

**Procedure:**
1. Run: `python integration_test_5_interrupt.py`
2. System plays TTS (simulated with beeps or actual Piper)
3. While TTS playing, **SPEAK** into microphone
4. Observe interruption detection
5. TTS should stop

**Expected Behavior:**
- ✅ Console shows "🛑 INTERRUPTION DETECTED!" when you speak
- ✅ TTS/playback stops immediately
- ✅ System switches to listening mode
- ✅ Captures your interrupting speech

**What to Check:**
- Is interruption detection fast? (<200ms)
- Does it require confirmation (3 frames = ~96ms of speech)?
- Does ambient noise cause false interruptions?

**Troubleshooting:**
- If too sensitive: Increase interruption_threshold (0.6 → 0.7)
- If not detecting: Check microphone input while audio playing
- If delayed: Check audio playback latency

---

### Test 6: Full Conversation Flow (10 minutes)
**Purpose:** Validate complete interaction cycle

**Procedure:**
1. Run: `python integration_test_6_full.py`
2. System in IDLE
3. **SAY**: "workshop"
4. Wait for wake confirmation
5. **SAY**: "what is the weather like"
6. System captures, "transcribes" (simulated), responds
7. Observe state transitions: IDLE → LISTENING → PROCESSING → SPEAKING → IDLE
8. **REPEAT** 2-3 times

**Expected Behavior:**
- ✅ Wake word triggers listening every time
- ✅ Commands captured completely
- ✅ State transitions logged correctly
- ✅ Can chain multiple interactions
- ✅ No hangs or crashes

**What to Check:**
- Does the flow feel natural?
- Are transitions smooth?
- Is timing appropriate?
- Does it handle back-to-back commands?

---

### Test 7: Edge Cases & Error Recovery (10 minutes)
**Purpose:** Test system handles unusual situations

**Test 7a: No Speech After Wake**
1. Run integration test
2. **SAY**: "workshop" (wake word)
3. **DON'T SPEAK** for 30 seconds
4. Observe timeout behavior

**Expected:** Returns to IDLE after timeout

---

**Test 7b: Background Noise**
1. Run integration test
2. Play music or have conversation in background
3. **SAY**: "workshop"
4. **SAY**: "command"
5. Observe detection accuracy

**Expected:** Still detects wake word and command despite noise

---

**Test 7c: Partial Wake Word**
1. Run integration test
2. **SAY**: "work" (not full wake word)
3. Observe no activation

**Expected:** Stays in IDLE, doesn't false trigger

---

**Test 7d: Rapid Speech**
1. Run integration test
2. **SAY**: "workshop what time is it" (no pause)
3. Observe handling

**Expected:** Detects wake, captures "what time is it" as command

---

**Test 7e: Multiple People Speaking**
1. Run integration test (if possible with helper)
2. Person A: "workshop"
3. Person B: speaks command
4. Observe whose voice is captured

**Expected:** Captures whoever speaks (non-speaker-dependent)

---

## 📊 Test Results Template

### Test Session Log

**Date:** ___________  
**Tester:** ___________  
**Environment:** ___________  

| Test | Status | Notes | Issues |
|------|--------|-------|--------|
| Test 1: VAD Detection | ☐ Pass ☐ Fail | | |
| Test 2: Segment Capture | ☐ Pass ☐ Fail | | |
| Test 3: Timeout | ☐ Pass ☐ Fail | | |
| Test 4: Wake Word | ☐ Pass ☐ Fail | | |
| Test 5: Interruption | ☐ Pass ☐ Fail | | |
| Test 6: Full Flow | ☐ Pass ☐ Fail | | |
| Test 7a: No Speech | ☐ Pass ☐ Fail | | |
| Test 7b: Background Noise | ☐ Pass ☐ Fail | | |
| Test 7c: Partial Wake | ☐ Pass ☐ Fail | | |
| Test 7d: Rapid Speech | ☐ Pass ☐ Fail | | |
| Test 7e: Multiple People | ☐ Pass ☐ Fail | | |

**Overall Assessment:**
- [ ] Ready for Phase 3
- [ ] Needs minor adjustments
- [ ] Major issues found

**Issues Found:**
1. 
2. 
3. 

**Recommended Tuning:**
- VAD Threshold: _____
- Interruption Threshold: _____
- Timeout Duration: _____
- Min Silence Duration: _____

---

## 🎚️ Configuration Tuning Guide

Based on test results, you may need to adjust parameters:

### If Speech Detection Too Sensitive (False Positives)
```python
# Increase VAD threshold
vad = VoiceActivityDetector(threshold=0.6)  # was 0.5

# Increase confirmation frames
detector = InterruptionDetector(confirmation_frames=5)  # was 3
```

### If Speech Detection Not Sensitive Enough (Missed Speech)
```python
# Decrease VAD threshold
vad = VoiceActivityDetector(threshold=0.4)  # was 0.5

# Decrease min speech duration
vad = VoiceActivityDetector(min_speech_duration_ms=150)  # was 250
```

### If Speech Cuts Off Too Early
```python
# Increase silence duration
vad = VoiceActivityDetector(min_silence_duration_ms=500)  # was 300
```

### If Commands Feel Too Long to Capture
```python
# Decrease timeout
pipeline = AudioFramePipeline(timeout_s=15.0)  # was 30.0
```

### If Wake Word Too Sensitive
```python
# Adjust in wake_word.py or try different model
detector = WakeWordDetector(threshold=0.6)  # if supported
```

---

## 🐛 Common Issues & Solutions

### Issue: Microphone not detected
**Symptoms:** No audio input, errors about audio device  
**Solution:**
```bash
# List devices
python -c "import sounddevice; print(sounddevice.query_devices())"

# Set default device (if needed)
# In audio_realtime.py, specify device_index
```

### Issue: VAD always detects silence
**Symptoms:** Never transitions to speaking state  
**Solution:**
- Check microphone input level (too quiet?)
- Lower threshold: `VoiceActivityDetector(threshold=0.3)`
- Test with louder speech

### Issue: VAD always detects speech
**Symptoms:** Never transitions to silence  
**Solution:**
- Check for background noise
- Increase threshold: `VoiceActivityDetector(threshold=0.7)`
- Move to quieter environment

### Issue: Wake word not detected
**Symptoms:** "workshop" doesn't trigger  
**Solution:**
- Speak clearly and at normal volume
- Ensure wake word model loaded
- Try different pronunciations
- Check model file path

### Issue: Interruption not working
**Symptoms:** Can't stop TTS by speaking  
**Solution:**
- Verify `set_assistant_speaking(True)` called
- Check interruption threshold (might be too high)
- Ensure microphone active during TTS playback

### Issue: Choppy speech detection
**Symptoms:** Speech state flickers on/off  
**Solution:**
- Increase smoothing frames in VAD
- Adjust min_speech_duration_ms
- Check for audio buffer issues

---

## ✅ Success Criteria

Phase 2 is ready for Phase 3 when:

### Functional Requirements
- [ ] Wake word triggers >90% of the time
- [ ] Speech segments captured completely
- [ ] Timeout enforces limit correctly
- [ ] Interruptions detected <200ms latency
- [ ] No crashes during normal use
- [ ] Error recovery works (timeouts, no speech)

### Performance Requirements
- [ ] Wake detection latency <500ms
- [ ] Speech capture feels natural (300-500ms pause)
- [ ] Interruption response <200ms
- [ ] CPU usage reasonable (<20% on laptop)
- [ ] No audio glitches/dropouts

### User Experience Requirements
- [ ] Interaction feels conversational
- [ ] Timing feels natural (not too fast/slow)
- [ ] False positives rare (<5%)
- [ ] System responsive and predictable

---

## 📝 Post-Test Actions

After completing integration tests:

1. **Document Issues:**
   - Log any bugs found
   - Note configuration needed
   - Record unexpected behaviors

2. **Apply Tuning:**
   - Adjust thresholds based on test results
   - Update default parameters
   - Re-test critical scenarios

3. **Create Configuration File:**
   - Save tuned parameters
   - Document why each value chosen
   - Create config file for easy loading

4. **Update Documentation:**
   - Add any new findings
   - Update troubleshooting guide
   - Record performance metrics

5. **Plan Phase 3:**
   - Based on integration experience
   - Identify pain points to address
   - Prioritize features

---

## 🚀 Ready for Phase 3?

**Sign-off Checklist:**
- [ ] All integration tests passing
- [ ] Configuration tuned for your environment
- [ ] No critical bugs blocking usage
- [ ] Performance acceptable
- [ ] User experience feels natural
- [ ] Documentation updated

**Signed:** ___________  
**Date:** ___________

---

## 📞 Need Help?

If you encounter issues during integration testing:

1. **Check logs:** Enable DEBUG logging for detailed output
2. **Run unit tests:** Verify components individually still work
3. **Simplify:** Test components in isolation
4. **Document:** Record exact steps to reproduce issue
5. **Adjust:** Try configuration tuning suggestions above

**Remember:** Integration testing often reveals edge cases unit tests miss. This is normal and expected! Document everything you find.

---

**Good luck with integration testing! 🎉**

Run the tests, document your findings, and we'll address any issues before Phase 3!