# Master Project Specification: Workshop Phase 2 - Real-Time Voice

**Generated:** 2025-12-13  
**Target:** Production-ready streaming voice interface

---

## 1. Project Purpose

**Problem:** Phase 1's batch processing creates multi-second latency and walkie-talkie interaction patterns. Users must wait for silence detection, full transcription, complete LLM generation, and full TTS synthesis before hearing responses. This breaks the flow of thinking out loud.

**Who:** Christian doing software development, business development, and rapid technical work requiring real-time AI collaboration.

**Value:** Real-time voice communication cuts development time by double-digit percentages. Initial development accelerates, but the bigger unlock is debugging and problem-solving through natural think-out-loud workflows. Voice responses must feel conversational, not transactional.

---

## 2. Essential Functionality

1. **Natural Conversation Flow** - Interrupt mid-sentence, system stops and listens immediately
2. **Instant Response Initiation** - <500ms from speech end to first audio output
3. **Streaming Architecture** - Every component streams (VAD → Whisper → LLM → TTS) with no batch boundaries
4. **Real-Time Backchanneling** - Model provides acknowledgment ("uh huh", "yep") while user speaks (stretch goal)

---

## 3. Scope Boundaries

### NOW (Make It Work)
- ✅ Streaming Whisper transcription (overlapping chunks, partial results)
- ✅ Streaming LLM response (token-by-token generation)
- ✅ Streaming TTS synthesis (sentence-by-sentence audio)
- ✅ Voice Activity Detection (<1ms per audio chunk)
- ✅ Wake word detection ("Workshop")
- ✅ Interruption handling (stop playback immediately on new speech)
- ✅ Visual feedback (listening indicator, speaking indicator)
- ✅ <1 second total response time (speech end → first audio)

### NOT (Out of Scope - Phase 2)
- ❌ Gesture recognition
- ❌ Visual constructs / spatial UI
- ❌ Hardware integration
- ❌ Multi-user support
- ❌ Cloud services

### NEXT (Phase 3)
- 🔮 Automated context engineering workflows
- 🔮 Workflow detection and context assembly
- 🔮 MediaPipe hand tracking
- 🔮 Spatial computing layer

---

## 4. Technical Context

**Platform:**
- Local execution only
- GPU: 20GB VRAM limit (shared across Whisper + Ollama + any other models)
- CPU: Available for VAD, audio processing, orchestration

**Latency Targets:**
- Voice Activity Detection: <1ms per audio frame
- Wake word detection: <100ms
- Transcription start: <200ms from speech end
- LLM first token: <300ms from transcription complete
- TTS first audio: <500ms total from speech end
- Interruption response: <100ms (immediate playback stop)

**Components:**
- `silero-vad` - Voice activity detection
- `OpenWakeWord` or `Porcupine` - Wake word detection
- `faster-whisper` - Streaming transcription
- `Ollama` - Streaming LLM inference
- `Piper` - Streaming TTS (or investigate alternatives)
- Audio I/O via `sounddevice`

---

## 5. Workflow Details

### Workflow 1: Wake Word Activation
**Goal:** Detect "Workshop" wake word and enter listening mode

**Steps:**
1. Background audio stream continuously feeds wake word detector
2. "Workshop" detected → trigger activation
3. Visual indicator shows "listening" state
4. Audio buffer begins capturing for transcription

**Expected Outcome:** User sees listening indicator within 100ms of saying "Workshop"

---

### Workflow 2: Real-Time Speech-to-Intent
**Goal:** Convert user speech to text and model context as quickly as possible

**Steps:**
1. VAD detects speech activity
2. Audio chunks (100-200ms) feed streaming Whisper
3. Partial transcriptions emit immediately
4. Speech end detected (VAD silence threshold)
5. Final transcription refinement (if needed)
6. Transcription sent to LLM

**Expected Outcome:** LLM receives complete user intent <200ms after user stops speaking

---

### Workflow 3: Streaming LLM Response
**Goal:** Generate and deliver response with minimal perceivable latency

**Steps:**
1. LLM receives user transcription + conversation context
2. Token generation begins immediately (streaming mode)
3. Tokens accumulate into sentences
4. Each complete sentence sent to TTS pipeline
5. While sentence N plays, sentence N+1 generates

**Expected Outcome:** User hears first audio output <500ms after they stop speaking

---

### Workflow 4: Interruption Handling
**Goal:** Stop current response immediately when user speaks again

**Steps:**
1. VAD detects new speech while system is speaking
2. Audio playback stops within 100ms
3. TTS pipeline clears buffered audio
4. LLM generation pauses (or continues silently for context)
5. System enters listening mode
6. New transcription captures user's interruption
7. LLM receives interruption context + continues

**Expected Outcome:** System stops talking and starts listening instantly

---

### Workflow 5: Real-Time Backchanneling (Stretch)
**Goal:** Provide acknowledgment while user speaks (like a human listener)

**Steps:**
1. VAD detects user is speaking (>1 second continuous speech)
2. Streaming transcription provides partial context
3. LLM generates brief acknowledgments ("mm-hmm", "right", "okay")
4. Acknowledgments play without interrupting user speech
5. Acknowledgment timing based on natural pause points

**Expected Outcome:** User feels heard in real-time, system feels conversational

---

## 6. Success Criteria

**Quantitative:**
- Wake word detection accuracy: >95%
- False positive rate: <5% (shouldn't activate on "workshop" in normal conversation)
- Transcription accuracy: >90% (faster-whisper base.en baseline)
- Response initiation: <500ms (speech end → first audio)
- Total response time: <1 second (speech end → user hears complete first sentence)
- Interruption response: <100ms (speech start → playback stop)

**Qualitative:**
- Feels natural (testers say "I can think out loud")
- No noticeable lag (testers don't mention latency unprompted)
- Interruptions feel instant (testers say "it just stops")
- Visual feedback is clear (testers know when to speak)

---

## 7. Constraints

**Hardware:**
- 20GB VRAM shared budget
- Whisper base.en: ~1GB VRAM
- Ollama (llama3.1:8b): ~5GB VRAM
- Room for optimization: ~14GB available

**Latency Budget Breakdown:**
```
User stops speaking                    T=0ms
├─ VAD detects silence                T=50ms   (50ms)
├─ Whisper final transcription        T=200ms  (150ms)
├─ LLM first token generated          T=350ms  (150ms)
├─ TTS first sentence synthesized     T=450ms  (100ms)
└─ First audio plays                  T=500ms  (50ms buffer)
```

**Dependencies:**
- Python 3.10+
- CUDA-capable GPU
- ~10GB disk space for models

---

**END OF MASTER PROJECT SPEC**

---

# Next: Feature Extraction (5 minutes)

I'm now going to extract atomic features from this spec. Hold tight—generating feature inventory NOW.

# Feature Inventory: Workshop Phase 2

**Generated:** 2025-12-13  
**Total Features:** 20 core + 3 stretch

---

## Category: AUDIO (Audio I/O Infrastructure)

| ID | Feature | Complexity | Description |
|----|---------|------------|-------------|
| AUDIO-001 | Continuous audio stream capture | Easy | Capture audio from microphone in continuous stream using sounddevice |
| AUDIO-002 | Audio frame processing | Easy | Process audio in fixed-size frames (512-1024 samples) for VAD/wake word |
| AUDIO-003 | Audio playback pipeline | Medium | Play synthesized audio with low latency, support immediate abort |

---

## Category: VAD (Voice Activity Detection)

| ID | Feature | Complexity | Description |
|----|---------|------------|-------------|
| VAD-001 | Real-time VAD using silero-vad | Medium | Detect speech vs silence in <1ms per frame |
| VAD-002 | Speech end detection | Easy | Detect when user stops speaking (configurable silence threshold) |
| VAD-003 | Interruption detection | Medium | Detect new speech while system is speaking |

---

## Category: WAKE (Wake Word Detection)

| ID | Feature | Complexity | Description |
|----|---------|------------|-------------|
| WAKE-001 | "Workshop" wake word detection | Medium | Detect activation phrase with <5% false positive rate |
| WAKE-002 | Wake word model loading | Easy | Load and initialize wake word detector on startup |

---

## Category: STT (Speech-to-Text)

| ID | Feature | Complexity | Description |
|----|---------|------------|-------------|
| STT-001 | Streaming Whisper setup | Medium | Initialize faster-whisper in streaming mode |
| STT-002 | Chunk-based transcription | Hard | Transcribe overlapping audio chunks, emit partial results |
| STT-003 | Transcription buffering | Medium | Accumulate partial transcriptions into coherent text |
| STT-004 | Final transcription refinement | Easy | Clean up and finalize transcription on speech end |

---

## Category: LLM (Language Model Integration)

| ID | Feature | Complexity | Description |
|----|---------|------------|-------------|
| LLM-001 | Ollama streaming client | Easy | Connect to Ollama with streaming enabled |
| LLM-002 | Token-by-token generation | Easy | Receive and process individual tokens as generated |
| LLM-003 | Sentence boundary detection | Medium | Detect sentence completion from token stream |
| LLM-004 | Generation interruption | Medium | Pause/stop generation when user interrupts |
| LLM-005 | Backchannel generation | Hard | Generate brief acknowledgments during user speech (STRETCH) |

---

## Category: TTS (Text-to-Speech)

| ID | Feature | Complexity | Description |
|----|---------|------------|-------------|
| TTS-001 | Piper streaming setup | Medium | Initialize Piper for sentence-level synthesis |
| TTS-002 | Sentence-by-sentence synthesis | Medium | Synthesize audio for each complete sentence immediately |
| TTS-003 | Audio buffer management | Medium | Queue and play audio chunks with minimal latency |
| TTS-004 | Synthesis interruption | Easy | Stop synthesis pipeline immediately on interrupt |
| TTS-005 | Backchannel synthesis | Easy | Synthesize brief acknowledgments quickly (STRETCH) |

---

## Category: ORCH (Orchestration & State Management)

| ID | Feature | Complexity | Description |
|----|---------|------------|-------------|
| ORCH-001 | State machine (idle/listening/processing/speaking) | Medium | Manage system state transitions and valid actions |
| ORCH-002 | Audio buffer coordination | Medium | Coordinate audio capture between wake word, VAD, and STT |
| ORCH-003 | Pipeline coordination (LLM→TTS→Audio) | Hard | Coordinate streaming from LLM through TTS to audio playback |
| ORCH-004 | Interruption handling | Hard | Handle interruptions cleanly across all components |
| ORCH-005 | Backchannel timing | Hard | Time backchannels to not overlap user speech (STRETCH) |

---

## Category: UI (Visual Feedback)

| ID | Feature | Complexity | Description |
|----|---------|------------|-------------|
| UI-001 | Listening indicator | Easy | Visual indicator showing system is listening |
| UI-002 | Speaking indicator | Easy | Visual indicator showing system is speaking |
| UI-003 | Transcription display | Easy | Show real-time transcription of user speech |

---

## Summary Stats

**By Complexity:**
- Easy: 9 features
- Medium: 14 features  
- Hard: 4 features

**By Category:**
- AUDIO: 3 features
- VAD: 3 features
- WAKE: 2 features
- STT: 4 features
- LLM: 5 features (3 core + 2 stretch)
- TTS: 5 features (3 core + 2 stretch)
- ORCH: 5 features (3 core + 2 stretch)
- UI: 3 features

**Core: 20 features | Stretch: 3 features**

---
