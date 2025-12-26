**CRITICAL NOTE CAPTURED: AMD 7900 XT (ROCm, not CUDA)**

This changes our STT approach:

**Affected Components:**
- ❌ `faster-whisper` (CTranslate2 = CUDA-only)
- ✅ `Ollama` (excellent ROCm support)
- ✅ `silero-vad` (PyTorch = ROCm compatible)
- ✅ `Piper` (CPU-only, no issue)

**STT Solution Options:**
1. **whisper.cpp** (ROCm support, fastest)
2. **Original OpenAI Whisper** (PyTorch + ROCm, proven)
3. **Faster-Whisper via ROCm** (experimental, risky)

**Recommendation:** Use **whisper.cpp with ROCm** for STT-001 through STT-004. It's the fastest and has solid AMD support.

---

# Dependency Matrix: Workshop Phase 2

```
Legend: X = Row depends on Column

FEATURE     |AUDIO|AUDIO|AUDIO|VAD |VAD |VAD |WAKE|WAKE|STT |STT |STT |STT |LLM |LLM |LLM |LLM |TTS |TTS |TTS |TTS |ORCH|ORCH|ORCH|ORCH|UI  |UI  |UI  |
            | 001 | 002 | 003 | 001| 002| 003| 001| 002| 001| 002| 003| 004| 001| 002| 003| 004| 001| 002| 003| 004| 001| 002| 003| 004| 001| 002| 003|
------------|-----|-----|-----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
AUDIO-001   |  -  |     |     |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
AUDIO-002   |  X  |  -  |     |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
AUDIO-003   |     |     |  -  |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
VAD-001     |  X  |  X  |     |  - |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
VAD-002     |     |     |     |  X |  - |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
VAD-003     |     |     |     |  X |    |  - |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
WAKE-001    |  X  |  X  |     |    |    |    |  - |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
WAKE-002    |     |     |     |    |    |    |    |  - |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
STT-001     |     |     |     |    |    |    |    |  X |  - |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
STT-002     |  X  |  X  |     |  X |    |    |    |    |  X |  - |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
STT-003     |     |     |     |    |    |    |    |    |    |  X |  - |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
STT-004     |     |     |     |    |  X |    |    |    |    |    |  X |  - |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
LLM-001     |     |     |     |    |    |    |    |    |    |    |    |    |  - |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
LLM-002     |     |     |     |    |    |    |    |    |    |    |    |    |  X |  - |    |    |    |    |    |    |    |    |    |    |    |    |    |
LLM-003     |     |     |     |    |    |    |    |    |    |    |    |    |    |  X |  - |    |    |    |    |    |    |    |    |    |    |    |    |
LLM-004     |     |     |     |    |    |    |    |    |    |    |    |    |    |  X |    |  - |    |    |    |    |    |    |    |    |    |    |    |
TTS-001     |     |     |     |    |    |    |    |    |    |    |    |    |    |    |    |    |  - |    |    |    |    |    |    |    |    |    |    |
TTS-002     |     |     |     |    |    |    |    |    |    |    |    |    |    |    |  X |    |  X |  - |    |    |    |    |    |    |    |    |    |
TTS-003     |     |     |  X  |    |    |    |    |    |    |    |    |    |    |    |    |    |    |  X |  - |    |    |    |    |    |    |    |    |
TTS-004     |     |     |     |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |  X |  - |    |    |    |    |    |    |    |
ORCH-001    |     |     |     |    |    |    |  X |    |    |    |    |  X |    |    |    |    |    |    |    |    |  - |    |    |    |    |    |    |
ORCH-002    |  X  |  X  |     |  X |    |    |  X |    |    |    |    |    |    |    |    |    |    |    |    |    |  X |  - |    |    |    |    |    |
ORCH-003    |     |     |  X  |    |    |    |    |    |    |    |    |  X |  X |  X |  X |    |  X |  X |  X |    |  X |    |  - |    |    |    |    |
ORCH-004    |     |     |     |    |    |  X |    |    |    |    |    |    |    |    |    |  X |    |    |    |  X |  X |    |  X |  - |    |    |    |
UI-001      |     |     |     |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |  X |    |    |    |  - |    |    |
UI-002      |     |     |     |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |  X |    |    |    |    |  - |    |
UI-003      |     |     |     |    |    |    |    |    |    |    |  X |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |  - |
```

---

# Implementation Plan: Workshop Phase 2

## Phase 1: Foundation (No Dependencies)
**Goal:** Core audio, models loaded, basic state management

### Features (8)
1. **AUDIO-001:** Continuous audio stream capture
2. **AUDIO-003:** Audio playback pipeline
3. **WAKE-002:** Wake word model loading
4. **STT-001:** Streaming Whisper setup (whisper.cpp + ROCm)
5. **LLM-001:** Ollama streaming client
6. **TTS-001:** Piper streaming setup
7. **ORCH-001:** State machine (idle/listening/processing/speaking)

**Validation Gate:**
- ✅ Audio captures from microphone
- ✅ Audio plays through speakers
- ✅ Wake word model loaded
- ✅ whisper.cpp initialized
- ✅ Ollama connected
- ✅ Piper initialized
- ✅ State machine transitions work

**Estimated Time:** 4-6 hours

---

## Phase 2: Detection Layer
**Goal:** VAD, wake word, frame processing working

### Features (5)
8. **AUDIO-002:** Audio frame processing
9. **VAD-001:** Real-time VAD using silero-vad
10. **VAD-002:** Speech end detection
11. **VAD-003:** Interruption detection
12. **WAKE-001:** "Workshop" wake word detection

**Validation Gate:**
- ✅ VAD detects speech vs silence accurately
- ✅ Speech end detected within 100ms of silence
- ✅ "Workshop" wake word triggers activation (<5% false positives)
- ✅ Interruptions detected while system speaks

**Estimated Time:** 3-4 hours

---

## Phase 3: STT Pipeline
**Goal:** Speech converts to text with streaming

### Features (3)
13. **STT-002:** Chunk-based transcription
14. **STT-003:** Transcription buffering
15. **STT-004:** Final transcription refinement

**Validation Gate:**
- ✅ Partial transcriptions emit during speech
- ✅ Final transcription accurate (>90%)
- ✅ Transcription completes <200ms after speech end

**Estimated Time:** 4-5 hours

---

## Phase 4: LLM Streaming
**Goal:** Model generates responses token-by-token

### Features (3)
16. **LLM-002:** Token-by-token generation
17. **LLM-003:** Sentence boundary detection
18. **LLM-004:** Generation interruption

**Validation Gate:**
- ✅ Tokens stream from Ollama
- ✅ Sentences detected correctly
- ✅ Generation stops cleanly on interrupt
- ✅ First token arrives <300ms after transcription

**Estimated Time:** 2-3 hours

---

## Phase 5: TTS Pipeline
**Goal:** Text becomes audio, streams to playback

### Features (3)
19. **TTS-002:** Sentence-by-sentence synthesis
20. **TTS-003:** Audio buffer management
21. **TTS-004:** Synthesis interruption

**Validation Gate:**
- ✅ Each sentence synthesizes independently
- ✅ Audio plays while next sentence generates
- ✅ Synthesis stops immediately on interrupt
- ✅ First audio plays <500ms after speech end

**Estimated Time:** 3-4 hours

---

## Phase 6: Full Integration
**Goal:** All pipelines coordinated, system works end-to-end

### Features (3)
22. **ORCH-002:** Audio buffer coordination
23. **ORCH-003:** Pipeline coordination (LLM→TTS→Audio)
24. **ORCH-004:** Interruption handling across all components

**Validation Gate:**
- ✅ Wake word → Listen → Transcribe → Generate → Speak (full loop)
- ✅ Interruptions work at any point
- ✅ Total response time <1 second
- ✅ No audio glitches or dropped frames

**Estimated Time:** 4-6 hours

---

## Phase 7: UI & Polish
**Goal:** Visual feedback, final tuning

### Features (3)
25. **UI-001:** Listening indicator
26. **UI-002:** Speaking indicator
27. **UI-003:** Transcription display

**Validation Gate:**
- ✅ Visual feedback matches system state
- ✅ User knows when to speak
- ✅ Transcription visible in real-time

**Estimated Time:** 2-3 hours

---

## Total Estimated Time: 22-31 hours

**Critical Path:** AUDIO-001 → AUDIO-002 → VAD-001 → STT-002 → STT-003 → STT-004 → LLM-002 → LLM-003 → TTS-002 → TTS-003 → ORCH-003

---

# READY TO START BUILDING

**Phase 1, Feature 1: AUDIO-001 (Continuous audio stream capture)**

Should I generate the feature spec for AUDIO-001 and start implementing immediately?

Or do you want to review the plan first?

**Say "GO" and I start coding.**