# Workshop Development Roadmap (Post-Research)

Based on professional research findings. Prioritized by impact and cost.

---

## Immediate Actions (No Hardware Cost)

### 1. ✅ UI: Switch to Hybrid Rendering
**Status**: Implemented

- WebGL for ambient 3D effects (particles, glow, grid)
- HTML overlay for panels (code, terminal, file tree)
- Panels positioned via CSS transforms with parallax from camera movement
- Depth through effects (blur, shadows, opacity) not Z-position

**Why**: CSS3DRenderer is fundamentally broken for text. Hybrid is how AAA games do HUDs.

### 2. Voice Stack Upgrade
**Priority**: High  
**Effort**: 2-4 hours

| Current | Upgrade To | Benefit |
|---------|------------|---------|
| Vosk (wake word) | OpenWakeWord | More accurate, trainable, free |
| Whisper | faster-whisper distil-large-v3 | 5.8x faster, ~4GB VRAM |
| Ollama (llama3.1) | Qwen3-8B | Best function calling accuracy |
| None | Piper TTS | Voice responses, CPU-only |

**Implementation**:
```python
# faster-whisper
from faster_whisper import WhisperModel
model = WhisperModel("distil-large-v3", device="cuda", compute_type="int8_float16")
segments, _ = model.transcribe(audio, beam_size=5, vad_filter=True)

# Piper TTS
piper --model en_US-lessac-medium --output_file response.wav < text.txt
```

### 3. Input Abstraction Layer
**Priority**: Medium  
**Effort**: 4-6 hours

Map all inputs (voice, keyboard, gesture) to unified **intents**:

```python
class InputIntent:
    SELECT = "select"
    NAVIGATE = "navigate"
    ACTIVATE = "activate"
    CANCEL = "cancel"
    COMMAND = "command"

# Voice: "select the battery guardian file" → Intent(SELECT, target="battery_guardian")
# Keyboard: Enter → Intent(ACTIVATE)
# Gesture: Pinch → Intent(SELECT)
```

**Why**: Enables seamless addition of gesture input later without changing core logic.

### 4. Display Abstraction Layer
**Priority**: Medium  
**Effort**: 4-6 hours

```python
class DisplayTarget(Protocol):
    def render(self, constructs: list[Construct]) -> None: ...
    def get_capabilities(self) -> DisplayCapabilities: ...

class ScreenDisplay(DisplayTarget): ...      # Current Electron UI
class ProjectionDisplay(DisplayTarget): ...  # Future
class ARDisplay(DisplayTarget): ...          # Future
```

**Why**: Enables switching between screen, projection, AR without rewriting construct logic.

### 5. Consider MCP (Model Context Protocol)
**Priority**: Low (nice to have)  
**Effort**: 8-12 hours

Anthropic's standard for tool integration. Benefits:
- Add tools without modifying core Workshop code
- Growing ecosystem of pre-built servers
- Works with any LLM

**Trade-off**: Current tool registry works fine. MCP adds complexity for extensibility we may not need yet.

---

## Phase 2: Gesture Input (Future, ~$139)

When ready to add hand tracking:

**Hardware**: Leap Motion Controller 2 ($139)
- IR-based, works in variable workshop lighting
- 115fps, 27 joints per hand
- Best-in-class for desk-level interaction

**Fallback**: MediaPipe Hands (free, webcam-based)
- 21 landmarks, ~30fps
- Good enough for basic gestures
- Use when Leap Motion not connected

**Gesture Vocabulary**:
| Gesture | Action |
|---------|--------|
| Point | Raycast/cursor |
| Pinch | Select/click |
| Open palm | Cancel |
| Thumbs up | Confirm/voice toggle |

---

## Phase 3: Projection SAR (Future, ~$1,000-2,000)

For spatial computing in workshop environment:

**Hardware**:
- Projector: 3,000+ lumens, 1080p+, short-throw
- Depth camera: Intel RealSense D455 (~$350)

**Software**:
- TouchDesigner (free for non-commercial)
- OpenCV for calibration
- Custom calibration using gray code patterns

**Why Projection Over AR Glasses**:
- Multi-user (anyone can see)
- Hands-free (nothing to wear)
- Full peripheral vision (workshop safety)
- No battery/comfort limits

---

## Architecture Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| UI Rendering | Hybrid (WebGL + HTML) | Text quality, native interactions |
| STT | faster-whisper distil-large-v3 | Speed, accuracy, VRAM |
| LLM | Qwen3-8B | Function calling accuracy |
| TTS | Piper | CPU-only, instant |
| Wake word | OpenWakeWord | Free, trainable |
| Gesture (future) | Leap Motion 2 | IR-based, workshop-safe |
| Spatial (future) | Projection SAR | Multi-user, hands-free |

---

## VRAM Budget (20GB Available)

| Component | VRAM |
|-----------|------|
| faster-whisper distil-large-v3 | ~4GB |
| Qwen3-8B | ~8GB |
| Piper TTS | 0 (CPU) |
| **Total** | ~12GB |
| **Headroom** | 8GB |

---

## Immediate TODO

1. [ ] Test hybrid UI with real constructs
2. [ ] Install faster-whisper, run benchmark vs current Whisper
3. [ ] Try Qwen3-8B via Ollama for function calling
4. [ ] Install Piper, test voice output
5. [ ] Create InputIntent abstraction in agent.py
6. [ ] Create DisplayTarget abstraction in construct_manager.py
