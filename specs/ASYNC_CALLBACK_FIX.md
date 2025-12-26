# Async Callback Fix - Summary

## Problem Description

After the queue exhaustion fix, the system would detect wake words and capture speech successfully, but the callbacks (`_on_wake` and `_on_speech`) would never execute, causing the system to hang in the "processing" state with no transcription or agent response.

## Root Cause

The issue was an **event loop management problem** in the async/sync boundary:

1. `main.py` runs `run_voice_mode_phase2()` as an async function using `asyncio.run()`
2. `run_voice_mode_phase2()` called `wake_pipeline.run()` - a **synchronous blocking function**
3. This **blocked the entire event loop**, preventing any async tasks from running
4. When `wake_pipeline` tried to call async callbacks (`_on_wake`, `_on_speech`), it attempted to create tasks or run them in the blocked loop
5. Result: **deadlock** - callbacks never executed, system hung

### Technical Details

**Before (Broken):**
```python
# main.py:170 (BEFORE)
async def run_voice_mode_phase2(self):
    ...
    self.wake_pipeline.run()  # ❌ BLOCKS event loop!
```

```python
# wake_pipeline.py:181 (BEFORE)
if loop.is_running():
    asyncio.create_task(self.on_speech(segment, reason))  # ❌ Doesn't wait
else:
    loop.run_until_complete(self.on_speech(segment, reason))  # ❌ Fails
```

**Problems:**
- `wake_pipeline.run()` blocks event loop → no async tasks can run
- `create_task()` doesn't wait for completion → state machine moves on immediately
- `run_until_complete()` fails when loop is already running
- No mechanism to pass event loop from main thread to executor thread

## The Fix

### Three-Part Solution

#### 1. Run Wake Pipeline in Thread Pool Executor
**File**: `main.py:184`

```python
# AFTER: Run in executor to avoid blocking event loop
loop = asyncio.get_event_loop()
await loop.run_in_executor(None, self.wake_pipeline.run)
```

**Why**: This runs the synchronous `wake_pipeline.run()` in a separate thread, freeing the event loop to handle async tasks.

#### 2. Pass Event Loop to Wake Pipeline
**File**: `main.py:167-172`

```python
# Initialize wake pipeline with event loop
loop = asyncio.get_event_loop()
self.wake_pipeline = WakeWordPipeline(
    wake_word="alexa",
    timeout_s=30.0,
    workshop=self,
    event_loop=loop  # ✅ Pass event loop for async callbacks
)
```

**Why**: The wake pipeline needs a reference to the main event loop to schedule async callbacks from the executor thread.

#### 3. Use `asyncio.run_coroutine_threadsafe()`
**File**: `wake_pipeline.py:182-189`

```python
# Async callback - schedule on provided event loop
if self.event_loop:
    # We have an event loop - schedule callback and wait
    future = asyncio.run_coroutine_threadsafe(
        self.on_speech(segment, reason),
        self.event_loop
    )
    # Block until callback completes (with 60s timeout)
    future.result(timeout=60.0)
```

**Why**: `run_coroutine_threadsafe()` safely schedules coroutines on an event loop from a different thread (executor). The `future.result()` blocks the executor thread until the callback completes.

#### 4. Proper State Management
**File**: `main.py:129-156`

```python
async def _on_speech(self, segment, reason):
    ...
    # Process speech
    text = self.whisper.transcribe(segment)
    response = await self.process_input(text)

    # Set state during TTS
    self.wake_pipeline.set_state("speaking")
    await self.piper.speak(response)

    # Return to idle when done
    self.wake_pipeline.set_state("idle")  # ✅ Ready for next wake word
```

**Why**: The callback must explicitly return the state machine to "idle" after processing completes, otherwise it stays in "processing"/"speaking" and never listens for the next wake word.

## Architecture Diagram

```
Main Thread (Event Loop)                 Executor Thread (Wake Pipeline)
─────────────────────────                ─────────────────────────────
asyncio.run()
  └─> run_voice_mode_phase2()
        └─> run_in_executor()  ──────────────> wake_pipeline.run()
                │                                   │
                │                                   └─> wake detected
                │                                         │
                │                                         └─> run_coroutine_threadsafe()
                │                                               │
        ┌───────┘◄──────────────────────────────────────────────┘
        │
        └─> _on_speech() executes
              └─> whisper.transcribe()
              └─> agent.chat()
              └─> piper.speak()
              └─> set_state("idle")  ──────────────> State updated
                                                        │
                                                        └─> Continue wake loop
```

## Testing Results

**Before Fix:**
```
16:00:20 | Wake word detected
16:00:20 | State: idle → listening
16:00:21 | Speech captured
16:00:24 | State: listening → processing
[HANGS HERE - No transcription, no response]
```

**After Fix:**
```
[Expected flow with proper execution]
Wake word detected
State: idle → listening
Speech captured
State: listening → processing
[Callback executes: transcription → agent → TTS]
State: processing → speaking → idle
Ready for next wake word
```

## Files Modified

1. **main.py**
   - Lines 95: Initialize wake_pipeline as None (created in run method)
   - Lines 165-172: Pass event loop to wake_pipeline constructor
   - Line 184: Run wake_pipeline.run() in executor
   - Lines 129-156: Proper state management in `_on_speech()`

2. **wake_pipeline.py**
   - Line 32: Added `event_loop` parameter to `__init__`
   - Line 39: Store event loop reference
   - Lines 122-147: Use `run_coroutine_threadsafe()` for wake callback
   - Lines 180-205: Use `run_coroutine_threadsafe()` for speech callback

## Key Concepts

### Event Loop Threading Rules

1. **Never block the event loop**: Synchronous blocking calls prevent async tasks from running
2. **Use executors for blocking code**: `run_in_executor()` runs sync code in thread pool
3. **Cross-thread coroutine scheduling**: `run_coroutine_threadsafe()` schedules coroutines from non-async threads
4. **Wait for completion**: `future.result()` blocks until the coroutine completes

### State Machine Management

1. **Explicit transitions**: Callbacks must explicitly set next state
2. **Processing → Idle**: After handling speech, return to idle for next wake word
3. **Speaking state**: Set during TTS to enable interruption detection
4. **Error handling**: Return to idle on errors to prevent stuck states

## Prevention

To prevent similar issues in future:

1. **Never call blocking sync functions from async context** without using executor
2. **Always pass event loop** to components that need to schedule async callbacks
3. **Use `run_coroutine_threadsafe()`** when scheduling coroutines from threads
4. **Wait for callback completion** before transitioning states
5. **Test state transitions** thoroughly - verify each state returns to expected next state

## Next Steps

1. Test the full conversation flow: wake → listen → transcribe → respond → wake
2. Verify state transitions work correctly across multiple cycles
3. Test error handling (transcription failures, timeout, etc.)
4. Monitor for any new deadlocks or race conditions

---

**Status**: Fix implemented and ready for testing
**Date**: 2024-12-14
**Phase**: Phase 2 - Real-Time Voice Integration
**Issue**: Async callback deadlock preventing speech processing
**Solution**: Thread pool executor + run_coroutine_threadsafe + proper state management
