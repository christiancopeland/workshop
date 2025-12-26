"""
Workshop Phase 2: Voice State Machine
Manages state transitions for real-time voice system
"""

from enum import Enum
from typing import Optional, Callable
from logger import get_logger

log = get_logger("voice_state")


class VoiceState(Enum):
    """Voice system states"""
    IDLE = "idle"                # Waiting for wake word
    LISTENING = "listening"      # Capturing user speech
    PROCESSING = "processing"    # LLM generating response
    SPEAKING = "speaking"        # TTS playing response


class VoiceEvent(Enum):
    """Events that trigger state transitions"""
    WAKE_WORD_DETECTED = "wake_word_detected"
    SPEECH_END = "speech_end"
    RESPONSE_READY = "response_ready"
    PLAYBACK_COMPLETE = "playback_complete"
    USER_INTERRUPT = "user_interrupt"
    ERROR = "error"
    RESET = "reset"


class StateMachine:
    """
    Finite state machine for voice system.
    
    States: IDLE → LISTENING → PROCESSING → SPEAKING → IDLE
    
    Valid transitions:
        IDLE + wake_word → LISTENING
        LISTENING + speech_end → PROCESSING
        PROCESSING + response_ready → SPEAKING
        SPEAKING + playback_complete → IDLE
        ANY + user_interrupt → LISTENING
        ANY + error → IDLE
        ANY + reset → IDLE
    
    Example:
        sm = StateMachine()
        
        # Wake word detected
        if sm.transition(VoiceEvent.WAKE_WORD_DETECTED):
            start_listening()
        
        # User finished speaking
        if sm.transition(VoiceEvent.SPEECH_END):
            process_speech()
        
        # User interrupts
        if sm.transition(VoiceEvent.USER_INTERRUPT):
            stop_playback()
            start_listening()
    """
    
    def __init__(self, initial_state: VoiceState = VoiceState.IDLE):
        """
        Initialize state machine.
        
        Args:
            initial_state: Starting state (default: IDLE)
        """
        self._state = initial_state
        self._callbacks = {}  # Event callbacks
        
        # Define valid transitions
        self._transitions = {
            VoiceState.IDLE: {
                VoiceEvent.WAKE_WORD_DETECTED: VoiceState.LISTENING,
            },
            VoiceState.LISTENING: {
                VoiceEvent.SPEECH_END: VoiceState.PROCESSING,
            },
            VoiceState.PROCESSING: {
                VoiceEvent.RESPONSE_READY: VoiceState.SPEAKING,
            },
            VoiceState.SPEAKING: {
                VoiceEvent.PLAYBACK_COMPLETE: VoiceState.IDLE,
            },
        }
        
        # Universal transitions (work from any state)
        self._universal_transitions = {
            VoiceEvent.USER_INTERRUPT: VoiceState.LISTENING,
            VoiceEvent.ERROR: VoiceState.IDLE,
            VoiceEvent.RESET: VoiceState.IDLE,
        }
        
        log.info(f"StateMachine initialized: {initial_state.value}")
    
    @property
    def state(self) -> VoiceState:
        """Get current state."""
        return self._state
    
    def transition(self, event: VoiceEvent) -> bool:
        """
        Attempt state transition based on event.
        
        Args:
            event: Event triggering transition
            
        Returns:
            True if transition succeeded, False if invalid
        """
        current = self._state
        
        # Check universal transitions first
        if event in self._universal_transitions:
            new_state = self._universal_transitions[event]
            self._state = new_state
            log.info(f"State transition: {current.value} → {new_state.value} ({event.value})")
            self._trigger_callback(event, current, new_state)
            return True
        
        # Check state-specific transitions
        if current in self._transitions:
            if event in self._transitions[current]:
                new_state = self._transitions[current][event]
                self._state = new_state
                log.info(f"State transition: {current.value} → {new_state.value} ({event.value})")
                self._trigger_callback(event, current, new_state)
                return True
        
        # Invalid transition
        log.warning(f"Invalid transition: {current.value} + {event.value}")
        return False
    
    def on(self, event: VoiceEvent, callback: Callable):
        """
        Register callback for state transition.
        
        Args:
            event: Event to listen for
            callback: Function(old_state, new_state) to call
        """
        self._callbacks[event] = callback
    
    def _trigger_callback(self, event: VoiceEvent, old_state: VoiceState, new_state: VoiceState):
        """Trigger registered callback for event."""
        if event in self._callbacks:
            try:
                self._callbacks[event](old_state, new_state)
            except Exception as e:
                log.error(f"Callback error for {event.value}: {e}")
    
    def can_listen(self) -> bool:
        """Check if system can start listening."""
        return self._state in (VoiceState.IDLE, VoiceState.LISTENING)
    
    def can_speak(self) -> bool:
        """Check if system can start speaking."""
        return self._state in (VoiceState.PROCESSING, VoiceState.SPEAKING)
    
    def is_idle(self) -> bool:
        """Check if in idle state."""
        return self._state == VoiceState.IDLE
    
    def is_listening(self) -> bool:
        """Check if listening for speech."""
        return self._state == VoiceState.LISTENING
    
    def is_processing(self) -> bool:
        """Check if processing user input."""
        return self._state == VoiceState.PROCESSING
    
    def is_speaking(self) -> bool:
        """Check if playing response."""
        return self._state == VoiceState.SPEAKING
    
    def reset(self):
        """Force reset to IDLE state."""
        old_state = self._state
        self._state = VoiceState.IDLE
        log.info(f"State reset: {old_state.value} → IDLE")
    
    def get_valid_events(self) -> list[VoiceEvent]:
        """Get list of valid events for current state."""
        valid = list(self._universal_transitions.keys())
        
        if self._state in self._transitions:
            valid.extend(self._transitions[self._state].keys())
        
        return valid


# Testing
def test_state_machine():
    """Test state machine transitions"""
    print("Testing StateMachine...\n")
    
    sm = StateMachine()
    
    # Test 1: Normal flow
    print("Test 1: Normal flow (IDLE → LISTENING → PROCESSING → SPEAKING → IDLE)")
    assert sm.state == VoiceState.IDLE
    
    assert sm.transition(VoiceEvent.WAKE_WORD_DETECTED) == True
    assert sm.state == VoiceState.LISTENING
    
    assert sm.transition(VoiceEvent.SPEECH_END) == True
    assert sm.state == VoiceState.PROCESSING
    
    assert sm.transition(VoiceEvent.RESPONSE_READY) == True
    assert sm.state == VoiceState.SPEAKING
    
    assert sm.transition(VoiceEvent.PLAYBACK_COMPLETE) == True
    assert sm.state == VoiceState.IDLE
    print("✅ Normal flow works\n")
    
    # Test 2: Invalid transitions
    print("Test 2: Invalid transitions")
    assert sm.state == VoiceState.IDLE
    assert sm.transition(VoiceEvent.SPEECH_END) == False  # Can't go IDLE → PROCESSING
    assert sm.state == VoiceState.IDLE  # State unchanged
    print("✅ Invalid transitions rejected\n")
    
    # Test 3: User interrupt (from any state)
    print("Test 3: User interrupt from SPEAKING")
    sm.transition(VoiceEvent.WAKE_WORD_DETECTED)  # → LISTENING
    sm.transition(VoiceEvent.SPEECH_END)          # → PROCESSING
    sm.transition(VoiceEvent.RESPONSE_READY)      # → SPEAKING
    
    assert sm.state == VoiceState.SPEAKING
    assert sm.transition(VoiceEvent.USER_INTERRUPT) == True
    assert sm.state == VoiceState.LISTENING  # Back to listening
    print("✅ User interrupt works\n")
    
    # Test 4: Error recovery
    print("Test 4: Error recovery from PROCESSING")
    sm.transition(VoiceEvent.SPEECH_END)  # → PROCESSING
    assert sm.state == VoiceState.PROCESSING
    
    assert sm.transition(VoiceEvent.ERROR) == True
    assert sm.state == VoiceState.IDLE  # Back to idle
    print("✅ Error recovery works\n")
    
    # Test 5: State queries
    print("Test 5: State query methods")
    sm.reset()
    assert sm.is_idle() == True
    assert sm.can_listen() == True
    assert sm.can_speak() == False
    
    sm.transition(VoiceEvent.WAKE_WORD_DETECTED)
    assert sm.is_listening() == True
    assert sm.can_listen() == True
    
    sm.transition(VoiceEvent.SPEECH_END)
    assert sm.is_processing() == True
    assert sm.can_speak() == True
    
    sm.transition(VoiceEvent.RESPONSE_READY)
    assert sm.is_speaking() == True
    print("✅ State queries work\n")
    
    # Test 6: Callbacks
    print("Test 6: Event callbacks")
    callback_fired = [False]
    
    def on_wake_word(old, new):
        callback_fired[0] = True
        assert old == VoiceState.IDLE
        assert new == VoiceState.LISTENING
    
    sm.reset()
    sm.on(VoiceEvent.WAKE_WORD_DETECTED, on_wake_word)
    sm.transition(VoiceEvent.WAKE_WORD_DETECTED)
    
    assert callback_fired[0] == True
    print("✅ Callbacks work\n")
    
    # Test 7: Get valid events
    print("Test 7: Get valid events")
    sm.reset()
    valid = sm.get_valid_events()
    assert VoiceEvent.WAKE_WORD_DETECTED in valid
    assert VoiceEvent.USER_INTERRUPT in valid
    assert VoiceEvent.ERROR in valid
    assert VoiceEvent.SPEECH_END not in valid  # Not valid from IDLE
    print(f"  Valid events from IDLE: {[e.value for e in valid]}")
    print("✅ Valid events works\n")
    
    print("✅ All tests passed!")


if __name__ == "__main__":
    test_state_machine()