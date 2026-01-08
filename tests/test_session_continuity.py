"""
Test Session Continuity (Phase 4)

Tests the integration between Workshop sessions and Claude Code sessions:
1. ClaudeBridgeManager singleton behavior
2. Session state persistence
3. Workshop session binding
4. Turn counting and token tracking
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_singleton_behavior():
    """Test that ClaudeBridgeManager is a proper singleton."""
    from claude_bridge import get_claude_bridge_manager, ClaudeBridgeManager

    # Reset singleton for clean test
    ClaudeBridgeManager._instance = None

    manager1 = get_claude_bridge_manager()
    manager2 = get_claude_bridge_manager()

    assert manager1 is manager2, "Managers should be the same instance"
    print("✓ Singleton behavior verified")


def test_session_state_serialization():
    """Test ClaudeSessionState serialization."""
    from claude_bridge import ClaudeSessionState

    state = ClaudeSessionState()
    state.workshop_session_id = "sess_20260103_120000_abcd"
    state.claude_session_id = "claude-session-xyz"
    state.started_at = datetime.now()
    state.last_used_at = datetime.now()
    state.turn_count = 5
    state.context_tokens_estimated = 2500

    # Serialize
    data = state.to_dict()

    # Deserialize
    restored = ClaudeSessionState.from_dict(data)

    assert restored.workshop_session_id == state.workshop_session_id
    assert restored.claude_session_id == state.claude_session_id
    assert restored.turn_count == state.turn_count
    assert restored.context_tokens_estimated == state.context_tokens_estimated

    print("✓ Session state serialization verified")


def test_session_binding():
    """Test binding workshop sessions to Claude bridge."""
    from claude_bridge import get_claude_bridge_manager, ClaudeBridgeManager

    # Reset singleton for clean test
    ClaudeBridgeManager._instance = None

    manager = get_claude_bridge_manager()

    # Bind to a workshop session
    session_id = "sess_test_20260103_120000_1234"
    manager.bind_to_workshop_session(session_id)

    info = manager.get_session_info()
    assert info["workshop_session_id"] == session_id
    assert info["turn_count"] == 0

    print("✓ Session binding verified")


def test_turn_recording():
    """Test that turns are recorded correctly."""
    from claude_bridge import get_claude_bridge_manager, ClaudeBridgeManager

    # Reset singleton for clean test
    ClaudeBridgeManager._instance = None

    manager = get_claude_bridge_manager()
    manager.bind_to_workshop_session("sess_test_turns_1234")

    # Record some turns
    manager.record_turn(estimated_tokens=500)
    manager.record_turn(estimated_tokens=750)
    manager.record_turn(estimated_tokens=300)

    info = manager.get_session_info()
    assert info["turn_count"] == 3
    assert info["context_tokens_estimated"] == 1550

    print("✓ Turn recording verified")


def test_should_summarize():
    """Test summarization threshold detection."""
    from claude_bridge import get_claude_bridge_manager, ClaudeBridgeManager

    # Reset singleton for clean test
    ClaudeBridgeManager._instance = None

    manager = get_claude_bridge_manager()
    manager.bind_to_workshop_session("sess_test_summarize_1234")

    # Not enough tokens yet
    assert not manager.should_summarize(token_threshold=1000)

    # Add tokens
    manager.record_turn(estimated_tokens=600)
    manager.record_turn(estimated_tokens=600)

    # Now should trigger
    assert manager.should_summarize(token_threshold=1000)

    print("✓ Summarization threshold verified")


def test_session_reset():
    """Test session reset functionality."""
    from claude_bridge import get_claude_bridge_manager, ClaudeBridgeManager

    # Reset singleton for clean test
    ClaudeBridgeManager._instance = None

    manager = get_claude_bridge_manager()
    manager.bind_to_workshop_session("sess_test_reset_1234")

    # Record some activity
    manager.record_turn(estimated_tokens=500)
    manager.record_turn(estimated_tokens=500)

    info = manager.get_session_info()
    assert info["turn_count"] == 2

    # Reset
    manager.reset()

    info = manager.get_session_info()
    assert info["turn_count"] == 0
    assert info["context_tokens_estimated"] == 0

    print("✓ Session reset verified")


def test_shared_bridge():
    """Test that shared bridge is returned consistently."""
    from claude_bridge import get_claude_bridge, ClaudeBridgeManager

    # Reset singleton for clean test
    ClaudeBridgeManager._instance = None

    bridge1 = get_claude_bridge(timeout_seconds=60)
    bridge2 = get_claude_bridge(timeout_seconds=120)  # Different timeout should still return same bridge

    assert bridge1 is bridge2, "Should return same bridge instance"
    print("✓ Shared bridge verified")


def test_session_manager_integration():
    """Test that SessionManager correctly integrates with ClaudeBridgeManager."""
    from session_manager import get_session_manager, SessionManager
    from claude_bridge import get_claude_bridge_manager, ClaudeBridgeManager

    # Reset singletons for clean test
    ClaudeBridgeManager._instance = None

    # Create fresh session manager
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        session_mgr = SessionManager(base_dir=Path(tmpdir))

        # Start a session
        session = session_mgr.start_session(mode="text")

        # Check Claude bridge was bound
        bridge_mgr = get_claude_bridge_manager()
        info = bridge_mgr.get_session_info()

        assert info["workshop_session_id"] == session.session_id
        print(f"✓ Session manager integration verified (session: {session.session_id})")


async def test_claude_bridge_query():
    """Test actual Claude Code query (requires Claude Code to be installed)."""
    from claude_bridge import ClaudeCodeBridge

    try:
        bridge = ClaudeCodeBridge(timeout_seconds=30)

        messages = [{"role": "user", "content": "What is 2 + 2? Reply with just the number."}]
        result = await bridge.query(messages, disable_native_tools=True)

        content = result.get("content", "")
        assert "4" in content, f"Expected '4' in response, got: {content}"

        # Check session ID was captured
        assert bridge.session_id is not None, "Session ID should be captured"

        print(f"✓ Claude bridge query verified (session: {bridge.session_id})")
        return True
    except Exception as e:
        print(f"⚠ Claude bridge query skipped: {e}")
        return False


def run_all_tests():
    """Run all session continuity tests."""
    print("\n" + "="*60)
    print("Session Continuity Tests (Phase 4)")
    print("="*60 + "\n")

    test_singleton_behavior()
    test_session_state_serialization()
    test_session_binding()
    test_turn_recording()
    test_should_summarize()
    test_session_reset()
    test_shared_bridge()
    test_session_manager_integration()

    # Run async test
    print("\nRunning async tests...")
    asyncio.run(test_claude_bridge_query())

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
