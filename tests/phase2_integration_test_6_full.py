"""
Integration Test 6: Full Conversation Flow
Tests complete wake → listen → process → respond cycle.
"""

import numpy as np
import time
from wake_pipeline import WakeWordPipeline


def test_full_flow():
    """Test complete conversation flow."""
    print("\n" + "="*60)
    print("INTEGRATION TEST 6: Full Conversation Flow")
    print("="*60)
    print("\nThis test verifies complete interaction cycle.")
    print("\nInstructions:")
    print("1. Wait for 'IDLE' state")
    print("2. SAY: 'workshop' (wake word)")
    print("3. Wait for wake confirmation")
    print("4. SAY: a command (e.g., 'what time is it')")
    print("5. System processes (simulated)")
    print("6. Returns to IDLE")
    print("7. Repeat 3 times total")
    print("\nStarting in 3 seconds...\n")
    time.sleep(3)
    
    # Initialize pipeline
    pipeline = WakeWordPipeline(
        wake_word="alexa",
        timeout_s=15.0  # Shorter timeout for testing
    )
    
    # Track events
    events = {
        'wake_count': 0,
        'speech_count': 0,
        'wake_times': [],
        'speech_times': []
    }
    
    # Register callbacks
    def on_wake():
        events['wake_count'] += 1
        events['wake_times'].append(time.time())
        print("\n   🎤 WAKE WORD DETECTED!")
        print("   State: IDLE → LISTENING")
        print("   Now speak your command...")
        print()
    
    def on_speech(segment, reason):
        events['speech_count'] += 1
        events['speech_times'].append(time.time())
        
        duration = len(segment) / 16000.0
        
        print(f"\n   📝 SPEECH CAPTURED!")
        print(f"   Duration: {duration:.2f}s")
        print(f"   End reason: {reason}")
        print(f"   Samples: {len(segment)}")
        print(f"   State: LISTENING → PROCESSING")
        
        # Simulate processing
        print("\n   ⚙️  Processing (simulated)...")
        print("      [Whisper would transcribe here]")
        print("      [Ollama would generate response here]")
        time.sleep(1)  # Simulate processing delay
        
        print(f"   State: PROCESSING → IDLE")
        print(f"\n   Ready for next wake word!")
        print()
        
        # Return to idle
        pipeline.set_state("idle")
    
    pipeline.register_callbacks(on_wake=on_wake, on_speech=on_speech)
    
    try:
        # Start pipeline
        print("🚀 Pipeline starting...")
        print("   Initial state: IDLE")
        print()
        print("👂 Listening for wake word 'workshop'...")
        print("   Say 'workshop' to begin...")
        print()
        
        # Run in separate thread-like manner (manual control)
        pipeline.audio_pipeline.start_listening()
        
        target_cycles = 3
        
        while events['speech_count'] < target_cycles:
            # Get current state
            current_state = pipeline.state
            
            if current_state == "idle":
                # Check for wake word
                frame = pipeline.audio_pipeline.capture.get_frame()
                if frame is not None:
                    pipeline._process_idle_state(frame)
            
            elif current_state == "listening":
                # Capture speech
                try:
                    segment, reason = pipeline.audio_pipeline.capture_speech_segment(
                        max_wait_s=20.0,
                        require_speech=False
                    )
                    
                    if segment is not None:
                        # Trigger speech callback
                        if pipeline.on_speech:
                            pipeline.on_speech(segment, reason)
                    else:
                        # Timeout - no speech
                        print("\n   ⚠️  No speech detected, returning to IDLE")
                        pipeline.set_state("idle")
                        pipeline.audio_pipeline.stop_listening()
                        pipeline.audio_pipeline.start_listening()
                
                except Exception as e:
                    print(f"\n   ❌ Error: {e}")
                    pipeline.set_state("idle")
            
            time.sleep(0.01)
        
        # All cycles complete
        pipeline.audio_pipeline.stop_listening()
        
        print("\n" + "="*60)
        print("ALL CYCLES COMPLETE!")
        print("="*60)
        
        # Calculate statistics
        print(f"\nStatistics:")
        print(f"  Wake detections: {events['wake_count']}")
        print(f"  Speech captures: {events['speech_count']}")
        
        if len(events['wake_times']) > 1:
            wake_intervals = [
                events['wake_times'][i+1] - events['wake_times'][i]
                for i in range(len(events['wake_times'])-1)
            ]
            avg_interval = sum(wake_intervals) / len(wake_intervals)
            print(f"  Average cycle time: {avg_interval:.1f}s")
        
        # State transition stats
        stats = pipeline.get_stats()
        print(f"\nState transitions:")
        for transition in stats['state_durations'].items():
            state, duration = transition
            print(f"  {state}: {duration:.1f}s")
        
        # Assessment
        print("\n" + "="*60)
        print("ASSESSMENT")
        print("="*60)
        
        if events['wake_count'] == target_cycles:
            print(f"✅ PASS: All {target_cycles} wake words detected")
        else:
            print(f"⚠️  Expected {target_cycles} wakes, got {events['wake_count']}")
        
        if events['speech_count'] == target_cycles:
            print(f"✅ PASS: All {target_cycles} commands captured")
        else:
            print(f"⚠️  Expected {target_cycles} commands, got {events['speech_count']}")
        
        if len(pipeline.state_transitions) >= target_cycles * 2:
            print(f"✅ PASS: State machine working (idle↔listening transitions)")
        else:
            print(f"⚠️  Few state transitions: {len(pipeline.state_transitions)}")
        
        print("\n💡 Full flow tested:")
        print("   1. IDLE (waiting for wake)")
        print("   2. LISTENING (after 'workshop')")
        print("   3. PROCESSING (simulated)")
        print("   4. Back to IDLE")
        print("   ✅ Complete conversation cycle!")
        
        print("\n" + "="*60)
        print("Test complete!")
        print("="*60 + "\n")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Test stopped by user")
        print(f"Completed {events['speech_count']}/{target_cycles} cycles")
        pipeline.audio_pipeline.stop_listening()
    
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        pipeline.audio_pipeline.stop_listening()


if __name__ == "__main__":
    test_full_flow()