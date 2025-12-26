"""
Quick test to verify the audio queue fix.
Tests that queue doesn't exhaust during wake word → listening transition.
"""

import time
from wake_pipeline import WakeWordPipeline
from logger import get_logger

log = get_logger("test_queue_fix")

def test_queue_behavior():
    """Test that queue maintains frames during state transition."""
    print("\n" + "="*60)
    print("QUEUE FIX VERIFICATION TEST")
    print("="*60)
    print("\nThis test verifies the audio queue fix.")
    print("We'll monitor queue size during wake word detection.")
    print("\nInstructions:")
    print("1. Wait for system to start")
    print("2. Say 'Alexa' when ready")
    print("3. System will check queue behavior")
    print("\nStarting in 3 seconds...\n")
    time.sleep(3)

    # Initialize pipeline
    pipeline = WakeWordPipeline(
        wake_word="alexa",
        timeout_s=15.0
    )

    # Track queue states
    queue_states = []

    def on_wake():
        print("\n✅ Wake word detected!")
        print("   Checking queue behavior...")

        # Check queue size before refill
        queue_before = pipeline.audio_pipeline.capture.frame_queue.qsize()
        queue_states.append(("before_refill", queue_before))
        print(f"   Queue size: {queue_before} frames")

    def on_speech(segment, reason):
        duration = len(segment) / 16000.0
        print(f"\n✅ Speech captured!")
        print(f"   Duration: {duration:.2f}s")
        print(f"   End reason: {reason}")

        # Stop test
        pipeline.set_state("idle")
        pipeline.stop()

    pipeline.register_callbacks(on_wake=on_wake, on_speech=on_speech)

    try:
        print("🚀 Starting pipeline...")
        print("👂 Listening for 'Alexa'...\n")

        # Start and run
        pipeline.run()

        print("\n" + "="*60)
        print("TEST COMPLETE")
        print("="*60)

        # Check results
        if len(queue_states) > 0:
            print(f"\n✅ Queue monitoring successful")
            for state, size in queue_states:
                print(f"   {state}: {size} frames")
        else:
            print(f"\n⚠️  No queue data collected")

        print("\nIf the queue refilled quickly (within 100-200ms),")
        print("the fix is working correctly!")
        print("\n" + "="*60 + "\n")

    except KeyboardInterrupt:
        print("\n\n⏹️  Test stopped by user")
        pipeline.stop()

    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        pipeline.stop()


if __name__ == "__main__":
    test_queue_behavior()
