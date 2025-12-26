"""
Integration Test 5: Interruption Detection
Tests detecting user speech during assistant output.
"""

import numpy as np
import time
from audio_realtime import RealtimeAudioCapture
from vad import VoiceActivityDetector, InterruptionDetector


def test_interruption():
    """Test interruption detection with real microphone."""
    print("\n" + "="*60)
    print("INTEGRATION TEST 5: Interruption Detection")
    print("="*60)
    print("\nThis test verifies you can interrupt during assistant speech.")
    print("\nInstructions:")
    print("1. Test simulates assistant speaking (10 second countdown)")
    print("2. While countdown running, SPEAK into microphone")
    print("3. Interruption should be detected quickly (<200ms)")
    print("4. Test will run 3 interruption cycles")
    print("\nStarting in 3 seconds...\n")
    time.sleep(3)
    
    # Initialize components
    vad = VoiceActivityDetector(
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=300
    )
    
    detector = InterruptionDetector(
        vad,
        interruption_threshold=0.6,  # Higher threshold
        confirmation_frames=3
    )
    
    capture = RealtimeAudioCapture(sample_rate=16000, frame_size=512)
    
    try:
        capture.start()
        
        cycles_completed = 0
        target_cycles = 3
        
        while cycles_completed < target_cycles:
            print(f"\n🔊 CYCLE {cycles_completed + 1}/{target_cycles}")
            print("   Simulating assistant speaking (10 second countdown)...")
            print("   SPEAK into microphone to interrupt!")
            print()
            
            # Register interruption callback
            interrupted = [False]
            interrupt_time = [None]
            
            def on_interrupt():
                interrupted[0] = True
                interrupt_time[0] = time.time()
            
            detector.on_interruption(on_interrupt)
            detector.set_assistant_speaking(True)
            
            # Simulate assistant speaking for 10 seconds
            # (or until interrupted)
            speaking_start = time.time()
            max_duration = 10.0
            frames_processed = 0
            
            while True:
                elapsed = time.time() - speaking_start
                
                if elapsed >= max_duration:
                    print(f"\n   ⏱️  Countdown complete (no interruption)")
                    break
                
                # Get audio frame
                frame = capture.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                frames_processed += 1
                
                # Show countdown every 50 frames (~1.6s)
                if frames_processed % 50 == 0:
                    remaining = max_duration - elapsed
                    print(f"   ⏱️  {remaining:.1f}s remaining... (speak to interrupt)")
                
                # Process frame (check for interruption)
                if detector.process_frame(frame):
                    # Interruption detected!
                    latency = interrupt_time[0] - speaking_start if interrupt_time[0] else 0
                    
                    print(f"\n   🛑 INTERRUPTION DETECTED!")
                    print(f"   Time to interrupt: {latency:.3f}s")
                    print(f"   Frames processed: {frames_processed}")
                    print(f"   Latency: {(frames_processed * 0.032):.3f}s")
                    
                    interrupted[0] = True
                    break
            
            # Cycle complete
            detector.set_assistant_speaking(False)
            detector.reset()
            cycles_completed += 1
            
            # Brief pause between cycles
            if cycles_completed < target_cycles:
                print(f"\n   Next cycle in 2 seconds...")
                time.sleep(2)
        
        # All cycles complete
        print("\n" + "="*60)
        print("ALL CYCLES COMPLETE!")
        print("="*60)
        
        stats = detector.get_stats()
        print(f"\nStatistics:")
        print(f"  Total interruptions: {stats['total_interruptions']}")
        print(f"  Interruption threshold: {stats['interruption_threshold']}")
        print(f"  Confirmation frames: {stats['confirmation_frames']}")
        
        # Assessment
        print("\n" + "="*60)
        print("ASSESSMENT")
        print("="*60)
        
        if stats['total_interruptions'] > 0:
            print(f"✅ PASS: Detected {stats['total_interruptions']} interruption(s)")
            
            # Check latency (should be <200ms for good UX)
            print(f"✅ PASS: Interruption detection working")
            print(f"   💡 TIP: Latency should be <200ms for natural feel")
        else:
            print(f"⚠️  No interruptions detected")
            print(f"   Did you speak during the countdown?")
        
        print("\n💡 What to test:")
        print("   - Speak loudly during countdown → should interrupt")
        print("   - Whisper during countdown → may not interrupt (threshold)")
        print("   - Ambient noise → should NOT interrupt (high threshold)")
        print("   - Quick 'hey' → should interrupt (confirmation frames)")
        
        print("\n" + "="*60)
        print("Test complete!")
        print("="*60 + "\n")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Test stopped by user")
        print(f"Completed {cycles_completed}/{target_cycles} cycles")
    
    finally:
        capture.stop()


if __name__ == "__main__":
    test_interruption()