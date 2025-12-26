"""
Integration Test 3: Timeout Detection
Tests that timeout enforces speech duration limit.
"""

import numpy as np
import time
from audio_realtime import RealtimeAudioCapture
from vad import VoiceActivityDetector, SpeechEndDetector


def test_timeout():
    """Test timeout enforcement with real microphone."""
    print("\n" + "="*60)
    print("INTEGRATION TEST 3: Timeout Detection")
    print("="*60)
    print("\nThis test verifies timeout stops long speech.")
    print("\nInstructions:")
    print("1. Wait for 'Speak continuously...'")
    print("2. SPEAK without pausing for 10+ seconds")
    print("3. Count slowly: 'one, two, three, four...'")
    print("4. Timeout should trigger at ~5 seconds")
    print("\nStarting in 3 seconds...\n")
    time.sleep(3)
    
    # Initialize with SHORT timeout
    TIMEOUT_SECONDS = 5.0
    
    vad = VoiceActivityDetector(
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=300
    )
    
    detector = SpeechEndDetector(vad, timeout_s=TIMEOUT_SECONDS)
    capture = RealtimeAudioCapture(sample_rate=16000, frame_size=512)
    
    try:
        capture.start()
        
        print(f"⏱️  Timeout set to: {TIMEOUT_SECONDS} seconds")
        print("\n🎤 Speak continuously without pausing...")
        print("   Count slowly: 'one, two, three, four, five, six...'")
        print()
        
        segment_start = time.time()
        frames_processed = 0
        speech_started = False
        
        # Process frames until segment captured
        while True:
            frame = capture.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            frames_processed += 1
            
            # Track when speech starts
            if not speech_started and vad.is_speaking:
                speech_started = True
                speech_start_time = time.time()
                print("   🎤 Speech detected, timer started!")
            
            # Process through detector
            segment, reason = detector.process_frame(frame)
            
            # Show progress every 25 frames (~0.8 seconds)
            if speech_started and frames_processed % 25 == 0:
                elapsed = time.time() - speech_start_time
                remaining = TIMEOUT_SECONDS - elapsed
                
                if remaining > 0:
                    print(f"   ⏱️  Elapsed: {elapsed:.1f}s / {TIMEOUT_SECONDS}s "
                          f"(timeout in {remaining:.1f}s...)")
                else:
                    print(f"   ⏱️  Elapsed: {elapsed:.1f}s (past timeout, should capture soon...)")
            
            if segment is not None:
                # Segment captured!
                total_elapsed = time.time() - segment_start
                duration = len(segment) / 16000.0
                
                print(f"\n   ✅ SEGMENT CAPTURED!")
                print(f"   Segment duration: {duration:.2f}s")
                print(f"   End reason: {reason}")
                print(f"   Total time: {total_elapsed:.1f}s")
                
                # Check if timeout worked correctly
                print("\n" + "="*60)
                print("ASSESSMENT")
                print("="*60)
                
                if reason == "timeout":
                    print(f"✅ PASS: Timeout triggered correctly")
                    
                    # Check duration is approximately timeout value
                    if abs(duration - TIMEOUT_SECONDS) < 1.0:
                        print(f"✅ PASS: Duration ~{TIMEOUT_SECONDS}s (actual: {duration:.2f}s)")
                    else:
                        print(f"⚠️  WARNING: Duration {duration:.2f}s differs from timeout {TIMEOUT_SECONDS}s")
                else:
                    print(f"❌ FAIL: Expected 'timeout', got '{reason}'")
                    print(f"   Did you pause while speaking?")
                
                # Duration check
                if duration < 10.0:
                    print(f"✅ PASS: Speech cut off at {duration:.2f}s (not full 10s)")
                else:
                    print(f"❌ FAIL: Captured full {duration:.2f}s - timeout didn't work")
                
                print("\n💡 NOTE: You should have spoken for ~10 seconds,")
                print(f"         but segment is only ~{duration:.1f} seconds due to timeout.")
                
                break
        
        print("\n" + "="*60)
        print("Test complete!")
        print("="*60 + "\n")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Test stopped by user")
    
    finally:
        capture.stop()


if __name__ == "__main__":
    test_timeout()