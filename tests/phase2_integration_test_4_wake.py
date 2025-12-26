"""
Integration Test 4: Wake Word Detection
Tests wake word activation with OpenWakeWord.
"""

import numpy as np
import time
from audio_realtime import RealtimeAudioCapture
from wake_word import WakeWordDetector, WakeWordBuffer


def test_wake_word():
    """Test wake word detection with real microphone."""
    print("\n" + "="*60)
    print("INTEGRATION TEST 4: Wake Word Detection")
    print("="*60)
    print("\nThis test verifies 'workshop' activates listening.")
    print("\nInstructions:")
    print("1. Wait for 'Listening for wake word...'")
    print("2. SAY: 'workshop' (or 'hey jarvis' - similar model)")
    print("3. Observe wake detection")
    print("4. Test will detect 3 wake words then stop")
    print("\nNOTE: Speak clearly and at normal volume")
    print("\nStarting in 3 seconds...\n")
    time.sleep(3)
    
    # Initialize components
    wake_detector = WakeWordDetector(
        model_name="hey_jarvis",  # Closest model to "workshop"
        threshold=0.5
    )
    
    buffer = WakeWordBuffer(wake_detector)
    
    # Configure for Blue Yeti
    capture = RealtimeAudioCapture(
        sample_rate=16000,
        channels=2,
        frame_size=512,
        device_id=4,
        hardware_sample_rate=44100
    )
    
    try:
        print("👂 Listening for wake word...")
        print("   Say: 'workshop' or 'hey jarvis'")
        print()
        
        capture.start()
        
        detections = 0
        target_detections = 3
        frame_count = 0
        
        while detections < target_detections:
            # Get audio frame
            frame = capture.get_frame(timeout=0.05)
            if frame is None:
                continue
            
            frame_count += 1
            
            # Process through wake word detector
            detected = buffer.add_frame(frame)
            
            if detected:
                detections += 1
                stats = wake_detector.get_stats()
                
                print(f"🎯 WAKE WORD DETECTED! (#{detections}/{target_detections})")
                print(f"   Frames processed: {frame_count}")
                print(f"   Detection rate: {stats['detection_rate']:.4f}")
                print(f"   Model: {stats['model']}")
                print()
                
                if detections < target_detections:
                    print(f"👂 Listening for next wake word... ({detections}/{target_detections})")
                    print()
                
                # Brief pause to avoid double-detection
                time.sleep(1.0)
            
            # Status update every 5 seconds
            if frame_count % 156 == 0:  # ~5 seconds at 31 fps
                elapsed = frame_count / 31
                print(f"   ... listening ({elapsed:.0f}s, {frame_count} frames processed)")
        
        # Final statistics
        stats = wake_detector.get_stats()
        print("\n" + "="*60)
        print("ALL WAKE WORDS DETECTED!")
        print("="*60)
        print("Statistics:")
        print(f"  Total frames processed: {stats['frames_processed']}")
        print(f"  Total detections: {stats['detections']}")
        print(f"  Detection rate: {stats['detection_rate']:.4f}")
        print(f"  Model used: {stats['model']}")
        print(f"  Threshold: {stats['threshold']}")
        
        # Assessment
        print("\n" + "="*60)
        print("ASSESSMENT")
        print("="*60)
        
        if detections == target_detections:
            print(f"✅ PASS: Detected all {target_detections} wake words")
        else:
            print(f"❌ FAIL: Expected {target_detections}, got {detections}")
        
        if stats['detections'] > 0:
            print("✅ PASS: Wake word detection working")
        
        if stats['detection_rate'] < 0.1:
            print("✅ PASS: Low false positive rate (good!)")
        elif stats['detection_rate'] > 0.5:
            print("⚠️  WARNING: High detection rate - check threshold")
        
        print("\n" + "="*60)
        print("Test complete!")
        print("="*60 + "\n")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        
        stats = wake_detector.get_stats()
        print("\nPartial statistics:")
        print(f"  Detections: {detections}/{target_detections}")
        print(f"  Frames processed: {stats['frames_processed']}")
        print(f"  Detection rate: {stats['detection_rate']:.4f}")
    
    finally:
        capture.stop()


if __name__ == "__main__":
    test_wake_word()