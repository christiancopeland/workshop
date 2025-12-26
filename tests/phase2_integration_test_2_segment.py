"""
Integration Test 2: Speech Segment Capture
Tests complete speech segment collection with natural pause detection.
"""

import numpy as np
import time
from audio_realtime import RealtimeAudioCapture
from vad import VoiceActivityDetector, SpeechEndDetector


def test_segment_capture():
    """Test speech segment capture with natural pause detection."""
    print("\n" + "="*60)
    print("INTEGRATION TEST 2: Speech Segment Capture")
    print("="*60)
    print("\nThis test verifies complete speech segments are captured.")
    print("\nInstructions:")
    print("1. Wait for 'Speak now...'")
    print("2. SPEAK a sentence (e.g., 'Hello Workshop, how are you today?')")
    print("3. PAUSE for 1 second")
    print("4. Observe segment capture")
    print("5. Test will capture 3 segments then stop")
    print("\nStarting in 3 seconds...\n")
    time.sleep(3)
    
    # Initialize components
    vad = VoiceActivityDetector(
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=300
    )
    
    detector = SpeechEndDetector(vad, timeout_s=30.0)
    
    # Configure for Blue Yeti
    capture = RealtimeAudioCapture(
        sample_rate=16000,
        channels=2,
        frame_size=512,
        device_id=None,
        hardware_sample_rate=44100
    )
    
    try:
        capture.start()
        
        segments = []
        target_segments = 3
        
        for segment_num in range(1, target_segments + 1):
            print(f"🎤 Segment {segment_num}/{target_segments}")
            print("   Speak now, then pause...")
            
            start_time = time.time()
            waiting_reported = False
            speaking_last_report = 0
            
            while True:
                frame = capture.get_frame(timeout=0.05)
                if frame is None:
                    continue
                
                # Process frame
                segment, end_reason = detector.process_frame(frame)
                
                # Report waiting status (every 1.6s)
                elapsed = time.time() - start_time
                if not vad.is_speaking and elapsed - speaking_last_report >= 1.6:
                    if not waiting_reported or elapsed - speaking_last_report >= 1.6:
                        print(f"   ... waiting for speech ({elapsed:.1f}s)")
                        waiting_reported = True
                        speaking_last_report = elapsed
                
                # Report speaking status (every 1.6s)
                if vad.is_speaking and elapsed - speaking_last_report >= 1.6:
                    print(f"   🎤 Speaking... ({elapsed:.1f}s)")
                    speaking_last_report = elapsed
                
                # Check if segment complete
                if segment is not None:
                    # Skip empty segments (false triggers)
                    if len(segment) == 0:
                        print(f"   ⚠️  Empty segment detected (false trigger), ignoring...")
                        continue
                    
                    duration = len(segment) / 16000
                    total_time = time.time() - start_time
                    
                    print(f"   ✅ SEGMENT CAPTURED!")
                    print(f"   Duration: {duration:.2f}s")
                    print(f"   Samples: {len(segment)}")
                    print(f"   End reason: {end_reason}")
                    print(f"   Total time: {total_time:.1f}s")
                    print(f"   Sample range: [{segment.min():.3f}, {segment.max():.3f}]")
                    
                    segments.append((duration, segment, end_reason))
                    break
        
        # Final statistics
        stats = detector.get_stats()
        print("\n" + "="*60)
        print("ALL SEGMENTS CAPTURED!")
        print("="*60)
        print("Statistics:")
        print(f"  Total segments: {stats['total_segments']}")
        print(f"  Natural pause endings: {stats['natural_pause']}")
        print(f"  Timeout endings: {stats['timeout']}")
        print(f"  Natural pause %: {stats.get('natural_pause_pct', 0):.1f}%")
        print(f"  Timeout %: {stats.get('timeout_pct', 0):.1f}%")
        
        # Assessment
        print("="*60)
        print("ASSESSMENT")
        print("="*60)
        
        if len(segments) == target_segments:
            print(f"✅ PASS: Captured all {target_segments} segments")
        else:
            print(f"❌ FAIL: Expected {target_segments} segments, got {len(segments)}")
        
        if stats['natural_pause'] > 0:
            print("✅ PASS: Natural pause detection working")
        
        if all(reason == "natural_pause" for _, _, reason in segments):
            print("✅ PASS: All segments ended naturally (good!)")
        elif stats['timeout'] > 0:
            print("⚠️  WARNING: Some segments ended by timeout (try shorter phrases)")
        
        # Check segment quality
        print("\n" + "="*60)
        print("SEGMENT QUALITY")
        print("="*60)
        
        for i, (duration, segment, reason) in enumerate(segments, 1):
            sample_min = segment.min()
            sample_max = segment.max()
            is_clipping = abs(sample_min) >= 0.95 or abs(sample_max) >= 0.95
            
            print(f"Segment {i}:")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Samples: {len(segment)}")
            print(f"  End reason: {reason}")
            print(f"  Sample range: [{sample_min:.3f}, {sample_max:.3f}]", end="")
            
            if is_clipping:
                print(" ⚠️ CLIPPING DETECTED")
            else:
                print(" ✅")
        
        # Overall quality check
        all_good = all(
            abs(seg.min()) < 0.95 and abs(seg.max()) < 0.95 
            for _, seg, _ in segments
        )
        
        print()
        if all_good:
            print("✅ PASS: All segments have good audio quality (no clipping)")
        else:
            print("⚠️  WARNING: Some segments show clipping - check microphone gain")
        
        print("\n" + "="*60)
        print("Test complete!")
        print("="*60 + "\n")
    
    finally:
        capture.stop()


if __name__ == "__main__":
    test_segment_capture()