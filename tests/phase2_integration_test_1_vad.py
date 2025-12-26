"""
Integration Test 1: Basic VAD Detection
Tests voice activity detection with real microphone input.
"""

import numpy as np
import time
from audio_realtime import RealtimeAudioCapture
from vad import VoiceActivityDetector
import logger as log


def test_vad_detection():
    """Test basic VAD with real microphone."""
    print("\n" + "="*60)
    print("INTEGRATION TEST 1: Voice Activity Detection")
    print("="*60)
    print("\nThis test verifies VAD detects your voice vs silence.")
    print("\nInstructions:")
    print("1. Wait for 'Listening...'")
    print("2. SPEAK into microphone for 2-3 seconds")
    print("3. PAUSE for 1-2 seconds")
    print("4. Repeat a few times")
    print("5. Press Ctrl+C to stop")
    print("\nStarting in 3 seconds...\n")
    time.sleep(3)
    
    # Initialize components
    # Blue USB mic configuration: 44.1kHz hardware, 16kHz for VAD
    vad = VoiceActivityDetector(
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=300
    )
    
    # Configure for Blue Yeti (device 4, 44.1kHz, stereo)
    # Automatically resamples 44.1kHz → 16kHz for VAD
    capture = RealtimeAudioCapture(
        sample_rate=16000,           # Target rate for VAD
        channels=2,                  # Blue Yeti is stereo
        frame_size=512,              # At 16kHz (exactly 512 samples guaranteed)
        device_id=None,                 # Blue Yeti mic
        hardware_sample_rate=44100   # Mic's native rate
    )
    
    try:
        print("🎤 Listening for speech...")
        print("   (Speak to test detection, ambient noise should be ignored)")
        print()
        
        capture.start()
        
        frame_count = 0
        speech_count = 0
        silence_count = 0
        last_state = None
        
        # Manual tracking since VAD doesn't provide these stats
        total_speech_frames = 0
        total_silence_frames = 0
        
        while True:
            # Get audio frame (with short timeout to keep queue from filling)
            frame = capture.get_frame(timeout=0.05)
            if frame is None:
                continue
            
            # Verify frame size (debugging)
            if frame_count == 0:
                print(f"   First frame size: {len(frame)} samples (expected: 512)")
                if len(frame) != 512:
                    print(f"   ⚠️ WARNING: Frame size mismatch! VAD may fail.")
            
            # Process through VAD
            is_speaking = vad.process_frame(frame)
            frame_count += 1
            
            # Track totals manually
            if is_speaking:
                total_speech_frames += 1
            else:
                total_silence_frames += 1
            
            # Check state changes
            if is_speaking != last_state:
                if is_speaking:
                    speech_count += 1
                    print(f"🎤 SPEECH DETECTED! (transition #{speech_count})")
                    print(f"   Probability: {vad.get_probability():.3f}")
                else:
                    silence_count += 1
                    print(f"🔇 Silence (transition #{silence_count})")
                    print(f"   Probability: {vad.get_probability():.3f}")
                
                last_state = is_speaking
            
            # Periodic stats (every 200 frames = ~6.4 seconds)
            if frame_count % 200 == 0:
                total = total_speech_frames + total_silence_frames
                
                if total > 0:
                    speech_pct = (total_speech_frames / total) * 100
                    print(f"\n📊 Stats: {frame_count} frames processed")
                    print(f"   Speech: {total_speech_frames} frames ({speech_pct:.1f}%)")
                    print(f"   Silence: {total_silence_frames} frames ({100-speech_pct:.1f}%)")
                    print(f"   Transitions: {speech_count + silence_count}")
                    print()
    
    except KeyboardInterrupt:
        print("\n\nℹ️  Test stopped by user")
        
        # Final stats
        stats = vad.get_stats()
        print("\n" + "="*60)
        print("FINAL STATISTICS")
        print("="*60)
        print(f"Total frames processed: {frame_count}")
        print(f"Speech frames: {total_speech_frames}")
        print(f"Silence frames: {total_silence_frames}")
        print(f"Speech segments: {stats['speech_segments']}")
        print(f"Speech transitions: {speech_count}")
        print(f"Silence transitions: {silence_count}")
        print(f"Total transitions: {speech_count + silence_count}")
        print()
        
        # Assessment
        print("="*60)
        print("ASSESSMENT")
        print("="*60)
        
        if speech_count > 0:
            print("✅ PASS: VAD detected speech")
        else:
            print("❌ FAIL: No speech detected - check microphone or speak louder")
        
        if silence_count > 0:
            print("✅ PASS: VAD detected silence")
        else:
            print("⚠️  WARNING: No silence detected - constant noise?")
        
        total_transitions = speech_count + silence_count
        if total_transitions >= 2:
            print("✅ PASS: Multiple state transitions (natural)")
        else:
            print("⚠️  WARNING: Few transitions - try speaking and pausing more")
        
        # Speech detection quality
        if total_speech_frames > 0 and total_silence_frames > 0:
            speech_pct = (total_speech_frames / frame_count) * 100
            print(f"\n📊 Speech/Silence ratio: {speech_pct:.1f}% speech, {100-speech_pct:.1f}% silence")
            if 10 < speech_pct < 90:
                print("✅ PASS: Good balance of speech and silence detected")
            else:
                print("⚠️  WARNING: Unusual ratio - check microphone sensitivity")
        
        print("\n" + "="*60)
        print("Test complete!")
        print("="*60 + "\n")
    
    finally:
        capture.stop()


if __name__ == "__main__":
    test_vad_detection()