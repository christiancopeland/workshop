#!/usr/bin/env python3
"""
Voice Interface Test Script
Run this to debug microphone and Whisper setup
"""

import sys

def test_sounddevice():
    """Test 1: Can we import sounddevice?"""
    print("\n=== Test 1: sounddevice import ===")
    try:
        import sounddevice as sd
        print("✅ sounddevice imported successfully")
        
        print("\nAvailable audio devices:")
        print(sd.query_devices())
        
        default = sd.query_devices(kind='input')
        print(f"\nDefault input device: {default['name']}")
        print(f"  Sample rate: {default['default_samplerate']}")
        print(f"  Channels: {default['max_input_channels']}")
        
        return True
    except ImportError as e:
        print(f"❌ sounddevice not installed: {e}")
        print("   Install with: pip install sounddevice")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_recording():
    """Test 2: Can we record audio?"""
    print("\n=== Test 2: Recording ===")
    try:
        import sounddevice as sd
        import numpy as np
        
        print("Recording 2 seconds of audio... Speak now!")
        audio = sd.rec(
            int(2 * 16000),
            samplerate=16000,
            channels=1,
            dtype=np.float32
        )
        sd.wait()
        
        audio = audio.flatten()
        amplitude = np.abs(audio).mean()
        max_amp = np.abs(audio).max()
        
        print(f"✅ Recorded {len(audio)} samples")
        print(f"   Mean amplitude: {amplitude:.6f}")
        print(f"   Max amplitude: {max_amp:.6f}")
        
        if max_amp < 0.001:
            print("⚠️  Very low amplitude - microphone may not be working")
            return False
        
        return True, audio
        
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        return False, None

def test_whisper():
    """Test 3: Can we load and use Whisper?"""
    print("\n=== Test 3: Whisper ===")
    
    # Try faster-whisper first
    try:
        from faster_whisper import WhisperModel
        print("Loading faster-whisper (base.en)...")
        model = WhisperModel("base.en", device="cuda", compute_type="float16")
        print("✅ faster-whisper loaded (CUDA)")
        return model, "faster"
    except ImportError:
        print("faster-whisper not installed")
    except Exception as e:
        print(f"faster-whisper failed: {e}")
        # Try CPU
        try:
            from faster_whisper import WhisperModel
            print("Trying faster-whisper on CPU...")
            model = WhisperModel("base.en", device="cpu", compute_type="int8")
            print("✅ faster-whisper loaded (CPU)")
            return model, "faster"
        except Exception as e2:
            print(f"CPU also failed: {e2}")
    
    # Try openai-whisper
    try:
        import whisper
        print("Loading openai-whisper (base.en)...")
        model = whisper.load_model("base.en")
        print("✅ openai-whisper loaded")
        return model, "openai"
    except ImportError:
        print("❌ No Whisper library found")
        print("   Install with: pip install faster-whisper")
        print("   Or: pip install openai-whisper")
        return None, None
    except Exception as e:
        print(f"❌ openai-whisper failed: {e}")
        return None, None

def test_transcription(model, model_type, audio):
    """Test 4: Can we transcribe audio?"""
    print("\n=== Test 4: Transcription ===")
    
    if model is None:
        print("❌ Skipping - no model loaded")
        return
    
    try:
        print("Transcribing recorded audio...")
        
        if model_type == "faster":
            segments, info = model.transcribe(audio, language="en", vad_filter=True)
            text = " ".join(s.text for s in segments).strip()
        else:
            result = model.transcribe(audio, language="en")
            text = result["text"].strip()
        
        if text:
            print(f"✅ Transcription: '{text}'")
        else:
            print("⚠️  Empty transcription (was there speech?)")
            
    except Exception as e:
        print(f"❌ Transcription failed: {e}")

def test_wake_word(model, model_type):
    """Test 5: Wake word detection"""
    print("\n=== Test 5: Wake Word Detection ===")
    
    if model is None:
        print("❌ Skipping - no model loaded")
        return
    
    try:
        import sounddevice as sd
        import numpy as np
        
        print("Say 'workshop' in the next 3 seconds...")
        audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype=np.float32)
        sd.wait()
        audio = audio.flatten()
        
        print("Transcribing...")
        
        if model_type == "faster":
            segments, _ = model.transcribe(audio, language="en", vad_filter=True)
            text = " ".join(s.text for s in segments).strip().lower()
        else:
            result = model.transcribe(audio, language="en")
            text = result["text"].strip().lower()
        
        print(f"Heard: '{text}'")
        
        if "workshop" in text:
            print("✅ Wake word 'workshop' detected!")
        else:
            print("⚠️  Wake word not detected")
            print("   (You may need to speak louder/clearer)")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

def main():
    print("=" * 50)
    print("Workshop Voice Interface Test")
    print("=" * 50)
    
    # Test 1: sounddevice
    if not test_sounddevice():
        print("\n❌ Cannot proceed without sounddevice")
        return
    
    # Test 2: Recording
    result = test_recording()
    if isinstance(result, tuple):
        success, audio = result
    else:
        success, audio = result, None
    
    if not success:
        print("\n❌ Cannot proceed without recording")
        return
    
    # Test 3: Whisper
    model, model_type = test_whisper()
    
    # Test 4: Transcription
    if audio is not None and model is not None:
        test_transcription(model, model_type, audio)
    
    # Test 5: Wake word
    if model is not None:
        test_wake_word(model, model_type)
    
    print("\n" + "=" * 50)
    print("Test complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
