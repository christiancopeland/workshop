"""
Quick Audio Diagnostic
Tests microphone detection and resampling
"""

import sounddevice as sd
import numpy as np
from audio_realtime import AudioStream

print("\n" + "="*60)
print("AUDIO DIAGNOSTIC")
print("="*60)

# Step 1: List all audio devices
print("\n1. Available Audio Devices:")
print("-" * 60)
devices = sd.query_devices()
for i, dev in enumerate(devices):
    if dev['max_input_channels'] > 0:
        print(f"Device {i}: {dev['name']}")
        print(f"  Input channels: {dev['max_input_channels']}")
        print(f"  Default sample rate: {dev['default_samplerate']}")
        print()

# Step 2: Check default device
print("\n2. Default Input Device:")
print("-" * 60)
default_device = sd.query_devices(kind='input')
print(f"Name: {default_device['name']}")
print(f"Channels: {default_device['max_input_channels']}")
print(f"Sample rate: {default_device['default_samplerate']}")
print()

# Step 3: Find Blue USB mic
print("\n3. Looking for Blue USB Mic...")
print("-" * 60)
blue_device = None
for i, dev in enumerate(devices):
    if 'blue' in dev['name'].lower() and dev['max_input_channels'] > 0:
        blue_device = i
        print(f"✅ Found Blue USB at device {i}")
        print(f"   Name: {dev['name']}")
        print(f"   Channels: {dev['max_input_channels']}")
        print(f"   Native rate: {dev['default_samplerate']}")
        break

if blue_device is None:
    print("⚠️  Blue USB mic not found")
    print("   Using default device instead")
    blue_device = None  # Will use default

print()

# Step 4: Test resampling capture
print("\n4. Testing Audio Capture with Resampling:")
print("-" * 60)

try:
    # Create stream with resampling
    stream = AudioStream(
        sample_rate=16000,                    # Target for VAD
        channels=2,                           # Stereo
        frame_size=512,                       # Target frames
        device_id=blue_device,                # Your mic
        hardware_sample_rate=44100 if blue_device else None  # Hardware rate
    )
    
    print("Stream configuration:")
    stats = stream.get_stats()
    print(f"  Hardware rate: {stats['hardware_rate']} Hz")
    print(f"  Target rate: {stats['target_rate']} Hz")
    print(f"  Resampling: {stats['resampling']}")
    print(f"  Device: {stats['device_id']}")
    print()
    
    # Start capture
    print("Starting capture...")
    if stream.start():
        print("✅ Stream started successfully!")
        
        # Capture a few frames
        print("\nCapturing 10 frames...")
        for i in range(10):
            frame = stream.get_frame(timeout=1.0)
            if frame is not None:
                print(f"  Frame {i+1}: {len(frame)} samples, "
                      f"range [{frame.min():.3f}, {frame.max():.3f}]")
            else:
                print(f"  Frame {i+1}: None (timeout)")
        
        print("\n✅ Capture successful!")
        stream.stop()
    else:
        print("❌ Failed to start stream")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
print("\nIf all tests passed, run: python integration_test_1_vad.py")
print()