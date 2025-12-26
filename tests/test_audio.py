# Save as test_blue_mic.py
import asyncio
import sounddevice as sd
import numpy as np

async def test_blue_mic():
    print("=== Testing Blue Microphones (device 9) ===")
    
    # Device 9 details from your output
    device_info = sd.query_devices(9)
    print(f"Name: {device_info['name']}")
    print(f"Channels: {device_info['max_input_channels']}")
    print(f"Default SR: {device_info['default_samplerate']}")
    
    rates = [44100, 48000]  # Start with native 44100
    
    for rate in rates:
        try:
            print(f"\n→ Trying {rate}Hz...")
            chunks = 0
            
            def callback(indata, frames, time, status):
                nonlocal chunks
                if status:
                    print(f"  Status: {status}")
                print(f"  ✓ Chunk #{chunks} ({len(indata)} frames)")
                chunks += 1
                if chunks >= 5:  # Stop after 5 chunks (~500ms)
                    raise sd.CallbackStop
            
            with sd.InputStream(
                device=9,           # Blue mic
                samplerate=rate,
                channels=1,         # Mono (even though mic is stereo)
                dtype=np.float32,
                blocksize=2048,     # ~46ms @ 44.1kHz
                callback=callback,
                latency='low'
            ) as stream:
                stream.start()
                await asyncio.sleep(1)
                print(f"  → SUCCESS: {rate}Hz works!")
                return rate
                
        except Exception as e:
            print(f"  ✗ {rate}Hz failed: {e}")
    
    print("No working rate found!")
    return None

asyncio.run(test_blue_mic())
