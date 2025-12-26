"""
OpenWakeWord Prediction Key Diagnostic
Identifies the actual prediction dictionary keys returned by the model.
"""

import numpy as np
from openwakeword.model import Model

print("="*60)
print("OpenWakeWord Prediction Key Diagnostic")
print("="*60)

# Initialize model
print("\nInitializing model with 'hey_jarvis'...")
model = Model(wakeword_models=["hey_jarvis"])

# Create test audio (1280 samples of silence)
test_audio = np.zeros(1280, dtype=np.float32)

print("Generating prediction on silence...")
prediction = model.predict(test_audio)

print("\n" + "="*60)
print("PREDICTION DICTIONARY CONTENTS:")
print("="*60)

if not prediction:
    print("âŒ ERROR: Prediction dictionary is empty!")
else:
    print(f"Found {len(prediction)} keys in prediction:\n")
    for key, value in prediction.items():
        print(f"  Key: '{key}'")
        print(f"    Value: {value:.6f}")
        print(f"    Type: {type(value)}")
        print()

print("="*60)
print("DIAGNOSIS:")
print("="*60)

if not prediction:
    print("âŒ Model returned empty dictionary")
    print("   Possible causes:")
    print("   1. Model files corrupted")
    print("   2. OpenWakeWord installation issue")
    print("   3. Model not properly loaded")
else:
    # Check if "hey_jarvis" exists
    if "hey_jarvis" in prediction:
        print("âœ… Key 'hey_jarvis' exists in predictions")
    else:
        print("âŒ Key 'hey_jarvis' NOT FOUND")
        print(f"   Available keys: {list(prediction.keys())}")
        print("   ^ Use one of these keys instead")

print("\n" + "="*60)
print("SOLUTION:")
print("="*60)

if prediction:
    actual_key = list(prediction.keys())[0]  # Get first key
    print(f"Update wake_word.py line 75 to use: '{actual_key}'")
    print("\nOR better - use this pattern:")
    print("""
    # Get any prediction value (there should only be one model loaded)
    if prediction:
        score = list(prediction.values())[0]
    else:
        score = 0.0
    """)
else:
    print("1. Reinstall OpenWakeWord:")
    print("   pip uninstall openwakeword")
    print("   pip install openwakeword")
    print("2. Re-download models:")
    print("   python -c 'from openwakeword import utils; utils.download_models()'")

print("\n" + "="*60)