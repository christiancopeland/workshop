"""
Test OpenWakeWord's internal pipeline to find where it's failing.
The pipeline is: audio → melspectrogram → embedding → prediction
"""

import numpy as np
from openwakeword.model import Model

print("Testing OpenWakeWord internal pipeline...\n")

# Create model
print("Loading model...")
model = Model(wakeword_models=["hey_jarvis"])

# Create test audio - try different types
print("\n" + "="*60)
print("TEST 1: Random float32 audio (what we're using)")
print("="*60)
audio_float = np.random.randn(1280).astype(np.float32) * 0.1
prediction = model.predict(audio_float)
print(f"Input dtype: {audio_float.dtype}")
print(f"Input range: [{audio_float.min():.4f}, {audio_float.max():.4f}]")
print(f"Prediction: {prediction}")

print("\n" + "="*60)
print("TEST 2: Random int16 audio (alternative format)")
print("="*60)
audio_int16 = (np.random.randn(1280) * 10000).astype(np.int16)
prediction = model.predict(audio_int16)
print(f"Input dtype: {audio_int16.dtype}")
print(f"Input range: [{audio_int16.min()}, {audio_int16.max()}]")
print(f"Prediction: {prediction}")

print("\n" + "="*60)
print("TEST 3: Louder float32 audio")
print("="*60)
audio_loud = np.random.randn(1280).astype(np.float32) * 0.5
prediction = model.predict(audio_loud)
print(f"Input dtype: {audio_loud.dtype}")
print(f"Input range: [{audio_loud.min():.4f}, {audio_loud.max():.4f}]")
print(f"Prediction: {prediction}")

print("\n" + "="*60)
print("TEST 4: Sine wave (structured audio)")
print("="*60)
t = np.linspace(0, 0.08, 1280)  # 80ms
frequency = 440  # A note
audio_sine = (np.sin(2 * np.pi * frequency * t) * 0.3).astype(np.float32)
prediction = model.predict(audio_sine)
print(f"Input dtype: {audio_sine.dtype}")
print(f"Input range: [{audio_sine.min():.4f}, {audio_sine.max():.4f}]")
print(f"Prediction: {prediction}")

print("\n" + "="*60)
print("TEST 5: Check model internals")
print("="*60)

# Check what models are loaded
print(f"Model prediction_buffer_size: {model.prediction_buffer_size}")
print(f"Number of models: {len(model.models)}")

# Check if models dict is populated
if hasattr(model, 'models'):
    print(f"\nModels loaded:")
    for name in model.models.keys():
        print(f"  - {name}")

# Check melspectrogram model
if hasattr(model, 'melspectrogram_model'):
    print(f"\nMelspectrogram model exists: {model.melspectrogram_model is not None}")

# Check embedding model  
if hasattr(model, 'embedding_model'):
    print(f"Embedding model exists: {model.embedding_model is not None}")

print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)

# Check if ANY test produced non-zero score
all_zero = True
tests = [
    ("Random float32", audio_float),
    ("Random int16", audio_int16),
    ("Loud float32", audio_loud),
    ("Sine wave", audio_sine)
]

for name, audio in tests:
    pred = model.predict(audio)
    score = pred.get('hey_jarvis', 0.0)
    print(f"{name}: score = {score:.6f}")
    if score > 0.001:
        all_zero = False

if all_zero:
    print("\n❌ PROBLEM: ALL tests returned near-zero scores!")
    print("   This suggests the model pipeline is broken.")
    print("\n   Possible causes:")
    print("   1. Model files corrupted during download")
    print("   2. OpenWakeWord version incompatibility")
    print("   3. Audio preprocessing issue")
    print("\n   Try:")
    print("   1. pip install --upgrade openwakeword")
    print("   2. Redownload models")
    print("   3. Check OpenWakeWord version")
else:
    print("\n✅ At least one test worked!")
    print("   Issue is with audio format we're feeding it.")

# Check version
try:
    import openwakeword
    print(f"\nOpenWakeWord version: {openwakeword.__version__}")
except:
    print("\nCouldn't determine OpenWakeWord version")