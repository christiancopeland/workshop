"""Unit tests for WakeWordDetector"""

import sys
import numpy as np
from unittest.mock import MagicMock, Mock, patch

# Mock dependencies
sys.modules['logger'] = MagicMock()
sys.modules['openwakeword'] = MagicMock()
sys.modules['openwakeword.model'] = MagicMock()

from wake_word import WakeWordDetector, WakeWordBuffer, OPENWAKEWORD_AVAILABLE


def test_wake_word_init():
    """Test WakeWordDetector initialization"""
    print("Test 1: Initialization")
    
    # Mock Model class
    mock_model = Mock()
    
    with patch('wake_word.Model', return_value=mock_model):
        with patch('wake_word.OPENWAKEWORD_AVAILABLE', True):
            detector = WakeWordDetector(
                model_name="hey_jarvis",
                threshold=0.5
            )
            
            assert detector.model_name == "hey_jarvis"
            assert detector.threshold == 0.5
            assert detector.chunk_size == 1280
            assert detector.frames_processed == 0
            assert detector.detections == 0
            
            print("✅ Initialization works\n")


def test_wake_word_detect():
    """Test wake word detection"""
    print("Test 2: Detect wake word")
    
    # Mock Model
    mock_model = Mock()
    mock_model.predict.return_value = {"hey_jarvis": 0.8}  # High confidence
    
    with patch('wake_word.Model', return_value=mock_model):
        with patch('wake_word.OPENWAKEWORD_AVAILABLE', True):
            detector = WakeWordDetector(threshold=0.5)
            
            # Test with correct chunk size
            audio = np.random.randn(1280).astype(np.float32)
            detected = detector.detect(audio)
            
            assert detected == True  # 0.8 > 0.5 threshold
            assert detector.detections == 1
            assert detector.frames_processed == 1
            
            print(f"  Detected: {detected}")
            print("✅ Detection works\n")


def test_wake_word_no_detect():
    """Test no detection with low confidence"""
    print("Test 3: No detection (low confidence)")
    
    # Mock Model with low confidence
    mock_model = Mock()
    mock_model.predict.return_value = {"hey_jarvis": 0.2}
    
    with patch('wake_word.Model', return_value=mock_model):
        with patch('wake_word.OPENWAKEWORD_AVAILABLE', True):
            detector = WakeWordDetector(threshold=0.5)
            
            audio = np.random.randn(1280).astype(np.float32)
            detected = detector.detect(audio)
            
            assert detected == False  # 0.2 < 0.5 threshold
            assert detector.detections == 0
            assert detector.frames_processed == 1
            
            print(f"  Detected: {detected}")
            print("✅ No detection works\n")


def test_wake_word_buffer():
    """Test WakeWordBuffer"""
    print("Test 4: Wake word buffer")
    
    # Mock Model
    mock_model = Mock()
    mock_model.predict.return_value = {"hey_jarvis": 0.8}
    
    with patch('wake_word.Model', return_value=mock_model):
        with patch('wake_word.OPENWAKEWORD_AVAILABLE', True):
            detector = WakeWordDetector(threshold=0.5)
            buffer = WakeWordBuffer(detector)
            
            # Add frames smaller than chunk size
            # Need 1280 samples total = 3 frames of 512 samples
            
            # Frame 1: Not enough yet
            frame1 = np.random.randn(512).astype(np.float32)
            detected = buffer.add_frame(frame1)
            assert detected == False
            assert len(buffer.buffer) == 512
            
            # Frame 2: Still not enough
            frame2 = np.random.randn(512).astype(np.float32)
            detected = buffer.add_frame(frame2)
            assert detected == False
            assert len(buffer.buffer) == 1024
            
            # Frame 3: Now we have enough (1536 > 1280)
            frame3 = np.random.randn(512).astype(np.float32)
            detected = buffer.add_frame(frame3)
            assert detected == True  # Mock returns 0.8
            assert len(buffer.buffer) == 256  # Remaining samples
            
            print(f"  Final detection: {detected}")
            print(f"  Remaining buffer: {len(buffer.buffer)} samples")
            print("✅ Buffer works\n")


def test_wake_word_process_stream():
    """Test processing longer stream"""
    print("Test 5: Process stream")
    
    # Mock Model
    mock_model = Mock()
    mock_model.predict.side_effect = [
        {"hey_jarvis": 0.8},  # Chunk 0: detect
        {"hey_jarvis": 0.2},  # Chunk 1: no detect
        {"hey_jarvis": 0.9},  # Chunk 2: detect
    ]
    
    with patch('wake_word.Model', return_value=mock_model):
        with patch('wake_word.OPENWAKEWORD_AVAILABLE', True):
            detector = WakeWordDetector(threshold=0.5)
            
            # Create 3 chunks of audio
            audio = np.random.randn(1280 * 3).astype(np.float32)
            detections = detector.process_stream(audio)
            
            assert len(detections) == 2  # Chunks 0 and 2
            assert 0 in detections
            assert 2 in detections
            
            print(f"  Detections at chunks: {detections}")
            print("✅ Stream processing works\n")


def test_wake_word_stats():
    """Test statistics reporting"""
    print("Test 6: Stats")
    
    mock_model = Mock()
    mock_model.predict.return_value = {"hey_jarvis": 0.8}
    
    with patch('wake_word.Model', return_value=mock_model):
        with patch('wake_word.OPENWAKEWORD_AVAILABLE', True):
            detector = WakeWordDetector(threshold=0.5)
            
            # Process some frames
            for i in range(5):
                audio = np.random.randn(1280).astype(np.float32)
                detector.detect(audio)
            
            stats = detector.get_stats()
            
            assert stats["frames_processed"] == 5
            assert stats["detections"] == 5  # All above threshold
            assert stats["detection_rate"] == 1.0
            
            print(f"  Stats: {stats}")
            print("✅ Stats work\n")


def test_wake_word_reset():
    """Test reset functionality"""
    print("Test 7: Reset")
    
    mock_model = Mock()
    mock_model.predict.return_value = {"hey_jarvis": 0.8}
    
    with patch('wake_word.Model', return_value=mock_model):
        with patch('wake_word.OPENWAKEWORD_AVAILABLE', True):
            detector = WakeWordDetector(threshold=0.5)
            buffer = WakeWordBuffer(detector)
            
            # Add some audio
            buffer.add_frame(np.random.randn(512).astype(np.float32))
            assert len(buffer.buffer) > 0
            
            # Reset
            buffer.reset()
            
            assert len(buffer.buffer) == 0
            assert mock_model.reset.called
            
            print("✅ Reset works\n")


if __name__ == "__main__":
    print("Running WakeWordDetector unit tests...\n")
    
    test_wake_word_init()
    test_wake_word_detect()
    test_wake_word_no_detect()
    test_wake_word_buffer()
    test_wake_word_process_stream()
    test_wake_word_stats()
    test_wake_word_reset()
    
    print("✅ All unit tests passed!")
    print("\nNote: Hardware test requires openwakeword installed.")
    print("To test on your machine: python wake_word.py")