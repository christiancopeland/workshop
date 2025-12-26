"""Unit tests for WhisperCpp wrapper"""

import sys
from pathlib import Path
import numpy as np
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path

# Mock dependencies
sys.modules['logger'] = MagicMock()

from whisper_wrapper import WhisperCpp, StreamingWhisper

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))


def test_whisper_init():
    """Test WhisperCpp initialization"""
    print("Test 1: Initialization")
    
    with patch('whisper_wrapper.Path.exists', return_value=True):
        whisper = WhisperCpp(
            model_path="/fake/model.bin",
            whisper_bin="/fake/whisper-cli"
        )
        
        assert whisper.language == "en"
        assert whisper.threads == 4
        print("✅ Initialization works\n")


def test_whisper_transcribe_array():
    """Test transcribing numpy array"""
    print("Test 2: Transcribe numpy array")
    
    with patch('whisper_wrapper.Path.exists', return_value=True):
        whisper = WhisperCpp(
            model_path="/fake/model.bin",
            whisper_bin="/fake/whisper-cli"
        )
        
        # Mock subprocess
        mock_result = Mock()
        mock_result.stdout = "And so my fellow Americans"
        mock_result.returncode = 0
        
        with patch('subprocess.run', return_value=mock_result):
            audio = np.random.randn(16000).astype(np.float32)
            text = whisper.transcribe_array(audio)
            
            assert "Americans" in text
            print(f"  Transcribed: {text}")
            print("✅ Array transcription works\n")


def test_whisper_transcribe_chunks():
    """Test transcribing multiple chunks"""
    print("Test 3: Transcribe chunks")
    
    with patch('whisper_wrapper.Path.exists', return_value=True):
        whisper = WhisperCpp(
            model_path="/fake/model.bin",
            whisper_bin="/fake/whisper-cli"
        )
        
        # Mock subprocess
        mock_result = Mock()
        mock_result.stdout = "This is a test"
        
        with patch('subprocess.run', return_value=mock_result):
            # Create 3 chunks
            chunks = [
                np.random.randn(8000).astype(np.float32),
                np.random.randn(8000).astype(np.float32),
                np.random.randn(8000).astype(np.float32)
            ]
            
            text = whisper.transcribe_chunks(chunks)
            
            assert "test" in text
            print(f"  Transcribed: {text}")
            print("✅ Chunk transcription works\n")


def test_streaming_whisper():
    """Test streaming transcription"""
    print("Test 4: Streaming transcription")
    
    with patch('whisper_wrapper.Path.exists', return_value=True):
        whisper = WhisperCpp(
            model_path="/fake/model.bin",
            whisper_bin="/fake/whisper-cli"
        )
        
        streamer = StreamingWhisper(
            whisper,
            window_size=3.0,
            overlap=0.5,
            sample_rate=16000
        )
        
        # Check buffer starts empty
        assert len(streamer.buffer) == 0
        
        # Add chunk (not enough for window)
        chunk1 = np.random.randn(8000).astype(np.float32)  # 0.5s
        result = streamer.add_audio(chunk1)
        assert result is None  # Not enough yet
        assert len(streamer.buffer) == 8000
        
        # Add more chunks to reach window size
        mock_result = Mock()
        mock_result.stdout = "Transcribed window"
        
        with patch('subprocess.run', return_value=mock_result):
            for i in range(5):  # Add 2.5s more = 3s total
                chunk = np.random.randn(8000).astype(np.float32)
                result = streamer.add_audio(chunk)
                
                if result:
                    print(f"    Window transcribed: {result}")
                    # Check buffer kept overlap
                    assert len(streamer.buffer) == streamer.overlap_samples
                    break
        
        print("✅ Streaming works\n")


def test_streaming_flush():
    """Test flushing remaining audio"""
    print("Test 5: Flush remaining audio")
    
    with patch('whisper_wrapper.Path.exists', return_value=True):
        whisper = WhisperCpp(
            model_path="/fake/model.bin",
            whisper_bin="/fake/whisper-cli"
        )
        
        streamer = StreamingWhisper(whisper)
        
        # Add some audio (not enough for full window)
        chunk = np.random.randn(8000).astype(np.float32)
        streamer.add_audio(chunk)
        
        assert len(streamer.buffer) == 8000
        
        # Flush
        mock_result = Mock()
        mock_result.stdout = "Final text"
        
        with patch('subprocess.run', return_value=mock_result):
            text = streamer.flush()
            
            assert "Final" in text
            assert len(streamer.buffer) == 0  # Buffer cleared
            print(f"  Flushed: {text}")
            print("✅ Flush works\n")


def test_whisper_stats():
    """Test stats reporting"""
    print("Test 6: Stats")
    
    with patch('whisper_wrapper.Path.exists', return_value=True):
        whisper = WhisperCpp(
            model_path="/fake/model.bin",
            whisper_bin="/fake/whisper-cli"
        )
        
        stats = whisper.get_stats()
        
        assert "model" in stats
        assert "language" in stats
        assert stats["threads"] == 4
        print(f"  Stats: {stats}")
        print("✅ Stats works\n")


if __name__ == "__main__":
    print("Running WhisperCpp unit tests...\n")
    
    test_whisper_init()
    test_whisper_transcribe_array()
    test_whisper_transcribe_chunks()
    test_streaming_whisper()
    test_streaming_flush()
    test_whisper_stats()
    
    print("✅ All unit tests passed!")
    print("\nNote: Hardware test requires actual whisper.cpp installation.")
    print("To test on your machine: python whisper_wrapper.py")