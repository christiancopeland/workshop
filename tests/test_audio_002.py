"""
Unit tests for AudioFramePipeline (AUDIO-002)
Tests frame processing pipeline with mocked dependencies.
"""

import sys
import numpy as np
import time
from unittest.mock import MagicMock, Mock, patch, call

# Mock dependencies BEFORE imports
sys.modules['logger'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['torch.hub'] = MagicMock()
sys.modules['audio_realtime'] = MagicMock()
sys.modules['audio_playback'] = MagicMock()

# Import module
import audio_pipeline


def test_pipeline_initialization():
    """Test pipeline initializes all components"""
    print("Test 1: Pipeline initialization")
    
    mock_model = Mock()
    mock_utils = [Mock(), Mock(), Mock()]
    
    with patch('vad.torch') as mock_torch:
        mock_torch.hub.load.return_value = (mock_model, mock_utils)
        
        # Mock audio capture
        with patch('audio_pipeline.RealtimeAudioCapture') as mock_capture_class:
            mock_capture = Mock()
            mock_capture_class.return_value = mock_capture
            
            pipeline = audio_pipeline.AudioFramePipeline(
                sample_rate=16000,
                frame_size=512,
                timeout_s=30.0
            )
            
            # Verify components initialized
            assert pipeline.sample_rate == 16000
            assert pipeline.frame_size == 512
            assert pipeline.timeout_s == 30.0
            assert pipeline.vad is not None
            assert pipeline.speech_detector is not None
            assert pipeline.interruption_detector is not None
            assert not pipeline.listening
            assert not pipeline.assistant_speaking
            
            # Verify audio capture created correctly
            mock_capture_class.assert_called_once_with(
                sample_rate=16000,
                frame_size=512
            )
            
            print("  All components initialized ✓")
            print("✅ Initialization works\n")


def test_start_stop_listening():
    """Test start/stop listening control"""
    print("Test 2: Start/stop listening")
    
    mock_model = Mock()
    mock_utils = [Mock(), Mock(), Mock()]
    
    with patch('vad.torch') as mock_torch:
        mock_torch.hub.load.return_value = (mock_model, mock_utils)
        
        with patch('audio_pipeline.RealtimeAudioCapture') as mock_capture_class:
            mock_capture = Mock()
            mock_capture_class.return_value = mock_capture
            
            pipeline = audio_pipeline.AudioFramePipeline()
            
            # Initially not listening
            assert not pipeline.listening
            
            # Start listening
            pipeline.start_listening()
            assert pipeline.listening
            mock_capture.start.assert_called_once()
            print("  Started listening ✓")
            
            # Stop listening
            pipeline.stop_listening()
            assert not pipeline.listening
            mock_capture.stop.assert_called_once()
            print("  Stopped listening ✓")
            
            print("✅ Start/stop works\n")


def test_assistant_speaking_state():
    """Test assistant speaking state management"""
    print("Test 3: Assistant speaking state")
    
    mock_model = Mock()
    mock_utils = [Mock(), Mock(), Mock()]
    
    with patch('vad.torch') as mock_torch:
        mock_torch.hub.load.return_value = (mock_model, mock_utils)
        
        with patch('audio_pipeline.RealtimeAudioCapture'):
            pipeline = audio_pipeline.AudioFramePipeline()
            
            # Initially not speaking
            assert not pipeline.assistant_speaking
            
            # Set speaking
            pipeline.set_assistant_speaking(True)
            assert pipeline.assistant_speaking
            assert pipeline.interruption_detector.assistant_speaking
            print("  Assistant speaking set ✓")
            
            # Clear speaking
            pipeline.set_assistant_speaking(False)
            assert not pipeline.assistant_speaking
            assert not pipeline.interruption_detector.assistant_speaking
            print("  Assistant speaking cleared ✓")
            
            print("✅ State management works\n")


def test_frame_routing_normal():
    """Test frame routing in normal listening mode"""
    print("Test 4: Frame routing (normal mode)")
    
    mock_model = Mock()
    mock_model.return_value.item.return_value = 0.8
    mock_utils = [Mock(), Mock(), Mock()]
    
    with patch('vad.torch') as mock_torch:
        mock_torch.hub.load.return_value = (mock_model, mock_utils)
        mock_tensor = Mock()
        mock_tensor.float.return_value = mock_tensor
        mock_torch.from_numpy.return_value = mock_tensor
        
        with patch('audio_pipeline.RealtimeAudioCapture'):
            pipeline = audio_pipeline.AudioFramePipeline()
            
            # Not assistant speaking - should route to speech detector
            pipeline.assistant_speaking = False
            
            audio = np.random.randn(512).astype(np.float32)
            
            # Process frames (won't return segment without full sequence)
            segment, reason = pipeline.process_frame(audio)
            
            # Should return None/None (no complete segment yet)
            assert segment is None
            assert reason is None
            
            print("  Frame routed to speech detector ✓")
            print("✅ Normal routing works\n")


def test_frame_routing_interruption():
    """Test frame routing during assistant speech"""
    print("Test 5: Frame routing (interruption mode)")
    
    mock_model = Mock()
    mock_model.return_value.item.return_value = 0.8
    mock_utils = [Mock(), Mock(), Mock()]
    
    with patch('vad.torch') as mock_torch:
        mock_torch.hub.load.return_value = (mock_model, mock_utils)
        mock_tensor = Mock()
        mock_tensor.float.return_value = mock_tensor
        mock_torch.from_numpy.return_value = mock_tensor
        
        with patch('audio_pipeline.RealtimeAudioCapture'):
            pipeline = audio_pipeline.AudioFramePipeline()
            
            # Set assistant speaking
            pipeline.set_assistant_speaking(True)
            
            audio = np.random.randn(512).astype(np.float32)
            
            # Process frames to trigger interruption (need 3 frames)
            for i in range(3):
                segment, reason = pipeline.process_frame(audio)
                
                if i < 2:
                    assert segment is None
                else:
                    # 3rd frame should trigger interruption
                    assert segment is None
                    assert reason == "interrupted"
                    assert pipeline.interruptions_detected == 1
            
            print("  Interruption detected ✓")
            print("✅ Interruption routing works\n")


def test_statistics_aggregation():
    """Test statistics from all components"""
    print("Test 6: Statistics aggregation")
    
    mock_model = Mock()
    mock_utils = [Mock(), Mock(), Mock()]
    
    with patch('vad.torch') as mock_torch:
        mock_torch.hub.load.return_value = (mock_model, mock_utils)
        
        with patch('audio_pipeline.RealtimeAudioCapture'):
            pipeline = audio_pipeline.AudioFramePipeline(timeout_s=15.0)
            
            stats = pipeline.get_stats()
            
            # Check pipeline stats
            assert stats["sample_rate"] == 16000
            assert stats["frame_size"] == 512
            assert stats["timeout_s"] == 15.0
            assert stats["listening"] == False
            assert stats["segments_captured"] == 0
            assert stats["interruptions_detected"] == 0
            
            # Check component stats included
            assert "vad" in stats
            assert "speech_detector" in stats
            assert "interruption_detector" in stats
            
            print(f"  Pipeline stats: sample_rate={stats['sample_rate']}, timeout={stats['timeout_s']} ✓")
            print(f"  Component stats included: VAD, speech_detector, interruption_detector ✓")
            print("✅ Statistics work\n")


def test_context_manager():
    """Test context manager interface"""
    print("Test 7: Context manager")
    
    mock_model = Mock()
    mock_utils = [Mock(), Mock(), Mock()]
    
    with patch('vad.torch') as mock_torch:
        mock_torch.hub.load.return_value = (mock_model, mock_utils)
        
        with patch('audio_pipeline.RealtimeAudioCapture') as mock_capture_class:
            mock_capture = Mock()
            mock_capture_class.return_value = mock_capture
            
            # Use as context manager
            with audio_pipeline.AudioFramePipeline() as pipeline:
                # Should be listening
                assert pipeline.listening
                mock_capture.start.assert_called_once()
                print("  Started on __enter__ ✓")
            
            # Should have stopped
            mock_capture.stop.assert_called_once()
            print("  Stopped on __exit__ ✓")
            print("✅ Context manager works\n")


def test_reset_functionality():
    """Test reset clears all state"""
    print("Test 8: Reset functionality")
    
    mock_model = Mock()
    mock_utils = [Mock(), Mock(), Mock()]
    
    with patch('vad.torch') as mock_torch:
        mock_torch.hub.load.return_value = (mock_model, mock_utils)
        
        with patch('audio_pipeline.RealtimeAudioCapture'):
            pipeline = audio_pipeline.AudioFramePipeline()
            
            # Build up state
            pipeline.segments_captured = 5
            pipeline.interruptions_detected = 2
            
            # Reset
            pipeline.reset()
            
            # Verify cleared
            assert pipeline.segments_captured == 0
            assert pipeline.interruptions_detected == 0
            
            print("  Pipeline state cleared ✓")
            print("✅ Reset works\n")


def test_interruption_callback():
    """Test interruption callback registration and firing"""
    print("Test 9: Interruption callback")
    
    mock_model = Mock()
    mock_model.return_value.item.return_value = 0.8
    mock_utils = [Mock(), Mock(), Mock()]
    
    with patch('vad.torch') as mock_torch:
        mock_torch.hub.load.return_value = (mock_model, mock_utils)
        mock_tensor = Mock()
        mock_tensor.float.return_value = mock_tensor
        mock_torch.from_numpy.return_value = mock_tensor
        
        with patch('audio_pipeline.RealtimeAudioCapture'):
            pipeline = audio_pipeline.AudioFramePipeline()
            
            # Register callback
            callback_fired = [False]
            def callback():
                callback_fired[0] = True
            
            pipeline.on_interruption(callback)
            pipeline.set_assistant_speaking(True)
            
            # Trigger interruption
            audio = np.random.randn(512).astype(np.float32)
            for i in range(3):
                pipeline.process_frame(audio)
            
            assert callback_fired[0], "Callback should have fired"
            
            print("  Callback registered ✓")
            print("  Callback fired on interruption ✓")
            print("✅ Callback mechanism works\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("WORKSHOP PHASE 2: AUDIO-002 UNIT TESTS")
    print("AudioFramePipeline (Frame processing pipeline)")
    print("="*60 + "\n")
    
    try:
        test_pipeline_initialization()
        test_start_stop_listening()
        test_assistant_speaking_state()
        test_frame_routing_normal()
        test_frame_routing_interruption()
        test_statistics_aggregation()
        test_context_manager()
        test_reset_functionality()
        test_interruption_callback()
        
        print("="*60)
        print("✅ ALL 9 TESTS PASSED!")
        print("="*60)
        print("\nAUDIO-002: Frame Processing Pipeline - COMPLETE ✅")
        print("\nFeatures validated:")
        print("  • Pipeline initialization with all components")
        print("  • Start/stop listening control")
        print("  • Assistant speaking state management")
        print("  • Frame routing (normal vs interruption mode)")
        print("  • Interruption detection during TTS")
        print("  • Statistics aggregation from all components")
        print("  • Context manager interface")
        print("  • Reset functionality")
        print("  • Interruption callback mechanism")
        print("\nNext: WAKE-001 - Wake word integration")
        print("="*60 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)