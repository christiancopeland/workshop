"""
Unit tests for Workshop context awareness.

Run with: python -m pytest test_context.py -v
Or: python test_context.py
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import os

# Import the module to test
import sys
sys.path.insert(0, str(Path(__file__).parent))

from context import (
    DeviceInfo, FileChange, DevContext, ContextAwareness,
    get_context_manager, get_current_context
)


class TestDeviceInfo(unittest.TestCase):
    """Test DeviceInfo dataclass"""
    
    def test_arduino_detection_esp32(self):
        """ESP32 devices should be detected as Arduino"""
        device = DeviceInfo(
            port="/dev/ttyUSB0",
            description="CP2102 USB to UART Bridge Controller (ESP32)"
        )
        self.assertTrue(device.is_arduino)
    
    def test_arduino_detection_nano(self):
        """Arduino Nano should be detected"""
        device = DeviceInfo(
            port="/dev/ttyACM0",
            description="Arduino Nano ESP32"
        )
        self.assertTrue(device.is_arduino)
    
    def test_non_arduino_device(self):
        """Non-Arduino devices should not be flagged"""
        device = DeviceInfo(
            port="/dev/ttyS0",
            description="Standard Serial Port"
        )
        self.assertFalse(device.is_arduino)
    
    def test_device_string_representation(self):
        """Device should have readable string representation"""
        device = DeviceInfo(
            port="/dev/ttyUSB0",
            description="ESP32 Board"
        )
        self.assertEqual(str(device), "ESP32 Board (/dev/ttyUSB0)")


class TestFileChange(unittest.TestCase):
    """Test FileChange dataclass"""
    
    def test_age_calculation(self):
        """Should correctly calculate file age"""
        # File modified 30 seconds ago
        past_time = datetime.now() - timedelta(seconds=30)
        file_change = FileChange(
            path=Path("/test/file.py"),
            modified_time=past_time,
            size=1024
        )
        
        # Age should be approximately 30 seconds (allow 1 second tolerance)
        self.assertAlmostEqual(file_change.age_seconds, 30, delta=1)
    
    def test_string_representation(self):
        """Should have readable string representation"""
        past_time = datetime.now() - timedelta(seconds=45)
        file_change = FileChange(
            path=Path("/test/example.ino"),
            modified_time=past_time,
            size=2048
        )
        
        result = str(file_change)
        self.assertIn("example.ino", result)
        self.assertIn("45s ago", result)


class TestDevContext(unittest.TestCase):
    """Test DevContext dataclass"""
    
    def test_empty_context_string(self):
        """Empty context should still produce valid string"""
        context = DevContext()
        result = str(context)
        self.assertIn("Development Context", result)
    
    def test_context_with_devices(self):
        """Context with devices should include them in string"""
        context = DevContext(
            connected_devices=[
                DeviceInfo(port="/dev/ttyUSB0", description="ESP32"),
                DeviceInfo(port="/dev/ttyACM0", description="Arduino Nano")
            ]
        )
        result = str(context)
        self.assertIn("ESP32", result)
        self.assertIn("Arduino Nano", result)
    
    def test_context_with_recent_files(self):
        """Context with recent files should include them"""
        context = DevContext(
            recent_files=[
                FileChange(
                    path=Path("/test/sketch.ino"),
                    modified_time=datetime.now(),
                    size=1024
                )
            ]
        )
        result = str(context)
        self.assertIn("sketch.ino", result)


class TestContextAwareness(unittest.TestCase):
    """Test ContextAwareness main class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary project directory
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir) / "test_project"
        self.project_path.mkdir()
        
        # Create test files
        self.test_files = []
        for name in ["sketch.ino", "main.cpp", "config.h", "README.md"]:
            file_path = self.project_path / name
            file_path.write_text(f"Test content for {name}")
            self.test_files.append(file_path)
        
        # Initialize context manager
        self.context = ContextAwareness(
            project_paths=[self.project_path],
            indexable_extensions={'.ino', '.cpp', '.h', '.md', '.py'}
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Context manager should initialize properly"""
        self.assertIsNotNone(self.context)
        self.assertEqual(len(self.context.project_paths), 1)
        self.assertEqual(self.context.project_paths[0], self.project_path)
    
    def test_get_context(self):
        """Should return DevContext object"""
        context = self.context.get_context()
        self.assertIsInstance(context, DevContext)
        self.assertIsInstance(context.timestamp, datetime)
    
    @patch('context.subprocess.run')
    def test_active_window_macos(self, mock_run):
        """Should detect active window on macOS"""
        # Mock osascript response
        mock_run.return_value = Mock(stdout="Visual Studio Code\n")
        
        # Force macOS detection
        self.context._system = "Darwin"
        
        window, app = self.context._get_active_window()
        self.assertEqual(app, "Visual Studio Code")
    
    @patch('serial.tools.list_ports.comports')
    def test_device_detection(self, mock_comports):
        """Should detect connected USB devices"""
        # Mock serial ports
        mock_port = Mock()
        mock_port.device = "/dev/ttyUSB0"
        mock_port.description = "CP2102 USB to UART Bridge Controller"
        mock_port.hwid = "USB VID:PID=10C4:EA60"
        mock_port.vid = 0x10C4
        mock_port.pid = 0xEA60
        
        mock_comports.return_value = [mock_port]
        
        devices = self.context._get_connected_devices()
        
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0].port, "/dev/ttyUSB0")
        self.assertTrue(devices[0].is_arduino)
    
    def test_recent_files_detection(self):
        """Should detect recently modified files"""
        # Touch one of the test files to make it "recent"
        recent_file = self.project_path / "sketch.ino"
        recent_file.touch()
        
        # Get files modified in last 60 minutes
        recent = self.context._get_recent_files(minutes=60)
        
        # Should find at least the touched file
        file_names = [f.path.name for f in recent]
        self.assertIn("sketch.ino", file_names)
    
    def test_recent_files_filtering(self):
        """Should only include files with indexable extensions"""
        # Create non-indexable file
        ignored_file = self.project_path / "binary.bin"
        ignored_file.write_bytes(b"\x00\x01\x02")
        
        recent = self.context._get_recent_files(minutes=60)
        
        # Binary file should not be included
        file_names = [f.path.name for f in recent]
        self.assertNotIn("binary.bin", file_names)
    
    def test_detect_arduino_boards(self):
        """Should filter for only Arduino boards"""
        with patch('serial.tools.list_ports.comports') as mock_comports:
            # Mock both Arduino and non-Arduino devices
            arduino_port = Mock()
            arduino_port.device = "/dev/ttyUSB0"
            arduino_port.description = "Arduino Nano ESP32"
            arduino_port.hwid = "USB"
            arduino_port.vid = 0x2341
            arduino_port.pid = 0x0043
            
            other_port = Mock()
            other_port.device = "/dev/ttyS0"
            other_port.description = "Standard Serial Port"
            other_port.hwid = ""
            other_port.vid = None
            other_port.pid = None
            
            mock_comports.return_value = [arduino_port, other_port]
            
            boards = self.context.detect_arduino_boards()
            
            # Should only return Arduino board
            self.assertEqual(len(boards), 1)
            self.assertEqual(boards[0].port, "/dev/ttyUSB0")
    
    def test_project_detection_for_file(self):
        """Should correctly identify which project a file belongs to"""
        test_file = self.project_path / "sketch.ino"
        
        project = self.context.get_project_for_file(test_file)
        
        self.assertEqual(project, "test_project")
    
    def test_format_context_for_llm(self):
        """Should format context as LLM-readable string"""
        # Create context with known data
        context = DevContext(
            active_window="sketch.ino - Arduino IDE",
            active_app="Arduino IDE",
            connected_devices=[
                DeviceInfo(port="/dev/ttyUSB0", description="ESP32 Dev Board")
            ]
        )
        
        formatted = self.context.format_context_for_llm(context)
        
        # Should mention the sketch being edited
        self.assertIn("sketch.ino", formatted)
        
        # Should mention connected board
        self.assertIn("ESP32", formatted)
    
    def test_should_suggest_compile(self):
        """Should suggest compile when board connected and sketch modified"""
        # Create context with Arduino board and recent .ino file
        context = DevContext(
            connected_devices=[
                DeviceInfo(port="/dev/ttyUSB0", description="Arduino Nano ESP32")
            ],
            recent_files=[
                FileChange(
                    path=Path("/test/sketch.ino"),
                    modified_time=datetime.now() - timedelta(seconds=60),
                    size=1024
                )
            ]
        )
        
        should_suggest = self.context.should_suggest_compile(context)
        self.assertTrue(should_suggest)
    
    def test_should_not_suggest_compile_no_board(self):
        """Should not suggest compile without board connected"""
        context = DevContext(
            connected_devices=[],  # No devices
            recent_files=[
                FileChange(
                    path=Path("/test/sketch.ino"),
                    modified_time=datetime.now(),
                    size=1024
                )
            ]
        )
        
        should_suggest = self.context.should_suggest_compile(context)
        self.assertFalse(should_suggest)
    
    def test_should_not_suggest_compile_old_sketch(self):
        """Should not suggest compile for old sketch files"""
        context = DevContext(
            connected_devices=[
                DeviceInfo(port="/dev/ttyUSB0", description="Arduino Nano")
            ],
            recent_files=[
                FileChange(
                    path=Path("/test/sketch.ino"),
                    modified_time=datetime.now() - timedelta(hours=2),  # 2 hours old
                    size=1024
                )
            ]
        )
        
        should_suggest = self.context.should_suggest_compile(context)
        self.assertFalse(should_suggest)
    
    def test_should_suggest_upload(self):
        """Should suggest upload for very recent sketch changes"""
        context = DevContext(
            connected_devices=[
                DeviceInfo(port="/dev/ttyUSB0", description="ESP32")
            ],
            recent_files=[
                FileChange(
                    path=Path("/test/sketch.ino"),
                    modified_time=datetime.now() - timedelta(seconds=30),  # 30s ago
                    size=1024
                )
            ]
        )
        
        should_suggest = self.context.should_suggest_upload(context)
        self.assertTrue(should_suggest)


class TestContextManagerSingleton(unittest.TestCase):
    """Test singleton context manager"""
    
    def test_get_context_manager_singleton(self):
        """Should return same instance"""
        manager1 = get_context_manager()
        manager2 = get_context_manager()
        
        self.assertIs(manager1, manager2)
    
    def test_get_current_context(self):
        """get_current_context() should work as shortcut"""
        context = get_current_context()
        self.assertIsInstance(context, DevContext)


class TestIntegration(unittest.TestCase):
    """Integration tests for context awareness"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir) / "battery_guardian"
        self.project_path.mkdir()
        
        # Create realistic project structure
        firmware_dir = self.project_path / "firmware"
        firmware_dir.mkdir()
        
        sketch = firmware_dir / "battery_guardian.ino"
        sketch.write_text("""
        void setup() {
            Serial.begin(115200);
        }
        
        void loop() {
            // Monitor battery voltage
        }
        """)
        
        readme = self.project_path / "README.md"
        readme.write_text("# Battery Guardian\n\nSmart LiPo monitoring")
        
        self.context = ContextAwareness(
            project_paths=[self.project_path],
            indexable_extensions={'.ino', '.cpp', '.h', '.md', '.py'}
        )
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('serial.tools.list_ports.comports')
    def test_battery_guardian_workflow(self, mock_comports):
        """Simulate Battery Guardian development workflow"""
        # Mock ESP32 Nano connected
        mock_port = Mock()
        mock_port.device = "/dev/ttyACM0"
        mock_port.description = "Arduino Nano ESP32"
        mock_port.hwid = "USB VID:PID=2341:0070"
        mock_port.vid = 0x2341
        mock_port.pid = 0x0070
        mock_comports.return_value = [mock_port]
        
        # Touch the sketch file (simulate editing)
        sketch = self.project_path / "firmware" / "battery_guardian.ino"
        sketch.touch()
        
        # Get context
        context = self.context.get_context()
        
        # Should detect the board
        self.assertEqual(len(context.connected_devices), 1)
        self.assertTrue(context.connected_devices[0].is_arduino)
        
        # Should detect recent sketch modification
        sketch_files = [f for f in context.recent_files if f.path.suffix == '.ino']
        self.assertGreater(len(sketch_files), 0)
        
        # Should suggest compile
        should_compile = self.context.should_suggest_compile(context)
        self.assertTrue(should_compile)
        
        # Format for LLM should be informative
        formatted = self.context.format_context_for_llm(context)
        self.assertIn("esp32", formatted.lower())  # Case-insensitive check
        self.assertIn("battery_guardian", formatted.lower())


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDeviceInfo))
    suite.addTests(loader.loadTestsFromTestCase(TestFileChange))
    suite.addTests(loader.loadTestsFromTestCase(TestDevContext))
    suite.addTests(loader.loadTestsFromTestCase(TestContextAwareness))
    suite.addTests(loader.loadTestsFromTestCase(TestContextManagerSingleton))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")
    
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)