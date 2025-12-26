"""
Workshop Context Awareness
Tracks development environment state for proactive assistance
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import platform

from logger import get_logger

log = get_logger("context")


@dataclass
class DeviceInfo:
    """Information about a connected USB device"""
    port: str
    description: str
    hwid: str = ""
    vid: str = ""
    pid: str = ""
    
    @property
    def is_arduino(self) -> bool:
        """Check if this looks like an Arduino/ESP32 board"""
        desc_lower = self.description.lower()
        return any(x in desc_lower for x in [
            'arduino', 'esp32', 'esp8266', 'ch340', 'cp210', 
            'ftdi', 'usb serial', 'nano'
        ])
    
    def __str__(self):
        return f"{self.description} ({self.port})"


@dataclass
class FileChange:
    """Information about a recently modified file"""
    path: Path
    modified_time: datetime
    size: int
    
    @property
    def age_seconds(self) -> float:
        """How long ago was this file modified"""
        return (datetime.now() - self.modified_time).total_seconds()
    
    def __str__(self):
        return f"{self.path.name} (modified {int(self.age_seconds)}s ago)"


@dataclass
class DevContext:
    """Current development context"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Environment
    active_window: str = ""
    active_app: str = ""
    current_directory: str = ""
    
    # Hardware
    connected_devices: List[DeviceInfo] = field(default_factory=list)
    
    # Files
    recent_files: List[FileChange] = field(default_factory=list)
    
    # Project
    active_project: Optional[str] = None
    project_path: Optional[str] = None
    
    def __str__(self):
        lines = ["Development Context:"]
        
        if self.active_window:
            lines.append(f"  Window: {self.active_window}")
        
        if self.connected_devices:
            lines.append(f"  Devices:")
            for dev in self.connected_devices:
                lines.append(f"    - {dev}")
        
        if self.recent_files:
            lines.append(f"  Recent files:")
            for file in self.recent_files[:5]:
                lines.append(f"    - {file}")
        
        if self.active_project:
            lines.append(f"  Project: {self.active_project}")
        
        return "\n".join(lines)


class ContextAwareness:
    """
    Tracks development environment state.
    
    Monitors:
    - Active window/application
    - Connected USB devices (Arduino, ESP32)
    - Recently modified files in project paths
    - Current working directory
    """
    
    def __init__(self, project_paths: List[Path] = None, indexable_extensions: set = None):
        self.project_paths = project_paths or []
        self.indexable_extensions = indexable_extensions or {
            '.py', '.ino', '.cpp', '.h', '.js', '.md', '.txt'
        }
        self._last_context: Optional[DevContext] = None
        self._system = platform.system()
        
        log.info(f"Context awareness initialized for {self._system}")
    
    def get_context(self) -> DevContext:
        """Get current development context"""
        context = DevContext()
        
        # Gather all context
        context.active_window, context.active_app = self._get_active_window()
        context.current_directory = os.getcwd()
        context.connected_devices = self._get_connected_devices()
        context.recent_files = self._get_recent_files(minutes=60)
        
        self._last_context = context
        log.debug(f"Context captured: {len(context.connected_devices)} devices, "
                 f"{len(context.recent_files)} recent files")
        
        return context
    
    def _get_active_window(self) -> tuple[str, str]:
        """Get the active window title and application name"""
        try:
            if self._system == "Darwin":  # macOS
                # Get app name
                app_script = '''
                tell application "System Events"
                    set frontApp to name of first application process whose frontmost is true
                    return frontApp
                end tell
                '''
                result = subprocess.run(
                    ['osascript', '-e', app_script],
                    capture_output=True,
                    text=True,
                    timeout=1
                )
                app_name = result.stdout.strip()
                
                # Get window title
                window_script = f'''
                tell application "System Events"
                    tell process "{app_name}"
                        try
                            set windowTitle to name of front window
                            return windowTitle
                        end try
                    end tell
                end tell
                '''
                result = subprocess.run(
                    ['osascript', '-e', window_script],
                    capture_output=True,
                    text=True,
                    timeout=1
                )
                window_title = result.stdout.strip()
                
                return window_title, app_name
            
            elif self._system == "Linux":
                # Try xdotool first
                try:
                    # Get window ID
                    window_id = subprocess.run(
                        ['xdotool', 'getactivewindow'],
                        capture_output=True,
                        text=True,
                        timeout=1
                    ).stdout.strip()
                    
                    # Get window name
                    window_name = subprocess.run(
                        ['xdotool', 'getwindowname', window_id],
                        capture_output=True,
                        text=True,
                        timeout=1
                    ).stdout.strip()
                    
                    # Get app name from window class
                    window_class = subprocess.run(
                        ['xdotool', 'getwindowclassname', window_id],
                        capture_output=True,
                        text=True,
                        timeout=1
                    ).stdout.strip()
                    
                    return window_name, window_class
                
                except FileNotFoundError:
                    # xdotool not installed, try wmctrl
                    result = subprocess.run(
                        ['wmctrl', '-l', '-p'],
                        capture_output=True,
                        text=True,
                        timeout=1
                    )
                    # Parse wmctrl output for active window
                    # This is best-effort
                    pass
        
        except Exception as e:
            log.debug(f"Could not get active window: {e}")
        
        return "", ""
    
    def _get_connected_devices(self) -> List[DeviceInfo]:
        """Get list of connected USB devices (serial ports)"""
        devices = []
        
        try:
            import serial.tools.list_ports
            
            ports = serial.tools.list_ports.comports()
            for port in ports:
                device = DeviceInfo(
                    port=port.device,
                    description=port.description,
                    hwid=port.hwid or "",
                    vid=hex(port.vid) if port.vid else "",
                    pid=hex(port.pid) if port.pid else ""
                )
                devices.append(device)
                
                if device.is_arduino:
                    log.debug(f"Arduino/ESP32 detected: {device}")
        
        except ImportError:
            log.warning("pyserial not installed - cannot detect USB devices")
        except Exception as e:
            log.error(f"Error detecting devices: {e}")
        
        return devices
    
    def _get_recent_files(self, minutes: int = 60) -> List[FileChange]:
        """Get files modified in the last N minutes"""
        import time
        
        recent = []
        cutoff = time.time() - (minutes * 60)
        
        for project_path in self.project_paths:
            if not project_path.exists():
                continue
            
            try:
                # Walk project directory
                for file_path in project_path.rglob("*"):
                    # Skip directories and hidden files
                    if not file_path.is_file() or file_path.name.startswith('.'):
                        continue
                    
                    # Skip non-indexable extensions
                    if file_path.suffix not in self.indexable_extensions:
                        continue
                    
                    # Check modification time
                    mtime = file_path.stat().st_mtime
                    if mtime > cutoff:
                        file_change = FileChange(
                            path=file_path,
                            modified_time=datetime.fromtimestamp(mtime),
                            size=file_path.stat().st_size
                        )
                        recent.append(file_change)
            
            except Exception as e:
                log.debug(f"Error scanning {project_path}: {e}")
        
        # Sort by modification time (newest first)
        recent.sort(key=lambda f: f.modified_time, reverse=True)
        
        return recent
    
    def detect_arduino_boards(self) -> List[DeviceInfo]:
        """Get only Arduino/ESP32 boards from connected devices"""
        return [d for d in self._get_connected_devices() if d.is_arduino]
    
    def get_project_for_file(self, file_path: Path) -> Optional[str]:
        """Determine which project a file belongs to"""
        file_path = file_path.resolve()
        
        for project_path in self.project_paths:
            try:
                if project_path in file_path.parents or file_path == project_path:
                    return project_path.name
            except:
                pass
        
        return None
    
    def format_context_for_llm(self, context: DevContext = None) -> str:
        """Format context as a string suitable for LLM injection"""
        if context is None:
            context = self.get_context()
        
        parts = []
        
        # Active window gives us a hint about what user is doing
        if context.active_window:
            # Extract useful info from window title
            window_lower = context.active_window.lower()
            
            # Detect if editing specific files
            if any(ext in window_lower for ext in ['.py', '.ino', '.cpp', '.js', '.md']):
                parts.append(f"User is editing: {context.active_window}")
            elif context.active_app:
                parts.append(f"Active app: {context.active_app}")
        
        # Connected hardware
        arduino_boards = [d for d in context.connected_devices if d.is_arduino]
        if arduino_boards:
            board_list = ", ".join(str(b) for b in arduino_boards)
            parts.append(f"Connected boards: {board_list}")
        
        # Recent file activity
        if context.recent_files:
            # Group by project
            by_project = {}
            for file in context.recent_files[:10]:  # Top 10
                project = self.get_project_for_file(file.path)
                if project:
                    if project not in by_project:
                        by_project[project] = []
                    by_project[project].append(file)
            
            if by_project:
                parts.append("Recent activity:")
                for project, files in by_project.items():
                    file_names = ", ".join(f.path.name for f in files[:3])
                    parts.append(f"  • {project}: {file_names}")
        
        return "\n".join(parts) if parts else ""
    
    def should_suggest_compile(self, context: DevContext = None) -> bool:
        """Check if we should suggest compiling based on context"""
        if context is None:
            context = self.get_context()
        
        # Has Arduino board connected?
        has_board = any(d.is_arduino for d in context.connected_devices)
        
        # Recently modified .ino files?
        has_recent_sketch = any(
            f.path.suffix == '.ino' and f.age_seconds < 300  # 5 minutes
            for f in context.recent_files
        )
        
        return has_board and has_recent_sketch
    
    def should_suggest_upload(self, context: DevContext = None) -> bool:
        """Check if we should suggest uploading firmware"""
        # Same as compile, but only if very recent changes (< 1 min)
        if context is None:
            context = self.get_context()
        
        has_board = any(d.is_arduino for d in context.connected_devices)
        
        very_recent_sketch = any(
            f.path.suffix == '.ino' and f.age_seconds < 60
            for f in context.recent_files
        )
        
        return has_board and very_recent_sketch


# === Singleton instance ===
_context_manager: Optional[ContextAwareness] = None


def get_context_manager(project_paths: List[Path] = None, 
                       indexable_extensions: set = None) -> ContextAwareness:
    """Get or create the global context manager"""
    global _context_manager
    
    if _context_manager is None:
        _context_manager = ContextAwareness(project_paths, indexable_extensions)
    
    return _context_manager


def get_current_context() -> DevContext:
    """Quick access to current context"""
    manager = get_context_manager()
    return manager.get_context()