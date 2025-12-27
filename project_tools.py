"""
Workshop Project & Arduino Tools
Tools for project context management and Arduino development
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from logger import get_logger

log = get_logger("project_tools")


def register_project_tools(registry, memory, construct_manager=None):
    """Register project and Arduino tools"""
    
    # === Project Context Tools ===
    
    async def work_on(project: str, path: str = None, description: str = None) -> str:
        """Set the active project for this session"""
        result = memory.set_active_project(project, path, description)
        
        # Notify UI of project change
        if construct_manager:
            await construct_manager.broadcast_project_changed(project)
        
        # Get project context to show
        context = memory.get_project_context()
        last_session = memory.get_last_session()
        
        response = [result]
        if last_session and last_session.get('summary'):
            response.append(f"\nLast session: {last_session['summary']}")
        
        return "\n".join(response)
    
    registry.register(
        name="work_on",
        func=work_on,
        description="Set the active project. Use when user says 'working on X' or 'switch to project X'",
        signature="work_on(project: str, path: str = None, description: str = None)"
    )
    
    def project_note(content: str, category: str = "general") -> str:
        """Add a quick note to the active project"""
        return memory.add_project_note(content, category)
    
    registry.register(
        name="project_note",
        func=project_note,
        description="Add a note to the current project. Faster than take_note for project-specific things.",
        signature="project_note(content: str, category: str = 'general')"
    )
    
    def what_project(**kwargs) -> str:
        """Get info about the current active project"""
        # Ignore any args LLM might pass - we only care about the active project
        project = memory.get_active_project()
        if not project:
            projects = memory.list_projects()
            if projects:
                project_names = [p['name'] for p in projects[:5]]
                return f"No active project. Recent projects: {', '.join(project_names)}"
            return "No active project set. Say 'working on [project name]' to set one."
        
        return memory.get_project_context()
    
    registry.register(
        name="what_project",
        func=what_project,
        description="Get information about the current active project",
        signature="what_project()"
    )
    
    def list_projects() -> str:
        """List all known projects"""
        projects = memory.list_projects()
        if not projects:
            return "No projects yet. Start one with 'working on [project name]'"
        
        lines = ["Known projects:"]
        for p in projects:
            line = f"  • {p['name']}"
            if p.get('path'):
                line += f" ({p['path']})"
            lines.append(line)
        
        return "\n".join(lines)
    
    registry.register(
        name="list_projects",
        func=list_projects,
        description="List all known projects",
        signature="list_projects()"
    )
    
    def session_summary(summary: str) -> str:
        """Save a summary of what was done this session"""
        return memory.save_session_summary(summary)
    
    registry.register(
        name="session_summary",
        func=session_summary,
        description="Save a summary of what was accomplished this session",
        signature="session_summary(summary: str)"
    )
    
    def last_session(project: str = None) -> str:
        """Get info about the last session on a project"""
        session = memory.get_last_session(project)
        if not session:
            return "No previous session found."
        
        result = f"Last session: {session.get('started_at', 'unknown')}"
        if session.get('summary'):
            result += f"\nSummary: {session['summary']}"
        return result
    
    registry.register(
        name="last_session",
        func=last_session,
        description="Get info about the last work session on a project",
        signature="last_session(project: str = None)"
    )
    
    # === Arduino CLI Tools ===
    
    def _find_arduino_cli() -> Optional[str]:
        """Find arduino-cli executable"""
        # Check common locations
        locations = [
            shutil.which("arduino-cli"),
            "/usr/local/bin/arduino-cli",
            str(Path.home() / ".local" / "bin" / "arduino-cli"),
            str(Path.home() / "bin" / "arduino-cli"),
        ]
        
        for loc in locations:
            if loc and Path(loc).exists():
                return loc
        
        return None
    
    def _find_sketch(name: str) -> Optional[Path]:
        """Find an Arduino sketch by name"""
        from config import Config
        config = Config()
        
        # Common sketch locations
        search_paths = [
            Path.home() / "Arduino" / "sketches",
            Path.home() / "Arduino",
            Path.home() / "Documents" / "Arduino",
        ]
        search_paths.extend(config.PROJECT_PATHS)
        
        name_lower = name.lower().replace('.ino', '')
        
        for base in search_paths:
            if not base.exists():
                continue
            
            # Look for sketch folder or .ino file
            for item in base.rglob("*"):
                if item.is_dir() and item.name.lower() == name_lower:
                    ino_file = item / f"{item.name}.ino"
                    if ino_file.exists():
                        return ino_file
                elif item.is_file() and item.suffix == '.ino':
                    if item.stem.lower() == name_lower:
                        return item
        
        return None
    
    async def arduino_compile(sketch: str = None, board: str = "arduino:esp32:nano_nora", show_output: bool = True) -> str:
        """Compile an Arduino sketch with streaming output"""
        import asyncio
        
        cli = _find_arduino_cli()
        if not cli:
            return "Error: arduino-cli not found. Install it from https://arduino.github.io/arduino-cli/"
        
        # Use active project if no sketch specified
        if not sketch:
            project = memory.get_active_project()
            if project:
                sketch = project['name'].replace(' ', '_')
                log.info(f"Using active project: {sketch}")
            else:
                return "Error: No sketch specified and no active project. Say 'working on [project]' first, or specify a sketch."
        
        # Find the sketch
        sketch_path = _find_sketch(sketch)
        if not sketch_path:
            return f"Error: Sketch '{sketch}' not found. Try providing the full path."
        
        log.info(f"Compiling {sketch_path} for {board}")
        
        cmd = [cli, "compile", "--fqbn", board, str(sketch_path.parent)]
        
        # Create terminal construct first if available
        term_id = None
        if construct_manager and show_output:
            term_id = await construct_manager.create_terminal(
                content=f"$ {' '.join(cmd)}\n\n",
                title=f"Compiling: {sketch_path.name}..."
            )
        
        try:
            # Use asyncio subprocess for non-blocking streaming
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            
            output_lines = []
            
            # Stream output line by line
            async def read_stream():
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    text = line.decode('utf-8', errors='ignore')
                    output_lines.append(text)
                    if term_id:
                        await construct_manager.append_content(term_id, text)
            
            # Wait for output with timeout
            try:
                await asyncio.wait_for(read_stream(), timeout=120)
            except asyncio.TimeoutError:
                process.kill()
                if term_id:
                    await construct_manager.append_content(term_id, "\n\n⚠️ Compilation timed out!")
                    await construct_manager.update_title(term_id, f"Compile: {sketch_path.name} ⚠️")
                return "Error: Compilation timed out (120s limit)"
            
            await process.wait()
            success = process.returncode == 0
            output = ''.join(output_lines)
            
            # Update terminal title with result
            if term_id:
                if success:
                    await construct_manager.update_title(term_id, f"Compile: {sketch_path.name} ✓")
                else:
                    await construct_manager.update_title(term_id, f"Compile: {sketch_path.name} ✗")
            
            if success:
                # Extract binary size info
                size_info = [l for l in output_lines if 'bytes' in l.lower()]
                if size_info:
                    return f"✓ Compiled successfully\n" + "".join(size_info[-3:])
                return "✓ Compiled successfully"
            else:
                # Find error lines
                error_lines = [l for l in output_lines if 'error:' in l.lower()]
                if error_lines:
                    return f"✗ Compilation failed:\n" + "".join(error_lines[:5])
                return f"✗ Compilation failed:\n{output[:500]}"
            
        except Exception as e:
            log.error(f"Arduino compile error: {e}")
            if term_id:
                await construct_manager.append_content(term_id, f"\n\n❌ Error: {e}")
                await construct_manager.update_title(term_id, f"Compile: {sketch_path.name} ✗")
            return f"Error: {e}"
    
    registry.register(
        name="arduino_compile",
        func=arduino_compile,
        description="Compile an Arduino sketch. Finds sketch by name automatically.",
        signature="arduino_compile(sketch: str, board: str = 'esp32:esp32:esp32', show_output: bool = True)"
    )
    
    async def arduino_upload(sketch: str = None, port: str = None, board: str = "arduino:esp32:nano_nora") -> str:
        """Upload an Arduino sketch to a connected board with streaming output"""
        import asyncio
        
        cli = _find_arduino_cli()
        if not cli:
            return "Error: arduino-cli not found."
        
        # Use active project if no sketch specified
        if not sketch:
            project = memory.get_active_project()
            if project:
                sketch = project['name'].replace(' ', '_')
                log.info(f"Using active project: {sketch}")
            else:
                return "Error: No sketch specified and no active project. Say 'working on [project]' first, or specify a sketch."
        
        sketch_path = _find_sketch(sketch)
        if not sketch_path:
            return f"Error: Sketch '{sketch}' not found."
        
        # Auto-detect port if not specified
        if not port:
            # Try common ports (ACM first for Nano ESP32)
            common_ports = ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB0", "/dev/ttyUSB1"]
            for p in common_ports:
                if Path(p).exists():
                    port = p
                    break
            
            if not port:
                return "Error: No port specified and couldn't auto-detect. Use port='/dev/ttyACM0' or similar."
        
        log.info(f"Uploading {sketch_path} to {port}")
        
        cmd = [cli, "upload", "-p", port, "--fqbn", board, str(sketch_path.parent)]
        
        # Create terminal construct first
        term_id = None
        if construct_manager:
            term_id = await construct_manager.create_terminal(
                content=f"$ {' '.join(cmd)}\n\n",
                title=f"Uploading: {sketch_path.name}..."
            )
        
        try:
            # Use asyncio subprocess for non-blocking streaming
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            
            output_lines = []
            
            # Stream output line by line
            async def read_stream():
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    text = line.decode('utf-8', errors='ignore')
                    output_lines.append(text)
                    if term_id:
                        await construct_manager.append_content(term_id, text)
            
            # Wait for output with timeout
            try:
                await asyncio.wait_for(read_stream(), timeout=120)
            except asyncio.TimeoutError:
                process.kill()
                if term_id:
                    await construct_manager.append_content(term_id, "\n\n⚠️ Upload timed out!")
                    await construct_manager.update_title(term_id, f"Upload: {sketch_path.name} ⚠️")
                return "Error: Upload timed out (120s limit)"
            
            await process.wait()
            success = process.returncode == 0
            
            # Update terminal title with result
            if term_id:
                if success:
                    await construct_manager.update_title(term_id, f"Upload: {sketch_path.name} ✓")
                else:
                    await construct_manager.update_title(term_id, f"Upload: {sketch_path.name} ✗")
            
            if success:
                return f"✓ Uploaded to {port}"
            else:
                output = ''.join(output_lines)
                return f"✗ Upload failed:\n{output[:500]}"
            
        except Exception as e:
            log.error(f"Arduino upload error: {e}")
            if term_id:
                await construct_manager.append_content(term_id, f"\n\n❌ Error: {e}")
                await construct_manager.update_title(term_id, f"Upload: {sketch_path.name} ✗")
            return f"Error: {e}"
    
    registry.register(
        name="arduino_upload",
        func=arduino_upload,
        description="Upload a compiled sketch to an Arduino/ESP32 board",
        signature="arduino_upload(sketch: str, port: str = None, board: str = 'esp32:esp32:esp32')"
    )
    
    async def arduino_monitor(port: str = None, baud: int = 115200, duration: int = 10) -> str:
        """Open serial monitor for a brief capture"""
        cli = _find_arduino_cli()
        if not cli:
            return "Error: arduino-cli not found."
        
        # Auto-detect port if not specified
        if not port:
            common_ports = ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB0", "/dev/ttyUSB1"]
            for p in common_ports:
                if Path(p).exists():
                    port = p
                    break
            
            if not port:
                return "Error: No port found. Specify port or connect a board."
        
        if not Path(port).exists():
            return f"Error: Port {port} not found."
        
        log.info(f"Starting serial monitor on {port} at {baud} baud for {duration}s")
        
        cmd = [cli, "monitor", "-p", port, "-c", f"baudrate={baud}"]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=duration
            )
            output = result.stdout + result.stderr
            
        except subprocess.TimeoutExpired as e:
            # This is expected - we capture for duration then stop
            output = e.stdout.decode() if e.stdout else ""
            output += e.stderr.decode() if e.stderr else ""
        except Exception as e:
            return f"Error: {e}"
        
        if construct_manager:
            await construct_manager.create_terminal(
                content=output,
                title=f"Serial: {port}"
            )
        
        return f"Captured {len(output)} bytes from {port}"
    
    registry.register(
        name="arduino_monitor",
        func=arduino_monitor,
        description="Capture serial output from a connected board for a specified duration",
        signature="arduino_monitor(port: str = '/dev/ttyUSB0', baud: int = 115200, duration: int = 10)"
    )
    
    def arduino_boards() -> str:
        """List connected Arduino/ESP32 boards"""
        cli = _find_arduino_cli()
        if not cli:
            return "Error: arduino-cli not found."
        
        try:
            result = subprocess.run(
                [cli, "board", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return result.stdout or "No boards found."
            return f"Error: {result.stderr}"
            
        except Exception as e:
            return f"Error: {e}"
    
    registry.register(
        name="arduino_boards",
        func=arduino_boards,
        description="List connected Arduino/ESP32 boards",
        signature="arduino_boards()"
    )
    
    log.info(f"Registered project tools (with Arduino CLI)")
