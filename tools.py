"""
Workshop Tools
Extensible tool framework for agent capabilities
"""

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from datetime import datetime
import json

from config import Config
from logger import get_logger

from context import get_context_manager, get_current_context

log = get_logger("tools")


def resolve_path_case_insensitive(path_str: str) -> Path:
    """
    Resolve a path with case-insensitive matching.
    Handles ~/arduino/Sketches -> ~/Arduino/sketches
    """
    # Expand ~ first
    if path_str.startswith('~'):
        expanded = str(Path.home()) + path_str[1:]
    else:
        expanded = path_str
    
    # If it exists as-is, return it
    p = Path(expanded)
    if p.exists():
        return p
    
    # Try to resolve case-insensitively
    parts = expanded.split('/')
    
    # Handle absolute vs relative paths
    if expanded.startswith('/'):
        resolved = Path('/')
        start_idx = 1
    else:
        resolved = Path('.')
        start_idx = 0
    
    for part in parts[start_idx:]:
        if not part:
            continue
        
        next_path = resolved / part
        if next_path.exists():
            resolved = next_path
        else:
            # Try case-insensitive match
            found = False
            if resolved.exists() and resolved.is_dir():
                try:
                    for item in resolved.iterdir():
                        if item.name.lower() == part.lower():
                            resolved = item
                            found = True
                            log.debug(f"Case-insensitive match: {part} -> {item.name}")
                            break
                except PermissionError:
                    pass
            
            if not found:
                # Can't resolve further, return best effort
                resolved = resolved / part
                break
    
    return resolved


def find_file(filename: str, search_paths: list = None) -> Optional[Path]:
    """
    Helper to find a file by name across common directories.
    Useful when user provides just a filename without full path.
    """
    config = Config()
    
    if search_paths is None:
        search_paths = [
            Path.home(),
            Path.home() / "projects",
            Path.home() / "Arduino",
            Path.home() / "Documents",
            Path.home() / "Downloads",
            Path.cwd(),
        ]
        search_paths.extend(config.PROJECT_PATHS)
    
    filename_lower = filename.strip().lower()
    
    for base in search_paths:
        if not base.exists():
            continue
        
        try:
            # Try exact match first (faster)
            for file in base.rglob("*"):
                if file.is_file() and file.name.lower() == filename_lower:
                    return file
        except PermissionError:
            continue
    
    return None


class ToolRegistry:
    """Registry for agent tools"""
    
    # Common argument aliases - LLMs often use different names
    ARG_ALIASES = {
        "directory": "path",
        "dir": "path",
        "folder": "path",
        "filepath": "path",
        "file_path": "path",
        "file": "path",
        "filename": "path",
        "location": "path",
        "text": "content",
        "message": "content",
        "body": "content",
        "data": "content",
        "search": "query",
        "term": "query",
        "q": "query",
        "expr": "expression",
        "math": "expression",
        "name": "title",
        "cmd": "command",
    }
    
    def __init__(self):
        self._tools: Dict[str, dict] = {}
        self.config = Config()
    
    def register(
        self,
        name: str,
        func: Callable,
        description: str,
        signature: str = "",
        examples: list = None
    ):
        """Register a new tool"""
        self._tools[name] = {
            "func": func,
            "description": description,
            "signature": signature,
            "examples": examples or []
        }
        log.debug(f"Registered tool: {name}")
    
    def get_all_tools(self) -> Dict[str, dict]:
        """Get all registered tools (without functions)"""
        return {
            name: {k: v for k, v in tool.items() if k != "func"}
            for name, tool in self._tools.items()
        }
    
    async def execute(self, name: str, args: dict) -> Any:
        """Execute a tool by name"""
        if name not in self._tools:
            log.error(f"Unknown tool: {name}")
            raise ValueError(f"Unknown tool: {name}")
        
        # Normalize argument names using aliases
        normalized_args = {}
        for key, value in args.items():
            canonical = self.ARG_ALIASES.get(key.lower(), key)
            normalized_args[canonical] = value
        
        log.info(f"Executing tool: {name}")
        log.debug(f"  Original args: {args}")
        if args != normalized_args:
            log.debug(f"  Normalized args: {normalized_args}")
        
        func = self._tools[name]["func"]
        
        try:
            # Handle both sync and async functions
            if asyncio.iscoroutinefunction(func):
                result = await func(**normalized_args)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: func(**normalized_args))
            
            log.debug(f"  Result: {str(result)[:200]}...")
            return result
        except TypeError as e:
            # Handle missing/extra arguments
            log.error(f"  Argument error: {e}")
            return f"Error: {e}"


def register_default_tools(registry: ToolRegistry, memory: "MemorySystem"):
    """Register the default set of tools"""
    
    config = Config()
    
    # === File Tools ===
    
    def read_file(path: str, max_lines: int = 500) -> str:
        """Read contents of a file"""
        log.info(f"read_file called with path: {path}")
        
        # Try case-insensitive resolution
        p = resolve_path_case_insensitive(path)
        
        # If still doesn't exist and looks like just a filename, search for it
        if not p.exists() and '/' not in path and '~' not in path:
            log.debug(f"File not found directly, searching for: {path}")
            found = find_file(path)
            if found:
                log.info(f"Found file at: {found}")
                p = found
            else:
                return f"Error: File not found: {path}\n\nTip: Try providing the full path, or use search_files to find it."
        
        if not p.exists():
            return f"Error: File not found: {path}"
        
        if not p.is_file():
            return f"Error: Not a file: {path}"
        
        if not config.is_path_allowed(p):
            return f"Error: Access to {path} is not allowed"
        
        try:
            content = p.read_text(errors='ignore')
            lines = content.split('\n')
            
            result = f"File: {p}\n"
            result += f"Size: {len(content):,} bytes, {len(lines)} lines\n"
            result += "-" * 40 + "\n"
            
            if len(lines) > max_lines:
                result += f"[Showing first {max_lines} of {len(lines)} lines]\n"
                result += '\n'.join(lines[:max_lines])
            else:
                result += content
            
            return result
        except Exception as e:
            return f"Error reading file: {e}"
    
    registry.register(
        name="read_file",
        func=read_file,
        description="Read the contents of a file. Will search for the file if path is just a filename.",
        signature="read_file(path: str, max_lines: int = 500)"
    )
    
    def write_file(path: str, content: str, append: bool = False) -> str:
        """Write content to a file"""
        p = Path(path).expanduser()
        
        if not config.is_path_allowed(p):
            return f"Error: Access to {path} is not allowed"
        
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            mode = 'a' if append else 'w'
            with open(p, mode) as f:
                f.write(content)
            return f"Successfully wrote {len(content)} bytes to {p}"
        except Exception as e:
            return f"Error writing file: {e}"
    
    registry.register(
        name="write_file",
        func=write_file,
        description="Write content to a file. Creates parent directories if needed.",
        signature="write_file(path: str, content: str, append: bool = False)"
    )
    
    def list_directory(path: str = ".", pattern: str = None, max_items: int = 100) -> str:
        """List files in a directory"""
        log.info(f"list_directory called with path: {path}")
        
        # Try case-insensitive resolution
        p = resolve_path_case_insensitive(path)
        log.debug(f"Resolved to: {p}")
        
        if not p.exists():
            return f"Error: Directory not found: {path}"
        
        if not p.is_dir():
            return f"Error: Not a directory: {path}"
        
        if not config.is_path_allowed(p):
            return f"Error: Access to {path} is not allowed"
        
        try:
            if pattern:
                items = list(p.glob(pattern))
            else:
                items = list(p.iterdir())
            
            # Sort: directories first, then by name
            items.sort(key=lambda x: (x.is_file(), x.name.lower()))
            
            result = [f"Directory: {p}", f"Total items: {len(items)}", "-" * 40]
            
            for item in items[:max_items]:
                if item.is_dir():
                    result.append(f"📁 {item.name}/")
                else:
                    try:
                        size = item.stat().st_size
                        if size > 1024*1024:
                            size_str = f"{size/1024/1024:.1f}MB"
                        elif size > 1024:
                            size_str = f"{size/1024:.1f}KB"
                        else:
                            size_str = f"{size}B"
                        result.append(f"📄 {item.name} ({size_str})")
                    except:
                        result.append(f"📄 {item.name}")
            
            if len(items) > max_items:
                result.append(f"... and {len(items) - max_items} more items")
            
            return "\n".join(result)
        except Exception as e:
            return f"Error listing directory: {e}"
    
    registry.register(
        name="list_directory",
        func=list_directory,
        description="List files and folders in a directory",
        signature="list_directory(path: str = '.', pattern: str = None)"
    )
    
    def search_files(query: str, path: str = "~", extensions: list = None, max_results: int = 20) -> str:
        """Search for text in files or find files by name"""
        log.info(f"search_files called: query='{query}', path='{path}'")
        
        # Try case-insensitive resolution
        p = resolve_path_case_insensitive(path)
        log.debug(f"Resolved to: {p}")
        
        if not p.exists():
            return f"Error: Path not found: {path}"
        
        if not config.is_path_allowed(p):
            return f"Error: Access to {path} is not allowed"
        
        if extensions is None:
            extensions = ['.py', '.md', '.txt', '.json', '.yaml', '.yml', '.ino', '.cpp', '.c', '.h']
        
        results = []
        files_searched = 0
        
        skip_dirs = {'node_modules', '__pycache__', '.git', 'venv', '.venv', 'build'}
        
        try:
            files = p.rglob("*") if p.is_dir() else [p]
            
            for file in files:
                if not file.is_file():
                    continue
                
                # Skip ignored directories
                if any(skip in file.parts for skip in skip_dirs):
                    continue
                
                # Check if filename matches query
                if query.lower() in file.name.lower():
                    results.append(f"📄 {file}")
                    if len(results) >= max_results:
                        break
                    continue
                
                # Search content only for certain extensions
                if file.suffix.lower() not in extensions:
                    continue
                
                files_searched += 1
                
                try:
                    content = file.read_text(errors='ignore')
                    if query.lower() in content.lower():
                        # Find matching lines
                        for i, line in enumerate(content.split('\n'), 1):
                            if query.lower() in line.lower():
                                results.append(f"{file}:{i}: {line.strip()[:80]}")
                                if len(results) >= max_results:
                                    break
                except:
                    continue
                
                if len(results) >= max_results:
                    break
            
            if results:
                header = f"Search results for '{query}' ({files_searched} files searched):\n"
                return header + "\n".join(results)
            return f"No matches found for '{query}' in {path} ({files_searched} files searched)"
            
        except Exception as e:
            return f"Error searching: {e}"
    
    registry.register(
        name="search_files",
        func=search_files,
        description="Search for text in files or find files by name",
        signature="search_files(query: str, path: str = '~', extensions: list = None)"
    )
    
    # === Shell Tools ===
    
    def run_shell(command: str, timeout: int = 30) -> str:
        """Run a shell command"""
        parts = command.split()
        if not parts:
            return "Error: Empty command"
        
        base_cmd = parts[0]
        
        if base_cmd not in config.ALLOWED_SHELL_COMMANDS:
            return f"Error: Command '{base_cmd}' not allowed.\nAllowed: {', '.join(sorted(config.ALLOWED_SHELL_COMMANDS))}"
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(Path.home())
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            
            return output.strip() or "(no output)"
            
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout}s"
        except Exception as e:
            return f"Error: {e}"
    
    registry.register(
        name="run_shell",
        func=run_shell,
        description="Run a shell command (limited to safe commands)",
        signature="run_shell(command: str, timeout: int = 30)"
    )
    
    # === Memory Tools ===
    
    def remember(content: str = None, category: str = "general", **kwargs) -> str:
        """Store something in long-term memory"""
        # Handle various arg formats from different LLMs
        # Qwen3 uses key/value, others use content
        if content is None:
            if 'value' in kwargs:
                content = kwargs['value']
                if 'key' in kwargs:
                    category = kwargs['key']
            elif 'text' in kwargs:
                content = kwargs['text']
            elif 'info' in kwargs:
                content = kwargs['info']
            else:
                return "Error: Nothing to remember. Provide content."
        
        # Clean up quoted content
        if isinstance(content, str):
            content = content.strip('"\'')
        
        memory.add_memory(content, category=category)
        memory.set_fact(f"memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}", content, category)
        log.info(f"Remembered ({category}): {content[:50]}...")
        return f"✓ Remembered: {content[:100]}..."
    
    registry.register(
        name="remember",
        func=remember,
        description="Store information in long-term memory. Use when user says 'remember X' or shares important info.",
        signature="remember(content: str, category: str = 'general')"
    )
    
    def recall(query: str, limit: int = 5) -> str:
        """Search long-term memory"""
        results = memory.search_memories(query, k=limit)
        if results:
            formatted = [f"Found {len(results)} relevant memories:", ""]
            for i, r in enumerate(results, 1):
                formatted.append(f"{i}. {r[:200]}...")
            return "\n".join(formatted)
        return f"No memories found for '{query}'"
    
    registry.register(
        name="recall",
        func=recall,
        description="Search long-term memory for relevant information",
        signature="recall(query: str, limit: int = 5)"
    )
    
    def take_note(title: str, content: str) -> str:
        """Create a timestamped note"""
        notes_dir = config.DATA_DIR / "notes"
        notes_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c if c.isalnum() or c in ' _-' else '_' for c in title)
        filename = f"{timestamp}_{safe_title[:50]}.md"
        
        note_content = f"""# {title}

_Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_

{content}
"""
        
        note_path = notes_dir / filename
        note_path.write_text(note_content)
        
        # Also remember it
        memory.add_memory(f"Note '{title}': {content}", category="notes")
        
        log.info(f"Note created: {note_path}")
        return f"✓ Note saved: {note_path}"
    
    registry.register(
        name="take_note",
        func=take_note,
        description="Create a timestamped note file",
        signature="take_note(title: str, content: str)"
    )
    
    def list_notes(limit: int = 10) -> str:
        """List recent notes"""
        notes_dir = config.DATA_DIR / "notes"
        if not notes_dir.exists():
            return "No notes yet"
        
        notes = sorted(notes_dir.glob("*.md"), reverse=True)[:limit]
        
        if not notes:
            return "No notes yet"
        
        result = [f"Recent notes ({len(notes)}):", ""]
        for note in notes:
            name = note.stem.split('_', 2)[-1].replace('_', ' ')
            result.append(f"• {name}")
            result.append(f"  {note}")
        
        return "\n".join(result)
    
    registry.register(
        name="list_notes",
        func=list_notes,
        description="List recent notes",
        signature="list_notes(limit: int = 10)"
    )
    
    # === Utility Tools ===
    
    def get_current_time() -> str:
        """Get current date and time"""
        now = datetime.now()
        return now.strftime("%A, %B %d, %Y at %I:%M %p")
    
    registry.register(
        name="get_current_time",
        func=get_current_time,
        description="Get the current date and time",
        signature="get_current_time()"
    )
    
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression"""
        import math
        
        allowed = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "len": len, "pow": pow,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "log10": math.log10,
            "pi": math.pi, "e": math.e,
        }
        
        try:
            safe_expr = expression.replace('^', '**')
            result = eval(safe_expr, {"__builtins__": {}}, allowed)
            return f"{expression} = {result}"
        except Exception as e:
            return f"Error calculating '{expression}': {e}"
    
    registry.register(
        name="calculate",
        func=calculate,
        description="Evaluate a mathematical expression",
        signature="calculate(expression: str)"
    )
    
    # === Web Search ===
    
    async def web_search(query: str, max_results: int = 5) -> str:
        """Search the web"""
        try:
            # Try new package name first
            try:
                from ddgs import DDGS
            except ImportError:
                # Fallback to old name (with deprecation warning)
                from duckduckgo_search import DDGS
            
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            
            if not results:
                return f"No results found for: {query}"
            
            formatted = [f"Web search results for '{query}':", ""]
            for r in results:
                title = r.get('title', 'No title')
                body = r.get('body', '')[:200]
                url = r.get('href', '')
                formatted.append(f"**{title}**")
                formatted.append(f"{body}")
                formatted.append(f"URL: {url}")
                formatted.append("")
            
            return "\n".join(formatted)
            
        except ImportError:
            return "Web search unavailable. Install: pip install ddgs"
        except Exception as e:
            log.error(f"Web search error: {e}")
            return f"Search error: {e}"
    
    registry.register(
        name="web_search",
        func=web_search,
        description="Search the web using DuckDuckGo",
        signature="web_search(query: str, max_results: int = 5)"
    )
    
    log.info(f"Registered {len(registry._tools)} tools")

    # === Context Awareness Tools ===
    from context import get_context_manager
    
    context_mgr = get_context_manager(
        project_paths=config.PROJECT_PATHS,
        indexable_extensions=config.INDEXABLE_EXTENSIONS
    )
    
    def get_dev_context() -> str:
        '''Get current development environment context'''
        context = context_mgr.get_context()
        
        # Add active project from memory
        active_project = memory.get_active_project()
        if active_project:
            context.active_project = active_project.get('name')
            context.project_path = active_project.get('path')
        
        return str(context)
    
    registry.register(
        name="get_dev_context",
        func=get_dev_context,
        description="Get current development environment context",
        signature="get_dev_context()"
    )
    
    def list_connected_devices() -> str:
        '''List all connected USB devices'''
        context = context_mgr.get_context()
        
        if not context.connected_devices:
            return "No USB devices detected."
        
        arduino_boards = [d for d in context.connected_devices if d.is_arduino]
        
        lines = []
        if arduino_boards:
            lines.append("Arduino/ESP32 boards:")
            for dev in arduino_boards:
                lines.append(f"  • {dev.description} ({dev.port})")
        
        return "\\n".join(lines) if lines else "No Arduino boards detected."
    
    registry.register(
        name="list_connected_devices",
        func=list_connected_devices,
        description="List connected USB devices, especially Arduino/ESP32",
        signature="list_connected_devices()"
    )
    
    log.info("Registered context awareness tools")
    
    return context_mgr  # Return for use in agent
