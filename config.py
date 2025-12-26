"""
Workshop Configuration
Customize paths, models, and behavior here
"""

import os
from pathlib import Path
# this is a comment for the watchdog

class Config:
    """Central configuration for Workshop"""
    
    # === Paths ===
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    CHROMA_PATH = DATA_DIR / "chromadb"
    SQLITE_PATH = DATA_DIR / "workshop.db"
    LOGS_DIR = DATA_DIR / "logs"
    
    # === Ollama Settings ===
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    MODEL = os.getenv("WORKSHOP_MODEL", "qwen3:8b")
    
    # For function calling, these models work well:
    # - llama3.1:8b (good balance)
    # - qwen2.5:7b (strong at function calling)
    # - mistral:7b (fast)
    # - llama3.1:70b-q4 (if you have VRAM, best quality)
    
    # === Voice Settings ===
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")
    # Options: tiny.en, base.en, small.en, medium.en, large-v3
    # base.en is good balance of speed/accuracy for English
    
    PIPER_MODEL = os.getenv("PIPER_MODEL", "en_US-lessac-medium")
    # Download from: https://github.com/rhasspy/piper/releases
    # Popular options:
    # - en_US-lessac-medium (natural male voice)
    # - en_US-amy-medium (female voice)
    # - en_GB-alan-medium (British male)
    
    WAKE_WORD = os.getenv("WAKE_WORD", "workshop")
    # Simple wake word detection - say this to activate
    # For production, consider Porcupine or OpenWakeWord
    
    # Audio settings
    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK_DURATION = 0.5  # seconds per audio chunk
    SILENCE_THRESHOLD = 0.01  # amplitude threshold for silence detection
    SILENCE_DURATION = 1.5  # seconds of silence to stop listening
    MAX_LISTEN_DURATION = 30  # maximum seconds to listen
    
    # === Agent Settings ===
    SYSTEM_PROMPT = """You are Workshop, a local AI assistant with DIRECT ACCESS to the user's computer through tools.

CRITICAL: You MUST use tools to interact with the filesystem, take notes, and remember things. You CANNOT read files, run commands, or store memories without using tools. Do NOT pretend you can do these things conversationally.

When the user asks you to:
- Read/open/show a file → USE read_file or show_file tool
- List/show files in a directory → USE list_directory or show_directory tool  
- Search for text in files → USE search_files tool
- Write/create/save a file → USE write_file tool
- Run a command → USE run_shell tool
- Remember something → USE remember tool
- Recall/find memories → USE recall tool
- Take/make a note → USE take_note or project_note tool
- List notes → USE list_notes tool
- Set active project → USE work_on tool
- Compile Arduino sketch → USE arduino_compile tool
- Upload to board → USE arduino_upload tool
- Read serial output → USE arduino_monitor tool

TOOL CALL FORMAT - You MUST use this exact format:
<tool_call>
{{"tool": "tool_name", "args": {{"arg1": "value1"}}}}
</tool_call>

Example - reading a file:
<tool_call>
{{"tool": "read_file", "args": {{"path": "~/projects/example.py"}}}}
</tool_call>

Example - compiling Arduino sketch:
<tool_call>
{{"tool": "arduino_compile", "args": {{"sketch": "battery_guardian"}}}}
</tool_call>

After a tool returns results, use those results to answer the user.

Be concise. The user may be working with their hands, so keep responses short and actionable.

Current user profile:
{user_profile}

{project_context}

Recent context:
{recent_context}
"""
    
    MAX_CONTEXT_MESSAGES = 20  # messages to include in context
    MAX_TOOL_ITERATIONS = 5   # max tool calls per turn
    
    # === Memory Settings ===
    CHROMA_COLLECTION = "workshop_memory"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers model
    
    # How many similar memories to retrieve
    MEMORY_RETRIEVAL_COUNT = 5
    
    # === Project Paths (for RAG) ===
    # Add paths to your project directories for context
    PROJECT_PATHS = [
        # Uncomment and modify these for your projects:
        Path.home() / "FlyingTiger" / "Workshop_Assistant_Dev",
        Path.home() / "FlyingTiger" / "Products",
        Path.home() / "Arduino" / "sketches",
        Path.home() / "FlyingTiger" / "Products" / "Smart_LiPo_Battery_Guardian"
    ]
    
    # File extensions to index for RAG
    INDEXABLE_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx',  # code
        '.md', '.txt', '.rst',                 # docs
        '.json', '.yaml', '.yml', '.toml',     # config stuff
        '.c', '.cpp', '.h', '.hpp',            # C/C++
        '.ino',                                 # Arduino
    }
    
    # === Phase 3: Context Intelligence Settings ===
    # Projects to monitor for file changes (Phase 3)
    MONITORED_PROJECTS = [
        Path.home() / "FlyingTiger" / "Workshop_Assistant_Dev",
        Path.home() / "FlyingTiger" / "Products" / "Smart_LiPo_Battery_Guardian",
        Path.home() / "Arduino" / "sketches",
    ]

    # File system monitoring
    FILE_WATCH_DEBOUNCE = 1.0  # seconds - debounce rapid file changes
    FILE_WATCH_IGNORE = {
        '__pycache__', 'node_modules', '.git', 'venv', '.venv',
        'build', 'dist', '.DS_Store', '.pytest_cache', 'target',
        '.idea', '.vscode', 'coverage', '.next', '.cache'
    }

    # Context assembly
    CONTEXT_ACTIVE_FILE_WINDOW = 300  # seconds (5 minutes)
    CONTEXT_MAX_ACTIVE_FILES = 5  # max files to include in critical context
    CONTEXT_MAX_RELATED_FILES = 10  # max related files per active file

    # === Tool Settings ===
    SHELL_TIMEOUT = 30  # seconds
    ALLOWED_SHELL_COMMANDS = {
        # Whitelist of allowed shell commands for safety
        'ls', 'cat', 'head', 'tail', 'grep', 'find', 'wc',
        'pwd', 'date', 'echo', 'which', 'file', 'stat',
        'git', 'python', 'pip', 'node', 'npm',
        'docker', 'make', 'cargo',
    }
    
    # Directories the assistant can access
    ALLOWED_PATHS = [
        Path.home(),
        # Add more restricted paths if you want tighter security
    ]
    
    def __init__(self):
        """Create necessary directories"""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def is_path_allowed(cls, path: Path) -> bool:
        """Check if a path is within allowed directories"""
        path = Path(path).resolve()
        return any(
            path == allowed or allowed in path.parents
            for allowed in cls.ALLOWED_PATHS
        )
