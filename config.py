"""
Workshop Configuration
Customize paths, models, and behavior here
"""

import os
from pathlib import Path

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
    # MODEL = os.getenv("WORKSHOP_MODEL", "phi4:14b")
    MODEL = os.getenv("WORKSHOP_MODEL", "phi4:14b")


    # Model selection:
    # - phi4:14b (RECOMMENDED - best reasoning, good tool use, ~8GB VRAM)
    # - qwen2.5:7b (fast, good tool use, ~4.5GB VRAM)
    # - llama3.1:8b (good balance of speed/accuracy)
    # - llama3-groq-tool-use:8b (fast routing only, terrible at reasoning)
    #
    # NOTE: Model MUST support Ollama's native tool calling API.
    # Check for "tools" badge on ollama.com/library
    
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
    SILENCE_DURATION = 3.0  # seconds of silence to stop listening
    MAX_LISTEN_DURATION = 90  # maximum seconds to listen
    
    # === Routing Settings (Phase 5: Fabric-Style Two-Stage Routing) ===
    #
    # Two-stage routing inspired by Daniel Miessler's Fabric:
    # Stage 1: Router prompt (phi3:mini) classifies intent → skill name
    # Stage 2: Skill executor (phi4:14b) runs with NON-NATIVE tool calling
    #
    # Uses <tool_call> format instead of native tools, allowing phi4's superior
    # reasoning to be used for both argument extraction and response synthesis.
    # This fixes the "bad query extraction" problem where regex captured garbage,
    # AND avoids using models with native tool support but poor reasoning.
    #
    # To use the OLD routing system, set: Agent(use_legacy_routing=True)

    # Router model for Stage 1 intent classification
    ROUTER_MODEL = os.getenv("WORKSHOP_ROUTER_MODEL", "phi3:mini")  # ~2GB VRAM
    ROUTER_MAX_TOKENS = 20  # Router outputs only skill name
    ROUTER_TEMPERATURE = 0.1  # Low temperature for deterministic classification

    # Skill execution model for Stage 2 (NON-NATIVE tool calling)
    # Uses phi4 for excellent reasoning - outputs <tool_call> format which we parse
    # Does NOT need to support Ollama's native tool calling API
    SKILL_EXECUTION_MODEL = os.getenv("WORKSHOP_SKILL_MODEL", "phi4:14b")

    # Semantic pre-filtering thresholds (optional optimization)
    # When embedding similarity is very high, we can skip the router prompt
    SEMANTIC_ROUTING_ENABLED = True  # Enable semantic pre-filtering
    SEMANTIC_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # CPU (~1.2GB RAM)
    SEMANTIC_EMBEDDINGS_CACHE = DATA_DIR / "skill_embeddings.json"
    SEMANTIC_BYPASS_THRESHOLD = 0.85  # Skip router prompt, go direct to skill
    SEMANTIC_CONFIRM_THRESHOLD = 0.45  # Router confirms from top candidates

    # Legacy settings (only used if use_legacy_routing=True)
    SEMANTIC_HIGH_CONFIDENCE = 0.85  # Legacy: Direct routing threshold
    SEMANTIC_MEDIUM_CONFIDENCE = 0.45  # Legacy: LLM confirmation threshold
    ROUTING_PRIORITIES = {
        "workflow_triggers": 0,   # Legacy: Highest priority
        "skill_patterns": 1,      # Legacy: Pattern-based
        "semantic_routing": 2,    # Legacy: Embedding similarity
        "hardcoded_patterns": 3,  # Legacy: Tool intent patterns
        "llm_decision": 10,       # Legacy: Fallback
    }

    # === VRAM Management (12GB constraint) ===
    VRAM_BUDGET_GB = 12
    VRAM_HEADROOM_GB = 2  # Reserve for KV cache and operations

    # Model VRAM estimates at Q4_K_M quantization
    MODEL_VRAM_ESTIMATES = {
        "phi3:mini": 2.0,           # Router model (semantic confirmation)
        "phi4:14b": 8.0,            # Primary agent - best reasoning
        "qwen2.5:7b": 4.5,
        "qwen2.5-coder:7b": 4.5,
        "mistral:7b": 4.5,
        "llama3.1:8b": 5.5,
        "llama3-groq-tool-use:8b": 5.5,  # Deprecated - bad at reasoning
    }

    # Model swapping settings
    OLLAMA_MAX_LOADED_MODELS = 2  # Router + one specialist
    OLLAMA_KEEP_ALIVE_DEFAULT = "5m"  # Default keep-alive for models
    OLLAMA_AGGRESSIVE_UNLOAD = True  # Unload immediately after response

    # Specialist agent models
    SPECIALIST_MODELS = {
        "primary": "phi4:14b",              # Main agent - reasoning + tool use
        "researcher": "phi4:14b",           # Research synthesis
        "coder": "qwen2.5-coder:7b",        # Code generation
        "writer": "phi4:14b",               # Writing and analysis
    }

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
        # Example paths - customize for your environment:
        # Path.home() / "Projects" / "my-app",
        # Path.home() / "Arduino" / "sketches",
        # Path.home() / "code" / "my-project",
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
        # Example paths - customize for your environment:
        # Path.home() / "Projects" / "my-app",
        # Path.home() / "Arduino" / "sketches",
        # Path.home() / "code" / "my-project",
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

    # === Crawl4AI Settings (Web Crawling with JS Rendering) ===
    # Crawl4AI provides JavaScript rendering, anti-bot bypass, and LLM-ready output
    CRAWL4AI_CONFIG = {
        "enabled": True,                    # Enable Crawl4AI as primary fetcher
        "headless": True,                   # Run browser in headless mode
        "cache_enabled": True,              # Cache crawled pages
        "timeout_ms": 30000,                # Page timeout in milliseconds
        "stealth_mode": True,               # Anti-bot detection bypass
        "browser_type": "chromium",         # chromium | firefox | webkit

        # Content filtering
        "pruning_threshold": 0.48,          # Noise filtering threshold
        "min_word_threshold": 0,            # Minimum words per block

        # Rate limiting
        "max_concurrent": 5,                # Max concurrent requests
        "delay_between_requests": 0.5,      # Delay in seconds

        # Fallback behavior
        "fallback_to_trafilatura": True,    # Fall back to Trafilatura on failure
    }

    # Ollama settings for Crawl4AI LLM extraction (uses local Ollama)
    CRAWL4AI_LLM_CONFIG = {
        "provider": "ollama/phi4:14b",      # Match Workshop's main model
        "api_token": "ollama",              # Ollama doesn't need real token
        "base_url": OLLAMA_URL,             # Use same Ollama instance
        "backoff_base_delay": 2,            # Retry delay in seconds
        "backoff_max_attempts": 3,          # Max retry attempts
        "backoff_exponential_factor": 2,    # Exponential backoff multiplier
    }

    # === Trace Logging Settings ===
    # Enable detailed context tracing for subagent debugging
    TRACE_LOGGING_ENABLED = os.getenv("WORKSHOP_TRACE_LOGGING", "true").lower() == "true"
    TRACE_LOG_FULL_CONTEXT = os.getenv("WORKSHOP_TRACE_FULL_CONTEXT", "true").lower() == "true"
    TRACE_LOG_TOOL_RESULTS = os.getenv("WORKSHOP_TRACE_TOOL_RESULTS", "true").lower() == "true"
    TRACE_MAX_CONTENT_LENGTH = int(os.getenv("WORKSHOP_TRACE_MAX_LENGTH", "5000"))  # Truncate after this

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
