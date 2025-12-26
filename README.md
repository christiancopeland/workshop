# Workshop

**Local Agentic Voice Assistant**

A Jarvis-style AI assistant that runs entirely on your machine. Voice or text input, tool use, persistent memory, and extensible architecture.

## Quick Start

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) running locally
- A model that supports function calling (recommended: `llama3.1:8b`)

```bash
# Install Ollama, then pull a model
ollama pull llama3.1:8b
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run in Text Mode (easiest start)

```bash
python main.py --mode text
```

### 4. Run in Voice Mode (requires additional setup)

```bash
# Install voice dependencies
pip install faster-whisper sounddevice soundfile

# Download Piper TTS (optional but recommended)
# See: https://github.com/rhasspy/piper

python main.py --mode voice
```

## Modes

| Mode | Command | Description |
|------|---------|-------------|
| Text | `--mode text` | Type input, read output. Best for development. |
| Voice | `--mode voice` | Say "workshop" to activate, speak commands. |
| Hybrid | `--mode hybrid` | Both voice and text input available. |

## Available Tools

Workshop comes with these built-in tools:

### File Operations
- `read_file` - Read file contents
- `write_file` - Write/append to files
- `list_directory` - List directory contents
- `search_files` - Search for text in files

### Shell Commands
- `run_shell` - Execute whitelisted shell commands

### Memory
- `remember` - Store information long-term
- `recall` - Search memories semantically
- `take_note` - Create timestamped note files
- `list_notes` - List recent notes

### Utilities
- `get_current_time` - Current date/time
- `calculate` - Math expressions
- `web_search` - DuckDuckGo search (requires `duckduckgo-search`)

## Adding Custom Tools

Edit `tools.py` to add your own tools:

```python
def my_custom_tool(arg1: str, arg2: int = 10) -> str:
    """Description of what this tool does"""
    # Your implementation
    return f"Result: {arg1}, {arg2}"

registry.register(
    name="my_custom_tool",
    func=my_custom_tool,
    description="What this tool does",
    signature="my_custom_tool(arg1: str, arg2: int = 10)",
    examples=[
        {"tool": "my_custom_tool", "args": {"arg1": "test"}}
    ]
)
```

## Configuration

Edit `config.py` to customize:

```python
# Change the model
MODEL = "qwen2.5:7b"  # Good at function calling

# Add project paths for RAG indexing
PROJECT_PATHS = [
    Path.home() / "projects" / "battery-guardian",
    Path.home() / "projects" / "flying-tiger-rc",
]

# Modify allowed shell commands
ALLOWED_SHELL_COMMANDS = {
    'ls', 'cat', 'git', 'python', ...
}
```

## Project Structure

```
workshop/
├── main.py          # Entry point and orchestration
├── config.py        # Configuration settings
├── voice.py         # Whisper STT + Piper TTS
├── agent.py         # Ollama integration + tool calling
├── tools.py         # Tool definitions
├── memory.py        # ChromaDB + SQLite memory
├── requirements.txt # Dependencies
└── data/            # Created at runtime
    ├── chromadb/    # Vector memory
    ├── workshop.db  # SQLite database
    ├── logs/        # Conversation logs
    └── notes/       # Note files
```

## Memory Architecture

Workshop uses a tiered memory system:

1. **Immediate** - Current conversation (in-memory)
2. **Session** - SQLite for structured facts, user profile
3. **Long-term** - ChromaDB for semantic search

The agent automatically:
- Updates the user profile based on conversation patterns
- Stores important information to long-term memory
- Retrieves relevant context for each query

## Voice Setup (Detailed)

### Whisper (Speech-to-Text)

faster-whisper runs on GPU for real-time transcription:

```bash
pip install faster-whisper

# Models (pick based on accuracy vs speed):
# tiny.en  - Fastest, less accurate
# base.en  - Good balance (default)
# small.en - Better accuracy
# medium.en - Even better
# large-v3 - Best, but slower
```

### Piper (Text-to-Speech)

Piper provides fast, natural local TTS:

```bash
# Download from releases:
# https://github.com/rhasspy/piper/releases

# Get a voice model:
# https://huggingface.co/rhasspy/piper-voices

# Add to PATH or specify in config
```

### Audio Configuration

If you have issues with audio:

```bash
# List audio devices
python -c "import sounddevice; print(sounddevice.query_devices())"

# Set specific device in config.py if needed
```

## Roadmap

This is Phase 1 of a larger project. Future phases:

- **Phase 2**: Gesture input (MediaPipe hand tracking)
- **Phase 3**: Visual constructs (spatial UI elements)
- **Phase 4**: Physical awareness (object recognition)
- **Phase 5**: Hardware integration (Battery Guardian, etc.)

## Troubleshooting

### "Connection error: Is Ollama running?"
```bash
ollama serve  # Start Ollama server
```

### "ChromaDB not installed"
```bash
pip install chromadb
```

### Voice not working
```bash
# Check audio devices
python -c "import sounddevice; print(sounddevice.query_devices())"

# Test recording
python -c "import sounddevice as sd; print(sd.rec(16000, samplerate=16000, channels=1))"
```

### Model not responding well to tool calls
Try a different model. Good options:
- `llama3.1:8b` (balanced)
- `qwen2.5:7b` (strong at function calling)
- `mistral:7b` (fast)

## License

MIT - Use it, modify it, build with it.
