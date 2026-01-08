# Workshop

**Local Agentic Voice Assistant with Context Intelligence**

A Jarvis-style AI assistant that runs entirely on your machine. Voice or text input, intelligent tool organization, dual-layer context awareness, and extensible architecture powered by proven PAI patterns.

## What's New

**Research Platform Overhaul (2026-01-02):** Fixed critical issues with research document generation:
- **No More Arbitrary Truncation** - Content is NEVER truncated to arbitrary character limits
- **LLM-Powered Distillation** - All content reduction uses phi4:14b for intelligent distillation
- **Chunked Processing** - Large content split into 40KB semantic chunks for processing
- **Recursive Synthesis** - Multiple distillations automatically synthesized into coherent output
- **Comprehensive Handoffs** - Agent handoffs now preserve full context with phi4:14b (was phi3:mini with 500 char truncation)
- **Rich Research Summaries** - Final summaries synthesize ALL sources with citations and evidence

See [ARCHITECTURE.md](ARCHITECTURE.md#research-processing-pipeline) for technical details.

**Phase 5: Fabric-Style Patterns & Validated Research (2024-12-30):** Quality-focused research and decision support:
- **9 New Patterns** - Including `evaluate_search_results`, `analyze_tradeoffs`, `compare_technologies`
- **ValidatedResearch Workflow** - Research with built-in quality gates and source credibility checks
- **ImplementFeature Workflow** - End-to-end feature implementation with research
- **Source Quality Tools** - URL validation, metadata extraction, domain reputation database
- **19 Total Patterns** - Fabric-inspired text transformation patterns

See [docs/NEW_PATTERNS_AND_WORKFLOWS.md](docs/NEW_PATTERNS_AND_WORKFLOWS.md) for details.

**Phase 4: Semantic Routing & Multi-Agent (2024-12-28):** Intelligent intent routing and specialist agents:
- **Semantic Routing** - Embedding-based intent matching (no more brittle keyword patterns)
- **Two-Stage Routing** - Fast similarity check + LLM confirmation for ambiguous cases
- **Specialist Agents** - Dedicated models for code, writing, research synthesis
- **VRAM-Aware Model Swapping** - Preloading and parallel swaps for 12GB constraint
- **Context Compression** - Efficient handoffs between agents

See [docs/Phase4_Semantic_Routing_Complete.md](docs/Phase4_Semantic_Routing_Complete.md) for details.

**Telemetry & Observability System (2024-12-27):** Complete visibility into Workshop's decision-making:
- 📊 **Full Request Tracing** - Every interaction captured with 50+ telemetry fields
- 🔍 **Context Transparency** - See exactly what context was loaded from Telos, memory, and context manager
- 🛠️ **Tool Call Analysis** - Track every tool execution with timing, args, and results
- 📈 **Training Data Export** - Export traces in JSONL format for model fine-tuning
- 🐛 **Debug Mode** - Real-time trace summaries with `WORKSHOP_DEBUG=1`
- 📋 **CLI Viewer** - Inspect past interactions with `./trace_viewer.py`

See [docs/Telemetry_System.md](docs/Telemetry_System.md) for complete documentation.

**Phase 3 Complete (2025-12-26):** Workshop now features world-class context intelligence:
- 🎯 **Telos Personal Context** - Define who you are, your goals, and mission in human-editable markdown files
- 🔍 **Automatic Context Intelligence** - File watching, workflow detection, relationship tracking
- 🛠️ **30 Tools across 6 Skills** - Hierarchical, self-documenting tool organization
- 📝 **PAI-Inspired Patterns** - Template-based prompts with voice mode support

See [docs/Phase3_Telos_Personal_Context_Complete.md](docs/Phase3_Telos_Personal_Context_Complete.md) for details.

## Quick Start

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) running locally
- A model that supports function calling (recommended: `llama3.1:8b` or `qwen2.5:7b`)

```bash
# Install Ollama, then pull a model
ollama pull llama3.1:8b
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Workshop

```bash
# Text mode (easiest start)
python main.py --mode text

# Voice mode (requires audio setup)
python main.py --mode voice
```

On first run, Workshop automatically creates:
- Your personal context files at `~/.workshop/Telos/` (profile, goals, mission)
- Skills directory at `.workshop/Skills/`
- Memory databases at `data/`

### 4. Personalize Your Context

Edit your personal context to get better, more personalized assistance:

```bash
# Edit who you are
nano ~/.workshop/Telos/profile.md

# Edit your goals
nano ~/.workshop/Telos/goals.md

# Edit your long-term mission
nano ~/.workshop/Telos/mission.md

# In Workshop: "reload telos" to apply changes
```

## Architecture

### Skills-Based Tool Organization

Workshop uses a hierarchical Skills architecture (inspired by Daniel Miessler's PAI):

```
.workshop/Skills/
├── FileOperations/     # File system operations (4 tools)
├── Memory/             # Memory and notes (4 tools)
├── ContextIntelligence/# Code navigation & context (8 tools)
├── Research/           # Web search (1 tool)
├── VisualConstructs/   # Spatial UI elements (6 tools)
└── Telos/              # Personal context management (7 tools)
```

**Total:** **40+ tools** intelligently organized and routed by user intent (including pattern tools).

### Dual-Layer Context System

Workshop combines two types of context:

**1. Automatic Context (ContextManager)**
- File watching with debouncing
- Workflow detection (debugging, feature development, configuration, research)
- File relationship tracking (imports, co-edits)
- Recent edit history

**2. Personal Context (Telos)**
- Your identity, preferences, tech stack
- Current goals (weekly/monthly)
- Long-term mission and vision
- Project-specific context

Together, these provide Workshop with complete awareness of both what you're doing (automatic) and who you are (personal).

## Available Skills & Tools

### FileOperations (4 tools)
- `read_file` - Read file contents with auto-search and case-insensitive resolution
- `write_file` - Write/append to files, auto-creates parent directories
- `list_directory` - List directory with icons and human-readable sizes
- `search_files` - Search text in files or find by filename

### Memory (4 tools)
- `remember` - Store information in ChromaDB + SQLite dual storage
- `recall` - Semantic memory search
- `take_note` - Create timestamped markdown notes
- `list_notes` - List recent notes

### ContextIntelligence (8 tools) ⚡ HIGH PRIORITY
- `get_file_content` - Read file with line range support
- `search_project_files` - Semantic project search
- `find_definition` - Multi-language symbol definition finder
- `find_references` - Cross-project symbol usage
- `get_related_files` - Imports + co-edits from context graph
- `get_recent_edits` - Edit history
- `search_web_docs` - External documentation search
- `get_context_stats` - Workflow + active files statistics

**Unique Advantage:** Leverages file watching, workflow detection, and context graph.

### Research (6+ tools) ⭐ ENHANCED
- `web_search` - DuckDuckGo integration
- `deep_research` - Multi-query autonomous research
- `validate_url` - Check URL accessibility and detect paywalls
- `extract_page_metadata` - Get title, author, date, word count
- `check_source_reputation` - Domain tier and bias assessment
- Plus 9 pattern tools for text transformation

### VisualConstructs (6 tools)
- `show_file` - Display file in visual code panel
- `show_directory` - Tree visualization
- `create_note` - Floating note cards
- `close_construct` - Close specific construct
- `close_all_constructs` - Clear all
- `list_constructs` - Show active constructs

**Vision:** Spatial computing UI (Phase 4 roadmap)

### Telos (7 tools) ⭐ NEW
- `edit_profile` - Open profile.md for editing
- `edit_goals` - Open goals.md for editing
- `edit_mission` - Open mission.md for editing
- `reload_telos` - Reload all personal context
- `show_telos_stats` - Display Telos statistics
- `list_projects` - List all project contexts
- `create_project` - Create new project context

**Purpose:** Manage your personal identity, goals, and project-specific context.

## Voice Mode

Workshop features voice-optimized prompts and hands-free operation:

```bash
# Install voice dependencies
pip install faster-whisper sounddevice soundfile

# Download Piper TTS (recommended)
# See: https://github.com/rhasspy/piper

# Run in voice mode
python main.py --mode voice

# Say "workshop" to activate, then speak your command
```

**Voice optimizations:**
- Concise responses (under 3 sentences when possible)
- Confirms actions before executing
- Natural speech patterns
- Summarizes long outputs

## Observability & Debugging

### Telemetry System (NEW)

Workshop captures complete telemetry for every interaction:

```bash
# View recent traces
./trace_viewer.py

# Inspect specific trace
./trace_viewer.py --id <trace_id>

# Show statistics
./trace_viewer.py --stats

# Export for training data
./trace_viewer.py --export training_data.jsonl

# Debug mode - print traces in real-time
WORKSHOP_DEBUG=1 python main.py --mode text
```

**What's captured:**
- Full context assembly (Telos, memory, context manager)
- Intent detection and pattern matching
- LLM calls with prompts and responses
- Tool executions with arguments and results
- Complete timing breakdown
- Error tracebacks

See [docs/Telemetry_System.md](docs/Telemetry_System.md) for details.

## Memory & Context Architecture

### Memory System (ChromaDB + SQLite)

1. **Immediate** - Current conversation (in-memory)
2. **Session** - SQLite for structured facts, user profile
3. **Long-term** - ChromaDB for semantic search
4. **Telemetry** - SQLite trace storage for observability

### Context Intelligence (Phase 3)

**Automatic Context:**
- Monitors project directories with file watching
- Detects workflow patterns (debugging, feature dev, config, research)
- Builds context graph of file relationships
- Tracks recent edits and active files

**Personal Context (Telos):**
- User-defined identity and preferences
- Current goals and objectives
- Long-term mission and vision
- Project-specific context files

Both contexts are automatically assembled and included in LLM prompts for truly context-aware assistance.

## Configuration

Edit `config.py` to customize:

```python
# Change the model
MODEL = "qwen2.5:7b"  # Excellent at function calling

# Add monitored projects for context intelligence
MONITORED_PROJECTS = [
    Path.home() / "projects" / "my-project",
    Path.home() / "Arduino",
]

# File extensions to index
INDEXABLE_EXTENSIONS = {'.py', '.ino', '.cpp', '.h', '.md'}
```

## Project Structure

```
workshop/
├── main.py                 # Entry point and orchestration
├── config.py               # Configuration settings
├── agent.py                # Ollama + tool calling + Pattern integration
├── skill_registry.py       # Skills system with routing
├── pattern_loader.py       # PAI-style prompt templates
├── pattern_executor.py     # Fabric-style pattern execution
├── pattern_tools.py        # Pattern tool functions (19 patterns)
├── research_tools.py       # URL validation, metadata, reputation
├── context_manager.py      # Automatic context intelligence
├── telos_manager.py        # Personal context system
├── memory.py               # ChromaDB + SQLite memory
├── telemetry.py            # Observability and tracing
├── trace_viewer.py         # CLI tool for trace inspection
│
├── .workshop/              # Project-specific configuration
│   ├── patterns/           # Prompt templates (base/system.md, voice_system.md)
│   ├── Skills/             # Tool organization (6 skills, 40+ tools)
│   │   └── Research/
│   │       └── Workflows/  # ValidatedResearch, ImplementFeature
│   └── agents/             # Specialist agent definitions
│
├── ~/.workshop/            # Global configuration
│   ├── Telos/              # Personal context (profile, goals, mission)
│   └── patterns/           # Fabric-style patterns (19 patterns)
│       ├── evaluate/       # Search quality, source credibility
│       ├── extract/        # Wisdom, ideas, implementation steps
│       ├── analyze/        # Claims, papers, tradeoffs
│       ├── compare/        # Technology comparisons
│       ├── create/         # Summaries, learning paths, decisions
│       ├── improve/        # Writing improvement
│       └── transform/      # Text cleaning
│
├── data/                   # Created at runtime
│   ├── chromadb/           # Vector memory
│   ├── workshop.db         # SQLite database
│   ├── telemetry.db        # Trace storage
│   ├── traces/             # JSON trace files
│   ├── logs/               # Conversation logs
│   └── notes/              # Note files
│
├── docs/                   # Documentation
│   ├── NEW_PATTERNS_AND_WORKFLOWS.md  # Phase 5 additions
│   ├── Telemetry_System.md
│   ├── Phase4_Semantic_Routing_Complete.md
│   └── ...more docs
│
└── tests/
    ├── test_skills_system.py
    ├── test_telos_system.py
    └── test_pattern_loader.py
```

## Adding Custom Skills

Create a new skill by adding a directory in `.workshop/Skills/`:

```
.workshop/Skills/MySkill/
├── SKILL.md              # Routing patterns, keywords, priority
├── __init__.py
└── tools/
    ├── __init__.py
    ├── my_tool.py        # Tool implementation
    └── another_tool.py
```

**Tool file format:**

```python
"""
my_tool - Brief description

Part of MySkill.
"""

from logger import get_logger

log = get_logger("MySkill.my_tool")

TOOL_DESCRIPTION = "What this tool does"
TOOL_SIGNATURE = "my_tool(arg1: str, arg2: int = 0) -> str"
TOOL_EXAMPLES = [
    "my_tool('test', 42)"
]

async def my_tool(arg1: str, arg2: int = 0, _deps: dict = None) -> str:
    """Tool implementation"""
    deps = _deps or {}
    config = deps.get("config")

    # Your logic here
    return f"Result: {arg1}, {arg2}"
```

Skills are automatically loaded by SkillRegistry on startup!

## Design Philosophy

Workshop follows proven architectural patterns from Daniel Miessler's Personal AI Infrastructure (PAI):

1. **Scaffolding > Model** - Well-designed system beats smart model in poor system
2. **UNIX Philosophy** - Small, composable, single-purpose tools
3. **Code Before Prompts** - Solve with architecture, not prompt engineering
4. **Template-Based Prompts** - Standardized, reusable prompt structures
5. **Skills Architecture** - Hierarchical tool organization with routing

**Workshop's Unique Advantages:**
- Phase 3 context intelligence (file watching, workflow detection)
- Dual context system (automatic + personal)
- ChromaDB semantic memory (superior to filesystem)
- Voice-first design with optimized prompts
- Visual constructs for spatial computing

## Testing

Comprehensive test suites included:

```bash
# Test Skills system (23 tools)
python test_skills_system.py

# Test Telos system (7 tools)
python test_telos_system.py
```

All tests pass ✅

## Roadmap

### Completed
- **Phase 1:** PAI-style prompt templates with voice variants
- **Phase 2:** Skills architecture (30 tools across 6 skills)
- **Phase 3:** Dual-layer context intelligence (Automatic + Telos)
- **Phase 4:** Semantic routing & multi-agent orchestration
- **Phase 5:** Fabric-style patterns & validated research (19 patterns, 2 workflows)

### In Progress
- Voice pipeline optimization
- Spatial UI for visual constructs

### Planned
- **Phase 6:** Provider abstraction (OpenAI, Anthropic, Gemini support)
- **Phase 7:** Advanced Telos features (file watching, versioning, persona profiles)
- **Phase 8:** Hardware integration (Battery Guardian, Arduino projects)

## Troubleshooting

### "Connection error: Is Ollama running?"
```bash
ollama serve  # Start Ollama server
```

### "ChromaDB not installed"
```bash
pip install chromadb
```

### Skills not loading
```bash
# Check Skills directory
ls .workshop/Skills/

# Verify each skill has SKILL.md and tools/
# Check logs for loading errors
```

### Telos context not working
```bash
# Check Telos files exist
ls ~/.workshop/Telos/

# Reload context
# In Workshop: "reload telos"

# View stats
# In Workshop: "show telos stats"
```

### Voice not working
```bash
# Check audio devices
python -c "import sounddevice; print(sounddevice.query_devices())"

# Test recording
python -c "import sounddevice as sd; print(sd.rec(16000, samplerate=16000, channels=1))"
```

### Model not responding well to tool calls
Try a different model. Recommended:
- `llama3.1:8b` (balanced, excellent tool use)
- `qwen2.5:7b` (very strong at function calling)
- `mistral:7b` (fast)

## Documentation

Comprehensive documentation in `docs/`:
- [Development Plan](docs/Development_Plan_PAI_Integration.md) - Overall integration roadmap
- [PAI Learnings](docs/PAI_v2_Learnings.md) - Daniel Miessler's design principles
- [Skills Architecture](docs/Skills_Architecture_Specification.md) - Technical specification
- [Phase 2 Complete](docs/Phase2_Skills_Migration_Complete.md) - Skills migration details
- [Phase 3 Complete](docs/Phase3_Telos_Personal_Context_Complete.md) - Telos implementation
- [Phase 4 Complete](docs/Phase4_Semantic_Routing_Complete.md) - Semantic routing & multi-agent
- [Telemetry System](docs/Telemetry_System.md) - Observability, debugging, and training data
- [Subagent Architecture](docs/Subagent_Architecture.md) - Multi-agent orchestration

## Acknowledgments

**Inspired by:**
- **Daniel Miessler's Personal AI Infrastructure (PAI)** - Skills architecture, pattern templates, Telos concept
- **Fabric** (34,900 stars) - Pattern system and prompt engineering approach
- **Greek Philosophy** - Telos (τέλος) = "ultimate purpose"

**Workshop's Implementation:**
- Dual-layer context system (unique)
- Phase 3 automatic context intelligence
- Voice-first optimizations
- ChromaDB semantic memory
- Spatial computing vision

## Contributing

Workshop is a personal project, but contributions welcome:
- Bug reports and feature requests via GitHub Issues
- Code improvements via Pull Requests
- Documentation enhancements
- New Skills and tools

## License

MIT - Use it, modify it, build with it.

---

**Workshop - Your context-aware AI assistant, powered by proven patterns and personal context.**
