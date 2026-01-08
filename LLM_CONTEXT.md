# Workshop - LLM Developer Context

This document provides a comprehensive overview for LLM developers working on the Workshop codebase.

## Architecture Overview

Workshop is a Python-based AI agent system featuring two-stage Fabric-style routing, skill-based tool execution, **Claude Code as the reasoning engine** (via subscription CLI), persistent memory (SQLite + ChromaDB), and real-time WebSocket dashboard observability.

**January 2026 Update:** Workshop now uses Claude Code CLI (`claude -p`) as the primary reasoning backend instead of local Ollama models. This provides significantly improved reasoning quality while Workshop maintains control over routing, tool execution, and memory.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INPUT                                     │
│                        (Text / Voice / Dashboard)                           │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           WORKSHOP AGENT                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STAGE 0: Context Injection                                         │    │
│  │  - ContextManager: active files, recent changes, detected workflow  │    │
│  │  - TelosManager: user profile, goals, project context               │    │
│  │  - TaskManager: current task list and progress                      │    │
│  └─────────────────────────────────┬───────────────────────────────────┘    │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STAGE 1: Semantic-First Routing (Phase 3)                          │    │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │    │
│  │  │ SemanticRouter →  │ PromptRouter │ →  │ SkillExecutor│           │    │
│  │  │ (embeddings) │    │ (Claude fallback) │ (Claude Code)│           │    │
│  │  └──────────────┘    └──────────────┘    └──────────────┘           │    │
│  └─────────────────────────────────┬───────────────────────────────────┘    │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STAGE 2: Skill Execution (via Claude Code CLI)                     │    │
│  │  - ClaudeCodeBridge: subprocess wrapper for `claude -p`             │    │
│  │  - Non-native tool calling (<tool_call> XML format)                 │    │
│  │  - Workshop executes tools locally, feeds results back to Claude    │    │
│  └─────────────────────────────────┬───────────────────────────────────┘    │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STAGE 3: Auto-Continue & Task Advancement                          │    │
│  │  - Track task completion with work evidence validation              │    │
│  │  - Recursively continue if pending tasks exist                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
┌─────────────────────┐ ┌─────────────────┐ ┌─────────────────────┐
│   MemorySystem      │ │   Telemetry     │ │   DashboardServer   │
│ SQLite + ChromaDB   │ │ Traces + Stats  │ │   WebSocket:8766    │
└─────────────────────┘ └─────────────────┘ └─────────────────────┘
```

## Directory Structure

```
workshop/
├── main.py                    # Entry point, CLI, Workshop class
├── agent.py                   # Core Agent class, conversation loop, context assembly
├── config.py                  # All configuration options, model settings
├── logger.py                  # Logging setup (file + console handlers)
│
├── claude_bridge.py           # Claude Code CLI wrapper (subprocess + JSON parsing)
├── hooks.py                   # NEW: Hook system for lifecycle events (SESSION_START, POST_TOOL_USE)
├── router.py                  # PromptRouter: skill discovery, intent classification
├── semantic_router.py         # SemanticRouter: embedding-based pre-filtering
├── skill_executor.py          # SkillExecutor: constrained tool execution (uses Claude Code)
├── skill_registry.py          # SkillRegistry: tool registration, dependency injection
│
├── pattern_executor.py        # PatternExecutor, PatternRegistry, PatternPipeline
├── pattern_loader.py          # PAI-style templates, PromptPrimitives
├── pattern_tools.py           # Pattern-to-tool bridge (extract_wisdom, etc.)
│
├── memory.py                  # MemorySystem: SQLite + ChromaDB, three-tier memory
├── session_manager.py         # SessionManager: session lifecycle, isolation
├── task_manager.py            # TaskManager: task tracking, work evidence validation
├── telos_manager.py           # TelosManager: personal context (profile/goals/projects)
│
├── subagent_manager.py        # SubagentManager: model swapping, context snapshots
├── telemetry.py               # TelemetryCollector: traces, LLM/tool call tracking
├── dashboard.py               # DashboardServer: WebSocket events, REST API
├── dashboard_integration.py   # Async event emission helpers
├── trace_dashboard.py         # Flask web UI for real-time trace visualization (port 5001)
│
├── project_tools.py           # Project context and Arduino tools
├── research_tools.py          # URL validation, source reputation checking
│
├── .workshop/                 # User configuration directory
│   ├── Skills/                # Skill definitions with tools
│   │   ├── Research/          # Research skill (SKILL.md, system.md, tools/)
│   │   │   └── tools/
│   │   │       ├── web_search.py
│   │   │       ├── deep_research.py
│   │   │       ├── fetch_url.py
│   │   │       ├── save_research_output.py   # NEW: Scoped file write
│   │   │       ├── archive_source.py         # NEW: URL to database
│   │   │       ├── index_findings.py         # NEW: Content to ChromaDB
│   │   │       ├── _research_state.py        # UPDATED: Auto-load from disk
│   │   │       └── continue_research.py      # UPDATED: Disk fallback
│   │   ├── FileOperations/    # File operations skill
│   │   ├── Memory/            # Memory tools (remember, recall)
│   │   ├── TaskManagement/    # Task tools (task_write, task_read)
│   │   └── Subagents/         # Subagent spawning tool
│   ├── patterns/              # Fabric-style patterns
│   │   ├── base/              # system.md, voice_system.md
│   │   ├── extract/           # extract_wisdom, extract_ideas, etc.
│   │   ├── analyze/           # analyze_claims, analyze_paper, etc.
│   │   └── create/            # create_summary, create_synthesis, etc.
│   ├── agents/                # Subagent definitions (YAML frontmatter + prompt)
│   │   ├── primary.md
│   │   ├── research-summarizer.md
│   │   ├── coder.md           # Code generation specialist
│   │   ├── writer.md          # Documentation specialist
│   │   ├── web-researcher.md  # Web research with multi-query expansion
│   │   ├── codebase-analyst.md # Codebase understanding specialist
│   │   ├── tech-comparator.md # Technology comparison specialist
│   │   └── snapshots/         # Context snapshots for handoffs
│   ├── Telos/                 # Personal context
│   │   ├── profile.md         # User identity, preferences
│   │   ├── goals.md           # Current objectives
│   │   ├── mission.md         # Long-term vision
│   │   └── projects/          # Project-specific context
│   ├── sessions/              # Session management
│   │   ├── current.json       # Active session
│   │   └── archive/           # Archived sessions
│   └── tasks/                 # Task persistence
│       └── current.json       # Active task list
│
├── data/                      # Runtime data (auto-created)
│   ├── chromadb/              # Vector database
│   ├── workshop.db            # SQLite metadata
│   ├── telemetry.db           # Telemetry traces
│   ├── logs/                  # Session logs (workshop_*.log)
│   └── skill_embeddings.json  # Cached skill embeddings
│
└── dashboard/                 # React dashboard (separate LLM_CONTEXT.md)
```

## Key Files Reference

### Claude Code Integration

**`claude_bridge.py`** - Claude Code CLI wrapper and session management (~650 lines):

**Core Bridge:**
- `ClaudeCodeBridge.__init__()`: Initialize bridge, verify Claude Code installation
- `ClaudeCodeBridge.query()`: Main method - sends messages to Claude Code via subprocess
- `ClaudeCodeBridge._messages_to_prompt()`: Convert OpenAI-style messages to CLI prompt
- `ClaudeCodeBridge._extract_tool_calls_from_content()`: Parse `<tool_call>` JSON from response
- Uses explicit binary path `~/.local/bin/claude` for security
- Disables Claude's native tools with `--tools ''` so Workshop handles tool execution
- Returns dict compatible with old `_call_ollama` interface: `{"content": "...", "tool_calls": [...]}`

**Session Management (Phase 4):**
- `ClaudeSessionState`: Persisted state linking Workshop sessions to Claude Code sessions
- `ClaudeBridgeManager`: Singleton manager providing:
  - Shared bridge instance for session continuity across components
  - Session state persistence to `~/.workshop/sessions/claude_session.json`
  - Turn counting and token estimation
  - Session summarization for context efficiency
- `get_claude_bridge()`: Get shared bridge instance (recommended for most use cases)
- `get_claude_bridge_manager()`: Get manager instance for advanced session control

### Hook System

**`hooks.py`** - Extensible lifecycle event system (~600 lines):

**Core Components:**
- `HookType` enum: `SESSION_START`, `POST_TOOL_USE` (future: PRE_TOOL_USE, SESSION_END)
- `HookContext` dataclass: Carries state through hook chain with mutable `system_prompt_additions`
- `HookManager` class: Singleton registry with `register()` and `execute()` methods
- `get_hook_manager()`: Get the global singleton instance

**Built-in Handlers:**
| Handler | Hook Type | Priority | Purpose |
|---------|-----------|----------|---------|
| `hydrate_capabilities` | SESSION_START | 5 | Injects available tools and skills manifest |
| `hydrate_telos_context` | SESSION_START | 10 | Injects user profile, goals, mission, active project |
| `hydrate_active_tasks` | SESSION_START | 20 | Injects current task list |
| `auto_persist_research` | POST_TOOL_USE | 50 | Auto-indexes research results to ChromaDB |

**Priority Ranges:**
- 0-20: Core/system handlers (telos, tasks)
- 21-50: Standard handlers
- 51-80: Enhancement handlers
- 81-100: Logging/observability handlers

**Terminal Visualization:**
Hooks display execution status in terminal:
```
🔗 [HOOKS] ━━━ SESSION_START ━━━ (3 handlers)
   ├─ [05] hydrate_capabilities... ✓ (+1 sections, [10 skills, 55 tools])
   ├─ [10] hydrate_telos_context... ✓ (+4 sections, [profile, goals, mission, project:workshop])
   ├─ [20] hydrate_active_tasks... ✓ ([2 active, 5 pending])
   └─ Total: 6 sections, 13000 chars injected

🔧 [HOOKS] ━━━ POST_TOOL_USE [web_search] ━━━ (1 handlers)
   ├─ [50] auto_persist_research... ✓ (indexed to memory, logged to index)
```

**Usage Example:**
```python
from hooks import get_hook_manager, HookType, HookContext

async def my_custom_handler(ctx: HookContext) -> HookContext:
    ctx.add_system_context("Custom Section", "Custom content")
    return ctx

get_hook_manager().register(HookType.SESSION_START, my_custom_handler, priority=30)
```

**Integration Points:**
- `agent.py`: Fires SESSION_START hooks in `_two_stage_chat()`, line ~906
- `skill_executor.py`: Fires POST_TOOL_USE hooks after tool execution, line ~532

### Core Agent Loop

**`agent.py`** - The heart of the system (~2700 lines):
- `Agent.__init__()` (lines 105-179): Initialize with all managers
- `Agent.chat()` (lines 1120-1132): Entry point, routes to two-stage or legacy
- `Agent._two_stage_chat()` (lines 826-1118): Main conversation flow
- `Agent._build_system_message()` (lines 1626-1791): Context assembly for LLM
- `Agent._execute_tool_traced()` (lines 2280-2344): Tool execution with telemetry

**`main.py`** - Entry point and CLI:
- `Workshop.__init__()` (lines 68-182): Initialize all subsystems
- `Workshop.process_input()` (lines 219-242): Process user input
- `main()` (lines 525-567): CLI argument parsing

### Two-Stage Routing

**`router.py`** - PromptRouter for skill classification:
- Skills discovered from `~/.workshop/Skills/`
- Each skill has: `SKILL.md` (metadata), `system.md` (prompt), `tools/` (functions)
- Routes user input to appropriate skill based on intent patterns

**`semantic_router.py`** - Embedding-based pre-filtering:
- Model: `sentence-transformers/all-MiniLM-L6-v2` (CPU, ~1.2GB RAM)
- Bypass threshold: 0.85 (skip router, go direct to skill)
- Confirm threshold: 0.45 (router confirms from top candidates)

**`skill_executor.py`** - Constrained tool execution:
- `SkillExecutor.execute()` (lines 619-645): Main execution method
- Non-native tool calling: `<tool_call>{"tool": "name", "args": {...}}</tool_call>`
- Parses XML tool calls from LLM response (allows phi4:14b reasoning)

### Skill & Tool System

**`skill_registry.py`** - Central tool management:
- `SkillRegistry.__init__()` (lines 605-632): Load skills, inject dependencies
- `SkillRegistry.execute()` (lines 706-764): Execute tool with normalization
- `ToolInfo` dataclass (lines 28-37): name, func, description, signature
- Argument aliases (lines 580-603): LLM flexibility for parameter names
  - Path aliases: `directory`→`path`, `filepath`→`path`, `file`→`path`, etc.
  - Content aliases: `text`→`content`, `message`→`content`, `body`→`content`
  - Research aliases: `notes`→`summary`, `description`→`summary` (prevents hallucination errors)

**Skill Directory Structure:**
```
~/.workshop/Skills/{SkillName}/
├── SKILL.md         # Skill metadata (name, priority, semantic_utterances)
├── system.md        # Skill-specific system prompt
└── tools/
    ├── tool_name.py # Tool implementation with TOOL_DESCRIPTION, TOOL_SIGNATURE
    └── ...
```

**SKILL.md Semantic Utterances:**
Skills use embedding-based routing via semantic utterances in SKILL.md:
```markdown
## Semantic Utterances
These natural language examples are used for embedding-based semantic routing:

### Category 1
- example phrase one
- example phrase two

### Category 2
- another example phrase
```

Run `python generate_embeddings.py` after modifying semantic utterances.

**System Prompt Tool-First Approach:**
All skill system.md files should include anti-hallucination instructions:
```markdown
# CRITICAL: TOOL-FIRST APPROACH

**NEVER generate responses from general knowledge.**
- ALWAYS use tools before making claims
- If you can't help with available tools, say so
- Do NOT hallucinate answers
```

### Research Persistence Tools (NEW - January 2026)

The Research skill now includes three tools for persisting findings. These fix the "discover but not persist" gap where research would evaporate when sessions ended.

**Files:**
- `.workshop/Skills/Research/tools/save_research_output.py` - Scoped file write
- `.workshop/Skills/Research/tools/archive_source.py` - URL to database
- `.workshop/Skills/Research/tools/index_findings.py` - Content to ChromaDB
- `.workshop/Skills/Research/tools/_research_state.py` - Research state with disk persistence

**Tool Overview:**
| Tool | Storage | Idempotency | Force Override |
|------|---------|-------------|----------------|
| `save_research_output(filename, content, format)` | File + ChromaDB | SHA256 hash in `_content_index.json` | `force_update=True` |
| `archive_source(url, title, summary, tags)` | SQLite + ChromaDB | URL hash in facts table | `force_update=True` |
| `index_findings(content, topic, tags)` | ChromaDB | MD5 content hash check | `force_reindex=True` |

**Key Design Decisions:**
1. **Scoped Write**: `save_research_output` can ONLY write to `data/research/`. No general filesystem access.
2. **Dual Storage**: `archive_source` saves to both SQLite (structured) and ChromaDB (semantic search).
3. **Idempotent by Default**: All tools check for existing content and skip duplicates.
4. **Argument Aliases**: `save_research_output` accepts both `filename` and `path` for LLM compatibility.

**Research State Persistence:**
`_research_state.py` now auto-loads from disk when the in-memory researcher instance is None. This fixes context loss between queries:
```python
def get_researcher(auto_load: bool = True):
    if researcher_instance is None and auto_load:
        return _load_researcher_from_disk()  # Loads from ~/.workshop/research/_active.json
```

**Storage Locations:**
```
data/research/                    # Research output files
├── FAS_Intel_Resources.md
├── _content_index.json           # Hash index for idempotency
└── ...

~/.workshop/research/             # Research platform state
├── _active.json                  # Pointer to current platform
└── {topic}.json                  # Platform data (sources, findings)
```

### Pattern System (Fabric-style)

**`pattern_executor.py`** - Pattern execution engine:
- `PatternExecutor.execute()` (lines 187-368): Execute single pattern
- `PatternPipeline.execute()` (lines 370-512): Unix pipe-style chaining
- `PatternRegistry` (lines 89-184): Pattern discovery and metadata
- Patterns loaded from `~/.workshop/patterns/{category}/{pattern_name}/system.md`

**`pattern_loader.py`** - PAI-style templates:
- `PatternLoader.load_pattern()` (lines 45-98): Load with variable substitution
- `PromptPrimitives` (lines 249-384): Composable prompt blocks
  - `roster()`: Formatted tool/skill lists
  - `voice()`: Communication style (concise/detailed/debug)
  - `structure()`: Response format (json/markdown/yaml)
  - `briefing()`: Context information blocks

**Available Patterns:**
- `extract_*`: wisdom, ideas, questions, implementation_steps
- `analyze_*`: claims, paper, tradeoffs
- `create_*`: summary, synthesis, learning_path, decision_doc
- `improve_*`: writing
- `compare_*`: technologies
- `evaluate_*`: search_results

### Memory System

**`memory.py`** - Three-tier memory (~990 lines):
- **Immediate**: In-memory conversation buffer (`_messages` list)
- **Session**: SQLite for facts, projects, messages
- **Long-term**: ChromaDB for semantic search

**Key Tables (SQLite):**
| Table | Purpose |
|-------|---------|
| `user_profile` | User metadata (single row) |
| `facts` | Key-value facts store (includes `archived_source_*` keys) |
| `messages` | Conversation history (session-scoped) |
| `projects` | Project metadata |
| `project_notes` | Project-specific notes |
| `sessions` | Session metadata |
| `indexed_files` | File indexing cache |

**Key Methods:**
- `add_message(role, content)` (lines 214-236): Log conversation
- `search_memories(query, k, include_metadata)` (lines 429-479): Semantic search with optional rich results
- `set_active_project(name)` (lines 640-672): Project management
- `get_project_context()` (lines 767-801): Formatted context for LLM

**ChromaDB Categories:**
| Category | Purpose |
|----------|---------|
| `research_output` | Saved research files (via `save_research_output`) |
| `research_sources` | Archived URLs (via `archive_source`) |
| `research:{topic}` | Indexed findings by topic (via `index_findings`) |
| `research_findings` | General indexed findings |
| `project` | Indexed project files |
| `conversation` | Logged conversation snippets |

### Session & Task Management

**`session_manager.py`** - Session lifecycle:
- `Session` dataclass (lines 25-67): session_id, timestamps, mode, status
- `SessionManager.start_session()` (lines 104-146): Start new, archive previous
- `SessionManager.detect_stale_state()` (lines 412-446): Find orphan tasks
- Session ID format: `sess_YYYYMMDD_HHMMSS_xxxx`

**`task_manager.py`** - Task tracking:
- `Task` dataclass (lines 32-69): content, status, active_form
- `TaskStatus` enum: PENDING, IN_PROGRESS, COMPLETED
- **Work Evidence Validation** (lines 312-336): Anti-hallucination check
- `TaskManager.bind_to_session()` (lines 159-172): Session isolation

**`telos_manager.py`** - Personal context:
- `TelosContext` dataclass (lines 19-30): profile, goals, mission, projects
- Loads from `~/.workshop/Telos/` (profile.md, goals.md, mission.md)
- 5-minute cache for performance
- `format_for_llm()` (lines 284-352): Inject into system prompt

### Subagent Orchestration

**`subagent_manager.py`** - Multi-model coordination (~1550 lines):
- `SubagentManager.spawn_subagent()` (lines 1020-1193): Main orchestration
- `SubagentManager.swap_models()` (lines 926-958): Parallel preloading
- `SubagentManager._execute_subagent_tool()` (lines 741-800): Tool execution with skill_registry fallback
- `ContextSnapshot` (lines 261-323): Serializable context for handoffs
- `ContextCompressor` (lines 342-583): Context preservation strategies

**Available Subagent Specialists:**
| Agent | Model | Purpose |
|-------|-------|---------|
| `coder` | qwen2.5-coder:7b | Code generation, refactoring, review |
| `writer` | mistral:7b | Documentation, tutorials, explanations |
| `research-summarizer` | phi4:14b | Synthesizing research findings |
| `web-researcher` | phi4:14b | Web research with multi-query expansion |
| `codebase-analyst` | qwen2.5-coder:7b | Codebase understanding, pattern finding |
| `tech-comparator` | phi4:14b | Technology comparison, tradeoff analysis |

**Subagent Tool Access:**
Subagents can use any tool registered in the skill_registry. The `_execute_subagent_tool()` method first tries skill_registry.execute() for comprehensive tool access, then falls back to local subagent tools.

**VRAM Budget (12GB constraint):**
```python
VRAM_BUDGET_GB = 12
VRAM_HEADROOM_GB = 2  # Reserved for KV cache
# Available: 10GB for models

MODEL_VRAM_ESTIMATES = {
    "phi3:mini": 2.0,           # Router
    "phi4:14b": 8.0,            # Primary agent
    "qwen2.5-coder:7b": 4.5,    # Code specialist
}
```

**Agent Definition Format:**
```yaml
---
name: research-summarizer
model: phi4:14b
purpose: Conduct deep research and synthesize findings
tools:                          # Tools this agent can use
  - web_search
  - deep_research
  - fetch_url
context_requirements: [research_platform, task_description]
output_format: markdown
max_tokens: 3000
temperature: 0.3
---
# System prompt follows...
```

### Telemetry & Dashboard

**`telemetry.py`** - Execution tracing:
- `Trace` dataclass (lines 191-469): Complete request lifecycle
- `LLMCall` (lines 134-188): Single LLM API call record
- `ToolCallTrace` (lines 60-130): Tool execution record
- `TelemetryCollector` (lines 471-1032): Storage and retrieval

**`dashboard.py`** - WebSocket server (port 8766):
- `DashboardServer` (lines 119-1388): Event emission, command handling
- 50+ event types for real-time visualization
- Commands: send_message, emergency_stop, memory inspection, session control

### Configuration

**`config.py`** - All settings:
| Setting | Default | Purpose |
|---------|---------|---------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `MODEL` | `phi4:14b` | Primary agent model |
| `ROUTER_MODEL` | `phi3:mini` | Intent classification |
| `SKILL_EXECUTION_MODEL` | `phi4:14b` | Tool execution |
| `SEMANTIC_BYPASS_THRESHOLD` | 0.85 | Skip router, direct to skill |
| `SEMANTIC_CONFIRM_THRESHOLD` | 0.45 | Router confirms candidates |
| `MAX_TOOL_ITERATIONS` | 5 | Max tool calls per turn |
| `MAX_CONTEXT_MESSAGES` | 20 | Context window size |

**`logger.py`** - Dual-handler logging:
- File: DEBUG level to `data/logs/workshop_*.log`
- Console: WARNING level (clean user experience)

## WebSocket Protocol

### Dashboard Commands (Client → Server)

```typescript
{ action: 'send_message', message: string }
{ action: 'emergency_stop' }
{ action: 'get_current_tasks' }
{ action: 'get_research_library' }
{ action: 'get_subagent_history' }
{ action: 'get_snapshot_detail', snapshot_id: string }
{ action: 'search_memory', query: string, limit: number }
{ action: 'get_memory_facts' }
{ action: 'get_session_info' }
{ action: 'start_new_session', mode: string }
{ action: 'detect_stale_state' }
```

### Dashboard Events (Server → Client)

**Session Events:**
- `session_started`, `session_ended`, `session_resumed`
- `stale_state_detected`, `stale_state_cleared`

**Processing Events:**
- `user_input`, `assistant_response`
- `context_loading`, `context_loaded`
- `intent_detected`, `skill_matched`

**Tool Events:**
- `tool_calling`, `tool_result`, `tool_error`

**LLM Events:**
- `llm_calling`, `llm_streaming`, `llm_complete`
- `llm_calling` event includes: `messages` array, `system_prompt`, `message_count`, `system_prompt_length`

**Task Events:**
- `task_list_updated`, `task_started`, `task_completed`

**Subagent Events:**
- `subagent_spawning`, `subagent_model_swap`
- `subagent_executing`, `subagent_complete`
- `context_snapshot`

## Common Patterns

### Adding a New Skill

1. Create directory: `~/.workshop/Skills/{SkillName}/`
2. Create `SKILL.md` with metadata:
```markdown
---
name: MySkill
priority: MEDIUM
intent_patterns:
  - "do something"
  - "perform action"
---
# Description of skill
```
3. Create `system.md` with skill-specific prompt
4. Add tools in `tools/` directory:
```python
# tools/my_tool.py
TOOL_DESCRIPTION = "What this tool does"
TOOL_SIGNATURE = "my_tool(arg: str) -> str"

async def my_tool(arg: str, _deps: dict = None) -> str:
    memory = _deps.get('memory')  # Access injected dependencies
    return "result"
```

### Adding a New Pattern

1. Create directory: `~/.workshop/patterns/{category}/{pattern_name}/`
2. Create `system.md` with pattern prompt
3. Add to `_registry.json` (optional, for triggers):
```json
{
  "patterns": {
    "my_pattern": {
      "category": "transform",
      "purpose": "Transform input in specific way",
      "triggers": ["transform this", "convert to"],
      "model_preference": "phi4:14b"
    }
  }
}
```
4. Optionally add to `pattern_tools.py` for skill integration

### Adding Store State

1. Add type to `telemetry.py` or appropriate dataclass
2. Add field to `Trace`, `Session`, or `Task` dataclass
3. Update `to_dict()` and `from_dict()` methods
4. Handle in WebSocket if dashboard needs it

## Data Flow Summary

### Request Lifecycle

```
User Input
    ↓
main.py: Workshop.process_input()
    ├─ memory.add_message("user", text)
    ├─ agent.chat(text)
    │      ↓
    │  agent.py: _two_stage_chat()
    │      ├─ STAGE 0: Context injection (if needed)
    │      │      ├─ context_manager.get_context()
    │      │      └─ enriched_input = input + [Current Context]
    │      ├─ STAGE 1: Routing
    │      │      └─ prompt_router.route(enriched_input)
    │      ├─ STAGE 1.75: Auto-create tasks (if none exist)
    │      ├─ STAGE 2: Skill execution
    │      │      ├─ skill_executor.execute(skill_name, input)
    │      │      └─ For each tool_call: registry.execute(tool, args)
    │      ├─ STAGE 2.5: Task advancement
    │      │      └─ Mark current task completed, next in_progress
    │      └─ STAGE 3: Auto-continue (if pending tasks)
    │             └─ Recursively call _two_stage_chat("Continue.")
    │      ↓
    └─ memory.add_message("assistant", response)
    ↓
Response to User
```

### Tool Execution Flow

```
LLM Response with <tool_call>...</tool_call>
    ↓
skill_executor._parse_tool_calls()
    ↓
For each tool call:
    ├─ skill_registry.execute(tool_name, args)
    │      ├─ Normalize arguments (aliases)
    │      ├─ Find skill that owns tool
    │      ├─ Call skill.execute_tool(name, args)
    │      │      └─ Inject _deps (memory, config, etc.)
    │      └─ Return result
    ├─ Record in telemetry trace
    └─ Emit dashboard event
    ↓
Append tool results to conversation
    ↓
Continue LLM loop (if more tool calls)
```

## Design Tokens

| Token | Value | Usage |
|-------|-------|-------|
| Primary Model | `claude-code` (CLI) | Main agent reasoning (via subscription) |
| Router Model | Semantic + Claude fallback | Intent classification (Phase 3: no Ollama required) |
| Embedding Model | `all-MiniLM-L6-v2` | Semantic routing |
| Claude Binary | `~/.local/bin/claude` | Explicit path for security |
| Dashboard Port | 8766 | WebSocket server (React dashboard) |
| Trace Dashboard Port | 5001 | Flask trace visualization |
| Context Cache | 5 minutes | Telos reload interval |
| Max Tool Iterations | 5 | Per-turn limit |

## Build & Development

```bash
# Install dependencies
pip install -r requirements.txt

# === Claude Code Setup (Required) ===
# Install Claude Code CLI
curl -fsSL https://claude.ai/install.sh | sh

# Authenticate with your subscription (opens browser)
claude login

# Verify subscription status
claude /status

# IMPORTANT: Remove any API key from environment to use subscription
# Check: echo $ANTHROPIC_API_KEY
# If set, remove from ~/.bashrc or ~/.zshrc

# === Running Workshop ===
# Run in text mode
python main.py --mode text

# Run with dashboard
python main.py --mode text --dashboard

# Run in voice mode (Phase 2)
python main.py --mode voice

# Development dashboard
cd dashboard && npm run dev
```

## Key Design Decisions

1. **Claude Code as Reasoning Engine**: Uses Claude Code CLI (`claude -p`) via subprocess for superior reasoning quality. Workshop maintains control over tools by disabling Claude's native tools with `--tools ''`.

2. **Semantic-First Routing (Phase 3)**: Intent classification uses semantic embeddings first. High/medium confidence matches route directly to skills. Only low-confidence queries use Claude Code for clarification. Ollama is no longer required.

3. **Non-Native Tool Calling**: Uses `<tool_call>` XML format. Claude outputs tool requests in this format, Workshop parses and executes them locally, then feeds results back. This keeps tool execution in your environment, not Claude's sandbox.

4. **Session Isolation**: Tasks bound to sessions, prevents stale state leakage.

5. **Work Evidence Validation**: Anti-hallucination check requires actual tool execution before marking tasks complete.

6. **Context Layering**: Five independent sources (automatic, personal, task, research, conversational) assembled into system prompt for Claude.

7. **Singleton Managers**: TaskManager, SessionManager, TelosManager use singleton pattern for global access.

8. **Tool-First Approach**: All skill system prompts explicitly forbid generating responses from general knowledge - the LLM must use tools to gather information before making claims. This prevents hallucination.

9. **Semantic Routing with Utterances**: Skills define semantic utterances in SKILL.md that are embedded and used for high-confidence routing bypass (threshold 0.85).

10. **Specialized Subagents**: Subagent AGENT.md files become system prompts for separate `claude -p` calls. Fresh context window per delegation prevents context pollution.

11. **Batch Browser Fetching**: Deep research uses `fetch_urls_for_research()` to fetch multiple URLs with a SINGLE browser instance (via Crawl4AI's `fetch_many()`). Failed fetches retry with exponential backoff.

12. **Subscription-Based Billing**: Uses Claude Pro/Max subscription via CLI (no API key). Fixed monthly cost instead of per-token. Monitor usage with `/status` command.

13. **Hook-Based Context Injection**: All dynamic context (capabilities, Telos profile/goals, tasks) is injected via SESSION_START hooks, not hard-coded in agent.py. This separation allows easy extension without modifying core code. `_build_system_message()` only extracts user_profile for templates and captures telemetry - it does NOT inject context directly.

## Dashboard Features

### LLM Context Visibility
The dashboard's `EventItem` component shows LLM context preview for `llm_calling` events:
- Message count indicator
- System prompt character length
- "Context logged" indicator when full messages are captured
- User message preview (first 80 chars)

The `EventModal` component provides full LLM context view with:
- Scrollable message list
- Color-coded roles (system, user, assistant)
- Expandable system prompt
- Character counts per message

## Changelog

### January 4, 2026 - Research Persistence & Idempotency

**New Features:**
- Added `save_research_output` tool: Scoped write to `data/research/` only
- Added `archive_source` tool: Archive URLs to SQLite + ChromaDB
- Added `index_findings` tool: Index content to ChromaDB with auto-chunking
- All three tools are **idempotent** - duplicate content is detected and skipped

**Bug Fixes:**
- Fixed research context loss between queries (`_research_state.py` now auto-loads from disk)
- Fixed `continue_research.py` to fallback to disk when in-memory state is lost
- Fixed `save_research_output` argument mismatch (now accepts both `filename` and `path`)
- Added `include_metadata` parameter to `memory.search_memories()` for rich results

**Architecture Changes:**
- Research state persisted to `~/.workshop/research/_active.json`
- Content index for idempotency: `data/research/_content_index.json`
- ChromaDB categories added: `research_output`, `research_sources`, `research:{topic}`

**Files Modified:**
- `.workshop/Skills/Research/tools/_research_state.py` - Auto-load from disk
- `.workshop/Skills/Research/tools/continue_research.py` - Disk fallback
- `.workshop/Skills/Research/tools/save_research_output.py` - NEW + idempotency
- `.workshop/Skills/Research/tools/archive_source.py` - NEW + idempotency
- `.workshop/Skills/Research/tools/index_findings.py` - NEW + idempotency
- `.workshop/Skills/Research/SKILL.md` - Updated tool signatures
- `memory.py` - Added `include_metadata` to search_memories()

### January 4, 2026 - Hook System Implementation

**New Features:**
- Added `hooks.py`: Complete lifecycle hook system for extensible behavior
- `SESSION_START` hook: Fires at conversation start for context hydration
- `POST_TOOL_USE` hook: Fires after each tool execution for persistence/reactions
- Built-in handlers: Telos hydration, Task hydration, Research auto-persist
- Terminal visualization: Real-time display of hook execution with status indicators

**Terminal Output Example:**
```
🔗 [HOOKS] ━━━ SESSION_START ━━━ (3 handlers)
   ├─ [05] hydrate_capabilities... ✓ (+1 sections, [10 skills, 55 tools])
   ├─ [10] hydrate_telos_context... ✓ (+4 sections, [profile, goals, mission, project:workshop])
   ├─ [20] hydrate_active_tasks... ✓ ([2 active, 5 pending])
   └─ Total: 6 sections, 13000 chars injected
```

**Architecture:**
- Singleton `HookManager` with priority-ordered handler execution
- Error isolation: One handler failing doesn't break the chain
- `HookContext` dataclass carries state through handler chain
- Handlers can modify context, add system prompt sections, or signal chain stop

**Files Created:**
- `hooks.py` - Core hook system (~600 lines)
- `tests/test_hooks.py` - Comprehensive unit tests

**Files Modified:**
- `agent.py` - Added SESSION_START hook integration in `_two_stage_chat()`
- `skill_executor.py` - Added POST_TOOL_USE hook integration after tool execution

**Usage:**
```python
from hooks import get_hook_manager, HookType, HookContext

async def my_handler(ctx: HookContext) -> HookContext:
    ctx.add_system_context("My Section", "Content")
    return ctx

get_hook_manager().register(HookType.SESSION_START, my_handler, priority=30)
```

### January 5, 2026 - Capabilities Manifest & Context Deduplication

**New Features:**
- Added `hydrate_capabilities` hook (priority 5): Injects available tools/skills into context
- Added `generate_capabilities_manifest()` to SkillRegistry: Full markdown manifest
- Added `get_capabilities_summary()` to SkillRegistry: Condensed version for prompts
- Auto-generates `~/.workshop/CAPABILITIES.md` on startup

**Architecture Cleanup:**
- Removed duplicate context injection from `_build_system_message()`
- All dynamic context (capabilities, Telos, tasks) now flows through SESSION_START hooks
- `_build_system_message()` only extracts user_profile and captures telemetry
- Eliminates ~8KB of redundant context that was being injected twice

**Files Modified:**
- `skill_registry.py` - Added manifest generation methods
- `hooks.py` - Added `hydrate_capabilities` handler and cache
- `agent.py` - Removed Telos/Task context injection (now via hooks only)
- `LLM_CONTEXT.md` - Updated documentation

### January 6, 2026 - Voice Mode Fixes (Fix 1 & Fix 3)

**Problem Addressed:**
Voice mode was broken for skill execution - skills returned markdown-formatted responses that were sent directly to TTS, resulting in poor audio quality. Additionally, users had no feedback during long operations (5+ minutes of silence).

**Fix 1: SkillExecutor Voice Mode Passthrough**

The SkillExecutor was ignoring `voice_mode` entirely. Now:
- `voice_mode` is passed from main.py → Agent → SkillExecutor
- Voice formatting requirements are injected into every skill's system prompt
- Post-processing sanitizes any remaining markdown before TTS

**Key Components:**
```python
# skill_executor.py
def _get_voice_requirements(self) -> str:
    """Returns TTS formatting instructions appended to skill prompts."""

def _sanitize_for_voice(self, text: str) -> str:
    """Removes markdown artifacts: **bold**, headers, lists, URLs, etc."""

def _prepare_final_response(self, content: str) -> str:
    """Combines tool artifact cleanup with voice sanitization."""
```

**Fix 3: Real-time Haiku Progress Updates**

New `HaikuProgressManager` provides contextual voice feedback during long operations:
- Uses Claude Haiku to generate natural progress summaries
- Rate-limited (8 second minimum between updates)
- Tracks tool execution context for informative updates
- Graceful fallback to simple messages if Haiku unavailable

**Key Components:**
```python
# haiku_progress.py
class HaikuProgressManager:
    """Manages real-time progress updates using Haiku."""

    async def on_tool_start(tool_name, args):
        """Called before tool execution - may generate starting update."""

    async def on_tool_complete(tool_name, result, duration_ms):
        """Called after tool execution - generates completion update."""
```

**Integration Flow:**
```
User speaks → Workshop processes → Tool starts
                                      ↓
                               HaikuProgressManager.on_tool_start()
                                      ↓
                               [Optional: "Searching the web now."]
                                      ↓
                               Tool executes (may take minutes)
                                      ↓
                               HaikuProgressManager.on_tool_complete()
                                      ↓
                               [Haiku generates contextual update]
                                      ↓
                               TTS speaks: "Found 3 sources about..."
```

**Files Created:**
- `haiku_progress.py` - HaikuProgressManager (~450 lines)

**Files Modified:**
- `skill_executor.py` - Added voice_mode, tts_callback, progress manager integration
- `agent.py` - Added tts_callback parameter, updated get_skill_executor
- `main.py` - Set tts_callback on agent after voice stack initialization

**Logging:**
All voice mode operations are logged with `[VOICE]` prefix:
```
[VOICE] Progress session started for skill: Research
[VOICE] Tool started: web_search
[VOICE] Tool completed: web_search (1500ms)
[VOICE] Update #1: "Found several sources about agent orchestration..." (350ms)
[VOICE] Progress session complete: 5 tools, 3 updates, 45s
[VOICE] Sanitized response for TTS (1200 chars)
```

**Configuration:**
- `MIN_UPDATE_INTERVAL_SEC = 8` - Minimum seconds between updates
- Progress updates only for certain tools: `web_search`, `deep_research`, `fetch_url`, etc.
- Voice requirements injected only when `voice_mode=True`

### January 6, 2026 - Argument Aliasing & Error Logging Fixes (Fix 4 & 5)

**Fix 4: Signature-Aware Argument Aliasing**

**Problem:** The argument aliasing system blindly converted `file_path` → `path`, breaking functions like `get_related_files()` that explicitly require `file_path` as the parameter name.

**Solution:** Implemented signature-aware argument aliasing that inspects the target function's signature before applying aliases.

**Files Modified:**
- `skill_registry.py`:
  - Added `_normalize_args_for_function()` method - Inspects function signature before aliasing
  - Updated `execute()` method - Uses signature-aware normalization
  - Original `_normalize_args()` method preserved for backwards compatibility

**Key Behavior:**
- Aliases (e.g., `directory` → `path`) only applied when target function accepts the aliased name
- Original parameter names preserved when function signature expects them
- Logged warnings when arguments don't match any valid parameter

**Fix 5: Improved Callback Error Logging**

**Problem:** Speech and wake callback errors in `wake_pipeline.py` logged empty messages when exceptions had no string representation.

**Solution:** Enhanced error logging to include exception type and use proper logging for tracebacks.

**Files Modified:**
- `wake_pipeline.py`:
  - Updated wake callback error handler (lines 144-147)
  - Updated speech callback error handler (lines 213-216)

**Changes:**
```python
# Before:
log.error(f"Speech callback error: {e}")
import traceback
traceback.print_exc()

# After:
log.error(f"Speech callback error: {type(e).__name__}: {e}")
log.debug("Speech callback traceback:", exc_info=True)
```

**Benefits:**
- Exception type always visible even if message is empty
- Traceback uses proper logging system (respects log levels, goes to log file)
- Cleaner output at runtime (traceback only shown at DEBUG level)

### January 6, 2026 - Hybrid Voice+Text Input Mode

**Feature:** Allow text input while in voice mode using `/text` prefix.

**Problem:** Voice mode was exclusively voice-driven. Users couldn't type text input, which is inconvenient when:
- In a noisy environment
- Need to input precise text (URLs, code snippets)
- Wake word detection is unreliable

**Solution:** Run voice pipeline and text input loop concurrently using asyncio.

**Usage:**
```
# In voice mode, type:
/text what is the weather today
/text search for python tutorials
exit  # to quit
```

**Files Modified:**
- `main.py`:
  - Added `_text_input_loop()` method - Async loop reading stdin for `/text` commands
  - Added `_process_text_in_voice_mode()` method - Process text and speak response via TTS
  - Updated `run_voice_mode_phase2()` - Runs voice and text loops concurrently with `asyncio.wait()`
- `terminal_ui.py`:
  - Updated `voice_mode_start()` - Shows hint for `/text` command

**Architecture:**
```
run_voice_mode_phase2()
    │
    ├── voice_task (executor) ─── wake_pipeline.run() ─── Wake word detection
    │
    └── text_task (async) ─────── _text_input_loop() ─── /text command parsing
                                        │
                                        └── _process_text_in_voice_mode()
                                              │
                                              ├── process_input() (Agent)
                                              └── piper.speak() (TTS)
```

**Behavior:**
- Both input methods work simultaneously
- Text input pauses wake word detection during processing
- Response is spoken via TTS (same as voice input)
- Exit via "exit" command or Ctrl+C terminates both loops

### January 7, 2026 - Claude Bridge Tool Call Fix & Logging

**Problem 1: Silent Logging**

Modules `claude_bridge.py` and `hooks.py` used `logging.getLogger(__name__)` which created loggers named `claude_bridge` and `hooks` respectively. These were NOT children of the `workshop` logger, so they didn't inherit the file handler - logs went nowhere.

**Solution:** Changed to use Workshop's logger system.

**Files Modified:**
- `claude_bridge.py:35-37`: Changed to `from logger import get_logger; log = get_logger("claude_bridge")`
- `hooks.py:35-37`: Changed to `from logger import get_logger; logger = get_logger("hooks")`

**Problem 2: Tool Calls Extracted But Not Used**

The `claude_bridge.query()` method correctly extracted `<tool_call>` tags from Claude's response, but then **cleaned the content** (removing those tags) before returning. The `skill_executor` received the cleaned content and tried to re-parse it - finding nothing.

**Root Cause Flow:**
```
claude_bridge.query():
  1. Get response from Claude with <tool_call> tags
  2. Extract tool_calls → [{"tool": "parallel_dispatch", ...}]  ✓
  3. Clean content → removes <tool_call> tags
  4. Return {"content": cleaned, "tool_calls": [...]}

skill_executor._call_llm_with_tools():
  1. result = await bridge.query(...)
  2. content = result.get("content")  ← cleaned, no tags
  3. tool_calls = self._parse_tool_calls(content)  ← finds nothing!
  4. result.get("tool_calls") was IGNORED
```

**Solution:** Updated `skill_executor.py` to use pre-extracted tool calls from the bridge response.

**Files Modified:**
- `skill_executor.py:473-474`: Added `bridge_tool_calls = []` tracking variable
- `skill_executor.py:486-488`: Capture `result.get("tool_calls", [])` from Claude bridge
- `skill_executor.py:550-561`: Use `bridge_tool_calls` if available, convert to `ToolCall` objects

**Key Code Change:**
```python
# Before (broken):
tool_calls = self._parse_tool_calls(content)  # content is cleaned!

# After (fixed):
if bridge_tool_calls:
    # Convert bridge format to ToolCall objects
    tool_calls = [ToolCall(name=tc.get("tool"), arguments=tc.get("args", {}))
                  for tc in bridge_tool_calls if tc.get("tool")]
else:
    tool_calls = self._parse_tool_calls(content)  # Fallback for Ollama
```

**Diagnostic Logging Added:**
- `claude_bridge.py:452`: Logs `disable_native_tools` parameter value
- `claude_bridge.py:460-463`: Logs full CLI command being executed
- `claude_bridge.py:466-468`: Logs prompt and system prompt sizes
- `skill_executor.py:488`: Logs tool calls received from bridge
- `skill_executor.py:558`: Logs when using bridge tool calls

**Log Output Example:**
```
[CLAUDE_BRIDGE] disable_native_tools parameter = True
[CLAUDE_BRIDGE] ⚡ --tools '' flag APPLIED - Claude native tools DISABLED
[CLAUDE_BRIDGE] ========== FULL COMMAND ==========
[CLAUDE_BRIDGE] cmd = ['/home/user/.local/bin/claude', '-p', '-', '--output-format', 'json', ...]
[CLAUDE_BRIDGE] '--tools' in cmd: True
[CLAUDE_BRIDGE] Prompt length: 3021 chars
[CLAUDE_BRIDGE] System prompt length: 12366 chars
[TOOL_PARSE] Extracted tool call (tool_call tag): parallel_dispatch
[CLAUDE_BRIDGE] Received 1 tool calls from bridge: ['parallel_dispatch']
[SKILL_EXECUTOR] Using 1 tool calls from Claude bridge
```

### January 7, 2026 - Subagent Tool Call Fix & Voice Progress

**Problem 1: Subagent Tool Calls Extracted But Not Used**

The same bug that affected `skill_executor.py` also affected `subagent_manager.py`. The `_call_subagent_llm()` method called `claude_bridge.query()` which extracts tool calls and cleans them from content, but the extracted tool calls were discarded:

```python
# Before (broken):
content = result.get("content", "")  # Gets CLEANED content
return content, duration_ms, 0, 0    # tool_calls discarded!

# Then in _execute_subagent:
tool_calls = _parse_tool_calls(output)  # Parses cleaned content - finds NOTHING!
```

**Problem 2: No Tool-Calling Format Instructions**

Subagent system prompts (from `~/.workshop/agents/*.md`) listed available tools but didn't tell Claude HOW to call them. Claude would respond with "I'll search for..." instead of actually emitting `<tool_call>` blocks.

**Problem 3: Tools List Not Loaded from YAML**

The `AgentDefinition` dataclass didn't have a `tools` field, so the `tools:` list in agent YAML frontmatter was being ignored.

**Solution:**

1. **Fixed tool call passthrough** - `_call_subagent_llm()` now returns 5 values including `bridge_tool_calls`
2. **Added tool format instructions** - New `_build_tool_instructions()` method generates explicit `<tool_call>` format instructions
3. **Load tools from YAML** - Added `tools: List[str]` to `AgentDefinition` dataclass

**Files Modified:**
- `subagent_manager.py:1314-1341`: Return `bridge_tool_calls` from `_call_subagent_llm()`
- `subagent_manager.py:1505-1511`: Use pre-extracted tool_calls in `_execute_subagent()`
- `subagent_manager.py:921-979`: Added `_build_tool_instructions()` method
- `subagent_manager.py:360`: Added `tools: List[str]` field to `AgentDefinition`
- `subagent_manager.py:914`: Load tools list from YAML in `load_agent_definition()`

**Log Output Before (broken):**
```
Found 0 <tool_call> blocks in 105 char response
Subagent complete after 1 iterations, 0 tool calls
```

**Log Output After (fixed):**
```
[SUBAGENT_TOOLS] Agent web-researcher has 6 tools: ['web_search', 'deep_research', ...]
[SUBAGENT_LLM] Claude bridge extracted 4 tool calls: ['deep_research', 'web_search', ...]
[SUBAGENT_EXEC] Using 4 pre-extracted tool calls from Claude bridge
Subagent complete after 5 iterations, 18 tool calls
```

### January 7, 2026 - Subagent Haiku Progress Updates

**Problem:** Subagents provided voice updates only when completing or running out of iterations. Users had no feedback during the 5+ minutes of subagent tool execution.

**Solution:** Integrated `HaikuProgressManager` into `SubagentManager` using the same pattern as `SkillExecutor`.

**Architecture:**
```
spawn_subagent() / parallel_dispatch()
    │
    ├── Extract voice_mode, tts_callback from _deps
    │
    └── SubagentManager(voice_mode=True, tts_callback=piper.speak)
            │
            ├── _progress_manager = HaikuProgressManager(...)
            │
            └── _execute_subagent()
                    │
                    ├── start_session(query, skill_name="subagent:web-researcher")
                    │
                    ├── For each tool:
                    │   ├── on_tool_start(tool_name, args)
                    │   ├── Execute tool
                    │   └── on_tool_complete(tool_name, result, duration_ms)
                    │
                    └── get_session_summary() on completion
```

**Files Modified:**
- `subagent_manager.py:38-44`: Import `HaikuProgressManager`
- `subagent_manager.py:674-712`: Add `voice_mode`, `tts_callback` params; init progress manager
- `subagent_manager.py:1568-1578`: Call `start_session()` before execution loop
- `subagent_manager.py:1683-1702`: Call `on_tool_start()` and `on_tool_complete()` around tool execution
- `subagent_manager.py:1650-1656`: Call `get_session_summary()` on completion
- `.workshop/Skills/Orchestration/tools/spawn_subagent.py:182-194`: Pass voice params to SubagentManager
- `.workshop/Skills/Orchestration/tools/parallel_dispatch.py:198-210`: Pass voice params to SubagentManager
- `main.py:318-323`: Inject `voice_mode` and `tts_callback` into skill registry dependencies

**Log Output Example:**
```
[SUBAGENT_VOICE] Voice mode enabled for subagent progress updates
[SUBAGENT_VOICE] Progress session started for web-researcher
[HAIKU_PROGRESS] Tool started: web_search
[HAIKU_PROGRESS] Update #1: "Found several sources about agent orchestration..." (350ms)
[HAIKU_PROGRESS] Tool completed: web_search (3500ms)
[SUBAGENT_VOICE] Progress session complete: 5 tools, 3 updates, 45s
```

**Voice Flow:**
1. User speaks query in voice mode
2. Orchestrator dispatches subagent via `parallel_dispatch`
3. Subagent executes tools (web_search, fetch_url, etc.)
4. After each tool completes, Haiku generates a natural progress update
5. TTS speaks the update: "Found several sources about agent orchestration..."
6. User hears real-time progress instead of silence

### January 7, 2026 - Comprehensive Trace Logging & Dashboard

**Problem:** Subagent dispatch latency was extreme (38-104 seconds per LLM call) and debugging was difficult because:
1. Context sizes weren't tracked - prompts grew to 115KB+ without warning
2. Tool arguments and results were truncated in logs
3. No visibility into what context each agent received/produced
4. No real-time monitoring capability

**Analysis Findings:**
| LLM Call | Response Time | Content Size |
|----------|---------------|--------------|
| codebase-analyst iter 1 | 38.6 seconds | 105 chars |
| web-researcher iter 1 | 47.5 seconds | 255 chars |
| codebase-analyst iter 2 | **104.2 seconds** | 106 chars |

Root causes: Context accumulation (tool results appended each iteration), parallel subagents competing, Haiku progress updates adding overhead.

**Solution:** Added comprehensive trace logging system and Flask dashboard.

**Config Settings (`config.py`):**
```python
# === Trace Logging Settings ===
TRACE_LOGGING_ENABLED = os.getenv("WORKSHOP_TRACE_LOGGING", "true").lower() == "true"
TRACE_LOG_FULL_CONTEXT = os.getenv("WORKSHOP_TRACE_FULL_CONTEXT", "true").lower() == "true"
TRACE_LOG_TOOL_RESULTS = os.getenv("WORKSHOP_TRACE_TOOL_RESULTS", "true").lower() == "true"
TRACE_MAX_CONTENT_LENGTH = int(os.getenv("WORKSHOP_TRACE_MAX_LENGTH", "5000"))
```

**Trace Logging Functions (in `subagent_manager.py` and `skill_executor.py`):**
- `_trace_log(tag, content)` - Consistent tag-based logging with truncation
- `_trace_context_sizes(messages, iteration)` - Size breakdown with warnings at 50KB/100KB
- `_trace_messages_content(messages, label)` - Full message content logging
- `_trace_tool_execution(tool_name, args, result, duration_ms)` - Tool args and results

**Log Output Example:**
```
[TRACE:CONTEXT_SIZE] Iteration 1 - Total: 15,234 chars
[TRACE:CONTEXT_SIZE] System prompt: 7,035 chars
[TRACE:CONTEXT_SIZE] Messages (4):
  msg[0] system: 7,035 chars
  msg[1] user: 500 chars
  msg[2] assistant: 105 chars
  msg[3] user: 7,594 chars
[TRACE:CONTEXT_SIZE] ⚠️ LARGE CONTEXT WARNING: 115,423 chars
[TRACE:LLM_RESPONSE] Skill=Orchestration Iter=1 Duration=15386ms Chars=280
[TRACE:TOOL_EXEC] web_search (1523ms)
[TRACE:TOOL_ARGS:web_search] {"query": "LangGraph agent orchestration"}
[TRACE:TOOL_RESULT:web_search] Web search results for...
[TRACE:CONTEXT_OUT:web-researcher:ITER1] I'll search for information about...
```

**Flask Trace Dashboard (`trace_dashboard.py`):**
Real-time web UI at `http://localhost:5001` providing:
- Live log parsing with auto-refresh (1 second polling)
- Filtering by type: LLM, Tools, Context, Warnings, Subagents
- Summary metrics: LLM calls, avg latency, tool calls, warnings
- Color-coded entries by type and severity
- Large context warnings highlighted
- Auto-scroll with manual override

**Usage:**
```bash
# Terminal 1: Run trace dashboard
python trace_dashboard.py

# Terminal 2: Run Workshop
python main.py --mode text
```

**Files Created:**
- `trace_dashboard.py` - Flask web dashboard (~400 lines)

**Files Modified:**
- `config.py:258-263` - Added trace logging config settings
- `subagent_manager.py:57-139` - Added trace logging functions
- `subagent_manager.py:1668-1671` - Context size tracking before LLM call
- `subagent_manager.py:1698-1699` - LLM response logging
- `subagent_manager.py:1804-1805` - Tool execution logging
- `skill_executor.py:54-115` - Added trace logging functions (mirrors subagent_manager)
- `skill_executor.py:525-527` - Context size tracking before LLM call
- `skill_executor.py:614-616` - LLM response logging
- `skill_executor.py:712-713,729` - Tool execution logging
- `requirements.txt:55-56` - Added Flask dependency

**Environment Variables:**
| Variable | Default | Purpose |
|----------|---------|---------|
| `WORKSHOP_TRACE_LOGGING` | `true` | Master switch for trace logging |
| `WORKSHOP_TRACE_FULL_CONTEXT` | `true` | Log full message contents |
| `WORKSHOP_TRACE_TOOL_RESULTS` | `true` | Log full tool results |
| `WORKSHOP_TRACE_MAX_LENGTH` | `5000` | Truncation limit for logged content |
