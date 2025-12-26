# Master Project Specification: Workshop Phase 3 - Context Intelligence

**Generated:** 2025-12-14
**Target:** Intelligent context assembly and workflow detection
**Status:** Planning (Phase 2 complete)

---

## 1. Project Purpose

**Problem:** Phase 2 achieved natural conversation flow, but Workshop still requires manual context specification. When you ask "Why is cell 2 reading high?", Workshop doesn't know you just opened `battery_guardian.ino`, doesn't see your recent edits to voltage calibration, and can't automatically pull in datasheet information for the BQ76940 chip. Every query requires explicit "read this file, check that config" preambles.

**Who:** Christian doing rapid prototyping across multiple projects (Arduino firmware, React Native apps, Python tools) where context switches are frequent and manual file specification breaks flow.

**Value:** Automatic context assembly eliminates 30-50% of query overhead. Instead of "Read battery_guardian.ino and battery_config.h then explain why cell 2 voltage is wrong", you say "Why is cell 2 reading high?" and Workshop already knows what files matter, what you changed recently, and what external knowledge (datasheets, libraries) to fetch.

**Long-term Vision:** Replace Claude Code by adding visual constructs, tool integration, and context intelligence that Claude Code cannot provide. Phase 3 is the context intelligence foundation.

---

## 2. Essential Functionality

### Core Capabilities (Phase 3 MVP)

1. **File System Monitoring** - Track which files are opened, edited, saved in registered project directories
2. **Automatic Context Assembly** - Build relevant context graph when user asks a question
3. **Workflow Detection** - Recognize debugging, feature development, configuration editing patterns
4. **Context Visualization** - Show user what Workshop "sees" and knows about current session
5. **Smart Context Retrieval** - LLM can request additional context via tools when needed
6. **Persistent Context Relationships** - Remember file relationships, dependencies, common co-edits

### Context Strategy: Instructions vs Info

**Critical Context (System Prompt - "Instructions"):**
- Currently open file(s)
- User's explicit project focus ("working on Battery Guardian")
- Established constraints (hardware limitations, API versions)
- User preferences (coding style, libraries to use/avoid)

**Retrievable Context (Tool Calls - "Info"):**
- File contents when needed
- Related files (imports, dependencies)
- Recent edit history
- External knowledge (datasheets, documentation)
- Previous solutions to similar problems

**Why this split?**
- System prompt stays focused, doesn't blow up token count
- LLM decides what info it needs, retrieves on-demand
- Enables reasoning about "what context do I need?" before fetching

---

## 3. Scope Boundaries

### NOW (Phase 3 MVP)

- ✅ Monitor configured project directories for file changes
- ✅ Detect which file is currently active (via heuristics: recent edit, multiple saves)
- ✅ Build context graph: file → imports → dependencies → related files
- ✅ Workflow detection: debugging (repeated compile/upload), feature dev (new files), config editing
- ✅ Context visualization UI (show active files, detected workflow, available context)
- ✅ LLM tools for context retrieval (`get_file_content`, `search_related_files`, `find_definition`)
- ✅ Persistent context storage (file relationships in ChromaDB, session state in SQLite)
- ✅ Project registration (add paths to `config.py`, automatic initialization)

### NOT (Out of Scope - Phase 3)

- ❌ VSCode extension (use file system monitoring instead)
- ❌ Git integration (no branch/commit awareness yet)
- ❌ Cursor/selection tracking (no IDE integration)
- ❌ Proactive interruptions (wait for user to speak first)
- ❌ Cross-project context by default (only on explicit request)

### NEXT (Phase 4+)

- 🔮 Git awareness (detect branch, recent commits, diff analysis)
- 🔮 Proactive suggestions (interrupt with "I noticed..." prompts)
- 🔮 Cross-project context (link Guardian firmware with mobile app code)
- 🔮 External knowledge caching (store fetched datasheets, docs for reuse)
- 🔮 Learning user patterns (remember common debugging workflows)

---

## 4. Technical Context

**Current State (End of Phase 2):**
- Real-time voice pipeline working (wake word → VAD → Whisper → Agent → Piper)
- Memory system: SQLite (conversations, notes) + ChromaDB (semantic search, RAG)
- Tool registry: 12 base tools + 9 construct tools + 10 project tools
- Agent uses Ollama (qwen3:8b) with tool calling

**Platform:**
- OS: Ubuntu (Lubuntu)
- IDE: VSCodium (VSCode fork)
- Projects: Arduino (.ino), Python (.py), React Native (.js/.jsx), config files (.yaml/.json)
- No Windows/Mac support needed

**Dependencies:**
- File system monitoring: `watchdog` library
- Context graph storage: ChromaDB (already integrated)
- Session state: SQLite (already integrated)
- LLM: Ollama with tool calling (already working)

---

## 5. Workflow Details

### Workflow 1: Project Registration & Monitoring

**Goal:** User adds project path, Workshop starts monitoring automatically

**Steps:**
1. User edits `config.py` and adds project path:
   ```python
   MONITORED_PROJECTS = [
       "/home/bron/Projects/BatteryGuardian",
       "/home/bron/Projects/RobotArm",
       "/home/bron/FlyingTiger/Workshop_Assistant_Dev"
   ]
   ```
2. Workshop initializes file watchers for each project
3. Workshop scans project structure, identifies file types, builds initial context graph
4. Workshop registers project in memory (name, path, main files, language/framework)

**Expected Outcome:** Workshop silently monitors all registered projects, updates context graph as files change

---

### Workflow 2: Automatic Context Assembly (The Core Feature)

**Goal:** User asks question, Workshop automatically assembles relevant context

**Example Query:** "Why is cell 2 reading high?"

**Behind the Scenes:**

1. **Detect Active Context**
   - Recently edited files: `battery_guardian.ino` (5 seconds ago), `battery_config.h` (2 minutes ago)
   - Current project: Battery Guardian (inferred from file path)
   - Active workflow: Debugging (detected: repeated Serial monitor opens, voltage calibration edits)

2. **Build Context Graph**
   - Primary: `battery_guardian.ino` (active file)
   - Secondary: `battery_config.h` (included by primary, recently edited)
   - Tertiary: `bq76940.h` (chip driver, imported by primary)
   - Related: Previous session notes about cell balancing issues

3. **Inject Critical Context (System Prompt)**
   ```
   USER CONTEXT:
   - Project: Battery Guardian (Arduino ESP32)
   - Active File: battery_guardian.ino (edited 5s ago)
   - Recent Changes: Modified CELL_VOLTAGE_SCALE in battery_config.h
   - Detected Workflow: Debugging voltage readings
   ```

4. **LLM Reasoning**
   - "Cell 2 reading high" suggests voltage calibration issue
   - User just edited voltage scale constants
   - Need to check: current scale values, cell 2 reading logic, BQ76940 datasheet specs

5. **LLM Retrieves Additional Context (Tool Calls)**
   ```python
   get_file_content("battery_config.h")  # Check CELL_VOLTAGE_SCALE
   get_file_content("battery_guardian.ino", lines=45-67)  # Cell 2 reading code
   search_web("BQ76940 cell voltage ADC calibration")  # Datasheet info
   ```

6. **LLM Assembles Full Picture**
   - Reads `battery_config.h`: `CELL_VOLTAGE_SCALE = 0.003` (changed from 0.0033)
   - Reads cell 2 code: `cell2_voltage = adc_reading * CELL_VOLTAGE_SCALE`
   - Fetches datasheet: BQ76940 ADC is 14-bit, 3.3V reference → scale should be 3.3/16384 = 0.000201
   - **Diagnosis:** User's scale is 15x too large! Should be ~0.0002, not 0.003

7. **LLM Responds**
   ```
   "Cell 2 is reading high because CELL_VOLTAGE_SCALE in battery_config.h
   is incorrect. You set it to 0.003, but for a 14-bit ADC with 3.3V
   reference, it should be 3.3/16384 = 0.000201. Try changing line 12 to:

   #define CELL_VOLTAGE_SCALE 0.000201

   This will fix all cell voltage readings, not just cell 2."
   ```

**Key Point:** User never said "read battery_config.h" or "check the BQ76940 datasheet". Workshop figured out what context was needed and fetched it automatically.

---

### Workflow 3: Workflow Detection

**Goal:** Recognize what user is doing, adjust context priorities

**Detectable Patterns:**

| Pattern | Indicators | Context Priority Adjustments |
|---------|-----------|------------------------------|
| **Debugging** | - Repeated compile/upload<br>- Serial monitor opens<br>- Same file edited multiple times<br>- Error-related searches | - Prioritize error logs<br>- Load recent changes first<br>- Fetch hardware datasheets<br>- Check for common bugs (buffer overflow, timing) |
| **Feature Development** | - New file created<br>- Multiple files edited (header + implementation)<br>- Test file created/modified | - Prioritize related files (headers, dependencies)<br>- Suggest file updates (config, makefile)<br>- Check for API compatibility |
| **Configuration Editing** | - Config files modified (.h, .yaml, .json)<br>- No code changes<br>- Repeated value tweaking | - Prioritize config schema/docs<br>- Check for value ranges<br>- Warn about common misconfigurations |
| **Research/Learning** | - Web searches<br>- Documentation file reading<br>- Code experimentation (create → test → delete) | - Maintain research notes<br>- Link concepts to code<br>- Suggest next steps |

**Example:**
```
User edits battery_config.h → CELL_VOLTAGE_SCALE 5 times in 3 minutes
Workshop detects: "Configuration tuning workflow"
Workshop adjusts: Prioritize datasheet specs, previous calibration values,
                  suggest validation test (read known voltage, check accuracy)
```

---

### Workflow 4: Context Visualization

**Goal:** Show user what Workshop knows, enable manual context adjustment

**UI Elements (in Construct system):**

1. **Active Context Panel**
   ```
   📂 Battery Guardian
   📄 battery_guardian.ino (active)
   📄 battery_config.h (recent edit)
   📚 bq76940.h (imported)

   🔍 Detected: Debugging workflow
   ⏱️  Session: 23 minutes
   ```

2. **Context Graph Visualization**
   - Node graph showing file relationships
   - Highlighted: active files (green), recently edited (yellow), available (gray)
   - Edges: imports (solid), references (dashed), co-edited (dotted)

3. **Manual Controls**
   - "Add file to context" button
   - "Remove from context" button
   - "Focus on this feature" (filter context to specific subsystem)
   - "Ignore this directory" (exclude build artifacts, libraries)

**Why visualize?**
- User can verify Workshop "sees" the right context
- Manual override when automatic detection fails
- Educational: user learns what context matters for different tasks

---

### Workflow 5: On-Demand Context Retrieval

**Goal:** LLM fetches additional context as needed via tool calls

**New Tools (Phase 3):**

| Tool | Purpose | Example |
|------|---------|---------|
| `get_file_content(path, lines=None)` | Read file or specific line range | `get_file_content("battery_guardian.ino", lines=45-67)` |
| `search_project_files(query, project=None)` | Semantic search across project files | `search_project_files("BQ76940 initialization")` |
| `find_definition(symbol, project=None)` | Find where variable/function is defined | `find_definition("CELL_VOLTAGE_SCALE")` |
| `find_references(symbol, project=None)` | Find all uses of variable/function | `find_references("read_cell_voltage")` |
| `get_related_files(file_path)` | Get imports, includes, dependencies | `get_related_files("battery_guardian.ino")` |
| `get_recent_edits(file_path, limit=10)` | Get recent change history | `get_recent_edits("battery_config.h")` |
| `search_web_docs(query)` | Fetch external docs/datasheets | `search_web_docs("BQ76940 ADC calibration")` |

**LLM Decision Tree:**
```
User asks question
  ↓
Check active context (system prompt)
  ↓
Is context sufficient? → YES → Answer directly
  ↓ NO
What type of info needed?
  ├─ File content → get_file_content()
  ├─ Code search → search_project_files()
  ├─ Symbol location → find_definition()
  ├─ Dependencies → get_related_files()
  └─ External knowledge → search_web_docs()
  ↓
Fetch context via tool call(s)
  ↓
Re-evaluate: Is context sufficient now?
  ↓ YES
Answer question
```

---

## 6. Success Criteria

### Quantitative

- **Context Assembly Time:** <200ms to build initial context graph for typical project
- **Relevance Accuracy:** >80% (correct files loaded automatically in test scenarios)
- **Workflow Detection Accuracy:** >70% (identifies correct pattern from test cases)
- **Tool Retrieval Precision:** >75% (LLM fetches relevant context without over-fetching)
- **False Positive Rate:** <15% (doesn't load completely irrelevant files)
- **Memory Usage:** <500MB additional RAM for context graph (typical 100-file project)

### Qualitative

- **User says:** "It just knew what I needed"
- **Reduced query preambles:** Baseline 5-10 words of file specification → Target <2 words
- **Increased voice usage:** Phase 2: 30% voice, 70% typing → Phase 3 target: 60% voice, 40% typing
- **Context accuracy:** User rarely needs to correct/add context manually
- **Workflow detection feels right:** Detected pattern matches user's perception

### Test Scenarios (Must Pass)

1. **Battery Guardian Debug:**
   - Open `battery_guardian.ino`, edit voltage calibration
   - Ask "Why is cell 2 reading high?"
   - **Expected:** Workshop loads config file, finds calibration constant, checks datasheet, explains issue

2. **New Feature Development:**
   - Create `temperature_monitor.cpp` and `temperature_monitor.h`
   - Ask "Do I need to add this to the build system?"
   - **Expected:** Workshop checks Makefile/platformio.ini, suggests adding new files

3. **Cross-File Debugging:**
   - Error in `robot_arm.ino` references `servo_controller.cpp`
   - Ask "What's causing this servo initialization error?"
   - **Expected:** Workshop loads both files, traces initialization path, finds issue

4. **External Knowledge:**
   - Editing BLE code using NimBLE library
   - Ask "How do I set custom advertisement data?"
   - **Expected:** Workshop searches web for NimBLE docs, provides code example

5. **Context Persistence:**
   - Debug battery issue Monday, stop session
   - Resume Tuesday, ask "What was I working on?"
   - **Expected:** Workshop recalls previous session, mentions battery calibration debugging

---

## 7. Technical Architecture

### Component Structure

```
workshop/
├── context_manager.py          # NEW: Core context assembly
│   ├── ContextManager          # Orchestrates context assembly
│   ├── FileWatcher             # Monitors project directories
│   ├── ContextGraph            # Builds/queries file relationships
│   └── WorkflowDetector        # Recognizes patterns
│
├── context_tools.py            # NEW: LLM-callable context tools
│   ├── get_file_content()
│   ├── search_project_files()
│   ├── find_definition()
│   ├── find_references()
│   └── get_related_files()
│
├── context_viz.py              # NEW: Visualization constructs
│   ├── ContextGraphViz         # Show file relationships
│   ├── ActiveContextPanel      # Show current context
│   └── WorkflowIndicator       # Show detected workflow
│
├── agent.py                    # MODIFIED: Inject context into prompts
├── memory.py                   # MODIFIED: Store context relationships
├── config.py                   # MODIFIED: Add MONITORED_PROJECTS
└── main.py                     # MODIFIED: Initialize context manager
```

### Data Flow

```
File System Change
  ↓
FileWatcher detects event
  ↓
ContextGraph updates relationships
  ↓
WorkflowDetector analyzes patterns
  ↓
ContextManager updates active context
  ↓
[User asks question via voice]
  ↓
Agent receives active context in system prompt
  ↓
LLM reasons: "Need more context"
  ↓
LLM calls context tools (get_file_content, etc.)
  ↓
ContextManager retrieves and returns
  ↓
LLM answers with full context
```

### Storage Schema

**ChromaDB Collections:**
```python
# File relationships (vector embeddings for semantic search)
collection: "file_contexts"
documents: [
    {
        "path": "/home/bron/Projects/BatteryGuardian/battery_guardian.ino",
        "content": "/* file content */",
        "metadata": {
            "project": "BatteryGuardian",
            "language": "cpp",
            "last_modified": "2025-12-14T16:30:00",
            "imports": ["battery_config.h", "bq76940.h"],
            "symbols_defined": ["setup", "loop", "read_cell_voltage"],
            "last_edit_workflow": "debugging"
        }
    }
]

# Context relationships (which files co-occur in sessions)
collection: "file_cooccurrence"
documents: [
    {
        "file_pair": "battery_guardian.ino:battery_config.h",
        "metadata": {
            "cooccurrence_count": 47,
            "relationship_type": "include",
            "projects": ["BatteryGuardian"]
        }
    }
]
```

**SQLite Tables:**
```sql
-- Active session state
CREATE TABLE session_context (
    session_id TEXT,
    timestamp DATETIME,
    active_files TEXT,  -- JSON array
    detected_workflow TEXT,
    project_focus TEXT
);

-- File edit history
CREATE TABLE file_edits (
    file_path TEXT,
    timestamp DATETIME,
    edit_type TEXT,  -- 'create', 'modify', 'delete'
    project TEXT
);

-- Workflow history
CREATE TABLE workflow_sessions (
    session_id TEXT,
    workflow_type TEXT,  -- 'debugging', 'feature_dev', 'config_edit'
    start_time DATETIME,
    end_time DATETIME,
    primary_files TEXT,  -- JSON array
    outcome TEXT  -- 'completed', 'abandoned', 'ongoing'
);
```

---

## 8. Implementation Plan

### Phase 3.1: File Monitoring Foundation (Week 1)

**Goals:**
- File system monitoring working
- Context graph building from project scans
- Basic file relationship detection

**Deliverables:**
- `context_manager.py` with `FileWatcher` class
- `ContextGraph` builds import/include relationships
- Project registration via `config.py`
- Unit tests for file monitoring

**Success Metric:** Workshop detects file changes in registered projects within 1 second

---

### Phase 3.2: Context Assembly (Week 2)

**Goals:**
- Automatic context assembly on user query
- LLM receives active context in system prompt
- Context relevance ranking

**Deliverables:**
- `ContextManager.assemble_context()` method
- Agent integration (inject context into system prompt)
- Relevance scoring algorithm
- Test scenarios 1-3 passing

**Success Metric:** >70% context relevance in test scenarios

---

### Phase 3.3: Context Retrieval Tools (Week 3)

**Goals:**
- LLM can fetch additional context via tools
- File content, search, definition lookup working
- External knowledge retrieval (web docs)

**Deliverables:**
- `context_tools.py` with all retrieval tools
- Tool registration in agent
- Web search integration for datasheets/docs
- Test scenario 4 passing

**Success Metric:** LLM successfully retrieves needed context in >80% of test queries

---

### Phase 3.4: Workflow Detection (Week 4)

**Goals:**
- Recognize debugging, feature dev, config editing patterns
- Adjust context priorities based on workflow
- Persist workflow history

**Deliverables:**
- `WorkflowDetector` class
- Pattern matching for 4 core workflows
- Context priority adjustments
- Workflow history in SQLite

**Success Metric:** >70% workflow detection accuracy

---

### Phase 3.5: Context Visualization (Week 5)

**Goals:**
- User can see active context
- Manual context controls (add/remove files)
- Workflow indicator

**Deliverables:**
- `ContextGraphViz` construct (node graph)
- `ActiveContextPanel` construct (file list)
- `WorkflowIndicator` construct (current pattern)
- Manual override controls

**Success Metric:** User can identify and adjust context via UI

---

### Phase 3.6: Integration & Refinement (Week 6)

**Goals:**
- All test scenarios passing
- Performance optimization
- Documentation

**Deliverables:**
- Full integration test suite
- Performance benchmarks (<200ms context assembly)
- User documentation (how to register projects, interpret visualizations)
- Phase 3 complete marker

**Success Metric:** All quantitative success criteria met

---

## 9. Constraints & Risks

### Technical Constraints

- **File System Latency:** Watchdog has ~100ms delay on some file systems → Cache heavily
- **Context Graph Size:** 1000+ file projects → Limit depth, use lazy loading
- **LLM Context Window:** qwen3:8b has 32k token limit → Keep system prompt <4k tokens
- **Memory Overhead:** Context graph in RAM → Use SQLite for cold storage, RAM for hot

### Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| File monitoring misses rapid changes | Medium | Low | Debounce events, queue processing |
| Context graph too large for RAM | High | Medium | Implement graph pruning, lazy loading |
| Workflow detection false positives | Medium | High | Require confidence threshold >70%, allow manual override |
| LLM over-fetches context (tool spam) | Medium | Medium | Add fetch budget (max 5 tool calls/query), teach via examples |
| User privacy (monitor personal files) | High | Low | Explicit project registration, exclude patterns in config |

### Performance Targets

- Context assembly: <200ms
- File change detection: <1s
- Tool retrieval: <500ms per call
- Graph query: <50ms
- Total overhead per query: <1s

---

## 10. Dependencies

### New Python Packages
```
watchdog>=3.0.0        # File system monitoring
tree-sitter>=0.20.0    # Code parsing (find definitions, references)
pygments>=2.15.0       # Syntax highlighting for viz
```

### Existing Dependencies (Already Installed)
- ChromaDB (context storage)
- SQLite (session state)
- Ollama (LLM with tool calling)

### External Services (Optional)
- Web search API for documentation fetching (fallback to scraping if needed)

---

## 11. Out of Scope (Explicitly NOT Phase 3)

- ❌ **Git integration** - No branch/commit awareness
- ❌ **IDE extensions** - No VSCode plugin development
- ❌ **Cursor tracking** - No selection/caret position monitoring
- ❌ **Proactive interruptions** - Workshop waits for user to speak first
- ❌ **Multi-user support** - Single developer only
- ❌ **Cloud sync** - Local only
- ❌ **Natural language code generation** - Context assembly only, not code writing
- ❌ **Automated refactoring** - Suggestions only, no auto-apply

---

## 12. Success Validation

### User Acceptance Test

You (Christian) use Workshop for 1 week of real development work:
- Work on Battery Guardian (Arduino)
- Work on another project (React Native or Python)
- Switch between projects multiple times
- Ask 50+ questions via voice

**Success if:**
- >80% of queries don't require file specification
- You say "it knew what I needed" at least 3 times unprompted
- Workflow detection matches your perception >70% of time
- You use voice more than typing for questions
- Context visualization helps you understand/verify context at least once

### Benchmark Comparison

**Before Phase 3 (Phase 2 only):**
- Average query: "Read battery_guardian.ino and battery_config.h, then explain why cell 2 reads high"
- Words: 14 words (9 words of context specification)
- Time to answer: LLM wait + file reading + reasoning

**After Phase 3:**
- Average query: "Why does cell 2 read high?"
- Words: 6 words (0 words of context specification)
- Time to answer: LLM wait + automatic context + reasoning

**Target improvement:** 60% reduction in query overhead

---

## 13. Next Steps (Phase 4 Preview)

After Phase 3 completes, Phase 4 will add:

1. **Git Integration** - Branch awareness, commit analysis, diff understanding
2. **Proactive Suggestions** - "I noticed you're debugging servo jitter, want me to check PWM frequency?"
3. **Cross-Project Context** - Link Battery Guardian firmware with mobile app code
4. **External Knowledge Caching** - Store fetched datasheets permanently
5. **Pattern Learning** - Remember your debugging workflows, suggest shortcuts

**Phase 3 → Phase 4 Unlock:** Once Workshop reliably assembles context automatically (Phase 3), it can start *predicting* what you need (Phase 4).

---

**END OF MASTER PROJECT SPEC - PHASE 3**

---

*This spec was generated based on architectural decisions from Christian (2025-12-14). Implementation begins after Phase 2 integration testing completes.*
