# Phase 3 Implementation Plan

**Date:** 2025-12-14
**Status:** Planning Complete, Ready for Implementation

---

## Current State Assessment

### Already Implemented (Phase 2)
- ✅ Basic `context.py` with ContextAwareness class
- ✅ USB device detection (Arduino/ESP32)
- ✅ Recent file tracking (60-minute window)
- ✅ Active window detection (Linux/macOS)
- ✅ Project path configuration in config.py
- ✅ Agent context injection mechanism
- ✅ Memory system with project support (SQLite + ChromaDB)

### Gap Analysis - What Phase 3 Adds

**Missing from Phase 2:**
1. ❌ **Continuous file monitoring** (currently polls on-demand)
2. ❌ **Context graph** (file relationships, imports, dependencies)
3. ❌ **Workflow detection** (no pattern recognition)
4. ❌ **Context retrieval tools** (LLM can't fetch additional context)
5. ❌ **Session tracking** (no workflow history)
6. ❌ **Context visualization** (no UI for context)

---

## Implementation Strategy

### Phase 3.1: File Monitoring Foundation (Week 1)

**Goal:** Replace on-demand scanning with continuous monitoring

**Files to Create/Modify:**
- `context_manager.py` - NEW (orchestrates all context features)
- `config.py` - MODIFY (add MONITORED_PROJECTS)
- `memory.py` - MODIFY (add new SQLite tables)

**Implementation:**

1. **Create `context_manager.py`:**
```python
class FileWatcher:
    """Monitor project directories for file changes"""
    - Use watchdog library for filesystem events
    - Track: file creates, modifies, deletes
    - Debounce rapid changes (e.g., save-on-every-keystroke)
    - Update ContextGraph on changes

class ContextGraph:
    """Build and query file relationships"""
    - Parse imports/includes using tree-sitter
    - Track: which files import which
    - Co-occurrence matrix (files edited together)
    - Store in ChromaDB for semantic search

class ContextManager:
    """Main orchestrator for Phase 3 features"""
    - Owns FileWatcher, ContextGraph, WorkflowDetector
    - Assembles context on query
    - Ranks relevance of files
```

2. **Update `config.py`:**
```python
# Add to Config class:
MONITORED_PROJECTS = [
    Path.home() / "FlyingTiger" / "Workshop_Assistant_Dev",
    Path.home() / "FlyingTiger" / "Products" / "Smart_LiPo_Battery_Guardian",
    Path.home() / "Arduino" / "sketches",
]

# For file system monitoring
FILE_WATCH_DEBOUNCE = 1.0  # seconds
FILE_WATCH_IGNORE = {
    '__pycache__', 'node_modules', '.git', 'venv', '.venv',
    'build', 'dist', '.DS_Store'
}
```

3. **Extend `memory.py` - New Tables:**
```sql
-- Active session context
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
    workflow_type TEXT,
    start_time DATETIME,
    end_time DATETIME,
    primary_files TEXT,  -- JSON array
    outcome TEXT
);
```

**Success Criteria:**
- [ ] File changes detected within 1 second
- [ ] Context graph builds for 100+ file project in <200ms
- [ ] ContextManager initializes without errors

---

### Phase 3.2: Context Assembly (Week 2)

**Goal:** Automatic context assembly when user asks a question

**Files to Modify:**
- `context_manager.py` - ADD context assembly logic
- `agent.py` - MODIFY to use enhanced context

**Implementation:**

1. **ContextManager.assemble_context():**
```python
def assemble_context(self, user_query: str) -> dict:
    """
    Build relevant context for a user query.

    Returns:
        {
            'critical': {  # Goes in system prompt
                'active_files': [...],
                'recent_changes': [...],
                'detected_workflow': '...',
            },
            'retrievable': {  # Available via tools
                'related_files': [...],
                'file_graph': {...},
                'previous_sessions': [...],
            }
        }
    """
    # 1. Get active files (recently edited)
    # 2. Build file graph (imports, dependencies)
    # 3. Rank by relevance to query
    # 4. Split into critical vs. retrievable
```

2. **Relevance Scoring:**
```python
def score_file_relevance(self, file_path: Path, query: str) -> float:
    """
    Score 0-1 for file relevance to query.

    Factors:
    - Recency of edits (higher = more relevant)
    - Query keyword match in filename/path
    - File type relevance (.ino for hardware queries)
    - Import relationships to active files
    - Co-occurrence in past sessions
    """
```

3. **Agent Integration:**
```python
# In agent.py chat() method:
if self.context_manager:
    context = self.context_manager.assemble_context(user_input)

    # Inject critical context into system prompt
    system_prompt = self._build_system_message_with_context(context['critical'])

    # Make retrievable context available via tools
    self.context_manager.current_retrievable = context['retrievable']
```

**Success Criteria:**
- [ ] >70% of test queries get correct files in context
- [ ] Context assembly completes in <200ms
- [ ] User says "it knew what I needed" in manual testing

---

### Phase 3.3: Context Retrieval Tools (Week 3)

**Goal:** LLM can fetch additional context via tool calls

**Files to Create:**
- `context_tools.py` - NEW (LLM-callable context tools)

**Implementation:**

Create 7 new tools in `context_tools.py`:

```python
def get_file_content(path: str, lines: str = None) -> str:
    """
    Read file or specific line range.

    Examples:
        get_file_content("battery_guardian.ino")
        get_file_content("config.h", lines="10-25")
    """

def search_project_files(query: str, project: str = None) -> str:
    """
    Semantic search across project files using ChromaDB.

    Example:
        search_project_files("BQ76940 initialization")
    """

def find_definition(symbol: str, project: str = None) -> str:
    """
    Find where a variable/function is defined.
    Uses tree-sitter for parsing.

    Example:
        find_definition("CELL_VOLTAGE_SCALE")
    """

def find_references(symbol: str, project: str = None) -> str:
    """
    Find all uses of a variable/function.

    Example:
        find_references("read_cell_voltage")
    """

def get_related_files(file_path: str) -> str:
    """
    Get imports, includes, dependencies.

    Returns:
        - Files this imports
        - Files that import this
        - Files co-edited in past sessions
    """

def get_recent_edits(file_path: str, limit: int = 10) -> str:
    """
    Get recent change history for a file.

    Returns list of edit events with timestamps.
    """

def search_web_docs(query: str) -> str:
    """
    Fetch external docs/datasheets.
    Wraps existing web_search but focused on technical docs.
    """
```

**Tool Registration:**
```python
# In main.py or tools.py:
from context_tools import register_context_tools

register_context_tools(
    registry=tool_registry,
    context_manager=context_mgr,
    memory=memory_system
)
```

**Success Criteria:**
- [ ] All 7 tools work correctly
- [ ] LLM successfully uses tools in >80% of test queries
- [ ] Tool calls complete in <500ms each

---

### Phase 3.4: Workflow Detection (Week 4)

**Goal:** Recognize debugging, feature dev, config editing patterns

**Files to Modify:**
- `context_manager.py` - ADD WorkflowDetector class

**Implementation:**

```python
class WorkflowDetector:
    """Recognize development workflow patterns"""

    WORKFLOWS = {
        'debugging': {
            'indicators': [
                'repeated_compiles',
                'serial_monitor_opens',
                'same_file_edited_multiple_times',
                'error_keywords_in_searches',
            ],
            'context_priority': ['error_logs', 'recent_changes', 'datasheets'],
        },
        'feature_development': {
            'indicators': [
                'new_file_created',
                'multiple_files_edited',
                'test_file_created',
            ],
            'context_priority': ['related_files', 'api_docs', 'similar_code'],
        },
        'configuration': {
            'indicators': [
                'config_file_modified',
                'no_code_changes',
                'repeated_value_changes',
            ],
            'context_priority': ['config_schema', 'valid_ranges', 'examples'],
        },
    }

    def detect_workflow(self, recent_activity: List[FileEdit]) -> str:
        """
        Analyze recent activity and return workflow type.

        Returns: 'debugging' | 'feature_development' | 'configuration' | 'unknown'
        """
        scores = {workflow: 0 for workflow in self.WORKFLOWS}

        # Score each workflow based on indicators
        for workflow, config in self.WORKFLOWS.items():
            for indicator in config['indicators']:
                if self._check_indicator(indicator, recent_activity):
                    scores[workflow] += 1

        # Return highest scoring workflow (if confidence > 70%)
        max_score = max(scores.values())
        if max_score >= 0.7 * len(self.WORKFLOWS[workflow]['indicators']):
            return max(scores, key=scores.get)

        return 'unknown'
```

**Success Criteria:**
- [ ] >70% workflow detection accuracy on test scenarios
- [ ] Context priorities adjust based on detected workflow
- [ ] Workflow history persists to SQLite

---

### Phase 3.5: Context Visualization (Week 5)

**Goal:** User can see and adjust context

**Files to Create:**
- `context_viz.py` - NEW (visualization constructs)

**Implementation:**

Use existing construct system (from Phase 2) to visualize context:

```python
class ContextGraphViz(Construct):
    """Show file relationship graph"""

    def render(self) -> str:
        # Generate ASCII graph or JSON for 3D UI
        return """
        battery_guardian.ino
        ├── battery_config.h (edited 2m ago)
        ├── bq76940.h
        └── Arduino.h
        """

class ActiveContextPanel(Construct):
    """Show current active context"""

    def render(self) -> str:
        return """
        📂 Battery Guardian
        📄 battery_guardian.ino (active)
        📄 battery_config.h (recent edit)
        📚 bq76940.h (imported)

        🔍 Detected: Debugging workflow
        ⏱️  Session: 23 minutes
        """

class WorkflowIndicator(Construct):
    """Show detected workflow"""

    def render(self) -> str:
        workflow = self.context_manager.detect_workflow()
        return f"🔧 Workflow: {workflow}"
```

**Manual Controls:**
```python
# New tools for manual context adjustment:
def add_file_to_context(path: str) -> str:
    """Manually add a file to active context"""

def remove_from_context(path: str) -> str:
    """Remove a file from active context"""

def focus_on_feature(feature_name: str) -> str:
    """Filter context to specific subsystem"""
```

**Success Criteria:**
- [ ] User can see active context via construct
- [ ] Manual override controls work
- [ ] Visualization helps user understand context

---

### Phase 3.6: Integration & Testing (Week 6)

**Goal:** Everything works together, meets success criteria

**Tasks:**

1. **Integration Testing:**
   - Run all 5 test scenarios from spec
   - Measure performance metrics
   - Fix integration issues

2. **Performance Optimization:**
   - Profile context assembly (target <200ms)
   - Optimize file watching (reduce CPU usage)
   - Cache graph queries

3. **Documentation:**
   - User guide: how to register projects
   - Developer docs: architecture diagrams
   - Examples: common workflows

4. **User Acceptance Testing:**
   - Use Workshop for 1 week of real work
   - Track query overhead reduction
   - Collect feedback

**Success Criteria:**
- [ ] All quantitative metrics met
- [ ] All test scenarios pass
- [ ] User acceptance criteria met

---

## Dependencies to Install

```bash
pip install watchdog>=3.0.0      # File system monitoring
pip install tree-sitter>=0.20.0  # Code parsing
pip install pygments>=2.15.0     # Syntax highlighting
```

---

## Quick Start

1. Install dependencies
2. Add projects to `config.py` MONITORED_PROJECTS
3. Run Workshop - file monitoring starts automatically
4. Test with: "What files have I been working on?"

---

## Rollout Plan

- Week 1: File monitoring foundation
- Week 2: Context assembly
- Week 3: Retrieval tools
- Week 4: Workflow detection
- Week 5: Visualization
- Week 6: Integration & polish

**Total: 6 weeks to Phase 3 MVP**
