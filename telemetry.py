"""
Workshop Telemetry System
Comprehensive tracing for observability, debugging, and training data collection.

Every interaction is captured as a Trace with full context, tool calls, and timing.
Designed for:
1. Real-time debugging ("what context was loaded?")
2. Performance analysis (latency, token usage)
3. Training data extraction (complete input/output pairs with reasoning)
"""

import json
import time
import uuid
import sqlite3
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

from logger import get_logger

log = get_logger("telemetry")


class TraceStatus(Enum):
    """Status of a trace through its lifecycle"""
    STARTED = "started"
    CONTEXT_LOADED = "context_loaded"
    INTENT_DETECTED = "intent_detected"
    LLM_CALLED = "llm_called"
    TOOLS_EXECUTED = "tools_executed"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ContextSource:
    """A single source of context that was loaded"""
    source_type: str           # 'telos_profile', 'telos_goals', 'telos_project',
                               # 'context_manager', 'memory_search', 'recent_messages'
    source_path: Optional[str] # File path if applicable
    content: str               # Actual content loaded
    content_length: int        # Character count
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "source_type": self.source_type,
            "source_path": self.source_path,
            "content": self.content,
            "content_length": self.content_length,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ToolCallTrace:
    """Complete record of a single tool invocation"""
    call_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp_start: datetime = field(default_factory=datetime.now)
    timestamp_end: Optional[datetime] = None

    # Tool identification
    tool_name: str = ""
    skill_name: str = ""

    # Arguments
    args_raw: Dict = field(default_factory=dict)        # As received from LLM
    args_normalized: Dict = field(default_factory=dict) # After alias resolution
    args_with_deps: Dict = field(default_factory=dict)  # With injected dependencies listed

    # Execution
    dependencies_available: List[str] = field(default_factory=list)  # memory, config, etc.
    dependencies_used: List[str] = field(default_factory=list)       # Actually accessed

    # Result
    result_raw: str = ""
    result_type: str = ""      # 'success', 'error', 'empty', 'truncated'
    result_length: int = 0
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    # Timing
    duration_ms: int = 0

    # For pattern-matched direct calls
    pattern_matched: Optional[str] = None  # The regex pattern that matched
    was_direct_call: bool = False          # True if bypassed LLM

    def complete(self, result: str, error: str = None):
        """Mark tool call as complete"""
        self.timestamp_end = datetime.now()
        self.duration_ms = int((self.timestamp_end - self.timestamp_start).total_seconds() * 1000)
        self.result_raw = result
        self.result_length = len(result) if result else 0

        if error:
            self.result_type = "error"
            self.error_message = error
        elif not result or len(result.strip()) == 0:
            self.result_type = "empty"
        elif len(result) > 10000:
            self.result_type = "truncated"
        else:
            self.result_type = "success"

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "timestamp_start": self.timestamp_start.isoformat(),
            "timestamp_end": self.timestamp_end.isoformat() if self.timestamp_end else None,
            "tool_name": self.tool_name,
            "skill_name": self.skill_name,
            "args_raw": self.args_raw,
            "args_normalized": self.args_normalized,
            "args_with_deps": self.args_with_deps,
            "dependencies_available": self.dependencies_available,
            "dependencies_used": self.dependencies_used,
            "result_raw": self.result_raw,
            "result_type": self.result_type,
            "result_length": self.result_length,
            "error_message": self.error_message,
            "error_traceback": self.error_traceback,
            "duration_ms": self.duration_ms,
            "pattern_matched": self.pattern_matched,
            "was_direct_call": self.was_direct_call
        }


@dataclass
class LLMCall:
    """Record of a single LLM API call (there may be multiple per trace)"""
    call_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp_start: datetime = field(default_factory=datetime.now)
    timestamp_end: Optional[datetime] = None

    # Request
    model: str = ""
    messages: List[Dict] = field(default_factory=list)  # Full message history sent
    system_prompt: str = ""
    system_prompt_length: int = 0

    # Response
    response_raw: str = ""
    response_length: int = 0

    # Extracted from response
    tool_calls_extracted: List[Dict] = field(default_factory=list)  # JSON tool calls found

    # Timing & tokens
    duration_ms: int = 0
    tokens_prompt: Optional[int] = None      # If available from API
    tokens_completion: Optional[int] = None
    tokens_total: Optional[int] = None

    # Iteration info (for multi-turn tool use)
    iteration: int = 1
    max_iterations: int = 5

    def complete(self, response: str):
        """Mark LLM call as complete"""
        self.timestamp_end = datetime.now()
        self.duration_ms = int((self.timestamp_end - self.timestamp_start).total_seconds() * 1000)
        self.response_raw = response
        self.response_length = len(response) if response else 0

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "timestamp_start": self.timestamp_start.isoformat(),
            "timestamp_end": self.timestamp_end.isoformat() if self.timestamp_end else None,
            "model": self.model,
            "messages": self.messages,
            "system_prompt": self.system_prompt,
            "system_prompt_length": self.system_prompt_length,
            "response_raw": self.response_raw,
            "response_length": self.response_length,
            "tool_calls_extracted": self.tool_calls_extracted,
            "duration_ms": self.duration_ms,
            "tokens_prompt": self.tokens_prompt,
            "tokens_completion": self.tokens_completion,
            "tokens_total": self.tokens_total,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations
        }


@dataclass
class Trace:
    """
    Complete record of a single user interaction (request-response cycle).

    This is the primary unit for:
    - Debugging ("what happened when I said X?")
    - Training data ("input -> reasoning -> output")
    - Evaluation metrics
    """

    # Identification
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    timestamp_start: datetime = field(default_factory=datetime.now)
    timestamp_end: Optional[datetime] = None
    status: TraceStatus = TraceStatus.STARTED

    # User Input
    user_input_raw: str = ""
    user_input_length: int = 0
    user_input_enriched: str = ""      # After context injection
    user_input_enriched_length: int = 0

    # Context Loading (critical for training data)
    context_injection_attempted: bool = False
    context_injection_succeeded: bool = False
    context_should_inject_reason: str = ""  # Why we decided to inject (or not)
    context_sources: List[ContextSource] = field(default_factory=list)
    context_assembled: str = ""         # Final assembled context string
    context_assembled_length: int = 0

    # Telos Personal Context
    telos_loaded: bool = False
    telos_profile: str = ""
    telos_goals: str = ""
    telos_active_project: Optional[str] = None
    telos_project_context: str = ""

    # Automatic Context (ContextManager)
    context_manager_used: bool = False
    active_files: List[str] = field(default_factory=list)
    recent_changes: List[Dict] = field(default_factory=list)
    detected_workflow: Optional[str] = None
    detected_workflow_confidence: float = 0.0

    # Memory Context
    memory_search_query: str = ""
    memory_results: List[str] = field(default_factory=list)
    memory_results_count: int = 0
    user_profile_loaded: str = ""
    recent_messages_count: int = 0

    # Intent Detection
    intent_detection_attempted: bool = False
    intent_detected: bool = False
    intent_tool_name: Optional[str] = None
    intent_pattern_matched: Optional[str] = None
    intent_args_extracted: Dict = field(default_factory=dict)
    intent_detection_time_ms: int = 0

    # LLM Interactions (may be multiple for tool-use loops)
    llm_calls: List[LLMCall] = field(default_factory=list)
    llm_total_calls: int = 0
    llm_total_duration_ms: int = 0

    # Tool Execution
    tool_calls: List[ToolCallTrace] = field(default_factory=list)
    tool_total_calls: int = 0
    tool_total_duration_ms: int = 0
    tools_succeeded: int = 0
    tools_failed: int = 0

    # Final Output
    response_raw: str = ""             # Before any cleaning
    response_final: str = ""           # What user actually sees
    response_length: int = 0
    response_was_formatted: bool = False  # If we asked LLM to format tool result

    # Timing
    total_duration_ms: int = 0

    # Error tracking
    error_occurred: bool = False
    error_stage: str = ""              # 'context', 'intent', 'llm', 'tool', 'format'
    error_message: str = ""
    error_traceback: str = ""

    # Environment (useful for reproducibility)
    model_name: str = ""
    voice_mode: bool = False
    constructs_enabled: bool = False

    # Metadata for training data extraction
    metadata: Dict = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)  # Manual tags for categorization

    def add_context_source(self, source_type: str, content: str, source_path: str = None):
        """Add a context source with timestamp"""
        self.context_sources.append(ContextSource(
            source_type=source_type,
            source_path=source_path,
            content=content,
            content_length=len(content)
        ))

    def start_llm_call(self, model: str, messages: List[Dict], system_prompt: str, iteration: int = 1) -> LLMCall:
        """Start tracking an LLM call"""
        llm_call = LLMCall(
            model=model,
            messages=messages.copy(),
            system_prompt=system_prompt,
            system_prompt_length=len(system_prompt),
            iteration=iteration
        )
        self.llm_calls.append(llm_call)
        self.llm_total_calls += 1
        return llm_call

    def start_tool_call(self, tool_name: str, skill_name: str, args: Dict) -> ToolCallTrace:
        """Start tracking a tool call"""
        tool_trace = ToolCallTrace(
            tool_name=tool_name,
            skill_name=skill_name,
            args_raw=args.copy()
        )
        self.tool_calls.append(tool_trace)
        self.tool_total_calls += 1
        return tool_trace

    def complete(self, response: str):
        """Mark trace as complete"""
        self.timestamp_end = datetime.now()
        self.total_duration_ms = int((self.timestamp_end - self.timestamp_start).total_seconds() * 1000)
        self.response_final = response
        self.response_length = len(response) if response else 0
        self.status = TraceStatus.COMPLETED

        # Aggregate tool stats
        self.tool_total_duration_ms = sum(t.duration_ms for t in self.tool_calls)
        self.tools_succeeded = sum(1 for t in self.tool_calls if t.result_type == "success")
        self.tools_failed = sum(1 for t in self.tool_calls if t.result_type == "error")

        # Aggregate LLM stats
        self.llm_total_duration_ms = sum(c.duration_ms for c in self.llm_calls)

    def mark_error(self, stage: str, message: str, traceback: str = ""):
        """Mark trace as failed with error info"""
        self.timestamp_end = datetime.now()
        self.total_duration_ms = int((self.timestamp_end - self.timestamp_start).total_seconds() * 1000)
        self.status = TraceStatus.ERROR
        self.error_occurred = True
        self.error_stage = stage
        self.error_message = message
        self.error_traceback = traceback

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            # Identification
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "timestamp_start": self.timestamp_start.isoformat(),
            "timestamp_end": self.timestamp_end.isoformat() if self.timestamp_end else None,
            "status": self.status.value,

            # User Input
            "user_input_raw": self.user_input_raw,
            "user_input_length": self.user_input_length,
            "user_input_enriched": self.user_input_enriched,
            "user_input_enriched_length": self.user_input_enriched_length,

            # Context
            "context_injection_attempted": self.context_injection_attempted,
            "context_injection_succeeded": self.context_injection_succeeded,
            "context_should_inject_reason": self.context_should_inject_reason,
            "context_sources": [s.to_dict() for s in self.context_sources],
            "context_assembled": self.context_assembled,
            "context_assembled_length": self.context_assembled_length,

            # Telos
            "telos_loaded": self.telos_loaded,
            "telos_profile": self.telos_profile,
            "telos_goals": self.telos_goals,
            "telos_active_project": self.telos_active_project,
            "telos_project_context": self.telos_project_context,

            # Context Manager
            "context_manager_used": self.context_manager_used,
            "active_files": self.active_files,
            "recent_changes": self.recent_changes,
            "detected_workflow": self.detected_workflow,
            "detected_workflow_confidence": self.detected_workflow_confidence,

            # Memory
            "memory_search_query": self.memory_search_query,
            "memory_results": self.memory_results,
            "memory_results_count": self.memory_results_count,
            "user_profile_loaded": self.user_profile_loaded,
            "recent_messages_count": self.recent_messages_count,

            # Intent
            "intent_detection_attempted": self.intent_detection_attempted,
            "intent_detected": self.intent_detected,
            "intent_tool_name": self.intent_tool_name,
            "intent_pattern_matched": self.intent_pattern_matched,
            "intent_args_extracted": self.intent_args_extracted,
            "intent_detection_time_ms": self.intent_detection_time_ms,

            # LLM
            "llm_calls": [c.to_dict() for c in self.llm_calls],
            "llm_total_calls": self.llm_total_calls,
            "llm_total_duration_ms": self.llm_total_duration_ms,

            # Tools
            "tool_calls": [t.to_dict() for t in self.tool_calls],
            "tool_total_calls": self.tool_total_calls,
            "tool_total_duration_ms": self.tool_total_duration_ms,
            "tools_succeeded": self.tools_succeeded,
            "tools_failed": self.tools_failed,

            # Output
            "response_raw": self.response_raw,
            "response_final": self.response_final,
            "response_length": self.response_length,
            "response_was_formatted": self.response_was_formatted,

            # Timing
            "total_duration_ms": self.total_duration_ms,

            # Errors
            "error_occurred": self.error_occurred,
            "error_stage": self.error_stage,
            "error_message": self.error_message,
            "error_traceback": self.error_traceback,

            # Environment
            "model_name": self.model_name,
            "voice_mode": self.voice_mode,
            "constructs_enabled": self.constructs_enabled,

            # Metadata
            "metadata": self.metadata,
            "tags": self.tags
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        """Get a concise summary for CLI display"""
        lines = [
            f"Trace: {self.trace_id[:8]}",
            f"  Status: {self.status.value}",
            f"  Input: {self.user_input_raw[:50]}{'...' if len(self.user_input_raw) > 50 else ''}",
            f"  Duration: {self.total_duration_ms}ms",
        ]

        if self.context_sources:
            lines.append(f"  Context: {len(self.context_sources)} sources ({self.context_assembled_length} chars)")

        if self.intent_detected:
            lines.append(f"  Direct Intent: {self.intent_tool_name}")

        if self.llm_calls:
            lines.append(f"  LLM Calls: {self.llm_total_calls} ({self.llm_total_duration_ms}ms)")

        if self.tool_calls:
            tools = [t.tool_name for t in self.tool_calls]
            lines.append(f"  Tools: {tools} ({self.tools_succeeded} ok, {self.tools_failed} failed)")

        if self.error_occurred:
            lines.append(f"  ERROR at {self.error_stage}: {self.error_message[:50]}")

        lines.append(f"  Output: {self.response_length} chars")

        return "\n".join(lines)


class TelemetryCollector:
    """
    Collects, stores, and retrieves traces.

    Storage options:
    1. SQLite (persistent, queryable)
    2. JSON files (easy export, training data)
    3. In-memory (for current session)
    """

    def __init__(
        self,
        sqlite_path: Path = None,
        json_dir: Path = None,
        keep_in_memory: int = 100,  # Keep last N traces in memory
        auto_save: bool = True
    ):
        self.sqlite_path = sqlite_path
        self.json_dir = json_dir
        self.keep_in_memory = keep_in_memory
        self.auto_save = auto_save

        # In-memory cache
        self._traces: List[Trace] = []
        self._current_trace: Optional[Trace] = None

        # Session tracking
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initialize storage
        if sqlite_path:
            self._init_sqlite()

        if json_dir:
            json_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"TelemetryCollector initialized (session: {self.session_id})")

    def _init_sqlite(self):
        """Initialize SQLite tables for trace storage"""
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.sqlite_path))
        conn.executescript("""
            -- Main traces table
            CREATE TABLE IF NOT EXISTS traces (
                trace_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp_start DATETIME NOT NULL,
                timestamp_end DATETIME,
                status TEXT NOT NULL,

                -- User input
                user_input_raw TEXT,
                user_input_length INTEGER,
                user_input_enriched TEXT,

                -- Context summary
                context_injection_succeeded BOOLEAN,
                context_assembled_length INTEGER,
                context_sources_count INTEGER,

                -- Telos
                telos_loaded BOOLEAN,
                telos_active_project TEXT,

                -- Context Manager
                detected_workflow TEXT,
                detected_workflow_confidence REAL,

                -- Intent
                intent_detected BOOLEAN,
                intent_tool_name TEXT,

                -- LLM summary
                llm_total_calls INTEGER,
                llm_total_duration_ms INTEGER,

                -- Tool summary
                tool_total_calls INTEGER,
                tool_total_duration_ms INTEGER,
                tools_succeeded INTEGER,
                tools_failed INTEGER,

                -- Output
                response_final TEXT,
                response_length INTEGER,

                -- Timing
                total_duration_ms INTEGER,

                -- Errors
                error_occurred BOOLEAN,
                error_stage TEXT,
                error_message TEXT,

                -- Environment
                model_name TEXT,
                voice_mode BOOLEAN,

                -- Full JSON (for complete data)
                full_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_traces_session ON traces(session_id);
            CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON traces(timestamp_start);
            CREATE INDEX IF NOT EXISTS idx_traces_status ON traces(status);
            CREATE INDEX IF NOT EXISTS idx_traces_error ON traces(error_occurred);

            -- Tool calls table (for detailed analysis)
            CREATE TABLE IF NOT EXISTS tool_calls (
                call_id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL,
                timestamp_start DATETIME NOT NULL,
                timestamp_end DATETIME,

                tool_name TEXT NOT NULL,
                skill_name TEXT,

                args_raw TEXT,      -- JSON
                args_normalized TEXT,

                result_type TEXT,
                result_length INTEGER,
                duration_ms INTEGER,

                error_message TEXT,
                was_direct_call BOOLEAN,
                pattern_matched TEXT,

                FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
            );

            CREATE INDEX IF NOT EXISTS idx_tool_calls_trace ON tool_calls(trace_id);
            CREATE INDEX IF NOT EXISTS idx_tool_calls_tool ON tool_calls(tool_name);

            -- LLM calls table
            CREATE TABLE IF NOT EXISTS llm_calls (
                call_id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL,
                timestamp_start DATETIME NOT NULL,
                timestamp_end DATETIME,

                model TEXT,
                system_prompt_length INTEGER,
                messages_count INTEGER,

                response_length INTEGER,
                tool_calls_extracted INTEGER,
                duration_ms INTEGER,

                tokens_prompt INTEGER,
                tokens_completion INTEGER,
                iteration INTEGER,

                FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
            );

            CREATE INDEX IF NOT EXISTS idx_llm_calls_trace ON llm_calls(trace_id);

            -- Context sources table
            CREATE TABLE IF NOT EXISTS context_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,

                source_type TEXT NOT NULL,
                source_path TEXT,
                content TEXT,
                content_length INTEGER,

                FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
            );

            CREATE INDEX IF NOT EXISTS idx_context_sources_trace ON context_sources(trace_id);
            CREATE INDEX IF NOT EXISTS idx_context_sources_type ON context_sources(source_type);
        """)
        conn.commit()
        conn.close()

        log.info(f"Telemetry SQLite initialized at {self.sqlite_path}")

    def start_trace(self, user_input: str) -> Trace:
        """Start a new trace for a user interaction"""
        trace = Trace(
            session_id=self.session_id,
            user_input_raw=user_input,
            user_input_length=len(user_input)
        )

        self._current_trace = trace
        log.debug(f"Started trace: {trace.trace_id[:8]}")

        return trace

    def get_current_trace(self) -> Optional[Trace]:
        """Get the currently active trace"""
        return self._current_trace

    def complete_trace(self, trace: Trace, response: str):
        """Complete and save a trace"""
        trace.complete(response)

        # Add to memory cache
        self._traces.append(trace)
        if len(self._traces) > self.keep_in_memory:
            self._traces = self._traces[-self.keep_in_memory:]

        # Auto-save if enabled
        if self.auto_save:
            self.save_trace(trace)

        # Clear current
        if self._current_trace == trace:
            self._current_trace = None

        log.info(f"Completed trace: {trace.trace_id[:8]} ({trace.total_duration_ms}ms)")

        # Print summary if debug mode
        if os.environ.get("WORKSHOP_DEBUG"):
            print(f"\n{trace.summary()}\n")

    def save_trace(self, trace: Trace):
        """Save trace to storage"""
        # Save to SQLite
        if self.sqlite_path:
            self._save_to_sqlite(trace)

        # Save to JSON file
        if self.json_dir:
            self._save_to_json(trace)

    def _save_to_sqlite(self, trace: Trace):
        """Save trace to SQLite"""
        try:
            conn = sqlite3.connect(str(self.sqlite_path))

            # Main trace record
            conn.execute("""
                INSERT OR REPLACE INTO traces (
                    trace_id, session_id, timestamp_start, timestamp_end, status,
                    user_input_raw, user_input_length, user_input_enriched,
                    context_injection_succeeded, context_assembled_length, context_sources_count,
                    telos_loaded, telos_active_project,
                    detected_workflow, detected_workflow_confidence,
                    intent_detected, intent_tool_name,
                    llm_total_calls, llm_total_duration_ms,
                    tool_total_calls, tool_total_duration_ms, tools_succeeded, tools_failed,
                    response_final, response_length,
                    total_duration_ms,
                    error_occurred, error_stage, error_message,
                    model_name, voice_mode,
                    full_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trace.trace_id,
                trace.session_id,
                trace.timestamp_start.isoformat(),
                trace.timestamp_end.isoformat() if trace.timestamp_end else None,
                trace.status.value,
                trace.user_input_raw,
                trace.user_input_length,
                trace.user_input_enriched,
                trace.context_injection_succeeded,
                trace.context_assembled_length,
                len(trace.context_sources),
                trace.telos_loaded,
                trace.telos_active_project,
                trace.detected_workflow,
                trace.detected_workflow_confidence,
                trace.intent_detected,
                trace.intent_tool_name,
                trace.llm_total_calls,
                trace.llm_total_duration_ms,
                trace.tool_total_calls,
                trace.tool_total_duration_ms,
                trace.tools_succeeded,
                trace.tools_failed,
                trace.response_final,
                trace.response_length,
                trace.total_duration_ms,
                trace.error_occurred,
                trace.error_stage,
                trace.error_message,
                trace.model_name,
                trace.voice_mode,
                trace.to_json()
            ))

            # Tool calls
            for tool_call in trace.tool_calls:
                conn.execute("""
                    INSERT OR REPLACE INTO tool_calls (
                        call_id, trace_id, timestamp_start, timestamp_end,
                        tool_name, skill_name,
                        args_raw, args_normalized,
                        result_type, result_length, duration_ms,
                        error_message, was_direct_call, pattern_matched
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    tool_call.call_id,
                    trace.trace_id,
                    tool_call.timestamp_start.isoformat(),
                    tool_call.timestamp_end.isoformat() if tool_call.timestamp_end else None,
                    tool_call.tool_name,
                    tool_call.skill_name,
                    json.dumps(tool_call.args_raw),
                    json.dumps(tool_call.args_normalized),
                    tool_call.result_type,
                    tool_call.result_length,
                    tool_call.duration_ms,
                    tool_call.error_message,
                    tool_call.was_direct_call,
                    tool_call.pattern_matched
                ))

            # LLM calls
            for llm_call in trace.llm_calls:
                conn.execute("""
                    INSERT OR REPLACE INTO llm_calls (
                        call_id, trace_id, timestamp_start, timestamp_end,
                        model, system_prompt_length, messages_count,
                        response_length, tool_calls_extracted, duration_ms,
                        tokens_prompt, tokens_completion, iteration
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    llm_call.call_id,
                    trace.trace_id,
                    llm_call.timestamp_start.isoformat(),
                    llm_call.timestamp_end.isoformat() if llm_call.timestamp_end else None,
                    llm_call.model,
                    llm_call.system_prompt_length,
                    len(llm_call.messages),
                    llm_call.response_length,
                    len(llm_call.tool_calls_extracted),
                    llm_call.duration_ms,
                    llm_call.tokens_prompt,
                    llm_call.tokens_completion,
                    llm_call.iteration
                ))

            # Context sources
            for source in trace.context_sources:
                conn.execute("""
                    INSERT INTO context_sources (
                        trace_id, timestamp, source_type, source_path, content, content_length
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    trace.trace_id,
                    source.timestamp.isoformat(),
                    source.source_type,
                    source.source_path,
                    source.content,
                    source.content_length
                ))

            conn.commit()
            conn.close()

        except Exception as e:
            log.error(f"Failed to save trace to SQLite: {e}")

    def _save_to_json(self, trace: Trace):
        """Save trace to JSON file"""
        try:
            # Organize by date
            date_dir = self.json_dir / trace.timestamp_start.strftime("%Y-%m-%d")
            date_dir.mkdir(parents=True, exist_ok=True)

            # Filename includes timestamp and short ID
            filename = f"{trace.timestamp_start.strftime('%H%M%S')}_{trace.trace_id[:8]}.json"
            filepath = date_dir / filename

            filepath.write_text(trace.to_json())

        except Exception as e:
            log.error(f"Failed to save trace to JSON: {e}")

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get a trace by ID"""
        # Check memory first
        for trace in self._traces:
            if trace.trace_id == trace_id or trace.trace_id.startswith(trace_id):
                return trace

        # Check SQLite
        if self.sqlite_path and self.sqlite_path.exists():
            try:
                conn = sqlite3.connect(str(self.sqlite_path))
                conn.row_factory = sqlite3.Row

                row = conn.execute(
                    "SELECT full_json FROM traces WHERE trace_id LIKE ?",
                    (f"{trace_id}%",)
                ).fetchone()

                conn.close()

                if row:
                    return self._trace_from_json(row["full_json"])

            except Exception as e:
                log.error(f"Failed to load trace from SQLite: {e}")

        return None

    def get_recent_traces(self, limit: int = 10) -> List[Trace]:
        """Get most recent traces"""
        # Return from memory cache
        return list(reversed(self._traces[-limit:]))

    def query_traces(
        self,
        session_id: str = None,
        status: str = None,
        has_error: bool = None,
        tool_name: str = None,
        since: datetime = None,
        limit: int = 100
    ) -> List[Dict]:
        """Query traces with filters (returns summaries, not full traces)"""
        if not self.sqlite_path or not self.sqlite_path.exists():
            return []

        try:
            conn = sqlite3.connect(str(self.sqlite_path))
            conn.row_factory = sqlite3.Row

            query = "SELECT * FROM traces WHERE 1=1"
            params = []

            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)

            if status:
                query += " AND status = ?"
                params.append(status)

            if has_error is not None:
                query += " AND error_occurred = ?"
                params.append(has_error)

            if since:
                query += " AND timestamp_start >= ?"
                params.append(since.isoformat())

            query += " ORDER BY timestamp_start DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            conn.close()

            return [dict(row) for row in rows]

        except Exception as e:
            log.error(f"Failed to query traces: {e}")
            return []

    def _trace_from_json(self, json_str: str) -> Trace:
        """Reconstruct Trace from JSON (simplified - returns basic trace)"""
        data = json.loads(json_str)
        trace = Trace(
            trace_id=data["trace_id"],
            session_id=data["session_id"],
            user_input_raw=data["user_input_raw"]
        )
        # Populate key fields
        trace.response_final = data.get("response_final", "")
        trace.total_duration_ms = data.get("total_duration_ms", 0)
        trace.status = TraceStatus(data.get("status", "completed"))
        return trace

    def export_training_data(
        self,
        output_path: Path,
        format: str = "jsonl",  # 'jsonl' or 'json'
        include_errors: bool = False,
        min_response_length: int = 10
    ) -> int:
        """
        Export traces as training data.

        Returns number of traces exported.
        """
        if not self.sqlite_path or not self.sqlite_path.exists():
            log.warning("No SQLite database for export")
            return 0

        try:
            conn = sqlite3.connect(str(self.sqlite_path))
            conn.row_factory = sqlite3.Row

            query = "SELECT full_json FROM traces WHERE response_length >= ?"
            params = [min_response_length]

            if not include_errors:
                query += " AND error_occurred = 0"

            rows = conn.execute(query, params).fetchall()
            conn.close()

            count = 0

            with open(output_path, 'w') as f:
                if format == "json":
                    traces = [json.loads(row["full_json"]) for row in rows]
                    json.dump(traces, f, indent=2)
                    count = len(traces)
                else:  # jsonl
                    for row in rows:
                        f.write(row["full_json"] + "\n")
                        count += 1

            log.info(f"Exported {count} traces to {output_path}")
            return count

        except Exception as e:
            log.error(f"Failed to export training data: {e}")
            return 0

    def get_stats(self) -> Dict:
        """Get telemetry statistics"""
        stats = {
            "session_id": self.session_id,
            "traces_in_memory": len(self._traces),
            "current_trace_active": self._current_trace is not None
        }

        if self.sqlite_path and self.sqlite_path.exists():
            try:
                conn = sqlite3.connect(str(self.sqlite_path))

                stats["total_traces"] = conn.execute(
                    "SELECT COUNT(*) FROM traces"
                ).fetchone()[0]

                stats["total_tool_calls"] = conn.execute(
                    "SELECT COUNT(*) FROM tool_calls"
                ).fetchone()[0]

                stats["total_llm_calls"] = conn.execute(
                    "SELECT COUNT(*) FROM llm_calls"
                ).fetchone()[0]

                stats["error_count"] = conn.execute(
                    "SELECT COUNT(*) FROM traces WHERE error_occurred = 1"
                ).fetchone()[0]

                # Average response time
                avg = conn.execute(
                    "SELECT AVG(total_duration_ms) FROM traces WHERE status = 'completed'"
                ).fetchone()[0]
                stats["avg_response_time_ms"] = int(avg) if avg else 0

                conn.close()

            except Exception as e:
                log.error(f"Failed to get stats: {e}")

        return stats


# === Global instance ===
_telemetry_instance: Optional[TelemetryCollector] = None


def get_telemetry(
    sqlite_path: Path = None,
    json_dir: Path = None
) -> TelemetryCollector:
    """Get or create the global telemetry collector"""
    global _telemetry_instance

    if _telemetry_instance is None:
        # Default paths
        if sqlite_path is None:
            sqlite_path = Path(__file__).parent / "data" / "telemetry.db"
        if json_dir is None:
            json_dir = Path(__file__).parent / "data" / "traces"

        _telemetry_instance = TelemetryCollector(
            sqlite_path=sqlite_path,
            json_dir=json_dir
        )

    return _telemetry_instance


def get_current_trace() -> Optional[Trace]:
    """Convenience function to get current trace"""
    if _telemetry_instance:
        return _telemetry_instance.get_current_trace()
    return None
