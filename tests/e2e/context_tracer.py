"""
Context Pipeline Tracer

Captures every layer of context assembly, routing decisions, Claude Code
invocations, and tool executions for comprehensive E2E testing and debugging.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import json
import hashlib


class TraceStage(Enum):
    """Stages of the pipeline that can be traced."""
    INITIALIZED = "initialized"
    CONTEXT_TELOS = "context_telos"
    CONTEXT_TASKS = "context_tasks"
    CONTEXT_AUTO = "context_auto"
    CONTEXT_MEMORY = "context_memory"
    CONTEXT_ASSEMBLED = "context_assembled"
    ROUTING_SEMANTIC = "routing_semantic"
    ROUTING_PATTERN = "routing_pattern"
    ROUTING_CLAUDE = "routing_claude"
    ROUTING_DECIDED = "routing_decided"
    PROMPT_ASSEMBLED = "prompt_assembled"
    CLAUDE_INVOKED = "claude_invoked"
    TOOL_EXECUTING = "tool_executing"
    TOOL_COMPLETED = "tool_completed"
    ITERATION_COMPLETE = "iteration_complete"
    RESPONSE_GENERATED = "response_generated"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ContextLayer:
    """A single layer of context that was assembled."""
    layer_name: str  # telos_profile, telos_goals, task_context, auto_context, memory
    source_path: Optional[str] = None  # File path if applicable
    content: str = ""
    content_length: int = 0
    loaded_successfully: bool = True
    load_duration_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.content and not self.content_length:
            self.content_length = len(self.content)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "source_path": self.source_path,
            "content_preview": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "content_length": self.content_length,
            "loaded_successfully": self.loaded_successfully,
            "load_duration_ms": self.load_duration_ms,
            "metadata": self.metadata,
        }


@dataclass
class RoutingDecision:
    """Complete record of how a query was routed."""
    # Semantic routing
    semantic_enabled: bool = True
    semantic_score: float = 0.0
    semantic_matched_skill: Optional[str] = None
    semantic_matched_utterance: Optional[str] = None
    semantic_all_scores: List[Tuple[str, float]] = field(default_factory=list)
    semantic_duration_ms: int = 0

    # Pattern detection
    pattern_detected: bool = False
    pattern_name: Optional[str] = None
    pattern_pipeline: Optional[List[str]] = None
    pattern_confidence: float = 0.0

    # Active skill continuation
    active_skill_used: bool = False
    active_skill_name: Optional[str] = None

    # Claude fallback
    claude_fallback_used: bool = False
    claude_routing_prompt: Optional[str] = None
    claude_routing_response: Optional[str] = None
    claude_routing_duration_ms: int = 0

    # Final decision
    final_skill: str = ""
    final_method: str = ""  # semantic_direct, semantic_trusted, pattern, pipeline, claude_fallback, active_skill
    final_confidence: float = 0.0

    # Thresholds used
    bypass_threshold: float = 0.85
    confirm_threshold: float = 0.45

    def to_dict(self) -> Dict[str, Any]:
        return {
            "semantic": {
                "enabled": self.semantic_enabled,
                "score": self.semantic_score,
                "matched_skill": self.semantic_matched_skill,
                "matched_utterance": self.semantic_matched_utterance,
                "top_candidates": self.semantic_all_scores[:5],
                "duration_ms": self.semantic_duration_ms,
            },
            "pattern": {
                "detected": self.pattern_detected,
                "name": self.pattern_name,
                "pipeline": self.pattern_pipeline,
                "confidence": self.pattern_confidence,
            },
            "active_skill": {
                "used": self.active_skill_used,
                "name": self.active_skill_name,
            },
            "claude_fallback": {
                "used": self.claude_fallback_used,
                "duration_ms": self.claude_routing_duration_ms,
            },
            "decision": {
                "skill": self.final_skill,
                "method": self.final_method,
                "confidence": self.final_confidence,
            },
            "thresholds": {
                "bypass": self.bypass_threshold,
                "confirm": self.confirm_threshold,
            },
        }


@dataclass
class LLMInvocation:
    """Record of a single LLM (Claude Code) invocation."""
    invocation_id: str = ""
    iteration: int = 1

    # Command construction
    command: List[str] = field(default_factory=list)
    working_dir: str = ""
    timeout_seconds: int = 180

    # Session info
    claude_session_id: Optional[str] = None
    turn_number: int = 1

    # Input
    system_prompt: str = ""
    system_prompt_length: int = 0
    system_prompt_sections: Dict[str, int] = field(default_factory=dict)  # section -> char count
    user_message: str = ""
    user_message_length: int = 0
    messages_history: List[Dict[str, str]] = field(default_factory=list)

    # Output
    response_raw: str = ""
    response_length: int = 0
    tool_calls_detected: int = 0

    # Timing
    timestamp_start: Optional[datetime] = None
    timestamp_end: Optional[datetime] = None
    duration_ms: int = 0

    # Status
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "invocation_id": self.invocation_id,
            "iteration": self.iteration,
            "command_preview": " ".join(self.command[:3]) + "..." if self.command else "",
            "session": {
                "claude_session_id": self.claude_session_id,
                "turn_number": self.turn_number,
            },
            "input": {
                "system_prompt_length": self.system_prompt_length,
                "system_prompt_sections": self.system_prompt_sections,
                "user_message_length": self.user_message_length,
                "messages_history_count": len(self.messages_history),
            },
            "output": {
                "response_length": self.response_length,
                "tool_calls_detected": self.tool_calls_detected,
            },
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class ToolCallDetail:
    """Detailed trace of a single tool call."""
    call_id: str = ""
    iteration: int = 1
    sequence_in_iteration: int = 1

    # Tool info
    tool_name: str = ""
    skill_name: str = ""

    # Arguments
    args_from_llm: Dict[str, Any] = field(default_factory=dict)
    args_normalized: Dict[str, Any] = field(default_factory=dict)
    args_with_deps: Dict[str, Any] = field(default_factory=dict)

    # Dependencies
    dependencies_available: List[str] = field(default_factory=list)
    dependencies_used: List[str] = field(default_factory=list)

    # Execution
    timestamp_start: Optional[datetime] = None
    timestamp_end: Optional[datetime] = None
    duration_ms: int = 0

    # Result
    result: str = ""
    result_length: int = 0
    result_truncated: bool = False
    result_type: str = "success"  # success, error, empty, truncated

    # Error
    error: Optional[str] = None
    error_traceback: Optional[str] = None

    # Context at time of call
    messages_before_call: List[Dict[str, str]] = field(default_factory=list)
    system_prompt_at_call: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "iteration": self.iteration,
            "sequence": self.sequence_in_iteration,
            "tool": {
                "name": self.tool_name,
                "skill": self.skill_name,
            },
            "args": {
                "from_llm": self.args_from_llm,
                "normalized": self.args_normalized,
            },
            "dependencies": {
                "available": self.dependencies_available,
                "used": self.dependencies_used,
            },
            "result": {
                "type": self.result_type,
                "length": self.result_length,
                "truncated": self.result_truncated,
                "preview": self.result[:200] + "..." if len(self.result) > 200 else self.result,
            },
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


@dataclass
class ContextPipelineTrace:
    """
    Complete trace of context assembly for a single request.

    This captures every layer of the Workshop context pipeline:
    1. Telos Personal Context (profile, goals, mission, project)
    2. Task Context (active tasks, work evidence)
    3. Automatic Context (active files, recent edits, workflow detection)
    4. Memory Context (semantic search results, recent messages)
    5. Routing Decision (semantic scores, pattern detection, fallback)
    6. System Prompt Assembly (all sections, total size)
    7. Claude Code Invocation (command, session, response)
    8. Tool Execution Loop (each tool call with full context)
    9. Final Response
    """

    # Identification
    trace_id: str = ""
    session_id: str = ""
    timestamp_start: Optional[datetime] = None
    timestamp_end: Optional[datetime] = None

    # Current stage
    current_stage: TraceStage = TraceStage.INITIALIZED
    stage_history: List[Tuple[TraceStage, datetime]] = field(default_factory=list)

    # Input
    user_input_raw: str = ""
    user_input_enriched: str = ""

    # Layer 1: Telos Personal Context
    telos_loaded: bool = False
    telos_layers: List[ContextLayer] = field(default_factory=list)
    telos_project_detected: Optional[str] = None
    telos_total_chars: int = 0
    telos_load_duration_ms: int = 0

    # Layer 2: Task Context
    tasks_session_id: str = ""
    tasks_bound_correctly: bool = True
    tasks_active: List[Dict[str, Any]] = field(default_factory=list)
    tasks_pending_count: int = 0
    tasks_completed_count: int = 0
    tasks_in_progress_count: int = 0
    task_original_request: str = ""
    work_evidence_tools: List[str] = field(default_factory=list)
    task_context_formatted: str = ""
    task_context_length: int = 0

    # Layer 3: Automatic Context
    auto_context_enabled: bool = True
    auto_context_injection_reason: str = ""
    active_files: List[str] = field(default_factory=list)
    recent_edits: List[Dict[str, Any]] = field(default_factory=list)
    detected_workflow: str = ""
    workflow_confidence: float = 0.0
    auto_context_formatted: str = ""
    auto_context_length: int = 0

    # Layer 4: Memory Context
    memory_search_performed: bool = False
    memory_search_query: str = ""
    memory_results: List[str] = field(default_factory=list)
    memory_results_count: int = 0
    recent_messages_included: int = 0
    memory_context_formatted: str = ""
    memory_context_length: int = 0

    # Routing Decision
    routing: RoutingDecision = field(default_factory=RoutingDecision)
    routing_total_duration_ms: int = 0

    # System Prompt Assembly
    system_prompt_full: str = ""
    system_prompt_length: int = 0
    system_prompt_sections: Dict[str, str] = field(default_factory=dict)
    system_prompt_section_sizes: Dict[str, int] = field(default_factory=dict)
    skill_system_prompt: str = ""
    skill_tools_available: List[str] = field(default_factory=list)

    # Claude Code Invocations
    llm_invocations: List[LLMInvocation] = field(default_factory=list)
    llm_total_invocations: int = 0
    llm_total_duration_ms: int = 0

    # Tool Execution
    tool_calls: List[ToolCallDetail] = field(default_factory=list)
    tool_total_calls: int = 0
    tool_total_duration_ms: int = 0
    max_iterations: int = 100
    iterations_used: int = 0

    # Response
    response_raw: str = ""
    response_final: str = ""
    response_length: int = 0

    # Timing breakdown
    duration_context_ms: int = 0
    duration_routing_ms: int = 0
    duration_llm_ms: int = 0
    duration_tools_ms: int = 0
    duration_total_ms: int = 0

    # Status
    success: bool = True
    error_stage: Optional[TraceStage] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.trace_id:
            self.trace_id = self._generate_trace_id()
        if not self.timestamp_start:
            self.timestamp_start = datetime.now()

    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID."""
        import uuid
        return f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def set_stage(self, stage: TraceStage):
        """Update current stage and record in history."""
        self.current_stage = stage
        self.stage_history.append((stage, datetime.now()))

    def add_telos_layer(self, layer: ContextLayer):
        """Add a Telos context layer."""
        self.telos_layers.append(layer)
        self.telos_total_chars += layer.content_length
        if layer.loaded_successfully:
            self.telos_loaded = True

    def add_llm_invocation(self, invocation: LLMInvocation):
        """Add an LLM invocation record."""
        self.llm_invocations.append(invocation)
        self.llm_total_invocations += 1
        self.llm_total_duration_ms += invocation.duration_ms

    def add_tool_call(self, tool_call: ToolCallDetail):
        """Add a tool call record."""
        self.tool_calls.append(tool_call)
        self.tool_total_calls += 1
        self.tool_total_duration_ms += tool_call.duration_ms

    def complete(self, response: str):
        """Mark trace as complete with final response."""
        self.timestamp_end = datetime.now()
        self.response_final = response
        self.response_length = len(response)
        self.set_stage(TraceStage.COMPLETED)

        if self.timestamp_start and self.timestamp_end:
            self.duration_total_ms = int(
                (self.timestamp_end - self.timestamp_start).total_seconds() * 1000
            )

    def fail(self, stage: TraceStage, error: str, traceback: str = None):
        """Mark trace as failed."""
        self.timestamp_end = datetime.now()
        self.success = False
        self.error_stage = stage
        self.error_message = error
        self.error_traceback = traceback
        self.set_stage(TraceStage.ERROR)

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary for serialization."""
        return {
            "identification": {
                "trace_id": self.trace_id,
                "session_id": self.session_id,
                "timestamp_start": self.timestamp_start.isoformat() if self.timestamp_start else None,
                "timestamp_end": self.timestamp_end.isoformat() if self.timestamp_end else None,
            },
            "input": {
                "user_input_raw": self.user_input_raw,
                "user_input_enriched_length": len(self.user_input_enriched),
            },
            "context": {
                "telos": {
                    "loaded": self.telos_loaded,
                    "layers": [l.to_dict() for l in self.telos_layers],
                    "project_detected": self.telos_project_detected,
                    "total_chars": self.telos_total_chars,
                },
                "tasks": {
                    "session_bound": self.tasks_bound_correctly,
                    "pending": self.tasks_pending_count,
                    "in_progress": self.tasks_in_progress_count,
                    "completed": self.tasks_completed_count,
                    "original_request": self.task_original_request,
                    "context_length": self.task_context_length,
                },
                "auto": {
                    "enabled": self.auto_context_enabled,
                    "injection_reason": self.auto_context_injection_reason,
                    "active_files": self.active_files,
                    "workflow": self.detected_workflow,
                    "workflow_confidence": self.workflow_confidence,
                    "context_length": self.auto_context_length,
                },
                "memory": {
                    "search_performed": self.memory_search_performed,
                    "results_count": self.memory_results_count,
                    "recent_messages": self.recent_messages_included,
                    "context_length": self.memory_context_length,
                },
            },
            "routing": self.routing.to_dict(),
            "system_prompt": {
                "total_length": self.system_prompt_length,
                "sections": self.system_prompt_section_sizes,
                "tools_available": self.skill_tools_available,
            },
            "llm_invocations": [inv.to_dict() for inv in self.llm_invocations],
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "response": {
                "length": self.response_length,
                "preview": self.response_final[:500] + "..." if len(self.response_final) > 500 else self.response_final,
            },
            "timing": {
                "context_ms": self.duration_context_ms,
                "routing_ms": self.duration_routing_ms,
                "llm_ms": self.duration_llm_ms,
                "tools_ms": self.duration_tools_ms,
                "total_ms": self.duration_total_ms,
            },
            "status": {
                "success": self.success,
                "stage": self.current_stage.value,
                "error_stage": self.error_stage.value if self.error_stage else None,
                "error_message": self.error_message,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert trace to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def get_content_hash(self) -> str:
        """Generate a hash of the trace content for comparison."""
        content = f"{self.user_input_raw}:{self.routing.final_skill}:{self.tool_total_calls}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class PipelineTracer:
    """
    Context manager for tracing a complete pipeline execution.

    Usage:
        async with PipelineTracer(user_input) as tracer:
            # Context assembly
            tracer.trace.add_telos_layer(...)
            tracer.trace.set_stage(TraceStage.CONTEXT_ASSEMBLED)

            # Routing
            tracer.trace.routing.semantic_score = 0.91
            tracer.trace.set_stage(TraceStage.ROUTING_DECIDED)

            # ... etc

        # tracer.trace now contains complete pipeline trace
    """

    def __init__(self, user_input: str, session_id: str = ""):
        self.trace = ContextPipelineTrace(
            user_input_raw=user_input,
            session_id=session_id,
        )
        self._start_time = None

    async def __aenter__(self):
        self._start_time = datetime.now()
        self.trace.timestamp_start = self._start_time
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.trace.timestamp_end = datetime.now()
        if exc_type is not None:
            self.trace.fail(
                self.trace.current_stage,
                str(exc_val),
                str(exc_tb) if exc_tb else None
            )
        return False  # Don't suppress exceptions

    def start_context_layer(self, layer_name: str, source_path: str = None) -> ContextLayer:
        """Start tracing a context layer."""
        return ContextLayer(
            layer_name=layer_name,
            source_path=source_path,
        )

    def start_llm_invocation(self, iteration: int = 1) -> LLMInvocation:
        """Start tracing an LLM invocation."""
        import uuid
        return LLMInvocation(
            invocation_id=f"llm_{uuid.uuid4().hex[:8]}",
            iteration=iteration,
            timestamp_start=datetime.now(),
        )

    def start_tool_call(self, tool_name: str, skill_name: str, iteration: int = 1) -> ToolCallDetail:
        """Start tracing a tool call."""
        import uuid
        return ToolCallDetail(
            call_id=f"tool_{uuid.uuid4().hex[:8]}",
            tool_name=tool_name,
            skill_name=skill_name,
            iteration=iteration,
            timestamp_start=datetime.now(),
        )
