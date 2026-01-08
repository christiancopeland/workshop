"""
Dashboard Integration Helper

Provides a simple interface for emitting dashboard events from anywhere in Workshop.
Automatically checks if dashboard is enabled before emitting.
"""

import asyncio
from typing import Optional, Dict, Any

# Dashboard singleton - will be set when dashboard is started
_dashboard = None


def set_dashboard(dashboard):
    """Set the global dashboard instance."""
    global _dashboard
    _dashboard = dashboard


def get_dashboard():
    """Get the global dashboard instance."""
    return _dashboard


def is_enabled() -> bool:
    """Check if dashboard is enabled and connected."""
    return _dashboard is not None


async def emit(event_type, data: Dict = None, trace_id: str = None):
    """Emit an event if dashboard is enabled."""
    if _dashboard:
        await _dashboard.emit(event_type, data, trace_id)


# Convenience functions that handle the async call in sync contexts
def emit_sync(event_type, data: Dict = None, trace_id: str = None):
    """Emit an event synchronously (for use in sync code)."""
    if _dashboard:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_dashboard.emit(event_type, data, trace_id))
            else:
                loop.run_until_complete(_dashboard.emit(event_type, data, trace_id))
        except RuntimeError:
            # No event loop - skip
            pass


# Specific event helpers
async def user_input(text: str, trace_id: str = None):
    """Emit user input event."""
    if _dashboard:
        await _dashboard.emit_user_input(text, trace_id)


async def tool_calling(tool_name: str, skill_name: str, args: Dict, call_id: str = None, trace_id: str = None) -> Optional[str]:
    """Emit tool calling event. Returns call_id."""
    if _dashboard:
        return await _dashboard.emit_tool_calling(tool_name, skill_name, args, call_id, trace_id)
    return call_id


async def tool_result(call_id: str, result: str, duration_ms: int = None, trace_id: str = None):
    """Emit tool result event."""
    if _dashboard:
        await _dashboard.emit_tool_result(call_id, result, duration_ms, trace_id)


async def tool_error(call_id: str, error: str, trace_id: str = None):
    """Emit tool error event."""
    if _dashboard:
        await _dashboard.emit_tool_error(call_id, error, trace_id)


async def llm_calling(model: str, message_count: int, messages: list = None, system_prompt: str = None, trace_id: str = None):
    """Emit LLM calling event with full message context."""
    if _dashboard:
        await _dashboard.emit_llm_calling(model, message_count, messages, system_prompt, trace_id)


async def llm_complete(response_length: int, duration_ms: int, tool_calls: int = 0, trace_id: str = None):
    """Emit LLM complete event."""
    if _dashboard:
        await _dashboard.emit_llm_complete(response_length, duration_ms, tool_calls, trace_id)


async def workflow_started(workflow_name: str, skill_name: str, trigger: str = None, trace_id: str = None):
    """Emit workflow started event."""
    if _dashboard:
        await _dashboard.emit_workflow_started(workflow_name, skill_name, trigger, trace_id)


async def skill_matched(skill_name: str, reason: str, trace_id: str = None):
    """Emit skill matched event."""
    if _dashboard:
        await _dashboard.emit_skill_matched(skill_name, reason, trace_id)


async def research_started(query: str, trace_id: str = None):
    """Emit research started event."""
    if _dashboard:
        await _dashboard.emit_research_started(query, trace_id)


async def research_query(query: str, angle: str = None, trace_id: str = None):
    """Emit research query event."""
    if _dashboard:
        await _dashboard.emit_research_query(query, angle, trace_id)


async def research_fetching(url: str, title: str = None, trace_id: str = None):
    """Emit research fetching event."""
    if _dashboard:
        await _dashboard.emit_research_fetching(url, title, trace_id)


async def research_complete(source_count: int, queries: int, trace_id: str = None):
    """Emit research complete event."""
    if _dashboard:
        await _dashboard.emit_research_complete(source_count, queries, trace_id)


async def response(text: str, trace_id: str = None):
    """Emit assistant response event."""
    if _dashboard:
        await _dashboard.emit_response(text, trace_id)


async def error(error: str, stage: str = None, trace_id: str = None):
    """Emit error event."""
    if _dashboard:
        await _dashboard.emit_error(error, stage, trace_id)


async def info(message: str, trace_id: str = None):
    """Emit info event."""
    if _dashboard:
        await _dashboard.emit_info(message, trace_id)


async def context_loading(sources: list, trace_id: str = None):
    """Emit context loading event."""
    if _dashboard:
        await _dashboard.emit_context_loading(sources, trace_id)


async def context_loaded(sources: list, total_length: int, trace_id: str = None):
    """Emit context loaded event."""
    if _dashboard:
        await _dashboard.emit_context_loaded(sources, total_length, trace_id)


async def intent_detected(tool_name: str, skill_name: str, pattern: str = None, trace_id: str = None):
    """Emit intent detected event."""
    if _dashboard:
        await _dashboard.emit_intent_detected(tool_name, skill_name, pattern, trace_id)


# === Agent State Events (for orchestration dashboard) ===

async def agent_state(state: str, details: str = None, trace_id: str = None):
    """
    Emit agent state change event.

    States: idle, routing, thinking, executing, synthesizing, responding
    """
    if _dashboard:
        await _dashboard.emit("agent_state", {
            "state": state,
            "details": details,
        }, trace_id)


async def routing_started(user_input: str, trace_id: str = None):
    """Emit routing started event."""
    if _dashboard:
        await _dashboard.emit("routing_started", {
            "input": user_input[:100] + "..." if len(user_input) > 100 else user_input,
        }, trace_id)


async def thinking_started(model: str, skill_name: str = None, trace_id: str = None):
    """Emit thinking started event (LLM is generating)."""
    if _dashboard:
        await _dashboard.emit("thinking_started", {
            "model": model,
            "skill": skill_name,
        }, trace_id)


async def thinking_progress(elapsed_ms: int, tokens_so_far: int = 0, trace_id: str = None):
    """Emit thinking progress event."""
    if _dashboard:
        await _dashboard.emit("thinking_progress", {
            "elapsed_ms": elapsed_ms,
            "tokens": tokens_so_far,
        }, trace_id)


async def model_loading(model: str, trace_id: str = None):
    """Emit model loading event (useful to show VRAM activity)."""
    if _dashboard:
        await _dashboard.emit("model_loading", {
            "model": model,
        }, trace_id)


async def model_ready(model: str, load_time_ms: int = None, trace_id: str = None):
    """Emit model ready event."""
    if _dashboard:
        await _dashboard.emit("model_ready", {
            "model": model,
            "load_time_ms": load_time_ms,
        }, trace_id)


# === Pattern Events (Fabric-style patterns) ===

async def pattern_detected(pattern_name: str, method: str, trace_id: str = None):
    """
    Emit pattern detection event.

    Args:
        pattern_name: Name of the detected pattern (e.g., "extract_wisdom")
        method: Detection method ("trigger", "pipeline", "explicit")
    """
    if _dashboard:
        await _dashboard.emit("pattern_detected", {
            "pattern": pattern_name,
            "method": method,
        }, trace_id)


async def pipeline_detected(patterns: list, trace_id: str = None):
    """
    Emit pipeline detection event (multiple patterns chained).

    Args:
        patterns: List of pattern names in execution order
    """
    if _dashboard:
        await _dashboard.emit("pipeline_detected", {
            "patterns": patterns,
            "stage_count": len(patterns),
        }, trace_id)


async def pattern_executing(pattern_name: str, stage: int = 1, total_stages: int = 1, trace_id: str = None):
    """
    Emit pattern execution started event.

    Args:
        pattern_name: Name of the pattern being executed
        stage: Current stage in pipeline (1-indexed)
        total_stages: Total number of stages in pipeline
    """
    if _dashboard:
        await _dashboard.emit("pattern_executing", {
            "pattern": pattern_name,
            "stage": stage,
            "total_stages": total_stages,
        }, trace_id)


async def pattern_complete(pattern_name: str, output_length: int, duration_ms: int, stage: int = 1, trace_id: str = None):
    """
    Emit pattern execution complete event.

    Args:
        pattern_name: Name of the completed pattern
        output_length: Length of the output text
        duration_ms: Execution time in milliseconds
        stage: Stage number in pipeline
    """
    if _dashboard:
        await _dashboard.emit("pattern_complete", {
            "pattern": pattern_name,
            "output_length": output_length,
            "duration_ms": duration_ms,
            "stage": stage,
        }, trace_id)


async def pipeline_complete(patterns: list, total_duration_ms: int, trace_id: str = None):
    """
    Emit pipeline execution complete event.

    Args:
        patterns: List of pattern names that were executed
        total_duration_ms: Total execution time for the pipeline
    """
    if _dashboard:
        await _dashboard.emit("pipeline_complete", {
            "patterns": patterns,
            "total_duration_ms": total_duration_ms,
            "stage_count": len(patterns),
        }, trace_id)
