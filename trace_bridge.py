"""
Trace Bridge

Bridges between the existing Telemetry system (telemetry.py) and the new
E2E ContextPipelineTrace format (tests/e2e/context_tracer.py).

This allows us to leverage existing instrumentation while enabling the
new trace visualization capabilities for --trace-mode.
"""

from datetime import datetime
from typing import Optional

from telemetry import Trace, ToolCallTrace, LLMCall
from tests.e2e.context_tracer import (
    ContextPipelineTrace,
    RoutingDecision,
    ToolCallDetail,
    LLMInvocation,
    ContextLayer,
    TraceStage,
)


def telemetry_to_pipeline_trace(trace: Trace) -> ContextPipelineTrace:
    """
    Convert a telemetry.Trace to ContextPipelineTrace.

    This preserves backward compatibility while enabling the new
    visualization and export capabilities of --trace-mode.

    Args:
        trace: The existing telemetry.Trace object

    Returns:
        A ContextPipelineTrace with all data mapped
    """
    pipeline = ContextPipelineTrace(
        user_input_raw=trace.user_input_raw,
        session_id=trace.session_id or "unknown",
    )

    # Basic identification
    pipeline.trace_id = trace.trace_id

    # Copy timing
    pipeline.duration_total_ms = trace.total_duration_ms or 0
    pipeline.duration_routing_ms = trace.intent_detection_time_ms or 0
    pipeline.duration_llm_ms = sum(llm.duration_ms for llm in trace.llm_calls) if trace.llm_calls else 0
    pipeline.duration_tools_ms = sum(tc.duration_ms for tc in trace.tool_calls) if trace.tool_calls else 0
    pipeline.duration_context_ms = max(0, pipeline.duration_total_ms - pipeline.duration_routing_ms - pipeline.duration_llm_ms - pipeline.duration_tools_ms)

    # Copy input
    pipeline.user_input_enriched = trace.user_input_enriched or trace.user_input_raw

    # Copy routing decision
    pipeline.routing.semantic_enabled = True  # Assume semantic routing is used
    pipeline.routing.final_skill = trace.intent_tool_name or "general"
    pipeline.routing.final_confidence = trace.intent_detection_time_ms / 1000 if trace.intent_detection_time_ms else 0
    pipeline.routing.final_method = "pattern" if trace.intent_pattern_matched else "semantic"

    if trace.intent_pattern_matched:
        pipeline.routing.pattern_detected = True
        pipeline.routing.pattern_name = trace.intent_pattern_matched

    # Copy Telos context
    pipeline.telos_loaded = trace.telos_loaded
    pipeline.telos_project_detected = trace.telos_active_project

    if trace.telos_profile:
        pipeline.add_telos_layer(ContextLayer(
            layer_name="profile",
            content=trace.telos_profile,
            content_length=len(trace.telos_profile),
            loaded_successfully=True,
        ))

    if trace.telos_goals:
        pipeline.add_telos_layer(ContextLayer(
            layer_name="goals",
            content=trace.telos_goals,
            content_length=len(trace.telos_goals),
            loaded_successfully=True,
        ))

    if trace.telos_project_context:
        pipeline.add_telos_layer(ContextLayer(
            layer_name=f"project:{trace.telos_active_project or 'active'}",
            content=trace.telos_project_context,
            content_length=len(trace.telos_project_context),
            loaded_successfully=True,
        ))

    # Copy auto context
    pipeline.auto_context_enabled = trace.context_manager_used
    pipeline.active_files = trace.active_files or []
    pipeline.recent_edits = trace.recent_changes or []
    pipeline.detected_workflow = trace.detected_workflow or ""
    pipeline.workflow_confidence = trace.detected_workflow_confidence or 0.0
    pipeline.auto_context_length = trace.context_assembled_length or 0
    pipeline.auto_context_injection_reason = trace.context_should_inject_reason or ""

    # Copy memory context
    pipeline.memory_search_performed = bool(trace.memory_search_query)
    pipeline.memory_search_query = trace.memory_search_query or ""
    pipeline.memory_results = trace.memory_results or []
    pipeline.memory_results_count = trace.memory_results_count or 0
    pipeline.recent_messages_included = trace.recent_messages_count or 0

    # Copy context sources
    for src in trace.context_sources or []:
        if "telos" in src.source_type.lower():
            pipeline.add_telos_layer(ContextLayer(
                layer_name=src.source_type,
                source_path=src.source_path,
                content=src.content,
                content_length=src.content_length,
                loaded_successfully=True,
            ))

    # Copy LLM calls
    for llm in trace.llm_calls or []:
        invocation = LLMInvocation(
            invocation_id=llm.call_id,
            iteration=llm.iteration,
            system_prompt=llm.system_prompt,
            system_prompt_length=llm.system_prompt_length,
            user_message_length=len(llm.messages[-1].get("content", "")) if llm.messages else 0,
            messages_history=llm.messages,
            response_raw=llm.response_raw,
            response_length=llm.response_length,
            tool_calls_detected=len(llm.tool_calls_extracted),
            timestamp_start=llm.timestamp_start,
            timestamp_end=llm.timestamp_end,
            duration_ms=llm.duration_ms,
            success=True,
        )
        pipeline.add_llm_invocation(invocation)

    # Copy tool calls
    for tc in trace.tool_calls or []:
        tool_detail = ToolCallDetail(
            call_id=tc.call_id,
            tool_name=tc.tool_name,
            skill_name=tc.skill_name,
            args_from_llm=tc.args_raw,
            args_normalized=tc.args_normalized,
            dependencies_available=tc.dependencies_available,
            dependencies_used=tc.dependencies_used,
            timestamp_start=tc.timestamp_start,
            timestamp_end=tc.timestamp_end,
            duration_ms=tc.duration_ms,
            result=tc.result_raw,
            result_length=tc.result_length,
            result_type=tc.result_type,
            error=tc.error_message,
            error_traceback=tc.error_traceback,
        )
        pipeline.add_tool_call(tool_detail)

    # Copy response
    pipeline.response_raw = trace.response_raw or ""
    pipeline.response_final = trace.response_final or ""
    pipeline.response_length = trace.response_length or 0

    # Set status
    if trace.error_occurred:
        pipeline.success = False
        pipeline.error_message = trace.error_message
        pipeline.error_traceback = trace.error_traceback
        # Map error stage
        stage_map = {
            "context": TraceStage.CONTEXT_ASSEMBLED,
            "intent": TraceStage.ROUTING_DECIDED,
            "llm": TraceStage.CLAUDE_INVOKED,
            "tool": TraceStage.TOOL_EXECUTING,
            "format": TraceStage.RESPONSE_GENERATED,
        }
        pipeline.error_stage = stage_map.get(trace.error_stage, TraceStage.ERROR)
        pipeline.set_stage(TraceStage.ERROR)
    else:
        pipeline.success = True
        pipeline.set_stage(TraceStage.COMPLETED)

    pipeline.timestamp_end = trace.timestamp_end

    return pipeline


def create_pipeline_tracer_from_input(
    user_input: str,
    session_id: str = "",
) -> ContextPipelineTrace:
    """
    Create a fresh ContextPipelineTrace for a new request.

    This is used when trace mode is enabled to create a trace
    that will be populated during pipeline execution.

    Args:
        user_input: The raw user input
        session_id: The current session ID

    Returns:
        A new ContextPipelineTrace ready for instrumentation
    """
    return ContextPipelineTrace(
        user_input_raw=user_input,
        session_id=session_id,
        timestamp_start=datetime.now(),
    )


def merge_telemetry_into_pipeline_trace(
    pipeline: ContextPipelineTrace,
    trace: Trace,
) -> ContextPipelineTrace:
    """
    Merge data from a telemetry.Trace into an existing ContextPipelineTrace.

    This is useful when you've been building up a pipeline trace during
    execution and want to also capture telemetry data.

    Args:
        pipeline: The existing pipeline trace
        trace: The telemetry trace to merge from

    Returns:
        The updated pipeline trace
    """
    # Only merge fields that haven't been set in the pipeline trace
    if not pipeline.duration_total_ms and trace.total_duration_ms:
        pipeline.duration_total_ms = trace.total_duration_ms

    if not pipeline.tool_calls and trace.tool_calls:
        for tc in trace.tool_calls:
            tool_detail = ToolCallDetail(
                call_id=tc.call_id,
                tool_name=tc.tool_name,
                skill_name=tc.skill_name,
                args_from_llm=tc.args_raw,
                args_normalized=tc.args_normalized,
                duration_ms=tc.duration_ms,
                result=tc.result_raw,
                result_length=tc.result_length,
                result_type=tc.result_type,
                error=tc.error_message,
            )
            pipeline.add_tool_call(tool_detail)

    if not pipeline.llm_invocations and trace.llm_calls:
        for llm in trace.llm_calls:
            invocation = LLMInvocation(
                invocation_id=llm.call_id,
                iteration=llm.iteration,
                system_prompt_length=llm.system_prompt_length,
                response_length=llm.response_length,
                duration_ms=llm.duration_ms,
            )
            pipeline.add_llm_invocation(invocation)

    return pipeline
