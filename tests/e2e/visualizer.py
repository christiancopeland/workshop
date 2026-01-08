"""
Trace Visualizer

Generates ASCII and markdown visualizations of pipeline traces
for debugging and test reports.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import textwrap

from .context_tracer import (
    ContextPipelineTrace,
    ContextLayer,
    ToolCallDetail,
    LLMInvocation,
    RoutingDecision,
    TraceStage,
)


class TraceVisualizer:
    """
    Generates visual representations of pipeline traces.

    Usage:
        visualizer = TraceVisualizer()
        ascii_output = visualizer.to_ascii(trace)
        markdown_output = visualizer.to_markdown(trace)

        # Print to console
        print(visualizer.to_ascii(trace))

        # Save to file
        with open("trace_report.md", "w") as f:
            f.write(visualizer.to_markdown(trace))
    """

    def __init__(self, width: int = 80):
        self.width = width
        self.box_width = width - 4  # Account for box borders

    def to_ascii(self, trace: ContextPipelineTrace) -> str:
        """Generate ASCII visualization of trace."""
        lines = []

        # Header
        lines.append(self._header_box(f"E2E TEST TRACE: {trace.trace_id}"))
        lines.append("")

        # Input section
        lines.append(f"INPUT: \"{self._truncate(trace.user_input_raw, 60)}\"")
        lines.append(self._divider())
        lines.append("")

        # Context Assembly
        lines.extend(self._context_section(trace))
        lines.append("")

        # Routing Decision
        lines.extend(self._routing_section(trace))
        lines.append("")

        # System Prompt
        lines.extend(self._prompt_section(trace))
        lines.append("")

        # Claude Invocations
        lines.extend(self._llm_section(trace))
        lines.append("")

        # Tool Execution
        lines.extend(self._tools_section(trace))
        lines.append("")

        # Response
        lines.extend(self._response_section(trace))
        lines.append("")

        # Status footer
        status = "PASSED" if trace.success else f"FAILED at {trace.error_stage.value if trace.error_stage else 'unknown'}"
        symbol = "✓" if trace.success else "✗"
        lines.append(self._header_box(f"{symbol} TEST {status}"))

        return "\n".join(lines)

    def to_markdown(self, trace: ContextPipelineTrace) -> str:
        """Generate markdown report of trace."""
        lines = []

        # Header
        lines.append(f"# E2E Test Trace: {trace.trace_id}")
        lines.append("")
        lines.append(f"**Session:** {trace.session_id}")
        lines.append(f"**Started:** {trace.timestamp_start}")
        lines.append(f"**Duration:** {trace.duration_total_ms}ms")
        lines.append(f"**Status:** {'✅ PASSED' if trace.success else '❌ FAILED'}")
        lines.append("")

        # Input
        lines.append("## Input")
        lines.append("```")
        lines.append(trace.user_input_raw)
        lines.append("```")
        lines.append("")

        # Context Assembly
        lines.append("## Context Assembly")
        lines.append("")

        # Telos
        lines.append("### Layer 1: Telos Personal Context")
        lines.append(f"- **Loaded:** {trace.telos_loaded}")
        lines.append(f"- **Total Size:** {trace.telos_total_chars} chars")
        lines.append(f"- **Project Detected:** {trace.telos_project_detected or 'None'}")
        if trace.telos_layers:
            lines.append("")
            lines.append("| Layer | Source | Size | Status |")
            lines.append("|-------|--------|------|--------|")
            for layer in trace.telos_layers:
                status = "✅" if layer.loaded_successfully else "❌"
                lines.append(f"| {layer.layer_name} | {layer.source_path or '-'} | {layer.content_length} | {status} |")
        lines.append("")

        # Tasks
        lines.append("### Layer 2: Task Context")
        lines.append(f"- **Session Bound:** {trace.tasks_bound_correctly}")
        lines.append(f"- **Pending:** {trace.tasks_pending_count}")
        lines.append(f"- **In Progress:** {trace.tasks_in_progress_count}")
        lines.append(f"- **Completed:** {trace.tasks_completed_count}")
        lines.append(f"- **Context Size:** {trace.task_context_length} chars")
        lines.append("")

        # Auto Context
        lines.append("### Layer 3: Automatic Context")
        lines.append(f"- **Enabled:** {trace.auto_context_enabled}")
        lines.append(f"- **Injection Reason:** {trace.auto_context_injection_reason or 'N/A'}")
        lines.append(f"- **Workflow Detected:** {trace.detected_workflow or 'None'} ({trace.workflow_confidence:.0%})")
        lines.append(f"- **Active Files:** {len(trace.active_files)}")
        if trace.active_files:
            for f in trace.active_files[:5]:
                lines.append(f"  - {f}")
            if len(trace.active_files) > 5:
                lines.append(f"  - ... and {len(trace.active_files) - 5} more")
        lines.append("")

        # Memory
        lines.append("### Layer 4: Memory Context")
        lines.append(f"- **Search Performed:** {trace.memory_search_performed}")
        lines.append(f"- **Results Count:** {trace.memory_results_count}")
        lines.append(f"- **Recent Messages:** {trace.recent_messages_included}")
        lines.append("")

        # Routing
        lines.append("## Routing Decision")
        lines.append("")
        r = trace.routing
        lines.append(f"**Final Decision:** `{r.final_skill}` via `{r.final_method}` (confidence: {r.final_confidence:.2f})")
        lines.append("")

        if r.semantic_enabled:
            lines.append("### Semantic Routing")
            lines.append(f"- **Score:** {r.semantic_score:.3f}")
            lines.append(f"- **Matched Skill:** {r.semantic_matched_skill}")
            lines.append(f"- **Matched Utterance:** \"{r.semantic_matched_utterance}\"")
            if r.semantic_all_scores:
                lines.append("")
                lines.append("**Top Candidates:**")
                for skill, score in r.semantic_all_scores[:5]:
                    bar = "█" * int(score * 20)
                    lines.append(f"- {skill}: {score:.3f} {bar}")
            lines.append("")

        if r.pattern_detected:
            lines.append("### Pattern Detection")
            lines.append(f"- **Pattern:** {r.pattern_name}")
            lines.append(f"- **Pipeline:** {r.pattern_pipeline}")
            lines.append("")

        if r.claude_fallback_used:
            lines.append("### Claude Fallback")
            lines.append(f"- **Duration:** {r.claude_routing_duration_ms}ms")
            lines.append("")

        # System Prompt
        lines.append("## System Prompt Assembly")
        lines.append(f"**Total Length:** {trace.system_prompt_length} chars")
        lines.append("")
        if trace.system_prompt_section_sizes:
            lines.append("| Section | Size |")
            lines.append("|---------|------|")
            for section, size in trace.system_prompt_section_sizes.items():
                lines.append(f"| {section} | {size} |")
        lines.append("")

        if trace.skill_tools_available:
            lines.append("**Available Tools:**")
            for tool in trace.skill_tools_available:
                lines.append(f"- `{tool}`")
            lines.append("")

        # LLM Invocations
        lines.append("## Claude Code Invocations")
        lines.append(f"**Total:** {trace.llm_total_invocations}")
        lines.append(f"**Duration:** {trace.llm_total_duration_ms}ms")
        lines.append("")

        for inv in trace.llm_invocations:
            lines.append(f"### Invocation {inv.iteration}")
            lines.append(f"- **Session:** {inv.claude_session_id}")
            lines.append(f"- **Turn:** {inv.turn_number}")
            lines.append(f"- **System Prompt:** {inv.system_prompt_length} chars")
            lines.append(f"- **User Message:** {inv.user_message_length} chars")
            lines.append(f"- **Response:** {inv.response_length} chars")
            lines.append(f"- **Tool Calls:** {inv.tool_calls_detected}")
            lines.append(f"- **Duration:** {inv.duration_ms}ms")
            lines.append("")

        # Tool Calls
        lines.append("## Tool Execution")
        lines.append(f"**Total Calls:** {trace.tool_total_calls}")
        lines.append(f"**Total Duration:** {trace.tool_total_duration_ms}ms")
        lines.append(f"**Iterations Used:** {trace.iterations_used}/{trace.max_iterations}")
        lines.append("")

        for tc in trace.tool_calls:
            lines.append(f"### {tc.tool_name} (Iteration {tc.iteration})")
            lines.append(f"- **Skill:** {tc.skill_name}")
            lines.append(f"- **Duration:** {tc.duration_ms}ms")
            lines.append(f"- **Result Type:** {tc.result_type}")
            lines.append(f"- **Result Size:** {tc.result_length} chars")
            lines.append("")
            lines.append("**Arguments (from LLM):**")
            lines.append("```json")
            import json
            lines.append(json.dumps(tc.args_from_llm, indent=2))
            lines.append("```")
            if tc.error:
                lines.append(f"**Error:** {tc.error}")
            lines.append("")

        # Response
        lines.append("## Response")
        lines.append(f"**Length:** {trace.response_length} chars")
        lines.append("")
        lines.append("```")
        lines.append(trace.response_final[:1000])
        if len(trace.response_final) > 1000:
            lines.append(f"... ({len(trace.response_final) - 1000} more chars)")
        lines.append("```")
        lines.append("")

        # Timing
        lines.append("## Timing Breakdown")
        lines.append("")
        lines.append("| Stage | Duration | Percentage |")
        lines.append("|-------|----------|------------|")
        total = trace.duration_total_ms or 1
        stages = [
            ("Context", trace.duration_context_ms),
            ("Routing", trace.duration_routing_ms),
            ("LLM", trace.duration_llm_ms),
            ("Tools", trace.duration_tools_ms),
        ]
        for name, duration in stages:
            pct = (duration / total) * 100 if total > 0 else 0
            bar = "█" * int(pct / 5)
            lines.append(f"| {name} | {duration}ms | {pct:.1f}% {bar} |")
        lines.append(f"| **Total** | **{total}ms** | |")
        lines.append("")

        # Error (if any)
        if not trace.success:
            lines.append("## Error")
            lines.append(f"**Stage:** {trace.error_stage.value if trace.error_stage else 'unknown'}")
            lines.append(f"**Message:** {trace.error_message}")
            if trace.error_traceback:
                lines.append("")
                lines.append("**Traceback:**")
                lines.append("```")
                lines.append(trace.error_traceback)
                lines.append("```")
            lines.append("")

        return "\n".join(lines)

    def to_compact(self, trace: ContextPipelineTrace) -> str:
        """Generate compact one-line summary."""
        status = "✓" if trace.success else "✗"
        return (
            f"{status} {trace.trace_id} | "
            f"skill={trace.routing.final_skill} | "
            f"method={trace.routing.final_method} ({trace.routing.final_confidence:.2f}) | "
            f"tools={trace.tool_total_calls} | "
            f"{trace.duration_total_ms}ms"
        )

    def _header_box(self, text: str) -> str:
        """Create a header box."""
        padding = (self.width - len(text) - 2) // 2
        return (
            "═" * self.width + "\n" +
            " " * padding + text + "\n" +
            "═" * self.width
        )

    def _divider(self) -> str:
        """Create a divider line."""
        return "─" * self.width

    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate text with ellipsis."""
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."

    def _box(self, title: str, content: List[str]) -> List[str]:
        """Create a content box."""
        lines = []
        lines.append(f"┌{'─' * (self.box_width + 2)}┐")
        lines.append(f"│ {title.ljust(self.box_width)} │")
        lines.append(f"├{'─' * (self.box_width + 2)}┤")
        for line in content:
            wrapped = textwrap.wrap(line, self.box_width)
            for w in wrapped:
                lines.append(f"│ {w.ljust(self.box_width)} │")
        lines.append(f"└{'─' * (self.box_width + 2)}┘")
        return lines

    def _context_section(self, trace: ContextPipelineTrace) -> List[str]:
        """Generate context assembly section."""
        lines = [f"▼ CONTEXT ASSEMBLY (took {trace.duration_context_ms}ms)"]

        # Telos
        telos_lines = [
            f"├── Loaded: {'✓' if trace.telos_loaded else '✗'}",
            f"├── Total: {trace.telos_total_chars} chars",
            f"└── Project: {trace.telos_project_detected or 'None'}",
        ]
        lines.extend(self._box("LAYER 1: TELOS PERSONAL CONTEXT", telos_lines))

        # Tasks
        task_lines = [
            f"├── Session Bound: {'✓' if trace.tasks_bound_correctly else '✗'}",
            f"├── Pending: {trace.tasks_pending_count}",
            f"├── In Progress: {trace.tasks_in_progress_count}",
            f"└── Completed: {trace.tasks_completed_count}",
        ]
        lines.extend(self._box("LAYER 2: TASK CONTEXT", task_lines))

        # Auto context
        auto_lines = [
            f"├── Enabled: {'✓' if trace.auto_context_enabled else '✗'}",
            f"├── Reason: {trace.auto_context_injection_reason or 'N/A'}",
            f"├── Workflow: {trace.detected_workflow or 'None'} ({trace.workflow_confidence:.0%})",
            f"└── Active Files: {len(trace.active_files)}",
        ]
        lines.extend(self._box("LAYER 3: AUTOMATIC CONTEXT", auto_lines))

        # Memory
        mem_lines = [
            f"├── Search: {'✓' if trace.memory_search_performed else '✗'}",
            f"├── Results: {trace.memory_results_count}",
            f"└── Recent Messages: {trace.recent_messages_included}",
        ]
        lines.extend(self._box("LAYER 4: MEMORY CONTEXT", mem_lines))

        return lines

    def _routing_section(self, trace: ContextPipelineTrace) -> List[str]:
        """Generate routing decision section."""
        r = trace.routing
        lines = [f"▼ ROUTING DECISION (took {trace.routing_total_duration_ms}ms)"]

        content = [
            f"├── Semantic Score: {r.semantic_score:.3f}",
            f"├── Matched Skill: {r.semantic_matched_skill}",
            f"├── Threshold Check: {r.semantic_score:.2f} {'≥' if r.semantic_score >= r.bypass_threshold else '<'} {r.bypass_threshold}",
            f"├── Method: {r.final_method}",
            f"└── Final: {r.final_skill} (confidence: {r.final_confidence:.2f})",
        ]

        if r.semantic_all_scores:
            content.append("")
            content.append("Top Candidates:")
            for skill, score in r.semantic_all_scores[:3]:
                bar = "█" * int(score * 20)
                content.append(f"  {skill}: {score:.2f} {bar}")

        lines.extend(self._box("SEMANTIC ROUTING", content))
        return lines

    def _prompt_section(self, trace: ContextPipelineTrace) -> List[str]:
        """Generate system prompt section."""
        lines = [f"▼ SYSTEM PROMPT ASSEMBLY ({trace.system_prompt_length} chars total)"]

        content = []
        for section, size in trace.system_prompt_section_sizes.items():
            content.append(f"├── [{section}]: {size} chars")

        if trace.skill_tools_available:
            content.append("")
            content.append(f"Tools: {', '.join(trace.skill_tools_available)}")

        lines.extend(self._box("SECTIONS", content))
        return lines

    def _llm_section(self, trace: ContextPipelineTrace) -> List[str]:
        """Generate LLM invocations section."""
        lines = ["▼ CLAUDE CODE INVOCATIONS"]

        for inv in trace.llm_invocations:
            content = [
                f"├── Session: {inv.claude_session_id}",
                f"├── Turn: {inv.turn_number}",
                f"├── System Prompt: {inv.system_prompt_length} chars",
                f"├── Response: {inv.response_length} chars",
                f"├── Tool Calls: {inv.tool_calls_detected}",
                f"└── Duration: {inv.duration_ms}ms",
            ]
            lines.extend(self._box(f"ITERATION {inv.iteration}", content))

        return lines

    def _tools_section(self, trace: ContextPipelineTrace) -> List[str]:
        """Generate tool execution section."""
        lines = ["▼ TOOL EXECUTION LOOP"]

        for tc in trace.tool_calls:
            status = "✓" if tc.result_type == "success" else "✗"
            content = [
                f"├── Skill: {tc.skill_name}",
                f"├── Args: {self._truncate(str(tc.args_from_llm), 50)}",
                f"├── Duration: {tc.duration_ms}ms",
                f"└── Result: {status} {tc.result_type} ({tc.result_length} chars)",
            ]
            if tc.error:
                content.append(f"    Error: {tc.error}")

            lines.extend(self._box(f"{tc.tool_name} (Iter {tc.iteration})", content))

        return lines

    def _response_section(self, trace: ContextPipelineTrace) -> List[str]:
        """Generate response section."""
        lines = ["▼ RESPONSE"]

        content = [
            f"├── Length: {trace.response_length} chars",
            f"├── Duration: {trace.duration_total_ms}ms total",
            f"│   - Context: {trace.duration_context_ms}ms ({trace.duration_context_ms * 100 // max(trace.duration_total_ms, 1)}%)",
            f"│   - Routing: {trace.duration_routing_ms}ms ({trace.duration_routing_ms * 100 // max(trace.duration_total_ms, 1)}%)",
            f"│   - Tools: {trace.duration_tools_ms}ms ({trace.duration_tools_ms * 100 // max(trace.duration_total_ms, 1)}%)",
            f"│   - Claude: {trace.duration_llm_ms}ms ({trace.duration_llm_ms * 100 // max(trace.duration_total_ms, 1)}%)",
            f"└── Preview:",
            f"    \"{self._truncate(trace.response_final, 60)}\"",
        ]

        lines.extend(self._box("RESPONSE", content))
        return lines


class TraceReporter:
    """
    Generates test reports from multiple traces.

    Usage:
        reporter = TraceReporter()
        reporter.add_trace(trace1)
        reporter.add_trace(trace2)

        summary = reporter.generate_summary()
        print(summary)
    """

    def __init__(self):
        self.traces: List[ContextPipelineTrace] = []
        self.visualizer = TraceVisualizer()

    def add_trace(self, trace: ContextPipelineTrace):
        """Add a trace to the report."""
        self.traces.append(trace)

    def clear(self):
        """Clear all traces."""
        self.traces.clear()

    def generate_summary(self) -> str:
        """Generate a summary of all traces."""
        if not self.traces:
            return "No traces recorded."

        lines = []
        lines.append("# E2E Test Report")
        lines.append("")
        lines.append(f"**Total Traces:** {len(self.traces)}")

        passed = sum(1 for t in self.traces if t.success)
        failed = len(self.traces) - passed
        lines.append(f"**Passed:** {passed}")
        lines.append(f"**Failed:** {failed}")
        lines.append("")

        # Summary table
        lines.append("## Trace Summary")
        lines.append("")
        lines.append("| Trace | Skill | Method | Tools | Duration | Status |")
        lines.append("|-------|-------|--------|-------|----------|--------|")

        for trace in self.traces:
            status = "✅" if trace.success else "❌"
            lines.append(
                f"| {trace.trace_id[:12]}... | "
                f"{trace.routing.final_skill} | "
                f"{trace.routing.final_method} | "
                f"{trace.tool_total_calls} | "
                f"{trace.duration_total_ms}ms | "
                f"{status} |"
            )

        lines.append("")

        # Detailed traces
        if failed > 0:
            lines.append("## Failed Traces")
            lines.append("")
            for trace in self.traces:
                if not trace.success:
                    lines.append(self.visualizer.to_markdown(trace))
                    lines.append("")
                    lines.append("---")
                    lines.append("")

        return "\n".join(lines)

    def save_report(self, path: str):
        """Save report to file."""
        report = self.generate_summary()
        with open(path, "w") as f:
            f.write(report)

    def print_compact(self):
        """Print compact summary to console."""
        for trace in self.traces:
            print(self.visualizer.to_compact(trace))
