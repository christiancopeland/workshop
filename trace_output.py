"""
Trace Output Manager

Handles saving and formatting ContextPipelineTrace objects for --trace-mode.
Produces JSON, Markdown, and ASCII output for offline review and debugging.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from tests.e2e.context_tracer import ContextPipelineTrace
from tests.e2e.visualizer import TraceVisualizer


class TraceOutputManager:
    """
    Manages trace output to files and terminal.

    Features:
    - JSON output: Complete trace data for machine processing
    - Markdown output: Human-readable reports
    - ASCII output: Terminal visualization

    Usage:
        manager = TraceOutputManager()
        saved_files = manager.save_trace(trace)
        # Returns: {"json": Path(...), "markdown": Path(...)}
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        format: str = "both",
        print_ascii: bool = True,
    ):
        """
        Initialize the trace output manager.

        Args:
            output_dir: Directory for trace files. Defaults to ~/.workshop/traces/
            format: Output format - "json", "markdown", "both", or "ascii"
            print_ascii: Whether to print ASCII visualization to terminal
        """
        self.output_dir = output_dir or Path.home() / ".workshop" / "traces"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format = format
        self.print_ascii = print_ascii or format == "ascii"
        self.visualizer = TraceVisualizer()

    def save_trace(self, trace: ContextPipelineTrace) -> Dict[str, Path]:
        """
        Save a completed trace to files.

        Args:
            trace: The completed pipeline trace

        Returns:
            Dict with paths to saved files, e.g.:
            {"json": Path("..."), "markdown": Path("...")}
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_id = trace.trace_id.split("_")[-1][:8] if "_" in trace.trace_id else trace.trace_id[:8]
        base_name = f"{timestamp}_{trace_id}"

        saved_files = {}

        # JSON output (complete data)
        if self.format in ("json", "both"):
            json_path = self.output_dir / f"{base_name}.json"
            json_path.write_text(trace.to_json())
            saved_files["json"] = json_path

        # Markdown output (human readable)
        if self.format in ("markdown", "both"):
            md_path = self.output_dir / f"{base_name}.md"
            md_path.write_text(self.visualizer.to_markdown(trace))
            saved_files["markdown"] = md_path

        # ASCII to terminal
        if self.print_ascii:
            self._print_ascii_trace(trace)

        return saved_files

    def _print_ascii_trace(self, trace: ContextPipelineTrace):
        """Print ASCII visualization to terminal."""
        print("\n" + "=" * 80)
        print("PIPELINE TRACE OUTPUT")
        print("=" * 80)

        # Compact summary line
        status = "PASSED" if trace.success else "FAILED"
        status_symbol = "✓" if trace.success else "✗"
        print(f"\n{status_symbol} {status} | {trace.trace_id}")
        print(f"Input: \"{trace.user_input_raw[:60]}{'...' if len(trace.user_input_raw) > 60 else ''}\"")
        print("-" * 80)

        # Context Assembly Summary
        print(f"\n📦 CONTEXT ASSEMBLY ({trace.duration_context_ms}ms)")
        print(f"   ├─ Telos: {'✓' if trace.telos_loaded else '✗'} ({trace.telos_total_chars} chars)")
        if trace.telos_layers:
            for layer in trace.telos_layers:
                status = "✓" if layer.loaded_successfully else "✗"
                print(f"   │   └─ {status} {layer.layer_name} ({layer.content_length} chars)")
        print(f"   ├─ Tasks: {trace.tasks_in_progress_count} in_progress, {trace.tasks_pending_count} pending")
        print(f"   ├─ Auto Context: {'✓' if trace.auto_context_enabled else '✗'} ({trace.auto_context_length} chars)")
        if trace.detected_workflow:
            print(f"   │   └─ Workflow: {trace.detected_workflow} ({trace.workflow_confidence:.0%})")
        print(f"   └─ Memory: {trace.memory_results_count} results")

        # Routing Summary
        r = trace.routing
        print(f"\n🧭 ROUTING DECISION ({trace.routing_total_duration_ms}ms)")
        print(f"   ├─ Method: {r.final_method}")
        print(f"   ├─ Skill: {r.final_skill}")
        print(f"   ├─ Confidence: {r.final_confidence:.2f}")
        if r.semantic_enabled:
            threshold_check = "≥" if r.semantic_score >= r.bypass_threshold else "<"
            print(f"   └─ Semantic: {r.semantic_score:.3f} {threshold_check} {r.bypass_threshold} (bypass)")
            if r.semantic_matched_utterance:
                print(f"       Matched: \"{r.semantic_matched_utterance[:50]}...\"")

        # LLM Invocations Summary
        print(f"\n🤖 CLAUDE CODE INVOCATIONS ({trace.llm_total_invocations} calls, {trace.llm_total_duration_ms}ms)")
        for inv in trace.llm_invocations:
            print(f"   ├─ Iter {inv.iteration}: {inv.system_prompt_length} sys + {inv.user_message_length} user → {inv.response_length} chars ({inv.duration_ms}ms)")
            if inv.tool_calls_detected > 0:
                print(f"   │   └─ Tool calls detected: {inv.tool_calls_detected}")

        # Tool Execution Summary
        if trace.tool_calls:
            print(f"\n🔧 TOOL EXECUTION ({trace.tool_total_calls} calls, {trace.tool_total_duration_ms}ms)")
            for tc in trace.tool_calls:
                status = "✓" if tc.result_type == "success" else "✗"
                print(f"   ├─ {status} {tc.tool_name} ({tc.duration_ms}ms)")
                if tc.args_from_llm:
                    args_preview = str(tc.args_from_llm)[:50]
                    print(f"   │   Args: {args_preview}{'...' if len(str(tc.args_from_llm)) > 50 else ''}")
                print(f"   │   Result: {tc.result_type} ({tc.result_length} chars)")
                if tc.error:
                    print(f"   │   Error: {tc.error[:60]}")

        # Response Summary
        print(f"\n📤 RESPONSE ({trace.response_length} chars)")
        preview = trace.response_final[:150].replace('\n', ' ')
        print(f"   \"{preview}{'...' if len(trace.response_final) > 150 else ''}\"")

        # Timing Breakdown
        total = max(trace.duration_total_ms, 1)
        print(f"\n⏱️  TIMING BREAKDOWN ({total}ms total)")
        stages = [
            ("Context", trace.duration_context_ms),
            ("Routing", trace.duration_routing_ms),
            ("LLM", trace.duration_llm_ms),
            ("Tools", trace.duration_tools_ms),
        ]
        for name, duration in stages:
            pct = (duration / total) * 100
            bar = "█" * int(pct / 5)
            print(f"   ├─ {name:8s}: {duration:5d}ms ({pct:5.1f}%) {bar}")

        # Error info if failed
        if not trace.success:
            print(f"\n❌ ERROR at {trace.error_stage.value if trace.error_stage else 'unknown'}")
            print(f"   {trace.error_message}")

        print("=" * 80 + "\n")

    def get_recent_traces(self, limit: int = 10) -> list:
        """
        Get paths to recent trace files.

        Args:
            limit: Maximum number of traces to return

        Returns:
            List of Path objects to trace files (newest first)
        """
        if not self.output_dir.exists():
            return []

        # Get all JSON files (primary format)
        traces = sorted(
            self.output_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        return traces[:limit]

    def load_trace(self, trace_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load a trace from JSON file.

        Args:
            trace_path: Path to the JSON trace file

        Returns:
            Parsed trace dict or None if failed
        """
        import json
        try:
            return json.loads(trace_path.read_text())
        except Exception as e:
            print(f"Failed to load trace: {e}")
            return None
