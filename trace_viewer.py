#!/usr/bin/env python3
"""
Workshop Trace Viewer
CLI tool for inspecting telemetry traces for debugging and analysis.

Usage:
    python trace_viewer.py                    # Show recent traces
    python trace_viewer.py --id <trace_id>    # Show specific trace
    python trace_viewer.py --export out.jsonl # Export for training
    python trace_viewer.py --stats            # Show statistics
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from telemetry import get_telemetry, TelemetryCollector


def format_duration(ms: int) -> str:
    """Format milliseconds as human readable"""
    if ms < 1000:
        return f"{ms}ms"
    elif ms < 60000:
        return f"{ms/1000:.1f}s"
    else:
        return f"{ms/60000:.1f}m"


def print_trace_summary(trace_data: dict):
    """Print a one-line summary of a trace"""
    trace_id = trace_data.get('trace_id', '')[:8]
    timestamp = trace_data.get('timestamp_start', '')[:19]
    status = trace_data.get('status', 'unknown')
    duration = format_duration(trace_data.get('total_duration_ms', 0))

    user_input = trace_data.get('user_input_raw', '')[:40]
    if len(trace_data.get('user_input_raw', '')) > 40:
        user_input += '...'

    # Status emoji
    status_emoji = {
        'completed': '✓',
        'error': '✗',
        'started': '○'
    }.get(status, '?')

    # Tools used
    tool_count = trace_data.get('tool_total_calls', 0)
    llm_count = trace_data.get('llm_total_calls', 0)

    info = []
    if tool_count:
        info.append(f"{tool_count} tools")
    if llm_count:
        info.append(f"{llm_count} LLM calls")
    if trace_data.get('context_injection_succeeded'):
        info.append("ctx")
    if trace_data.get('telos_loaded'):
        info.append("telos")

    info_str = f"[{', '.join(info)}]" if info else ""

    print(f"{status_emoji} {trace_id} | {timestamp} | {duration:>6} | {user_input} {info_str}")


def print_trace_detail(trace_data: dict):
    """Print detailed view of a trace"""
    print("\n" + "=" * 70)
    print(f"TRACE: {trace_data.get('trace_id', 'unknown')}")
    print("=" * 70)

    # Basic info
    print(f"\n📅 Timestamp: {trace_data.get('timestamp_start', 'unknown')}")
    print(f"⏱️  Duration: {format_duration(trace_data.get('total_duration_ms', 0))}")
    print(f"📊 Status: {trace_data.get('status', 'unknown')}")
    print(f"🤖 Model: {trace_data.get('model_name', 'unknown')}")
    print(f"🎙️  Voice Mode: {trace_data.get('voice_mode', False)}")

    # User Input
    print(f"\n{'─' * 70}")
    print("USER INPUT:")
    print(f"  Raw ({trace_data.get('user_input_length', 0)} chars):")
    print(f"    {trace_data.get('user_input_raw', '')[:200]}")
    if trace_data.get('user_input_enriched_length', 0) > trace_data.get('user_input_length', 0):
        print(f"  Enriched ({trace_data.get('user_input_enriched_length', 0)} chars)")

    # Context Sources
    context_sources = trace_data.get('context_sources', [])
    if context_sources:
        print(f"\n{'─' * 70}")
        print(f"CONTEXT SOURCES ({len(context_sources)}):")
        for src in context_sources:
            src_type = src.get('source_type', 'unknown')
            src_len = src.get('content_length', 0)
            src_path = src.get('source_path', '')
            print(f"  • {src_type}: {src_len} chars")
            if src_path:
                print(f"    Path: {src_path}")
            # Show first 100 chars of content
            content = src.get('content', '')[:100]
            if content:
                print(f"    Preview: {content}...")

    # Telos
    if trace_data.get('telos_loaded'):
        print(f"\n{'─' * 70}")
        print("TELOS CONTEXT:")
        if trace_data.get('telos_active_project'):
            print(f"  Active Project: {trace_data['telos_active_project']}")
        if trace_data.get('telos_profile'):
            print(f"  Profile: {trace_data['telos_profile'][:100]}...")

    # Context Manager
    if trace_data.get('context_manager_used'):
        print(f"\n{'─' * 70}")
        print("CONTEXT MANAGER:")
        if trace_data.get('active_files'):
            print(f"  Active Files: {trace_data['active_files']}")
        if trace_data.get('detected_workflow'):
            print(f"  Workflow: {trace_data['detected_workflow']} "
                  f"({trace_data.get('detected_workflow_confidence', 0)*100:.0f}%)")

    # Intent Detection
    if trace_data.get('intent_detection_attempted'):
        print(f"\n{'─' * 70}")
        print("INTENT DETECTION:")
        print(f"  Detected: {trace_data.get('intent_detected', False)}")
        if trace_data.get('intent_detected'):
            print(f"  Tool: {trace_data.get('intent_tool_name', 'unknown')}")
            print(f"  Pattern: {trace_data.get('intent_pattern_matched', 'unknown')}")
            print(f"  Args: {trace_data.get('intent_args_extracted', {})}")

    # LLM Calls
    llm_calls = trace_data.get('llm_calls', [])
    if llm_calls:
        print(f"\n{'─' * 70}")
        print(f"LLM CALLS ({len(llm_calls)}):")
        for i, call in enumerate(llm_calls):
            print(f"\n  Call {i+1}:")
            print(f"    Model: {call.get('model', 'unknown')}")
            print(f"    Duration: {format_duration(call.get('duration_ms', 0))}")
            print(f"    System Prompt: {call.get('system_prompt_length', 0)} chars")
            print(f"    Messages: {len(call.get('messages', []))}")
            print(f"    Response: {call.get('response_length', 0)} chars")
            if call.get('tool_calls_extracted'):
                print(f"    Tool Calls Extracted: {len(call['tool_calls_extracted'])}")

    # Tool Calls
    tool_calls = trace_data.get('tool_calls', [])
    if tool_calls:
        print(f"\n{'─' * 70}")
        print(f"TOOL CALLS ({len(tool_calls)}):")
        for i, call in enumerate(tool_calls):
            status = "✓" if call.get('result_type') == 'success' else "✗"
            print(f"\n  {status} {call.get('tool_name', 'unknown')} ({call.get('skill_name', 'unknown')})")
            print(f"    Duration: {format_duration(call.get('duration_ms', 0))}")
            print(f"    Args Raw: {call.get('args_raw', {})}")
            if call.get('args_normalized') != call.get('args_raw'):
                print(f"    Args Normalized: {call.get('args_normalized', {})}")
            print(f"    Result: {call.get('result_type', 'unknown')} ({call.get('result_length', 0)} chars)")
            if call.get('was_direct_call'):
                print(f"    Direct Call: Yes (pattern: {call.get('pattern_matched', 'unknown')})")
            if call.get('error_message'):
                print(f"    Error: {call['error_message']}")

    # Response
    print(f"\n{'─' * 70}")
    print("RESPONSE:")
    print(f"  Length: {trace_data.get('response_length', 0)} chars")
    response = trace_data.get('response_final', '')[:300]
    print(f"  Content: {response}{'...' if len(trace_data.get('response_final', '')) > 300 else ''}")

    # Error
    if trace_data.get('error_occurred'):
        print(f"\n{'─' * 70}")
        print("ERROR:")
        print(f"  Stage: {trace_data.get('error_stage', 'unknown')}")
        print(f"  Message: {trace_data.get('error_message', 'unknown')}")
        if trace_data.get('error_traceback'):
            print(f"  Traceback:\n{trace_data['error_traceback']}")

    print("\n" + "=" * 70)


def print_stats(telemetry: TelemetryCollector):
    """Print telemetry statistics"""
    stats = telemetry.get_stats()

    print("\n" + "=" * 50)
    print("TELEMETRY STATISTICS")
    print("=" * 50)

    print(f"\n📊 Session: {stats.get('session_id', 'unknown')}")
    print(f"💾 Traces in memory: {stats.get('traces_in_memory', 0)}")
    print(f"🗄️  Total traces: {stats.get('total_traces', 0)}")
    print(f"🔧 Total tool calls: {stats.get('total_tool_calls', 0)}")
    print(f"🤖 Total LLM calls: {stats.get('total_llm_calls', 0)}")
    print(f"❌ Errors: {stats.get('error_count', 0)}")
    print(f"⏱️  Avg response time: {format_duration(stats.get('avg_response_time_ms', 0))}")

    print("\n" + "=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Workshop Trace Viewer - Inspect telemetry for debugging and analysis"
    )

    parser.add_argument(
        '--id', '-i',
        help='Show specific trace by ID (or prefix)'
    )
    parser.add_argument(
        '--last', '-n',
        type=int,
        default=10,
        help='Number of recent traces to show (default: 10)'
    )
    parser.add_argument(
        '--stats', '-s',
        action='store_true',
        help='Show telemetry statistics'
    )
    parser.add_argument(
        '--export', '-e',
        type=str,
        help='Export traces to file (JSONL format)'
    )
    parser.add_argument(
        '--export-json',
        type=str,
        help='Export traces to file (JSON format)'
    )
    parser.add_argument(
        '--errors',
        action='store_true',
        help='Show only traces with errors'
    )
    parser.add_argument(
        '--session',
        type=str,
        help='Filter by session ID'
    )
    parser.add_argument(
        '--since',
        type=str,
        help='Show traces since time (e.g., "1h", "30m", "2023-12-01")'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )
    parser.add_argument(
        '--db',
        type=str,
        help='Path to telemetry database'
    )

    args = parser.parse_args()

    # Initialize telemetry
    db_path = Path(args.db) if args.db else Path(__file__).parent / "data" / "telemetry.db"
    json_dir = Path(__file__).parent / "data" / "traces"

    if not db_path.exists():
        print(f"No telemetry database found at {db_path}")
        print("Run Workshop to generate traces first.")
        return 1

    telemetry = TelemetryCollector(
        sqlite_path=db_path,
        json_dir=json_dir,
        auto_save=False
    )

    # Stats mode
    if args.stats:
        print_stats(telemetry)
        return 0

    # Export mode
    if args.export or args.export_json:
        output = Path(args.export or args.export_json)
        format_type = 'json' if args.export_json else 'jsonl'
        count = telemetry.export_training_data(
            output,
            format=format_type,
            include_errors=args.errors
        )
        print(f"Exported {count} traces to {output}")
        return 0

    # Specific trace
    if args.id:
        trace = telemetry.get_trace(args.id)
        if trace:
            print_trace_detail(trace.to_dict())
        else:
            # Try to load from SQLite directly
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT full_json FROM traces WHERE trace_id LIKE ?",
                (f"{args.id}%",)
            ).fetchone()
            conn.close()

            if row:
                trace_data = json.loads(row['full_json'])
                print_trace_detail(trace_data)
            else:
                print(f"Trace not found: {args.id}")
                return 1
        return 0

    # Parse since filter
    since = None
    if args.since:
        if args.since.endswith('h'):
            hours = int(args.since[:-1])
            since = datetime.now() - timedelta(hours=hours)
        elif args.since.endswith('m'):
            minutes = int(args.since[:-1])
            since = datetime.now() - timedelta(minutes=minutes)
        else:
            since = datetime.fromisoformat(args.since)

    # Query traces
    traces = telemetry.query_traces(
        session_id=args.session,
        has_error=True if args.errors else None,
        since=since,
        limit=args.last
    )

    if not traces:
        print("No traces found matching criteria")
        return 0

    # Print traces
    print(f"\n{'─' * 70}")
    print(f"RECENT TRACES ({len(traces)})")
    print(f"{'─' * 70}\n")

    for trace_data in traces:
        if args.verbose:
            # Load full trace and show detail
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT full_json FROM traces WHERE trace_id = ?",
                (trace_data['trace_id'],)
            ).fetchone()
            conn.close()

            if row:
                full_trace = json.loads(row['full_json'])
                print_trace_detail(full_trace)
        else:
            print_trace_summary(trace_data)

    print(f"\n{'─' * 70}")
    print(f"Use --id <trace_id> to see details, --stats for statistics")
    print(f"{'─' * 70}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
