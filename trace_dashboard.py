#!/usr/bin/env python3
"""
Workshop Trace Dashboard - Real-time visualization of agent execution traces.

This Flask app provides a web interface for watching trace logs as they develop,
with filtering by agent, context size tracking, and latency analysis.

Run with: python trace_dashboard.py
Access at: http://localhost:5001
"""

import os
import re
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from flask import Flask, render_template_string, jsonify, request
from config import Config

app = Flask(__name__)

# =============================================================================
# Log Parsing
# =============================================================================

def get_latest_log_file():
    """Find the most recent workshop log file."""
    logs_dir = Config.LOGS_DIR
    if not logs_dir.exists():
        return None

    log_files = list(logs_dir.glob("workshop_*.log"))
    if not log_files:
        return None

    return max(log_files, key=lambda f: f.stat().st_mtime)


def parse_trace_entries(log_file: Path, last_position: int = 0, max_entries: int = 500):
    """
    Parse trace entries from a log file.

    Returns:
        (entries, new_position) - List of trace entries and new file position
    """
    if not log_file or not log_file.exists():
        return [], 0

    entries = []

    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        f.seek(last_position)
        content = f.read()
        new_position = f.tell()

    # Parse different trace entry types
    patterns = {
        'context_size': re.compile(
            r'(\d{2}:\d{2}:\d{2}).*\[TRACE:CONTEXT_SIZE\]\s*(.*)',
            re.MULTILINE
        ),
        'context_in': re.compile(
            r'(\d{2}:\d{2}:\d{2}).*\[TRACE:(SUBAGENT|SKILL):([\w-]+):ITER(\d+):msg\[(\d+)\]:(\w+)\]\s*(.*)',
            re.MULTILINE | re.DOTALL
        ),
        'context_out': re.compile(
            r'(\d{2}:\d{2}:\d{2}).*\[TRACE:CONTEXT_OUT:([\w-]+):ITER(\d+)\]\s*(.*)',
            re.MULTILINE | re.DOTALL
        ),
        'tool_exec': re.compile(
            r'(\d{2}:\d{2}:\d{2}).*\[TRACE:TOOL_EXEC\]\s*(\w+)\s*\((\d+)ms\)',
            re.MULTILINE
        ),
        'tool_args': re.compile(
            r'(\d{2}:\d{2}:\d{2}).*\[TRACE:TOOL_ARGS:(\w+)\]\s*(.*?)(?=\n\d{2}:\d{2}:\d{2}|\Z)',
            re.MULTILINE | re.DOTALL
        ),
        'tool_result': re.compile(
            r'(\d{2}:\d{2}:\d{2}).*\[TRACE:TOOL_RESULT:(\w+)\]\s*(.*?)(?=\n\d{2}:\d{2}:\d{2}|\Z)',
            re.MULTILINE | re.DOTALL
        ),
        'llm_response': re.compile(
            r'(\d{2}:\d{2}:\d{2}).*\[TRACE:LLM_RESPONSE\]\s*Skill=(\w+)\s*Iter=(\d+)\s*Duration=(\d+)ms\s*Chars=(\d+)',
            re.MULTILINE
        ),
        'subagent_exec': re.compile(
            r'(\d{2}:\d{2}:\d{2}).*\[SUBAGENT_EXEC\]\s*Iteration\s*(\d+)/(\d+)\s*\(model=([\w-]+)\)',
            re.MULTILINE
        ),
        'large_context_warning': re.compile(
            r'(\d{2}:\d{2}:\d{2}).*\[TRACE:CONTEXT_SIZE\].*(?:LARGE CONTEXT|CRITICAL).*?(\d+,?\d*)\s*chars',
            re.MULTILINE
        ),
    }

    # Extract context size entries
    for match in patterns['context_size'].finditer(content):
        timestamp, message = match.groups()
        entries.append({
            'type': 'context_size',
            'timestamp': timestamp,
            'message': message,
            'raw': match.group(0)
        })

    # Extract LLM response entries
    for match in patterns['llm_response'].finditer(content):
        timestamp, skill, iteration, duration, chars = match.groups()
        entries.append({
            'type': 'llm_response',
            'timestamp': timestamp,
            'skill': skill,
            'iteration': int(iteration),
            'duration_ms': int(duration),
            'chars': int(chars),
            'raw': match.group(0)
        })

    # Extract tool execution entries
    for match in patterns['tool_exec'].finditer(content):
        timestamp, tool_name, duration = match.groups()
        entries.append({
            'type': 'tool_exec',
            'timestamp': timestamp,
            'tool': tool_name,
            'duration_ms': int(duration),
            'raw': match.group(0)
        })

    # Extract subagent execution entries
    for match in patterns['subagent_exec'].finditer(content):
        timestamp, iteration, max_iter, model = match.groups()
        entries.append({
            'type': 'subagent_iteration',
            'timestamp': timestamp,
            'iteration': int(iteration),
            'max_iterations': int(max_iter),
            'model': model,
            'raw': match.group(0)
        })

    # Extract large context warnings
    for match in patterns['large_context_warning'].finditer(content):
        timestamp, size = match.groups()
        entries.append({
            'type': 'warning',
            'timestamp': timestamp,
            'message': f'Large context: {size} chars',
            'severity': 'warning' if 'LARGE' in match.group(0) else 'critical',
            'raw': match.group(0)
        })

    # Sort by timestamp and limit
    entries.sort(key=lambda e: e['timestamp'])

    return entries[-max_entries:], new_position


def compute_summary(entries: list) -> dict:
    """Compute summary statistics from trace entries."""
    summary = {
        'total_entries': len(entries),
        'llm_calls': 0,
        'total_llm_time_ms': 0,
        'avg_llm_time_ms': 0,
        'tool_calls': 0,
        'total_tool_time_ms': 0,
        'warnings': 0,
        'skills_active': set(),
        'subagent_iterations': 0,
    }

    llm_times = []

    for entry in entries:
        if entry['type'] == 'llm_response':
            summary['llm_calls'] += 1
            llm_times.append(entry['duration_ms'])
            summary['skills_active'].add(entry.get('skill', 'unknown'))
        elif entry['type'] == 'tool_exec':
            summary['tool_calls'] += 1
            summary['total_tool_time_ms'] += entry.get('duration_ms', 0)
        elif entry['type'] == 'warning':
            summary['warnings'] += 1
        elif entry['type'] == 'subagent_iteration':
            summary['subagent_iterations'] += 1

    if llm_times:
        summary['total_llm_time_ms'] = sum(llm_times)
        summary['avg_llm_time_ms'] = sum(llm_times) // len(llm_times)

    summary['skills_active'] = list(summary['skills_active'])

    return summary


# =============================================================================
# State Management
# =============================================================================

class DashboardState:
    """Thread-safe state management for the dashboard."""

    def __init__(self):
        self.lock = threading.Lock()
        self.entries = []
        self.last_position = 0
        self.log_file = None
        self.last_update = None

    def update(self):
        """Update entries from log file."""
        with self.lock:
            # Check for new or changed log file
            latest = get_latest_log_file()
            if latest != self.log_file:
                self.log_file = latest
                self.last_position = 0
                self.entries = []

            if self.log_file:
                new_entries, self.last_position = parse_trace_entries(
                    self.log_file, self.last_position
                )
                self.entries.extend(new_entries)
                # Keep only last 1000 entries
                self.entries = self.entries[-1000:]

            self.last_update = datetime.now().isoformat()

    def get_state(self):
        """Get current state."""
        with self.lock:
            return {
                'entries': self.entries[-200:],  # Last 200 entries
                'log_file': str(self.log_file) if self.log_file else None,
                'last_update': self.last_update,
                'summary': compute_summary(self.entries)
            }


state = DashboardState()


# =============================================================================
# Routes
# =============================================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Workshop Trace Dashboard</title>
    <meta charset="utf-8">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Consolas', 'Monaco', monospace;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .header {
            background: #16213e;
            padding: 15px 20px;
            border-bottom: 2px solid #0f3460;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .header h1 {
            font-size: 1.4em;
            color: #e94560;
        }
        .header .status {
            font-size: 0.9em;
            color: #888;
        }
        .header .status.live {
            color: #4caf50;
        }
        .summary {
            background: #16213e;
            padding: 15px 20px;
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
            border-bottom: 1px solid #0f3460;
        }
        .summary-item {
            text-align: center;
        }
        .summary-item .value {
            font-size: 1.8em;
            font-weight: bold;
            color: #e94560;
        }
        .summary-item .label {
            font-size: 0.75em;
            color: #888;
            text-transform: uppercase;
        }
        .summary-item.warning .value { color: #ff9800; }
        .summary-item.critical .value { color: #f44336; }
        .filters {
            background: #16213e;
            padding: 10px 20px;
            border-bottom: 1px solid #0f3460;
        }
        .filters button {
            background: #0f3460;
            border: none;
            color: #eee;
            padding: 5px 12px;
            margin-right: 8px;
            cursor: pointer;
            border-radius: 3px;
        }
        .filters button:hover { background: #e94560; }
        .filters button.active { background: #e94560; }
        .entries {
            padding: 10px;
            max-height: calc(100vh - 220px);
            overflow-y: auto;
        }
        .entry {
            background: #16213e;
            margin: 5px;
            padding: 10px;
            border-radius: 5px;
            border-left: 3px solid #0f3460;
            font-size: 0.85em;
        }
        .entry.llm_response {
            border-left-color: #4caf50;
        }
        .entry.tool_exec {
            border-left-color: #2196f3;
        }
        .entry.warning {
            border-left-color: #ff9800;
            background: #2a1f00;
        }
        .entry.critical {
            border-left-color: #f44336;
            background: #2a0000;
        }
        .entry.context_size {
            border-left-color: #9c27b0;
        }
        .entry.subagent_iteration {
            border-left-color: #00bcd4;
        }
        .entry .timestamp {
            color: #888;
            font-size: 0.85em;
        }
        .entry .type {
            display: inline-block;
            padding: 2px 6px;
            background: #0f3460;
            border-radius: 3px;
            font-size: 0.75em;
            margin-left: 8px;
            text-transform: uppercase;
        }
        .entry .content {
            margin-top: 5px;
            white-space: pre-wrap;
            word-break: break-all;
        }
        .entry .metrics {
            display: flex;
            gap: 15px;
            margin-top: 5px;
            font-size: 0.85em;
        }
        .entry .metrics span {
            color: #888;
        }
        .entry .metrics .value {
            color: #e94560;
            font-weight: bold;
        }
        .auto-scroll-notice {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #e94560;
            color: white;
            padding: 10px 15px;
            border-radius: 5px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Workshop Trace Dashboard</h1>
        <div class="status" id="status">Connecting...</div>
    </div>

    <div class="summary" id="summary">
        <div class="summary-item">
            <div class="value" id="llm-calls">0</div>
            <div class="label">LLM Calls</div>
        </div>
        <div class="summary-item">
            <div class="value" id="avg-latency">0ms</div>
            <div class="label">Avg Latency</div>
        </div>
        <div class="summary-item">
            <div class="value" id="tool-calls">0</div>
            <div class="label">Tool Calls</div>
        </div>
        <div class="summary-item">
            <div class="value" id="subagent-iters">0</div>
            <div class="label">Subagent Iters</div>
        </div>
        <div class="summary-item" id="warnings-container">
            <div class="value" id="warnings">0</div>
            <div class="label">Warnings</div>
        </div>
    </div>

    <div class="filters">
        <button class="active" data-filter="all">All</button>
        <button data-filter="llm_response">LLM</button>
        <button data-filter="tool_exec">Tools</button>
        <button data-filter="context_size">Context</button>
        <button data-filter="warning">Warnings</button>
        <button data-filter="subagent_iteration">Subagents</button>
    </div>

    <div class="entries" id="entries">
        <div class="entry">Loading traces...</div>
    </div>

    <script>
        let activeFilter = 'all';
        let autoScroll = true;
        let lastEntryCount = 0;

        function formatEntry(entry) {
            let content = '';
            let metrics = '';

            if (entry.type === 'llm_response') {
                content = `Skill: ${entry.skill} | Iteration: ${entry.iteration}`;
                metrics = `<span>Duration: <span class="value">${entry.duration_ms}ms</span></span>
                          <span>Chars: <span class="value">${entry.chars}</span></span>`;
            } else if (entry.type === 'tool_exec') {
                content = `Tool: ${entry.tool}`;
                metrics = `<span>Duration: <span class="value">${entry.duration_ms}ms</span></span>`;
            } else if (entry.type === 'context_size') {
                content = entry.message;
            } else if (entry.type === 'warning') {
                content = entry.message;
            } else if (entry.type === 'subagent_iteration') {
                content = `Model: ${entry.model} | Iteration: ${entry.iteration}/${entry.max_iterations}`;
            } else {
                content = entry.raw || JSON.stringify(entry);
            }

            let typeClass = entry.type;
            if (entry.severity === 'critical') typeClass = 'critical';

            return `
                <div class="entry ${typeClass}" data-type="${entry.type}">
                    <span class="timestamp">${entry.timestamp}</span>
                    <span class="type">${entry.type.replace('_', ' ')}</span>
                    <div class="content">${content}</div>
                    ${metrics ? '<div class="metrics">' + metrics + '</div>' : ''}
                </div>
            `;
        }

        function updateUI(data) {
            // Update status
            const status = document.getElementById('status');
            status.textContent = `Live | ${data.log_file ? data.log_file.split('/').pop() : 'No log file'}`;
            status.className = 'status live';

            // Update summary
            const summary = data.summary;
            document.getElementById('llm-calls').textContent = summary.llm_calls;
            document.getElementById('avg-latency').textContent = summary.avg_llm_time_ms + 'ms';
            document.getElementById('tool-calls').textContent = summary.tool_calls;
            document.getElementById('subagent-iters').textContent = summary.subagent_iterations;
            document.getElementById('warnings').textContent = summary.warnings;

            // Highlight warnings
            const warningsContainer = document.getElementById('warnings-container');
            if (summary.warnings > 0) {
                warningsContainer.className = 'summary-item warning';
            }

            // Update entries
            const entriesDiv = document.getElementById('entries');
            const filteredEntries = data.entries.filter(e =>
                activeFilter === 'all' || e.type === activeFilter
            );

            if (filteredEntries.length !== lastEntryCount) {
                entriesDiv.innerHTML = filteredEntries.map(formatEntry).join('');
                lastEntryCount = filteredEntries.length;

                if (autoScroll) {
                    entriesDiv.scrollTop = entriesDiv.scrollHeight;
                }
            }
        }

        async function fetchData() {
            try {
                const response = await fetch('/api/traces');
                const data = await response.json();
                updateUI(data);
            } catch (e) {
                document.getElementById('status').textContent = 'Error: ' + e.message;
                document.getElementById('status').className = 'status';
            }
        }

        // Filter buttons
        document.querySelectorAll('.filters button').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                activeFilter = btn.dataset.filter;
                lastEntryCount = -1; // Force refresh
                fetchData();
            });
        });

        // Initial fetch and polling
        fetchData();
        setInterval(fetchData, 1000);

        // Scroll detection
        document.getElementById('entries').addEventListener('scroll', function() {
            const isAtBottom = this.scrollHeight - this.scrollTop - this.clientHeight < 50;
            autoScroll = isAtBottom;
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Serve the dashboard HTML."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/traces')
def get_traces():
    """API endpoint to get trace entries."""
    state.update()
    return jsonify(state.get_state())


@app.route('/api/clear')
def clear_traces():
    """Clear current trace entries."""
    with state.lock:
        state.entries = []
    return jsonify({'status': 'cleared'})


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Workshop Trace Dashboard")
    print("=" * 60)
    print(f"Watching logs in: {Config.LOGS_DIR}")
    print(f"Dashboard URL: http://localhost:5001")
    print("=" * 60)

    # Initial state update
    state.update()

    app.run(host='127.0.0.1', port=5001, debug=False, threaded=True)
