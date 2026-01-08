"""
Claude Code Bridge - Subscription-based Claude Code integration for Workshop

This module provides a wrapper around the Claude Code CLI that uses subscription
authentication (no per-token billing). It replaces Ollama as the reasoning engine
while Workshop maintains orchestration control.

Usage:
    # Get the shared singleton instance (recommended)
    bridge = get_claude_bridge()
    response = await bridge.query(messages, system_prompt="You are Workshop...")

    # Or create a dedicated instance (for isolated contexts)
    bridge = ClaudeCodeBridge()

The response format is compatible with the existing _call_ollama interface:
    {"content": "...", "tool_calls": [...]}

Phase 4 Update (Jan 2026):
- Added singleton pattern for shared session management
- Added session persistence to Workshop session files
- Added session continuity across multi-turn conversations
"""

import subprocess
import json
import os
import asyncio
import re
from typing import Optional, List, Dict, Any, Generator
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

# Use Workshop's logger to ensure logs go to the log file
from logger import get_logger
log = get_logger("claude_bridge")

# Explicit path to Claude Code binary - more secure than PATH lookup
CLAUDE_BINARY = os.path.expanduser("~/.local/bin/claude")


@dataclass
class ClaudeResponse:
    """Structured response from Claude Code CLI"""
    content: str
    session_id: Optional[str] = None
    cost_usd: float = 0.0  # Will be 0 with subscription
    model: str = "claude-sonnet-4-20250514"
    tool_calls: List[Dict] = field(default_factory=list)
    raw_output: Optional[Dict] = None


class ClaudeCodeBridge:
    """
    Wrapper for Claude Code CLI that uses subscription authentication.
    No API key required - uses browser-authenticated session.

    This is the core replacement for Ollama in Workshop's architecture.
    """

    def __init__(
        self,
        working_dir: Optional[str] = None,
        timeout_seconds: int = 120
    ):
        self.working_dir = working_dir or os.getcwd()
        self.timeout = timeout_seconds
        self._session_id: Optional[str] = None
        self._verify_installation()

    def _verify_installation(self):
        """Ensure Claude Code is installed and no API key is set."""
        # Check for API key that would override subscription
        if os.environ.get('ANTHROPIC_API_KEY'):
            log.warning(
                "ANTHROPIC_API_KEY is set! This will cause per-token billing. "
                "Consider removing it from your environment to use subscription auth."
            )

        # Verify Claude Code is installed at expected path
        if not os.path.isfile(CLAUDE_BINARY):
            raise RuntimeError(
                f"Claude Code not found at {CLAUDE_BINARY}. "
                "Install with: curl -fsSL https://claude.ai/install.sh | sh"
            )

        try:
            result = subprocess.run(
                [CLAUDE_BINARY, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError(f"Claude Code not working: {result.stderr}")
            log.info(f"Claude Code version: {result.stdout.strip()}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Claude Code --version timed out")

    def _clean_env(self) -> Dict[str, str]:
        """Return environment with ANTHROPIC_API_KEY removed to ensure subscription auth."""
        return {k: v for k, v in os.environ.items() if k != 'ANTHROPIC_API_KEY'}

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """
        Convert OpenAI-style messages array to a single prompt string.

        Claude Code CLI takes a single prompt, not a messages array.
        We need to format the conversation history appropriately.
        """
        prompt_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # System messages are handled separately via --append-system-prompt
                continue
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            elif role == "tool":
                # Tool results from previous calls
                tool_name = msg.get("name", "tool")
                prompt_parts.append(f"Tool Result ({tool_name}): {content}")

        return "\n\n".join(prompt_parts)

    def _extract_system_prompt(self, messages: List[Dict]) -> Optional[str]:
        """Extract system prompt from messages array."""
        for msg in messages:
            if msg.get("role") == "system":
                return msg.get("content", "")
        return None

    def _get_workshop_tool_instructions(self) -> str:
        """
        Return instructions for Workshop tool calling format.

        When native tools are disabled, Claude needs to know:
        1. What tools are available via Workshop
        2. The exact XML format to use for tool calls
        3. That it must STOP and wait after each tool call
        """
        return '''
## WORKSHOP TOOL EXECUTION

You do NOT have direct tool access. To use tools, output this exact XML format:

<tool_call>
{"tool": "tool_name", "args": {"param1": "value1"}}
</tool_call>

After outputting a tool call, STOP and wait. Workshop will execute the tool and provide results in the next message.

### CRITICAL RULES
1. Output the <tool_call> tag exactly as shown - do NOT just describe what you would do
2. The JSON inside must be valid (double quotes, no trailing commas)
3. After each <tool_call>, STOP your response and wait for results
4. Do NOT fabricate or hallucinate information - use tools to get real data

### Available Tools

**Research Tools:**
- web_search(query, max_results=5) - Search the web via DuckDuckGo
- deep_research(query, depth="standard") - Multi-query research with source synthesis
- fetch_url(url) - Fetch and extract content from a URL

**Orchestration Tools:**
- spawn_subagent(agent_name, task, input_data={}) - Spawn a specialist subagent
  - agent_name: "web-researcher", "coder", "writer", "codebase-analyst", "tech-comparator"
- parallel_dispatch(tasks) - Run multiple subagent tasks in parallel
  - tasks: [{"agent": "name", "prompt": "task description"}, ...]
- checkpoint(summary, options=[]) - Pause for human direction

**File Tools:**
- read_file(path) - Read file contents
- write_file(path, content) - Write file contents
- search_project_files(pattern) - Find files matching pattern
- get_file_content(path) - Get file with line numbers

**Memory Tools:**
- remember(content, category) - Store to long-term memory
- recall(query, limit=5) - Search memory semantically

**Task Tools:**
- task_write(tasks) - Update task list
- task_read() - Get current tasks

### Example Usage

User asks: "Search for ESP32 specifications"

Your response:
<tool_call>
{"tool": "web_search", "args": {"query": "ESP32 microcontroller specifications datasheet", "max_results": 5}}
</tool_call>

Searching for ESP32 specifications now.

[STOP HERE - Workshop will provide results]

### Example: Spawning a Subagent

User asks: "Have a researcher look into agent orchestration"

Your response:
<tool_call>
{"tool": "spawn_subagent", "args": {"agent_name": "web-researcher", "task": "Research modern agent orchestration frameworks including LangGraph, CrewAI, and AutoGen. Focus on how they handle context management and multi-agent coordination."}}
</tool_call>

Dispatching a web-researcher subagent to investigate agent orchestration frameworks.

[STOP HERE - Workshop will spawn the subagent and return results]
'''

    def _extract_tool_calls_from_content(self, content: str) -> List[Dict]:
        """
        Extract tool calls from response content in various formats.

        Supports multiple formats Claude might use:
        1. <tool_call>{"tool": "...", "args": {...}}</tool_call> (Workshop format)
        2. JSON in code blocks
        3. <function_calls><invoke name="...">...</invoke></function_calls> (Anthropic XML format)
        """
        tool_calls = []

        # Strategy 1: XML-style <tool_call> tags (Workshop's primary format)
        for match in re.finditer(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL):
            try:
                call = json.loads(match.group(1).strip())
                if call not in tool_calls:
                    tool_calls.append(call)
                    log.debug(f"[TOOL_PARSE] Extracted tool call (tool_call tag): {call.get('tool', 'unknown')}")
            except json.JSONDecodeError as e:
                log.warning(f"[TOOL_PARSE] Failed to parse tool call: {e}")

        # Strategy 2: JSON in code blocks (fallback)
        for match in re.finditer(r'```(?:json)?\s*(\{[^`]+\})\s*```', content, re.DOTALL):
            try:
                data = json.loads(match.group(1))
                if "tool" in data and data not in tool_calls:
                    tool_calls.append(data)
                    log.debug(f"[TOOL_PARSE] Extracted tool call (code block): {data.get('tool', 'unknown')}")
            except json.JSONDecodeError:
                pass

        # Strategy 3: Anthropic XML function_calls format
        # <function_calls><invoke name="ToolName"><parameter name="arg">value</parameter></invoke></function_calls>
        if '<function_calls>' in content or '<invoke name=' in content:
            log.debug("[TOOL_PARSE] Detected Anthropic XML format, parsing...")
            # Find all <invoke> blocks
            for invoke_match in re.finditer(
                r'<invoke\s+name=["\']([^"\']+)["\']>(.*?)</invoke>',
                content,
                re.DOTALL | re.IGNORECASE
            ):
                tool_name = invoke_match.group(1)
                invoke_content = invoke_match.group(2)

                # Extract parameters
                args = {}
                for param_match in re.finditer(
                    r'<parameter\s+name=["\']([^"\']+)["\']>(.*?)</parameter>',
                    invoke_content,
                    re.DOTALL | re.IGNORECASE
                ):
                    param_name = param_match.group(1)
                    param_value = param_match.group(2).strip()
                    args[param_name] = param_value

                if tool_name:
                    call = {"tool": tool_name, "args": args}
                    if call not in tool_calls:
                        tool_calls.append(call)
                        log.info(f"[TOOL_PARSE] Extracted tool call (Anthropic XML): {tool_name}({list(args.keys())})")

        # Map Claude Code tool names to Workshop tool names
        TOOL_NAME_MAP = {
            "Task": "spawn_subagent",  # Claude Code's Task -> Workshop's spawn_subagent
            "Read": "read_file",
            "Write": "write_file",
            "Edit": "edit_file",
            "Glob": "glob_files",
            "Grep": "grep_content",
            "Bash": "run_command",
            "WebFetch": "fetch_url",
            "WebSearch": "web_search",
        }

        # Map tool names and normalize args
        for call in tool_calls:
            original_name = call.get("tool", "")
            if original_name in TOOL_NAME_MAP:
                mapped_name = TOOL_NAME_MAP[original_name]
                log.info(f"[TOOL_PARSE] Mapped tool name: {original_name} -> {mapped_name}")
                call["tool"] = mapped_name

            # Map Claude Code arg names to Workshop arg names for Task/spawn_subagent
            if original_name == "Task" and "args" in call:
                args = call["args"]
                # Map 'prompt' -> 'task', 'subagent_type' -> 'agent_name'
                if "prompt" in args and "task" not in args:
                    args["task"] = args.pop("prompt")
                if "subagent_type" in args and "agent_name" not in args:
                    # Map subagent types: research -> web-researcher, code -> coder
                    subagent_type = args.pop("subagent_type")
                    type_map = {
                        "research": "web-researcher",
                        "code": "coder",
                        "general-purpose": "web-researcher",
                        "Explore": "web-researcher",
                    }
                    args["agent_name"] = type_map.get(subagent_type, subagent_type)
                    log.debug(f"[TOOL_PARSE] Mapped subagent_type '{subagent_type}' -> agent_name '{args['agent_name']}'")

        if tool_calls:
            log.info(f"[TOOL_PARSE] Total tool calls extracted: {len(tool_calls)}")
        return tool_calls

    def _clean_response_content(self, content: str) -> str:
        """
        Clean response content by removing internal tags that shouldn't be displayed.

        Removes:
        - <thinking>...</thinking> blocks
        - <function_calls>...</function_calls> blocks
        - <tool_call>...</tool_call> blocks (already parsed)
        - <invoke>...</invoke> blocks
        """
        import re

        cleaned = content

        # Remove <thinking> blocks
        cleaned = re.sub(r'<thinking>.*?</thinking>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

        # Remove <function_calls> blocks
        cleaned = re.sub(r'<function_calls>.*?</function_calls>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

        # Remove standalone <invoke> blocks (outside function_calls)
        cleaned = re.sub(r'<invoke\s+name=[^>]+>.*?</invoke>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

        # Remove <tool_call> blocks
        cleaned = re.sub(r'<tool_call>.*?</tool_call>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

        # Remove other known internal tags (be specific to avoid breaking code examples)
        # Note: Avoid generic XML patterns as they would break code examples with HTML/XML

        # Remove <parameter> tags (used in function calls)
        cleaned = re.sub(r'<parameter\s+name=[^>]+>.*?</parameter>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'<parameter>.*?</parameter>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

        # Remove <result> and <output> tags
        cleaned = re.sub(r'<result>.*?</result>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'<output>.*?</output>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

        # Remove <commentary> tags
        cleaned = re.sub(r'<commentary>.*?</commentary>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

        # Clean up excessive whitespace left behind
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = cleaned.strip()

        return cleaned

    async def query(
        self,
        messages: List[Dict],
        system_prompt: Optional[str] = None,
        continue_session: bool = False,
        disable_native_tools: bool = True,
        max_turns: int = 1,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a query to Claude Code using subscription auth.

        This method is designed to be a drop-in replacement for _call_ollama.

        Args:
            messages: OpenAI-style conversation messages
            system_prompt: System prompt (if not in messages)
            continue_session: Continue the most recent conversation
            disable_native_tools: If True, disable Claude's built-in tools so
                                  Workshop can handle tool execution via <tool_call> format.
                                  Default True to maintain Workshop control.
            max_turns: Max agentic turns (1 = single response)
            model: Model override (sonnet, opus, haiku)

        Returns:
            Dict with 'content' (str) and 'tool_calls' (list) - compatible with _call_ollama
        """
        # Extract or use provided system prompt
        msg_system_prompt = self._extract_system_prompt(messages)
        effective_system_prompt = system_prompt or msg_system_prompt

        # When native tools are disabled, inject Workshop tool instructions
        # This tells Claude what tools are available and the exact format to use
        if disable_native_tools:
            tool_instructions = self._get_workshop_tool_instructions()
            if effective_system_prompt:
                effective_system_prompt = effective_system_prompt + "\n\n" + tool_instructions
            else:
                effective_system_prompt = tool_instructions
            log.debug("Injected Workshop tool instructions into system prompt")

        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)

        if not prompt.strip():
            return {"content": "Error: Empty prompt", "tool_calls": []}

        # Build command - use stdin for prompt to avoid "Argument list too long" error
        # The prompt can be very large with all context, so we pass it via stdin
        cmd = [CLAUDE_BINARY, '-p', '-', '--output-format', 'json']

        # Add system prompt if we have one
        # If system prompt is also very large, we need to pass it via a temp file
        if effective_system_prompt:
            # Check if system prompt is small enough for command line (limit ~100KB to be safe)
            if len(effective_system_prompt) < 100000:
                cmd.extend(['--append-system-prompt', effective_system_prompt])
            else:
                # For very large system prompts, prepend to the prompt instead
                prompt = f"[System Context]\n{effective_system_prompt}\n\n[User Request]\n{prompt}"
                log.warning(f"System prompt too large ({len(effective_system_prompt)} chars), prepending to prompt")

        # Session management
        if continue_session and self._session_id:
            cmd.extend(['--resume', self._session_id])
        elif continue_session:
            cmd.append('--continue')

        # Max turns - allow multi-turn for tool use
        # Use higher value to let Claude complete research/tool operations
        effective_max_turns = max_turns if max_turns > 1 else 10
        cmd.extend(['--max-turns', str(effective_max_turns)])

        # Model selection (optional)
        if model:
            cmd.extend(['--model', model])

        # Bypass permissions for tool use - Workshop is the orchestrator
        # This allows Claude to use WebSearch, Read, etc. without interactive approval
        cmd.append('--dangerously-skip-permissions')

        # Disable native tools so Workshop handles tool execution via <tool_call> format
        # When disabled, Claude outputs tool calls as XML that Workshop parses and executes
        log.info(f"[CLAUDE_BRIDGE] disable_native_tools parameter = {disable_native_tools}")
        if disable_native_tools:
            cmd.extend(['--tools', ''])
            log.info("[CLAUDE_BRIDGE] ⚡ --tools '' flag APPLIED - Claude native tools DISABLED")
        else:
            log.warning("[CLAUDE_BRIDGE] ⚠️ Native tools ENABLED - Claude will use its own tools!")

        # Log the FULL command for debugging (critical for diagnosing tool issues)
        log.info(f"[CLAUDE_BRIDGE] ========== FULL COMMAND ==========")
        log.info(f"[CLAUDE_BRIDGE] cmd = {cmd}")
        log.info(f"[CLAUDE_BRIDGE] '--tools' in cmd: {'--tools' in cmd}")
        log.info(f"[CLAUDE_BRIDGE] ===================================")

        # Also log sizes
        log.info(f"[CLAUDE_BRIDGE] Prompt length: {len(prompt)} chars")
        if effective_system_prompt:
            log.info(f"[CLAUDE_BRIDGE] System prompt length: {len(effective_system_prompt)} chars")

        try:
            # Run Claude Code subprocess - pass prompt via stdin to avoid arg list limits
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    input=prompt,  # Pass prompt via stdin
                    capture_output=True,
                    text=True,
                    cwd=self.working_dir,
                    env=self._clean_env(),
                    timeout=self.timeout
                )
            )

            if result.returncode != 0:
                error_msg = result.stderr or "Unknown error"
                log.error(f"Claude Code error (exit {result.returncode}): {error_msg}")
                return {
                    "content": f"Error: Claude Code returned exit code {result.returncode}: {error_msg}",
                    "tool_calls": []
                }

            # Parse JSON output
            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                log.error(f"Failed to parse Claude Code output: {e}")
                log.error(f"Raw output: {result.stdout[:500]}")
                # Fall back to treating output as plain text
                return {
                    "content": result.stdout,
                    "tool_calls": []
                }

            # Extract content from response
            content = ""
            if isinstance(data, dict):
                # JSON output format has 'result' field
                content = data.get('result', data.get('content', ''))
                self._session_id = data.get('session_id')

                # Log cost/usage info if present
                if data.get('cost_usd'):
                    log.info(f"Query cost: ${data.get('cost_usd', 0):.4f}")
                if data.get('model'):
                    log.debug(f"Model used: {data.get('model')}")
            elif isinstance(data, str):
                content = data
            else:
                content = str(data)

            # Extract tool calls from content
            tool_calls = self._extract_tool_calls_from_content(content)

            log.debug(f"Response length: {len(content)} chars")
            if tool_calls:
                log.info(f"Extracted {len(tool_calls)} tool calls from response")

            # Clean the response content to remove internal XML tags
            # This prevents <thinking>, <function_calls>, etc. from being displayed to the user
            cleaned_content = self._clean_response_content(content)
            if cleaned_content != content:
                log.debug(f"[RESPONSE_CLEAN] Cleaned content: {len(content)} -> {len(cleaned_content)} chars")

            return {
                "content": cleaned_content,
                "tool_calls": tool_calls
            }

        except subprocess.TimeoutExpired:
            log.error(f"Claude Code timed out after {self.timeout}s")
            return {
                "content": f"Error: Claude Code timed out after {self.timeout} seconds",
                "tool_calls": []
            }
        except Exception as e:
            log.error(f"Unexpected error calling Claude Code: {e}", exc_info=True)
            return {
                "content": f"Error: {e}",
                "tool_calls": []
            }

    async def query_streaming(
        self,
        messages: List[Dict],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream responses for real-time output.

        Yields JSON objects as they arrive from Claude Code.
        """
        msg_system_prompt = self._extract_system_prompt(messages)
        effective_system_prompt = system_prompt or msg_system_prompt
        prompt = self._messages_to_prompt(messages)

        cmd = [CLAUDE_BINARY, '-p', prompt, '--output-format', 'stream-json']

        if effective_system_prompt:
            cmd.extend(['--append-system-prompt', effective_system_prompt])

        if kwargs.get('continue_session'):
            cmd.append('--continue')

        cmd.extend(['--max-turns', '1'])

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.working_dir,
            env=self._clean_env()
        )

        async for line in process.stdout:
            line = line.decode().strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    yield {'type': 'text', 'content': line}

        await process.wait()

    def reset_session(self):
        """Reset session state for a new conversation."""
        self._session_id = None
        log.debug("Session reset")

    @property
    def session_id(self) -> Optional[str]:
        """Get current session ID if any."""
        return self._session_id

    def set_session_id(self, session_id: str):
        """Set the Claude session ID (for resuming from persisted state)."""
        self._session_id = session_id
        log.debug(f"Session ID set to: {session_id}")


# ============================================================================
# PHASE 4: Session Persistence and Singleton Management
# ============================================================================

# Path for persisting Claude session state
CLAUDE_SESSION_FILE = Path.home() / ".workshop" / "sessions" / "claude_session.json"


class ClaudeSessionState:
    """
    Persisted state linking Workshop sessions to Claude Code sessions.

    This allows multi-turn conversations to maintain context across:
    - Multiple Workshop components (router, executor, subagents)
    - Workshop restarts (if session is still valid)
    - Session resume operations
    """

    def __init__(self):
        self.workshop_session_id: Optional[str] = None
        self.claude_session_id: Optional[str] = None
        self.started_at: Optional[datetime] = None
        self.last_used_at: Optional[datetime] = None
        self.turn_count: int = 0
        self.context_tokens_estimated: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workshop_session_id": self.workshop_session_id,
            "claude_session_id": self.claude_session_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "turn_count": self.turn_count,
            "context_tokens_estimated": self.context_tokens_estimated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaudeSessionState":
        state = cls()
        state.workshop_session_id = data.get("workshop_session_id")
        state.claude_session_id = data.get("claude_session_id")
        state.started_at = datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
        state.last_used_at = datetime.fromisoformat(data["last_used_at"]) if data.get("last_used_at") else None
        state.turn_count = data.get("turn_count", 0)
        state.context_tokens_estimated = data.get("context_tokens_estimated", 0)
        return state


class ClaudeBridgeManager:
    """
    Singleton manager for the Claude Code bridge.

    Provides:
    - Single shared bridge instance for session continuity
    - Session state persistence across Workshop restarts
    - Automatic session linking with Workshop sessions
    - Context usage tracking for efficiency monitoring

    Usage:
        manager = get_claude_bridge_manager()
        bridge = manager.get_bridge()
        await bridge.query(...)
    """

    _instance: Optional["ClaudeBridgeManager"] = None

    def __init__(self):
        self._bridge: Optional[ClaudeCodeBridge] = None
        self._state: Optional[ClaudeSessionState] = None
        self._workshop_session_id: Optional[str] = None
        self._load_state()

    @classmethod
    def get_instance(cls) -> "ClaudeBridgeManager":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_state(self):
        """Load persisted session state."""
        if CLAUDE_SESSION_FILE.exists():
            try:
                with open(CLAUDE_SESSION_FILE) as f:
                    data = json.load(f)
                self._state = ClaudeSessionState.from_dict(data)
                log.debug(f"Loaded Claude session state: {self._state.claude_session_id}")
            except Exception as e:
                log.warning(f"Could not load Claude session state: {e}")
                self._state = ClaudeSessionState()
        else:
            self._state = ClaudeSessionState()

    def _save_state(self):
        """Persist session state to disk."""
        try:
            CLAUDE_SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CLAUDE_SESSION_FILE, "w") as f:
                json.dump(self._state.to_dict(), f, indent=2)
            log.debug("Saved Claude session state")
        except Exception as e:
            log.warning(f"Could not save Claude session state: {e}")

    def get_bridge(self, timeout_seconds: int = 180) -> ClaudeCodeBridge:
        """
        Get the shared bridge instance.

        Creates a new bridge if none exists, or returns the existing one
        with session continuity preserved.
        """
        if self._bridge is None:
            self._bridge = ClaudeCodeBridge(timeout_seconds=timeout_seconds)

            # Restore session ID if we have a valid one for the current Workshop session
            if (self._state.claude_session_id and
                self._state.workshop_session_id == self._workshop_session_id):
                self._bridge.set_session_id(self._state.claude_session_id)
                log.info(f"Restored Claude session: {self._state.claude_session_id}")

        return self._bridge

    def bind_to_workshop_session(self, workshop_session_id: str):
        """
        Bind the Claude bridge to a Workshop session.

        If the Workshop session changes, we start a new Claude session
        to ensure clean context separation.
        """
        if self._workshop_session_id != workshop_session_id:
            log.info(f"Binding to Workshop session: {workshop_session_id}")

            # Check if we're resuming the same session
            if (self._state.workshop_session_id == workshop_session_id and
                self._state.claude_session_id):
                # Resume the previous Claude session
                self._workshop_session_id = workshop_session_id
                if self._bridge:
                    self._bridge.set_session_id(self._state.claude_session_id)
                log.info(f"Resuming Claude session for Workshop session: {workshop_session_id}")
            else:
                # New Workshop session = new Claude session
                self._workshop_session_id = workshop_session_id
                self._state.workshop_session_id = workshop_session_id
                self._state.claude_session_id = None
                self._state.started_at = datetime.now()
                self._state.turn_count = 0
                self._state.context_tokens_estimated = 0

                if self._bridge:
                    self._bridge.reset_session()

                self._save_state()
                log.info(f"Started new Claude session for Workshop session: {workshop_session_id}")

    def record_turn(self, claude_session_id: Optional[str] = None, estimated_tokens: int = 0):
        """
        Record a conversation turn and update session state.

        Called after each successful Claude query to track:
        - Claude session ID (from response)
        - Turn count for the session
        - Estimated token usage
        """
        if claude_session_id:
            self._state.claude_session_id = claude_session_id
        elif self._bridge and self._bridge.session_id:
            self._state.claude_session_id = self._bridge.session_id

        self._state.last_used_at = datetime.now()
        self._state.turn_count += 1
        self._state.context_tokens_estimated += estimated_tokens

        self._save_state()
        log.debug(f"Recorded turn {self._state.turn_count}, session: {self._state.claude_session_id}")

    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information for debugging/dashboard."""
        return {
            "workshop_session_id": self._workshop_session_id,
            "claude_session_id": self._state.claude_session_id if self._state else None,
            "turn_count": self._state.turn_count if self._state else 0,
            "context_tokens_estimated": self._state.context_tokens_estimated if self._state else 0,
            "started_at": self._state.started_at.isoformat() if self._state and self._state.started_at else None,
            "last_used_at": self._state.last_used_at.isoformat() if self._state and self._state.last_used_at else None,
        }

    def reset(self):
        """Reset the bridge and session state (for new conversations)."""
        if self._bridge:
            self._bridge.reset_session()

        self._state = ClaudeSessionState()
        self._state.workshop_session_id = self._workshop_session_id
        self._save_state()
        log.info("Claude session reset")

    async def summarize_session(self) -> Optional[str]:
        """
        Generate a summary of the current session for context efficiency.

        This is useful when the context window is getting full and we need
        to compress the conversation while preserving key information.

        Returns:
            Session summary string, or None if no session exists
        """
        if not self._bridge or not self._state.claude_session_id:
            log.debug("No active Claude session to summarize")
            return None

        summary_prompt = """Summarize the key points from this conversation session:
1. What tasks were discussed or worked on?
2. What decisions were made?
3. What progress was achieved?
4. Are there any pending items or next steps?

Keep the summary concise (under 500 words) but comprehensive."""

        try:
            result = await self._bridge.query(
                messages=[{"role": "user", "content": summary_prompt}],
                continue_session=True,
                disable_native_tools=True,
            )

            summary = result.get("content", "")
            if summary:
                log.info(f"Generated session summary: {len(summary)} chars")
                # Store summary for reference
                self._save_summary(summary)
            return summary

        except Exception as e:
            log.error(f"Failed to summarize session: {e}")
            return None

    def _save_summary(self, summary: str):
        """Save session summary to disk for reference."""
        if not self._state.workshop_session_id:
            return

        summary_file = CLAUDE_SESSION_FILE.parent / f"summary_{self._state.workshop_session_id}.txt"
        try:
            with open(summary_file, "w") as f:
                f.write(f"Session: {self._state.workshop_session_id}\n")
                f.write(f"Claude Session: {self._state.claude_session_id}\n")
                f.write(f"Turns: {self._state.turn_count}\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write("-" * 40 + "\n\n")
                f.write(summary)
            log.debug(f"Saved session summary to {summary_file}")
        except Exception as e:
            log.warning(f"Could not save session summary: {e}")

    def should_summarize(self, token_threshold: int = 100000) -> bool:
        """
        Check if the session should be summarized based on estimated token usage.

        Args:
            token_threshold: Trigger summarization when estimated tokens exceed this

        Returns:
            True if summarization is recommended
        """
        if not self._state:
            return False
        return self._state.context_tokens_estimated > token_threshold


# Singleton instance
_bridge_manager: Optional[ClaudeBridgeManager] = None


def get_claude_bridge_manager() -> ClaudeBridgeManager:
    """Get the global ClaudeBridgeManager singleton."""
    global _bridge_manager
    if _bridge_manager is None:
        _bridge_manager = ClaudeBridgeManager.get_instance()
    return _bridge_manager


def get_claude_bridge(timeout_seconds: int = 180) -> ClaudeCodeBridge:
    """
    Get the shared Claude Code bridge instance.

    This is the recommended way to get a bridge for most use cases.
    The shared instance maintains session continuity across components.

    For isolated contexts (e.g., subagent with separate context window),
    create a dedicated ClaudeCodeBridge() instance instead.
    """
    return get_claude_bridge_manager().get_bridge(timeout_seconds)


# Convenience function for simple queries
async def quick_query(prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Quick one-off query to Claude Code.

    Usage:
        result = await quick_query("What is 2 + 2?")
    """
    bridge = ClaudeCodeBridge()
    messages = [{"role": "user", "content": prompt}]
    response = await bridge.query(messages, system_prompt=system_prompt)
    return response.get("content", "")


# ============================================================================
# RESEARCH HELPER FUNCTIONS
# These replace Ollama phi4:14b calls in deep_research.py and fetch_url.py
# ============================================================================

# Claude's context window is 200K+ tokens (~800K chars)
# We can process most web pages in a single pass without chunking
CLAUDE_SINGLE_PASS_LIMIT = 400_000  # ~100K tokens, leaves room for output


async def distill_content(
    content: str,
    focus: str = "",
    context_title: str = "",
    preserve_technical: bool = True
) -> str:
    """
    Distill content using Claude Code, preserving key information.

    Replaces Ollama phi4:14b distillation in deep_research.py and fetch_url.py.
    With Claude's 200K+ context, we can process much larger content without chunking.

    Args:
        content: Raw content to distill (up to 400K chars in single pass)
        focus: Topic focus to prioritize during distillation
        context_title: Source title for context (e.g., "Section 1 of 3")
        preserve_technical: If True, preserve code, numbers, technical details

    Returns:
        Distilled content string (or original content on failure)
    """
    if not content or len(content.strip()) < 50:
        return content

    bridge = get_claude_bridge(timeout_seconds=180)

    system_prompt = """You are a research content distillation specialist. Extract and preserve ALL important information while condensing length.

Your distillation MUST include:
- Core facts, claims, and arguments with context
- Important quotes (preserve exact wording when significant)
- Specific data: numbers, statistics, percentages, dates, versions
- Technical details, terminology, definitions, specifications
- Code examples, commands, API details (preserve exactly)
- Examples, case studies, and evidence
- Relationships, causation, and logical connections
- Names, entities, organizations, and their roles
- URLs, file paths, and references mentioned

Be COMPREHENSIVE - this is distillation for research, not summarization.
Preserve specifics and technical details. Write in clear, organized prose."""

    focus_line = f"Topic focus: {focus}\n" if focus else ""
    context_line = f"Source: {context_title}\n" if context_title else ""

    user_prompt = f"""Distill this content into rich, detailed key information for research.
{focus_line}{context_line}
CONTENT:
{content}

COMPREHENSIVE DISTILLATION (preserve all important information):"""

    messages = [{"role": "user", "content": user_prompt}]

    try:
        result = await bridge.query(
            messages,
            system_prompt=system_prompt,
            disable_native_tools=True,
            max_turns=1
        )
        distilled = result.get("content", "")
        if distilled and len(distilled) > 50:
            log.debug(f"Distilled content: {len(content)} -> {len(distilled)} chars")
            return distilled
        return content  # Return original if distillation failed
    except Exception as e:
        log.error(f"Distillation failed: {e}")
        return content  # Fallback to original content


async def extract_key_points(
    content: str,
    source_title: str = "",
    max_points: int = 10
) -> List[str]:
    """
    Extract key points from content using Claude Code.

    Replaces Ollama phi4:14b key point extraction in deep_research.py.

    Args:
        content: Distilled or raw content to analyze
        source_title: Source title for context
        max_points: Maximum key points to extract (5-10 recommended)

    Returns:
        List of key point strings (empty list on failure)
    """
    if not content or len(content.strip()) < 100:
        return []

    bridge = get_claude_bridge(timeout_seconds=60)

    source_line = f"Source: {source_title}\n" if source_title else ""

    prompt = f"""Extract the most important key points from this research content.
{source_line}
CONTENT:
{content}

List the {max_points} most important key points as a numbered list. Each point should be a complete, specific statement that captures important information (facts, data, findings, recommendations).

KEY POINTS:"""

    messages = [{"role": "user", "content": prompt}]

    try:
        result = await bridge.query(
            messages,
            disable_native_tools=True,
            max_turns=1
        )

        response = result.get("content", "")

        # Parse numbered list from response
        points = []
        for line in response.split('\n'):
            line = line.strip()
            match = re.match(r'^[\d\.\-\*]+\s*(.+)$', line)
            if match:
                point = match.group(1).strip()
                if len(point) > 20:  # Filter out short/empty points
                    points.append(point)

        log.debug(f"Extracted {len(points)} key points from content")
        return points[:max_points]

    except Exception as e:
        log.error(f"Key point extraction failed: {e}")
        return []


async def synthesize_content(
    sources: List[str],
    topic: str,
    is_final: bool = False
) -> str:
    """
    Synthesize multiple content sources/distillations into coherent output.

    Replaces Ollama phi4:14b synthesis in deep_research.py.

    Args:
        sources: List of content strings to synthesize
        topic: Research topic for context
        is_final: If True, generate final research summary; if False, intermediate synthesis

    Returns:
        Synthesized content string (or combined sources on failure)
    """
    if not sources:
        return ""

    if len(sources) == 1:
        return sources[0]

    bridge = get_claude_bridge(timeout_seconds=180)

    if is_final:
        system_prompt = """You are synthesizing research findings into a final comprehensive summary.

Your synthesis MUST:
- Integrate information from all sources coherently
- Preserve ALL key facts, data, technical details, and specifics
- Eliminate redundancy while keeping all unique information
- Organize by theme or topic for clarity
- Maintain accuracy - do not add information not in the sources
- Include specific numbers, quotes, code examples, URLs mentioned
- Note any contradictions or varying perspectives between sources
- Conclude with confidence assessment and knowledge gaps if relevant"""
    else:
        system_prompt = """You are synthesizing multiple content sections into a coherent intermediate summary.

Your task:
- Combine information from all sections logically
- Preserve ALL key facts, data, quotes, and specifics - this is for research
- Remove redundancy but keep all unique information
- Maintain technical accuracy and detail"""

    # Combine sources with clear separation
    combined_sources = "\n\n---\n\n".join(
        f"### Source {i+1}\n{source}" for i, source in enumerate(sources)
    )

    summary_type = "summary" if is_final else "intermediate synthesis"
    prompt = f"""Synthesize these {len(sources)} research sections into a comprehensive {summary_type}.

Topic: {topic}

SECTIONS TO SYNTHESIZE:
{combined_sources}

COMPREHENSIVE SYNTHESIS (preserve all important information):"""

    messages = [{"role": "user", "content": prompt}]

    try:
        result = await bridge.query(
            messages,
            system_prompt=system_prompt,
            disable_native_tools=True,
            max_turns=1
        )
        synthesized = result.get("content", "")
        if synthesized and len(synthesized) > 50:
            log.debug(f"Synthesized {len(sources)} sources into {len(synthesized)} chars")
            return synthesized
        return combined_sources  # Return combined on failure
    except Exception as e:
        log.error(f"Synthesis failed: {e}")
        return combined_sources  # Fallback to combined sources


# Test function
async def _test_bridge():
    """Test the Claude Code bridge."""
    print("Testing Claude Code Bridge...")

    bridge = ClaudeCodeBridge()

    # Test 1: Simple query
    print("\n1. Simple query test:")
    messages = [{"role": "user", "content": "What is 2 + 2? Reply with just the number."}]
    response = await bridge.query(messages)
    print(f"   Response: {response['content'][:100]}")

    # Test 2: With system prompt
    print("\n2. System prompt test:")
    messages = [{"role": "user", "content": "Introduce yourself briefly."}]
    response = await bridge.query(
        messages,
        system_prompt="You are Workshop, a helpful AI assistant. Keep responses under 50 words."
    )
    print(f"   Response: {response['content'][:200]}")

    # Test 3: Tool call format
    print("\n3. Tool call extraction test:")
    messages = [{"role": "user", "content": "Read the file at ~/test.txt"}]
    response = await bridge.query(
        messages,
        system_prompt="""You are Workshop. When asked to read a file, output:
<tool_call>
{"tool": "read_file", "args": {"path": "the/file/path"}}
</tool_call>
Then wait for the result."""
    )
    print(f"   Content: {response['content'][:200]}")
    print(f"   Tool calls: {response['tool_calls']}")

    print("\nAll tests completed!")


if __name__ == "__main__":
    asyncio.run(_test_bridge())
