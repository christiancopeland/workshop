"""
Haiku Progress Manager - Real-time contextual progress updates for voice mode.

Uses Claude Haiku to generate natural, contextual progress summaries
that help users understand what Workshop is doing during long operations.

This addresses the "5+ minutes of silence" problem where users have no
feedback during complex tool execution chains.

Architecture:
- HaikuProgressManager tracks tool execution context
- Generates spoken updates via Haiku model (fast, cheap)
- Rate-limited to prevent spam (MIN_UPDATE_INTERVAL_SEC)
- Graceful fallback to simple updates if Haiku unavailable

Integration points:
- skill_executor.py: on_tool_start(), on_tool_complete()
- hooks.py: Can also be triggered via POST_TOOL_USE hooks

Logging:
- All operations logged to data/logs/ via workshop logger
- Debug level: detailed context, prompt generation
- Info level: update generation, TTS dispatch
- Warning level: failures, fallbacks
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Awaitable
from datetime import datetime

from logger import get_logger

log = get_logger("haiku_progress")


@dataclass
class ToolExecution:
    """Record of a single tool execution for context building."""
    tool_name: str
    args: Dict[str, Any]
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: int = 0
    result_summary: str = ""
    status: str = "running"  # running, complete, failed
    error: Optional[str] = None


@dataclass
class ProgressContext:
    """
    Accumulated context for generating progress updates.

    Maintains state across multiple tool calls to provide
    contextual, informative progress updates.
    """
    session_start: datetime = field(default_factory=datetime.now)
    original_query: str = ""
    current_task: str = ""
    skill_name: str = ""
    tools_executed: List[ToolExecution] = field(default_factory=list)
    findings_so_far: List[str] = field(default_factory=list)
    last_update_time: Optional[datetime] = None
    update_count: int = 0
    total_duration_ms: int = 0


class HaikuProgressManager:
    """
    Manages real-time progress updates using Haiku for natural language generation.

    Design principles:
    - Updates should be 1-2 sentences, natural for TTS
    - Context-aware: knows what tools ran, what was found
    - Rate-limited: no more than 1 update per MIN_UPDATE_INTERVAL_SEC
    - Graceful degradation: if Haiku fails, fall back to simple updates

    Usage:
        manager = HaikuProgressManager(
            claude_bridge=bridge,
            tts_callback=voice_stack.speak_async
        )
        manager.start_session("research Anthropic agent patterns", "Research")

        await manager.on_tool_start("web_search", {"query": "agent patterns"})
        # ... tool executes ...
        await manager.on_tool_complete("web_search", result, 1500)
    """

    # Rate limiting: minimum seconds between updates
    MIN_UPDATE_INTERVAL_SEC = 8

    # Tool categories for contextual update generation
    RESEARCH_TOOLS = {"web_search", "deep_research", "fetch_url", "fetch_urls_for_research"}
    FILE_TOOLS = {"read_file", "write_file", "list_files", "get_file_content"}
    MEMORY_TOOLS = {"remember", "recall", "search_memories"}

    def __init__(
        self,
        claude_bridge=None,
        tts_callback: Optional[Callable[[str], Awaitable[None]]] = None,
        voice_mode: bool = True,
    ):
        """
        Initialize the progress manager.

        Args:
            claude_bridge: ClaudeCodeBridge instance for Haiku queries
            tts_callback: Async function to speak updates (e.g., voice_stack.speak_async)
            voice_mode: Whether to actually speak updates (False = log only)
        """
        self.bridge = claude_bridge
        self.tts_callback = tts_callback
        self.voice_mode = voice_mode
        self.context = ProgressContext()
        self._lock = asyncio.Lock()
        self._enabled = True

        log.info(f"[HAIKU_PROGRESS] Initialized (voice_mode={voice_mode}, bridge={'yes' if self.bridge else 'no'}, tts={'yes' if tts_callback else 'no'})")

    def start_session(self, query: str, skill_name: str = "", task: str = ""):
        """
        Start a new progress tracking session.

        Called at the beginning of skill execution to establish context.

        Args:
            query: The user's original query
            skill_name: Name of the skill being executed
            task: Current task description (if any)
        """
        self.context = ProgressContext(
            session_start=datetime.now(),
            original_query=query,
            skill_name=skill_name,
            current_task=task,
        )
        log.info(f"[HAIKU_PROGRESS] Session started: skill={skill_name}, query={query[:50]}...")
        log.debug(f"[HAIKU_PROGRESS] Full query: {query}")

    def set_current_task(self, task: str):
        """Update the current task being worked on."""
        self.context.current_task = task
        log.debug(f"[HAIKU_PROGRESS] Task updated: {task[:100]}")

    def disable(self):
        """Disable progress updates (useful for batch operations)."""
        self._enabled = False
        log.info("[HAIKU_PROGRESS] Progress updates disabled")

    def enable(self):
        """Re-enable progress updates."""
        self._enabled = True
        log.info("[HAIKU_PROGRESS] Progress updates enabled")

    async def on_tool_start(self, tool_name: str, args: Dict[str, Any]):
        """
        Called when a tool starts executing.

        Records the tool execution and optionally generates a "starting" update.

        Args:
            tool_name: Name of the tool being called
            args: Arguments passed to the tool
        """
        if not self._enabled:
            return

        # Record the execution
        execution = ToolExecution(
            tool_name=tool_name,
            args=args,
            started_at=datetime.now(),
        )
        self.context.tools_executed.append(execution)

        log.info(f"[HAIKU_PROGRESS] Tool started: {tool_name}")
        log.debug(f"[HAIKU_PROGRESS] Tool args: {self._safe_args_str(args)}")

        # Generate quick "starting" update for long-running tools
        if tool_name in self.RESEARCH_TOOLS:
            await self._maybe_generate_update(
                event_type="tool_start",
                tool_name=tool_name,
                args=args
            )

    async def on_tool_complete(
        self,
        tool_name: str,
        result: Any,
        duration_ms: int,
        error: Optional[str] = None
    ):
        """
        Called when a tool completes.

        Updates the execution record and generates a completion update.

        Args:
            tool_name: Name of the tool that completed
            result: Result returned by the tool
            duration_ms: Execution duration in milliseconds
            error: Error message if tool failed
        """
        if not self._enabled:
            return

        # Find and update the execution record
        for execution in reversed(self.context.tools_executed):
            if execution.tool_name == tool_name and execution.status == "running":
                execution.completed_at = datetime.now()
                execution.duration_ms = duration_ms
                execution.status = "failed" if error else "complete"
                execution.error = error
                execution.result_summary = self._summarize_result(tool_name, result)
                break

        self.context.total_duration_ms += duration_ms

        log.info(f"[HAIKU_PROGRESS] Tool completed: {tool_name} ({duration_ms}ms)")
        log.debug(f"[HAIKU_PROGRESS] Result summary: {execution.result_summary[:200] if execution.result_summary else 'none'}")

        # Extract key finding if present
        finding = self._extract_finding(tool_name, result)
        if finding:
            self.context.findings_so_far.append(finding)
            log.debug(f"[HAIKU_PROGRESS] Finding extracted: {finding}")

        # Generate completion update
        await self._maybe_generate_update(
            event_type="tool_complete",
            tool_name=tool_name,
            result=result,
            duration_ms=duration_ms,
            error=error
        )

    async def _maybe_generate_update(self, event_type: str, **kwargs):
        """
        Generate and speak a progress update if appropriate.

        Implements rate limiting and delegates to Haiku for natural language.
        """
        async with self._lock:
            # Rate limiting check
            now = datetime.now()
            if self.context.last_update_time:
                elapsed = (now - self.context.last_update_time).total_seconds()
                if elapsed < self.MIN_UPDATE_INTERVAL_SEC:
                    log.debug(f"[HAIKU_PROGRESS] Rate limited ({elapsed:.1f}s < {self.MIN_UPDATE_INTERVAL_SEC}s)")
                    return

            tool_name = kwargs.get("tool_name", "")
            log.debug(f"[HAIKU_PROGRESS] Generating update: event={event_type}, tool={tool_name}")

            # Generate update text
            start_time = time.time()
            update_text = await self._generate_haiku_update(event_type, **kwargs)
            generation_time = int((time.time() - start_time) * 1000)

            if update_text:
                self.context.last_update_time = now
                self.context.update_count += 1

                log.info(f"[HAIKU_PROGRESS] Update #{self.context.update_count}: \"{update_text}\" ({generation_time}ms)")

                # Speak if TTS available and voice mode enabled
                if self.voice_mode and self.tts_callback:
                    try:
                        log.debug(f"[HAIKU_PROGRESS] Dispatching to TTS: {update_text[:50]}...")
                        await self.tts_callback(update_text)
                        log.debug("[HAIKU_PROGRESS] TTS dispatch successful")
                    except Exception as e:
                        log.warning(f"[HAIKU_PROGRESS] TTS dispatch failed: {type(e).__name__}: {e}")
                else:
                    # Log why TTS was skipped
                    log.warning(f"[HAIKU_PROGRESS] TTS SKIPPED: voice_mode={self.voice_mode}, tts_callback={'set' if self.tts_callback else 'NONE'}")
            else:
                log.debug("[HAIKU_PROGRESS] No update generated (empty or filtered)")

    async def _generate_haiku_update(self, event_type: str, **kwargs) -> Optional[str]:
        """
        Use Haiku to generate a natural progress update.

        Falls back to simple updates if Haiku is unavailable.
        """
        if not self.bridge:
            log.debug("[HAIKU_PROGRESS] No bridge available, using fallback")
            return self._fallback_update(event_type, **kwargs)

        # Build context summary for Haiku
        elapsed = (datetime.now() - self.context.session_start).total_seconds()
        tools_count = len(self.context.tools_executed)
        findings_count = len(self.context.findings_so_far)
        completed_tools = [t for t in self.context.tools_executed if t.status == "complete"]

        tool_name = kwargs.get("tool_name", "")
        result = kwargs.get("result", "")
        error = kwargs.get("error")

        # Build the prompt
        prompt = f"""Generate a brief spoken progress update (1-2 sentences max).

CONTEXT:
- User asked: "{self.context.original_query[:100]}..."
- Skill: {self.context.skill_name}
- Time elapsed: {int(elapsed)} seconds
- Tools used so far: {tools_count}
- Completed successfully: {len(completed_tools)}
- Findings extracted: {findings_count}

CURRENT EVENT: {event_type}
- Tool: {tool_name}
{self._format_event_details(event_type, kwargs)}

REQUIREMENTS:
- Natural speech only, NO markdown formatting
- 1-2 sentences maximum
- Be specific about what's happening or what was found
- Sound helpful and informative, not robotic
- If error occurred, briefly mention it

Generate the spoken update:"""

        log.debug(f"[HAIKU_PROGRESS] Haiku prompt:\n{prompt[:500]}...")

        try:
            result = await self.bridge.query(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You generate brief spoken progress updates for a voice assistant. Be concise and natural. Never use markdown, bullet points, or formatting. Just speak naturally in 1-2 sentences.",
                model="haiku",
                disable_native_tools=True,
                max_turns=1
            )

            text = result.get("content", "").strip()
            log.debug(f"[HAIKU_PROGRESS] Haiku raw response: {text[:200]}")

            # Sanitize any accidental markdown
            text = self._sanitize_for_voice(text)

            # Validate length (reject if too short or too long)
            if len(text) < 10:
                log.debug(f"[HAIKU_PROGRESS] Response too short ({len(text)} chars), using fallback")
                return self._fallback_update(event_type, **kwargs)
            if len(text) > 250:
                log.debug(f"[HAIKU_PROGRESS] Response too long ({len(text)} chars), truncating")
                # Find a good sentence break point
                text = self._truncate_to_sentence(text, 200)

            return text

        except Exception as e:
            log.warning(f"[HAIKU_PROGRESS] Haiku query failed: {e}")
            return self._fallback_update(event_type, **kwargs)

    def _fallback_update(self, event_type: str, **kwargs) -> str:
        """
        Simple fallback updates when Haiku is unavailable.

        Returns a basic but informative update string.
        """
        tool_name = kwargs.get("tool_name", "")
        error = kwargs.get("error")

        log.debug(f"[HAIKU_PROGRESS] Using fallback for {event_type}/{tool_name}")

        if error:
            return f"The {tool_name.replace('_', ' ')} encountered an issue. Moving on."

        fallbacks = {
            "tool_start": {
                "web_search": "Searching the web now.",
                "deep_research": "Starting deep research. This may take a moment.",
                "fetch_url": "Fetching the page content.",
                "fetch_urls_for_research": "Fetching multiple pages for analysis.",
                "read_file": "Reading the file.",
                "get_file_content": "Reading the file content.",
                "search_project_files": "Searching through project files.",
                "remember": "Saving to memory.",
                "recall": "Searching my memory.",
            },
            "tool_complete": {
                "web_search": "Search complete. Processing results.",
                "deep_research": "Research phase complete.",
                "fetch_url": "Page content retrieved.",
                "fetch_urls_for_research": "All pages fetched. Analyzing content.",
                "read_file": "File read successfully.",
                "get_file_content": "Got the file content.",
                "search_project_files": "Found some relevant files.",
                "remember": "Saved to memory.",
                "recall": "Retrieved from memory.",
            }
        }

        event_fallbacks = fallbacks.get(event_type, {})
        fallback = event_fallbacks.get(tool_name)

        if fallback:
            return fallback

        # Generic fallback
        action = "Starting" if event_type == "tool_start" else "Completed"
        return f"{action} {tool_name.replace('_', ' ')}."

    def _format_event_details(self, event_type: str, kwargs: Dict) -> str:
        """Format event-specific details for the Haiku prompt."""
        if event_type == "tool_complete":
            result = kwargs.get("result", "")
            duration = kwargs.get("duration_ms", 0)
            error = kwargs.get("error")

            if error:
                return f"- Duration: {duration}ms\n- ERROR: {error}"

            summary = self._summarize_result(kwargs.get("tool_name", ""), result)
            return f"- Duration: {duration}ms\n- Result: {summary[:300]}"

        elif event_type == "tool_start":
            args = kwargs.get("args", {})
            return f"- Args: {self._safe_args_str(args)}"

        return ""

    def _summarize_result(self, tool_name: str, result: Any) -> str:
        """Create a brief summary of a tool result for context."""
        if result is None:
            return "No result returned"

        result_str = str(result)

        # Tool-specific summarization
        if tool_name == "web_search":
            # Count results if it looks like a list/dict
            if "results" in result_str.lower():
                return f"Search returned results ({len(result_str)} chars)"
            return f"Search results ({len(result_str)} chars)"

        elif tool_name in ("fetch_url", "fetch_urls_for_research"):
            return f"Retrieved content ({len(result_str)} chars)"

        elif tool_name == "read_file":
            lines = result_str.count('\n') + 1
            return f"File content ({lines} lines, {len(result_str)} chars)"

        # Generic summary
        if len(result_str) > 200:
            return f"{result_str[:200]}... ({len(result_str)} chars total)"
        return result_str

    def _extract_finding(self, tool_name: str, result: Any) -> Optional[str]:
        """Extract a key finding from tool result if present."""
        if not result:
            return None

        result_str = str(result)

        # Tool-specific extraction
        if tool_name == "web_search":
            if len(result_str) > 50:
                return f"Web search returned results"
        elif tool_name in ("fetch_url", "fetch_urls_for_research"):
            if len(result_str) > 100:
                return f"Retrieved web content ({len(result_str)} chars)"
        elif tool_name == "deep_research":
            if "sources" in result_str.lower() or "findings" in result_str.lower():
                return "Deep research completed with findings"

        return None

    def _safe_args_str(self, args: Dict[str, Any], max_len: int = 100) -> str:
        """Safely convert args to string, truncating long values."""
        safe_args = {}
        for k, v in args.items():
            v_str = str(v)
            if len(v_str) > 50:
                safe_args[k] = f"{v_str[:50]}..."
            else:
                safe_args[k] = v

        result = str(safe_args)
        if len(result) > max_len:
            return result[:max_len] + "..."
        return result

    def _sanitize_for_voice(self, text: str) -> str:
        """Remove any markdown artifacts from Haiku's response."""
        import re

        # Remove markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold**
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic*
        text = re.sub(r'`([^`]+)`', r'\1', text)        # `code`
        text = re.sub(r'#{1,6}\s*', '', text)           # Headers
        text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)  # Bullets
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # Numbers

        # Clean whitespace
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r' +', ' ', text)

        return text.strip()

    def _truncate_to_sentence(self, text: str, max_len: int) -> str:
        """Truncate text to a sentence boundary within max_len."""
        if len(text) <= max_len:
            return text

        # Find last sentence end within limit
        truncated = text[:max_len]

        for end_char in ['. ', '! ', '? ']:
            last_pos = truncated.rfind(end_char)
            if last_pos > max_len // 2:  # At least half the content
                return truncated[:last_pos + 1].strip()

        # No good break found, just truncate at word boundary
        last_space = truncated.rfind(' ')
        if last_space > max_len // 2:
            return truncated[:last_space].strip() + "."

        return truncated.strip() + "."

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the progress session.

        Useful for debugging and telemetry.
        """
        elapsed = (datetime.now() - self.context.session_start).total_seconds()
        completed = [t for t in self.context.tools_executed if t.status == "complete"]
        failed = [t for t in self.context.tools_executed if t.status == "failed"]

        summary = {
            "duration_seconds": int(elapsed),
            "total_tool_time_ms": self.context.total_duration_ms,
            "tools_executed": len(self.context.tools_executed),
            "tools_completed": len(completed),
            "tools_failed": len(failed),
            "updates_generated": self.context.update_count,
            "findings_extracted": len(self.context.findings_so_far),
            "skill_name": self.context.skill_name,
            "original_query": self.context.original_query[:100],
        }

        log.info(f"[HAIKU_PROGRESS] Session summary: {summary}")
        return summary


# Singleton instance for easy access
_progress_manager: Optional[HaikuProgressManager] = None


def get_progress_manager() -> Optional[HaikuProgressManager]:
    """Get the global progress manager instance."""
    return _progress_manager


def init_progress_manager(
    claude_bridge=None,
    tts_callback=None,
    voice_mode: bool = True
) -> HaikuProgressManager:
    """
    Initialize the global progress manager.

    Should be called once during Workshop startup when voice mode is enabled.
    """
    global _progress_manager
    _progress_manager = HaikuProgressManager(
        claude_bridge=claude_bridge,
        tts_callback=tts_callback,
        voice_mode=voice_mode
    )
    log.info("[HAIKU_PROGRESS] Global progress manager initialized")
    return _progress_manager


def reset_progress_manager():
    """Reset the global progress manager (useful for testing)."""
    global _progress_manager
    _progress_manager = None
    log.info("[HAIKU_PROGRESS] Global progress manager reset")
