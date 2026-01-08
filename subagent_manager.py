"""
Subagent Manager - Orchestrates specialized agent spawning.

Phase 5 Migration (January 2026): Now supports Claude Code as reasoning backend.

Features:
1. Claude Code integration via ClaudeCodeBridge (subscription-based, no per-token cost)
2. Legacy Ollama support for local model fallback
3. Context compression for agent handoffs
4. Specialist agent spawning with fresh context windows

When using Claude Code (use_claude=True, default):
- Each subagent gets a fresh context window (prevents context pollution)
- No VRAM management needed (cloud-based)
- Superior reasoning quality over local models

Legacy Ollama mode (use_claude=False):
- VRAM-aware model selection for 12GB constraint
- Model preloading to hide cold-start latency
- Parallel model swap orchestration
"""

import asyncio
import aiohttp
import json
import re
import yaml
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable, Awaitable
from dataclasses import dataclass, field, asdict

from config import Config
from logger import get_logger

# Haiku progress manager for voice updates during subagent execution
try:
    from haiku_progress import HaikuProgressManager
    HAIKU_AVAILABLE = True
except ImportError:
    HaikuProgressManager = None
    HAIKU_AVAILABLE = False

# Claude Code integration for subscription-based reasoning
# Phase 4: Subagents use DEDICATED bridges (not shared) for isolated context windows
try:
    from claude_bridge import ClaudeCodeBridge, get_claude_bridge
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    get_claude_bridge = None

log = get_logger("subagent_manager")

# =============================================================================
# Trace Logging System - Comprehensive context visibility for debugging
# =============================================================================

def _trace_log(tag: str, content: str, max_length: int = None):
    """
    Log trace information with consistent formatting.

    Tags:
    - CONTEXT_IN: Full context being sent to LLM
    - CONTEXT_OUT: Full response from LLM
    - TOOL_ARGS: Full tool arguments
    - TOOL_RESULT: Full tool execution result
    - CONTEXT_SIZE: Size metrics for context
    """
    if not Config.TRACE_LOGGING_ENABLED:
        return

    max_len = max_length or Config.TRACE_MAX_CONTENT_LENGTH

    if len(content) > max_len:
        truncated = content[:max_len] + f"\n... [TRUNCATED - {len(content)} total chars]"
        log.info(f"[TRACE:{tag}] {truncated}")
    else:
        log.info(f"[TRACE:{tag}] {content}")


def _trace_context_sizes(messages: list, system_prompt: str = None, iteration: int = 0):
    """Log detailed size breakdown of context being sent to LLM."""
    if not Config.TRACE_LOGGING_ENABLED:
        return

    sizes = []
    total_chars = 0

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        size = len(content)
        total_chars += size
        sizes.append(f"  msg[{i}] {role}: {size:,} chars")

    sys_size = len(system_prompt) if system_prompt else 0
    total_chars += sys_size

    breakdown = "\n".join(sizes)
    log.info(f"[TRACE:CONTEXT_SIZE] Iteration {iteration} - Total: {total_chars:,} chars")
    log.info(f"[TRACE:CONTEXT_SIZE] System prompt: {sys_size:,} chars")
    log.info(f"[TRACE:CONTEXT_SIZE] Messages ({len(messages)}):\n{breakdown}")

    # Warn if context is getting large
    if total_chars > 50000:
        log.warning(f"[TRACE:CONTEXT_SIZE] ⚠️ LARGE CONTEXT WARNING: {total_chars:,} chars")
    if total_chars > 100000:
        log.error(f"[TRACE:CONTEXT_SIZE] 🚨 CRITICAL: Context exceeds 100KB ({total_chars:,} chars)")


def _trace_messages_content(messages: list, label: str = "CONTEXT_IN"):
    """Log full message contents for debugging."""
    if not Config.TRACE_LOGGING_ENABLED or not Config.TRACE_LOG_FULL_CONTEXT:
        return

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        _trace_log(f"{label}:msg[{i}]:{role}", content)


def _trace_tool_execution(tool_name: str, args: dict, result: str, duration_ms: int):
    """Log full tool execution details."""
    if not Config.TRACE_LOGGING_ENABLED:
        return

    # Always log tool args
    args_str = json.dumps(args, indent=2, default=str)
    log.info(f"[TRACE:TOOL_EXEC] {tool_name} ({duration_ms}ms)")
    _trace_log(f"TOOL_ARGS:{tool_name}", args_str)

    # Log full result if enabled
    if Config.TRACE_LOG_TOOL_RESULTS:
        _trace_log(f"TOOL_RESULT:{tool_name}", result)
    else:
        log.info(f"[TRACE:TOOL_RESULT:{tool_name}] {len(result):,} chars (content logging disabled)")


# =============================================================================
# Tool Parsing for Subagents (non-native tool calling)
# =============================================================================

def _parse_tool_calls(content: str) -> List[Dict]:
    """
    Parse <tool_call> blocks from LLM output.

    Returns list of dicts with 'tool' and 'args' keys.
    """
    tool_calls = []

    # Find all <tool_call>...</tool_call> blocks
    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)

    log.debug(f"[SUBAGENT_TOOL_PARSE] Found {len(matches)} <tool_call> blocks in {len(content)} char response")

    for match in matches:
        try:
            data = _repair_and_parse_json(match)
            if data and isinstance(data, dict):
                tool_name = data.get("tool", "")
                tool_args = data.get("args", {})
                if tool_name:
                    tool_calls.append({
                        "tool": tool_name,
                        "args": tool_args if isinstance(tool_args, dict) else {},
                        "raw": match
                    })
                    log.debug(f"[SUBAGENT_TOOL_PARSE] Parsed tool: {tool_name}({list(tool_args.keys())})")
        except Exception as e:
            log.warning(f"[SUBAGENT_TOOL_PARSE] Failed to parse tool call: {e}\nRaw: {match[:200]}")

    return tool_calls


def _repair_and_parse_json(text: str) -> Optional[Dict]:
    """Attempt to parse JSON with repair strategies for common LLM mistakes."""
    text = text.strip()

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Replace single quotes with double
    try:
        fixed = text.replace("'", '"')
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Fix common issues
    try:
        fixed = text
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)  # Remove trailing commas
        fixed = re.sub(r"'([^']*)'", r'"\1"', fixed)  # Replace single quotes
        fixed = re.sub(r'(?<!\\)\n', r'\\n', fixed)   # Escape newlines
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Strategy 4: Extract key-value pairs manually
    try:
        tool_match = re.search(r'"?tool"?\s*:\s*"([^"]+)"', text)
        args_match = re.search(r'"?args"?\s*:\s*(\{[^}]*\})', text)

        if tool_match:
            tool_name = tool_match.group(1)
            args = {}
            if args_match:
                try:
                    args = json.loads(args_match.group(1))
                except:
                    arg_pattern = r'"([^"]+)"\s*:\s*"([^"]*)"'
                    for key, value in re.findall(arg_pattern, args_match.group(1)):
                        args[key] = value
            return {"tool": tool_name, "args": args}
    except:
        pass

    log.warning(f"Could not repair JSON: {text[:100]}")
    return None


def _clean_tool_calls_from_response(content: str) -> str:
    """Remove <tool_call> blocks from response text."""
    cleaned = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()

# Try to import dashboard for event emission
try:
    from dashboard import get_dashboard
    _dashboard_available = True
except ImportError:
    _dashboard_available = False


async def _emit_dashboard_event(event_name: str, trace_id: str = None, **kwargs):
    """Emit a dashboard event if dashboard is available.

    Args:
        event_name: Name of the event (e.g., 'subagent_spawning')
        trace_id: Optional trace ID to correlate events with parent request
        **kwargs: Event-specific data
    """
    if not _dashboard_available:
        return
    try:
        dashboard = get_dashboard()
        if dashboard and dashboard.clients:
            emit_func = getattr(dashboard, f"emit_{event_name}", None)
            if emit_func:
                # Include trace_id in the call
                await emit_func(trace_id=trace_id, **kwargs)
    except Exception as e:
        log.debug(f"Dashboard event emission failed: {e}")


@dataclass
class SubagentToolCall:
    """Record of a single tool call made by a subagent."""
    call_id: str
    tool_name: str
    args: Dict[str, Any]
    result: str
    duration_ms: int
    timestamp: str
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SubagentLLMCall:
    """Record of a single LLM call made by a subagent."""
    call_id: str
    iteration: int
    model: str
    messages_count: int
    response_length: int
    duration_ms: int
    timestamp: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls_found: int = 0
    response_preview: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SubagentTrace:
    """Complete execution trace for a subagent."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    parent_trace_id: Optional[str] = None

    # Timing
    started_at: str = ""
    completed_at: str = ""
    total_duration_ms: int = 0

    # Execution stats
    llm_call_count: int = 0
    tool_call_count: int = 0
    total_tokens: int = 0

    # Detailed records
    llm_calls: List[SubagentLLMCall] = field(default_factory=list)
    tool_calls: List[SubagentToolCall] = field(default_factory=list)

    # Context provided to subagent
    system_prompt: str = ""
    context_injected: str = ""
    user_message: str = ""

    # Final output
    final_output: str = ""
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "trace_id": self.trace_id,
            "parent_trace_id": self.parent_trace_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_duration_ms": self.total_duration_ms,
            "llm_call_count": self.llm_call_count,
            "tool_call_count": self.tool_call_count,
            "total_tokens": self.total_tokens,
            "llm_calls": [c.to_dict() for c in self.llm_calls],
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "system_prompt": self.system_prompt,
            "context_injected": self.context_injected,
            "user_message": self.user_message,
            "final_output": self.final_output,
            "success": self.success,
            "error_message": self.error_message
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SubagentTrace':
        llm_calls = [SubagentLLMCall(**c) for c in data.get("llm_calls", [])]
        tool_calls = [SubagentToolCall(**c) for c in data.get("tool_calls", [])]
        return cls(
            trace_id=data.get("trace_id", ""),
            parent_trace_id=data.get("parent_trace_id"),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at", ""),
            total_duration_ms=data.get("total_duration_ms", 0),
            llm_call_count=data.get("llm_call_count", 0),
            tool_call_count=data.get("tool_call_count", 0),
            total_tokens=data.get("total_tokens", 0),
            llm_calls=llm_calls,
            tool_calls=tool_calls,
            system_prompt=data.get("system_prompt", ""),
            context_injected=data.get("context_injected", ""),
            user_message=data.get("user_message", ""),
            final_output=data.get("final_output", ""),
            success=data.get("success", True),
            error_message=data.get("error_message")
        )


@dataclass
class ContextSnapshot:
    """Serializable snapshot of agent context for swap preservation."""
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Primary agent state
    primary_model: str = ""
    conversation_summary: str = ""
    last_user_message: str = ""
    pending_task: str = ""

    # Telos context
    telos_profile_summary: str = ""
    telos_active_project: str = ""
    telos_current_goals: List[str] = field(default_factory=list)

    # Research context
    research_platform_path: str = ""
    research_topic: str = ""
    research_source_count: int = 0
    research_key_findings: str = ""

    # Subagent task
    subagent_name: str = ""
    subagent_model: str = ""
    task_description: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)

    # Execution trace (populated after subagent completes)
    trace: Optional[SubagentTrace] = None

    # Execution status
    status: str = "pending"  # pending, running, completed, failed
    output: str = ""
    duration_ms: int = 0

    def to_dict(self) -> Dict:
        result = asdict(self)
        # Handle nested trace object
        if self.trace:
            result["trace"] = self.trace.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> 'ContextSnapshot':
        trace_data = data.pop("trace", None)
        # Remove fields not in __init__
        data.pop("telos_current_goals", None)
        snapshot = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        if trace_data:
            snapshot.trace = SubagentTrace.from_dict(trace_data)
        return snapshot

    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        log.info(f"Context snapshot saved: {self.snapshot_id}")

    @classmethod
    def load(cls, path: Path) -> 'ContextSnapshot':
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


@dataclass
class AgentDefinition:
    """Parsed agent definition from markdown file."""
    name: str
    model: str
    purpose: str
    context_requirements: List[str]
    output_format: str
    max_tokens: int
    temperature: float
    system_prompt: str
    quantization: str = "Q4_K_M"
    estimated_vram: str = "8GB"
    is_primary: bool = False
    tools: List[str] = field(default_factory=list)  # Tools this agent can use


class ContextCompressor:
    """
    Compress conversation context for agent handoffs.

    Two strategies:
    1. Observation masking: Replace old tool outputs with summaries (fast, preserves reasoning)
    2. LLM summarization: Full summarization for critical handoffs (slower, more complete)

    Supports both Claude Code (cloud) and Ollama (local) for summarization.
    """

    def __init__(self, ollama_url: str = "http://localhost:11434", use_claude: bool = True):
        self.ollama_url = ollama_url
        self.use_claude = use_claude and CLAUDE_AVAILABLE
        self._claude_bridge: Optional[ClaudeCodeBridge] = None
        if self.use_claude:
            try:
                self._claude_bridge = ClaudeCodeBridge(timeout_seconds=120)
            except Exception as e:
                log.warning(f"ContextCompressor: Claude Code not available, using Ollama: {e}")
                self.use_claude = False

    def observation_masking(
        self,
        messages: List[Dict],
        keep_recent: int = 10,
        max_tool_output_chars: int = 200
    ) -> List[Dict]:
        """
        Preserve reasoning, compress tool outputs.

        This is faster than LLM summarization and works well for most handoffs.
        Old tool outputs are replaced with brief summaries.
        """
        result = []
        total_messages = len(messages)

        for i, msg in enumerate(messages):
            # Keep recent messages intact
            if i >= total_messages - keep_recent:
                result.append(msg)
                continue

            # Compress old tool outputs
            if msg.get("role") == "tool":
                tool_name = msg.get("tool_name", "tool")
                content = msg.get("content", "")
                content_len = len(content)

                if content_len > max_tool_output_chars:
                    # Create summary placeholder
                    result.append({
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": f"[{tool_name} output: {content_len} chars, truncated]"
                    })
                else:
                    result.append(msg)
            else:
                # Keep non-tool messages (reasoning, user input)
                result.append(msg)

        log.debug(f"Observation masking: {len(messages)} -> {len(result)} messages")
        return result

    async def summarize_for_handoff(
        self,
        messages: List[Dict],
        model: str = "phi4:14b",
        max_context_chars: int = 50000  # phi4:14b has ~16K token context (~64K chars)
    ) -> str:
        """
        Full summarization for critical handoffs.

        Use when transferring complex multi-step task context to a new agent.

        Context Priority (never truncate):
        1. User query - the original request that drives the task
        2. System messages - agent instructions and capabilities
        3. Recent assistant reasoning and findings
        4. Tool results and research data
        """
        # Extract paramount context - user query and system messages
        user_query = None
        system_messages = []
        conversation_messages = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                system_messages.append(content)
            elif role == "user" and user_query is None:
                # First user message is the original query - paramount
                user_query = content
            else:
                conversation_messages.append({"role": role, "content": content})

        # Build structured context - NO truncation
        context_parts = []

        # 1. User Query (paramount - always include in full)
        if user_query:
            context_parts.append(f"=== ORIGINAL USER QUERY (PARAMOUNT) ===\n{user_query}\n")

        # 2. System Context (key agent instructions)
        if system_messages:
            system_context = "\n---\n".join(system_messages)
            context_parts.append(f"=== SYSTEM CONTEXT ===\n{system_context}\n")

        # 3. Full Conversation (no truncation - preserve all reasoning and findings)
        if conversation_messages:
            conv_parts = []
            for msg in conversation_messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                conv_parts.append(f"[{role}]: {content}")
            context_parts.append(f"=== CONVERSATION HISTORY ===\n" + "\n\n".join(conv_parts))

        full_context = "\n\n".join(context_parts)

        # If context fits in model window, use it directly
        # phi4:14b can handle ~50K chars comfortably with room for output
        if len(full_context) <= max_context_chars:
            conversation = full_context
        else:
            # Context exceeds limit - use LLM to intelligently compress
            # while preserving user query and system intent
            log.info(f"Context exceeds limit ({len(full_context)} > {max_context_chars}), using intelligent compression")
            conversation = await self._intelligent_context_compression(
                user_query=user_query,
                system_messages=system_messages,
                conversation_messages=conversation_messages,
                target_chars=max_context_chars
            )

        prompt = f"""Create a comprehensive handoff summary for another AI agent. This summary must enable the receiving agent to continue the task seamlessly.

CRITICAL REQUIREMENTS:
- Preserve the EXACT user query and intent - do not paraphrase or lose details
- Include ALL key findings, data, and research discovered so far
- Document decisions made and reasoning behind them
- List specific constraints, preferences, or requirements from the user
- Clearly state what remains to be done
- Include any URLs, file paths, code snippets, or technical details needed to continue

CONTEXT:
{conversation}

COMPREHENSIVE HANDOFF SUMMARY:"""

        try:
            if self.use_claude and self._claude_bridge:
                # === CLAUDE CODE PATH ===
                messages = [{"role": "user", "content": prompt}]
                result = await self._claude_bridge.query(
                    messages=messages,
                    disable_native_tools=True,
                )
                summary = result.get("content", "")
                log.info(f"Generated handoff summary (Claude): {len(summary)} chars")
                return summary
            else:
                # === OLLAMA PATH ===
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.ollama_url}/api/generate",
                        json={
                            "model": model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "num_predict": 2000,  # Allow rich, detailed handoff summaries
                                "temperature": 0.2    # Low temp for accuracy
                            }
                        },
                        timeout=aiohttp.ClientTimeout(total=180)  # Allow time for thorough processing
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            summary = data.get("response", "")
                            log.info(f"Generated handoff summary: {len(summary)} chars")
                            return summary
                        else:
                            log.warning(f"Summarization failed: {resp.status}")
                            # Fallback: return critical context directly
                            return f"USER QUERY: {user_query}\n\nCONTEXT: {conversation[:10000]}"
        except Exception as e:
            log.error(f"Summarization error: {e}")
            # Fallback: return critical context directly
            return f"USER QUERY: {user_query}\n\nCONTEXT: {conversation[:10000]}"

    async def _intelligent_context_compression(
        self,
        user_query: str,
        system_messages: List[str],
        conversation_messages: List[Dict],
        target_chars: int
    ) -> str:
        """
        Intelligently compress context while preserving paramount information.

        Uses LLM to summarize conversation middle while keeping:
        - Full user query (never compressed)
        - Key system instructions
        - Recent messages in full
        - Compressed summary of older messages
        """
        # User query is always preserved in full
        preserved_parts = [f"=== ORIGINAL USER QUERY (PARAMOUNT) ===\n{user_query}\n"]

        # Extract key points from system messages
        if system_messages:
            system_summary = system_messages[0][:2000] if system_messages else ""
            preserved_parts.append(f"=== SYSTEM CONTEXT ===\n{system_summary}\n")

        # Keep recent messages in full (last 5)
        recent_messages = conversation_messages[-5:] if len(conversation_messages) > 5 else conversation_messages
        older_messages = conversation_messages[:-5] if len(conversation_messages) > 5 else []

        # Compress older messages if they exist
        if older_messages:
            older_text = "\n".join([f"[{m['role']}]: {m['content']}" for m in older_messages])

            # Use LLM to compress older conversation
            compress_prompt = f"""Summarize this conversation history, preserving ALL:
- Key findings and data discovered
- Decisions made and their reasoning
- Important URLs, file paths, code
- User preferences and constraints

CONVERSATION TO SUMMARIZE:
{older_text}

DETAILED SUMMARY (preserve all important information):"""

            try:
                if self.use_claude and self._claude_bridge:
                    # === CLAUDE CODE PATH ===
                    messages = [{"role": "user", "content": compress_prompt}]
                    result = await self._claude_bridge.query(
                        messages=messages,
                        disable_native_tools=True,
                    )
                    compressed = result.get("content", "")
                    preserved_parts.append(f"=== EARLIER CONVERSATION (SUMMARIZED) ===\n{compressed}\n")
                else:
                    # === OLLAMA PATH ===
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{self.ollama_url}/api/generate",
                            json={
                                "model": "phi4:14b",
                                "prompt": compress_prompt,
                                "stream": False,
                                "options": {"num_predict": 1500, "temperature": 0.2}
                            },
                            timeout=aiohttp.ClientTimeout(total=120)
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                compressed = data.get("response", "")
                                preserved_parts.append(f"=== EARLIER CONVERSATION (SUMMARIZED) ===\n{compressed}\n")
            except Exception as e:
                log.error(f"Compression error: {e}")
                # Fallback: include what fits
                preserved_parts.append(f"=== EARLIER CONVERSATION ===\n{older_text[:5000]}...\n")

        # Add recent messages in full
        if recent_messages:
            recent_text = "\n\n".join([f"[{m['role']}]: {m['content']}" for m in recent_messages])
            preserved_parts.append(f"=== RECENT CONVERSATION ===\n{recent_text}")

        return "\n\n".join(preserved_parts)


class SubagentManager:
    """
    Manages subagent lifecycle with model swapping.

    Phase 4 Enhanced Flow:
    1. Primary agent calls spawn_subagent()
    2. Manager serializes context with compression
    3. Preload subagent model while unloading primary (parallel)
    4. Execute subagent with injected context
    5. Preload primary model while capturing output
    6. Return result with rehydrated context

    VRAM Strategy:
    - Router model (~2GB) can stay loaded alongside specialists
    - Only one specialist (~5GB) loaded at a time
    - Preloading hides cold-start latency (3-10s for 7B models)
    """

    def __init__(
        self,
        agents_dir: Path = None,
        ollama_url: str = None,
        snapshot_dir: Path = None,
        config: Config = None,
        skill_registry = None,
        use_claude: bool = True,  # Use Claude Code by default
        voice_mode: bool = False,  # Enable voice progress updates
        tts_callback: Optional[Callable[[str], Awaitable[None]]] = None,  # TTS function for voice
    ):
        self.config = config or Config()
        self.agents_dir = agents_dir or Path.home() / ".workshop" / "agents"
        self.ollama_url = ollama_url or self.config.OLLAMA_URL
        self.snapshot_dir = snapshot_dir or self.agents_dir / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Voice mode for progress updates
        self.voice_mode = voice_mode
        self.tts_callback = tts_callback

        # Claude Code integration
        # NOTE: Subagents use DEDICATED bridges (not shared) for isolated context windows
        # This prevents context pollution between specialist subagents and main agent
        self.use_claude = use_claude and CLAUDE_AVAILABLE
        self._claude_bridge: Optional[ClaudeCodeBridge] = None
        if self.use_claude:
            try:
                # Dedicated instance for subagent isolation (not using get_claude_bridge)
                self._claude_bridge = ClaudeCodeBridge(timeout_seconds=300)  # 5 min for complex subagent tasks
                log.info("Claude Code bridge initialized for subagent execution (isolated instance)")
            except Exception as e:
                log.warning(f"Failed to initialize Claude Code, falling back to Ollama: {e}")
                self.use_claude = False

        # Haiku progress manager for voice updates during subagent tool execution
        self._progress_manager: Optional[HaikuProgressManager] = None
        if self.voice_mode and HAIKU_AVAILABLE and self._claude_bridge:
            try:
                self._progress_manager = HaikuProgressManager(
                    claude_bridge=self._claude_bridge,
                    tts_callback=self.tts_callback,
                    voice_mode=self.voice_mode
                )
                log.info("[SUBAGENT_VOICE] Haiku progress manager initialized for subagent updates")
            except Exception as e:
                log.warning(f"[SUBAGENT_VOICE] Failed to initialize progress manager: {e}")

        # Cache for loaded agent definitions
        self._agents: Dict[str, AgentDefinition] = {}

        # Track current state (only used for Ollama mode)
        self.current_model: Optional[str] = None
        self.current_snapshot: Optional[ContextSnapshot] = None

        # Preloading state (only used for Ollama mode)
        self._preload_task: Optional[asyncio.Task] = None
        self._preloaded_model: Optional[str] = None

        # Context compressor for handoffs
        self.compressor = ContextCompressor(self.ollama_url, use_claude=self.use_claude)

        # Specialist models from config (used for Ollama mode)
        self.specialist_models = self.config.SPECIALIST_MODELS
        self.vram_estimates = self.config.MODEL_VRAM_ESTIMATES
        self.vram_budget = self.config.VRAM_BUDGET_GB - self.config.VRAM_HEADROOM_GB

        # Tool execution support
        self.skill_registry = skill_registry
        self._subagent_tools = self._build_subagent_tools()

    def _build_subagent_tools(self) -> Dict[str, callable]:
        """Build the set of tools available to subagents."""
        tools = {}

        # Web research tools
        async def web_search(query: str) -> str:
            """Search the web for information."""
            try:
                # Try to use the skill registry's web_search
                if self.skill_registry:
                    result = await self.skill_registry.execute("web_search", {"query": query})
                    return str(result)
                return f"Web search not available (no skill registry)"
            except Exception as e:
                return f"Web search error: {e}"

        async def fetch_url(url: str) -> str:
            """Fetch and extract content from a URL."""
            try:
                if self.skill_registry:
                    result = await self.skill_registry.execute("fetch_url", {"url": url})
                    if isinstance(result, dict):
                        if result.get("success"):
                            return result.get("content", "")[:20000]  # Truncate for subagent context
                        else:
                            return f"Fetch failed: {result.get('error', 'Unknown error')}"
                    return str(result)[:20000]
                return f"URL fetch not available (no skill registry)"
            except Exception as e:
                return f"Fetch error: {e}"

        # File reading tools
        async def read_file(path: str) -> str:
            """Read content from a local file."""
            try:
                file_path = Path(path).expanduser()
                if file_path.exists():
                    content = file_path.read_text()
                    return content[:30000]  # Truncate for context
                return f"File not found: {path}"
            except Exception as e:
                return f"Read error: {e}"

        async def glob_files(pattern: str) -> str:
            """Find files matching a pattern."""
            try:
                from glob import glob
                matches = glob(pattern, recursive=True)
                if matches:
                    return "\n".join(matches[:50])  # Limit results
                return f"No files match pattern: {pattern}"
            except Exception as e:
                return f"Glob error: {e}"

        async def grep_content(pattern: str, path: str = ".") -> str:
            """Search for text patterns in files."""
            try:
                import subprocess
                result = subprocess.run(
                    ["grep", "-r", "-l", "--include=*.md", "--include=*.txt", "--include=*.py", pattern, path],
                    capture_output=True, text=True, timeout=10
                )
                if result.stdout:
                    return result.stdout[:5000]
                return f"No matches for pattern: {pattern}"
            except Exception as e:
                return f"Grep error: {e}"

        # Research platform tools
        async def get_research_section(section: str) -> str:
            """Get a section from the research platform."""
            try:
                if self.skill_registry:
                    result = await self.skill_registry.execute("get_research_section", {"section": section})
                    return str(result)[:20000]
                return "Research platform not available"
            except Exception as e:
                return f"Research section error: {e}"

        async def show_research_platform() -> str:
            """Show the current research platform status."""
            try:
                if self.skill_registry:
                    result = await self.skill_registry.execute("show_research_platform", {})
                    return str(result)[:10000]
                return "Research platform not available"
            except Exception as e:
                return f"Research platform error: {e}"

        # Register all tools
        tools["web_search"] = web_search
        tools["fetch_url"] = fetch_url
        tools["read_file"] = read_file
        tools["glob_files"] = glob_files
        tools["grep_content"] = grep_content
        tools["get_research_section"] = get_research_section
        tools["show_research_platform"] = show_research_platform

        return tools

    async def _execute_subagent_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        trace_id: str = None
    ) -> str:
        """Execute a tool for a subagent and return the result.

        First tries local subagent tools, then falls back to skill_registry.
        This allows subagents to use any registered skill tool.
        """
        tool_start = time.time()
        log.debug(f"[SUBAGENT_TOOL_EXEC] Starting: {tool_name}({list(args.keys())})")

        # First, try skill_registry for comprehensive tool access
        if self.skill_registry and tool_name not in self._subagent_tools:
            try:
                result = await self.skill_registry.execute(tool_name, args)
                duration_ms = int((time.time() - tool_start) * 1000)
                result_str = str(result)

                await _emit_dashboard_event(
                    "tool_result",
                    trace_id=trace_id,
                    call_id=f"subagent_{tool_name}_{int(time.time()*1000)}",
                    result=result_str[:5000] if len(result_str) > 5000 else result_str,
                    duration_ms=duration_ms
                )

                log.info(f"[SUBAGENT_TOOL_EXEC] {tool_name} (via registry): {len(result_str)} chars in {duration_ms}ms")
                return result_str
            except ValueError:
                # Tool not in registry, continue to local tools
                pass
            except Exception as e:
                log.error(f"[SUBAGENT_TOOL_EXEC] Skill registry tool {tool_name} failed: {type(e).__name__}: {e}")
                return f"Tool error: {e}"

        # Fall back to local subagent tools
        if tool_name not in self._subagent_tools:
            log.warning(f"[SUBAGENT_TOOL_EXEC] Unknown tool: {tool_name}. Available: {list(self._subagent_tools.keys())}")
            return f"Unknown tool: {tool_name}. Available: {list(self._subagent_tools.keys())}"

        tool_func = self._subagent_tools[tool_name]
        log.debug(f"[SUBAGENT_TOOL_EXEC] Using local subagent tool: {tool_name}")

        try:
            # Execute the tool
            result = await tool_func(**args)
            duration_ms = int((time.time() - tool_start) * 1000)

            # Emit dashboard event for tool execution
            # Note: Full result is preserved in the return value below
            # Dashboard preview can be larger to show meaningful research content
            result_str = str(result)
            await _emit_dashboard_event(
                "tool_result",
                trace_id=trace_id,
                call_id=f"subagent_{tool_name}_{int(time.time()*1000)}",
                result=result_str[:5000] if len(result_str) > 5000 else result_str,
                duration_ms=duration_ms
            )

            log.info(f"[SUBAGENT_TOOL_EXEC] {tool_name} (local): {len(str(result))} chars in {duration_ms}ms")
            return str(result)

        except Exception as e:
            log.error(f"[SUBAGENT_TOOL_EXEC] Local tool {tool_name} failed: {type(e).__name__}: {e}")
            return f"Tool error: {e}"

    def load_agent_definition(self, agent_name: str) -> AgentDefinition:
        """Load and parse an agent definition from markdown file."""
        if agent_name in self._agents:
            return self._agents[agent_name]

        agent_path = self.agents_dir / f"{agent_name}.md"
        if not agent_path.exists():
            raise ValueError(f"Agent definition not found: {agent_path}")

        content = agent_path.read_text()

        # Parse YAML frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = yaml.safe_load(parts[1])
                system_prompt = parts[2].strip()
            else:
                raise ValueError(f"Invalid frontmatter in {agent_path}")
        else:
            raise ValueError(f"Agent definition must have YAML frontmatter: {agent_path}")

        agent = AgentDefinition(
            name=frontmatter.get("name", agent_name),
            model=frontmatter.get("model", "llama3-groq-tool-use:8b"),
            purpose=frontmatter.get("purpose", ""),
            context_requirements=frontmatter.get("context_requirements", []),
            output_format=frontmatter.get("output_format", "markdown"),
            max_tokens=frontmatter.get("max_tokens", 2000),
            temperature=frontmatter.get("temperature", 0.7),
            quantization=frontmatter.get("quantization", "Q4_K_M"),
            estimated_vram=frontmatter.get("estimated_vram", "8GB"),
            is_primary=frontmatter.get("is_primary", False),
            system_prompt=system_prompt,
            tools=frontmatter.get("tools", [])  # Load tools list from YAML
        )

        self._agents[agent_name] = agent
        log.info(f"Loaded agent definition: {agent_name} ({agent.model})")
        return agent

    def _build_tool_instructions(self, agent_def: AgentDefinition) -> str:
        """Build tool-calling format instructions for the subagent.

        This is CRITICAL: Claude needs explicit instructions on the <tool_call> XML format
        Workshop uses. Without this, Claude will just describe what it would do instead of
        actually calling tools.
        """
        # Get tool descriptions from the skill registry if available
        tool_descriptions = []
        agent_tools = agent_def.tools if agent_def.tools else []

        if self.skill_registry and agent_tools:
            for tool_name in agent_tools:
                tool_info = self.skill_registry.get_tool(tool_name)
                if tool_info:
                    # Include signature for clarity on parameters
                    tool_descriptions.append(f"- **{tool_name}**({tool_info.signature}): {tool_info.description}")
                else:
                    # Tool in agent definition but not in registry - still list it
                    tool_descriptions.append(f"- **{tool_name}**: (tool available)")

            log.debug(f"[SUBAGENT_TOOLS] Agent {agent_def.name} has {len(tool_descriptions)} tools: {agent_tools}")
        else:
            log.warning(f"[SUBAGENT_TOOLS] Agent {agent_def.name} has no tools defined or skill_registry not available")

        tools_list = "\n".join(tool_descriptions) if tool_descriptions else "No specific tools defined. Use available tools from the skill registry."

        return f"""
## CRITICAL: Tool Calling Instructions

**You MUST use tools to accomplish tasks. DO NOT just describe what you would do.**

To call a tool, output it in this EXACT format:

<tool_call>
{{"tool": "tool_name", "args": {{"param1": "value1", "param2": "value2"}}}}
</tool_call>

### Available Tools:
{tools_list}

### IMPORTANT RULES:
1. **Always call tools** - Don't say "I'll search for..." - actually call the tool!
2. **One tool per block** - Each tool call needs its own <tool_call> block
3. **Valid JSON** - Use double quotes for strings, proper JSON syntax
4. **Wait for results** - After calling a tool, you'll receive results to work with

### Example:
User: "Search for Python tutorials"

WRONG (just describing):
"I'll search for Python tutorials using web_search..."

CORRECT (actually calling):
<tool_call>
{{"tool": "web_search", "args": {{"query": "Python tutorials"}}}}
</tool_call>

After receiving results, then synthesize and respond.
"""

    async def get_loaded_models(self) -> List[Dict]:
        """Get list of currently loaded models from Ollama."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_url}/api/ps") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("models", [])
        except Exception as e:
            log.warning(f"Failed to get loaded models: {e}")
        return []

    async def unload_model(self, model_name: str) -> bool:
        """Unload a model from Ollama by setting keep_alive to 0."""
        log.info(f"Unloading model: {model_name}")
        try:
            async with aiohttp.ClientSession() as session:
                # Send a generate request with keep_alive=0 to unload
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": "",
                        "keep_alive": 0
                    }
                ) as resp:
                    if resp.status == 200:
                        log.info(f"Model unloaded: {model_name}")
                        return True
                    else:
                        log.warning(f"Unload returned status {resp.status}")
        except Exception as e:
            log.error(f"Failed to unload model: {e}")
        return False

    async def ensure_model_loaded(self, model_name: str) -> bool:
        """Ensure a specific model is loaded (triggers load if needed)."""
        # Check if already preloaded
        if self._preloaded_model == model_name and self._preload_task:
            try:
                await self._preload_task
                self._preload_task = None
                self._preloaded_model = None
                self.current_model = model_name
                log.info(f"Model ready (preloaded): {model_name}")
                return True
            except Exception as e:
                log.warning(f"Preload failed, loading normally: {e}")

        log.info(f"Ensuring model loaded: {model_name}")
        try:
            async with aiohttp.ClientSession() as session:
                # Simple generate request will load the model
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": "Hello",
                        "stream": False,
                        "options": {"num_predict": 1}
                    },
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status == 200:
                        log.info(f"Model ready: {model_name}")
                        self.current_model = model_name
                        return True
        except Exception as e:
            log.error(f"Failed to load model {model_name}: {e}")
        return False

    async def preload_model(self, model_name: str):
        """
        Start loading a model in the background (non-blocking).

        Call this before unloading the current model to hide cold-start latency.
        The model will be ready when ensure_model_loaded is called.
        """
        if self._preload_task and not self._preload_task.done():
            log.debug(f"Cancelling previous preload of {self._preloaded_model}")
            self._preload_task.cancel()

        log.info(f"Preloading model: {model_name}")
        self._preloaded_model = model_name

        async def _do_preload():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.ollama_url}/api/generate",
                        json={
                            "model": model_name,
                            "prompt": "Warming up",
                            "stream": False,
                            "keep_alive": -1,  # Keep loaded indefinitely
                            "options": {"num_predict": 1}
                        },
                        timeout=aiohttp.ClientTimeout(total=120)
                    ) as resp:
                        if resp.status == 200:
                            log.info(f"Preload complete: {model_name}")
                        else:
                            log.warning(f"Preload returned {resp.status}")
            except Exception as e:
                log.error(f"Preload failed: {e}")

        self._preload_task = asyncio.create_task(_do_preload())

    async def swap_models(
        self,
        from_model: str,
        to_model: str,
        parallel: bool = True
    ) -> bool:
        """
        Swap from one model to another with optional preloading.

        If parallel=True, starts loading the new model before fully unloading
        the old one. This can reduce perceived latency but requires more VRAM.
        """
        swap_start = time.time()

        if parallel and self._can_coexist(from_model, to_model):
            # Start loading new model first
            await self.preload_model(to_model)
            # Small delay to let preload start
            await asyncio.sleep(0.5)
            # Then unload old model
            await self.unload_model(from_model)
            # Wait for preload to complete
            success = await self.ensure_model_loaded(to_model)
        else:
            # Sequential: unload first, then load
            await self.unload_model(from_model)
            await asyncio.sleep(0.5)  # Allow VRAM release
            success = await self.ensure_model_loaded(to_model)

        swap_time = int((time.time() - swap_start) * 1000)
        log.info(f"Model swap {from_model} -> {to_model}: {swap_time}ms")

        return success

    def _can_coexist(self, model_a: str, model_b: str) -> bool:
        """Check if two models can coexist in VRAM during swap."""
        vram_a = self.vram_estimates.get(model_a, 5.0)
        vram_b = self.vram_estimates.get(model_b, 5.0)

        # Allow coexistence if total fits in budget
        # (with some margin for the KV cache overlap)
        can_fit = (vram_a + vram_b) <= (self.vram_budget + 1.0)

        log.debug(f"Can coexist check: {model_a}({vram_a}GB) + {model_b}({vram_b}GB) "
                  f"<= {self.vram_budget}GB = {can_fit}")

        return can_fit

    def get_specialist_model(self, role: str) -> str:
        """Get the model name for a specialist role."""
        return self.specialist_models.get(role, self.specialist_models.get("primary"))

    def create_snapshot(
        self,
        primary_model: str,
        conversation_summary: str,
        last_user_message: str,
        pending_task: str,
        telos_context: Dict = None,
        research_context: Dict = None,
        subagent_task: Dict = None
    ) -> ContextSnapshot:
        """Create a context snapshot for preservation during swap."""
        snapshot = ContextSnapshot(
            primary_model=primary_model,
            conversation_summary=conversation_summary,
            last_user_message=last_user_message,
            pending_task=pending_task
        )

        if telos_context:
            snapshot.telos_profile_summary = telos_context.get("profile_summary", "")
            snapshot.telos_active_project = telos_context.get("active_project", "")
            snapshot.telos_current_goals = telos_context.get("goals", [])

        if research_context:
            snapshot.research_platform_path = research_context.get("platform_path", "")
            snapshot.research_topic = research_context.get("topic", "")
            snapshot.research_source_count = research_context.get("source_count", 0)
            snapshot.research_key_findings = research_context.get("key_findings", "")

        if subagent_task:
            snapshot.subagent_name = subagent_task.get("agent_name", "")
            snapshot.subagent_model = subagent_task.get("model", "")
            snapshot.task_description = subagent_task.get("task", "")
            snapshot.input_data = subagent_task.get("input_data", {})

        # Save to disk
        snapshot_path = self.snapshot_dir / f"{snapshot.snapshot_id}.json"
        snapshot.save(snapshot_path)
        self.current_snapshot = snapshot

        return snapshot

    async def spawn_subagent(
        self,
        agent_name: str,
        task: str,
        input_data: Dict[str, Any],
        primary_model: str = "llama3-groq-tool-use:8b",
        conversation_context: Dict = None,
        telos_context: Dict = None,
        research_context: Dict = None,
        trace_id: str = None
    ) -> Dict[str, Any]:
        """
        Spawn a subagent, execute task, and return to primary agent.

        This is the main entry point for subagent execution.

        Args:
            agent_name: Name of the subagent to spawn
            task: Task description for the subagent
            input_data: Data to pass to the subagent
            primary_model: Model to return to after subagent completes
            conversation_context: Summary of conversation state
            telos_context: User profile, goals, project context
            research_context: Active research data
            trace_id: Parent trace ID for correlating dashboard events

        Returns:
            Dict with 'success', 'output', 'agent', 'duration_ms', 'llm_call_data'
        """
        import time
        start_time = time.time()

        log.info(f"=== SPAWNING SUBAGENT: {agent_name} ===")
        log.info(f"[SUBAGENT_DISPATCH] Task: {task[:150]}...")
        log.info(f"[SUBAGENT_DISPATCH] Backend: {'Claude Code' if self.use_claude else 'Ollama'}")
        log.debug(f"[SUBAGENT_DISPATCH] Bridge available: {self._claude_bridge is not None}")
        log.debug(f"[SUBAGENT_DISPATCH] Skill registry: {'yes' if self.skill_registry else 'no'}")
        if trace_id:
            log.debug(f"[SUBAGENT_DISPATCH] Parent trace: {trace_id}")

        # Load agent definition
        try:
            agent_def = self.load_agent_definition(agent_name)
        except ValueError as e:
            return {"success": False, "error": str(e), "output": None}

        # Determine effective model name for telemetry
        # When using Claude Code, we report "claude-code" instead of the Ollama model
        effective_model_for_spawn = "claude-code" if self.use_claude else agent_def.model

        # Emit dashboard event for subagent spawning
        await _emit_dashboard_event(
            "subagent_spawning",
            trace_id=trace_id,
            agent_name=agent_name,
            model=effective_model_for_spawn,
            task=task[:200]
        )

        # Create context snapshot
        snapshot = self.create_snapshot(
            primary_model=primary_model,
            conversation_summary=conversation_context.get("summary", "") if conversation_context else "",
            last_user_message=conversation_context.get("last_message", "") if conversation_context else "",
            pending_task=task,
            telos_context=telos_context,
            research_context=research_context,
            subagent_task={
                "agent_name": agent_name,
                "model": agent_def.model,
                "task": task,
                "input_data": input_data
            }
        )

        log.info(f"Context snapshot created: {snapshot.snapshot_id}")

        # Emit context snapshot event
        await _emit_dashboard_event(
            "context_snapshot",
            trace_id=trace_id,
            snapshot_id=snapshot.snapshot_id,
            research_topic=snapshot.research_topic
        )

        # Determine the effective model name for telemetry
        effective_model = "claude-code" if self.use_claude else agent_def.model

        # VRAM management only needed for Ollama mode
        if not self.use_claude:
            # Unload primary model
            if self.current_model and self.current_model != agent_def.model:
                await _emit_dashboard_event(
                    "subagent_model_swap",
                    trace_id=trace_id,
                    from_model=self.current_model,
                    to_model=agent_def.model,
                    action="unloading"
                )
                await self.unload_model(self.current_model)

            # Load subagent model
            await _emit_dashboard_event(
                "subagent_model_swap",
                trace_id=trace_id,
                from_model=primary_model,
                to_model=agent_def.model,
                action="loading"
            )
            model_loaded = await self.ensure_model_loaded(agent_def.model)
            if not model_loaded:
                return {
                    "success": False,
                    "error": f"Failed to load model: {agent_def.model}",
                    "output": None
                }

        # Execute subagent
        await _emit_dashboard_event(
            "subagent_executing",
            trace_id=trace_id,
            agent_name=agent_name,
            model=effective_model
        )
        llm_call_data = None
        subagent_trace = None
        snapshot.status = "running"

        try:
            output, llm_call_data, subagent_trace = await self._execute_subagent(
                agent_def, task, input_data, snapshot, trace_id=trace_id
            )
        except Exception as e:
            log.error(f"Subagent execution failed: {e}")
            output = None
            snapshot.status = "failed"

        # VRAM cleanup only needed for Ollama mode
        if not self.use_claude:
            # Unload subagent model
            await self.unload_model(agent_def.model)

            # Reload primary model
            await _emit_dashboard_event(
                "subagent_model_swap",
                trace_id=trace_id,
                from_model=agent_def.model,
                to_model=primary_model,
                action="loading"
            )
            primary_loaded = await self.ensure_model_loaded(primary_model)
            if not primary_loaded:
                log.error(f"Failed to reload primary model: {primary_model}")

        duration_ms = int((time.time() - start_time) * 1000)
        log.info(f"=== SUBAGENT COMPLETE: {agent_name} ({duration_ms}ms) ===")

        # Update snapshot with trace and status
        snapshot.trace = subagent_trace
        snapshot.output = output or ""
        snapshot.duration_ms = duration_ms
        snapshot.status = "completed" if output else "failed"

        # Re-save snapshot with trace data
        snapshot_path = self.snapshot_dir / f"{snapshot.snapshot_id}.json"
        snapshot.save(snapshot_path)
        log.info(f"Snapshot updated with trace: {snapshot.snapshot_id}")

        # Emit completion event
        await _emit_dashboard_event(
            "subagent_complete",
            trace_id=trace_id,
            agent_name=agent_name,
            model=effective_model,
            duration_ms=duration_ms,
            output_length=len(output) if output else 0,
            success=output is not None
        )

        return {
            "success": output is not None,
            "output": output,
            "agent": agent_name,
            "model": effective_model,  # Report actual model used (claude-code or ollama model)
            "snapshot_id": snapshot.snapshot_id,
            "duration_ms": duration_ms,
            "llm_call_data": llm_call_data  # Include LLM call info for telemetry
        }

    async def _call_subagent_llm(
        self,
        messages: List[Dict],
        agent_def: AgentDefinition,
    ) -> Tuple[str, int, int, int, List]:
        """
        Call the LLM (Claude Code or Ollama) and return the response.

        Returns:
            Tuple of (content, duration_ms, prompt_tokens, completion_tokens, bridge_tool_calls)
            where bridge_tool_calls are pre-extracted tool calls from Claude bridge (empty for Ollama)
        """
        llm_start_time = time.time()

        if self.use_claude and self._claude_bridge:
            # === CLAUDE CODE PATH ===
            log.debug(f"[SUBAGENT_LLM] Calling Claude Code (messages={len(messages)}, agent={agent_def.name})")
            try:
                result = await self._claude_bridge.query(
                    messages=messages,
                    disable_native_tools=True,  # Workshop handles tools via <tool_call>
                )
                content = result.get("content", "")
                # IMPORTANT: Claude bridge extracts tool_calls BEFORE cleaning the content
                # The cleaned content has <tool_call> tags removed, so we MUST pass these through
                bridge_tool_calls = result.get("tool_calls", [])
                duration_ms = int((time.time() - llm_start_time) * 1000)
                log.debug(f"[SUBAGENT_LLM] Claude Code response: {len(content)} chars in {duration_ms}ms")
                if bridge_tool_calls:
                    log.info(f"[SUBAGENT_LLM] Claude bridge extracted {len(bridge_tool_calls)} tool calls: {[tc.get('tool') for tc in bridge_tool_calls]}")
                # Claude doesn't return token counts in the same way
                return content, duration_ms, 0, 0, bridge_tool_calls

            except Exception as e:
                log.error(f"[SUBAGENT_LLM] Claude Code call failed: {type(e).__name__}: {e}")
                raise RuntimeError(f"Claude Code error: {e}")
        else:
            # === OLLAMA PATH ===
            log.debug(f"[SUBAGENT_LLM] Calling Ollama (model={agent_def.model}, messages={len(messages)})")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": agent_def.model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "num_predict": agent_def.max_tokens,
                            "temperature": agent_def.temperature
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as resp:
                    duration_ms = int((time.time() - llm_start_time) * 1000)

                    if resp.status != 200:
                        error = await resp.text()
                        log.error(f"[SUBAGENT_LLM] Ollama error {resp.status}: {error[:200]}")
                        raise RuntimeError(f"Ollama error {resp.status}: {error}")

                    data = await resp.json()
                    content = data.get("message", {}).get("content", "")
                    prompt_tokens = data.get("prompt_eval_count", 0)
                    completion_tokens = data.get("eval_count", 0)

                    log.debug(f"[SUBAGENT_LLM] Ollama response: {len(content)} chars, {prompt_tokens}+{completion_tokens} tokens in {duration_ms}ms")
                    # Ollama doesn't pre-extract tool calls, return empty list
                    return content, duration_ms, prompt_tokens, completion_tokens, []

    async def _execute_subagent(
        self,
        agent_def: AgentDefinition,
        task: str,
        input_data: Dict[str, Any],
        snapshot: ContextSnapshot,
        trace_id: str = None,
        max_tool_iterations: int = 50  # Increased from 10 - let subagents finish their work
    ) -> Tuple[Optional[str], Optional[Dict], Optional[SubagentTrace]]:
        """Execute the subagent with its specialized prompt and tool support.

        Implements a tool execution loop:
        1. Call LLM (Claude Code or Ollama based on configuration)
        2. Parse any <tool_call> blocks
        3. Execute tools and feed results back
        4. Repeat until no more tool calls or max iterations

        Returns:
            Tuple of (output_text, llm_call_data, subagent_trace) where:
            - llm_call_data contains timing and token info for telemetry
            - subagent_trace contains the full execution trace for dashboard
        """

        # Create trace to record all events
        subagent_trace = SubagentTrace(
            parent_trace_id=trace_id,
            started_at=datetime.now().isoformat()
        )
        execution_start = time.time()

        # Determine model name for telemetry
        model_name = "claude-code" if self.use_claude else agent_def.model

        # Build context-aware prompt
        context_parts = []

        # Add research context if available
        if snapshot.research_topic:
            context_parts.append(f"## Research Context")
            context_parts.append(f"Topic: {snapshot.research_topic}")
            context_parts.append(f"Sources: {snapshot.research_source_count}")
            if snapshot.research_key_findings:
                context_parts.append(f"Key Findings:\n{snapshot.research_key_findings}")
            context_parts.append("")

        # Add Telos context if available
        if snapshot.telos_active_project:
            context_parts.append(f"## User Context")
            context_parts.append(f"Active Project: {snapshot.telos_active_project}")
            if snapshot.telos_profile_summary:
                context_parts.append(f"User Profile: {snapshot.telos_profile_summary}")
            context_parts.append("")

        # Build the user message with task and data
        user_message_parts = [
            f"## Task\n{task}",
            ""
        ]

        if input_data:
            user_message_parts.append("## Input Data")
            user_message_parts.append(json.dumps(input_data, indent=2, default=str))

        context_str = "\n".join(context_parts)
        user_message = "\n".join(user_message_parts)

        # Build enhanced system prompt with tool-calling instructions
        # CRITICAL: Without these, Claude will describe actions instead of calling tools
        tool_instructions = self._build_tool_instructions(agent_def)
        enhanced_system_prompt = f"{agent_def.system_prompt}\n\n{tool_instructions}"

        # Record context in trace
        subagent_trace.system_prompt = enhanced_system_prompt
        subagent_trace.context_injected = context_str
        subagent_trace.user_message = user_message

        log.debug(f"[SUBAGENT_EXEC] System prompt enhanced with tool instructions ({len(tool_instructions)} chars)")

        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": enhanced_system_prompt},
        ]

        if context_str:
            messages.append({"role": "user", "content": f"Context:\n{context_str}"})
            messages.append({"role": "assistant", "content": "I understand the context. What would you like me to do?"})

        messages.append({"role": "user", "content": user_message})

        # Aggregated LLM call data
        total_llm_duration_ms = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tool_calls = 0
        llm_call_count = 0

        log.info(f"[SUBAGENT_EXEC] Starting execution loop (max_iterations={max_tool_iterations})")
        log.debug(f"[SUBAGENT_EXEC] Context: research_topic={snapshot.research_topic[:50] if snapshot.research_topic else 'none'}, telos_project={snapshot.telos_active_project or 'none'}")

        # Start voice progress session for this subagent
        if self._progress_manager:
            try:
                self._progress_manager.start_session(
                    query=task,
                    skill_name=f"subagent:{agent_def.name}",
                    task=task[:200]
                )
                log.debug(f"[SUBAGENT_VOICE] Progress session started for {agent_def.name}")
            except Exception as e:
                log.debug(f"[SUBAGENT_VOICE] Failed to start progress session: {e}")

        # Tool execution loop
        for iteration in range(max_tool_iterations):
            log.info(f"[SUBAGENT_EXEC] Iteration {iteration + 1}/{max_tool_iterations} (model={model_name})")

            # === TRACE: Log context being sent to LLM ===
            sys_prompt = messages[0].get("content") if messages and messages[0].get("role") == "system" else None
            _trace_context_sizes(messages, sys_prompt, iteration + 1)
            _trace_messages_content(messages, f"SUBAGENT:{agent_def.name}:ITER{iteration + 1}")

            # Emit LLM calling event for dashboard visibility with full message context
            await _emit_dashboard_event(
                "llm_calling",
                trace_id=trace_id,
                model=model_name,
                message_count=len(messages),
                messages=messages,
                system_prompt=sys_prompt
            )

            llm_call_id = f"llm_{iteration}_{int(time.time() * 1000)}"

            try:
                # Call LLM (Claude Code or Ollama)
                # bridge_tool_calls contains pre-extracted tool calls from Claude bridge (empty for Ollama)
                output, llm_duration_ms, prompt_tokens, completion_tokens, bridge_tool_calls = await self._call_subagent_llm(
                    messages, agent_def
                )

                total_llm_duration_ms += llm_duration_ms
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                llm_call_count += 1

                # === TRACE: Log LLM response ===
                log.info(f"Subagent response: {len(output)} chars in {llm_duration_ms}ms")
                _trace_log(f"CONTEXT_OUT:{agent_def.name}:ITER{iteration + 1}", output)

                # Use pre-extracted tool calls from Claude bridge if available
                # The bridge extracts tool_calls BEFORE cleaning the content, so we must use them
                # Fallback to parsing from content for Ollama or if bridge didn't extract any
                if bridge_tool_calls:
                    # Convert bridge format {"tool": "name", "args": {...}} to internal format
                    tool_calls = bridge_tool_calls
                    log.info(f"[SUBAGENT_EXEC] Using {len(tool_calls)} pre-extracted tool calls from Claude bridge")
                else:
                    # Fallback: parse from content (for Ollama)
                    tool_calls = _parse_tool_calls(output)

                # Record LLM call in trace
                subagent_trace.llm_calls.append(SubagentLLMCall(
                    call_id=llm_call_id,
                    iteration=iteration,
                    model=model_name,
                    messages_count=len(messages),
                    response_length=len(output),
                    duration_ms=llm_duration_ms,
                    timestamp=datetime.now().isoformat(),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    tool_calls_found=len(tool_calls),
                    response_preview=output[:500] if output else ""
                ))

                # Emit LLM complete event
                await _emit_dashboard_event(
                    "llm_complete",
                    trace_id=trace_id,
                    response_length=len(output),
                    duration_ms=llm_duration_ms,
                    tool_calls=len(tool_calls)
                )

                # If no tool calls, we're done - return the final response
                if not tool_calls:
                    log.info(f"Subagent complete after {iteration + 1} iterations, {total_tool_calls} tool calls")

                    # End voice progress session and log summary
                    if self._progress_manager:
                        try:
                            summary = self._progress_manager.get_session_summary()
                            log.info(f"[SUBAGENT_VOICE] Progress session complete: {summary['tools_executed']} tools, {summary['updates_generated']} updates, {summary['duration_seconds']}s")
                        except Exception as e:
                            log.debug(f"[SUBAGENT_VOICE] Failed to get session summary: {e}")

                    # Clean tool call artifacts from final output
                    final_output = _clean_tool_calls_from_response(output)

                    llm_call_data = {
                        "model": model_name,
                        "duration_ms": total_llm_duration_ms,
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": total_completion_tokens,
                        "response_length": len(final_output),
                        "message_count": len(messages),
                        "subagent_name": agent_def.name,
                        "tool_calls": total_tool_calls,
                        "llm_calls": llm_call_count
                    }

                    # Finalize trace
                    subagent_trace.final_output = final_output
                    subagent_trace.completed_at = datetime.now().isoformat()
                    subagent_trace.total_duration_ms = int((time.time() - execution_start) * 1000)
                    subagent_trace.llm_call_count = llm_call_count
                    subagent_trace.tool_call_count = total_tool_calls
                    subagent_trace.total_tokens = total_prompt_tokens + total_completion_tokens

                    return final_output, llm_call_data, subagent_trace

                # Execute tool calls
                log.info(f"Subagent requested {len(tool_calls)} tool(s): {[tc['tool'] for tc in tool_calls]}")
                total_tool_calls += len(tool_calls)

                tool_results = []
                for tc in tool_calls:
                    tool_name = tc["tool"]
                    tool_args = tc["args"]
                    tool_start = time.time()
                    tool_call_id = f"tool_{tool_name}_{int(tool_start * 1000)}"

                    # Emit tool calling event
                    await _emit_dashboard_event(
                        "tool_calling",
                        trace_id=trace_id,
                        tool_name=tool_name,
                        skill_name="subagent",
                        args=tool_args
                    )

                    # Voice progress: notify tool start
                    if self._progress_manager:
                        try:
                            await self._progress_manager.on_tool_start(tool_name, tool_args)
                        except Exception as e:
                            log.debug(f"[SUBAGENT_VOICE] Progress on_tool_start failed: {e}")

                    # Execute the tool
                    result = await self._execute_subagent_tool(tool_name, tool_args, trace_id)
                    tool_duration_ms = int((time.time() - tool_start) * 1000)

                    # === TRACE: Log tool execution details ===
                    _trace_tool_execution(tool_name, tool_args, result or "", tool_duration_ms)

                    # Voice progress: notify tool complete (may generate spoken update)
                    tool_error = result if result and "error" in result.lower() else None
                    if self._progress_manager:
                        try:
                            await self._progress_manager.on_tool_complete(
                                tool_name, result, tool_duration_ms, error=tool_error
                            )
                        except Exception as e:
                            log.debug(f"[SUBAGENT_VOICE] Progress on_tool_complete failed: {e}")

                    # Record tool call in trace
                    subagent_trace.tool_calls.append(SubagentToolCall(
                        call_id=tool_call_id,
                        tool_name=tool_name,
                        args=tool_args,
                        result=result[:5000] if result else "",  # Truncate large results
                        duration_ms=tool_duration_ms,
                        timestamp=datetime.now().isoformat(),
                        success="error" not in result.lower() if result else False,
                        error=tool_error
                    ))

                    tool_results.append({
                        "tool": tool_name,
                        "result": result
                    })

                # Add assistant message and tool results to conversation
                messages.append({"role": "assistant", "content": output})

                # Format tool results for the next message
                results_text = "\n\n".join([
                    f"**{tr['tool']}** result:\n{tr['result']}"
                    for tr in tool_results
                ])
                messages.append({
                    "role": "user",
                    "content": f"Tool execution results:\n\n{results_text}\n\nUse these results to continue your research. If you need more information, call more tools. When you have enough information, provide your final synthesis WITHOUT any tool calls."
                })

            except Exception as e:
                log.error(f"Subagent iteration failed: {e}")
                # Record failed LLM call in trace
                subagent_trace.llm_calls.append(SubagentLLMCall(
                    call_id=llm_call_id,
                    iteration=iteration,
                    model=model_name,
                    messages_count=len(messages),
                    response_length=0,
                    duration_ms=0,
                    timestamp=datetime.now().isoformat(),
                    response_preview=f"Error: {str(e)[:200]}"
                ))
                subagent_trace.success = False
                subagent_trace.error_message = str(e)
                subagent_trace.completed_at = datetime.now().isoformat()
                subagent_trace.total_duration_ms = int((time.time() - execution_start) * 1000)
                return None, None, subagent_trace

        # Max iterations reached
        log.warning(f"Subagent reached max iterations ({max_tool_iterations})")
        llm_call_data = {
            "model": model_name,
            "duration_ms": total_llm_duration_ms,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "response_length": 0,
            "message_count": len(messages),
            "subagent_name": agent_def.name,
            "tool_calls": total_tool_calls,
            "llm_calls": llm_call_count
        }

        # Finalize trace with max iterations warning
        subagent_trace.final_output = "Research incomplete - maximum iterations reached."
        subagent_trace.completed_at = datetime.now().isoformat()
        subagent_trace.total_duration_ms = int((time.time() - execution_start) * 1000)
        subagent_trace.llm_call_count = llm_call_count
        subagent_trace.tool_call_count = total_tool_calls
        subagent_trace.total_tokens = total_prompt_tokens + total_completion_tokens
        subagent_trace.error_message = "Max iterations reached"

        return "Research incomplete - maximum iterations reached.", llm_call_data, subagent_trace

    # =========================================================================
    # Parallel Dispatch Methods (Phase 3)
    # =========================================================================

    def spawn_parallel(
        self,
        tasks: List[Dict[str, Any]],
        max_workers: int = 4,
        trace_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        Spawn multiple subagents in parallel using ThreadPoolExecutor.

        This is the synchronous version - use spawn_parallel_async() for
        async contexts. Dramatically reduces latency when multiple independent
        tasks need to be executed (e.g., researching 5 topics).

        Args:
            tasks: List of task dicts, each with:
                - agent: agent name (e.g., "web-researcher", "coder")
                - prompt: the task prompt
                - input_data: optional data dict for the agent
                - context: optional context dict
            max_workers: Maximum concurrent subagents (default: 4)
            trace_id: Parent trace ID for correlating dashboard events

        Returns:
            List of results in same order as tasks, each with:
                - success: bool
                - result: output from subagent (if success)
                - error: error message (if failed)
                - agent: agent name
                - duration_ms: execution time

        Example:
            results = manager.spawn_parallel([
                {"agent": "web-researcher", "prompt": "Research INA219 accuracy"},
                {"agent": "web-researcher", "prompt": "Research ADS1115 specs"},
                {"agent": "coder", "prompt": "Write voltage calculation function"}
            ])
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        log.info(f"=== PARALLEL DISPATCH: {len(tasks)} tasks, {max_workers} workers ===")
        for i, task in enumerate(tasks):
            log.info(f"  Task {i+1}: [{task.get('agent', 'unknown')}] {task.get('prompt', '')[:60]}...")

        results = [None] * len(tasks)
        start_time = time.time()

        def run_subagent_sync(idx: int, task: Dict) -> Tuple[int, Dict]:
            """Run a single subagent synchronously and return (index, result)."""
            task_start = time.time()
            agent_name = task.get("agent", "web-researcher")
            prompt = task.get("prompt", "")
            input_data = task.get("input_data", {})
            context = task.get("context", {})

            log.info(f"[Worker {idx}] Starting: {agent_name} - {prompt[:50]}...")

            try:
                # Create event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    result = loop.run_until_complete(
                        self.spawn_subagent(
                            agent_name=agent_name,
                            task=prompt,
                            input_data=input_data,
                            conversation_context=context,
                            trace_id=trace_id
                        )
                    )

                    duration_ms = int((time.time() - task_start) * 1000)
                    log.info(f"[Worker {idx}] Complete: {agent_name} ({duration_ms}ms)")

                    return (idx, {
                        "success": result.get("success", False),
                        "result": result.get("output", ""),
                        "agent": agent_name,
                        "duration_ms": duration_ms,
                        "snapshot_id": result.get("snapshot_id"),
                        "model": result.get("model")
                    })

                finally:
                    loop.close()

            except Exception as e:
                duration_ms = int((time.time() - task_start) * 1000)
                log.error(f"[Worker {idx}] Failed: {agent_name} - {e}")
                return (idx, {
                    "success": False,
                    "error": str(e),
                    "agent": agent_name,
                    "duration_ms": duration_ms
                })

        # Execute tasks in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_subagent_sync, i, task): i
                for i, task in enumerate(tasks)
            }

            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        total_duration = int((time.time() - start_time) * 1000)
        success_count = sum(1 for r in results if r and r.get("success"))

        log.info(f"=== PARALLEL DISPATCH COMPLETE: {success_count}/{len(tasks)} succeeded ({total_duration}ms) ===")

        return results

    async def spawn_parallel_async(
        self,
        tasks: List[Dict[str, Any]],
        max_concurrent: int = 4,
        trace_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        Spawn multiple subagents in parallel using asyncio.

        This is the async version for use in async contexts. Uses a semaphore
        to limit concurrency and prevent overwhelming the Claude Code CLI.

        Args:
            tasks: List of task dicts, each with:
                - agent: agent name (e.g., "web-researcher", "coder")
                - prompt: the task prompt
                - input_data: optional data dict for the agent
                - context: optional context dict
            max_concurrent: Maximum concurrent subagents (default: 4)
            trace_id: Parent trace ID for correlating dashboard events

        Returns:
            List of results in same order as tasks, each with:
                - success: bool
                - result: output from subagent (if success)
                - error: error message (if failed)
                - agent: agent name
                - duration_ms: execution time

        Example:
            results = await manager.spawn_parallel_async([
                {"agent": "web-researcher", "prompt": "Research INA219 accuracy"},
                {"agent": "web-researcher", "prompt": "Research ADS1115 specs"},
            ], max_concurrent=2)
        """
        log.info(f"=== PARALLEL ASYNC DISPATCH: {len(tasks)} tasks, {max_concurrent} concurrent ===")
        for i, task in enumerate(tasks):
            log.info(f"  Task {i+1}: [{task.get('agent', 'unknown')}] {task.get('prompt', '')[:60]}...")

        semaphore = asyncio.Semaphore(max_concurrent)
        start_time = time.time()

        async def run_with_semaphore(idx: int, task: Dict) -> Dict:
            """Run a single subagent with semaphore control."""
            async with semaphore:
                task_start = time.time()
                agent_name = task.get("agent", "web-researcher")
                prompt = task.get("prompt", "")
                input_data = task.get("input_data", {})
                context = task.get("context", {})

                log.info(f"[Async {idx}] Starting: {agent_name} - {prompt[:50]}...")

                try:
                    result = await self.spawn_subagent(
                        agent_name=agent_name,
                        task=prompt,
                        input_data=input_data,
                        conversation_context=context,
                        trace_id=trace_id
                    )

                    duration_ms = int((time.time() - task_start) * 1000)
                    log.info(f"[Async {idx}] Complete: {agent_name} ({duration_ms}ms)")

                    return {
                        "success": result.get("success", False),
                        "result": result.get("output", ""),
                        "agent": agent_name,
                        "duration_ms": duration_ms,
                        "snapshot_id": result.get("snapshot_id"),
                        "model": result.get("model"),
                        "index": idx
                    }

                except Exception as e:
                    duration_ms = int((time.time() - task_start) * 1000)
                    log.error(f"[Async {idx}] Failed: {agent_name} - {e}")
                    return {
                        "success": False,
                        "error": str(e),
                        "agent": agent_name,
                        "duration_ms": duration_ms,
                        "index": idx
                    }

        # Launch all tasks with semaphore control
        coros = [run_with_semaphore(i, task) for i, task in enumerate(tasks)]
        results_unordered = await asyncio.gather(*coros, return_exceptions=True)

        # Handle any exceptions that slipped through
        results = []
        for i, r in enumerate(results_unordered):
            if isinstance(r, Exception):
                results.append({
                    "success": False,
                    "error": str(r),
                    "agent": tasks[i].get("agent", "unknown"),
                    "duration_ms": 0,
                    "index": i
                })
            else:
                results.append(r)

        # Sort by original index to maintain order
        results.sort(key=lambda x: x.get("index", 0))

        # Remove index from results
        for r in results:
            r.pop("index", None)

        total_duration = int((time.time() - start_time) * 1000)
        success_count = sum(1 for r in results if r.get("success"))

        log.info(f"=== PARALLEL ASYNC COMPLETE: {success_count}/{len(tasks)} succeeded ({total_duration}ms) ===")

        return results

    def get_last_snapshot(self) -> Optional[ContextSnapshot]:
        """Get the most recent context snapshot."""
        snapshots = sorted(self.snapshot_dir.glob("*.json"), reverse=True)
        if snapshots:
            return ContextSnapshot.load(snapshots[0])
        return None


# Convenience function for spawning from agent
async def spawn_research_summarizer(
    research_platform: Dict,
    task: str,
    focus_areas: List[str] = None,
    primary_model: str = "llama3-groq-tool-use:8b",
    trace_id: str = None
) -> Dict[str, Any]:
    """
    Convenience function to spawn the research-summarizer subagent.

    Args:
        research_platform: The research platform dict (from ResearchPlatform.to_dict())
        task: What to summarize/analyze
        focus_areas: Specific areas to focus on
        primary_model: Model to return to after summarization
        trace_id: Parent trace ID for correlating dashboard events

    Returns:
        Subagent result with synthesized summary
    """
    manager = SubagentManager()

    # Extract key findings for context
    key_findings = []
    for source in research_platform.get("sources", [])[:5]:
        key_findings.extend(source.get("key_points", [])[:2])

    return await manager.spawn_subagent(
        agent_name="research-summarizer",
        task=task,
        input_data={
            "research_platform": research_platform,
            "focus_areas": focus_areas or []
        },
        primary_model=primary_model,
        research_context={
            "platform_path": "",  # Will be set if available
            "topic": research_platform.get("topic", ""),
            "source_count": len(research_platform.get("sources", [])),
            "key_findings": "\n".join(f"- {p}" for p in key_findings[:10])
        },
        trace_id=trace_id
    )
