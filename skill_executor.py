"""
Skill Executor - Execute skills using Fabric-style prompts with constrained tools.

This is Stage 2 of the two-stage routing system. After the router determines
which skill to use, the executor:
1. Loads the skill's system.md (Fabric-style prompt)
2. Constrains tool availability to ONLY that skill's tools
3. Lets the LLM extract arguments and call tools

Key insight: The LLM is much better at extracting "Daniel Miessler PAI" from
"research this PAI concept Im hearing about from Daniel Miessler" than regex.
By constraining to 3-5 tools instead of 100+, we get reliable tool selection too.

IMPORTANT: Uses NON-NATIVE tool calling for better reasoning.
The LLM outputs <tool_call>{"tool": "name", "args": {...}}</tool_call> format
which we parse and execute.

MIGRATION NOTE (Jan 2026): This module now supports Claude Code as the reasoning
backend. Set use_claude=True to use Claude Code CLI instead of Ollama.
"""

import asyncio
import aiohttp
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime

from logger import get_logger
from config import Config
from router import PromptRouter, SkillInfo

# Claude Code integration for subscription-based reasoning
# Phase 4: Use shared bridge for session continuity
from claude_bridge import ClaudeCodeBridge, get_claude_bridge, get_claude_bridge_manager

# Hook system for extensible lifecycle events
from hooks import get_hook_manager, HookType, HookContext

# Haiku progress updates for voice mode (real-time contextual feedback)
from haiku_progress import HaikuProgressManager, init_progress_manager, get_progress_manager

# Dashboard integration - import but don't fail if not available
try:
    import dashboard_integration as dash
except ImportError:
    dash = None

log = get_logger("skill_executor")

# =============================================================================
# Trace Logging System - Mirrors subagent_manager for consistency
# =============================================================================

def _trace_log(tag: str, content: str, max_length: int = None):
    """Log trace information with consistent formatting."""
    if not Config.TRACE_LOGGING_ENABLED:
        return
    max_len = max_length or Config.TRACE_MAX_CONTENT_LENGTH
    if len(content) > max_len:
        truncated = content[:max_len] + f"\n... [TRUNCATED - {len(content)} total chars]"
        log.info(f"[TRACE:{tag}] {truncated}")
    else:
        log.info(f"[TRACE:{tag}] {content}")


def _trace_context_sizes(messages: list, iteration: int = 0, skill_name: str = ""):
    """Log detailed size breakdown of context being sent to LLM."""
    if not Config.TRACE_LOGGING_ENABLED:
        return

    sizes = []
    total_chars = 0
    sys_size = 0

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        size = len(content)
        total_chars += size
        if role == "system":
            sys_size = size
        sizes.append(f"  msg[{i}] {role}: {size:,} chars")

    breakdown = "\n".join(sizes)
    log.info(f"[TRACE:CONTEXT_SIZE] Skill={skill_name} Iteration={iteration} - Total: {total_chars:,} chars")
    log.info(f"[TRACE:CONTEXT_SIZE] System: {sys_size:,}, Messages: {len(messages)}")
    log.debug(f"[TRACE:CONTEXT_SIZE] Breakdown:\n{breakdown}")

    if total_chars > 50000:
        log.warning(f"[TRACE:CONTEXT_SIZE] ⚠️ LARGE CONTEXT: {total_chars:,} chars")


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
    args_str = json.dumps(args, indent=2, default=str)
    log.info(f"[TRACE:TOOL_EXEC] {tool_name} ({duration_ms}ms)")
    _trace_log(f"TOOL_ARGS:{tool_name}", args_str)
    if Config.TRACE_LOG_TOOL_RESULTS:
        _trace_log(f"TOOL_RESULT:{tool_name}", result)


@dataclass
class ToolCall:
    """Represents a tool call extracted from LLM response."""
    name: str
    arguments: Dict[str, Any]
    raw_response: str = ""
    result: Any = None           # Captured result from tool execution
    duration_ms: int = 0         # Execution duration in milliseconds
    error: Optional[str] = None  # Error message if tool failed


@dataclass
class ExecutionResult:
    """Result of skill execution."""
    success: bool
    response: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


class SkillExecutor:
    """
    Execute skills using Fabric-style prompts with constrained tools.

    Uses NON-NATIVE tool calling: The LLM outputs <tool_call> tags which we
    parse and execute. This allows using phi4:14b (excellent reasoning) instead
    of models that support native tool calling but have poor reasoning.

    The executor loads the skill's system.md prompt and calls the LLM with
    ONLY that skill's tools listed in the prompt. This dramatically improves:
    1. Tool selection accuracy (3-5 tools vs 100+)
    2. Argument extraction (LLM does it, not regex)
    3. Response quality (focused system prompt)

    Usage:
        executor = SkillExecutor(router, tool_registry)
        await executor.initialize()

        result = await executor.execute(
            skill_name="Research",
            user_input="research Daniel Miessler PAI",
            context={"working_directory": "/home/user/project"}
        )
    """

    def __init__(
        self,
        router: PromptRouter,
        tool_registry: Any = None,  # The tools object from agent
        ollama_url: str = None,
        execution_model: str = None,  # Model for skill execution (phi4 recommended)
        use_claude: bool = True,  # Use Claude Code instead of Ollama
        voice_mode: bool = False,  # Enable voice-optimized output formatting
        tts_callback: Optional[Callable[[str], Awaitable[None]]] = None,  # TTS function for voice updates
    ):
        self.router = router
        self.tool_registry = tool_registry
        self.ollama_url = ollama_url or Config.OLLAMA_URL
        self.use_claude = use_claude
        self.voice_mode = voice_mode  # Voice mode: format responses for TTS
        self.tts_callback = tts_callback  # TTS callback for progress updates

        # Use phi4 for skill execution (Ollama) or Claude Code
        self.execution_model = execution_model or "phi4:14b"

        # Initialize Claude Code bridge if using Claude
        # Phase 4: Use shared bridge for session continuity across components
        self._claude_bridge: Optional[ClaudeCodeBridge] = None
        if self.use_claude:
            try:
                # Use shared bridge instead of creating a new instance
                self._claude_bridge = get_claude_bridge(timeout_seconds=180)
                log.info("Claude Code bridge initialized for skill execution (shared instance)")
            except Exception as e:
                log.warning(f"Failed to initialize Claude Code, falling back to Ollama: {e}")
                self.use_claude = False

        # Haiku progress manager for real-time voice updates
        # Provides contextual progress feedback during long operations
        self._progress_manager: Optional[HaikuProgressManager] = None
        if self.voice_mode:
            log.info(f"[VOICE] Initializing progress manager: bridge={'yes' if self._claude_bridge else 'NO'}, tts_callback={'yes' if self.tts_callback else 'NO'}")
            try:
                self._progress_manager = HaikuProgressManager(
                    claude_bridge=self._claude_bridge,
                    tts_callback=self.tts_callback,
                    voice_mode=self.voice_mode
                )
                log.info("[VOICE] Haiku progress manager initialized for real-time updates")
            except Exception as e:
                log.warning(f"[VOICE] Failed to initialize progress manager: {e}")
        else:
            log.debug(f"[VOICE] Progress manager NOT initialized (voice_mode={self.voice_mode})")

        # Tool implementations (name -> async callable)
        self._tool_implementations: Dict[str, Callable] = {}

        self._initialized = False

    async def initialize(self):
        """Initialize the executor."""
        if self._initialized:
            return

        # Ensure router is initialized
        if not self.router._initialized:
            await self.router.initialize()

        self._initialized = True
        log.info(f"SkillExecutor initialized with model: {self.execution_model}")
        log.info(f"Voice mode: {'enabled' if self.voice_mode else 'disabled'}")

    def register_tool(self, name: str, implementation: Callable):
        """Register a tool implementation."""
        self._tool_implementations[name] = implementation
        log.debug(f"Registered tool: {name}")

    def register_tools_from_registry(self, registry: Any):
        """
        Register all tools from a tool registry object.

        The registry should have a method like get_all_tools() or be iterable.
        """
        if hasattr(registry, 'get_all_tools'):
            tools = registry.get_all_tools()
            for name, impl in tools.items():
                self.register_tool(name, impl)
        elif hasattr(registry, 'items'):
            for name, impl in registry.items():
                self.register_tool(name, impl)

    async def execute(
        self,
        skill_name: str,
        user_input: str,
        context: Dict[str, Any] = None,
        max_tool_calls: int = 100,  # Increased from 5 - Claude is efficient and needs room for fetch + analysis
        trace_id: str = None,
        trace: "Trace" = None,
        conversation_history: List[Dict[str, str]] = None,  # Recent messages for context continuity
    ) -> ExecutionResult:
        """
        Execute a skill with the user's input.

        Args:
            skill_name: Name of the skill to execute
            user_input: The user's message
            context: Optional context (working_directory, active_project, etc.)
            max_tool_calls: Maximum number of tool calls allowed
            trace_id: Optional trace ID for dashboard events
            trace: Optional Trace object for recording LLM calls
            conversation_history: Recent messages from the conversation for context
                continuity. This allows skills to understand references like
                "save this research" by seeing prior assistant responses.

        Returns:
            ExecutionResult with response and tool call details
        """
        if not self._initialized:
            await self.initialize()

        skill = self.router.get_skill(skill_name)
        if not skill:
            return ExecutionResult(
                success=False,
                response=f"Unknown skill: {skill_name}",
                error=f"Skill '{skill_name}' not found in router",
            )

        # Start progress tracking session for voice mode
        if self._progress_manager:
            current_task = context.get("task_context", "") if context else ""
            self._progress_manager.start_session(
                query=user_input,
                skill_name=skill_name,
                task=current_task[:200] if current_task else ""
            )
            log.debug(f"[VOICE] Progress session started for skill: {skill_name}")

        # Build the system prompt with user input and tool descriptions
        system_prompt = self._build_prompt(skill, user_input, context)

        log.info(f"Executing skill '{skill_name}' with {len(skill.tools)} tools using {self.execution_model}")
        log.debug(f"Tools available: {skill.tools}")

        # Call LLM with non-native tool calling
        try:
            result = await self._call_llm_with_tools(
                system_prompt=system_prompt,
                user_input=user_input,
                skill=skill,
                max_tool_calls=max_tool_calls,
                trace_id=trace_id,
                trace=trace,
                conversation_history=conversation_history,
            )

            # Log progress session summary
            if self._progress_manager:
                summary = self._progress_manager.get_session_summary()
                log.info(f"[VOICE] Progress session complete: {summary['tools_executed']} tools, {summary['updates_generated']} updates, {summary['duration_seconds']}s")

            return result

        except Exception as e:
            log.error(f"Skill execution failed: {e}")
            if dash:
                await dash.error(str(e), "skill_execution", trace_id)
            return ExecutionResult(
                success=False,
                response=f"Sorry, I encountered an error: {str(e)}",
                error=str(e),
            )

    def _build_prompt(
        self,
        skill: SkillInfo,
        user_input: str,
        context: Dict[str, Any] = None,
    ) -> str:
        """Build the full system prompt for skill execution."""
        prompt = skill.system_prompt

        # Inject user input
        prompt = prompt.replace("{{user_input}}", user_input)

        # Inject context if provided
        if context:
            for key, value in context.items():
                placeholder = f"{{{{{key}}}}}"
                if placeholder in prompt:
                    prompt = prompt.replace(placeholder, str(value))

        # CRITICAL: Prepend task context if available
        # This ensures the LLM knows about active tasks and can continue work
        if context and context.get("task_context"):
            task_block = f"""# ACTIVE TASK CONTEXT

{context['task_context']}

IMPORTANT: You are continuing work on the tasks above. Update task status as you complete work.

---

"""
            prompt = task_block + prompt

        # Add tool call format instructions if not already present
        if "<tool_call>" not in prompt:
            tool_format = self._build_tool_format_instructions(skill)
            prompt = prompt + "\n\n" + tool_format

        # CRITICAL: Inject voice formatting requirements when in voice mode
        # This ensures skill responses are TTS-friendly
        if self.voice_mode:
            voice_requirements = self._get_voice_requirements()
            prompt = prompt + "\n\n" + voice_requirements
            log.debug("[VOICE] Injected voice formatting requirements into skill prompt")

        return prompt

    def _build_tool_format_instructions(self, skill: SkillInfo) -> str:
        """Build instructions for the <tool_call> format."""
        tool_descriptions = []

        for tool_name in skill.tools:
            desc = self._get_tool_description(tool_name)
            if desc:
                tool_descriptions.append(f"- **{tool_name}**: {desc}")

        tools_list = "\n".join(tool_descriptions) if tool_descriptions else "- No tools available"

        return f"""# TOOL CALLING FORMAT

When you need to use a tool, output it in this EXACT format:

<tool_call>
{{"tool": "tool_name", "args": {{"param1": "value1", "param2": "value2"}}}}
</tool_call>

## Available Tools for this Skill:
{tools_list}

## Rules:
1. ONLY use tools from the list above
2. Extract clean arguments from the user's request
3. Wait for tool results before responding
4. You can make multiple tool calls if needed
5. After getting results, synthesize them into a helpful response

## Example:
User: "search for Python tutorials"
<tool_call>
{{"tool": "web_search", "args": {{"query": "Python tutorials"}}}}
</tool_call>
"""

    def _get_voice_requirements(self) -> str:
        """
        Return voice formatting requirements to append to skill prompts.

        These instructions ensure LLM output is suitable for text-to-speech.
        Applied when voice_mode=True.
        """
        return """
## CRITICAL: Tool Calls Required + Voice Output

**IMPORTANT: You MUST use <tool_call> tags to execute tools.** The voice formatting rules below apply ONLY to your spoken response text, NOT to tool calls.

### Tool Calling (REQUIRED)
When you need to perform actions, you MUST output <tool_call> tags:

<tool_call>
{"tool": "tool_name", "args": {"param": "value"}}
</tool_call>

Do NOT just say "I'll do X" - actually call the tool! Narrating what you would do is NOT acceptable.

### Voice Output Formatting (for your response text only)
Your spoken response will be read aloud by text-to-speech.

**In your response text, DO NOT USE:**
- Markdown formatting: No **bold**, *italic*, `code`, headers
- Bullet points or numbered lists
- Tables, special characters, or raw URLs

**In your response text, DO USE:**
- Natural spoken sentences
- Concise responses (3-5 sentences)
- If you need to show code or structured data, save it to a file

**Example of correct behavior:**
<tool_call>
{"tool": "parallel_dispatch", "args": {"tasks": [{"agent": "web-researcher", "prompt": "Research agent orchestration"}]}}
</tool_call>

Then after getting results, respond with: "I've dispatched the research agents and they're working on it now."
"""

    def _get_tool_description(self, tool_name: str) -> Optional[str]:
        """Get a description for a tool."""
        # Try tool registry first
        if self.tool_registry and hasattr(self.tool_registry, 'get_all_tools'):
            all_tools = self.tool_registry.get_all_tools()
            if tool_name in all_tools:
                tool_info = all_tools[tool_name]
                return tool_info.get("description", f"{tool_name} tool")

        # Try implementation docstring
        if tool_name in self._tool_implementations:
            impl = self._tool_implementations[tool_name]
            if impl.__doc__:
                return impl.__doc__.split('\n')[0].strip()
            return f"Execute {tool_name}"

        return None

    async def _call_llm_with_tools(
        self,
        system_prompt: str,
        user_input: str,
        skill: SkillInfo,
        max_tool_calls: int = 15,  # Increased from 5 - Claude is efficient and needs room for fetch + analysis
        trace_id: str = None,
        trace: "Trace" = None,
        conversation_history: List[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """
        Call the LLM and handle the non-native tool calling loop.

        The LLM outputs <tool_call> tags which we parse, execute, and feed back.

        Args:
            conversation_history: Recent messages to include for context continuity.
                These are inserted between system prompt and current user input,
                allowing the LLM to understand references like "this research".
        """
        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Include recent conversation history for context continuity
        # This allows the LLM to understand references like "save this research"
        if conversation_history:
            # Filter to only include user/assistant messages (not system)
            # and limit to prevent context overflow
            relevant_history = [
                msg for msg in conversation_history
                if msg.get("role") in ("user", "assistant")
            ][-6:]  # Last 6 messages (3 turns) max

            if relevant_history:
                log.info(f"[CONTEXT] Including {len(relevant_history)} messages from conversation history")
                messages.extend(relevant_history)

        # Add the current user input
        messages.append({"role": "user", "content": user_input})

        all_tool_calls = []
        all_tool_results = []
        iterations = 0
        final_response = ""

        while iterations < max_tool_calls:
            iterations += 1

            # === TRACE: Log context being sent to LLM ===
            _trace_context_sizes(messages, iterations, skill.name)
            _trace_messages_content(messages, f"SKILL:{skill.name}:ITER{iterations}")

            # Emit state events for dashboard
            model_name = "claude-code" if self.use_claude else self.execution_model
            if dash:
                await dash.thinking_started(model_name, skill.name, trace_id)
                # Get system prompt from first message if available
                sys_prompt = messages[0].get("content") if messages and messages[0].get("role") == "system" else None
                await dash.llm_calling(model_name, len(messages), messages=messages, system_prompt=sys_prompt, trace_id=trace_id)

            start_time = time.time()

            # Call LLM - use Claude Code or Ollama based on configuration
            # Track tool_calls from Claude bridge (it extracts them before cleaning content)
            bridge_tool_calls = []

            if self.use_claude and self._claude_bridge:
                # === CLAUDE CODE PATH ===
                try:
                    result = await self._claude_bridge.query(
                        messages=messages,
                        disable_native_tools=True,  # Workshop handles tools via <tool_call>
                    )
                    content = result.get("content", "")
                    # IMPORTANT: Claude bridge extracts tool_calls BEFORE cleaning the content
                    # The cleaned content has <tool_call> tags removed, so we MUST use these
                    bridge_tool_calls = result.get("tool_calls", [])
                    if bridge_tool_calls:
                        log.info(f"[CLAUDE_BRIDGE] Received {len(bridge_tool_calls)} tool calls from bridge: {[tc.get('tool') for tc in bridge_tool_calls]}")

                    # Phase 4: Record turn for session continuity
                    try:
                        manager = get_claude_bridge_manager()
                        # Estimate tokens: ~4 chars per token
                        estimated_tokens = len(content) // 4
                        manager.record_turn(
                            claude_session_id=self._claude_bridge.session_id,
                            estimated_tokens=estimated_tokens
                        )
                    except Exception as e:
                        log.debug(f"Could not record turn: {e}")

                except Exception as e:
                    log.error(f"Claude Code call failed: {e}")
                    if dash:
                        await dash.error(f"Claude error: {e}", "llm_call", trace_id)
                    return ExecutionResult(
                        success=False,
                        response="Sorry, I couldn't process your request.",
                        error=f"Claude Code error: {e}",
                    )
            else:
                # === OLLAMA PATH ===
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "model": self.execution_model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": 2048,  # Allow longer responses for reasoning
                        }
                    }

                    async with session.post(
                        f"{self.ollama_url}/api/chat",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=180),  # 3 min for complex reasoning
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            log.error(f"LLM call failed: {response.status} - {error_text}")
                            if dash:
                                await dash.error(f"LLM error: {response.status}", "llm_call", trace_id)
                            return ExecutionResult(
                                success=False,
                                response="Sorry, I couldn't process your request.",
                                error=f"LLM returned {response.status}: {error_text}",
                            )

                        result = await response.json()

                message = result.get("message", {})
                content = message.get("content", "")

            duration_ms = int((time.time() - start_time) * 1000)

            # === TRACE: Log LLM response ===
            _trace_log(f"CONTEXT_OUT:{skill.name}:ITER{iterations}", content)
            log.info(f"[TRACE:LLM_RESPONSE] Skill={skill.name} Iter={iterations} Duration={duration_ms}ms Chars={len(content)}")

            # Parse tool calls from the response
            # For Claude bridge: use pre-extracted tool_calls (content is already cleaned)
            # For Ollama: parse from content
            if bridge_tool_calls:
                # Convert bridge format {"tool": "name", "args": {...}} to ToolCall objects
                tool_calls = []
                for tc in bridge_tool_calls:
                    tool_name = tc.get("tool", "")
                    tool_args = tc.get("args", {})
                    if tool_name:
                        tool_calls.append(ToolCall(name=tool_name, arguments=tool_args))
                log.info(f"[SKILL_EXECUTOR] Using {len(tool_calls)} tool calls from Claude bridge")
            else:
                # Fallback: parse from content (for Ollama or if bridge didn't extract)
                tool_calls = self._parse_tool_calls(content)

            # === RECORD LLM CALL IN TRACE ===
            if trace:
                from telemetry import LLMCall
                llm_call = LLMCall(
                    model=model_name,  # Use correct model name (claude-code or ollama model)
                    system_prompt=system_prompt[:2000],  # Truncate for storage
                    system_prompt_length=len(system_prompt),
                    messages=messages[-4:] if len(messages) > 4 else messages,  # Keep last 4 messages
                    iteration=iterations,
                    max_iterations=max_tool_calls,
                )
                llm_call.complete(content)
                llm_call.duration_ms = duration_ms
                llm_call.tool_calls_extracted = [{"name": tc.name, "args": tc.arguments} for tc in tool_calls]
                trace.llm_calls.append(llm_call)
                trace.llm_total_calls += 1  # Increment counter (normally done by start_llm_call)
                log.debug(f"[TRACE] Recorded LLM call {llm_call.call_id}: {duration_ms}ms, {len(tool_calls)} tools")

            # Emit LLM complete event
            if dash:
                await dash.llm_complete(len(content), duration_ms, len(tool_calls), trace_id)

            log.debug(f"LLM response ({duration_ms}ms): {len(content)} chars, {len(tool_calls)} tool calls")

            # If no tool calls, we're done - this is the final response
            if not tool_calls:
                # Clean response and apply voice sanitization if needed
                final_response = self._prepare_final_response(content)
                return ExecutionResult(
                    success=True,
                    response=final_response,
                    tool_calls=all_tool_calls,
                    tool_results=all_tool_results,
                )

            # Execute tool calls and collect results
            tool_results_text = []

            # Track all executed tools for work evidence validation
            executed_tool_names = [tc.name for tc in all_tool_calls]

            for tc in tool_calls:
                tool_name = tc.name
                tool_args = tc.arguments

                # Validate tool is in skill's allowed tools
                if tool_name not in skill.tools:
                    log.warning(f"Tool '{tool_name}' not in skill's tools: {skill.tools}")
                    tool_results_text.append(f"Error: Tool '{tool_name}' is not available for this task.")
                    continue

                all_tool_calls.append(tc)
                executed_tool_names.append(tool_name)

                # === WORK EVIDENCE INJECTION ===
                # When task_write is called, inject the list of tools that were executed
                # This allows TaskManager to validate that real work was done
                if tool_name == "task_write":
                    tool_args["_work_evidence"] = executed_tool_names.copy()
                    log.debug(f"Injected work evidence: {executed_tool_names}")

                # Emit tool calling event
                call_id = None
                if dash:
                    call_id = await dash.tool_calling(tool_name, skill.name, tool_args, trace_id=trace_id)

                log.info(f"Tool call: {tool_name}({tool_args})")

                # === PROGRESS UPDATE: Tool Starting ===
                if self._progress_manager:
                    try:
                        await self._progress_manager.on_tool_start(tool_name, tool_args)
                    except Exception as e:
                        log.debug(f"[VOICE] Progress on_tool_start failed: {e}")

                # Execute the tool
                tool_start = time.time()
                tool_error = None
                try:
                    tool_result = await self._execute_tool(tool_name, tool_args)
                    tool_duration = int((time.time() - tool_start) * 1000)

                    # === TRACE: Log tool execution details ===
                    _trace_tool_execution(tool_name, tool_args, str(tool_result), tool_duration)

                    # Emit tool result event (larger preview for research content)
                    if dash and call_id:
                        result_preview = str(tool_result)
                        await dash.tool_result(
                            call_id,
                            result_preview[:5000] if len(result_preview) > 5000 else result_preview,
                            tool_duration,
                            trace_id
                        )

                except Exception as e:
                    tool_result = f"Error: {str(e)}"
                    tool_error = str(e)
                    tool_duration = int((time.time() - tool_start) * 1000)
                    _trace_tool_execution(tool_name, tool_args, tool_result, tool_duration)
                    if dash and call_id:
                        await dash.tool_error(call_id, str(e), trace_id)

                # === PROGRESS UPDATE: Tool Complete ===
                if self._progress_manager:
                    try:
                        await self._progress_manager.on_tool_complete(
                            tool_name, tool_result, tool_duration, error=tool_error
                        )
                    except Exception as e:
                        log.debug(f"[VOICE] Progress on_tool_complete failed: {e}")

                # Capture result on the ToolCall object for tracing
                tc.result = tool_result
                tc.duration_ms = tool_duration
                tc.error = tool_error

                all_tool_results.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result": tool_result,
                })

                tool_results_text.append(f"[{tool_name}] Result:\n{tool_result}")

                # === POST_TOOL_USE HOOKS ===
                # Execute hooks to allow extensions to react to tool outputs
                try:
                    hook_ctx = HookContext(
                        session=None,  # Session not directly available here
                        tool_name=tool_name,
                        tool_args=tool_args,
                        tool_result=tool_result,
                        skill_name=skill.name
                    )
                    hook_ctx = await get_hook_manager().execute(
                        HookType.POST_TOOL_USE,
                        hook_ctx
                    )
                    # If hook modified the result, use the modified version
                    if hook_ctx.tool_result is not None and hook_ctx.tool_result != tool_result:
                        tool_result = hook_ctx.tool_result
                        tc.result = tool_result
                        all_tool_results[-1]["result"] = tool_result
                        log.debug(f"[HOOKS] POST_TOOL_USE modified result for {tool_name}")
                except Exception as e:
                    log.warning(f"[HOOKS] POST_TOOL_USE execution failed: {e}")

            # Add assistant message and tool results to conversation
            messages.append({"role": "assistant", "content": content})

            # Build a continuation prompt that encourages thorough completion
            # without hardcoding domain-specific behaviors
            continuation_prompt = f"""Tool execution results:

{chr(10).join(tool_results_text)}

You have completed {len(all_tool_calls)} tool call(s) so far.

Before responding, evaluate:
1. Does this fully address ALL aspects of the original request?
2. Would additional tool calls provide more complete or higher-quality results?
3. Did the user ask for something comprehensive that might need multiple angles?

If the request is NOT fully satisfied, call another tool now.
Only provide a final response when you have thoroughly addressed the request."""

            messages.append({
                "role": "user",
                "content": continuation_prompt
            })

        # Max iterations reached - do one final synthesis call (no tools)
        # This ensures we provide a useful response even when hitting limits
        if all_tool_results:
            log.info(f"Max iterations ({max_tool_calls}) reached, requesting final synthesis...")

            synthesis_prompt = f"""You have completed {len(all_tool_calls)} tool operations. The tool iteration limit has been reached.

Based on ALL the information gathered above, provide a comprehensive response to the user's original request.

IMPORTANT:
- Do NOT call any more tools
- Synthesize and summarize the gathered information
- Address the user's original question directly
- If the task is incomplete, explain what was accomplished and what remains

Original request: {user_input}"""

            messages.append({"role": "user", "content": synthesis_prompt})

            try:
                # One final call without tool parsing
                if self.use_claude and self._claude_bridge:
                    result = await self._claude_bridge.query(
                        messages=messages,
                        disable_native_tools=True,
                    )
                    final_response = result.get("content", "")
                else:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{self.ollama_url}/api/chat",
                            json={"model": self.execution_model, "messages": messages, "stream": False},
                            timeout=aiohttp.ClientTimeout(total=120),
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                final_response = result.get("message", {}).get("content", "")
                            else:
                                final_response = ""

                if final_response:
                    # Clean response and apply voice sanitization if needed
                    final_response = self._prepare_final_response(final_response)
                    return ExecutionResult(
                        success=True,
                        response=final_response,
                        tool_calls=all_tool_calls,
                        tool_results=all_tool_results,
                    )
            except Exception as e:
                log.warning(f"Synthesis call failed: {e}")

        # Fallback if synthesis fails or no results
        return ExecutionResult(
            success=True,
            response="I've gathered the requested information. The results from each operation are available above.",
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
        )

    def _parse_tool_calls(self, content: str) -> List[ToolCall]:
        """
        Parse <tool_call> blocks from LLM output.

        Handles malformed JSON with repair attempts.
        """
        tool_calls = []

        # Find all <tool_call>...</tool_call> blocks
        pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)

        for match in matches:
            try:
                # Try to parse as JSON
                data = self._repair_and_parse_json(match)

                if data and isinstance(data, dict):
                    tool_name = data.get("tool", "")
                    tool_args = data.get("args", {})

                    if tool_name:
                        tool_calls.append(ToolCall(
                            name=tool_name,
                            arguments=tool_args if isinstance(tool_args, dict) else {},
                            raw_response=match,
                        ))

            except Exception as e:
                log.warning(f"Failed to parse tool call: {e}\nRaw: {match[:200]}")

        return tool_calls

    def _repair_and_parse_json(self, text: str) -> Optional[Dict]:
        """
        Attempt to parse JSON with multiple repair strategies.

        Handles common LLM JSON mistakes:
        - Missing quotes around keys
        - Single quotes instead of double
        - Trailing commas
        - Unescaped newlines in strings
        - Missing closing braces
        """
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
            # Remove trailing commas before } or ]
            fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
            # Replace single quotes around keys/values
            fixed = re.sub(r"'([^']*)'", r'"\1"', fixed)
            # Escape unescaped newlines in strings
            fixed = re.sub(r'(?<!\\)\n', r'\\n', fixed)
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
                        # Try to extract individual args
                        arg_pattern = r'"([^"]+)"\s*:\s*"([^"]*)"'
                        for key, value in re.findall(arg_pattern, args_match.group(1)):
                            args[key] = value

                return {"tool": tool_name, "args": args}
        except:
            pass

        # Strategy 5: Look for common patterns
        try:
            # Match tool: "name" pattern
            tool_match = re.search(r'tool["\s:]+(\w+)', text, re.IGNORECASE)
            if tool_match:
                return {"tool": tool_match.group(1), "args": {}}
        except:
            pass

        log.warning(f"Could not repair JSON: {text[:100]}")
        return None

    def _clean_response(self, content: str) -> str:
        """Clean any tool call artifacts from the final response."""
        # Remove <tool_call> blocks
        cleaned = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL | re.IGNORECASE)
        # Remove empty lines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()

    def _sanitize_for_voice(self, text: str) -> str:
        """
        Remove any remaining markdown artifacts before TTS.

        This is a safety net for voice mode - even with voice instructions in the prompt,
        the LLM sometimes includes markdown formatting. This method strips it out.

        Applied only when voice_mode=True.
        """
        if not text:
            return text

        # Remove markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold** -> bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic* -> italic
        text = re.sub(r'`([^`]+)`', r'\1', text)        # `code` -> code
        text = re.sub(r'```[\s\S]*?```', '', text)      # Remove code blocks entirely
        text = re.sub(r'#{1,6}\s*', '', text)           # Remove headers (# ## ### etc)

        # Remove list markers
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)  # Bullet points
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # Numbered lists

        # Remove links but keep text: [text](url) -> text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

        # Remove raw URLs
        text = re.sub(r'https?://\S+', '', text)

        # Remove tables (lines with |)
        text = re.sub(r'^\|.*\|$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[-|:\s]+$', '', text, flags=re.MULTILINE)

        # Remove blockquotes
        text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)

        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        return text.strip()

    def _prepare_final_response(self, content: str) -> str:
        """
        Prepare the final response for output.

        Cleans tool artifacts and applies voice sanitization if in voice mode.
        """
        # Always clean tool call artifacts
        cleaned = self._clean_response(content)

        # Apply voice sanitization if in voice mode
        if self.voice_mode:
            cleaned = self._sanitize_for_voice(cleaned)
            log.debug(f"[VOICE] Sanitized response for TTS ({len(cleaned)} chars)")

        return cleaned

    async def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Execute a tool and return its result."""
        # Primary path: Use tool registry's execute method (SkillRegistry)
        # This handles dependency injection, argument normalization, and telemetry
        if self.tool_registry and hasattr(self.tool_registry, 'execute'):
            try:
                return await self.tool_registry.execute(tool_name, args)
            except ValueError as e:
                # Tool not found in registry - fall through to local implementations
                log.debug(f"Tool {tool_name} not in registry: {e}")
            except Exception as e:
                log.error(f"Tool {tool_name} failed: {e}")
                return f"Error: {str(e)}"

        # Fallback: Try locally registered implementations
        if tool_name in self._tool_implementations:
            impl = self._tool_implementations[tool_name]
            try:
                if asyncio.iscoroutinefunction(impl):
                    return await impl(**args)
                else:
                    return impl(**args)
            except Exception as e:
                log.error(f"Tool {tool_name} failed: {e}")
                return f"Error: {str(e)}"

        return f"Error: Unknown tool '{tool_name}'"


async def execute_chat(
    user_input: str,
    ollama_url: str = None,
    model: str = None,
    trace_id: str = None,
    use_claude: bool = True,  # Use Claude Code by default
) -> str:
    """
    Handle chat/conversational requests (no tools needed).

    This is called when the router returns "chat" - greetings, thanks, etc.
    """
    messages = [
        {
            "role": "system",
            "content": "You are Workshop, a friendly AI assistant. Be concise and helpful."
        },
        {"role": "user", "content": user_input},
    ]

    model_name = "claude-code" if use_claude else (model or "phi4:14b")

    # Emit LLM calling event with message context
    if dash:
        sys_prompt = messages[0].get("content") if messages and messages[0].get("role") == "system" else None
        await dash.llm_calling(model_name, len(messages), messages=messages, system_prompt=sys_prompt, trace_id=trace_id)

    start_time = time.time()

    try:
        if use_claude:
            # === CLAUDE CODE PATH ===
            # Phase 4: Use shared bridge for session continuity
            bridge = get_claude_bridge(timeout_seconds=60)
            result = await bridge.query(
                messages=messages,
                disable_native_tools=True,  # Just chat, no tools
            )
            content = result.get("content", "Hello!")

            # Phase 4: Record turn for session tracking
            try:
                manager = get_claude_bridge_manager()
                manager.record_turn(
                    claude_session_id=bridge.session_id,
                    estimated_tokens=len(content) // 4
                )
            except Exception:
                pass
        else:
            # === OLLAMA PATH ===
            ollama_url = ollama_url or Config.OLLAMA_URL
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{ollama_url}/api/chat",
                    json={
                        "model": model or "phi4:14b",
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": 0.7},
                    },
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    if response.status != 200:
                        return "Hello! How can I help you today?"

                    result = await response.json()
                    content = result.get("message", {}).get("content", "Hello!")

        # Emit LLM complete event
        duration_ms = int((time.time() - start_time) * 1000)
        if dash:
            await dash.llm_complete(len(content), duration_ms, 0, trace_id)

        return content

    except Exception as e:
        log.error(f"Chat failed: {e}")
        return "Hello! I'm Workshop. How can I help?"


def handle_clarification(user_input: str) -> str:
    """
    Handle clarification requests.

    Called when the router returns "clarify" - the request was too ambiguous.
    """
    return (
        "I'm not sure what you'd like me to do. Could you be more specific?\n\n"
        "For example:\n"
        "- \"research [topic]\" to search the web\n"
        "- \"read [file]\" to view a file\n"
        "- \"what files are in [directory]\" to list contents\n"
        "- \"remember [something]\" to store information\n"
        "- \"compile [sketch]\" for Arduino projects"
    )
