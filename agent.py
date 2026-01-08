"""
Workshop Agent
Core LLM integration with function calling via Ollama
"""

import json
import re
import time
import traceback
import aiohttp
from pathlib import Path
from typing import Any, Optional, List, Dict
from datetime import datetime

from config import Config
from logger import get_logger
from pattern_loader import PatternLoader, PromptPrimitives
from telemetry import get_telemetry, Trace, ToolCallTrace, LLMCall
from skill_registry import IntentMatch
from task_manager import TaskManager, get_task_manager
from subagent_manager import SubagentManager
import dashboard_integration as dash

# Hook system for extensible lifecycle events
from hooks import get_hook_manager, HookType, HookContext, register_builtin_handlers

# Lazy import for semantic routing (Phase 4)
_semantic_router = None

# Lazy imports for Fabric-style two-stage routing (Phase 5)
_prompt_router = None
_skill_executor = None

def get_semantic_router():
    """Lazy-load the semantic router to avoid startup cost."""
    global _semantic_router
    if _semantic_router is None:
        try:
            from semantic_router import SemanticRouter
            config = Config()
            _semantic_router = SemanticRouter(
                skills_dir=Path(__file__).parent / ".workshop" / "Skills",
                embeddings_cache_path=config.SEMANTIC_EMBEDDINGS_CACHE,
                embedding_model=config.SEMANTIC_EMBEDDING_MODEL,
                ollama_url=config.OLLAMA_URL,
                llm_model=config.ROUTER_MODEL,
            )
        except Exception as e:
            log.warning(f"Failed to initialize semantic router: {e}")
            _semantic_router = False  # Mark as failed
    return _semantic_router if _semantic_router else None


def get_prompt_router():
    """Lazy-load the Fabric-style prompt router (Phase 3: uses Claude fallback)."""
    global _prompt_router
    if _prompt_router is None:
        try:
            from router import PromptRouter
            config = Config()
            _prompt_router = PromptRouter(
                skills_dir=Path(__file__).parent / ".workshop" / "Skills",
                agents_dir=Path.home() / ".workshop" / "agents",
                ollama_url=config.OLLAMA_URL,
                router_model=config.ROUTER_MODEL,
                use_claude=True,  # Phase 3: Claude fallback for low-confidence routing
            )
        except Exception as e:
            log.warning(f"Failed to initialize prompt router: {e}")
            _prompt_router = False
    return _prompt_router if _prompt_router else None


def get_skill_executor(router, tool_registry=None, use_claude=True, voice_mode=False, tts_callback=None):
    """Lazy-load the skill executor."""
    global _skill_executor

    # If executor exists but voice_mode changed, update it
    if _skill_executor and hasattr(_skill_executor, 'voice_mode'):
        if _skill_executor.voice_mode != voice_mode:
            log.info(f"Updating skill executor voice_mode: {_skill_executor.voice_mode} -> {voice_mode}")
            _skill_executor.voice_mode = voice_mode
        # Also update tts_callback if progress manager exists
        if hasattr(_skill_executor, '_progress_manager') and _skill_executor._progress_manager:
            old_callback = _skill_executor._progress_manager.tts_callback
            _skill_executor._progress_manager.tts_callback = tts_callback
            _skill_executor.tts_callback = tts_callback
            log.info(f"[VOICE] Updated progress manager tts_callback: {'None' if old_callback is None else 'was set'} -> {'None' if tts_callback is None else 'set'}")

    if _skill_executor is None:
        try:
            from skill_executor import SkillExecutor
            config = Config()
            _skill_executor = SkillExecutor(
                router=router,
                tool_registry=tool_registry,
                ollama_url=config.OLLAMA_URL,
                execution_model="phi4:14b",  # Fallback model for Ollama
                use_claude=use_claude,  # Use Claude Code as reasoning engine
                voice_mode=voice_mode,  # Enable voice-optimized output formatting
                tts_callback=tts_callback,  # TTS callback for voice progress updates
            )
        except Exception as e:
            log.warning(f"Failed to initialize skill executor: {e}")
            _skill_executor = False
    return _skill_executor if _skill_executor else None


log = get_logger("agent")


class Agent:
    """LLM agent with tool use capabilities.

    Routing pipeline (Phase 4):
    1. Workflow triggers (highest priority) - from SkillRegistry
    2. Skill patterns (from SKILL.md files) - from SkillRegistry
    3. Semantic routing (embedding similarity) - from SemanticRouter
    4. LLM decision (fallback) - Ollama with tool roster

    Note: Legacy hardcoded TOOL_INTENT_PATTERNS have been removed.
    All patterns are now defined in SKILL.md files for maintainability.
    """

    def __init__(
        self,
        model: str,
        tools: "ToolRegistry",
        memory: "MemorySystem",
        ollama_url: str = "http://localhost:11434",
        construct_manager: "ConstructManager" = None,
        context_manager: "ContextAwareness" = None,  # Phase 3: Automatic context
        telos_manager: "TelosManager" = None,  # Phase 3 Telos: Personal context
        task_manager: "TaskManager" = None,  # Task tracking for multi-step work
        voice_mode: bool = False,  # Enable voice-optimized prompts
        tts_callback: "Callable" = None,  # TTS callback for voice progress updates
        semantic_routing: bool = None,  # Phase 4: Semantic routing (None = use config)
        use_legacy_routing: bool = False,  # Phase 5: Set True to use old routing system
    ):
        self.model = model
        self.tools = tools
        self.memory = memory
        self.ollama_url = ollama_url
        self.config = Config()
        self.construct_manager = construct_manager
        self.context_manager = context_manager
        self.telos_manager = telos_manager  # NEW: Personal context manager
        self.voice_mode = voice_mode
        self.tts_callback = tts_callback  # TTS callback for real-time voice progress updates

        # Task management - use singleton if not provided
        self.task_manager = task_manager or get_task_manager()

        # Session manager - set via set_session_manager() after construction
        self._session_manager = None

        # Phase 5: Two-stage Fabric-style routing (DEFAULT)
        # Set use_legacy_routing=True to use old semantic+regex routing
        self.use_legacy_routing = use_legacy_routing
        self._prompt_router = None
        self._skill_executor = None
        self._two_stage_initialized = False

        # Phase 4: Semantic routing (used by legacy system OR as pre-filter for two-stage)
        self.semantic_routing_enabled = (
            semantic_routing if semantic_routing is not None
            else self.config.SEMANTIC_ROUTING_ENABLED
        )
        self._semantic_router_initialized = False

        # Phase 4: Subagent orchestration for specialized tasks
        self.subagent_manager = SubagentManager(
            agents_dir=Path.home() / ".workshop" / "agents",
            ollama_url=ollama_url,
            config=self.config
        )

        # Initialize telemetry
        self.telemetry = get_telemetry()
        self._current_trace: Optional[Trace] = None
        self._current_matched_skill: Optional[str] = None  # For contextual tool filtering

        # Initialize pattern loader for PAI-style prompt templates
        self.pattern_loader = PatternLoader(
            Path(__file__).parent / ".workshop" / "patterns"
        )

        # Hook system for extensible lifecycle events
        self._hook_system_additions: str = ""  # Additions from SESSION_START hooks
        self._hook_context_documents: List[str] = []  # Context docs from hooks
        self._session_started_hooks_fired: bool = False  # Track if SESSION_START fired
        self._hooks_initialized: bool = False

        log.info(f"Agent initialized with model: {model}")
        log.info(f"Ollama URL: {ollama_url}")
        log.info(f"Voice mode: {'enabled' if voice_mode else 'disabled'}")
        log.info(f"Routing mode: {'LEGACY' if use_legacy_routing else 'TWO-STAGE (Fabric-style)'}")
        log.debug(f"Available tools: {list(tools.get_all_tools().keys())}")
        if construct_manager:
            log.info("Construct manager attached")
        if context_manager:
            log.info("Context awareness enabled (Phase 3 automatic)")
        if telos_manager:
            log.info("Personal context enabled (Telos)")
        if self.task_manager:
            log.info("Task management enabled")
        log.info("Telemetry enabled")

    def set_session_manager(self, session_manager):
        """
        Set the session manager for session validation.

        This enables:
        - Auto-continue only for tasks in current session
        - Proper session isolation between conversations
        - SESSION_START hooks firing on new sessions
        """
        self._session_manager = session_manager
        # Reset session hooks so they fire on new session
        self.reset_session_hooks()
        log.info("Session manager attached to Agent")

    def reset_session_hooks(self):
        """
        Reset session-related hook state.

        Call this when starting a new session to ensure SESSION_START
        hooks fire again.
        """
        self._session_started_hooks_fired = False
        self._hook_system_additions = ""
        self._hook_context_documents = []
        log.debug("Session hooks reset")

    def _should_use_subagent(self, skill_name: str, user_input: str) -> bool:
        """
        DEPRECATED: Keyword-based subagent triggering.

        This method is no longer used. Subagent spawning is now handled via
        the spawn_subagent tool that the LLM can call directly, allowing it
        to decide when to delegate based on task complexity rather than
        keyword matching.

        See: .workshop/Skills/Subagents/tools/spawn_subagent.py

        Kept for reference/rollback purposes.
        """
        # Always return False - subagent delegation is now LLM-driven
        return False

    def _track_auto_continue_progress(self, result: "ExecutionResult") -> bool:
        """
        Track if real progress was made in the last execution.

        Returns True if we should continue, False if we're stuck.
        This prevents infinite loops where the LLM only calls task_write
        without doing actual work.
        """
        # Get work tools (exclude task management tools)
        task_tools = ('task_write', 'task_read', 'task_clear')
        work_tools = [tc.name for tc in result.tool_calls if tc.name not in task_tools]

        # Initialize tracking if needed
        if not hasattr(self, '_no_progress_count'):
            self._no_progress_count = 0

        if len(work_tools) == 0:
            self._no_progress_count += 1
            log.warning(f"[PROGRESS] No work tools executed ({self._no_progress_count} consecutive turns)")

            # Stop after 2 consecutive turns with no progress
            if self._no_progress_count >= 2:
                log.warning("[PROGRESS] Stopping auto-continue: no progress for 2 turns")
                self._no_progress_count = 0
                return False
        else:
            # Reset counter on progress
            self._no_progress_count = 0
            log.debug(f"[PROGRESS] Work tools executed: {work_tools}")

        return True

    async def _execute_with_subagent(
        self,
        skill_name: str,
        user_input: str,
        context: Dict,
        trace: "Trace"
    ) -> str:
        """
        DEPRECATED: Direct subagent execution from agent.

        This method is no longer called. Subagent spawning is now handled via
        the spawn_subagent tool that the LLM can call directly.

        See: .workshop/Skills/Subagents/tools/spawn_subagent.py

        Kept for reference/rollback purposes.
        """
        log.info(f"[SUBAGENT] Handing off to specialist for {skill_name}")

        # Build context for subagent
        telos_context = None
        if self.telos_manager:
            try:
                telos_context = {
                    "profile_summary": self.telos_manager.get_context_for_llm(),
                    "active_project": self.telos_manager.auto_detect_project(),
                }
            except Exception as e:
                log.debug(f"Telos context not available: {e}")

        # Build research context if available
        research_context = None
        if context.get("research_data"):
            research_context = context["research_data"]

        # Spawn appropriate subagent
        if skill_name == "Research":
            result = await self.subagent_manager.spawn_subagent(
                agent_name="research-summarizer",
                task=user_input,
                input_data={"context": context},
                primary_model=self.model,
                telos_context=telos_context,
                research_context=research_context,
            )
        else:
            # Fallback - should not reach here
            log.warning(f"[SUBAGENT] No specialist for skill: {skill_name}")
            return f"I couldn't find a specialist for {skill_name}."

        if result.get("success"):
            trace.subagent_used = True
            trace.subagent_name = result.get("agent")
            trace.subagent_model = result.get("model")
            trace.subagent_duration_ms = result.get("duration_ms")
            log.info(f"[SUBAGENT] Complete: {result.get('agent')} ({result.get('duration_ms')}ms)")
            return result.get("output", "Subagent completed but returned no output.")
        else:
            error = result.get("error", "Unknown error")
            log.error(f"[SUBAGENT] Failed: {error}")
            return f"I encountered an issue with the specialist agent: {error}"

    def _get_active_project(self) -> Optional[str]:
        """
        Determine the currently active project.

        Priority:
        1. Telos active project
        2. ContextManager detected project
        3. Current working directory
        """
        import os

        # Try Telos first
        if self.telos_manager:
            try:
                active = self.telos_manager.auto_detect_project()
                if active:
                    return active
            except Exception:
                pass

        # Try ContextManager
        if self.context_manager:
            try:
                context = self.context_manager.get_context()
                if context.get('critical', {}).get('active_project'):
                    return context['critical']['active_project']
            except Exception:
                pass

        # Fallback to cwd
        return os.getcwd()

    def _extract_first_tool_from_workflow(
        self,
        workflow,
        user_input: str = "",
        extracted_args: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Extract the first tool call from a workflow's steps.

        Looks for patterns like:
        - Tool: `get_recent_edits(limit=10)`
        - Tool: `web_search(query, max_results=5)`
        - {"tool": "get_recent_edits", "args": {"limit": 10}}

        Args:
            workflow: WorkflowInfo object
            user_input: Original user query (for extracting dynamic args like query)
            extracted_args: Args already extracted from intent matching
        """
        steps = workflow.steps

        # Look for Tool: `tool_name(args)` pattern
        tool_pattern = r'Tool:\s*`?(\w+)\(([^)]*)\)`?'
        match = re.search(tool_pattern, steps)
        if match:
            tool_name = match.group(1)
            args_str = match.group(2).strip()

            # Parse args
            args = {}
            if args_str:
                # Handle key=value pairs and positional args
                for pair in args_str.split(','):
                    pair = pair.strip()
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        key = key.strip()
                        value = value.strip()

                        # Check if value is a placeholder
                        if value in ['query', 'user_query', 'search_query']:
                            value = self._extract_topic_from_input(user_input)
                        else:
                            # Try to parse as number
                            try:
                                value = int(value)
                            except ValueError:
                                try:
                                    value = float(value)
                                except ValueError:
                                    value = value.strip("'\"")
                        args[key] = value
                    else:
                        # Positional arg - check if it's a known placeholder
                        if pair in ['query', 'user_query', 'search_query']:
                            args['query'] = self._extract_topic_from_input(user_input)

            # Merge with extracted args from intent matching
            if extracted_args:
                for key, value in extracted_args.items():
                    if key not in args:
                        args[key] = value

            log.debug(f"Extracted tool from workflow: {tool_name}({args})")
            return {"tool": tool_name, "args": args}

        # Look for JSON tool call pattern
        json_pattern = r'\{"tool":\s*"(\w+)",\s*"args":\s*(\{[^}]+\})\}'
        match = re.search(json_pattern, steps)
        if match:
            tool_name = match.group(1)
            try:
                args = json.loads(match.group(2))
                log.debug(f"Extracted JSON tool from workflow: {tool_name}({args})")
                return {"tool": tool_name, "args": args}
            except json.JSONDecodeError:
                pass

        return None

    def _extract_topic_from_input(self, user_input: str) -> str:
        """
        Extract the topic/subject from a user query.

        Examples:
        - "do some research on Daniel Miessler" -> "Daniel Miessler"
        - "look into PAI architecture" -> "PAI architecture"
        - "research Ollama function calling" -> "Ollama function calling"
        - "research on this PAI concept from Daniel Miessler" -> "PAI concept Daniel Miessler"
        """
        # First, try to extract named entities or key phrases
        # Look for proper nouns and technical terms

        # Patterns to extract topic - more specific first
        patterns = [
            # "research on X from/by Y" pattern
            r'research\s+(?:on|about)\s+(?:this\s+)?(.+?)\s+(?:from|by)\s+(.+?)(?:\s+and\s+|,|$)',
            # "X concept/architecture/project by/from Y"
            r'(\w+(?:\s+\w+)?)\s+(?:concept|architecture|project|framework)\s+(?:from|by|created by)\s+(.+?)(?:\s+and\s+|,|$)',
            # Standard patterns
            r'(?:do\s+)?(?:some\s+)?research\s+(?:on|about)\s+(?:this\s+)?(.+?)(?:\s+and\s+see|\s+to\s+see|,|$)',
            r'look\s+into\s+(.+?)(?:\s+and\s+|,|$)',
            r'compile\s+(?:information\s+)?(?:on|about)\s+(.+?)(?:\s+and\s+|,|$)',
            r'search\s+(?:for\s+)?(.+?)(?:\s+and\s+|,|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    # Combine topic + source (e.g., "PAI concept" + "Daniel Miessler")
                    topic = f"{groups[0].strip()} {groups[1].strip()}"
                else:
                    topic = groups[0].strip()
                topic = topic.rstrip('.,;:!?')
                # Clean up common filler words
                topic = re.sub(r'^(?:this|the|a|an)\s+', '', topic, flags=re.IGNORECASE)
                if len(topic) > 5:  # Ensure we have something meaningful
                    return topic

        # Fallback: extract key terms (capitalized words, technical terms)
        # Look for proper nouns or technical terms
        key_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', user_input)
        if key_terms:
            return ' '.join(key_terms[:3])  # Take first 3 proper nouns

        return user_input[:100]  # Limit length as last resort

    def _parse_workflow_to_tasks(self, workflow) -> List[Dict[str, str]]:
        """
        Parse workflow steps into a task list.

        This is critical for the multi-model architecture:
        - The task list gives the groq model clear guidance on what to do next
        - Each step becomes a task with status tracking
        - The first task starts as "in_progress"

        Args:
            workflow: WorkflowInfo object with steps

        Returns:
            List of task dicts ready for task_manager.write_tasks()
        """
        tasks = []
        steps_text = workflow.steps

        # Try to parse numbered steps: "1. Do something" or "Step 1: Do something"
        step_patterns = [
            r'(\d+)\.\s*(?:\*\*)?(.+?)(?:\*\*)?(?:\n|$)',  # 1. Step content
            r'Step\s*(\d+)[:\s]+(.+?)(?:\n|$)',  # Step 1: Step content
            r'-\s+(?:\*\*)?(.+?)(?:\*\*)?(?:\n|$)',  # - Bullet point step
        ]

        for pattern in step_patterns:
            matches = re.findall(pattern, steps_text, re.IGNORECASE)
            if matches:
                for i, match in enumerate(matches):
                    if isinstance(match, tuple) and len(match) == 2:
                        # Numbered step
                        content = match[1].strip()
                    else:
                        # Bullet point
                        content = match.strip() if isinstance(match, str) else match[0].strip()

                    # Clean up content
                    content = re.sub(r'^(?:Tool:\s*)?`?', '', content)
                    content = re.sub(r'`?$', '', content)
                    content = content.strip()

                    if content and len(content) > 3:
                        # Generate active form (present continuous)
                        active_form = self._to_active_form(content)

                        tasks.append({
                            "content": content,
                            "status": "in_progress" if i == 0 else "pending",
                            "active_form": active_form
                        })
                break  # Use first matching pattern

        # If no numbered steps, try to extract from purpose/description
        if not tasks and workflow.purpose:
            tasks.append({
                "content": workflow.purpose,
                "status": "in_progress",
                "active_form": self._to_active_form(workflow.purpose)
            })

        return tasks

    def _to_active_form(self, content: str) -> str:
        """
        Convert imperative form to present continuous (active form).

        "Search for information" -> "Searching for information"
        "Analyze the results" -> "Analyzing the results"
        """
        # Common verb transformations
        transformations = [
            (r'^Search\b', 'Searching'),
            (r'^Find\b', 'Finding'),
            (r'^Get\b', 'Getting'),
            (r'^Fetch\b', 'Fetching'),
            (r'^Load\b', 'Loading'),
            (r'^Analyze\b', 'Analyzing'),
            (r'^Summarize\b', 'Summarizing'),
            (r'^Create\b', 'Creating'),
            (r'^Build\b', 'Building'),
            (r'^Write\b', 'Writing'),
            (r'^Read\b', 'Reading'),
            (r'^Execute\b', 'Executing'),
            (r'^Run\b', 'Running'),
            (r'^Call\b', 'Calling'),
            (r'^Check\b', 'Checking'),
            (r'^Verify\b', 'Verifying'),
            (r'^Process\b', 'Processing'),
            (r'^Expand\b', 'Expanding'),
            (r'^Synthesize\b', 'Synthesizing'),
            (r'^Compile\b', 'Compiling'),
            (r'^Upload\b', 'Uploading'),
        ]

        for pattern, replacement in transformations:
            if re.match(pattern, content, re.IGNORECASE):
                return re.sub(pattern, replacement, content, flags=re.IGNORECASE)

        # Default: prepend "Working on:"
        return f"Working on: {content}"

    def _generate_tasks_for_request(
        self,
        user_input: str,
        skill_name: str
    ) -> List[Dict[str, str]]:
        """
        Generate a task list for non-trivial requests.

        This automatically creates tasks based on the matched skill,
        framing the conversation productively even without an explicit workflow.

        Note: In two-stage routing, we don't pre-extract topics because the LLM
        will do it properly. Tasks are generic to the skill type.

        Args:
            user_input: The user's request
            skill_name: The matched skill name

        Returns:
            List of task dicts or empty list if not applicable
        """
        tasks = []

        # === RESEARCH SKILL ===
        # Research skill now creates its own task list via task_write tool
        # with request-specific tasks. No hardcoded tasks here.
        if skill_name == "Research":
            pass  # Skill handles its own task planning

        # === FILE OPERATIONS SKILL TASKS ===
        elif skill_name == "FileOperations":
            # Only create tasks for complex file operations
            if any(kw in user_input.lower() for kw in ['search', 'find', 'all', 'multiple']):
                tasks = [
                    {
                        "content": "Locate relevant files",
                        "status": "in_progress",
                        "active_form": "Locating files"
                    },
                    {
                        "content": "Process file contents",
                        "status": "pending",
                        "active_form": "Processing files"
                    },
                    {
                        "content": "Summarize results",
                        "status": "pending",
                        "active_form": "Summarizing results"
                    }
                ]

        # === CONTEXT INTELLIGENCE SKILL TASKS ===
        elif skill_name == "ContextIntelligence":
            tasks = [
                {
                    "content": "Analyze current project context",
                    "status": "in_progress",
                    "active_form": "Analyzing context"
                },
                {
                    "content": "Identify relevant files and patterns",
                    "status": "pending",
                    "active_form": "Identifying patterns"
                },
                {
                    "content": "Provide contextual insights",
                    "status": "pending",
                    "active_form": "Providing insights"
                }
            ]

        # === SUBAGENTS SKILL TASKS ===
        elif skill_name == "Subagents":
            tasks = [
                {
                    "content": "Identify appropriate specialist agent",
                    "status": "in_progress",
                    "active_form": "Identifying specialist"
                },
                {
                    "content": "Delegate task to specialist",
                    "status": "pending",
                    "active_form": "Delegating to specialist"
                },
                {
                    "content": "Process and present specialist output",
                    "status": "pending",
                    "active_form": "Processing specialist output"
                }
            ]

        return tasks

    def _infer_tool_from_semantic_match(
        self,
        semantic_match: Dict,
        skill_info,
        user_input: str
    ) -> Optional[Dict]:
        """
        Infer the correct tool to call based on semantic match.

        This bypasses the groq model entirely for high-confidence matches,
        routing directly to the appropriate tool.

        Args:
            semantic_match: Dict with matched_utterance, skill_name, confidence
            skill_info: Skill object with tools
            user_input: Original user query

        Returns:
            Tool call dict {"tool": name, "args": {}} or None
        """
        utterance = semantic_match.get('matched_utterance', '').lower()
        skill_name = semantic_match.get('skill_name', '')

        # === RESEARCH SKILL TOOL INFERENCE ===
        if skill_name == "Research":
            # Synthesis/analysis utterances -> summarize_research (uses phi4 subagent)
            synthesis_keywords = [
                'think', 'thoughts', 'analyze', 'synthesize', 'summary', 'summarize',
                'takeaways', 'insights', 'findings', 'apply', 'patterns'
            ]
            if any(kw in utterance for kw in synthesis_keywords):
                log.info("Semantic inference: Research synthesis -> summarize_research")
                return {"tool": "summarize_research", "args": {}}

            # Show research utterances -> show_research_platform
            show_keywords = [
                'show', 'display', 'list', 'articles', 'sources', 'blogs',
                'posts', 'what do we have', 'what did we find'
            ]
            if any(kw in utterance for kw in show_keywords):
                log.info("Semantic inference: Show research -> show_research_platform")
                return {"tool": "show_research_platform", "args": {}}

            # Search utterances -> web_search or deep_research
            search_keywords = ['search', 'google', 'look up', 'find', 'research']
            deep_keywords = ['deep', 'comprehensive', 'thorough', 'compile', 'distill']

            if any(kw in utterance for kw in search_keywords):
                # Extract the topic from user input
                topic = self._extract_topic_from_input(user_input)

                if any(kw in utterance for kw in deep_keywords):
                    log.info(f"Semantic inference: Deep research -> deep_research({topic})")
                    return {"tool": "deep_research", "args": {"query": topic}}
                else:
                    log.info(f"Semantic inference: Web search -> web_search({topic})")
                    return {"tool": "web_search", "args": {"query": topic}}

        # === FILE OPERATIONS SKILL TOOL INFERENCE ===
        elif skill_name == "FileOperations":
            if any(kw in utterance for kw in ['list', 'show', 'files', 'directory']):
                return {"tool": "list_directory", "args": {"path": "."}}
            if any(kw in utterance for kw in ['read', 'show', 'content']):
                # Would need to extract file path from user_input
                pass

        # === TASK MANAGEMENT SKILL TOOL INFERENCE ===
        elif skill_name == "TaskManagement":
            if any(kw in utterance for kw in ['show', 'list', 'current', 'progress']):
                return {"tool": "task_read", "args": {}}
            if any(kw in utterance for kw in ['clear', 'done', 'finish']):
                return {"tool": "task_clear", "args": {}}

        return None

    def _advance_task_on_tool_completion(self, tool_name: str) -> None:
        """
        Update task state when a tool completes.

        This advances the workflow by:
        1. Marking the current in_progress task as completed
        2. Setting the next pending task to in_progress

        Args:
            tool_name: Name of the tool that just completed
        """
        if not self.task_manager:
            return

        try:
            current_tasks = self.task_manager.read_tasks_raw()
            if not current_tasks:
                return

            # Find current in_progress task and mark it complete
            updated_tasks = []
            found_in_progress = False
            should_start_next = False

            for task in current_tasks:
                task_copy = task.copy()

                if task_copy.get("status") == "in_progress":
                    # Mark current task as completed
                    task_copy["status"] = "completed"
                    found_in_progress = True
                    should_start_next = True
                    log.info(f"Task completed: {task_copy.get('content', '')[:50]}...")

                elif should_start_next and task_copy.get("status") == "pending":
                    # Start the next pending task
                    task_copy["status"] = "in_progress"
                    should_start_next = False
                    log.info(f"Starting next task: {task_copy.get('content', '')[:50]}...")

                updated_tasks.append(task_copy)

            if found_in_progress:
                # Update the task list
                self.task_manager.write_tasks(updated_tasks)

        except Exception as e:
            log.warning(f"Failed to advance task state: {e}")

    def _should_inject_context(self, user_input: str) -> bool:
        """
        Determine if we should inject development context for this query.
        Returns True if the query is related to development/hardware work.
        """
        # Keywords that indicate development context would be helpful
        dev_keywords = [
            # Arduino/hardware
            'compile', 'upload', 'flash', 'build', 'verify', 'board', 'sketch',
            'arduino', 'esp32', 'nano', 'serial', 'monitor', 'firmware',
            # Files/code
            'file', 'directory', 'folder', 'code', 'script', 'program',
            '.ino', '.cpp', '.h', '.py', '.js',
            # Development actions
            'debug', 'test', 'run', 'execute', 'open', 'show', 'edit',
            'error', 'bug', 'fix', 'issue', 'problem',
            # Context queries
            'working on', 'project', 'current', 'this', 'recent',
            'connected', 'device', 'port',
        ]
        
        input_lower = user_input.lower()
        
        # Check for dev keywords
        for keyword in dev_keywords:
            if keyword in input_lower:
                return True
        
        # Check for file extensions
        if re.search(r'\.\w{1,4}\b', user_input):
            return True
        
        # Don't inject for greetings/small talk
        if len(user_input.split()) <= 3 and any(greeting in input_lower for greeting in [
            'hi', 'hello', 'hey', 'thanks', 'thank you', 'bye', 'goodbye'
        ]):
            return False
        
        return False

    async def _initialize_two_stage_routing(self):
        """Initialize the two-stage routing system (lazy loading)."""
        if self._two_stage_initialized:
            return

        log.info("Initializing two-stage Fabric-style routing...")

        # Get or create the prompt router
        self._prompt_router = get_prompt_router()
        if self._prompt_router:
            # Initialize with semantic router for pre-filtering (optional optimization)
            semantic_router = get_semantic_router() if self.semantic_routing_enabled else None
            await self._prompt_router.initialize(semantic_router=semantic_router)

            # Get or create the skill executor (pass voice_mode and tts_callback for real-time updates)
            self._skill_executor = get_skill_executor(
                self._prompt_router,
                self.tools,
                voice_mode=self.voice_mode,
                tts_callback=self.tts_callback
            )
            if self._skill_executor:
                await self._skill_executor.initialize()
                # Note: Don't register tools from get_all_tools() - it returns metadata dicts, not callables.
                # The skill_executor uses tool_registry.execute() directly to run tools.
                log.info("Skill executor initialized (tools executed via registry)")

        self._two_stage_initialized = True
        log.info("Two-stage routing initialized")

    async def _two_stage_chat(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """
        Process user input using Fabric-style two-stage routing.

        Stage 1: Router prompt classifies intent → skill name
        Stage 2: Skill executor runs with constrained tools → LLM extracts args

        This is the DEFAULT routing mode. The LLM always extracts arguments,
        not regex patterns. Tools are constrained to 3-5 per skill instead of 100+.

        Args:
            user_input: The user's message
            context: Optional context dict with keys like:
                - active_skill: The skill that was last executed (for auto-continue fallback)
        """
        # === START TRACE ===
        trace = self.telemetry.start_trace(user_input)
        self._current_trace = trace
        trace.model_name = self.model
        trace.voice_mode = self.voice_mode
        trace.constructs_enabled = self.construct_manager is not None

        log.info(f"=" * 50)
        log.info(f"[TWO-STAGE] TRACE {trace.trace_id[:8]} | USER INPUT: {user_input}")
        log.info(f"=" * 50)

        try:
            # === STEP -1: INITIALIZE AND EXECUTE HOOKS ===
            # Register built-in handlers if not already done
            if not self._hooks_initialized:
                try:
                    register_builtin_handlers()
                    self._hooks_initialized = True
                    log.info("[HOOKS] Built-in handlers registered")
                except Exception as e:
                    log.warning(f"[HOOKS] Failed to register built-in handlers: {e}")

            # Fire SESSION_START hooks on first message of session
            # (or on every message if session tracking not available)
            should_fire_session_hooks = not self._session_started_hooks_fired
            if self._session_manager:
                # If we have session manager, only fire on true session start
                current_session = self._session_manager.get_current_session()
                should_fire_session_hooks = (
                    current_session and
                    current_session.message_count <= 1 and
                    not self._session_started_hooks_fired
                )

            if should_fire_session_hooks:
                try:
                    # Get current session for hook context
                    session = None
                    if self._session_manager:
                        session = self._session_manager.get_current_session()

                    hook_ctx = HookContext(
                        session=session,
                        user_input=user_input
                    )

                    # Execute SESSION_START hooks
                    hook_ctx = await get_hook_manager().execute(
                        HookType.SESSION_START,
                        hook_ctx
                    )

                    # Store additions for system message assembly
                    self._hook_system_additions = hook_ctx.get_combined_system_additions()
                    self._hook_context_documents = hook_ctx.context_documents.copy()
                    self._session_started_hooks_fired = True

                    if self._hook_system_additions:
                        log.info(f"[HOOKS] SESSION_START additions: {len(self._hook_system_additions)} chars")
                        trace.add_context_source(
                            source_type="hooks_session_start",
                            content=self._hook_system_additions
                        )

                    # === PIPELINE TRACER: CAPTURE HOOK CONTEXT ===
                    pipeline_tracer = getattr(self, '_pipeline_tracer', None)
                    if pipeline_tracer:
                        from tests.e2e.context_tracer import TraceStage, ContextLayer
                        pipeline_tracer.set_stage(TraceStage.CONTEXT_TELOS)

                        # Capture each hook's system prompt addition
                        for section_name, section_content in hook_ctx.system_prompt_additions.items():
                            pipeline_tracer.add_telos_layer(ContextLayer(
                                layer_name=f"hook:{section_name}",
                                content=section_content,
                                content_length=len(section_content),
                                loaded_successfully=True,
                            ))

                except Exception as e:
                    log.warning(f"[HOOKS] SESSION_START execution failed: {e}")

            # === STEP 0: CONTEXT INJECTION ===
            # Inject ContextManager data (active files, recent changes, workflow detection)
            enriched_input = user_input
            should_inject = self._should_inject_context(user_input)
            trace.context_injection_attempted = should_inject

            if self.context_manager and should_inject:
                try:
                    context_data = self.context_manager.get_context()
                    trace.context_manager_used = True

                    # Capture active files in trace
                    if context_data.get('critical', {}).get('active_files'):
                        trace.active_files = [f['path'] for f in context_data['critical']['active_files']]
                    # Capture recent changes in trace
                    if context_data.get('critical', {}).get('recent_changes'):
                        trace.recent_changes = context_data['critical']['recent_changes']
                    # Capture detected workflow in trace
                    if context_data.get('critical', {}).get('detected_workflow'):
                        wf = context_data['critical']['detected_workflow']
                        trace.detected_workflow = wf.get('type')
                        trace.detected_workflow_confidence = float(wf.get('confidence', '0%').replace('%', '')) / 100

                    # Format context for LLM
                    context_str = self.context_manager.format_context_for_llm(context_data)
                    if context_str:
                        trace.add_context_source(source_type="context_manager", content=context_str)
                        enriched_input = f"{user_input}\n\n[Current Context:\n{context_str}]"
                        trace.context_injection_succeeded = True
                        trace.context_assembled = context_str
                        trace.context_assembled_length = len(context_str)
                        log.info(f"[TWO-STAGE] Context injected ({len(context_str)} chars)")

                        # === PIPELINE TRACER: CAPTURE AUTO CONTEXT ===
                        pipeline_tracer = getattr(self, '_pipeline_tracer', None)
                        if pipeline_tracer:
                            from tests.e2e.context_tracer import TraceStage
                            pipeline_tracer.set_stage(TraceStage.CONTEXT_AUTO)
                            pipeline_tracer.auto_context_enabled = True
                            pipeline_tracer.auto_context_formatted = context_str
                            pipeline_tracer.auto_context_length = len(context_str)
                            pipeline_tracer.auto_context_injection_reason = "Dev keywords matched"
                            pipeline_tracer.active_files = trace.active_files
                            pipeline_tracer.detected_workflow = trace.detected_workflow or ""
                            pipeline_tracer.workflow_confidence = trace.detected_workflow_confidence

                except Exception as e:
                    log.warning(f"Context injection failed: {e}")
                    trace.context_should_inject_reason = f"Error: {e}"

            trace.user_input_enriched = enriched_input
            trace.user_input_enriched_length = len(enriched_input)

            # Initialize two-stage routing if needed
            await self._initialize_two_stage_routing()

            if not self._prompt_router or not self._skill_executor:
                log.warning("Two-stage routing not available, falling back to legacy")
                self.use_legacy_routing = True
                return await self.chat(user_input)

            # === STAGE 1: ROUTE TO SKILL ===
            log.info("[STAGE 1] Routing to skill...")

            # Emit routing started event for dashboard
            if dash.is_enabled():
                await dash.routing_started(user_input, trace.trace_id)
                await dash.agent_state("routing", "Classifying intent...", trace.trace_id)

            # Build routing context - include task context for "continue" handling
            routing_context = {
                "working_directory": self._get_active_project() or ".",
            }

            # Merge in any context passed from caller (e.g., active_skill for auto-continue)
            if context:
                routing_context.update(context)

            # CRITICAL: Add task context so router knows about active workflows
            if self.task_manager and self.task_manager.has_tasks():
                task_context = self.task_manager.get_context_for_router()
                if task_context:
                    routing_context["task_context"] = task_context
                    log.info(f"[ROUTING] Task context: {task_context[:100]}...")

            routing_result = await self._prompt_router.route(user_input, context=routing_context)

            log.info(f"  Skill: {routing_result.skill_name}")
            log.info(f"  Method: {routing_result.method}")
            log.info(f"  Confidence: {routing_result.confidence:.2f}")

            # Record routing in trace
            trace.intent_detected = routing_result.is_skill
            trace.intent_skill_matched = routing_result.skill_name
            trace.intent_confidence = routing_result.confidence
            trace.intent_match_type = routing_result.method

            # === PIPELINE TRACER INSTRUMENTATION ===
            pipeline_tracer = getattr(self, '_pipeline_tracer', None)
            if pipeline_tracer:
                from tests.e2e.context_tracer import TraceStage
                pipeline_tracer.set_stage(TraceStage.ROUTING_DECIDED)
                pipeline_tracer.routing.final_skill = routing_result.skill_name
                pipeline_tracer.routing.final_method = routing_result.method
                pipeline_tracer.routing.final_confidence = routing_result.confidence
                pipeline_tracer.routing.semantic_enabled = True
                pipeline_tracer.routing.semantic_score = routing_result.confidence

                # Capture semantic routing details if available
                if hasattr(routing_result, 'semantic_score'):
                    pipeline_tracer.routing.semantic_score = routing_result.semantic_score
                if hasattr(routing_result, 'matched_utterance'):
                    pipeline_tracer.routing.semantic_matched_utterance = routing_result.matched_utterance
                if hasattr(routing_result, 'top_candidates'):
                    pipeline_tracer.routing.semantic_all_scores = routing_result.top_candidates

                # Capture timing
                pipeline_tracer.duration_routing_ms = trace.intent_detection_time_ms

            # Emit dashboard event
            if dash.is_enabled() and routing_result.is_skill:
                await dash.skill_matched(
                    routing_result.skill_name,
                    f"Method: {routing_result.method}, Confidence: {routing_result.confidence:.2f}"
                )

            # === HANDLE SPECIAL CASES ===
            if routing_result.needs_clarification:
                from skill_executor import handle_clarification
                response = handle_clarification(user_input)
                self.telemetry.complete_trace(trace, response)
                self._current_trace = None
                return response

            if routing_result.is_chat:
                from skill_executor import execute_chat
                response = await execute_chat(user_input, self.ollama_url, self.model, use_claude=True)
                self.telemetry.complete_trace(trace, response)
                self._current_trace = None
                return response

            # === AUTO-CREATE TASKS ===
            if self.task_manager and not self.task_manager.has_tasks():
                auto_tasks = self._generate_tasks_for_request(
                    user_input,
                    routing_result.skill_name
                )
                if auto_tasks:
                    self.task_manager.write_tasks(auto_tasks, original_request=user_input)
                    log.info(f"Auto-created {len(auto_tasks)} tasks for {routing_result.skill_name}")

                    if dash.is_enabled():
                        await dash.emit("task_list_updated", {
                            "task_count": len(auto_tasks),
                            "skill": routing_result.skill_name
                        })

            # === STAGE 2: EXECUTE SKILL ===
            # NOTE: Subagent spawning is now handled via the spawn_subagent tool
            # that the LLM can call directly. This allows the LLM to decide when
            # to delegate based on task complexity rather than keyword matching.
            log.info(f"[STAGE 2] Executing skill '{routing_result.skill_name}'...")

            # Emit state change for dashboard
            if dash.is_enabled():
                await dash.agent_state("executing", f"Running {routing_result.skill_name} skill...", trace.trace_id)

            # Build context for skill execution
            context = {
                "working_directory": self._get_active_project() or ".",
            }

            # Add Telos context if available
            if self.telos_manager:
                try:
                    telos_context = self.telos_manager.get_context_for_llm()
                    if telos_context:
                        context["telos"] = telos_context
                except Exception as e:
                    log.debug(f"Telos context not available: {e}")

            # CRITICAL: Add task context so LLM knows what tasks are active
            # This enables continuity across conversation turns
            if self.task_manager and self.task_manager.has_tasks():
                task_llm_context = self.task_manager.get_context_for_llm()
                if task_llm_context:
                    context["task_context"] = task_llm_context
                    log.info(f"[EXECUTION] Task context injected ({len(task_llm_context)} chars)")

            # Get recent conversation history for context continuity
            # This allows skills to understand references like "save this research"
            conversation_history = None
            if self.memory:
                try:
                    conversation_history = self.memory.get_recent_messages(n=6)
                    if conversation_history:
                        log.info(f"[CONTEXT] Passing {len(conversation_history)} messages to skill executor")
                except Exception as e:
                    log.debug(f"Could not get conversation history: {e}")

            # Execute the skill with enriched input (includes context if applicable)
            result = await self._skill_executor.execute(
                skill_name=routing_result.skill_name,
                user_input=enriched_input,
                context=context,
                trace=trace,
                conversation_history=conversation_history,
            )

            # Log tool calls to trace
            for tc in result.tool_calls:
                log.info(f"  Tool: {tc.name}({tc.arguments})")
                # Use the Trace's start_tool_call method to properly record
                tool_trace = trace.start_tool_call(
                    tool_name=tc.name,
                    skill_name=routing_result.skill_name,
                    args=tc.arguments
                )
                # Capture tool results in trace for full observability
                tool_trace.result_raw = str(tc.result)[:10000] if tc.result else ""
                tool_trace.result_length = len(str(tc.result)) if tc.result else 0
                tool_trace.duration_ms = tc.duration_ms
                tool_trace.result_type = "error" if tc.error else "success"
                tool_trace.error_message = tc.error

            # === PIPELINE TRACER: CAPTURE EXECUTION DETAILS ===
            pipeline_tracer = getattr(self, '_pipeline_tracer', None)
            if pipeline_tracer:
                from tests.e2e.context_tracer import (
                    TraceStage, ToolCallDetail, LLMInvocation, ContextLayer
                )

                # Capture tool calls
                for tc in result.tool_calls:
                    tool_detail = ToolCallDetail(
                        call_id=f"tool_{id(tc)}",
                        tool_name=tc.name,
                        skill_name=routing_result.skill_name,
                        args_from_llm=tc.arguments if isinstance(tc.arguments, dict) else {},
                        duration_ms=tc.duration_ms,
                        result=str(tc.result)[:1000] if tc.result else "",
                        result_length=len(str(tc.result)) if tc.result else 0,
                        result_type="error" if tc.error else "success",
                        error=tc.error,
                    )
                    pipeline_tracer.add_tool_call(tool_detail)

                # Capture LLM invocations if available
                if hasattr(result, 'llm_calls') and result.llm_calls:
                    for i, llm_call in enumerate(result.llm_calls):
                        invocation = LLMInvocation(
                            invocation_id=f"llm_{i}",
                            iteration=i + 1,
                            system_prompt_length=len(llm_call.get('system', '')),
                            user_message_length=len(llm_call.get('user', '')),
                            response_length=len(llm_call.get('response', '')),
                            duration_ms=llm_call.get('duration_ms', 0),
                            tool_calls_detected=llm_call.get('tool_calls', 0),
                        )
                        pipeline_tracer.add_llm_invocation(invocation)

                # Capture context layers if available
                if context.get("telos"):
                    pipeline_tracer.add_telos_layer(ContextLayer(
                        layer_name="telos_context",
                        content=context["telos"],
                        content_length=len(context["telos"]),
                    ))

                if context.get("task_context"):
                    pipeline_tracer.task_context_formatted = context["task_context"]
                    pipeline_tracer.task_context_length = len(context["task_context"])

                # Update timing
                pipeline_tracer.duration_tools_ms = sum(tc.duration_ms for tc in result.tool_calls)
                pipeline_tracer.iterations_used = result.iterations if hasattr(result, 'iterations') else len(result.tool_calls)

                pipeline_tracer.set_stage(TraceStage.ITERATION_COMPLETE)

            # Advance task state on tool completion
            if result.tool_calls:
                self._advance_task_on_tool_completion(result.tool_calls[-1].name)

            # Complete trace
            response = result.response
            log.info(f"[TWO-STAGE] RESPONSE: {response[:200]}...")

            self.telemetry.complete_trace(trace, response)
            self._current_trace = None

            # === AUTO-CONTINUE: Check if there are pending tasks to work on ===
            # If tasks remain and we just completed work, automatically continue
            if self.task_manager and self.task_manager.has_tasks():
                stats = self.task_manager.get_stats()
                pending_count = stats.get('pending', 0)
                in_progress_count = stats.get('in_progress', 0)

                # SESSION VALIDATION: Only auto-continue if tasks belong to current session
                # This prevents stale tasks from previous conversations from executing
                session_valid = True
                if hasattr(self, '_session_manager') and self._session_manager:
                    task_session = self.task_manager.get_session_id()
                    if task_session and not self._session_manager.is_current_session(task_session):
                        log.warning(f"[AUTO-CONTINUE] Skipping: tasks from different session {task_session}")
                        session_valid = False

                # === PROGRESS TRACKING ===
                # Detect if we're making real progress or just spinning
                making_progress = self._track_auto_continue_progress(result)

                # Only auto-continue if:
                # 1. There are pending tasks OR there's an in-progress task
                # 2. We're not stuck in a loop (limit recursion)
                # 3. The last response wasn't an error
                # 4. Tasks belong to the current session
                # 5. We're making actual progress (calling work tools)
                if (pending_count > 0 or in_progress_count > 0) and result.success and session_valid:
                    # Check if we're stuck (no real work being done)
                    if not making_progress:
                        log.warning("[AUTO-CONTINUE] Not making progress, stopping auto-continue")
                        return response + "\n\nI'm not making progress on the remaining tasks. Could you provide more guidance?"

                    # Check recursion depth to prevent infinite loops
                    recursion_depth = getattr(self, '_auto_continue_depth', 0)
                    max_depth = 10  # Maximum auto-continue iterations

                    if recursion_depth < max_depth:
                        self._auto_continue_depth = recursion_depth + 1
                        log.info(f"[AUTO-CONTINUE] {pending_count} pending, {in_progress_count} in progress. Continuing... (depth={recursion_depth + 1})")

                        # Emit state for dashboard
                        if dash.is_enabled():
                            await dash.agent_state("executing", f"Auto-continuing tasks ({stats['completed']}/{stats['total']} done)...", None)

                        # Continue work automatically
                        # Use neutral phrasing - the task context already shows what's in-progress
                        # Avoid saying "next task" which encourages skipping incomplete work
                        continue_response = await self._two_stage_chat(
                            "Continue.",
                            context={"active_skill": routing_result.skill_name}
                        )

                        # Reset recursion counter on exit
                        self._auto_continue_depth = 0

                        # Combine responses
                        return f"{response}\n\n---\n\n{continue_response}"
                    else:
                        log.warning(f"[AUTO-CONTINUE] Max depth ({max_depth}) reached, stopping.")
                        self._auto_continue_depth = 0

            # Emit state change for dashboard (only if not auto-continuing)
            if dash.is_enabled():
                await dash.agent_state("idle", "Ready", trace.trace_id)

            return response

        except Exception as e:
            trace.mark_error(
                stage="two_stage_chat",
                message=str(e),
                traceback=traceback.format_exc()
            )
            self.telemetry.complete_trace(trace, f"Error: {e}")
            self._current_trace = None
            log.error(f"Two-stage chat error: {e}", exc_info=True)
            return f"I encountered an error: {e}"

    async def chat(self, user_input: str) -> str:
        """
        Process user input and generate response.

        By default, uses the new Fabric-style two-stage routing:
        1. Router prompt classifies intent → skill name
        2. Skill executor runs with constrained tools → LLM extracts args

        Set use_legacy_routing=True in __init__ to use the old system.
        """
        # === USE TWO-STAGE ROUTING (DEFAULT) ===
        if not self.use_legacy_routing:
            return await self._two_stage_chat(user_input)

        # === LEGACY ROUTING BELOW ===
        # (Only used if use_legacy_routing=True)

        # === START TRACE ===
        trace = self.telemetry.start_trace(user_input)
        self._current_trace = trace
        trace.model_name = self.model
        trace.voice_mode = self.voice_mode
        trace.constructs_enabled = self.construct_manager is not None

        log.info(f"=" * 50)
        log.info(f"TRACE {trace.trace_id[:8]} | USER INPUT: {user_input}")
        log.info(f"=" * 50)

        try:
            # === STEP 0: AUTO-INJECT CONTEXT ===
            enriched_input = user_input
            context_injected = False

            # Determine if we should inject context and why
            should_inject = self._should_inject_context(user_input)
            trace.context_injection_attempted = should_inject

            if not should_inject:
                trace.context_should_inject_reason = "No dev keywords detected"
            else:
                trace.context_should_inject_reason = "Dev keywords matched"

            if self.context_manager and should_inject:
                try:
                    # Get current development context (Phase 3: returns assembled context dict)
                    context_data = self.context_manager.get_context()

                    # Trace: capture context manager data
                    trace.context_manager_used = True
                    if context_data.get('critical', {}).get('active_files'):
                        trace.active_files = [f['path'] for f in context_data['critical']['active_files']]
                    if context_data.get('critical', {}).get('recent_changes'):
                        trace.recent_changes = context_data['critical']['recent_changes']
                    if context_data.get('critical', {}).get('detected_workflow'):
                        wf = context_data['critical']['detected_workflow']
                        trace.detected_workflow = wf.get('type')
                        trace.detected_workflow_confidence = float(wf.get('confidence', '0%').replace('%', '')) / 100

                    # Format context for LLM
                    context_str = self.context_manager.format_context_for_llm(context_data)

                    if context_str:
                        # Trace: capture context source
                        trace.add_context_source(
                            source_type="context_manager",
                            content=context_str
                        )

                        # Inject context into the input
                        enriched_input = f"{user_input}\n\n[Current Context:\n{context_str}]"
                        context_injected = True
                        trace.context_injection_succeeded = True
                        trace.context_assembled = context_str
                        trace.context_assembled_length = len(context_str)

                        log.info(f"CONTEXT INJECTED ({len(context_str)} chars)")
                        log.debug(f"Context:\n{context_str}")

                except Exception as e:
                    log.warning(f"Context injection failed: {e}")
                    trace.context_should_inject_reason += f" | Error: {e}"
                    # Continue with original input if context fails

            trace.user_input_enriched = enriched_input
            trace.user_input_enriched_length = len(enriched_input)

            # === STEP 1: SkillRegistry-based intent routing ===
            # This is the primary routing mechanism using patterns from SKILL.md files
            trace.intent_detection_attempted = True
            intent_start = time.time()

            # Build runtime context for intent routing
            import os
            routing_context = {
                "working_directory": os.getcwd(),
                "active_project": self._get_active_project(),
            }

            # === ROUTING STRATEGY ===
            # 1. Try SEMANTIC routing FIRST (embedding similarity) - most reliable
            # 2. Fall back to regex/keyword if semantic fails or is disabled
            # 3. The groq model ONLY picks tools - it doesn't reason

            skill_match = None
            semantic_match = None

            # Step 1: Try semantic routing FIRST (PRIMARY method)
            if self.semantic_routing_enabled:
                semantic_match = await self._try_semantic_routing(
                    enriched_input, routing_context, trace
                )

                if semantic_match and semantic_match.get('skill_name'):
                    # Semantic routing succeeded - create a skill_match from it
                    skill_info = self.tools.skills.get(semantic_match['skill_name'])
                    if skill_info:
                        from skill_registry import IntentMatch
                        skill_match = IntentMatch(
                            skill=skill_info,
                            skill_name=semantic_match['skill_name'],
                            matched_pattern=f"semantic:{semantic_match.get('matched_utterance', '')}",
                            confidence=semantic_match['confidence'],
                            match_type="semantic"
                        )
                        log.info(f"SEMANTIC ROUTING (PRIMARY): {skill_match.skill_name}")
                        log.info(f"  Matched: {semantic_match.get('matched_utterance', '')[:50]}...")
                        log.info(f"  Confidence: {semantic_match['confidence']:.2f}")
                        log.info(f"  Method: {semantic_match['method']}")

                        # === AUTO-CREATE TASKS FOR SEMANTIC MATCHES ===
                        # Create tasks BEFORE direct execution so they show in dashboard
                        if self.task_manager and not self.task_manager.has_tasks():
                            auto_tasks = self._generate_tasks_for_request(
                                user_input,
                                skill_match.skill_name
                            )
                            if auto_tasks:
                                self.task_manager.write_tasks(auto_tasks, original_request=user_input)
                                log.info(f"Auto-created {len(auto_tasks)} tasks for semantic match")

                        # === DIRECT TOOL EXECUTION FOR SEMANTIC MATCHES ===
                        # If we have a good semantic match to a specific utterance,
                        # we can infer the correct tool and execute directly (bypassing LLM)
                        # Threshold: 0.60 - balances precision vs. letting LLM fail
                        if semantic_match['confidence'] >= 0.60:
                            direct_tool = self._infer_tool_from_semantic_match(
                                semantic_match,
                                skill_info,
                                user_input
                            )
                            if direct_tool:
                                log.info(f"DIRECT SEMANTIC EXECUTION: {direct_tool['tool']}")
                                result = await self._execute_tool_traced(direct_tool, direct_call=True)
                                trace.response_raw = result

                                response = await self._format_tool_result_response(
                                    user_input, direct_tool, result
                                )
                                self.telemetry.complete_trace(trace, response)
                                self._current_trace = None
                                self._current_matched_skill = None
                                return response

            # Step 2: Fall back to regex/keyword ONLY if semantic didn't match
            if not skill_match and hasattr(self.tools, 'route_by_intent'):
                skill_match = self.tools.route_by_intent(enriched_input, routing_context)
                if skill_match and skill_match.matched:
                    log.info(f"REGEX/KEYWORD ROUTING (FALLBACK): {skill_match}")

            # Record routing result
            if skill_match and skill_match.matched:
                trace.intent_detection_time_ms = int((time.time() - intent_start) * 1000)
                trace.intent_detected = True
                trace.intent_skill_matched = skill_match.skill_name
                trace.intent_pattern_matched = skill_match.matched_pattern or ""
                trace.intent_match_type = skill_match.match_type
                trace.intent_confidence = skill_match.confidence

                log.info(f"SKILL ROUTING: {skill_match}")
                log.info(f"  Pattern: {skill_match.matched_pattern}")
                log.info(f"  Confidence: {skill_match.confidence:.2f}")

                # Emit dashboard event - skill matched
                if dash.is_enabled():
                    await dash.skill_matched(
                        skill_match.skill_name,
                        f"Pattern: {skill_match.matched_pattern}, Confidence: {skill_match.confidence:.2f}"
                    )

                # === AUTO-CREATE TASK LIST FOR NON-TRIVIAL REQUESTS ===
                # Even without an explicit workflow, create tasks for complex work
                # This frames the conversation productively
                if self.task_manager and not self.task_manager.has_tasks():
                    auto_tasks = self._generate_tasks_for_request(
                        user_input,
                        skill_match.skill_name
                    )
                    if auto_tasks:
                        self.task_manager.write_tasks(auto_tasks, original_request=user_input)
                        log.info(f"Auto-created {len(auto_tasks)} tasks for {skill_match.skill_name} request")

                        if dash.is_enabled():
                            await dash.emit_event("task_list_updated", {
                                "task_count": len(auto_tasks),
                                "skill": skill_match.skill_name
                            })

                # Handle workflow matches
                if skill_match.workflow_name:
                    log.info(f"WORKFLOW DETECTED: {skill_match.workflow_name}")
                    trace.intent_workflow = skill_match.workflow_name

                    # Emit dashboard event - workflow started
                    if dash.is_enabled():
                        await dash.workflow_started(
                            skill_match.workflow_name,
                            skill_match.skill_name,
                            skill_match.matched_pattern
                        )

                    # Load workflow content for LLM guidance
                    workflow = self.tools.get_workflow(skill_match.workflow_name)
                    if workflow:
                        trace.workflow_injected = True
                        trace.workflow_content = workflow.steps[:500]

                        # === AUTO-CREATE TASK LIST FROM WORKFLOW ===
                        # This is critical - gives the groq model clear guidance
                        if self.task_manager:
                            workflow_tasks = self._parse_workflow_to_tasks(workflow)
                            if workflow_tasks:
                                self.task_manager.write_tasks(
                                    workflow_tasks,
                                    original_request=user_input
                                )
                                log.info(f"Auto-created {len(workflow_tasks)} tasks from workflow")

                                # Emit dashboard update
                                if dash.is_enabled():
                                    await dash.emit_event("task_list_updated", {
                                        "task_count": len(workflow_tasks),
                                        "workflow": skill_match.workflow_name
                                    })

                        # For high-confidence workflows, try to execute first step directly
                        if skill_match.confidence >= 0.9:
                            first_tool = self._extract_first_tool_from_workflow(
                                workflow,
                                user_input=user_input,
                                extracted_args=skill_match.extracted_args
                            )
                            if first_tool:
                                log.info(f"DIRECT WORKFLOW EXECUTION: {first_tool['tool']}")
                                result = await self._execute_tool_traced(first_tool, direct_call=True)
                                trace.response_raw = result

                                # Format the response
                                response = await self._format_tool_result_response(
                                    user_input, first_tool, result
                                )
                                self.telemetry.complete_trace(trace, response)
                                self._current_trace = None
                                self._current_matched_skill = None
                                return response

                        # Otherwise, add workflow to context for LLM
                        workflow_guidance = f"\n\n[WORKFLOW: {workflow.name}]\n{workflow.steps}"
                        enriched_input = enriched_input + workflow_guidance
                        log.debug(f"Injected workflow guidance ({len(workflow.steps)} chars)")

                # If we have extracted args and a skill, we might be able to
                # resolve the query to a specific tool call
                if hasattr(skill_match, 'extracted_args') and skill_match.extracted_args:
                    log.info(f"  Extracted args: {skill_match.extracted_args}")
                    trace.intent_args_extracted = skill_match.extracted_args

                    # Enrich the query with resolved context
                    enriched_input = self.tools.resolve_query_context(
                        enriched_input, routing_context
                    )
                    trace.user_input_enriched = enriched_input
                    log.debug(f"Query enriched with resolved context")

            trace.intent_detection_time_ms = int((time.time() - intent_start) * 1000)

            # === STEP 2: Proceed to LLM with matched skill context ===
            if skill_match and skill_match.matched:
                log.info(f"Proceeding with skill: {skill_match.skill_name}")
            else:
                log.info("No skill matched, LLM will decide...")
            if context_injected:
                log.debug("LLM will receive context-enriched input")

            # Build context from memory
            context = self._build_context(enriched_input)

            # Trace: capture memory context
            if context.get('user_profile'):
                trace.user_profile_loaded = context['user_profile']
                trace.add_context_source(
                    source_type="user_profile",
                    content=context['user_profile']
                )
            if context.get('relevant_memories'):
                trace.memory_results = context['relevant_memories']
                trace.memory_results_count = len(context['relevant_memories'])
                trace.memory_search_query = enriched_input[:100]
                for i, mem in enumerate(context['relevant_memories'][:3]):
                    trace.add_context_source(
                        source_type=f"memory_result_{i}",
                        content=mem
                    )
            if context.get('recent_messages'):
                trace.recent_messages_count = len(context['recent_messages'])

            log.debug(f"Context: profile={len(context.get('user_profile') or '')} chars, " +
                    f"memories={len(context.get('relevant_memories', []))}, " +
                    f"recent={len(context.get('recent_messages', []))}")

            # Build messages for LLM with matched skill for contextual tool injection
            matched_skill_name = skill_match.skill_name if (skill_match and skill_match.matched) else None
            self._current_matched_skill = matched_skill_name  # Store for native tool filtering
            messages = self._build_messages(enriched_input, context, matched_skill_name)
            log.debug(f"Built {len(messages)} messages for LLM (skill={matched_skill_name})")

            # Generate response with tools (traced internally)
            response = await self._generate_with_tools_traced(messages)
            log.info(f"FINAL RESPONSE: {response[:200]}...")

            # Complete trace and cleanup
            self.telemetry.complete_trace(trace, response)
            self._current_trace = None
            self._current_matched_skill = None  # Reset for next chat
            return response

        except Exception as e:
            # Trace: capture error
            trace.mark_error(
                stage="chat",
                message=str(e),
                traceback=traceback.format_exc()
            )
            self.telemetry.complete_trace(trace, f"Error: {e}")
            self._current_trace = None
            self._current_matched_skill = None  # Reset for next chat
            log.error(f"Chat error: {e}", exc_info=True)
            return f"I encountered an error: {e}"
    
    async def _format_tool_result_response(
        self,
        user_input: str,
        tool_call: Dict,
        result: str
    ) -> str:
        """Ask LLM to format tool result into a nice response"""

        # For errors, return directly without LLM formatting
        if result.startswith("Error:"):
            return result

        # Tools that should return their full results directly (no summarization)
        # These are "display" tools where the user explicitly asked to see the content
        direct_return_tools = {
            "show_research_platform",  # Full research platform document
            "read_file",               # File contents
            "show_file",               # File contents (construct version)
            "list_directory",          # Directory listing
            "show_directory",          # Directory listing (construct version)
            "recall",                  # Memory recall results
            "list_notes",              # Notes listing
        }

        if tool_call.get("tool") in direct_return_tools:
            return result

        # For short results, return directly
        if len(result) < 500:
            return result

        prompt = f"""The user asked: "{user_input}"

I executed the '{tool_call['tool']}' tool with args {json.dumps(tool_call['args'])}.

Result:
```
{result[:3000]}
```

Provide a brief, helpful summary of this result. Be concise (2-3 sentences max)."""

        messages = [
            {"role": "system", "content": "Summarize tool results concisely. Don't repeat the full content."},
            {"role": "user", "content": prompt}
        ]

        # Call without tools for simple formatting
        response = await self._call_ollama(messages)
        content = response.get("content", "") if isinstance(response, dict) else response
        return content if content and not content.startswith("Error") else result

    async def _try_semantic_routing(
        self,
        query: str,
        context: Dict,
        trace: "Trace"
    ) -> Optional[Dict]:
        """
        Try semantic routing using embedding similarity.

        Phase 4 feature: Uses pre-computed skill embeddings to find
        semantically similar intents when pattern matching fails.

        Args:
            query: User's query (possibly enriched with context)
            context: Routing context (working_directory, active_project, etc.)
            trace: Current telemetry trace

        Returns:
            Dict with skill_name, confidence, method or None
        """
        router = get_semantic_router()
        if not router:
            return None

        try:
            # Initialize router if needed (lazy loading)
            if not self._semantic_router_initialized:
                await router.initialize()
                self._semantic_router_initialized = True

            # Get available skill names from registry
            available_skills = list(self.tools.skills.keys()) if hasattr(self.tools, 'skills') else None

            # Route the query
            semantic_start = time.time()
            match = await router.route(
                query,
                context=context,
                available_skills=available_skills
            )
            semantic_time_ms = int((time.time() - semantic_start) * 1000)

            # Record in trace
            trace.semantic_routing_attempted = True
            trace.semantic_routing_time_ms = semantic_time_ms

            if match.matched:
                return {
                    "skill_name": match.skill_name,
                    "confidence": match.confidence,
                    "method": match.method,
                    "matched_utterance": match.matched_utterance,
                    "similarity_score": match.similarity_score,
                }

            return None

        except Exception as e:
            log.warning(f"Semantic routing failed: {e}")
            trace.semantic_routing_error = str(e)
            return None

    def _build_context(self, user_input: str) -> dict:
        """Build context for LLM from memory and conversation history"""
        context = {}
        
        # User profile
        profile = self.memory.get_user_profile()
        if profile:
            context["user_profile"] = profile
        
        # Relevant memories
        memories = self.memory.search_memories(user_input, k=3)
        if memories:
            context["relevant_memories"] = memories
        
        # Recent conversation
        recent = self.memory.get_recent_messages(10)
        if recent:
            context["recent_messages"] = recent
        
        return context
    
    def _build_messages(self, user_input: str, context: dict, matched_skill: Optional[str] = None) -> List[Dict]:
        """Build message list for Ollama"""
        messages = []

        # System message with tools (contextual based on matched skill)
        system_msg = self._build_system_message(context, matched_skill)
        messages.append({"role": "system", "content": system_msg})
        
        # Recent conversation history
        recent = context.get("recent_messages", [])
        if recent:
            # Add last few exchanges
            for msg in recent[-6:]:  # Last 3 exchanges
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Current user message
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    def _build_system_message(self, context: dict, matched_skill: Optional[str] = None) -> str:
        """
        Build system message using PAI-style pattern templates.
        Selects voice or text variant based on voice_mode.
        Integrates both automatic (ContextManager) and personal (Telos) context.

        Args:
            context: Context dict with user_profile, recent_messages, etc.
            matched_skill: If a skill was matched via routing, inject its tools
        """
        # Build skill-specific tools context (if skill matched) or empty string
        # This keeps the prompt focused and reduces cognitive load on the model
        skill_tools = self._build_skill_tools(matched_skill)

        # Get voice style primitive
        if self.voice_mode:
            voice_style = PromptPrimitives.voice(
                mode="concise",
                context="User is in voice mode, working hands-free"
            )
        else:
            voice_style = ""  # Text mode doesn't need voice guidelines

        # === TELOS: Extract user_profile for template variable ===
        # Note: Full Telos context is injected via SESSION_START hooks (hydrate_telos_context)
        # Here we only extract user_profile for the template and capture telemetry
        user_profile = ""

        if self.telos_manager:
            try:
                telos_obj = self.telos_manager.load_context()
                if telos_obj.profile:
                    user_profile = telos_obj.profile

                # === TELEMETRY: Capture Telos context ===
                if self._current_trace:
                    trace = self._current_trace
                    active_project = self.telos_manager.auto_detect_project()
                    telos_formatted = self.telos_manager.format_for_llm(active_project=active_project)

                    trace.telos_loaded = True
                    trace.telos_active_project = active_project
                    trace.telos_project_context = telos_formatted
                    telos_path = str(self.telos_manager.telos_dir)

                    if telos_obj.profile:
                        trace.telos_profile = telos_obj.profile
                        trace.add_context_source(
                            source_type="telos_profile",
                            content=telos_obj.profile,
                            source_path=f"{telos_path}/profile.md"
                        )
                    if telos_obj.goals:
                        trace.telos_goals = telos_obj.goals
                        trace.add_context_source(
                            source_type="telos_goals",
                            content=telos_obj.goals,
                            source_path=f"{telos_path}/goals.md"
                        )
                    trace.add_context_source(
                        source_type="telos_formatted",
                        content=telos_formatted,
                        source_path=telos_path
                    )

            except Exception as e:
                log.warning(f"Failed to load Telos context for telemetry: {e}")

        # Fallback user profile if no Telos
        if not user_profile:
            user_profile = context.get("user_profile", "No user profile available yet.")

        # === PHASE 3 AUTOMATIC CONTEXT ===
        # Format project context (from ContextManager or memory)
        project_context = context.get("project_context", "")
        if not project_context and self.context_manager:
            # Try to get from context manager if available
            project_context = "Active project context available via context intelligence."

        # Format recent context
        recent_messages = context.get("recent_messages", [])
        recent_context = self._format_recent_context(recent_messages)

        # Format relevant memories
        memories = context.get("relevant_memories", [])
        if memories:
            memory_lines = ["Relevant memories:"]
            for m in memories[:3]:
                memory_lines.append(f"- {m[:100]}...")
            project_context = project_context + "\n\n" + "\n".join(memory_lines)

        # === ACTIVE RESEARCH CONTEXT ===
        # If there's active research, inject a condensed summary
        research_context = self._get_active_research_context()
        if research_context:
            project_context = project_context + "\n\n" + research_context
            log.debug(f"Injected active research context ({len(research_context)} chars)")

            # Telemetry: capture research context
            if self._current_trace:
                self._current_trace.add_context_source(
                    source_type="active_research",
                    content=research_context
                )

        # === TASK MANAGEMENT TELEMETRY ===
        # Note: Task context is injected via SESSION_START hooks (hydrate_active_tasks)
        # Here we only capture telemetry for tracing
        if self.task_manager and self.task_manager.has_tasks():
            task_context = self.task_manager.format_for_context()
            if task_context and self._current_trace:
                self._current_trace.add_context_source(
                    source_type="task_list",
                    content=task_context
                )

        # === HOOK SYSTEM ADDITIONS ===
        # Inject any context additions from SESSION_START hooks
        if self._hook_system_additions:
            project_context = project_context + "\n\n" + self._hook_system_additions
            log.debug(f"Injected hook additions ({len(self._hook_system_additions)} chars)")

        # Load pattern template (voice or text mode)
        mode = "voice" if self.voice_mode else "text"

        # Get working directory for context
        import os
        working_directory = os.getcwd()

        try:
            system_prompt = self.pattern_loader.load_pattern(
                "base/system",
                mode=mode,
                skill_tools=skill_tools,
                voice_style=voice_style,
                user_profile=user_profile,
                project_context=project_context,
                recent_context=recent_context,
                working_directory=working_directory
            )
        except FileNotFoundError as e:
            # Fallback to basic prompt if pattern files don't exist yet
            log.warning(f"Pattern template not found: {e}. Using fallback prompt.")
            system_prompt = self._build_fallback_system_message(
                skill_tools, user_profile, project_context, recent_context
            )

        return system_prompt

    def _build_fallback_system_message(
        self,
        skill_tools: str,
        user_profile: str,
        project_context: str,
        recent_context: str
    ) -> str:
        """Fallback system message if pattern templates aren't available"""
        parts = [
            "You are Workshop, a helpful local AI assistant with access to tools.",
            "",
        ]

        if skill_tools:
            parts.append(skill_tools)
            parts.append("")

        parts.extend([
            "Use tools via native function calling. Never fabricate information.",
            "",
            f"User profile: {user_profile}",
        ])

        if project_context:
            parts.append(f"\nProject context: {project_context}")

        if recent_context:
            parts.append(f"\nRecent context:\n{recent_context}")

        return "\n".join(parts)

    def _get_active_research_context(self) -> str:
        """
        Get condensed active research context for system message injection.

        Returns a ~1500 char summary of the current research platform,
        including topic, key findings, and source titles - enough for the
        LLM to answer questions without overwhelming the context window.

        This loads directly from disk (~/.workshop/research/_active.json)
        rather than relying on in-memory singleton, so it works even after
        Workshop restarts.
        """
        try:
            from pathlib import Path
            import json

            # Load directly from the research directory
            research_dir = Path.home() / ".workshop" / "research"
            active_file = research_dir / "_active.json"

            if not active_file.exists():
                log.debug("No active research file found")
                return ""

            # Read the active research pointer
            with open(active_file, 'r') as f:
                active_data = json.load(f)

            platform_path = Path(active_data.get("active_platform", ""))
            if not platform_path.exists():
                log.debug(f"Active research platform not found: {platform_path}")
                return ""

            # Load the platform data
            with open(platform_path, 'r') as f:
                platform_data = json.load(f)

            # Format a condensed context summary
            return self._format_research_context(platform_data)

        except Exception as e:
            log.debug(f"No active research context: {e}")

        return ""

    def _format_research_context(self, platform_data: dict) -> str:
        """Format research platform data into a condensed context summary."""
        lines = []

        topic = platform_data.get("topic", "Unknown Topic")
        original_query = platform_data.get("original_query", "")
        sources = platform_data.get("sources", [])

        lines.append(f"## Active Research: {topic}")
        if original_query:
            lines.append(f"Query: {original_query}")
        lines.append(f"Sources: {len(sources)} analyzed")
        lines.append("")

        # Collect key findings from all sources, scored by importance
        scored_points = []
        for source in sources:
            key_points = source.get("key_points", [])
            source_title = source.get("title", "Unknown")
            relevance = source.get("relevance_score", 0.5) or 0.5

            for i, point in enumerate(key_points):
                if point and len(point) > 20:
                    # Score: earlier points + longer points + higher relevance
                    position_score = 1.0 / (i + 1)
                    length_score = min(len(point) / 100, 1.0)
                    score = position_score * length_score * relevance
                    scored_points.append((score, point, source_title))

        # Sort by score and deduplicate
        scored_points.sort(reverse=True, key=lambda x: x[0])

        seen_content = set()
        unique_points = []
        for score, point, source_title in scored_points:
            point_words = set(point.lower().split())
            is_duplicate = any(
                len(point_words & seen) / max(len(point_words), 1) > 0.6
                for seen in seen_content
            )
            if not is_duplicate:
                seen_content.add(frozenset(point_words))
                unique_points.append((point, source_title))

        # Add top findings to context
        lines.append("### Key Findings:")
        for point, source_title in unique_points[:15]:  # Top 15 unique findings
            lines.append(f"- {point}")

        # Add source list
        lines.append("")
        lines.append("### Sources:")
        for source in sources[:8]:  # Top 8 sources
            title = source.get("title", "Unknown")
            url = source.get("url", "")
            lines.append(f"- {title}")

        return "\n".join(lines)

    def _format_recent_context(self, messages: list, max_lines: int = 10) -> str:
        """Format recent messages for context"""
        lines = []
        for msg in messages[-max_lines:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _build_minimal_tools_roster(self) -> str:
        """
        Build minimal tools reference for native tool calling.
        Since Ollama provides full tool definitions, we just need a quick reference.
        """
        sections = []

        if hasattr(self.tools, 'skills'):
            for skill_name, skill in self.tools.skills.items():
                tool_names = [name for name in skill.tools.keys()]
                if tool_names:
                    sections.append(f"- **{skill_name}**: {', '.join(tool_names)}")

        return "## Available Tools\n\n" + "\n".join(sections) + "\n\n*Use tools via function calling. Don't make up information.*"

    def _build_skill_tools(self, skill_name: Optional[str] = None) -> str:
        """
        Build contextual tool documentation for the matched skill.

        When a skill is matched, we inject detailed tool info for that skill only.
        This keeps the prompt focused and reduces cognitive load on the model.

        Args:
            skill_name: The matched skill name, or None for minimal overview

        Returns:
            Formatted tool documentation for the matched skill
        """
        if not skill_name or not hasattr(self.tools, 'skills'):
            # No skill matched - return minimal guidance
            return ""

        skill = self.tools.skills.get(skill_name)
        if not skill:
            return ""

        lines = [f"## Active Skill: {skill_name}"]

        # Add skill purpose if available
        purpose = skill.routing_info.get('purpose', '')
        if purpose:
            lines.append(f"*{purpose}*\n")

        # List available tools with signatures
        lines.append("### Available Tools")
        for tool_name, tool_info in skill.tools.items():
            lines.append(f"- `{tool_info.signature}` - {tool_info.description}")

        # Add workflow hints if any
        if skill.workflows:
            lines.append("\n### Workflows")
            for wf_name, wf_info in skill.workflows.items():
                trigger = wf_info.trigger_pattern if hasattr(wf_info, 'trigger_pattern') else ""
                lines.append(f"- **{wf_name}**: {trigger}")

        lines.append("\n*Use these tools via function calling. Never fabricate results.*")

        return "\n".join(lines)

    def _get_skill_tool_names(self, skill_name: Optional[str] = None) -> set:
        """
        Get the set of tool names for a specific skill.

        Args:
            skill_name: The skill name, or None for all tools

        Returns:
            Set of tool names belonging to this skill
        """
        if not skill_name or not hasattr(self.tools, 'skills'):
            return set()  # Return empty set, will use all tools

        skill = self.tools.skills.get(skill_name)
        if not skill:
            return set()

        return set(skill.tools.keys())

    def _build_enhanced_tools_roster(self) -> str:
        """
        Build comprehensive tools documentation for the system prompt.
        Groups tools by skill with purpose, signatures, and usage examples.

        This gives the model enough context to know:
        1. WHAT tools are available
        2. WHEN to use each tool
        3. HOW to call each tool (exact JSON format)
        """
        sections = []

        # Check if we have a SkillRegistry (new architecture) vs old ToolRegistry
        if hasattr(self.tools, 'skills'):
            # New SkillRegistry - group by skill with rich context
            for skill_name, skill in self.tools.skills.items():
                skill_section = []

                # Skill header with purpose
                purpose = skill.routing_info.get('purpose', '')
                skill_section.append(f"### {skill_name}")
                if purpose:
                    skill_section.append(f"*{purpose}*\n")

                # Workflows (multi-step procedures) - NEW
                if hasattr(skill, 'workflows') and skill.workflows:
                    skill_section.append("**Workflows (for multi-step tasks):**")
                    for wf_name, wf_info in skill.workflows.items():
                        triggers_preview = ", ".join(wf_info.triggers[:2]) if wf_info.triggers else ""
                        skill_section.append(f"- **{wf_name}**: {wf_info.purpose}")
                        if triggers_preview:
                            skill_section.append(f"  *Triggers:* {triggers_preview}")
                    skill_section.append("")  # Blank line before tools

                # Tool signatures
                skill_section.append("**Tools:**")
                for tool_name, tool_info in skill.tools.items():
                    skill_section.append(f"- `{tool_info.signature}`: {tool_info.description}")

                    # Add usage examples if available
                    if tool_info.examples:
                        for example in tool_info.examples[:2]:  # Max 2 examples
                            skill_section.append(f"  Example: `{example}`")

                sections.append("\n".join(skill_section))
        else:
            # Fallback for old ToolRegistry format
            tool_items = []
            for name, tool_info in self.tools.get_all_tools().items():
                sig = tool_info.get("signature", name)
                desc = tool_info.get("description", "")
                tool_items.append(f"- `{sig}`: {desc}")
            sections.append("\n".join(tool_items))

        return "## Available Tools & Workflows\n\n" + "\n\n".join(sections)

    async def _generate_with_tools(self, messages: List[Dict]) -> str:
        """Generate response with native tool calling support"""
        max_iterations = 5
        iteration = 0
        tool_results = []

        # Build native tools for Ollama
        native_tools = self._build_native_tools()

        while iteration < max_iterations:
            iteration += 1
            log.debug(f"Generation iteration {iteration}/{max_iterations}")

            # Call Ollama with native tools
            response = await self._call_ollama(messages, tools=native_tools)
            content = response.get("content", "")
            native_tool_calls = response.get("tool_calls", [])

            if content.startswith("Error:"):
                return content

            # Check for native tool calls
            if native_tool_calls:
                log.debug(f"Found {len(native_tool_calls)} native tool calls")

                # Add assistant message
                messages.append({"role": "assistant", "content": content, "tool_calls": native_tool_calls})

                # Execute each tool
                for tc in native_tool_calls:
                    func_info = tc.get("function", {})
                    tool_name = func_info.get("name", "")
                    tool_args = func_info.get("arguments", {})

                    call = {"tool": tool_name, "args": tool_args}
                    result = await self._execute_tool(call)
                    tool_results.append({"tool": tool_name, "result": result})

                    # Add tool result
                    messages.append({"role": "tool", "content": str(result)})

                continue

            # Fallback: Check for text-based tool calls
            text_tool_calls = self._extract_tool_calls(content)
            if text_tool_calls:
                log.debug(f"Found {len(text_tool_calls)} text tool calls")
                for call in text_tool_calls:
                    result = await self._execute_tool(call)
                    tool_results.append({"tool": call["tool"], "result": result})

                results_text = self._format_tool_results(tool_results[-len(text_tool_calls):])
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": results_text})
                continue

            # No tools - clean and return
            clean = self._clean_response(content)

            if tool_results:
                return f"{clean}\n\n(Based on: {', '.join([r['tool'] for r in tool_results])})"

            return clean

        return "I ran out of iterations trying to complete your request."
    
    async def _call_ollama(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Make API call to Ollama with native tool calling support.

        Args:
            messages: Conversation messages
            tools: Optional list of tools in Ollama native format

        Returns:
            Dict with 'content' (str) and optionally 'tool_calls' (list)
        """
        url = f"{self.ollama_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }

        # Add tools if provided (native function calling)
        if tools:
            payload["tools"] = tools
            # Log first message (system prompt) for debugging
            if messages and messages[0].get("role") == "system":
                log.debug(f"System prompt length: {len(messages[0].get('content', ''))}")
                log.debug(f"System prompt preview: {messages[0].get('content', '')[:500]}")
            # Log number of tools being sent
            log.debug(f"Sending {len(tools)} tools to Ollama")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        log.error(f"Ollama error: {resp.status} - {error_text}")
                        return {"content": f"Error: Ollama returned {resp.status}"}

                    data = await resp.json()
                    message = data.get("message", {})

                    result = {
                        "content": message.get("content", ""),
                        "tool_calls": message.get("tool_calls", [])
                    }

                    # Debug: Log raw message from Ollama
                    log.debug(f"Raw Ollama message: {json.dumps(message, default=str)[:1000]}")

                    # Log tool calls if present
                    if result["tool_calls"]:
                        log.info(f"Ollama returned {len(result['tool_calls'])} tool calls")
                        for tc in result["tool_calls"]:
                            func = tc.get("function", {})
                            log.debug(f"  Tool call: {func.get('name')} with {func.get('arguments')}")
                    else:
                        log.debug(f"No tool_calls in response. Content: {result['content'][:200]}")

                    return result

        except aiohttp.ClientError as e:
            log.error(f"Connection error: {e}")
            return {"content": "Error: Could not connect to Ollama. Is it running?"}
        except Exception as e:
            log.error(f"Unexpected error: {e}", exc_info=True)
            return {"content": f"Error: {e}"}
    
    def _extract_tool_calls(self, response: str) -> List[Dict]:
        """Extract tool calls from LLM response"""
        tool_calls = []
        
        # Strategy 1: JSON in code blocks
        for match in re.finditer(r'```(?:json)?\s*(\{[^`]+\})\s*```', response, re.DOTALL):
            try:
                data = json.loads(match.group(1))
                if "tool" in data:
                    if data not in tool_calls:
                        tool_calls.append(data)
                        log.debug(f"Extracted (code block): {data}")
            except json.JSONDecodeError:
                pass
        
        # Strategy 2: Standalone JSON objects
        for match in re.finditer(r'\{[^{}]*"tool"[^{}]*\}', response):
            try:
                call = json.loads(match.group(0))
                if call not in tool_calls:
                    tool_calls.append(call)
                    log.debug(f"Extracted (standalone): {call}")
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: XML-style tags
        for match in re.finditer(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL):
            try:
                call = json.loads(match.group(1))
                if call not in tool_calls:
                    tool_calls.append(call)
                    log.debug(f"Extracted (code block): {call}")
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Raw JSON
        for match in re.finditer(r'\{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"args"\s*:\s*(\{[^}]+\})\s*\}', response):
            try:
                args = json.loads(match.group(2))
                call = {"tool": match.group(1), "args": args}
                if call not in tool_calls:
                    tool_calls.append(call)
                    log.debug(f"Extracted (raw): {call}")
            except:
                pass
        
        return tool_calls
    
    async def _execute_tool(self, call: dict) -> str:
        """Execute a tool call"""
        tool_name = call.get("tool")
        args = call.get("args", {})
        
        if not tool_name:
            return "Error: No tool specified"
        
        try:
            result = await self.tools.execute(tool_name, args)
            return str(result)
        except Exception as e:
            log.error(f"Tool error: {e}", exc_info=True)
            return f"Error: {e}"
    
    def _format_tool_results(self, results: list) -> str:
        """Format tool results for the next turn"""
        lines = ["Tool results:\n"]
        for r in results:
            lines.append(f"**{r['tool']}** result:\n```\n{r['result']}\n```\n")
        lines.append("Use these results to answer the user.")
        return "\n".join(lines)

    # === TELEMETRY-ENABLED METHODS ===

    async def _execute_tool_traced(self, call: dict, direct_call: bool = False) -> str:
        """Execute a tool call with full telemetry tracing and dashboard events"""
        tool_name = call.get("tool")
        args = call.get("args", {})

        if not tool_name:
            return "Error: No tool specified"

        # Start tool trace
        trace = self._current_trace
        tool_trace = None
        skill_name = "unknown"

        if trace:
            # Get skill name
            tool_info = self.tools.get_tool(tool_name)
            skill_name = tool_info.skill_name if tool_info else "unknown"

            tool_trace = trace.start_tool_call(tool_name, skill_name, args)
            tool_trace.was_direct_call = direct_call
            tool_trace.pattern_matched = call.get('pattern', '')
            tool_trace.dependencies_available = list(self.tools.dependencies.keys()) if hasattr(self.tools, 'dependencies') else []

        # Emit dashboard event - tool calling
        call_id = None
        start_time = time.time()
        if dash.is_enabled():
            call_id = await dash.tool_calling(tool_name, skill_name, args)

        try:
            # Execute tool
            result = await self.tools.execute(tool_name, args)
            result_str = str(result)

            # Complete trace
            if tool_trace:
                tool_trace.args_normalized = args  # Will be updated by registry
                tool_trace.complete(result_str)

            # Emit dashboard event - tool result
            if dash.is_enabled() and call_id:
                duration_ms = int((time.time() - start_time) * 1000)
                await dash.tool_result(call_id, result_str, duration_ms)

            # === UPDATE TASK STATE ===
            # When a tool completes, advance to the next task in the workflow
            if self.task_manager and self.task_manager.has_tasks():
                self._advance_task_on_tool_completion(tool_name)

            return result_str

        except Exception as e:
            error_msg = str(e)
            tb = traceback.format_exc()

            if tool_trace:
                tool_trace.complete("", error=error_msg)
                tool_trace.error_traceback = tb

            # Emit dashboard event - tool error
            if dash.is_enabled() and call_id:
                await dash.tool_error(call_id, error_msg)

            log.error(f"Tool error: {e}", exc_info=True)
            return f"Error: {e}"

    async def _generate_with_tools_traced(self, messages: List[Dict]) -> str:
        """
        Generate response with NATIVE Ollama tool calling and full telemetry.

        This uses Ollama's native function calling API instead of prompt-based
        tool extraction. The model returns structured tool_calls that we execute
        and feed back as tool role messages.
        """
        max_iterations = 5
        iteration = 0
        tool_results = []
        trace = self._current_trace

        # Build native tools for Ollama
        native_tools = self._build_native_tools()
        log.info(f"Loaded {len(native_tools)} tools for native function calling")

        # IMPORTANT: Enhance (don't replace) system prompt for native tool calling
        # The full system prompt contains tools_roster, task context, and user profile
        # We prepend critical tool usage guidance while preserving all context
        if messages and messages[0].get("role") == "system":
            import os
            working_dir = os.getcwd()
            original_prompt = messages[0]["content"]

            tool_guidance = (
                "## CRITICAL TOOL USAGE RULES\n\n"
                "You have access to tools via native function calling. Follow these rules:\n\n"
                "1. **ALWAYS use tools** to answer questions - NEVER make up information or URLs\n"
                "2. **Research tasks**: Use web_search for quick lookups, deep_research for comprehensive research\n"
                "3. **URL content**: Use fetch_url to retrieve actual content - don't summarize URLs you haven't fetched\n"
                "4. **File operations**: Use read_file, list_directory, etc. for file questions\n"
                "5. **Multi-step tasks**: Use task_write FIRST to plan, then update as you progress\n"
                "6. **When in doubt, use a tool** - it's better to call a tool than to guess\n\n"
                "If you don't have information from a tool result, call a tool to get it.\n"
                "If the user asks for content, URLs, or details you don't have, use the appropriate tool.\n\n"
                f"Current working directory: {working_dir}\n\n"
                "---\n\n"
            )

            messages[0]["content"] = tool_guidance + original_prompt
            log.debug(f"Enhanced system prompt with tool guidance ({len(tool_guidance)} chars prepended)")

        while iteration < max_iterations:
            iteration += 1
            log.debug(f"Generation iteration {iteration}/{max_iterations}")

            # Get system prompt from first message for tracing
            system_prompt = ""
            if messages and messages[0].get("role") == "system":
                system_prompt = messages[0].get("content", "")

            # Start LLM call trace
            llm_call = None
            if trace:
                llm_call = trace.start_llm_call(
                    model=self.model,
                    messages=messages,
                    system_prompt=system_prompt,
                    iteration=iteration
                )

            # Emit dashboard event - LLM calling with full message context
            llm_start = time.time()
            if dash.is_enabled():
                await dash.llm_calling(self.model, len(messages), messages=messages, system_prompt=system_prompt)

            # Call Ollama with native tools
            response = await self._call_ollama(messages, tools=native_tools)

            # Extract content and tool_calls from response
            content = response.get("content", "")
            native_tool_calls = response.get("tool_calls", [])

            # Complete LLM trace
            if llm_call:
                llm_call.complete(content)
                llm_call.tool_calls_extracted = native_tool_calls

            # Emit dashboard event - LLM complete
            if dash.is_enabled():
                duration_ms = int((time.time() - llm_start) * 1000)
                await dash.llm_complete(len(content), duration_ms, len(native_tool_calls))

            if content.startswith("Error:"):
                return content

            # Check for native tool calls from Ollama
            if native_tool_calls:
                log.info(f"Model requested {len(native_tool_calls)} tool calls (native)")

                # Add assistant message with tool calls to conversation
                assistant_msg = {"role": "assistant", "content": content}
                if native_tool_calls:
                    assistant_msg["tool_calls"] = native_tool_calls
                messages.append(assistant_msg)

                # Execute each tool and add results
                for tc in native_tool_calls:
                    func_info = tc.get("function", {})
                    tool_name = func_info.get("name", "")
                    tool_args = func_info.get("arguments", {})

                    # Normalize to our internal format
                    call = {"tool": tool_name, "args": tool_args}
                    log.info(f"Executing tool: {tool_name} with args: {tool_args}")

                    # Execute with tracing
                    result = await self._execute_tool_traced(call, direct_call=False)
                    tool_results.append({"tool": tool_name, "result": result})

                    # Add tool result as 'tool' role message (Ollama format)
                    messages.append({
                        "role": "tool",
                        "content": str(result)
                    })

                # Continue loop to let model process tool results
                continue

            # No tool calls - check for prompt-based fallback
            # (Some models may still use text-based tool calls)
            text_tool_calls = self._extract_tool_calls(content)
            if text_tool_calls:
                log.info(f"Found {len(text_tool_calls)} tool calls (text fallback)")

                # Execute tools
                for call in text_tool_calls:
                    result = await self._execute_tool_traced(call, direct_call=False)
                    tool_results.append({"tool": call["tool"], "result": result})

                # Add results to conversation (old format for compatibility)
                results_text = self._format_tool_results(tool_results[-len(text_tool_calls):])
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": results_text})
                continue

            # No tools requested - return final response
            clean = self._clean_response(content)

            # Record raw response in trace
            if trace:
                trace.response_raw = content

            # If we have tool results, incorporate them
            if tool_results:
                return f"{clean}\n\n(Based on: {', '.join([r['tool'] for r in tool_results])})"

            return clean

        return "I ran out of iterations trying to complete your request."
    
    def _clean_response(self, response: str) -> str:
        """Clean up response for user display"""
        response = re.sub(r"<tool_call>.*?</tool_call>", "", response, flags=re.DOTALL)
        response = re.sub(r"<thinking>.*?</thinking>", "", response, flags=re.DOTALL)
        response = re.sub(r'\n{3,}', '\n\n', response)
        return response.strip()

    def _build_native_tools(self) -> List[Dict]:
        """
        Convert Workshop tools to Ollama's native function calling format.

        If a skill was matched (stored in self._current_matched_skill), only
        include tools from that skill to reduce cognitive load on the model.

        Ollama expects:
        {
            "type": "function",
            "function": {
                "name": "tool_name",
                "description": "what it does",
                "parameters": {
                    "type": "object",
                    "required": ["param1"],
                    "properties": {
                        "param1": {"type": "string", "description": "..."}
                    }
                }
            }
        }
        """
        native_tools = []

        # Get tool filter based on matched skill
        skill_tool_names = self._get_skill_tool_names(self._current_matched_skill)

        # Get all tools from registry
        all_tools = self.tools.get_all_tools()

        for tool_name, tool_info in all_tools.items():
            # If skill matched, only include tools from that skill
            if skill_tool_names and tool_name not in skill_tool_names:
                continue

            # Parse signature to extract parameters
            # Signatures look like: "tool_name(param1: str, param2: int = None)"
            signature = tool_info.get("signature", "")
            params = self._parse_tool_signature(signature)

            native_tool = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_info.get("description", f"{tool_name} tool"),
                    "parameters": {
                        "type": "object",
                        "properties": params.get("properties", {}),
                        "required": params.get("required", [])
                    }
                }
            }
            native_tools.append(native_tool)

        if self._current_matched_skill:
            log.debug(f"Built {len(native_tools)} native tools for skill '{self._current_matched_skill}'")
        else:
            log.debug(f"Built {len(native_tools)} native tools (all skills)")
        return native_tools

    def _parse_tool_signature(self, signature: str) -> Dict:
        """
        Parse a tool signature string into parameter schema.

        Input: "read_file(path: str)" or "search_files(query: str, path: str = '.')"
        Output: {
            "properties": {"path": {"type": "string", "description": "path parameter"}},
            "required": ["path"]
        }
        """
        properties = {}
        required = []

        # Extract parameters from signature: func_name(params)
        match = re.search(r'\(([^)]*)\)', signature)
        if not match:
            return {"properties": {}, "required": []}

        params_str = match.group(1).strip()
        if not params_str:
            return {"properties": {}, "required": []}

        # Parse each parameter
        # Handle: "param: type", "param: type = default", "param=default"
        for param_part in params_str.split(','):
            param_part = param_part.strip()
            if not param_part or param_part == '_deps':
                continue

            # Check for default value
            has_default = '=' in param_part

            # Extract name and type
            if ':' in param_part:
                # "param: type" or "param: type = default"
                name_type = param_part.split('=')[0].strip()
                parts = name_type.split(':')
                param_name = parts[0].strip()
                param_type = parts[1].strip() if len(parts) > 1 else "string"
            else:
                # "param=default" (no type annotation)
                param_name = param_part.split('=')[0].strip()
                param_type = "string"

            # Skip internal parameters
            if param_name.startswith('_'):
                continue

            # Map Python types to JSON Schema types
            type_map = {
                "str": "string",
                "int": "integer",
                "float": "number",
                "bool": "boolean",
                "list": "array",
                "dict": "object",
                "List": "array",
                "Dict": "object",
                "Optional": "string",  # Simplified
            }

            # Clean up type string
            json_type = "string"
            for py_type, js_type in type_map.items():
                if py_type in param_type:
                    json_type = js_type
                    break

            properties[param_name] = {
                "type": json_type,
                "description": f"{param_name} parameter"
            }

            # Required if no default
            if not has_default:
                required.append(param_name)

        return {"properties": properties, "required": required}
    
    async def update_user_profile(self, recent_messages: list):
        """Update user profile based on recent conversation"""
        if not recent_messages:
            return

        current_profile = self.memory.get_user_profile() or ""

        prompt = f"""Update user profile based on conversation. Be concise .

Current: {current_profile}

Recent:
{self._format_recent_context(recent_messages)}

Updated profile:"""

        messages = [
            {"role": "system", "content": "Extract key facts about user. Be concise."},
            {"role": "user", "content": prompt}
        ]

        # Call without tools for simple generation
        response = await self._call_ollama(messages)
        new_profile = response.get("content", "") if isinstance(response, dict) else response
        if new_profile and len(new_profile) > 20:
            self.memory.set_user_profile(new_profile)
            log.info(f"Profile updated: {len(new_profile)} chars")