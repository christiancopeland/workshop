"""
Workshop Hook System
Provides extension points for skills and components to inject behavior
into Workshop's lifecycle without modifying core code.

Usage:
    from hooks import get_hook_manager, HookType, HookContext

    # Register a handler
    async def my_handler(ctx: HookContext) -> HookContext:
        ctx.add_system_context("MySection", "Some content")
        return ctx

    get_hook_manager().register(HookType.SESSION_START, my_handler, priority=50)

Hook Types:
    - SESSION_START: Fires at conversation start, handlers enrich context
    - POST_TOOL_USE: Fires after each tool execution, handlers can react/persist

Priority Ranges:
    - 0-20: Core/system handlers (e.g., telos, tasks)
    - 21-50: Standard handlers
    - 51-80: Enhancement handlers
    - 81-100: Logging/observability handlers
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Awaitable, Tuple, TYPE_CHECKING

# Forward references to avoid circular imports
if TYPE_CHECKING:
    from session_manager import Session

# Use Workshop's logger to ensure logs go to the log file
from logger import get_logger
logger = get_logger("hooks")


# =============================================================================
# HOOK-001: HookType Enum
# =============================================================================

class HookType(Enum):
    """
    Lifecycle events where handlers can be registered.

    Each hook type represents a specific point in Workshop's execution
    where extensions can inject behavior.
    """
    SESSION_START = "session_start"      # Fires at conversation start
    POST_TOOL_USE = "post_tool_use"      # Fires after each tool execution

    # Future hooks (not implemented yet):
    # PRE_TOOL_USE = "pre_tool_use"      # Before tool execution
    # PRE_RESPONSE = "pre_response"       # Before sending response to user
    # SESSION_END = "session_end"         # When session ends


# =============================================================================
# HOOK-002: HookContext Dataclass
# =============================================================================

@dataclass
class HookContext:
    """
    Context passed through hook chain. Handlers can read and modify.

    This dataclass carries state between hooks and allows handlers to:
    - Read session state and user input
    - Add system prompt sections
    - Access tool execution results
    - Signal control flow changes

    Attributes:
        session: The current Workshop session (immutable reference)
        user_input: The user's input that triggered this hook
        system_prompt_additions: List of strings to add to system prompt
        context_documents: Additional context documents to include
        metadata: Arbitrary metadata handlers can use to communicate
        tool_name: For POST_TOOL_USE - name of the tool that executed
        tool_args: For POST_TOOL_USE - arguments passed to the tool
        tool_result: For POST_TOOL_USE - result returned by the tool
        skill_name: For POST_TOOL_USE - skill that owns the tool
        skip_remaining: If True, remaining handlers in chain won't execute
    """

    # Immutable inputs
    session: Optional['Session'] = None
    user_input: Optional[str] = None

    # Mutable state - handlers append to these
    system_prompt_additions: List[str] = field(default_factory=list)
    context_documents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Tool-specific (for POST_TOOL_USE)
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None
    skill_name: Optional[str] = None

    # Control flow
    skip_remaining: bool = False  # Handler can set to stop chain

    def add_system_context(self, label: str, content: str) -> None:
        """
        Helper to add formatted system prompt section.

        Args:
            label: Section heading (e.g., "User Profile", "Active Tasks")
            content: Content to include under the heading
        """
        if content and content.strip():
            self.system_prompt_additions.append(f"## {label}\n{content}")

    def get_combined_system_additions(self) -> str:
        """Combine all additions into single string for system prompt."""
        return "\n\n".join(self.system_prompt_additions)

    def add_context_document(self, doc: str) -> None:
        """Add a context document."""
        if doc and doc.strip():
            self.context_documents.append(doc)

    def get_combined_context_documents(self) -> str:
        """Combine all context documents."""
        return "\n\n---\n\n".join(self.context_documents)


# =============================================================================
# Handler Type Definition
# =============================================================================

# Handler signature: async function taking HookContext, returning HookContext or None
HookHandler = Callable[[HookContext], Awaitable[Optional[HookContext]]]


# =============================================================================
# HOOK-003, HOOK-004: HookManager Class
# =============================================================================

class HookManager:
    """
    Central registry and executor for Workshop hooks.

    Manages registration and execution of hook handlers. Each handler is
    associated with a hook type and priority. When a hook fires, all
    handlers for that type execute in priority order.

    Usage:
        mgr = get_hook_manager()
        mgr.register(HookType.SESSION_START, my_handler, priority=50)

        ctx = HookContext(session=session, user_input="Hello")
        ctx = await mgr.execute(HookType.SESSION_START, ctx)
    """

    def __init__(self):
        """Initialize with empty handler lists for each hook type."""
        self._hooks: Dict[HookType, List[Tuple[int, HookHandler]]] = {
            h: [] for h in HookType
        }
        logger.debug("HookManager initialized")

    def register(
        self,
        hook_type: HookType,
        handler: HookHandler,
        priority: int = 50
    ) -> None:
        """
        Register a handler for a hook type.

        Args:
            hook_type: Which lifecycle event to hook
            handler: Async function taking HookContext, returning modified context or None
            priority: Execution order (lower = earlier). Default 50.
                      Suggested ranges:
                      - 0-20: Core/system handlers
                      - 21-50: Standard handlers
                      - 51-80: Enhancement handlers
                      - 81-100: Logging/observability handlers

        Raises:
            ValueError: If hook_type is not a valid HookType
        """
        if not isinstance(hook_type, HookType):
            raise ValueError(f"Invalid hook type: {hook_type}. Must be a HookType enum value.")

        self._hooks[hook_type].append((priority, handler))
        # Sort by priority (stable sort preserves registration order for same priority)
        self._hooks[hook_type].sort(key=lambda x: x[0])

        handler_name = getattr(handler, '__name__', str(handler))
        logger.debug(f"Registered handler '{handler_name}' for {hook_type.value} at priority {priority}")

    def unregister(
        self,
        hook_type: HookType,
        handler: HookHandler
    ) -> bool:
        """
        Unregister a handler from a hook type.

        Args:
            hook_type: Which lifecycle event
            handler: The handler function to remove

        Returns:
            True if handler was found and removed, False otherwise
        """
        original_length = len(self._hooks[hook_type])
        self._hooks[hook_type] = [
            (p, h) for p, h in self._hooks[hook_type] if h != handler
        ]
        return len(self._hooks[hook_type]) < original_length

    async def execute(
        self,
        hook_type: HookType,
        context: HookContext,
        verbose: bool = False  # Changed default to False for clean UX
    ) -> HookContext:
        """
        Execute all handlers for a hook type.

        Handlers run in priority order (lower first). Each handler receives
        the context and can modify it. If a handler returns None, the
        original context continues. If a handler raises an exception, it's
        logged but the chain continues.

        Args:
            hook_type: Which hook to execute
            context: Initial context to pass through chain
            verbose: If True, show terminal visualization (default False for clean UX)

        Returns:
            Final context after all handlers have run

        Note:
            - Handlers run in priority order (lower first)
            - Handler errors are logged but don't stop the chain
            - Handler can set context.skip_remaining=True to stop early
        """
        handlers = self._hooks.get(hook_type, [])

        if not handlers:
            logger.debug(f"No handlers registered for {hook_type.value}")
            return context

        # Use terminal UI if available, otherwise fall back to print
        term = None
        if verbose:
            try:
                from terminal_ui import get_terminal
                term = get_terminal()
            except ImportError:
                pass

        # Terminal visualization - header
        if verbose:
            tool_name = context.tool_name if hook_type == HookType.POST_TOOL_USE else ""
            if term and hasattr(term.config, 'show_hook_traces') and term.config.show_hook_traces:
                term.hook_start(hook_type.value, len(handlers), tool_name)
            elif term is None:
                # Fallback to print if terminal_ui not available
                hook_icon = "\U0001F517" if hook_type == HookType.SESSION_START else "\U0001F527"
                extra_info = f" [{context.tool_name}]" if tool_name else ""
                print(f"\n{hook_icon} [HOOKS] \u2500\u2500\u2500 {hook_type.value.upper()}{extra_info} \u2500\u2500\u2500 ({len(handlers)} handlers)")

        logger.debug(f"Executing {len(handlers)} handlers for {hook_type.value}")

        additions_before = len(context.system_prompt_additions)

        for priority, handler in handlers:
            handler_name = getattr(handler, '__name__', str(handler))

            # Terminal visualization - handler start
            if verbose and term and term.config.show_hook_traces:
                term.hook_handler(priority, handler_name, "running")
            elif verbose and term is None:
                print(f"   \u251C\u2500 [{priority:02d}] {handler_name}...", end=" ", flush=True)

            # Track metadata keys before handler runs
            metadata_keys_before = set(context.metadata.keys())

            try:
                result = await handler(context)

                # If handler returns a HookContext, use it
                if result is not None:
                    context = result

                # Terminal visualization - handler result
                if verbose and term and term.config.show_hook_traces:
                    term.hook_handler(priority, handler_name, "success")
                elif verbose and term is None:
                    additions_now = len(context.system_prompt_additions)
                    new_additions = additions_now - additions_before
                    additions_before = additions_now

                    # Find newly added metadata keys
                    new_metadata_keys = set(context.metadata.keys()) - metadata_keys_before

                    # Build detailed status message
                    status_parts = []

                    if new_additions > 0:
                        status_parts.append(f"+{new_additions} sections")

                    # Check for specific handler metadata (only newly added)
                    if 'telos_sections' in new_metadata_keys:
                        sections = context.metadata['telos_sections']
                        if sections:
                            status_parts.append(f"[{', '.join(sections)}]")

                    if 'task_counts' in new_metadata_keys:
                        tc = context.metadata['task_counts']
                        status_parts.append(f"[{tc['in_progress']} active, {tc['pending']} pending]")

                    if 'research_indexed' in new_metadata_keys and context.metadata.get('research_indexed'):
                        status_parts.append("indexed to memory")

                    if 'research_index_appended' in new_metadata_keys and context.metadata.get('research_index_appended'):
                        status_parts.append("logged to index")

                    if status_parts:
                        print(f"\u2713 ({', '.join(status_parts)})")
                    else:
                        print("\u2713 (no changes)")

                # Check if handler wants to stop the chain
                if context.skip_remaining:
                    if verbose and term is None:
                        print(f"   \u2514\u2500 (chain stopped by {handler_name})")
                    logger.debug(f"Hook chain stopped early by '{handler_name}' at priority {priority}")
                    break

            except Exception as e:
                # Terminal visualization - error
                if verbose and term and term.config.show_hook_traces:
                    term.hook_handler(priority, handler_name, "error")
                elif verbose and term is None:
                    print(f"\u2717 (error: {str(e)[:50]})")
                logger.error(
                    f"Hook handler '{handler_name}' failed for {hook_type.value}: {e}",
                    exc_info=True
                )
                # Continue with unchanged context

        # Terminal visualization - summary
        if verbose and context.system_prompt_additions:
            total_chars = len(context.get_combined_system_additions())
            if term and term.config.show_hook_traces:
                term.hook_summary(len(context.system_prompt_additions), total_chars)
            elif term is None:
                print(f"   \u2514\u2500 Total: {len(context.system_prompt_additions)} sections, {total_chars} chars injected")

        return context

    def get_handler_count(self, hook_type: HookType) -> int:
        """Get number of registered handlers for a hook type."""
        return len(self._hooks.get(hook_type, []))

    def get_all_handlers(self, hook_type: HookType) -> List[Tuple[int, str]]:
        """
        Get list of (priority, handler_name) for a hook type.
        Useful for debugging and introspection.
        """
        return [
            (priority, getattr(handler, '__name__', str(handler)))
            for priority, handler in self._hooks.get(hook_type, [])
        ]

    def clear_handlers(self, hook_type: HookType = None) -> None:
        """
        Clear handlers. If hook_type is None, clears all handlers.
        Primarily useful for testing.
        """
        if hook_type is None:
            for h in HookType:
                self._hooks[h] = []
        else:
            self._hooks[hook_type] = []


# =============================================================================
# HOOK-005: Singleton Pattern
# =============================================================================

_hook_manager_instance: Optional[HookManager] = None


def get_hook_manager() -> HookManager:
    """
    Get the global HookManager singleton.

    Returns:
        Shared HookManager instance

    Usage:
        from hooks import get_hook_manager, HookType

        mgr = get_hook_manager()
        mgr.register(HookType.SESSION_START, my_handler)
    """
    global _hook_manager_instance
    if _hook_manager_instance is None:
        _hook_manager_instance = HookManager()
    return _hook_manager_instance


def reset_hook_manager() -> None:
    """
    Reset the singleton (for testing only).

    WARNING: This clears all registered handlers. Only use in tests.
    """
    global _hook_manager_instance
    _hook_manager_instance = None


# =============================================================================
# Built-in Handler Registration
# =============================================================================

def register_builtin_handlers() -> None:
    """
    Register Workshop's built-in handlers.

    This is called during Workshop initialization to set up the default
    handlers for context hydration and research persistence.

    Built-in handlers:
    - Capabilities hydration (SESSION_START, priority 5) - what tools are available
    - Telos hydration (SESSION_START, priority 10) - user profile, goals, mission
    - Task hydration (SESSION_START, priority 20) - current task list
    - Research auto-persist (POST_TOOL_USE, priority 50) - auto-index research findings
    """
    mgr = get_hook_manager()

    # Import and register built-in handlers
    # These are defined below to keep everything in one file
    # Priority order: capabilities -> telos -> tasks (so agent knows tools before context)
    mgr.register(HookType.SESSION_START, hydrate_capabilities, priority=5)
    mgr.register(HookType.SESSION_START, hydrate_telos_context, priority=10)
    mgr.register(HookType.SESSION_START, hydrate_active_tasks, priority=20)
    mgr.register(HookType.POST_TOOL_USE, auto_persist_research, priority=50)

    logger.info("Built-in hook handlers registered")


# =============================================================================
# HOOK-008: Telos Hydration Handler
# =============================================================================

async def hydrate_telos_context(ctx: HookContext) -> HookContext:
    """
    Inject user profile, goals, and active projects into context.

    Priority: 10 (early - other handlers may need user context)

    This handler reads from TelosManager and adds formatted context
    sections for profile, goals, mission, and active project.
    """
    try:
        from telos_manager import get_telos_manager

        telos = get_telos_manager()
        telos_ctx = telos.load_context()

        if telos_ctx.is_empty():
            logger.debug("No Telos context available")
            return ctx

        sections_added = []

        if telos_ctx.profile:
            ctx.add_system_context("User Profile", telos_ctx.profile)
            sections_added.append("profile")

        if telos_ctx.goals:
            ctx.add_system_context("Current Goals", telos_ctx.goals)
            sections_added.append("goals")

        if telos_ctx.mission:
            ctx.add_system_context("Mission", telos_ctx.mission)
            sections_added.append("mission")

        # Active project context
        if telos_ctx.project_context:
            # Get the first/active project
            for project_name, project_ctx in telos_ctx.project_context.items():
                if project_ctx:
                    ctx.add_system_context(f"Active Project: {project_name}", project_ctx)
                    sections_added.append(f"project:{project_name}")
                    break  # Only include first project for now

        # Track what was added for verbose output
        ctx.metadata['telos_sections'] = sections_added
        logger.debug(f"Telos context hydrated: {sections_added}")

    except ImportError:
        logger.debug("TelosManager not available, skipping telos hydration")
    except Exception as e:
        logger.warning(f"Failed to hydrate telos context: {e}")

    return ctx


# =============================================================================
# HOOK-009: Task Hydration Handler
# =============================================================================

async def hydrate_active_tasks(ctx: HookContext) -> HookContext:
    """
    Inject current task list into context.

    Priority: 20 (after telos, before memory)

    This handler reads from TaskManager and formats the task list
    as a markdown checklist showing in-progress and pending tasks.
    """
    try:
        from task_manager import get_task_manager, TaskStatus

        task_mgr = get_task_manager()

        if not task_mgr.has_tasks():
            logger.debug("No tasks to hydrate")
            return ctx

        tasks = task_mgr.get_tasks()

        # Separate by status
        in_progress = [t for t in tasks if t.status == TaskStatus.IN_PROGRESS]
        pending = [t for t in tasks if t.status == TaskStatus.PENDING]

        if not in_progress and not pending:
            return ctx

        task_lines = []

        if in_progress:
            task_lines.append("**Currently Working On:**")
            for task in in_progress:
                task_lines.append(f"- [ ] {task.content}")

        if pending:
            if task_lines:
                task_lines.append("")  # Blank line separator
            task_lines.append("**Pending Tasks:**")
            for task in pending:
                task_lines.append(f"- [ ] {task.content}")

        ctx.add_system_context("Active Tasks", "\n".join(task_lines))

        # Track what was added for verbose output
        ctx.metadata['task_counts'] = {
            'in_progress': len(in_progress),
            'pending': len(pending)
        }
        logger.debug(f"Task context hydrated: {len(in_progress)} in-progress, {len(pending)} pending")

    except ImportError:
        logger.debug("TaskManager not available, skipping task hydration")
    except Exception as e:
        logger.warning(f"Failed to hydrate task context: {e}")

    return ctx


# =============================================================================
# HOOK-010: Research Auto-Persist Handler
# =============================================================================

# Tools that produce research content worth indexing
RESEARCH_TOOLS = {
    'web_search',
    'deep_research',
    'fetch_url',
    'fetch_urls_for_research'
}

async def auto_persist_research(ctx: HookContext) -> HookContext:
    """
    Auto-index research findings to long-term memory.

    Priority: 50 (standard handler)

    This handler watches for research-related tools and automatically
    indexes their results to ChromaDB for semantic search. Also appends
    to a research index file for quick reference.
    """
    # Only handle research tools
    if ctx.tool_name not in RESEARCH_TOOLS:
        return ctx

    if not ctx.tool_result:
        logger.debug(f"No result from {ctx.tool_name}, skipping indexing")
        return ctx

    try:
        from datetime import datetime
        from pathlib import Path
        import json

        # Extract indexable content from result
        content = _extract_indexable_content(ctx.tool_result)
        if not content:
            return ctx

        # Build metadata
        metadata = {
            'tool': ctx.tool_name,
            'skill': ctx.skill_name,
            'timestamp': datetime.now().isoformat(),
            'query': ctx.tool_args.get('query', '') if ctx.tool_args else ''
        }

        # Try to index to memory if available
        try:
            # Get memory from session or try to import
            memory = None
            if ctx.session and hasattr(ctx.session, 'memory'):
                memory = ctx.session.memory

            if memory is None:
                # Try to get from main workshop
                from memory import get_memory
                memory = get_memory()

            if memory and hasattr(memory, 'add_to_chromadb'):
                await memory.add_to_chromadb(
                    content=content[:5000],
                    metadata=metadata,
                    collection='research_findings'
                )
                ctx.metadata['research_indexed'] = True
                logger.debug(f"Indexed {ctx.tool_name} result to ChromaDB")

        except ImportError:
            logger.debug("Memory system not available for research indexing")
        except Exception as e:
            logger.warning(f"Failed to index research to memory: {e}")

        # Append to research index file
        try:
            _append_to_research_index(ctx, content)
            ctx.metadata['research_index_appended'] = True
        except Exception as e:
            logger.warning(f"Failed to append to research index: {e}")

    except Exception as e:
        logger.warning(f"Research auto-persist failed: {e}")

    return ctx


def _extract_indexable_content(result: Any) -> str:
    """
    Extract text content from various result formats.

    Handles strings, dicts with common keys, and lists.
    Truncates to 5000 chars to avoid huge embeddings.
    """
    import json

    if isinstance(result, str):
        return result[:5000]

    if isinstance(result, dict):
        # Handle common result formats
        if 'content' in result:
            return str(result['content'])[:5000]
        if 'results' in result:
            return json.dumps(result['results'], indent=2)[:5000]
        if 'summary' in result:
            return str(result['summary'])[:5000]
        if 'text' in result:
            return str(result['text'])[:5000]
        return json.dumps(result, indent=2)[:5000]

    if isinstance(result, list):
        return json.dumps(result, indent=2)[:5000]

    return str(result)[:5000]


def _append_to_research_index(ctx: HookContext, content: str) -> None:
    """
    Append entry to research index markdown file.

    Creates a human-readable log of all research findings.
    """
    from datetime import datetime
    from pathlib import Path

    index_path = Path.home() / '.workshop' / 'research_index.md'
    index_path.parent.mkdir(parents=True, exist_ok=True)

    query = ctx.tool_args.get('query', 'N/A') if ctx.tool_args else 'N/A'

    entry = f"""
## {datetime.now().strftime('%Y-%m-%d %H:%M')} - {ctx.tool_name}

**Query:** {query}

**Summary:** {content[:500]}{'...' if len(content) > 500 else ''}

---
"""

    with open(index_path, 'a', encoding='utf-8') as f:
        f.write(entry)


# =============================================================================
# HOOK-011: Capabilities Hydration Handler
# =============================================================================

# Cache for capabilities summary to avoid regenerating every session
_capabilities_cache: Optional[str] = None
_capabilities_cache_time: Optional[float] = None
CAPABILITIES_CACHE_TTL = 300  # 5 minutes


async def hydrate_capabilities(ctx: HookContext) -> HookContext:
    """
    Inject capabilities summary into context.

    Priority: 5 (very early - before telos, so agent knows what it can do)

    This handler generates a condensed capabilities manifest and injects it
    into the system prompt so the agent knows what tools and skills are
    available.
    """
    global _capabilities_cache, _capabilities_cache_time

    try:
        import time
        import os
        from pathlib import Path

        current_time = time.time()

        # Check if we have a valid cache
        if (_capabilities_cache is not None and
            _capabilities_cache_time is not None and
            current_time - _capabilities_cache_time < CAPABILITIES_CACHE_TTL):
            # Use cached version
            ctx.add_system_context("Available Capabilities", _capabilities_cache)
            ctx.metadata['capabilities_cached'] = True
            logger.debug("Using cached capabilities summary")
            return ctx

        # Generate fresh capabilities
        try:
            from skill_registry import SkillRegistry

            # Find skills directory - check multiple locations
            skills_dir = None
            candidates = [
                # 1. Relative to current working directory (project directory)
                Path.cwd() / ".workshop" / "Skills",
                # 2. Relative to this file's location
                Path(__file__).parent / ".workshop" / "Skills",
                # 3. Home directory
                Path.home() / ".workshop" / "Skills",
            ]

            for candidate in candidates:
                if candidate.exists():
                    skills_dir = candidate
                    logger.debug(f"Found skills directory at: {skills_dir}")
                    break

            if not skills_dir:
                logger.debug("Skills directory not found, skipping capabilities hydration")
                return ctx

            # Create registry instance to generate manifest
            # Note: This is a lightweight operation since skills are already loaded
            registry = SkillRegistry(skills_dir)

            # Generate full manifest (writes to file)
            registry.generate_capabilities_manifest(write_to_file=True)

            # Get condensed summary for context injection
            summary = registry.get_capabilities_summary()

            # Update cache
            _capabilities_cache = summary
            _capabilities_cache_time = current_time

            # Inject into context
            ctx.add_system_context("Available Capabilities", summary)

            # Track metadata
            ctx.metadata['capabilities_skills'] = len(registry.skills)
            ctx.metadata['capabilities_tools'] = len(registry.list_all_tools())
            ctx.metadata['capabilities_generated'] = True

            logger.debug(
                f"Capabilities manifest generated: "
                f"{len(registry.skills)} skills, {len(registry.list_all_tools())} tools"
            )

        except ImportError as e:
            logger.debug(f"SkillRegistry not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to generate capabilities manifest: {e}")

    except Exception as e:
        logger.warning(f"Failed to hydrate capabilities: {e}")

    return ctx


def invalidate_capabilities_cache() -> None:
    """
    Invalidate the capabilities cache.

    Call this when skills are added, removed, or modified to force
    regeneration of the capabilities manifest on next session start.
    """
    global _capabilities_cache, _capabilities_cache_time
    _capabilities_cache = None
    _capabilities_cache_time = None
    logger.debug("Capabilities cache invalidated")


# =============================================================================
# HOOK-012: Voice Progress Update Handler
# =============================================================================

# Tools that warrant spoken progress updates (long-running operations)
VOICE_UPDATE_TOOLS = {
    'deep_research': "researching",
    'fetch_url': "fetching the page",
    'fetch_urls_for_research': "fetching pages",
    'web_search': "searching the web",
    'read_file': "reading the file",
    'write_file': "writing the file",
    'search_files': "searching files",
    'run_shell': "running the command",
}

# Shared state for voice updates
_voice_update_enabled = False
_piper_tts = None


def enable_voice_progress_updates(piper_tts_instance=None):
    """
    Enable voice progress updates during tool execution.

    Call this during voice mode initialization to enable spoken updates.

    Args:
        piper_tts_instance: The PiperStreamingTTS instance to use for speech
    """
    global _voice_update_enabled, _piper_tts
    _voice_update_enabled = True
    _piper_tts = piper_tts_instance
    logger.info("Voice progress updates enabled")


def disable_voice_progress_updates():
    """Disable voice progress updates (for text mode)."""
    global _voice_update_enabled, _piper_tts
    _voice_update_enabled = False
    _piper_tts = None
    logger.debug("Voice progress updates disabled")


def _extract_tool_specifics(tool_name: str, tool_args: dict, tool_result: str) -> dict:
    """
    Extract specific, informative details from tool call for voice updates.

    Returns dict with 'action' (what we did) and 'detail' (specific info).
    """
    args = tool_args or {}
    result = str(tool_result) if tool_result else ""

    if tool_name == 'deep_research':
        topic = args.get('query', args.get('topic', ''))[:60]
        # Extract source count if available
        source_count = ""
        if 'sources' in result.lower():
            import re
            match = re.search(r'(\d+)\s*sources?', result.lower())
            if match:
                source_count = f" from {match.group(1)} sources"
        return {
            'action': f"researching {topic}" if topic else "doing research",
            'detail': f"completed{source_count}" if source_count else "research complete"
        }

    elif tool_name == 'fetch_url':
        url = args.get('url', '')
        # Extract domain from URL for brevity
        if url:
            import re
            match = re.search(r'https?://(?:www\.)?([^/]+)', url)
            domain = match.group(1) if match else url[:30]
            # Check result for content size
            char_match = re.search(r'(\d+)\s*chars?', result.lower()) if result else None
            if char_match:
                return {
                    'action': f"fetching {domain}",
                    'detail': f"got the page, about {int(int(char_match.group(1))/1000)}k characters"
                }
            return {'action': f"fetching {domain}", 'detail': "page loaded"}
        return {'action': "fetching the page", 'detail': "done"}

    elif tool_name == 'fetch_urls_for_research':
        urls = args.get('urls', [])
        count = len(urls) if isinstance(urls, list) else 1
        return {
            'action': f"fetching {count} pages",
            'detail': f"got content from {count} sources"
        }

    elif tool_name == 'web_search':
        query = args.get('query', '')[:50]
        # Try to extract result count
        if result:
            import re
            match = re.search(r'(\d+)\s*results?', result.lower())
            count = match.group(1) if match else "several"
        else:
            count = "some"
        return {
            'action': f"searching for {query}" if query else "searching the web",
            'detail': f"found {count} results"
        }

    elif tool_name == 'read_file':
        path = args.get('path', args.get('file_path', ''))
        # Get just the filename
        filename = path.split('/')[-1] if path else "the file"
        lines = ""
        if result:
            import re
            match = re.search(r'(\d+)\s*lines?', result.lower())
            if match:
                lines = f", {match.group(1)} lines"
        return {
            'action': f"reading {filename}",
            'detail': f"got the contents{lines}"
        }

    elif tool_name == 'write_file':
        path = args.get('path', args.get('file_path', ''))
        filename = path.split('/')[-1] if path else "the file"
        return {
            'action': f"saving {filename}",
            'detail': "file saved"
        }

    elif tool_name == 'search_files':
        query = args.get('query', args.get('pattern', ''))[:40]
        # Try to get match count
        if result:
            import re
            match = re.search(r'(\d+)\s*(?:matches?|results?|files?)', result.lower())
            count = match.group(1) if match else "several"
        else:
            count = "some"
        return {
            'action': f"searching files for {query}" if query else "searching files",
            'detail': f"found {count} matches"
        }

    elif tool_name == 'run_shell':
        cmd = args.get('command', '')[:30]
        return {
            'action': f"running {cmd}" if cmd else "running command",
            'detail': "command finished"
        }

    # Default fallback
    return {
        'action': VOICE_UPDATE_TOOLS.get(tool_name, "working"),
        'detail': "done"
    }


async def voice_progress_update(ctx: HookContext) -> HookContext:
    """
    Generate and speak a brief, INFORMATIVE progress update using Claude Haiku.

    Priority: 30 (before research persistence, after core handlers)

    This handler:
    1. Checks if we're in voice mode and the tool warrants an update
    2. Extracts specific details from the tool call (query, URL, filename, etc.)
    3. Uses Claude Haiku to generate a natural spoken update
    4. Speaks it via Piper TTS

    The update should tell the user WHAT just happened with SPECIFICS,
    so they can stay informed while working on something else.
    """
    global _voice_update_enabled, _piper_tts

    # Skip if voice mode not enabled
    if not _voice_update_enabled:
        return ctx

    # Skip if tool doesn't warrant an update
    if ctx.tool_name not in VOICE_UPDATE_TOOLS:
        return ctx

    # Skip if no result (tool failed or still running)
    if not ctx.tool_result:
        return ctx

    try:
        from claude_bridge import get_claude_bridge

        bridge = get_claude_bridge(timeout_seconds=15)  # Short timeout for quick updates

        # Extract specific details from the tool call
        specifics = _extract_tool_specifics(ctx.tool_name, ctx.tool_args, ctx.tool_result)

        # === TASK CONTEXT: What are we working on? ===
        current_task = ""
        try:
            from task_manager import get_task_manager
            task_mgr = get_task_manager()
            if task_mgr.has_tasks():
                tasks = task_mgr.read_tasks_raw()
                for t in tasks:
                    if t.get('status') == 'in_progress':
                        current_task = t.get('content', '')[:60]
                        break
        except Exception:
            pass

        task_context = f"Current task: {current_task}" if current_task else ""

        prompt = f"""Generate a brief spoken progress update for a voice assistant. The user is working hands-free and needs to stay informed about what's happening.

WHAT JUST HAPPENED:
- Action: {specifics['action']}
- Result: {specifics['detail']}
{f"- {task_context}" if task_context else ""}

YOUR JOB: Write ONE natural sentence that tells the user what just completed. Include the SPECIFIC details (what was searched, what file was read, how many results found). This goes directly to text-to-speech.

CRITICAL RULES:
1. ONE sentence, 10-20 words
2. INCLUDE the specific details (topic, filename, result count)
3. NO markdown symbols: no asterisks, backticks, underscores, hashtags, or colons
4. NO lists, NO formatting
5. Write EXACTLY as a person would say it out loud

GOOD examples (natural, informative, specific):
- "Searched for intelligence brief formats and found eight relevant results."
- "Finished reading the config file, it's about two hundred lines."
- "Grabbed the Wikipedia page on OSINT, got the full content."
- "Saved your research to the notes file."
- "Ran the build command, it completed successfully."
- "Fetched three web pages about machine learning basics."
- "Deep research on API authentication is done, pulled from six sources."

BAD examples (vague, formatted, robotic):
- "Done." (too vague - what was done?)
- "Research complete." (doesn't say what was researched)
- "**Completed** fetching `url`" (markdown will be read aloud)
- "Got it, the file is saved." (which file?)
- "Task done: web_search" (robotic, technical)

Your spoken update (one specific, natural sentence):"""

        messages = [{"role": "user", "content": prompt}]

        result = await bridge.query(
            messages,
            system_prompt="You generate brief spoken progress updates. Your output goes DIRECTLY to text-to-speech with no processing. Write exactly as a person would say it. Include specific details about what was done. No markdown, no formatting, no symbols.",
            model="haiku",
            disable_native_tools=True,
            max_turns=1
        )

        update_text = result.get("content", "").strip()

        # Aggressive cleanup of any formatting that slipped through
        update_text = update_text.replace("*", "").replace("`", "").replace("#", "")
        update_text = update_text.replace("_", " ").replace("**", "")
        update_text = update_text.strip('"\'')
        # Remove common prefixes
        for prefix in ["Here's ", "Here is ", "Update: ", "Status: ", "Result: "]:
            if update_text.startswith(prefix):
                update_text = update_text[len(prefix):]

        # Capitalize first letter after cleanup
        if update_text:
            update_text = update_text[0].upper() + update_text[1:]

        if update_text and len(update_text) > 10 and len(update_text) < 150:
            # Speak the update
            if _piper_tts:
                logger.info(f"Voice update: {update_text}")
                try:
                    await _piper_tts.speak(update_text)
                    ctx.metadata['voice_update_spoken'] = update_text
                except Exception as e:
                    logger.warning(f"Failed to speak voice update: {e}")
            else:
                # Log if no TTS available
                logger.info(f"Voice update (no TTS): {update_text}")

    except ImportError:
        logger.debug("Claude bridge not available for voice updates")
    except Exception as e:
        logger.warning(f"Voice progress update failed: {e}")

    return ctx


def register_voice_progress_handler():
    """
    Register the voice progress update handler.

    Call this during voice mode initialization after enabling voice updates.
    """
    mgr = get_hook_manager()
    mgr.register(HookType.POST_TOOL_USE, voice_progress_update, priority=30)
    logger.info("Voice progress update handler registered")
