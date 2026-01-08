"""
Unit tests for the Workshop Hook System

Tests cover:
- HOOK-001: HookType enum
- HOOK-002: HookContext dataclass
- HOOK-003: HookManager.register()
- HOOK-004: HookManager.execute()
- HOOK-005: Singleton pattern
- HOOK-008: Telos hydration handler
- HOOK-009: Task hydration handler
- HOOK-010: Research persistence handler
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass
from datetime import datetime

# Import the hook system
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hooks import (
    HookType,
    HookContext,
    HookManager,
    get_hook_manager,
    reset_hook_manager,
    hydrate_telos_context,
    hydrate_active_tasks,
    auto_persist_research,
    RESEARCH_TOOLS,
)


# =============================================================================
# HOOK-001: HookType Enum Tests
# =============================================================================

class TestHookType:
    """Tests for HOOK-001: HookType enum"""

    def test_hook_type_values(self):
        """Enum values should be human-readable strings"""
        assert HookType.SESSION_START.value == "session_start"
        assert HookType.POST_TOOL_USE.value == "post_tool_use"

    def test_hook_type_iteration(self):
        """Should be able to iterate over all hook types"""
        types = list(HookType)
        assert len(types) >= 2
        assert HookType.SESSION_START in types
        assert HookType.POST_TOOL_USE in types

    def test_hook_type_string_conversion(self):
        """Should convert to string for logging"""
        assert "session_start" in str(HookType.SESSION_START)


# =============================================================================
# HOOK-002: HookContext Dataclass Tests
# =============================================================================

class TestHookContext:
    """Tests for HOOK-002: HookContext dataclass"""

    def test_context_creation_minimal(self):
        """Create context with minimal args"""
        ctx = HookContext()
        assert ctx.session is None
        assert ctx.user_input is None
        assert ctx.system_prompt_additions == []
        assert ctx.tool_name is None
        assert ctx.skip_remaining is False

    def test_context_creation_with_session(self):
        """Create context with mock session"""
        mock_session = Mock()
        ctx = HookContext(session=mock_session, user_input="Hello")
        assert ctx.session == mock_session
        assert ctx.user_input == "Hello"

    def test_context_add_system_context(self):
        """add_system_context should format sections properly"""
        ctx = HookContext()
        ctx.add_system_context("Profile", "Christian, software engineer")

        assert len(ctx.system_prompt_additions) == 1
        assert "## Profile" in ctx.system_prompt_additions[0]
        assert "Christian" in ctx.system_prompt_additions[0]

    def test_context_add_multiple_sections(self):
        """Should support multiple system context sections"""
        ctx = HookContext()
        ctx.add_system_context("Profile", "User info")
        ctx.add_system_context("Goals", "Build Workshop")

        assert len(ctx.system_prompt_additions) == 2
        combined = ctx.get_combined_system_additions()
        assert "## Profile" in combined
        assert "## Goals" in combined

    def test_context_empty_content_ignored(self):
        """Empty content should not be added"""
        ctx = HookContext()
        ctx.add_system_context("Empty", "")
        ctx.add_system_context("Whitespace", "   ")

        assert len(ctx.system_prompt_additions) == 0

    def test_context_tool_fields(self):
        """Tool-specific fields should work for POST_TOOL_USE"""
        ctx = HookContext(
            tool_name="web_search",
            tool_args={"query": "test"},
            tool_result={"results": []},
            skill_name="Research"
        )
        assert ctx.tool_name == "web_search"
        assert ctx.tool_args == {"query": "test"}
        assert ctx.tool_result == {"results": []}
        assert ctx.skill_name == "Research"

    def test_context_combined_empty(self):
        """Combined additions should be empty string when no additions"""
        ctx = HookContext()
        assert ctx.get_combined_system_additions() == ""


# =============================================================================
# HOOK-003: HookManager.register() Tests
# =============================================================================

class TestHookManagerRegister:
    """Tests for HOOK-003: HookManager.register()"""

    def test_register_single_handler(self):
        """Register a single handler"""
        mgr = HookManager()

        async def handler(ctx):
            return ctx

        mgr.register(HookType.SESSION_START, handler)
        assert mgr.get_handler_count(HookType.SESSION_START) == 1

    def test_register_multiple_handlers(self):
        """Register multiple handlers for same hook"""
        mgr = HookManager()

        async def handler_a(ctx):
            return ctx

        async def handler_b(ctx):
            return ctx

        mgr.register(HookType.SESSION_START, handler_a, priority=60)
        mgr.register(HookType.SESSION_START, handler_b, priority=40)

        assert mgr.get_handler_count(HookType.SESSION_START) == 2

    def test_register_priority_ordering(self):
        """Handlers should be sorted by priority"""
        mgr = HookManager()

        async def handler_a(ctx):
            return ctx

        async def handler_b(ctx):
            return ctx

        mgr.register(HookType.SESSION_START, handler_a, priority=60)
        mgr.register(HookType.SESSION_START, handler_b, priority=40)

        handlers = mgr.get_all_handlers(HookType.SESSION_START)
        # handler_b (40) should be first
        assert handlers[0][0] == 40
        assert handlers[1][0] == 60

    def test_register_invalid_hook_type(self):
        """Should raise ValueError for invalid hook type"""
        mgr = HookManager()

        async def handler(ctx):
            return ctx

        with pytest.raises(ValueError):
            mgr.register("invalid_type", handler)

    def test_unregister_handler(self):
        """Should be able to unregister a handler"""
        mgr = HookManager()

        async def handler(ctx):
            return ctx

        mgr.register(HookType.SESSION_START, handler)
        assert mgr.get_handler_count(HookType.SESSION_START) == 1

        result = mgr.unregister(HookType.SESSION_START, handler)
        assert result is True
        assert mgr.get_handler_count(HookType.SESSION_START) == 0

    def test_unregister_nonexistent_handler(self):
        """Unregistering nonexistent handler should return False"""
        mgr = HookManager()

        async def handler(ctx):
            return ctx

        result = mgr.unregister(HookType.SESSION_START, handler)
        assert result is False


# =============================================================================
# HOOK-004: HookManager.execute() Tests
# =============================================================================

class TestHookManagerExecute:
    """Tests for HOOK-004: HookManager.execute()"""

    @pytest.mark.asyncio
    async def test_execute_single_handler(self):
        """Execute single handler"""
        mgr = HookManager()

        async def handler(ctx):
            ctx.add_system_context("Test", "from handler")
            return ctx

        mgr.register(HookType.SESSION_START, handler)

        ctx = HookContext()
        result = await mgr.execute(HookType.SESSION_START, ctx)

        assert len(result.system_prompt_additions) == 1
        assert "from handler" in result.get_combined_system_additions()

    @pytest.mark.asyncio
    async def test_execute_chain(self):
        """Execute chain of handlers in priority order"""
        mgr = HookManager()
        execution_order = []

        async def handler_a(ctx):
            execution_order.append("A")
            ctx.add_system_context("A", "from A")
            return ctx

        async def handler_b(ctx):
            execution_order.append("B")
            ctx.add_system_context("B", "from B")
            return ctx

        mgr.register(HookType.SESSION_START, handler_a, priority=20)
        mgr.register(HookType.SESSION_START, handler_b, priority=10)

        ctx = HookContext()
        result = await mgr.execute(HookType.SESSION_START, ctx)

        # B (priority 10) should run before A (priority 20)
        assert execution_order == ["B", "A"]
        assert len(result.system_prompt_additions) == 2

    @pytest.mark.asyncio
    async def test_execute_handler_returns_none(self):
        """Handler returning None should preserve context"""
        mgr = HookManager()

        async def handler_modifies(ctx):
            ctx.metadata["modified"] = True
            return ctx

        async def handler_returns_none(ctx):
            # Modify but don't return
            ctx.metadata["also_modified"] = True
            return None

        mgr.register(HookType.SESSION_START, handler_modifies, priority=10)
        mgr.register(HookType.SESSION_START, handler_returns_none, priority=20)

        ctx = HookContext()
        result = await mgr.execute(HookType.SESSION_START, ctx)

        # Both modifications should be present
        assert result.metadata.get("modified") is True
        assert result.metadata.get("also_modified") is True

    @pytest.mark.asyncio
    async def test_execute_error_isolation(self):
        """Handler error should not stop chain"""
        mgr = HookManager()

        async def handler_ok(ctx):
            ctx.metadata["ok"] = True
            return ctx

        async def handler_fail(ctx):
            raise RuntimeError("Handler failed")

        async def handler_after(ctx):
            ctx.metadata["after"] = True
            return ctx

        mgr.register(HookType.SESSION_START, handler_ok, priority=10)
        mgr.register(HookType.SESSION_START, handler_fail, priority=20)
        mgr.register(HookType.SESSION_START, handler_after, priority=30)

        ctx = HookContext()
        result = await mgr.execute(HookType.SESSION_START, ctx)

        # Both ok handlers should have run
        assert result.metadata.get("ok") is True
        assert result.metadata.get("after") is True

    @pytest.mark.asyncio
    async def test_execute_skip_remaining(self):
        """skip_remaining should stop chain"""
        mgr = HookManager()

        async def handler_stop(ctx):
            ctx.metadata["stopped_here"] = True
            ctx.skip_remaining = True
            return ctx

        async def handler_after(ctx):
            ctx.metadata["should_not_run"] = True
            return ctx

        mgr.register(HookType.SESSION_START, handler_stop, priority=10)
        mgr.register(HookType.SESSION_START, handler_after, priority=20)

        ctx = HookContext()
        result = await mgr.execute(HookType.SESSION_START, ctx)

        assert result.metadata.get("stopped_here") is True
        assert result.metadata.get("should_not_run") is None

    @pytest.mark.asyncio
    async def test_execute_no_handlers(self):
        """Execute with no handlers should return unchanged context"""
        mgr = HookManager()

        ctx = HookContext(user_input="test")
        result = await mgr.execute(HookType.SESSION_START, ctx)

        assert result.user_input == "test"


# =============================================================================
# HOOK-005: Singleton Pattern Tests
# =============================================================================

class TestSingleton:
    """Tests for HOOK-005: Singleton pattern"""

    def test_singleton_behavior(self):
        """get_hook_manager should return same instance"""
        reset_hook_manager()

        mgr1 = get_hook_manager()
        mgr2 = get_hook_manager()

        assert mgr1 is mgr2

    def test_reset_singleton(self):
        """reset should create new instance"""
        mgr1 = get_hook_manager()
        reset_hook_manager()
        mgr2 = get_hook_manager()

        assert mgr1 is not mgr2

    def test_handlers_persist_in_singleton(self):
        """Handlers registered via singleton should persist"""
        reset_hook_manager()

        async def handler(ctx):
            return ctx

        get_hook_manager().register(HookType.SESSION_START, handler)
        assert get_hook_manager().get_handler_count(HookType.SESSION_START) == 1


# =============================================================================
# HOOK-008: Telos Hydration Handler Tests
# =============================================================================

class TestTelosHandler:
    """Tests for HOOK-008: Telos hydration handler"""

    @pytest.mark.asyncio
    async def test_telos_handler_with_profile(self):
        """Handler should add profile to context"""
        mock_telos_ctx = Mock()
        mock_telos_ctx.is_empty.return_value = False
        mock_telos_ctx.profile = "Christian, software engineer"
        mock_telos_ctx.goals = ""
        mock_telos_ctx.mission = ""
        mock_telos_ctx.project_context = {}

        mock_manager = Mock()
        mock_manager.get_context.return_value = mock_telos_ctx

        with patch('hooks.get_telos_manager', return_value=mock_manager):
            ctx = HookContext()
            result = await hydrate_telos_context(ctx)

            assert "Christian" in result.get_combined_system_additions()
            assert "## User Profile" in result.get_combined_system_additions()

    @pytest.mark.asyncio
    async def test_telos_handler_with_goals(self):
        """Handler should add goals to context"""
        mock_telos_ctx = Mock()
        mock_telos_ctx.is_empty.return_value = False
        mock_telos_ctx.profile = ""
        mock_telos_ctx.goals = "Build Workshop v2"
        mock_telos_ctx.mission = ""
        mock_telos_ctx.project_context = {}

        mock_manager = Mock()
        mock_manager.get_context.return_value = mock_telos_ctx

        with patch('hooks.get_telos_manager', return_value=mock_manager):
            ctx = HookContext()
            result = await hydrate_telos_context(ctx)

            assert "Build Workshop" in result.get_combined_system_additions()
            assert "## Current Goals" in result.get_combined_system_additions()

    @pytest.mark.asyncio
    async def test_telos_handler_empty_context(self):
        """Handler should not add anything for empty context"""
        mock_telos_ctx = Mock()
        mock_telos_ctx.is_empty.return_value = True

        mock_manager = Mock()
        mock_manager.get_context.return_value = mock_telos_ctx

        with patch('hooks.get_telos_manager', return_value=mock_manager):
            ctx = HookContext()
            result = await hydrate_telos_context(ctx)

            assert len(result.system_prompt_additions) == 0

    @pytest.mark.asyncio
    async def test_telos_handler_import_error(self):
        """Handler should handle missing TelosManager gracefully"""
        with patch.dict('sys.modules', {'telos_manager': None}):
            with patch('hooks.get_telos_manager', side_effect=ImportError):
                ctx = HookContext()
                result = await hydrate_telos_context(ctx)

                # Should not raise, just return unchanged context
                assert result is ctx


# =============================================================================
# HOOK-009: Task Hydration Handler Tests
# =============================================================================

class TestTaskHandler:
    """Tests for HOOK-009: Task hydration handler"""

    @pytest.mark.asyncio
    async def test_task_handler_with_tasks(self):
        """Handler should add tasks to context"""
        from task_manager import TaskStatus

        mock_task1 = Mock()
        mock_task1.content = "Write hook spec"
        mock_task1.status = TaskStatus.IN_PROGRESS

        mock_task2 = Mock()
        mock_task2.content = "Implement hooks"
        mock_task2.status = TaskStatus.PENDING

        mock_manager = Mock()
        mock_manager.has_tasks.return_value = True
        mock_manager.get_tasks.return_value = [mock_task1, mock_task2]

        with patch('hooks.get_task_manager', return_value=mock_manager):
            ctx = HookContext()
            result = await hydrate_active_tasks(ctx)

            combined = result.get_combined_system_additions()
            assert "Write hook spec" in combined
            assert "Implement hooks" in combined
            assert "Currently Working On" in combined
            assert "Pending Tasks" in combined

    @pytest.mark.asyncio
    async def test_task_handler_no_tasks(self):
        """Handler should not add anything when no tasks"""
        mock_manager = Mock()
        mock_manager.has_tasks.return_value = False

        with patch('hooks.get_task_manager', return_value=mock_manager):
            ctx = HookContext()
            result = await hydrate_active_tasks(ctx)

            assert len(result.system_prompt_additions) == 0


# =============================================================================
# HOOK-010: Research Persistence Handler Tests
# =============================================================================

class TestResearchHandler:
    """Tests for HOOK-010: Research persistence handler"""

    def test_research_tools_defined(self):
        """RESEARCH_TOOLS should contain expected tools"""
        assert 'web_search' in RESEARCH_TOOLS
        assert 'deep_research' in RESEARCH_TOOLS
        assert 'fetch_url' in RESEARCH_TOOLS

    @pytest.mark.asyncio
    async def test_research_handler_ignores_non_research(self):
        """Handler should ignore non-research tools"""
        ctx = HookContext(
            tool_name="read_file",
            tool_result="file contents"
        )

        result = await auto_persist_research(ctx)

        # Should not have indexed anything
        assert result.metadata.get('research_indexed') is None

    @pytest.mark.asyncio
    async def test_research_handler_empty_result(self):
        """Handler should handle empty results"""
        ctx = HookContext(
            tool_name="web_search",
            tool_result=None
        )

        result = await auto_persist_research(ctx)

        # Should not have indexed anything
        assert result.metadata.get('research_indexed') is None

    @pytest.mark.asyncio
    async def test_research_handler_indexes_results(self):
        """Handler should index research results"""
        ctx = HookContext(
            tool_name="web_search",
            tool_args={"query": "LiPo battery safety"},
            tool_result={"results": [{"title": "Safety Guide", "content": "Important info"}]},
            skill_name="Research"
        )

        # Patch both memory and file operations
        with patch('hooks.Path') as mock_path:
            mock_file = MagicMock()
            mock_path.home.return_value.joinpath = lambda *args: mock_file
            mock_file.parent.mkdir = MagicMock()
            mock_file.open = MagicMock()

            result = await auto_persist_research(ctx)

            # Even without memory, should attempt file index
            assert result is ctx  # Returns context unchanged on indexing attempt


# =============================================================================
# Integration Test
# =============================================================================

class TestIntegration:
    """Integration tests for the hook system"""

    @pytest.mark.asyncio
    async def test_full_session_start_flow(self):
        """Test complete SESSION_START hook flow"""
        reset_hook_manager()

        additions_made = []

        async def test_handler_1(ctx):
            ctx.add_system_context("Handler1", "Content 1")
            additions_made.append("handler_1")
            return ctx

        async def test_handler_2(ctx):
            ctx.add_system_context("Handler2", "Content 2")
            additions_made.append("handler_2")
            return ctx

        mgr = get_hook_manager()
        mgr.register(HookType.SESSION_START, test_handler_1, priority=10)
        mgr.register(HookType.SESSION_START, test_handler_2, priority=20)

        ctx = HookContext(user_input="Hello Workshop")
        result = await mgr.execute(HookType.SESSION_START, ctx)

        # Both handlers should have run in order
        assert additions_made == ["handler_1", "handler_2"]

        # Both additions should be in result
        combined = result.get_combined_system_additions()
        assert "Content 1" in combined
        assert "Content 2" in combined

    @pytest.mark.asyncio
    async def test_full_post_tool_use_flow(self):
        """Test complete POST_TOOL_USE hook flow"""
        reset_hook_manager()

        tool_processed = []

        async def tool_tracker(ctx):
            tool_processed.append(ctx.tool_name)
            return ctx

        mgr = get_hook_manager()
        mgr.register(HookType.POST_TOOL_USE, tool_tracker)

        ctx = HookContext(
            tool_name="web_search",
            tool_args={"query": "test"},
            tool_result="results",
            skill_name="Research"
        )
        await mgr.execute(HookType.POST_TOOL_USE, ctx)

        assert tool_processed == ["web_search"]


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
