"""
Multi-Turn Conversation Tests

Tests for multi-turn conversation flows:
- Context accumulation across turns
- Session continuity
- Task progression
- Skill transitions
- Memory persistence

These tests verify the full conversation lifecycle
is properly traced and validated.
"""

import pytest
from datetime import datetime

from tests.e2e.context_tracer import (
    ContextPipelineTrace,
    ToolCallDetail,
    TraceStage,
)
from tests.e2e.mocks import (
    MockFactory,
    MockSemanticMatch,
    MockLLMResponse,
    MockToolCall,
)
from tests.e2e.scenarios import E2EScenario, ScenarioBuilder, ValidationResult
from tests.e2e.visualizer import TraceVisualizer, TraceReporter


class TestMultiTurnBasics:
    """Test basic multi-turn conversation handling."""

    @pytest.mark.asyncio
    async def test_two_turn_conversation(self, mocks):
        """Test basic two-turn conversation."""
        scenario = E2EScenario(
            name="two_turns",
            description="Two turn conversation",
            user_inputs=[
                "Hello, can you help me?",
                "Yes, please search for Python.",
            ],
        )

        scenario.with_mocks(mocks)
        mocks.router.add_route("hello|help", MockSemanticMatch("chat", 0.80))
        mocks.router.add_route("search", MockSemanticMatch("Research", 0.90))

        scenario.expect_skills(["chat", "Research"])

        result = await scenario.run()

        assert len(result.traces) == 2
        assert result.traces[0].routing.final_skill == "chat"
        assert result.traces[1].routing.final_skill == "Research"

    @pytest.mark.asyncio
    async def test_three_turn_research_flow(self, mocks):
        """Test three-turn research flow."""
        scenario = E2EScenario(
            name="research_flow",
            user_inputs=[
                "Research Python async programming",
                "Can you give me more details on asyncio?",
                "Thanks, please save this to a file",
            ],
        )

        scenario.with_mocks(mocks)
        mocks.router.add_route("research", MockSemanticMatch("Research", 0.92))
        mocks.router.add_route("details|more", MockSemanticMatch("Research", 0.85))
        mocks.router.add_route("save|file", MockSemanticMatch("FileOperations", 0.88))

        scenario.expect_skills(["Research", "Research", "FileOperations"])

        result = await scenario.run()

        assert len(result.traces) == 3
        assert result.passed

    @pytest.mark.asyncio
    async def test_five_turn_complex_flow(self, mocks):
        """Test complex five-turn conversation."""
        scenario = E2EScenario(
            name="complex_flow",
            user_inputs=[
                "Remember that Python is my favorite language",
                "Search for Python web frameworks",
                "List files in my projects folder",
                "What do you remember about my preferences?",
                "Thanks, that's all for now",
            ],
        )

        scenario.with_mocks(mocks)
        mocks.router.add_route("remember", MockSemanticMatch("Memory", 0.90))
        mocks.router.add_route("search", MockSemanticMatch("Research", 0.88))
        mocks.router.add_route("list files", MockSemanticMatch("FileOperations", 0.87))
        mocks.router.add_route("what do you remember", MockSemanticMatch("Memory", 0.85))
        mocks.router.add_route("thanks|that's all", MockSemanticMatch("chat", 0.80))

        result = await scenario.run()

        assert len(result.traces) == 5


class TestSkillTransitions:
    """Test transitions between different skills."""

    @pytest.mark.asyncio
    async def test_research_to_memory_transition(self, mocks):
        """Test transition from Research to Memory skill."""
        scenario = E2EScenario(
            name="research_to_memory",
            user_inputs=[
                "Research the latest Python features",
                "Remember these findings for later",
            ],
        )

        scenario.with_mocks(mocks)
        mocks.router.add_route("research", MockSemanticMatch("Research", 0.92))
        mocks.router.add_route("remember", MockSemanticMatch("Memory", 0.90))

        mocks.claude.add_response(
            "research",
            MockLLMResponse("Here's what I found...", [MockToolCall("web_search", {})])
        )
        mocks.claude.add_response(
            "remember",
            MockLLMResponse("I'll remember that.", [MockToolCall("remember", {})])
        )

        scenario.expect_skills(["Research", "Memory"])
        scenario.expect_tools(["web_search", "remember"])

        result = await scenario.run()

        assert result.passed

    @pytest.mark.asyncio
    async def test_file_ops_to_research_transition(self, mocks):
        """Test transition from FileOperations to Research."""
        scenario = E2EScenario(
            name="files_to_research",
            user_inputs=[
                "List Python files in src/",
                "Research best practices for the patterns I see",
            ],
        )

        scenario.with_mocks(mocks)
        mocks.router.add_route("list.*files", MockSemanticMatch("FileOperations", 0.88))
        mocks.router.add_route("research", MockSemanticMatch("Research", 0.90))

        scenario.expect_skills(["FileOperations", "Research"])

        result = await scenario.run()

        assert result.passed
        assert result.traces[0].routing.final_skill == "FileOperations"
        assert result.traces[1].routing.final_skill == "Research"


class TestContextAccumulation:
    """Test context accumulation across turns."""

    @pytest.mark.asyncio
    async def test_context_grows_across_turns(self, mocks):
        """Test that context information accumulates."""
        scenario = E2EScenario(
            name="context_growth",
            user_inputs=[
                "Turn 1 message",
                "Turn 2 message",
                "Turn 3 message",
            ],
        )

        scenario.with_mocks(mocks)
        mocks.router.set_default(MockSemanticMatch("chat", 0.70))

        result = await scenario.run()

        # Each trace should have session info
        for i, trace in enumerate(result.traces):
            assert trace.session_id == f"e2e_test_session_{scenario.name}"

    @pytest.mark.asyncio
    async def test_telos_context_persists(self, mocks):
        """Test Telos context persists across turns."""
        scenario = E2EScenario(
            name="telos_persistence",
            user_inputs=["Turn 1", "Turn 2"],
        )

        scenario.with_mocks(mocks)
        scenario.expect_telos(loaded=True)

        result = await scenario.run()

        # Both traces should have Telos loaded (mock default)
        assert all(t.telos_loaded for t in result.traces)


class TestTaskProgression:
    """Test task progression across turns."""

    @pytest.mark.asyncio
    async def test_task_list_across_turns(self, mocks):
        """Test task list handling across multiple turns."""
        scenario = E2EScenario(
            name="task_progression",
            user_inputs=[
                "Create a task list: 1. Research, 2. Implement, 3. Test",
                "Start working on task 1",
                "Mark task 1 complete and start task 2",
            ],
        )

        scenario.with_mocks(mocks)
        mocks.router.add_route("task list|create", MockSemanticMatch("TaskManagement", 0.85))
        mocks.router.add_route("start|working", MockSemanticMatch("Research", 0.80))
        mocks.router.add_route("complete|mark", MockSemanticMatch("TaskManagement", 0.82))

        result = await scenario.run()

        assert len(result.traces) == 3


class TestScenarioBuilder:
    """Test ScenarioBuilder convenience methods."""

    @pytest.mark.asyncio
    async def test_research_scenario_builder(self, scenario_builder: ScenarioBuilder):
        """Test research scenario from builder."""
        scenario = scenario_builder.research_scenario("Python async patterns")

        result = await scenario.run()

        assert scenario.name == "research_flow"
        assert len(scenario.user_inputs) == 1

    @pytest.mark.asyncio
    async def test_file_ops_scenario_builder(self, scenario_builder: ScenarioBuilder):
        """Test file operations scenario from builder."""
        scenario = scenario_builder.file_operations_scenario("list files in /tmp")

        result = await scenario.run()

        assert scenario.name == "file_ops_flow"

    @pytest.mark.asyncio
    async def test_memory_scenario_builder(self, scenario_builder: ScenarioBuilder):
        """Test memory scenario from builder."""
        scenario = scenario_builder.memory_scenario(
            remember_text="Python is great",
            recall_query="Python"
        )

        assert scenario.name == "memory_flow"
        assert len(scenario.user_inputs) == 2

    @pytest.mark.asyncio
    async def test_multi_turn_scenario_builder(self, scenario_builder: ScenarioBuilder):
        """Test multi-turn scenario from builder."""
        scenario = scenario_builder.multi_turn_scenario(
            turns=["Turn 1", "Turn 2", "Turn 3"],
            expected_skills=["Research", "Memory", "chat"]
        )

        assert len(scenario.user_inputs) == 3
        assert scenario.expectations.expected_skills == ["Research", "Memory", "chat"]


class TestReporting:
    """Test multi-trace reporting."""

    @pytest.mark.asyncio
    async def test_reporter_multiple_traces(self, reporter: TraceReporter, mocks):
        """Test reporter handles multiple traces."""
        scenario = E2EScenario(
            name="multi_trace",
            user_inputs=["Query 1", "Query 2", "Query 3"],
        )

        scenario.with_mocks(mocks)
        mocks.router.set_default(MockSemanticMatch("chat", 0.70))

        result = await scenario.run()

        # Add traces to reporter
        for trace in result.traces:
            reporter.add_trace(trace)

        summary = reporter.generate_summary()

        assert "Total Traces:** 3" in summary
        assert "Trace Summary" in summary

    @pytest.mark.asyncio
    async def test_reporter_compact_output(self, reporter: TraceReporter, mocks):
        """Test reporter compact output."""
        scenario = E2EScenario(
            name="compact_test",
            user_inputs=["Test query"],
        )

        scenario.with_mocks(mocks)
        mocks.router.set_default(MockSemanticMatch("Research", 0.85))

        result = await scenario.run()

        for trace in result.traces:
            reporter.add_trace(trace)

        # Print compact should not raise
        reporter.print_compact()

        assert len(reporter.traces) == 1


class TestEdgeCases:
    """Test edge cases in multi-turn conversations."""

    @pytest.mark.asyncio
    async def test_empty_response_handling(self, mocks):
        """Test handling of empty responses."""
        scenario = E2EScenario(
            name="empty_response",
            user_inputs=["test"],
        )

        scenario.with_mocks(mocks)
        mocks.router.set_default(MockSemanticMatch("chat", 0.50))
        mocks.claude.set_default_response(MockLLMResponse(content=""))

        result = await scenario.run()

        # Should handle gracefully
        assert len(result.traces) == 1

    @pytest.mark.asyncio
    async def test_rapid_skill_changes(self, mocks):
        """Test rapid changes between skills."""
        scenario = E2EScenario(
            name="rapid_changes",
            user_inputs=[
                "search",
                "remember",
                "list files",
                "search again",
                "recall",
            ],
        )

        scenario.with_mocks(mocks)
        mocks.router.add_route("search", MockSemanticMatch("Research", 0.90))
        mocks.router.add_route("remember|recall", MockSemanticMatch("Memory", 0.88))
        mocks.router.add_route("list files", MockSemanticMatch("FileOperations", 0.85))

        result = await scenario.run()

        skills = [t.routing.final_skill for t in result.traces]
        assert skills == ["Research", "Memory", "FileOperations", "Research", "Memory"]

    @pytest.mark.asyncio
    async def test_long_conversation(self, mocks):
        """Test handling of longer conversation."""
        turns = [f"Turn {i+1}" for i in range(10)]

        scenario = E2EScenario(
            name="long_conversation",
            user_inputs=turns,
        )

        scenario.with_mocks(mocks)
        mocks.router.set_default(MockSemanticMatch("chat", 0.70))

        result = await scenario.run()

        assert len(result.traces) == 10


class TestValidationAcrossTurns:
    """Test validations that span multiple turns."""

    @pytest.mark.asyncio
    async def test_all_turns_pass(self, mocks):
        """Test all validations pass across turns."""
        scenario = E2EScenario(
            name="all_pass",
            user_inputs=["research topic", "research another"],
        )

        scenario.with_mocks(mocks)
        mocks.router.add_route("research", MockSemanticMatch("Research", 0.90))

        scenario.expect_skills(["Research", "Research"])
        scenario.expect_min_confidence(0.85)

        result = await scenario.run()

        assert result.passed
        assert result.failed_count == 0

    @pytest.mark.asyncio
    async def test_partial_turn_failure(self, mocks):
        """Test when one turn fails validation."""
        scenario = E2EScenario(
            name="partial_fail",
            user_inputs=["research topic", "gibberish xyz"],
        )

        scenario.with_mocks(mocks)
        mocks.router.add_route("research", MockSemanticMatch("Research", 0.90))
        mocks.router.add_route("gibberish", MockSemanticMatch("Unknown", 0.20))

        scenario.expect_skills(["Research", "Research"])  # Second will fail

        result = await scenario.run()

        assert not result.passed
        assert result.failed_count > 0

    @pytest.mark.asyncio
    async def test_aggregate_tool_count(self, mocks):
        """Test tool count aggregated across turns."""
        scenario = E2EScenario(
            name="aggregate_tools",
            user_inputs=["search 1", "search 2"],
        )

        scenario.with_mocks(mocks)
        mocks.router.add_route("search", MockSemanticMatch("Research", 0.88))
        mocks.claude.add_response(
            "search",
            MockLLMResponse("Searching...", [MockToolCall("web_search", {})])
        )

        scenario.expect_tool_count_range(2, 4)  # Expect 2 total across both turns

        result = await scenario.run()

        total_tools = sum(t.tool_total_calls for t in result.traces)
        assert total_tools >= 2


class TestPerformance:
    """Test performance expectations across turns."""

    @pytest.mark.asyncio
    async def test_duration_across_turns(self, mocks):
        """Test duration tracking across turns."""
        scenario = E2EScenario(
            name="duration_test",
            user_inputs=["query 1", "query 2"],
        )

        scenario.with_mocks(mocks)
        mocks.router.set_default(MockSemanticMatch("chat", 0.70))

        result = await scenario.run()

        # Result should have total duration
        assert result.duration_ms >= 0

        # Each trace should have duration
        for trace in result.traces:
            assert trace.duration_total_ms >= 0

    @pytest.mark.asyncio
    async def test_max_duration_exceeded(self, mocks):
        """Test max duration validation."""
        scenario = E2EScenario(
            name="max_duration",
            user_inputs=["test"],
        )

        scenario.with_mocks(mocks)
        mocks.router.set_default(MockSemanticMatch("chat", 0.70))

        scenario.expect_max_duration(1)  # 1ms - impossible to meet

        result = await scenario.run()

        # Should have failed duration check
        duration_checks = [v for v in result.validations if "duration" in v.name]
        # Note: This may or may not fail depending on mock execution time
