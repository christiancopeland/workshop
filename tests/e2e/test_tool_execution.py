"""
Tool Execution Tests

Tests for the tool execution pipeline:
- Tool call parsing from LLM response
- Tool argument normalization
- Dependency injection
- Result capture and formatting
- Error handling
- Multi-iteration tool loops

These tests verify tool execution is properly traced
and validated in E2E scenarios.
"""

import pytest
from datetime import datetime

from tests.e2e.context_tracer import (
    ContextPipelineTrace,
    ToolCallDetail,
    LLMInvocation,
    TraceStage,
)
from tests.e2e.mocks import (
    MockFactory,
    MockClaudeBridge,
    MockLLMResponse,
    MockToolCall,
    MockSemanticMatch,
)
from tests.e2e.scenarios import E2EScenario, ValidationResult
from tests.e2e.visualizer import TraceVisualizer


class TestToolCallCapture:
    """Test capturing tool calls in traces."""

    def test_tool_call_basic_capture(self, trace: ContextPipelineTrace):
        """Test basic tool call capture."""
        tool_call = ToolCallDetail(
            call_id="tool_abc123",
            iteration=1,
            sequence_in_iteration=1,
            tool_name="web_search",
            skill_name="Research",
            args_from_llm={"query": "Python tutorials"},
            args_normalized={"query": "Python tutorials"},
            duration_ms=1500,
            result="5 results found...",
            result_length=500,
            result_type="success",
        )
        trace.add_tool_call(tool_call)

        assert trace.tool_total_calls == 1
        assert trace.tool_total_duration_ms == 1500
        assert trace.tool_calls[0].tool_name == "web_search"

    def test_tool_call_with_args_normalization(self, trace: ContextPipelineTrace):
        """Test capturing argument normalization."""
        tool_call = ToolCallDetail(
            call_id="tool_xyz789",
            tool_name="read_file",
            skill_name="FileOperations",
            args_from_llm={"filepath": "/path/to/file"},  # LLM used 'filepath'
            args_normalized={"path": "/path/to/file"},  # Normalized to 'path'
            result_type="success",
        )
        trace.add_tool_call(tool_call)

        assert trace.tool_calls[0].args_from_llm["filepath"] == "/path/to/file"
        assert trace.tool_calls[0].args_normalized["path"] == "/path/to/file"

    def test_tool_call_with_dependencies(self, trace: ContextPipelineTrace):
        """Test capturing dependency injection."""
        tool_call = ToolCallDetail(
            call_id="tool_dep123",
            tool_name="save_research_output",
            skill_name="Research",
            args_from_llm={"filename": "output.md", "content": "..."},
            dependencies_available=["memory", "config", "session"],
            dependencies_used=["memory"],
            result_type="success",
        )
        trace.add_tool_call(tool_call)

        assert "memory" in trace.tool_calls[0].dependencies_available
        assert "memory" in trace.tool_calls[0].dependencies_used

    def test_tool_call_error_capture(self, trace: ContextPipelineTrace):
        """Test capturing tool errors."""
        tool_call = ToolCallDetail(
            call_id="tool_err456",
            tool_name="fetch_url",
            skill_name="Research",
            args_from_llm={"url": "https://invalid.example"},
            result_type="error",
            error="Connection timeout after 30s",
            error_traceback="Traceback (most recent call last):\n...",
        )
        trace.add_tool_call(tool_call)

        assert trace.tool_calls[0].result_type == "error"
        assert "timeout" in trace.tool_calls[0].error

    def test_multiple_tool_calls(self, trace: ContextPipelineTrace):
        """Test capturing multiple tool calls."""
        for i in range(3):
            tool_call = ToolCallDetail(
                call_id=f"tool_{i}",
                iteration=1,
                sequence_in_iteration=i + 1,
                tool_name=f"tool_{i}",
                duration_ms=100 * (i + 1),
                result_type="success",
            )
            trace.add_tool_call(tool_call)

        assert trace.tool_total_calls == 3
        assert trace.tool_total_duration_ms == 600  # 100 + 200 + 300


class TestToolCallSerialization:
    """Test serialization of tool calls."""

    def test_tool_call_to_dict(self):
        """Test ToolCallDetail serialization."""
        tool_call = ToolCallDetail(
            call_id="tool_123",
            iteration=2,
            sequence_in_iteration=1,
            tool_name="web_search",
            skill_name="Research",
            args_from_llm={"query": "test"},
            args_normalized={"query": "test"},
            dependencies_available=["memory"],
            dependencies_used=["memory"],
            duration_ms=500,
            result="Results...",
            result_length=100,
            result_type="success",
        )

        data = tool_call.to_dict()

        assert data["call_id"] == "tool_123"
        assert data["iteration"] == 2
        assert data["tool"]["name"] == "web_search"
        assert data["tool"]["skill"] == "Research"
        assert data["args"]["from_llm"] == {"query": "test"}
        assert data["duration_ms"] == 500
        assert data["result"]["type"] == "success"

    def test_trace_includes_tool_calls(self, trace: ContextPipelineTrace):
        """Test trace serialization includes tool calls."""
        trace.add_tool_call(ToolCallDetail(
            call_id="test",
            tool_name="web_search",
            result_type="success",
        ))

        data = trace.to_dict()

        assert "tool_calls" in data
        assert len(data["tool_calls"]) == 1
        assert data["tool_calls"][0]["tool"]["name"] == "web_search"


class TestMockClaudeBridgeToolCalls:
    """Test MockClaudeBridge tool call handling."""

    @pytest.mark.asyncio
    async def test_mock_claude_returns_tool_calls(self, mock_claude: MockClaudeBridge):
        """Test mock Claude returns configured tool calls."""
        mock_claude.add_response(
            "search",
            MockLLMResponse(
                content="I'll search for that.",
                tool_calls=[MockToolCall("web_search", {"query": "test"})]
            )
        )

        response = await mock_claude.query([{"role": "user", "content": "search for something"}])

        assert len(response["tool_calls"]) == 1
        assert response["tool_calls"][0]["tool"] == "web_search"
        assert response["tool_calls"][0]["args"]["query"] == "test"

    @pytest.mark.asyncio
    async def test_mock_claude_multiple_tool_calls(self, mock_claude: MockClaudeBridge):
        """Test mock Claude returns multiple tool calls."""
        mock_claude.add_response(
            "research",
            MockLLMResponse(
                content="I'll gather information.",
                tool_calls=[
                    MockToolCall("web_search", {"query": "topic"}),
                    MockToolCall("fetch_url", {"url": "https://example.com"}),
                ]
            )
        )

        response = await mock_claude.query([{"role": "user", "content": "research something"}])

        assert len(response["tool_calls"]) == 2

    @pytest.mark.asyncio
    async def test_mock_claude_response_sequence(self, mock_claude: MockClaudeBridge):
        """Test mock Claude returns responses in sequence."""
        mock_claude.add_response_sequence([
            MockLLMResponse(
                content="Step 1",
                tool_calls=[MockToolCall("tool_1", {})]
            ),
            MockLLMResponse(
                content="Step 2",
                tool_calls=[MockToolCall("tool_2", {})]
            ),
            MockLLMResponse(
                content="Done",
                tool_calls=[]
            ),
        ])

        # First call
        r1 = await mock_claude.query([{"role": "user", "content": "start"}])
        assert len(r1["tool_calls"]) == 1
        assert r1["tool_calls"][0]["tool"] == "tool_1"

        # Second call
        r2 = await mock_claude.query([{"role": "user", "content": "continue"}])
        assert len(r2["tool_calls"]) == 1
        assert r2["tool_calls"][0]["tool"] == "tool_2"

        # Third call
        r3 = await mock_claude.query([{"role": "user", "content": "finish"}])
        assert len(r3["tool_calls"]) == 0

    @pytest.mark.asyncio
    async def test_mock_claude_tracks_invocations(self, mock_claude: MockClaudeBridge):
        """Test mock Claude tracks all invocations."""
        await mock_claude.query([{"role": "user", "content": "query 1"}])
        await mock_claude.query([{"role": "user", "content": "query 2"}])

        assert len(mock_claude.invocations) == 2
        assert mock_claude.invocations[0]["user_content"] == "query 1"
        assert mock_claude.invocations[1]["user_content"] == "query 2"


class TestToolExecutionScenarios:
    """Test tool execution in full E2E scenarios."""

    @pytest.mark.asyncio
    async def test_single_tool_execution(self, mocks):
        """Test scenario with single tool execution."""
        scenario = E2EScenario(
            name="single_tool",
            user_inputs=["search for Python tutorials"],
        )

        scenario.with_mocks(mocks)
        mocks.router.add_route("search", MockSemanticMatch("Research", 0.90))
        mocks.claude.add_response(
            "search",
            MockLLMResponse(
                content="Searching...",
                tool_calls=[MockToolCall("web_search", {"query": "Python tutorials"})]
            )
        )

        scenario.expect_tool("web_search")
        scenario.expect_tool_count(1)

        result = await scenario.run()

        assert result.passed
        assert result.traces[0].tool_total_calls == 1

    @pytest.mark.asyncio
    async def test_multi_tool_chain(self, mocks):
        """Test scenario with multiple tool calls in single response."""
        scenario = E2EScenario(
            name="multi_tool",
            user_inputs=["research and summarize Python asyncio"],
        )

        scenario.with_mocks(mocks)
        mocks.router.add_route("research", MockSemanticMatch("Research", 0.92))
        # Single response with multiple tool calls (realistic for single-turn mock)
        mocks.claude.add_response(
            "research",
            MockLLMResponse(
                content="I'll search and fetch details...",
                tool_calls=[
                    MockToolCall("web_search", {"query": "Python asyncio"}),
                    MockToolCall("fetch_url", {"url": "https://docs.python.org"}),
                ]
            )
        )

        scenario.expect_tools(["web_search", "fetch_url"])
        scenario.expect_tool_count_range(2, 5)

        result = await scenario.run()

        assert result.passed

    @pytest.mark.asyncio
    async def test_tool_count_validation(self, mocks):
        """Test tool count validation."""
        scenario = E2EScenario(
            name="tool_count",
            user_inputs=["test"],
        )

        scenario.with_mocks(mocks)
        mocks.router.add_route(".*", MockSemanticMatch("Research", 0.80))
        mocks.claude.add_response(
            ".*",
            MockLLMResponse(
                content="Done",
                tool_calls=[
                    MockToolCall("tool_1", {}),
                    MockToolCall("tool_2", {}),
                ]
            )
        )

        scenario.expect_tool_count(2)  # Exact count

        result = await scenario.run()

        count_checks = [v for v in result.validations if "tool_count" in v.name]
        assert len(count_checks) > 0

    @pytest.mark.asyncio
    async def test_tool_range_validation(self, mocks):
        """Test tool count range validation."""
        scenario = E2EScenario(
            name="tool_range",
            user_inputs=["test"],
        )

        scenario.with_mocks(mocks)
        mocks.router.add_route(".*", MockSemanticMatch("Research", 0.80))
        mocks.claude.add_response(
            ".*",
            MockLLMResponse(
                content="Done",
                tool_calls=[MockToolCall("tool_1", {})]
            )
        )

        scenario.expect_tool_count_range(1, 3)  # Range

        result = await scenario.run()

        range_checks = [v for v in result.validations if "tool_count_range" in v.name]
        assert len(range_checks) > 0
        assert all(v.result == ValidationResult.PASSED for v in range_checks)


class TestToolCallVisualization:
    """Test visualization of tool calls."""

    def test_ascii_tool_section(self, trace: ContextPipelineTrace, visualizer: TraceVisualizer):
        """Test ASCII visualization includes tool calls."""
        trace.add_tool_call(ToolCallDetail(
            call_id="tool_1",
            iteration=1,
            tool_name="web_search",
            skill_name="Research",
            args_from_llm={"query": "test"},
            duration_ms=500,
            result="Results...",
            result_length=100,
            result_type="success",
        ))
        trace.complete("Done")

        output = visualizer.to_ascii(trace)

        assert "TOOL EXECUTION" in output
        assert "web_search" in output
        assert "success" in output

    def test_markdown_tool_section(self, trace: ContextPipelineTrace, visualizer: TraceVisualizer):
        """Test markdown visualization includes tool calls."""
        trace.add_tool_call(ToolCallDetail(
            call_id="tool_1",
            iteration=1,
            tool_name="web_search",
            skill_name="Research",
            args_from_llm={"query": "Python tutorials"},
            args_normalized={"query": "Python tutorials"},
            duration_ms=500,
            result="Found 5 results",
            result_length=200,
            result_type="success",
        ))
        trace.iterations_used = 2
        trace.max_iterations = 100
        trace.complete("Summary response")

        output = visualizer.to_markdown(trace)

        assert "## Tool Execution" in output
        assert "web_search" in output
        assert "**Arguments (from LLM):**" in output
        assert "Python tutorials" in output


class TestLLMInvocationCapture:
    """Test capturing LLM invocations."""

    def test_llm_invocation_basic(self, trace: ContextPipelineTrace):
        """Test basic LLM invocation capture."""
        invocation = LLMInvocation(
            invocation_id="llm_abc123",
            iteration=1,
            system_prompt="You are a helpful assistant...",
            system_prompt_length=500,
            user_message="Search for Python",
            user_message_length=17,
            response_raw="I'll search for that.",
            response_length=22,
            tool_calls_detected=1,
            duration_ms=800,
        )
        trace.add_llm_invocation(invocation)

        assert trace.llm_total_invocations == 1
        assert trace.llm_total_duration_ms == 800

    def test_llm_invocation_with_session(self, trace: ContextPipelineTrace):
        """Test LLM invocation with session info."""
        invocation = LLMInvocation(
            invocation_id="llm_sess123",
            iteration=1,
            claude_session_id="claude_sess_20260104",
            turn_number=3,
            duration_ms=500,
        )
        trace.add_llm_invocation(invocation)

        assert trace.llm_invocations[0].claude_session_id == "claude_sess_20260104"
        assert trace.llm_invocations[0].turn_number == 3

    def test_llm_invocation_serialization(self):
        """Test LLM invocation serialization."""
        invocation = LLMInvocation(
            invocation_id="llm_123",
            iteration=2,
            system_prompt_length=1000,
            user_message_length=50,
            response_length=500,
            tool_calls_detected=2,
            duration_ms=600,
        )

        data = invocation.to_dict()

        assert data["invocation_id"] == "llm_123"
        assert data["iteration"] == 2
        assert data["input"]["system_prompt_length"] == 1000
        assert data["output"]["tool_calls_detected"] == 2


class TestToolExecutionLoop:
    """Test multi-iteration tool execution loops."""

    def test_iteration_tracking(self, trace: ContextPipelineTrace):
        """Test iteration tracking in trace."""
        # Simulate multi-iteration loop
        for i in range(3):
            trace.add_llm_invocation(LLMInvocation(
                invocation_id=f"llm_{i}",
                iteration=i + 1,
            ))
            if i < 2:  # Tools on first 2 iterations
                trace.add_tool_call(ToolCallDetail(
                    call_id=f"tool_{i}",
                    iteration=i + 1,
                    tool_name=f"tool_{i}",
                ))

        trace.iterations_used = 3
        trace.max_iterations = 100

        assert trace.llm_total_invocations == 3
        assert trace.tool_total_calls == 2
        assert trace.iterations_used == 3

    def test_max_iterations_capture(self, trace: ContextPipelineTrace):
        """Test capturing max iterations."""
        trace.max_iterations = 100
        trace.iterations_used = 5

        assert trace.max_iterations == 100
        assert trace.iterations_used == 5

    @pytest.mark.asyncio
    async def test_scenario_tracks_iterations(self, mocks):
        """Test scenario tracks iteration count."""
        scenario = E2EScenario(
            name="iterations",
            user_inputs=["multi-step task"],
        )

        scenario.with_mocks(mocks)
        mocks.router.add_route(".*", MockSemanticMatch("Research", 0.80))
        mocks.claude.add_response_sequence([
            MockLLMResponse("Step 1", [MockToolCall("t1", {})]),
            MockLLMResponse("Step 2", [MockToolCall("t2", {})]),
            MockLLMResponse("Done", []),
        ])

        result = await scenario.run()

        # Check iterations tracked
        assert len(result.traces) == 1
        # Tool calls should reflect the mock sequence
