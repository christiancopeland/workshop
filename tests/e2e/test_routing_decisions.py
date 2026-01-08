"""
Routing Decision Tests

Tests for the routing pipeline:
- Semantic routing with confidence thresholds
- Pattern detection
- Claude fallback routing
- Active skill continuation
- Routing visualization

These tests verify routing decisions are properly captured
and validated in E2E traces.
"""

import pytest
from datetime import datetime

from tests.e2e.context_tracer import (
    ContextPipelineTrace,
    RoutingDecision,
    TraceStage,
)
from tests.e2e.mocks import (
    MockFactory,
    MockSemanticRouter,
    MockSemanticMatch,
    MockClaudeBridge,
    MockLLMResponse,
)
from tests.e2e.scenarios import E2EScenario, ValidationResult
from tests.e2e.visualizer import TraceVisualizer


class TestRoutingDecisionCapture:
    """Test capturing routing decisions in traces."""

    def test_semantic_routing_capture(self, trace: ContextPipelineTrace):
        """Test capturing semantic routing results."""
        trace.routing.semantic_enabled = True
        trace.routing.semantic_score = 0.92
        trace.routing.semantic_matched_skill = "Research"
        trace.routing.semantic_matched_utterance = "research [topic]'s methodology"
        trace.routing.semantic_all_scores = [
            ("Research", 0.92),
            ("Memory", 0.45),
            ("FileOperations", 0.32),
        ]
        trace.routing.semantic_duration_ms = 25

        assert trace.routing.semantic_score == 0.92
        assert trace.routing.semantic_matched_skill == "Research"
        assert len(trace.routing.semantic_all_scores) == 3

    def test_pattern_detection_capture(self, trace: ContextPipelineTrace):
        """Test capturing pattern detection results."""
        trace.routing.pattern_detected = True
        trace.routing.pattern_name = "extract_wisdom"
        trace.routing.pattern_pipeline = None
        trace.routing.pattern_confidence = 0.95

        assert trace.routing.pattern_detected
        assert trace.routing.pattern_name == "extract_wisdom"

    def test_pipeline_detection_capture(self, trace: ContextPipelineTrace):
        """Test capturing pipeline detection results."""
        trace.routing.pattern_detected = True
        trace.routing.pattern_name = None
        trace.routing.pattern_pipeline = ["extract_wisdom", "create_summary"]
        trace.routing.pattern_confidence = 0.9

        assert trace.routing.pattern_pipeline is not None
        assert len(trace.routing.pattern_pipeline) == 2

    def test_claude_fallback_capture(self, trace: ContextPipelineTrace):
        """Test capturing Claude fallback routing."""
        trace.routing.claude_fallback_used = True
        trace.routing.claude_routing_prompt = "Classify this query..."
        trace.routing.claude_routing_response = "Research"
        trace.routing.claude_routing_duration_ms = 500

        assert trace.routing.claude_fallback_used
        assert trace.routing.claude_routing_duration_ms == 500

    def test_final_decision_capture(self, trace: ContextPipelineTrace):
        """Test capturing final routing decision."""
        trace.routing.final_skill = "Research"
        trace.routing.final_method = "semantic_direct"
        trace.routing.final_confidence = 0.92

        assert trace.routing.final_skill == "Research"
        assert trace.routing.final_method == "semantic_direct"


class TestRoutingThresholds:
    """Test routing threshold behavior."""

    def test_high_confidence_bypass(self):
        """Test HIGH_CONFIDENCE (>=0.85) results in direct routing."""
        routing = RoutingDecision(
            semantic_score=0.91,
            semantic_matched_skill="Research",
            bypass_threshold=0.85,
            confirm_threshold=0.45,
        )

        # Verify this would bypass router
        assert routing.semantic_score >= routing.bypass_threshold

    def test_medium_confidence_trusted(self):
        """Test MEDIUM_CONFIDENCE (0.45-0.85) trusts semantic."""
        routing = RoutingDecision(
            semantic_score=0.65,
            semantic_matched_skill="Memory",
            bypass_threshold=0.85,
            confirm_threshold=0.45,
        )

        assert routing.semantic_score < routing.bypass_threshold
        assert routing.semantic_score >= routing.confirm_threshold

    def test_low_confidence_fallback(self):
        """Test LOW_CONFIDENCE (<0.45) triggers fallback."""
        routing = RoutingDecision(
            semantic_score=0.30,
            semantic_matched_skill="General",
            bypass_threshold=0.85,
            confirm_threshold=0.45,
        )

        assert routing.semantic_score < routing.confirm_threshold


class TestRoutingSerialization:
    """Test serialization of routing decisions."""

    def test_routing_to_dict(self):
        """Test RoutingDecision serialization."""
        routing = RoutingDecision(
            semantic_enabled=True,
            semantic_score=0.92,
            semantic_matched_skill="Research",
            semantic_matched_utterance="research [topic]",
            semantic_all_scores=[("Research", 0.92), ("Memory", 0.45)],
            semantic_duration_ms=25,
            pattern_detected=False,
            claude_fallback_used=False,
            final_skill="Research",
            final_method="semantic_direct",
            final_confidence=0.92,
        )

        data = routing.to_dict()

        assert data["semantic"]["enabled"]
        assert data["semantic"]["score"] == 0.92
        assert data["decision"]["skill"] == "Research"
        assert data["decision"]["method"] == "semantic_direct"
        assert data["thresholds"]["bypass"] == 0.85

    def test_trace_includes_routing(self, trace: ContextPipelineTrace):
        """Test trace serialization includes routing."""
        trace.routing.final_skill = "Research"
        trace.routing.final_method = "semantic_direct"
        trace.routing.final_confidence = 0.92

        data = trace.to_dict()

        assert "routing" in data
        assert data["routing"]["decision"]["skill"] == "Research"


class TestMockRouterBehavior:
    """Test MockSemanticRouter behavior."""

    @pytest.mark.asyncio
    async def test_mock_router_pattern_match(self, mock_router: MockSemanticRouter):
        """Test mock router pattern matching."""
        mock_router.add_route(
            "research|search|find",
            MockSemanticMatch("Research", 0.92, "research [topic]", "semantic_direct")
        )

        match = await mock_router.route("I want to research Python")

        assert match.skill_name == "Research"
        assert match.confidence == 0.92

    @pytest.mark.asyncio
    async def test_mock_router_default(self, mock_router: MockSemanticRouter):
        """Test mock router returns default for unmatched."""
        mock_router.set_default(MockSemanticMatch("General", 0.3, "", "fallback"))

        match = await mock_router.route("random gibberish xyz")

        assert match.skill_name == "General"
        assert match.confidence == 0.3

    @pytest.mark.asyncio
    async def test_mock_router_tracks_calls(self, mock_router: MockSemanticRouter):
        """Test mock router tracks route calls."""
        await mock_router.route("query 1")
        await mock_router.route("query 2")

        assert len(mock_router.route_calls) == 2
        assert mock_router.route_calls[0]["query"] == "query 1"
        assert mock_router.route_calls[1]["query"] == "query 2"


class TestRoutingScenarios:
    """Test routing in full E2E scenarios."""

    @pytest.mark.asyncio
    async def test_high_confidence_research_routing(self, mocks):
        """Test high confidence routing to Research skill."""
        scenario = E2EScenario(
            name="research_routing",
            description="Test research query routing",
            user_inputs=["research Python async programming"],
        )

        scenario.with_mocks(mocks)
        mocks.router.add_route(
            "research",
            MockSemanticMatch("Research", 0.92, "research [topic]", "semantic_direct")
        )

        scenario.expect_skill("Research")
        scenario.expect_routing_method("semantic_direct")
        scenario.expect_min_confidence(0.85)

        result = await scenario.run()

        assert result.passed
        assert result.traces[0].routing.final_skill == "Research"
        assert result.traces[0].routing.final_confidence >= 0.85

    @pytest.mark.asyncio
    async def test_medium_confidence_routing(self, mocks):
        """Test medium confidence routing."""
        scenario = E2EScenario(
            name="medium_confidence",
            user_inputs=["look up something about asyncio"],
        )

        scenario.with_mocks(mocks)
        mocks.router.add_route(
            "look up|asyncio",
            MockSemanticMatch("Research", 0.65, "look up [topic]", "semantic_trusted")
        )

        scenario.expect_skill("Research")

        result = await scenario.run()

        assert result.passed
        trace = result.traces[0]
        assert 0.45 <= trace.routing.final_confidence < 0.85

    @pytest.mark.asyncio
    async def test_chat_pattern_routing(self, mocks):
        """Test chat pattern detection."""
        scenario = E2EScenario(
            name="chat_routing",
            user_inputs=["hello, how are you?"],
        )

        scenario.with_mocks(mocks)
        mocks.router.add_route(
            "hello|hi|how are you",
            MockSemanticMatch("chat", 0.95, "greeting", "pattern_chat")
        )

        scenario.expect_skill("chat")
        scenario.expect_routing_method("pattern_chat")

        result = await scenario.run()

        assert result.passed

    @pytest.mark.asyncio
    async def test_clarify_fallback_routing(self, mocks):
        """Test clarify fallback for ambiguous queries."""
        scenario = E2EScenario(
            name="clarify_routing",
            user_inputs=["do the thing"],
        )

        scenario.with_mocks(mocks)
        mocks.router.set_default(
            MockSemanticMatch("clarify", 0.25, "", "fallback")
        )

        scenario.expect_skill("clarify")

        result = await scenario.run()

        assert result.passed


class TestRoutingValidation:
    """Test validation of routing expectations."""

    @pytest.mark.asyncio
    async def test_validate_skill_match(self, mocks):
        """Test validation of expected skill."""
        scenario = E2EScenario(
            name="skill_validation",
            user_inputs=["remember this note"],
        )

        scenario.with_mocks(mocks)
        mocks.router.add_route(
            "remember",
            MockSemanticMatch("Memory", 0.90, "remember [content]", "semantic_direct")
        )

        scenario.expect_skill("Memory")

        result = await scenario.run()

        skill_checks = [v for v in result.validations if "skill" in v.name]
        assert len(skill_checks) > 0
        assert all(v.result == ValidationResult.PASSED for v in skill_checks)

    @pytest.mark.asyncio
    async def test_validate_wrong_skill(self, mocks):
        """Test validation fails when wrong skill."""
        scenario = E2EScenario(
            name="wrong_skill",
            user_inputs=["test query"],
        )

        scenario.with_mocks(mocks)
        mocks.router.add_route(
            ".*",
            MockSemanticMatch("Research", 0.80, "", "semantic")
        )

        scenario.expect_skill("Memory")  # Wrong expectation

        result = await scenario.run()

        skill_checks = [v for v in result.validations if "skill" in v.name]
        assert any(v.result == ValidationResult.FAILED for v in skill_checks)

    @pytest.mark.asyncio
    async def test_validate_routing_method(self, mocks):
        """Test validation of routing method."""
        scenario = E2EScenario(
            name="method_validation",
            user_inputs=["research something"],
        )

        scenario.with_mocks(mocks)
        mocks.router.add_route(
            "research",
            MockSemanticMatch("Research", 0.92, "", "semantic_direct")
        )

        scenario.expect_routing_method("semantic_direct")

        result = await scenario.run()

        method_checks = [v for v in result.validations if "routing_method" in v.name]
        assert len(method_checks) > 0
        assert all(v.result == ValidationResult.PASSED for v in method_checks)


class TestRoutingVisualization:
    """Test visualization of routing decisions."""

    def test_ascii_routing_section(self, trace: ContextPipelineTrace, visualizer: TraceVisualizer):
        """Test ASCII visualization includes routing."""
        trace.routing.semantic_score = 0.92
        trace.routing.semantic_matched_skill = "Research"
        trace.routing.semantic_all_scores = [
            ("Research", 0.92),
            ("Memory", 0.45),
        ]
        trace.routing.final_skill = "Research"
        trace.routing.final_method = "semantic_direct"
        trace.routing.final_confidence = 0.92
        trace.routing_total_duration_ms = 25
        trace.complete("Done")

        output = visualizer.to_ascii(trace)

        assert "ROUTING" in output
        assert "Research" in output
        assert "0.92" in output or "92" in output

    def test_markdown_routing_section(self, trace: ContextPipelineTrace, visualizer: TraceVisualizer):
        """Test markdown visualization includes routing."""
        trace.routing.semantic_score = 0.92
        trace.routing.semantic_matched_skill = "Research"
        trace.routing.semantic_matched_utterance = "research [topic]"
        trace.routing.semantic_all_scores = [
            ("Research", 0.92),
            ("Memory", 0.45),
            ("FileOperations", 0.32),
        ]
        trace.routing.final_skill = "Research"
        trace.routing.final_method = "semantic_direct"
        trace.routing.final_confidence = 0.92
        trace.complete("Done")

        output = visualizer.to_markdown(trace)

        assert "## Routing Decision" in output
        assert "### Semantic Routing" in output
        assert "Research" in output
        assert "Top Candidates" in output


class TestMultiSkillRouting:
    """Test routing across multiple turns with different skills."""

    @pytest.mark.asyncio
    async def test_multi_skill_conversation(self, mocks):
        """Test routing different queries to different skills."""
        scenario = E2EScenario(
            name="multi_skill",
            user_inputs=[
                "research Python tutorials",
                "remember this: Python is great",
                "list files in /tmp",
            ],
        )

        scenario.with_mocks(mocks)
        mocks.router.add_route("research", MockSemanticMatch("Research", 0.90))
        mocks.router.add_route("remember", MockSemanticMatch("Memory", 0.88))
        mocks.router.add_route("list files", MockSemanticMatch("FileOperations", 0.85))

        scenario.expect_skills(["Research", "Memory", "FileOperations"])

        result = await scenario.run()

        assert len(result.traces) == 3
        assert result.traces[0].routing.final_skill == "Research"
        assert result.traces[1].routing.final_skill == "Memory"
        assert result.traces[2].routing.final_skill == "FileOperations"
