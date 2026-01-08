"""
Workshop End-to-End Testing Framework

This package provides comprehensive E2E testing infrastructure for the Workshop
AI agent system, enabling full pipeline tracing, visualization, and validation.

Modules:
    context_tracer: Core dataclasses for capturing pipeline traces
    mocks: Mock infrastructure for Claude Code, HTTP, and other external services
    fixtures: Shared pytest fixtures for test setup
    visualizer: ASCII/markdown trace visualization
    scenarios: E2EScenario framework for defining test scenarios

Usage:
    from tests.e2e import ContextPipelineTrace, E2EScenario, MockClaudeBridge

    async def test_research_flow():
        scenario = E2EScenario(
            name="research_query",
            user_inputs=["research Daniel Miessler"],
            expected_skills=["Research"],
        )
        result = await scenario.run(workshop)
        assert result.passed
"""

from .context_tracer import (
    ContextPipelineTrace,
    ToolCallDetail,
    RoutingDecision,
    ContextLayer,
    LLMInvocation,
)
from .mocks import MockClaudeBridge, MockHTTPClient, MockMemorySystem
from .scenarios import E2EScenario, E2EResult
from .visualizer import TraceVisualizer

__all__ = [
    # Core trace types
    "ContextPipelineTrace",
    "ToolCallDetail",
    "RoutingDecision",
    "ContextLayer",
    "LLMInvocation",
    # Mocks
    "MockClaudeBridge",
    "MockHTTPClient",
    "MockMemorySystem",
    # Scenarios
    "E2EScenario",
    "E2EResult",
    # Visualization
    "TraceVisualizer",
]
