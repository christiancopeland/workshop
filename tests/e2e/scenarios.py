"""
E2E Scenario Framework

Provides a declarative way to define and execute end-to-end test scenarios
with full pipeline tracing and validation.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
import logging

from .context_tracer import (
    ContextPipelineTrace,
    ContextLayer,
    ToolCallDetail,
    LLMInvocation,
    RoutingDecision,
    TraceStage,
    PipelineTracer,
)
from .mocks import (
    MockClaudeBridge,
    MockHTTPClient,
    MockMemorySystem,
    MockSemanticRouter,
    MockTaskManager,
    MockFactory,
    MockLLMResponse,
    MockToolCall,
)
from .visualizer import TraceVisualizer, TraceReporter

log = logging.getLogger(__name__)


class ValidationResult(Enum):
    """Result of a single validation check."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ValidationCheck:
    """A single validation check result."""
    name: str
    result: ValidationResult
    expected: Any = None
    actual: Any = None
    message: str = ""

    def __bool__(self) -> bool:
        return self.result == ValidationResult.PASSED


@dataclass
class E2EResult:
    """Result of an E2E scenario execution."""
    scenario_name: str
    passed: bool
    traces: List[ContextPipelineTrace] = field(default_factory=list)
    validations: List[ValidationCheck] = field(default_factory=list)
    duration_ms: int = 0
    error: Optional[str] = None
    error_traceback: Optional[str] = None

    @property
    def passed_count(self) -> int:
        return sum(1 for v in self.validations if v.result == ValidationResult.PASSED)

    @property
    def failed_count(self) -> int:
        return sum(1 for v in self.validations if v.result == ValidationResult.FAILED)

    def summary(self) -> str:
        """Generate summary string."""
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        return (
            f"{status} {self.scenario_name}\n"
            f"  Traces: {len(self.traces)}\n"
            f"  Validations: {self.passed_count}/{len(self.validations)} passed\n"
            f"  Duration: {self.duration_ms}ms"
        )

    def failed_validations(self) -> List[ValidationCheck]:
        """Get list of failed validations."""
        return [v for v in self.validations if v.result == ValidationResult.FAILED]


@dataclass
class ScenarioExpectations:
    """Expected outcomes for a scenario."""
    # Routing expectations
    expected_skills: List[str] = field(default_factory=list)
    expected_routing_methods: List[str] = field(default_factory=list)
    min_routing_confidence: float = 0.0

    # Tool execution expectations
    expected_tools: List[str] = field(default_factory=list)
    expected_tool_count: Optional[int] = None
    min_tool_count: int = 0
    max_tool_count: int = 100

    # Context expectations
    expect_telos_loaded: Optional[bool] = None
    expect_task_context: Optional[bool] = None
    expect_auto_context: Optional[bool] = None
    expect_memory_search: Optional[bool] = None

    # Response expectations
    min_response_length: int = 0
    response_contains: List[str] = field(default_factory=list)
    response_not_contains: List[str] = field(default_factory=list)

    # Performance expectations
    max_duration_ms: int = 30000  # 30 seconds default
    max_tool_duration_ms: int = 10000  # 10 seconds per tool


class E2EScenario:
    """
    Define and execute an end-to-end test scenario.

    Usage:
        scenario = E2EScenario(
            name="research_query",
            description="Test basic research flow",
            user_inputs=["research Daniel Miessler"],
        )

        # Configure expectations
        scenario.expect_skill("Research")
        scenario.expect_tools(["web_search", "fetch_url"])

        # Run with mock workshop
        result = await scenario.run(mock_workshop)

        assert result.passed
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        user_inputs: List[str] = None,
        setup: Callable = None,
        teardown: Callable = None,
    ):
        self.name = name
        self.description = description
        self.user_inputs = user_inputs or []

        self._setup = setup
        self._teardown = teardown

        self.expectations = ScenarioExpectations()
        self.traces: List[ContextPipelineTrace] = []
        self.validations: List[ValidationCheck] = []

        self._mock_factory = MockFactory()
        self._mocks = None

        # Custom validators
        self._custom_validators: List[Callable[[ContextPipelineTrace], ValidationCheck]] = []

    # =========================================================================
    # Expectation Configuration
    # =========================================================================

    def expect_skill(self, skill_name: str) -> "E2EScenario":
        """Expect a specific skill to be routed to."""
        self.expectations.expected_skills.append(skill_name)
        return self

    def expect_skills(self, skills: List[str]) -> "E2EScenario":
        """Expect specific skills in order."""
        self.expectations.expected_skills.extend(skills)
        return self

    def expect_routing_method(self, method: str) -> "E2EScenario":
        """Expect a specific routing method."""
        self.expectations.expected_routing_methods.append(method)
        return self

    def expect_min_confidence(self, confidence: float) -> "E2EScenario":
        """Expect minimum routing confidence."""
        self.expectations.min_routing_confidence = confidence
        return self

    def expect_tool(self, tool_name: str) -> "E2EScenario":
        """Expect a specific tool to be called."""
        self.expectations.expected_tools.append(tool_name)
        return self

    def expect_tools(self, tools: List[str]) -> "E2EScenario":
        """Expect specific tools to be called."""
        self.expectations.expected_tools.extend(tools)
        return self

    def expect_tool_count(self, count: int) -> "E2EScenario":
        """Expect exact number of tool calls."""
        self.expectations.expected_tool_count = count
        return self

    def expect_tool_count_range(self, min_count: int, max_count: int) -> "E2EScenario":
        """Expect tool count within range."""
        self.expectations.min_tool_count = min_count
        self.expectations.max_tool_count = max_count
        return self

    def expect_telos(self, loaded: bool = True) -> "E2EScenario":
        """Expect Telos context to be loaded (or not)."""
        self.expectations.expect_telos_loaded = loaded
        return self

    def expect_task_context(self, present: bool = True) -> "E2EScenario":
        """Expect task context to be present (or not)."""
        self.expectations.expect_task_context = present
        return self

    def expect_auto_context(self, present: bool = True) -> "E2EScenario":
        """Expect automatic context to be injected (or not)."""
        self.expectations.expect_auto_context = present
        return self

    def expect_memory_search(self, performed: bool = True) -> "E2EScenario":
        """Expect memory search to be performed (or not)."""
        self.expectations.expect_memory_search = performed
        return self

    def expect_response_contains(self, text: str) -> "E2EScenario":
        """Expect response to contain specific text."""
        self.expectations.response_contains.append(text)
        return self

    def expect_response_not_contains(self, text: str) -> "E2EScenario":
        """Expect response to NOT contain specific text."""
        self.expectations.response_not_contains.append(text)
        return self

    def expect_min_response_length(self, length: int) -> "E2EScenario":
        """Expect minimum response length."""
        self.expectations.min_response_length = length
        return self

    def expect_max_duration(self, ms: int) -> "E2EScenario":
        """Expect maximum total duration."""
        self.expectations.max_duration_ms = ms
        return self

    def add_validator(
        self,
        validator: Callable[[ContextPipelineTrace], ValidationCheck]
    ) -> "E2EScenario":
        """Add a custom validator function."""
        self._custom_validators.append(validator)
        return self

    # =========================================================================
    # Mock Configuration
    # =========================================================================

    def with_mocks(self, mocks: MockFactory.MockSet) -> "E2EScenario":
        """Use provided mocks instead of creating new ones."""
        self._mocks = mocks
        return self

    def configure_claude_response(
        self,
        pattern: str,
        content: str,
        tool_calls: List[Tuple[str, Dict]] = None
    ) -> "E2EScenario":
        """Configure a mock Claude response."""
        if not self._mocks:
            self._mocks = self._mock_factory.create_all()

        mock_tools = [MockToolCall(t, a) for t, a in (tool_calls or [])]
        self._mocks.claude.add_response(
            pattern,
            MockLLMResponse(content=content, tool_calls=mock_tools)
        )
        return self

    def configure_routing(
        self,
        pattern: str,
        skill: str,
        confidence: float = 0.9,
        method: str = "semantic_direct"
    ) -> "E2EScenario":
        """Configure mock routing for a pattern."""
        if not self._mocks:
            self._mocks = self._mock_factory.create_all()

        from .mocks import MockSemanticMatch
        self._mocks.router.add_route(
            pattern,
            MockSemanticMatch(skill, confidence, method=method)
        )
        return self

    # =========================================================================
    # Execution
    # =========================================================================

    async def run(self, workshop=None) -> E2EResult:
        """
        Execute the scenario and return results.

        Args:
            workshop: Optional Workshop instance. If not provided,
                     uses mock infrastructure.

        Returns:
            E2EResult with traces and validations
        """
        start_time = datetime.now()
        self.traces.clear()
        self.validations.clear()

        try:
            # Setup
            if self._setup:
                await self._maybe_await(self._setup())

            # Initialize mocks if needed
            if not self._mocks:
                self._mocks = self._mock_factory.create_all()

            # Execute each user input
            for i, user_input in enumerate(self.user_inputs):
                trace = await self._execute_input(user_input, workshop, i + 1)
                self.traces.append(trace)

            # Validate results
            self._run_validations()

            # Calculate duration
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # Determine overall pass/fail
            passed = all(v.result != ValidationResult.FAILED for v in self.validations)

            return E2EResult(
                scenario_name=self.name,
                passed=passed,
                traces=self.traces,
                validations=self.validations,
                duration_ms=duration_ms,
            )

        except Exception as e:
            import traceback
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            return E2EResult(
                scenario_name=self.name,
                passed=False,
                traces=self.traces,
                validations=self.validations,
                duration_ms=duration_ms,
                error=str(e),
                error_traceback=traceback.format_exc(),
            )

        finally:
            # Teardown
            if self._teardown:
                await self._maybe_await(self._teardown())

    async def _execute_input(
        self,
        user_input: str,
        workshop,
        turn_number: int
    ) -> ContextPipelineTrace:
        """Execute a single user input and capture trace."""
        trace = ContextPipelineTrace(
            user_input_raw=user_input,
            session_id=f"e2e_test_session_{self.name}",
        )

        if workshop:
            # Real workshop execution with tracing
            # TODO: Integrate with actual Workshop class
            response = await workshop.process_input(user_input)
            trace.response_final = response
            trace.complete(response)
        else:
            # Mock execution
            trace = await self._mock_execute(user_input, trace)

        return trace

    async def _mock_execute(
        self,
        user_input: str,
        trace: ContextPipelineTrace
    ) -> ContextPipelineTrace:
        """Execute using mock infrastructure."""
        trace.set_stage(TraceStage.INITIALIZED)

        # Simulate context loading
        trace.set_stage(TraceStage.CONTEXT_TELOS)
        trace.telos_loaded = True
        trace.telos_total_chars = 5000
        trace.duration_context_ms = 50

        # Simulate routing
        trace.set_stage(TraceStage.ROUTING_SEMANTIC)
        if self._mocks:
            match = await self._mocks.router.route(user_input, {})
            trace.routing.semantic_score = match.confidence
            trace.routing.semantic_matched_skill = match.skill_name
            trace.routing.semantic_matched_utterance = match.matched_utterance
            trace.routing.final_skill = match.skill_name
            trace.routing.final_method = match.method
            trace.routing.final_confidence = match.confidence

        trace.set_stage(TraceStage.ROUTING_DECIDED)
        trace.routing_total_duration_ms = 25

        # Simulate LLM invocation
        trace.set_stage(TraceStage.CLAUDE_INVOKED)
        if self._mocks:
            messages = [{"role": "user", "content": user_input}]
            response = await self._mocks.claude.query(messages)

            llm_inv = LLMInvocation(
                invocation_id=f"mock_llm_{len(trace.llm_invocations) + 1}",
                iteration=1,
                system_prompt_length=10000,
                user_message=user_input,
                user_message_length=len(user_input),
                response_raw=response["content"],
                response_length=len(response["content"]),
                tool_calls_detected=len(response.get("tool_calls", [])),
                duration_ms=500,
            )
            trace.add_llm_invocation(llm_inv)

            # Simulate tool execution
            for tc in response.get("tool_calls", []):
                trace.set_stage(TraceStage.TOOL_EXECUTING)
                tool_detail = ToolCallDetail(
                    call_id=f"mock_tool_{trace.tool_total_calls + 1}",
                    tool_name=tc["tool"],
                    skill_name=trace.routing.final_skill,
                    args_from_llm=tc["args"],
                    args_normalized=tc["args"],
                    duration_ms=200,
                    result="Mock tool result",
                    result_length=100,
                    result_type="success",
                )
                trace.add_tool_call(tool_detail)

            trace.response_raw = response["content"]
            trace.response_final = response["content"]
            trace.response_length = len(response["content"])

        trace.set_stage(TraceStage.RESPONSE_GENERATED)
        trace.duration_total_ms = trace.duration_context_ms + trace.routing_total_duration_ms + trace.llm_total_duration_ms + trace.tool_total_duration_ms

        trace.complete(trace.response_final)
        return trace

    def _run_validations(self):
        """Run all validations against captured traces."""
        # Validate skills
        if self.expectations.expected_skills:
            actual_skills = [t.routing.final_skill for t in self.traces]
            for i, expected in enumerate(self.expectations.expected_skills):
                if i < len(actual_skills):
                    passed = actual_skills[i] == expected
                    self.validations.append(ValidationCheck(
                        name=f"skill_turn_{i+1}",
                        result=ValidationResult.PASSED if passed else ValidationResult.FAILED,
                        expected=expected,
                        actual=actual_skills[i] if i < len(actual_skills) else None,
                        message=f"Expected skill '{expected}' on turn {i+1}",
                    ))
                else:
                    self.validations.append(ValidationCheck(
                        name=f"skill_turn_{i+1}",
                        result=ValidationResult.FAILED,
                        expected=expected,
                        actual=None,
                        message=f"No trace for turn {i+1}",
                    ))

        # Validate routing methods
        if self.expectations.expected_routing_methods:
            actual_methods = [t.routing.final_method for t in self.traces]
            for i, expected in enumerate(self.expectations.expected_routing_methods):
                if i < len(actual_methods):
                    passed = actual_methods[i] == expected
                    self.validations.append(ValidationCheck(
                        name=f"routing_method_turn_{i+1}",
                        result=ValidationResult.PASSED if passed else ValidationResult.FAILED,
                        expected=expected,
                        actual=actual_methods[i] if i < len(actual_methods) else None,
                    ))

        # Validate minimum confidence
        if self.expectations.min_routing_confidence > 0:
            for i, trace in enumerate(self.traces):
                passed = trace.routing.final_confidence >= self.expectations.min_routing_confidence
                self.validations.append(ValidationCheck(
                    name=f"routing_confidence_turn_{i+1}",
                    result=ValidationResult.PASSED if passed else ValidationResult.FAILED,
                    expected=f">= {self.expectations.min_routing_confidence}",
                    actual=trace.routing.final_confidence,
                ))

        # Validate tools
        if self.expectations.expected_tools:
            all_tools = []
            for trace in self.traces:
                all_tools.extend([tc.tool_name for tc in trace.tool_calls])

            for expected_tool in self.expectations.expected_tools:
                passed = expected_tool in all_tools
                self.validations.append(ValidationCheck(
                    name=f"tool_called_{expected_tool}",
                    result=ValidationResult.PASSED if passed else ValidationResult.FAILED,
                    expected=expected_tool,
                    actual=all_tools,
                    message=f"Expected tool '{expected_tool}' to be called",
                ))

        # Validate tool count
        total_tools = sum(t.tool_total_calls for t in self.traces)
        if self.expectations.expected_tool_count is not None:
            passed = total_tools == self.expectations.expected_tool_count
            self.validations.append(ValidationCheck(
                name="tool_count_exact",
                result=ValidationResult.PASSED if passed else ValidationResult.FAILED,
                expected=self.expectations.expected_tool_count,
                actual=total_tools,
            ))

        if self.expectations.min_tool_count > 0 or self.expectations.max_tool_count < 100:
            passed = self.expectations.min_tool_count <= total_tools <= self.expectations.max_tool_count
            self.validations.append(ValidationCheck(
                name="tool_count_range",
                result=ValidationResult.PASSED if passed else ValidationResult.FAILED,
                expected=f"{self.expectations.min_tool_count}-{self.expectations.max_tool_count}",
                actual=total_tools,
            ))

        # Validate context expectations
        if self.expectations.expect_telos_loaded is not None:
            for i, trace in enumerate(self.traces):
                passed = trace.telos_loaded == self.expectations.expect_telos_loaded
                self.validations.append(ValidationCheck(
                    name=f"telos_loaded_turn_{i+1}",
                    result=ValidationResult.PASSED if passed else ValidationResult.FAILED,
                    expected=self.expectations.expect_telos_loaded,
                    actual=trace.telos_loaded,
                ))

        # Validate response content
        for trace in self.traces:
            for expected_text in self.expectations.response_contains:
                passed = expected_text.lower() in trace.response_final.lower()
                self.validations.append(ValidationCheck(
                    name=f"response_contains_{expected_text[:20]}",
                    result=ValidationResult.PASSED if passed else ValidationResult.FAILED,
                    expected=f"contains '{expected_text}'",
                    actual=f"response length {len(trace.response_final)}",
                ))

            for forbidden_text in self.expectations.response_not_contains:
                passed = forbidden_text.lower() not in trace.response_final.lower()
                self.validations.append(ValidationCheck(
                    name=f"response_not_contains_{forbidden_text[:20]}",
                    result=ValidationResult.PASSED if passed else ValidationResult.FAILED,
                    expected=f"not contains '{forbidden_text}'",
                    actual=f"found in response" if not passed else "not found",
                ))

        # Validate response length
        if self.expectations.min_response_length > 0:
            for i, trace in enumerate(self.traces):
                passed = trace.response_length >= self.expectations.min_response_length
                self.validations.append(ValidationCheck(
                    name=f"response_length_turn_{i+1}",
                    result=ValidationResult.PASSED if passed else ValidationResult.FAILED,
                    expected=f">= {self.expectations.min_response_length}",
                    actual=trace.response_length,
                ))

        # Validate duration
        total_duration = sum(t.duration_total_ms for t in self.traces)
        if total_duration > self.expectations.max_duration_ms:
            self.validations.append(ValidationCheck(
                name="total_duration",
                result=ValidationResult.FAILED,
                expected=f"<= {self.expectations.max_duration_ms}ms",
                actual=f"{total_duration}ms",
            ))

        # Run custom validators
        for validator in self._custom_validators:
            for trace in self.traces:
                check = validator(trace)
                self.validations.append(check)

    async def _maybe_await(self, result):
        """Await result if it's a coroutine."""
        if asyncio.iscoroutine(result):
            return await result
        return result


# =============================================================================
# Scenario Builders
# =============================================================================

class ScenarioBuilder:
    """
    Factory for creating common scenario types.

    Usage:
        builder = ScenarioBuilder()
        scenario = builder.research_scenario("research Python tutorials")
        result = await scenario.run()
    """

    def research_scenario(
        self,
        query: str,
        expected_sources: int = 1
    ) -> E2EScenario:
        """Create a research scenario."""
        scenario = E2EScenario(
            name="research_flow",
            description=f"Research: {query}",
            user_inputs=[query],
        )

        scenario.expect_skill("Research")
        scenario.expect_routing_method("semantic_direct")
        scenario.expect_min_confidence(0.8)
        scenario.expect_tools(["web_search"])
        scenario.expect_tool_count_range(1, 10)

        # Configure mocks
        scenario.configure_routing("research|search|find", "Research", 0.92)
        scenario.configure_claude_response(
            "research|search",
            "I'll search for information on that topic.",
            [("web_search", {"query": query})]
        )

        return scenario

    def file_operations_scenario(
        self,
        query: str,
        expected_tool: str = "list_files"
    ) -> E2EScenario:
        """Create a file operations scenario."""
        scenario = E2EScenario(
            name="file_ops_flow",
            description=f"File ops: {query}",
            user_inputs=[query],
        )

        scenario.expect_skill("FileOperations")
        scenario.expect_tool(expected_tool)

        scenario.configure_routing("file|list|read|directory", "FileOperations", 0.88)
        scenario.configure_claude_response(
            "file|list|read",
            "I'll examine the files.",
            [(expected_tool, {"path": "/tmp"})]
        )

        return scenario

    def memory_scenario(
        self,
        remember_text: str,
        recall_query: str
    ) -> E2EScenario:
        """Create a memory scenario with remember/recall."""
        scenario = E2EScenario(
            name="memory_flow",
            description=f"Remember and recall",
            user_inputs=[
                f"remember: {remember_text}",
                f"what did I say about {recall_query}?",
            ],
        )

        scenario.expect_skills(["Memory", "Memory"])
        scenario.expect_tools(["remember", "recall"])

        scenario.configure_routing("remember|recall|memory", "Memory", 0.90)
        scenario.configure_claude_response(
            "remember",
            "I'll remember that.",
            [("remember", {"content": remember_text})]
        )

        return scenario

    def multi_turn_scenario(
        self,
        turns: List[str],
        expected_skills: List[str] = None
    ) -> E2EScenario:
        """Create a multi-turn conversation scenario."""
        scenario = E2EScenario(
            name="multi_turn_conversation",
            description=f"Multi-turn: {len(turns)} turns",
            user_inputs=turns,
        )

        if expected_skills:
            scenario.expect_skills(expected_skills)

        return scenario

    def error_scenario(
        self,
        query: str,
        expected_error_stage: str = None
    ) -> E2EScenario:
        """Create a scenario expected to produce an error."""
        scenario = E2EScenario(
            name="error_scenario",
            description=f"Error case: {query}",
            user_inputs=[query],
        )

        # Configure to fail
        scenario.configure_routing(".*", "Unknown", 0.1)

        return scenario
