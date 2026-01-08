"""
Pytest Fixtures for E2E Testing

Provides shared fixtures for test setup, mock configuration,
and cleanup across all E2E tests.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Generator, AsyncGenerator
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.e2e.mocks import (
    MockClaudeBridge,
    MockHTTPClient,
    MockMemorySystem,
    MockSemanticRouter,
    MockTaskManager,
    MockFactory,
    MockLLMResponse,
    MockToolCall,
    MockSemanticMatch,
)
from tests.e2e.context_tracer import ContextPipelineTrace, PipelineTracer
from tests.e2e.scenarios import E2EScenario, ScenarioBuilder
from tests.e2e.visualizer import TraceVisualizer, TraceReporter


# =============================================================================
# Event Loop Configuration
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_factory() -> MockFactory:
    """Create a MockFactory instance."""
    return MockFactory()


@pytest.fixture
def mocks(mock_factory: MockFactory) -> MockFactory.MockSet:
    """Create a coordinated set of mocks."""
    mock_set = mock_factory.create_all()
    yield mock_set
    mock_set.reset_all()


@pytest.fixture
def mock_claude() -> MockClaudeBridge:
    """Create a standalone MockClaudeBridge."""
    bridge = MockClaudeBridge()
    yield bridge
    bridge.reset()


@pytest.fixture
def mock_http() -> MockHTTPClient:
    """Create a standalone MockHTTPClient."""
    client = MockHTTPClient()
    yield client
    client.reset()


@pytest.fixture
def mock_memory() -> MockMemorySystem:
    """Create a standalone MockMemorySystem."""
    memory = MockMemorySystem()
    yield memory
    memory.reset()


@pytest.fixture
def mock_router() -> MockSemanticRouter:
    """Create a standalone MockSemanticRouter."""
    router = MockSemanticRouter()
    yield router
    router.reset()


@pytest.fixture
def mock_tasks() -> MockTaskManager:
    """Create a standalone MockTaskManager."""
    tasks = MockTaskManager()
    yield tasks
    tasks.reset()


# =============================================================================
# Research Scenario Mocks
# =============================================================================

@pytest.fixture
def research_mocks(mock_factory: MockFactory) -> MockFactory.MockSet:
    """Create mocks configured for research scenarios."""
    return mock_factory.create_research_scenario()


@pytest.fixture
def file_ops_mocks(mock_factory: MockFactory) -> MockFactory.MockSet:
    """Create mocks configured for file operations scenarios."""
    return mock_factory.create_file_ops_scenario()


# =============================================================================
# Tracer Fixtures
# =============================================================================

@pytest.fixture
def trace() -> ContextPipelineTrace:
    """Create a fresh pipeline trace."""
    return ContextPipelineTrace(
        user_input_raw="test input",
        session_id="test_session_001",
    )


@pytest.fixture
def tracer():
    """Create a PipelineTracer for context manager usage."""
    def _create_tracer(user_input: str, session_id: str = "test_session"):
        return PipelineTracer(user_input, session_id)
    return _create_tracer


@pytest.fixture
def visualizer() -> TraceVisualizer:
    """Create a TraceVisualizer."""
    return TraceVisualizer()


@pytest.fixture
def reporter() -> TraceReporter:
    """Create a TraceReporter."""
    return TraceReporter()


# =============================================================================
# Scenario Fixtures
# =============================================================================

@pytest.fixture
def scenario_builder() -> ScenarioBuilder:
    """Create a ScenarioBuilder for creating scenarios."""
    return ScenarioBuilder()


@pytest.fixture
def research_scenario(scenario_builder: ScenarioBuilder) -> E2EScenario:
    """Create a pre-configured research scenario."""
    return scenario_builder.research_scenario("test research query")


@pytest.fixture
def file_ops_scenario(scenario_builder: ScenarioBuilder) -> E2EScenario:
    """Create a pre-configured file operations scenario."""
    return scenario_builder.file_operations_scenario("list files in /tmp")


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test artifacts."""
    tmpdir = Path(tempfile.mkdtemp(prefix="workshop_e2e_"))
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def temp_workshop_dir(temp_dir: Path) -> Path:
    """Create a temporary .workshop directory structure."""
    workshop_dir = temp_dir / ".workshop"
    workshop_dir.mkdir()

    # Create subdirectories
    (workshop_dir / "Skills").mkdir()
    (workshop_dir / "Telos").mkdir()
    (workshop_dir / "sessions").mkdir()
    (workshop_dir / "tasks").mkdir()

    # Create minimal Telos files
    (workshop_dir / "Telos" / "profile.md").write_text("# Test Profile\nA test user.")
    (workshop_dir / "Telos" / "goals.md").write_text("# Test Goals\n- Goal 1")
    (workshop_dir / "Telos" / "mission.md").write_text("# Test Mission\nTest mission.")

    return workshop_dir


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_messages():
    """Sample conversation messages."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "Can you help me research Python?"},
    ]


@pytest.fixture
def sample_tool_calls():
    """Sample tool calls for testing."""
    return [
        MockToolCall("web_search", {"query": "Python tutorials"}),
        MockToolCall("fetch_url", {"url": "https://python.org"}),
        MockToolCall("save_research_output", {"filename": "python.md", "content": "# Python\n..."}),
    ]


@pytest.fixture
def sample_routing_results():
    """Sample routing results for different query types."""
    return {
        "research": MockSemanticMatch("Research", 0.92, "research [topic]", "semantic_direct"),
        "files": MockSemanticMatch("FileOperations", 0.88, "list files", "semantic_direct"),
        "memory": MockSemanticMatch("Memory", 0.90, "remember this", "semantic_direct"),
        "chat": MockSemanticMatch("chat", 0.65, "hello", "pattern_chat"),
        "ambiguous": MockSemanticMatch("clarify", 0.35, "", "fallback"),
    }


# =============================================================================
# Report Output Fixtures
# =============================================================================

@pytest.fixture
def report_dir(temp_dir: Path) -> Path:
    """Create a directory for test reports."""
    report_path = temp_dir / "reports"
    report_path.mkdir()
    return report_path


@pytest.fixture
def save_trace_report(report_dir: Path, visualizer: TraceVisualizer):
    """Factory fixture to save trace reports."""
    def _save(trace: ContextPipelineTrace, name: str = "trace"):
        # Save ASCII
        ascii_path = report_dir / f"{name}.txt"
        ascii_path.write_text(visualizer.to_ascii(trace))

        # Save markdown
        md_path = report_dir / f"{name}.md"
        md_path.write_text(visualizer.to_markdown(trace))

        # Save JSON
        json_path = report_dir / f"{name}.json"
        json_path.write_text(trace.to_json())

        return {
            "ascii": ascii_path,
            "markdown": md_path,
            "json": json_path,
        }

    return _save


# =============================================================================
# Integration Test Markers
# =============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_ollama: marks tests that require Ollama"
    )
    config.addinivalue_line(
        "markers", "requires_claude: marks tests that require Claude Code CLI"
    )


# =============================================================================
# Async Test Helpers
# =============================================================================

@pytest.fixture
def run_async():
    """Helper to run async functions in sync context."""
    def _run(coro):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    return _run


# =============================================================================
# Mock Configuration Helpers
# =============================================================================

@pytest.fixture
def configure_research_flow(mocks: MockFactory.MockSet):
    """Configure mocks for a complete research flow."""
    # Configure routing
    mocks.router.add_route(
        "research|search|find|look up",
        MockSemanticMatch("Research", 0.92, "research [topic]", "semantic_direct")
    )

    # Configure Claude responses - sequence of tool calls
    mocks.claude.add_response_sequence([
        # First response: search
        MockLLMResponse(
            content="I'll search for information on that topic.",
            tool_calls=[MockToolCall("web_search", {"query": "test query"})]
        ),
        # Second response: fetch
        MockLLMResponse(
            content="Let me fetch that page.",
            tool_calls=[MockToolCall("fetch_url", {"url": "https://example.com"})]
        ),
        # Third response: synthesis
        MockLLMResponse(
            content="Based on my research, here are the findings...",
            tool_calls=[]
        ),
    ])

    return mocks


@pytest.fixture
def configure_memory_flow(mocks: MockFactory.MockSet):
    """Configure mocks for memory operations flow."""
    mocks.router.add_route(
        "remember|recall|memory|what did",
        MockSemanticMatch("Memory", 0.90, "remember this", "semantic_direct")
    )

    mocks.claude.add_response(
        "remember",
        MockLLMResponse(
            content="I'll remember that for you.",
            tool_calls=[MockToolCall("remember", {"content": "test memory"})]
        )
    )

    mocks.claude.add_response(
        "recall|what did",
        MockLLMResponse(
            content="Here's what I remember...",
            tool_calls=[MockToolCall("recall", {"query": "test"})]
        )
    )

    return mocks


# =============================================================================
# Cleanup
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_singletons():
    """Reset any singleton state between tests."""
    yield
    # Add singleton reset logic here if needed
    pass
