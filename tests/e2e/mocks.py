"""
Mock Infrastructure for E2E Testing

Provides mock implementations of external services and components:
- MockClaudeBridge: Simulates Claude Code CLI responses
- MockHTTPClient: Simulates web requests for research tools
- MockMemorySystem: Simulates memory without real database
- MockSemanticRouter: Simulates routing with predefined scores
"""

import json
import re
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, AsyncMock
import logging

log = logging.getLogger(__name__)


# =============================================================================
# Mock Response Builders
# =============================================================================

@dataclass
class MockToolCall:
    """A tool call to include in mock response."""
    tool: str
    args: Dict[str, Any]

    def to_xml(self) -> str:
        """Format as <tool_call> XML."""
        return f'<tool_call>{json.dumps({"tool": self.tool, "args": self.args})}</tool_call>'


@dataclass
class MockLLMResponse:
    """A predefined LLM response for testing."""
    content: str
    tool_calls: List[MockToolCall] = field(default_factory=list)
    session_id: str = "mock_session_001"

    def get_full_content(self) -> str:
        """Get content with embedded tool calls."""
        if not self.tool_calls:
            return self.content

        tool_xml = "\n".join(tc.to_xml() for tc in self.tool_calls)
        return f"{self.content}\n\n{tool_xml}"


# =============================================================================
# MockClaudeBridge
# =============================================================================

class MockClaudeBridge:
    """
    Mock implementation of ClaudeCodeBridge for testing.

    Allows predefined responses to be returned based on input patterns,
    while capturing all invocations for verification.

    Usage:
        mock = MockClaudeBridge()

        # Add expected responses
        mock.add_response(
            pattern="research",  # Trigger on input containing "research"
            response=MockLLMResponse(
                content="I'll search for that.",
                tool_calls=[MockToolCall("web_search", {"query": "test"})]
            )
        )

        # Use in tests
        response = await mock.query(messages, system_prompt="...")

        # Verify
        assert len(mock.invocations) == 1
        assert mock.invocations[0]["system_prompt_length"] > 0
    """

    def __init__(
        self,
        working_dir: str = "/tmp/mock_workshop",
        timeout_seconds: int = 120,
        default_model: str = "mock-claude-opus-4-5"
    ):
        self.working_dir = working_dir
        self.timeout = timeout_seconds
        self.default_model = default_model

        # Response configuration
        self._responses: List[Tuple[str, MockLLMResponse]] = []
        self._default_response = MockLLMResponse(
            content="I understand your request.",
            tool_calls=[]
        )
        self._response_sequence: List[MockLLMResponse] = []
        self._sequence_index = 0

        # Session state
        self._session_id: str = "mock_session_001"
        self._turn_count: int = 0

        # Invocation tracking
        self.invocations: List[Dict[str, Any]] = []
        self.tool_call_history: List[Dict[str, Any]] = []

        # Callbacks for custom behavior
        self._on_query: Optional[Callable] = None

    def add_response(
        self,
        pattern: str,
        response: MockLLMResponse,
        case_sensitive: bool = False
    ):
        """
        Add a response that triggers when input matches pattern.

        Args:
            pattern: Regex pattern or substring to match
            response: The response to return
            case_sensitive: Whether pattern match is case-sensitive
        """
        flags = 0 if case_sensitive else re.IGNORECASE
        compiled = re.compile(pattern, flags)
        self._responses.append((compiled, response))

    def add_response_sequence(self, responses: List[MockLLMResponse]):
        """
        Add a sequence of responses to return in order.

        Each call to query() returns the next response in sequence.
        """
        self._response_sequence = responses
        self._sequence_index = 0

    def set_default_response(self, response: MockLLMResponse):
        """Set the default response when no pattern matches."""
        self._default_response = response

    def on_query(self, callback: Callable):
        """Set a callback to be invoked on each query (for custom logic)."""
        self._on_query = callback

    def reset(self):
        """Reset all state for a new test."""
        self.invocations.clear()
        self.tool_call_history.clear()
        self._turn_count = 0
        self._sequence_index = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    async def query(
        self,
        messages: List[Dict],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_turns: int = 1,
        resume_session: bool = False,
        continue_session: bool = False,
    ) -> Dict[str, Any]:
        """
        Mock query method matching ClaudeCodeBridge.query() signature.

        Returns a response dict compatible with Workshop's expectations:
            {"content": "...", "tool_calls": [...]}
        """
        self._turn_count += 1

        # Extract user message for pattern matching
        user_content = self._extract_user_content(messages)
        extracted_system = self._extract_system_prompt(messages)

        # Record invocation
        invocation = {
            "timestamp": datetime.now().isoformat(),
            "messages": messages,
            "system_prompt": system_prompt or extracted_system,
            "system_prompt_length": len(system_prompt or extracted_system or ""),
            "model": model or self.default_model,
            "max_turns": max_turns,
            "resume_session": resume_session,
            "continue_session": continue_session,
            "user_content": user_content,
            "turn_number": self._turn_count,
        }
        self.invocations.append(invocation)

        # Call custom callback if set
        if self._on_query:
            custom_response = self._on_query(messages, system_prompt)
            if custom_response:
                return self._format_response(custom_response)

        # Find matching response
        response = self._find_response(user_content)

        return self._format_response(response)

    def _extract_user_content(self, messages: List[Dict]) -> str:
        """Extract the last user message content."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def _extract_system_prompt(self, messages: List[Dict]) -> Optional[str]:
        """Extract system prompt from messages."""
        for msg in messages:
            if msg.get("role") == "system":
                return msg.get("content", "")
        return None

    def _find_response(self, user_content: str) -> MockLLMResponse:
        """Find appropriate response based on user content."""
        # Check sequence first
        if self._response_sequence and self._sequence_index < len(self._response_sequence):
            response = self._response_sequence[self._sequence_index]
            self._sequence_index += 1
            return response

        # Check pattern matches
        for pattern, response in self._responses:
            if pattern.search(user_content):
                return response

        # Return default
        return self._default_response

    def _format_response(self, response: MockLLMResponse) -> Dict[str, Any]:
        """Format MockLLMResponse to Workshop-compatible dict."""
        tool_calls = []
        for tc in response.tool_calls:
            tool_calls.append({
                "tool": tc.tool,
                "args": tc.args
            })
            self.tool_call_history.append({
                "tool": tc.tool,
                "args": tc.args,
                "turn": self._turn_count,
            })

        return {
            "content": response.get_full_content(),
            "tool_calls": tool_calls,
            "session_id": response.session_id,
        }

    # Compatibility methods
    def _verify_installation(self):
        """No-op for mock."""
        pass

    def _clean_env(self) -> Dict[str, str]:
        """Return empty env for mock."""
        return {}


# =============================================================================
# MockHTTPClient
# =============================================================================

@dataclass
class MockHTTPResponse:
    """A mock HTTP response."""
    status_code: int = 200
    content: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    url: str = ""

    @property
    def text(self) -> str:
        return self.content


class MockHTTPClient:
    """
    Mock HTTP client for testing web-based tools.

    Allows predefined responses for specific URLs or patterns.

    Usage:
        mock = MockHTTPClient()
        mock.add_response("https://example.com", MockHTTPResponse(
            content="<html>Example content</html>"
        ))

        # In tool execution
        response = await mock.get("https://example.com")
        assert response.status_code == 200
    """

    def __init__(self):
        self._responses: Dict[str, MockHTTPResponse] = {}
        self._pattern_responses: List[Tuple[re.Pattern, MockHTTPResponse]] = []
        self._default_response = MockHTTPResponse(
            status_code=200,
            content="<html><body>Mock content</body></html>"
        )
        self.requests: List[Dict[str, Any]] = []

    def add_response(self, url: str, response: MockHTTPResponse):
        """Add a response for exact URL match."""
        self._responses[url] = response

    def add_pattern_response(self, pattern: str, response: MockHTTPResponse):
        """Add a response for URL pattern match."""
        compiled = re.compile(pattern, re.IGNORECASE)
        self._pattern_responses.append((compiled, response))

    def set_default_response(self, response: MockHTTPResponse):
        """Set default response for unmatched URLs."""
        self._default_response = response

    def reset(self):
        """Reset request history."""
        self.requests.clear()

    async def get(self, url: str, **kwargs) -> MockHTTPResponse:
        """Mock GET request."""
        return await self._request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> MockHTTPResponse:
        """Mock POST request."""
        return await self._request("POST", url, **kwargs)

    async def _request(self, method: str, url: str, **kwargs) -> MockHTTPResponse:
        """Process a mock request."""
        self.requests.append({
            "method": method,
            "url": url,
            "kwargs": kwargs,
            "timestamp": datetime.now().isoformat(),
        })

        # Check exact match
        if url in self._responses:
            response = self._responses[url]
            response.url = url
            return response

        # Check pattern match
        for pattern, response in self._pattern_responses:
            if pattern.search(url):
                response.url = url
                return response

        # Return default
        default = self._default_response
        default.url = url
        return default


# =============================================================================
# MockMemorySystem
# =============================================================================

class MockMemorySystem:
    """
    Mock implementation of MemorySystem for testing.

    Provides in-memory storage without SQLite/ChromaDB dependencies.

    Usage:
        mock = MockMemorySystem()
        mock.add_memory("Python is great", category="tech")

        results = mock.search_memories("programming")
        assert len(results) > 0
    """

    def __init__(self):
        self._messages: List[Dict[str, Any]] = []
        self._memories: List[Dict[str, Any]] = []
        self._facts: Dict[str, str] = {}
        self._projects: Dict[str, Dict] = {}
        self._active_project: Optional[str] = None
        self._session_id: Optional[str] = None
        self.message_count: int = 0

    def start_session(self, session_id: str):
        """Start a new session."""
        self._session_id = session_id
        self._messages.clear()
        self.message_count = 0

    def end_session(self) -> Dict[str, Any]:
        """End current session."""
        summary = {
            "session_id": self._session_id,
            "message_count": self.message_count,
        }
        self._session_id = None
        return summary

    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        self._messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "session_id": self._session_id,
        })
        self.message_count += 1

    def get_recent_messages(self, n: int = 10) -> List[Dict]:
        """Get recent messages."""
        return self._messages[-n:]

    def add_memory(self, content: str, category: str = "general", metadata: Dict = None):
        """Add to long-term memory."""
        self._memories.append({
            "content": content,
            "category": category,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        })

    def search_memories(
        self,
        query: str,
        k: int = 5,
        category: Optional[str] = None,
        include_metadata: bool = False
    ) -> List[Any]:
        """
        Simple keyword-based memory search (mock doesn't do embeddings).
        """
        results = []
        query_lower = query.lower()

        for memory in self._memories:
            if category and memory["category"] != category:
                continue
            if query_lower in memory["content"].lower():
                if include_metadata:
                    results.append(memory)
                else:
                    results.append(memory["content"])

        return results[:k]

    def set_fact(self, key: str, value: str):
        """Store a fact."""
        self._facts[key] = value

    def get_fact(self, key: str) -> Optional[str]:
        """Retrieve a fact."""
        return self._facts.get(key)

    def get_all_facts(self) -> Dict[str, str]:
        """Get all facts."""
        return dict(self._facts)

    def set_active_project(self, name: str, path: str = "", description: str = ""):
        """Set active project."""
        self._active_project = name
        self._projects[name] = {
            "name": name,
            "path": path,
            "description": description,
        }

    def get_active_project(self) -> Optional[Dict]:
        """Get active project."""
        if self._active_project:
            return self._projects.get(self._active_project)
        return None

    def reset(self):
        """Reset all state."""
        self._messages.clear()
        self._memories.clear()
        self._facts.clear()
        self._projects.clear()
        self._active_project = None
        self._session_id = None
        self.message_count = 0


# =============================================================================
# MockSemanticRouter
# =============================================================================

@dataclass
class MockSemanticMatch:
    """A mock semantic match result."""
    skill_name: str
    confidence: float
    matched_utterance: str = ""
    method: str = "semantic"


class MockSemanticRouter:
    """
    Mock semantic router for testing routing decisions.

    Allows predefined routing results based on query patterns.

    Usage:
        mock = MockSemanticRouter()
        mock.add_route("research", MockSemanticMatch("Research", 0.91))
        mock.add_route("remember", MockSemanticMatch("Memory", 0.88))

        match = await mock.route("research Python tutorials")
        assert match.skill_name == "Research"
        assert match.confidence >= 0.85
    """

    def __init__(self):
        self._routes: List[Tuple[re.Pattern, MockSemanticMatch]] = []
        self._default = MockSemanticMatch("General", 0.3, method="fallback")
        self._initialized = False
        self.route_calls: List[Dict[str, Any]] = []

    async def initialize(self, force_rebuild: bool = False):
        """Mock initialization."""
        self._initialized = True

    def add_route(
        self,
        pattern: str,
        match: MockSemanticMatch,
        case_sensitive: bool = False
    ):
        """Add a route for a pattern."""
        flags = 0 if case_sensitive else re.IGNORECASE
        compiled = re.compile(pattern, flags)
        self._routes.append((compiled, match))

    def set_default(self, match: MockSemanticMatch):
        """Set default match when no pattern matches."""
        self._default = match

    async def route(
        self,
        query: str,
        context: Dict[str, Any] = None
    ) -> MockSemanticMatch:
        """Route a query to a skill."""
        self.route_calls.append({
            "query": query,
            "context": context,
            "timestamp": datetime.now().isoformat(),
        })

        for pattern, match in self._routes:
            if pattern.search(query):
                return match

        return self._default

    def reset(self):
        """Reset route call history."""
        self.route_calls.clear()


# =============================================================================
# MockTaskManager
# =============================================================================

@dataclass
class MockTask:
    """A mock task."""
    content: str
    status: str = "pending"  # pending, in_progress, completed
    active_form: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class MockTaskManager:
    """
    Mock task manager for testing task lifecycle.

    Usage:
        mock = MockTaskManager()
        mock.write_tasks([
            {"content": "Research topic", "status": "in_progress", "active_form": "Researching..."}
        ])

        tasks = mock.get_tasks()
        assert len(tasks) == 1
    """

    def __init__(self):
        self._tasks: List[MockTask] = []
        self._session_id: Optional[str] = None
        self._original_request: str = ""
        self._work_evidence: List[str] = []

    def bind_to_session(self, session_id: str):
        """Bind to a session."""
        self._session_id = session_id

    def write_tasks(
        self,
        tasks: List[Dict[str, Any]],
        original_request: str = ""
    ):
        """Write tasks."""
        self._original_request = original_request
        self._tasks = [
            MockTask(
                content=t["content"],
                status=t.get("status", "pending"),
                active_form=t.get("active_form", t["content"]),
            )
            for t in tasks
        ]

    def get_tasks(self) -> List[MockTask]:
        """Get all tasks."""
        return self._tasks

    def get_current_task(self) -> Optional[MockTask]:
        """Get the current in-progress task."""
        for task in self._tasks:
            if task.status == "in_progress":
                return task
        return None

    def advance_task(self, work_evidence: List[str] = None):
        """Advance to next task."""
        if work_evidence:
            self._work_evidence.extend(work_evidence)

        current = self.get_current_task()
        if current:
            current.status = "completed"

        # Find next pending and mark in_progress
        for task in self._tasks:
            if task.status == "pending":
                task.status = "in_progress"
                break

    def clear_tasks(self):
        """Clear all tasks."""
        self._tasks.clear()
        self._original_request = ""
        self._work_evidence.clear()

    def format_for_context(self) -> str:
        """Format tasks for LLM context."""
        if not self._tasks:
            return ""

        lines = ["=== ACTIVE TASK CONTEXT ==="]
        if self._original_request:
            lines.append(f"Working on: {self._original_request}")
        lines.append("")
        lines.append("Task Progress:")

        for i, task in enumerate(self._tasks, 1):
            icon = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
            lines.append(f"  {icon.get(task.status, '[ ]')} {i}. {task.content}")

        current = self.get_current_task()
        if current:
            lines.append("")
            lines.append(f"CURRENT: {current.active_form}")

        lines.append("=== END TASK CONTEXT ===")
        return "\n".join(lines)

    def reset(self):
        """Reset all state."""
        self._tasks.clear()
        self._session_id = None
        self._original_request = ""
        self._work_evidence.clear()


# =============================================================================
# Mock Factory
# =============================================================================

class MockFactory:
    """
    Factory for creating coordinated mock instances.

    Usage:
        factory = MockFactory()
        mocks = factory.create_all()

        # Use mocks in tests
        response = await mocks.claude.query(messages)
        memory_results = mocks.memory.search_memories("test")
    """

    @dataclass
    class MockSet:
        """Collection of coordinated mocks."""
        claude: MockClaudeBridge
        http: MockHTTPClient
        memory: MockMemorySystem
        router: MockSemanticRouter
        tasks: MockTaskManager

        def reset_all(self):
            """Reset all mocks."""
            self.claude.reset()
            self.http.reset()
            self.memory.reset()
            self.router.reset()
            self.tasks.reset()

    def create_all(self) -> MockSet:
        """Create a coordinated set of mocks."""
        return self.MockSet(
            claude=MockClaudeBridge(),
            http=MockHTTPClient(),
            memory=MockMemorySystem(),
            router=MockSemanticRouter(),
            tasks=MockTaskManager(),
        )

    def create_research_scenario(self) -> MockSet:
        """Create mocks configured for a research scenario."""
        mocks = self.create_all()

        # Configure Claude to emit web_search tool call
        mocks.claude.add_response(
            "research|search|find|look up",
            MockLLMResponse(
                content="I'll search for information on that topic.",
                tool_calls=[
                    MockToolCall("web_search", {"query": "mock search query"})
                ]
            )
        )

        # Add follow-up synthesis response
        mocks.claude.add_response_sequence([
            MockLLMResponse(
                content="I'll search for that.",
                tool_calls=[MockToolCall("web_search", {"query": "test"})]
            ),
            MockLLMResponse(
                content="Based on my research, here are the findings...",
                tool_calls=[]
            )
        ])

        # Configure HTTP responses
        mocks.http.add_pattern_response(
            r".*google.*|.*duckduckgo.*",
            MockHTTPResponse(
                content='{"results": [{"title": "Test Result", "url": "https://example.com"}]}'
            )
        )

        # Configure router
        mocks.router.add_route("research", MockSemanticMatch("Research", 0.92))

        return mocks

    def create_file_ops_scenario(self) -> MockSet:
        """Create mocks configured for file operations."""
        mocks = self.create_all()

        mocks.claude.add_response(
            "read|list|files|directory",
            MockLLMResponse(
                content="I'll list the files in that directory.",
                tool_calls=[
                    MockToolCall("list_files", {"path": "/tmp/mock"})
                ]
            )
        )

        mocks.router.add_route("file|read|list|directory", MockSemanticMatch("FileOperations", 0.89))

        return mocks
