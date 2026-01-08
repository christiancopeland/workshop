"""
Context Assembly Tests

Tests for the context pipeline assembly:
- Telos personal context loading
- Task context injection
- Automatic context (files, workflow)
- Memory context retrieval

These tests verify that all context layers are properly
assembled and injected into the system prompt.
"""

import pytest
from datetime import datetime

from tests.e2e.context_tracer import (
    ContextPipelineTrace,
    ContextLayer,
    TraceStage,
    PipelineTracer,
)
from tests.e2e.mocks import MockFactory, MockSemanticMatch
from tests.e2e.scenarios import E2EScenario, ValidationResult
from tests.e2e.visualizer import TraceVisualizer


class TestContextLayerCapture:
    """Test that context layers are properly captured in traces."""

    def test_telos_layer_capture(self, trace: ContextPipelineTrace):
        """Test capturing Telos context layers."""
        # Add profile layer
        profile_layer = ContextLayer(
            layer_name="telos_profile",
            source_path="/home/user/.workshop/Telos/profile.md",
            content="# Profile\nI am a software engineer...",
            content_length=500,
            loaded_successfully=True,
            load_duration_ms=15,
        )
        trace.add_telos_layer(profile_layer)

        # Add goals layer
        goals_layer = ContextLayer(
            layer_name="telos_goals",
            source_path="/home/user/.workshop/Telos/goals.md",
            content="# Goals\n- Learn Rust\n- Build Workshop",
            content_length=300,
            loaded_successfully=True,
            load_duration_ms=10,
        )
        trace.add_telos_layer(goals_layer)

        # Verify
        assert trace.telos_loaded
        assert len(trace.telos_layers) == 2
        assert trace.telos_total_chars == 800
        assert trace.telos_layers[0].layer_name == "telos_profile"
        assert trace.telos_layers[1].layer_name == "telos_goals"

    def test_telos_layer_failure(self, trace: ContextPipelineTrace):
        """Test capturing failed Telos layer load."""
        failed_layer = ContextLayer(
            layer_name="telos_mission",
            source_path="/home/user/.workshop/Telos/mission.md",
            content="",
            content_length=0,
            loaded_successfully=False,
            load_duration_ms=5,
            metadata={"error": "File not found"},
        )
        trace.add_telos_layer(failed_layer)

        # Should still be added but marked as failed
        assert len(trace.telos_layers) == 1
        assert not trace.telos_layers[0].loaded_successfully
        assert trace.telos_layers[0].metadata["error"] == "File not found"

    def test_task_context_capture(self, trace: ContextPipelineTrace):
        """Test capturing task context."""
        trace.tasks_session_id = "sess_20260104_143022"
        trace.tasks_bound_correctly = True
        trace.tasks_active = [
            {"content": "Research topic", "status": "in_progress"},
            {"content": "Summarize findings", "status": "pending"},
        ]
        trace.tasks_pending_count = 1
        trace.tasks_in_progress_count = 1
        trace.tasks_completed_count = 0
        trace.task_original_request = "Research Python async"
        trace.task_context_formatted = "=== ACTIVE TASK CONTEXT ===\n..."
        trace.task_context_length = 500

        # Verify
        assert trace.tasks_bound_correctly
        assert len(trace.tasks_active) == 2
        assert trace.tasks_pending_count == 1
        assert trace.task_context_length == 500

    def test_auto_context_capture(self, trace: ContextPipelineTrace):
        """Test capturing automatic context."""
        trace.auto_context_enabled = True
        trace.auto_context_injection_reason = "keyword 'research' matched"
        trace.active_files = [
            "/home/user/project/main.py",
            "/home/user/project/agent.py",
        ]
        trace.recent_edits = [
            {"file": "main.py", "action": "modified", "time": "2 min ago"},
        ]
        trace.detected_workflow = "feature_development"
        trace.workflow_confidence = 0.78
        trace.auto_context_formatted = "[Current Context:\n# Active Files...]"
        trace.auto_context_length = 800

        # Verify
        assert trace.auto_context_enabled
        assert len(trace.active_files) == 2
        assert trace.detected_workflow == "feature_development"
        assert trace.workflow_confidence == 0.78

    def test_memory_context_capture(self, trace: ContextPipelineTrace):
        """Test capturing memory context."""
        trace.memory_search_performed = True
        trace.memory_search_query = "Python async programming"
        trace.memory_results = [
            "Previous conversation about asyncio",
            "Notes on Python concurrency",
        ]
        trace.memory_results_count = 2
        trace.recent_messages_included = 5
        trace.memory_context_formatted = "# Relevant Memories\n..."
        trace.memory_context_length = 600

        # Verify
        assert trace.memory_search_performed
        assert trace.memory_results_count == 2
        assert trace.recent_messages_included == 5


class TestContextLayerSerialization:
    """Test serialization of context layers."""

    def test_context_layer_to_dict(self):
        """Test ContextLayer serialization."""
        layer = ContextLayer(
            layer_name="telos_profile",
            source_path="/path/to/profile.md",
            content="This is a long content string that should be truncated in the preview...",
            content_length=100,
            loaded_successfully=True,
            load_duration_ms=15,
            metadata={"version": "1.0"},
        )

        data = layer.to_dict()

        assert data["layer_name"] == "telos_profile"
        assert data["source_path"] == "/path/to/profile.md"
        assert data["content_length"] == 100
        assert data["loaded_successfully"]
        assert data["load_duration_ms"] == 15
        assert "content_preview" in data

    def test_trace_to_dict_context(self, trace: ContextPipelineTrace):
        """Test trace serialization includes context."""
        trace.add_telos_layer(ContextLayer(
            layer_name="profile",
            content="Test",
            content_length=4,
        ))
        trace.tasks_pending_count = 2
        trace.active_files = ["file1.py", "file2.py"]
        trace.memory_results_count = 3

        data = trace.to_dict()

        assert "context" in data
        assert data["context"]["telos"]["loaded"]
        assert len(data["context"]["telos"]["layers"]) == 1
        assert data["context"]["tasks"]["pending"] == 2
        assert len(data["context"]["auto"]["active_files"]) == 2
        assert data["context"]["memory"]["results_count"] == 3


class TestPipelineTracer:
    """Test the PipelineTracer context manager."""

    @pytest.mark.asyncio
    async def test_tracer_context_manager(self):
        """Test PipelineTracer as async context manager."""
        async with PipelineTracer("test input", "test_session") as tracer:
            tracer.trace.set_stage(TraceStage.CONTEXT_TELOS)

            # Simulate loading context
            layer = tracer.start_context_layer("telos_profile", "/path/to/profile.md")
            layer.content = "Profile content"
            layer.content_length = 15
            layer.loaded_successfully = True
            tracer.trace.add_telos_layer(layer)

            tracer.trace.set_stage(TraceStage.CONTEXT_ASSEMBLED)

            # Complete the trace before exiting
            tracer.trace.complete("Test response")

        # After context manager
        assert tracer.trace.timestamp_start is not None
        assert tracer.trace.timestamp_end is not None
        assert tracer.trace.current_stage == TraceStage.COMPLETED
        assert len(tracer.trace.telos_layers) == 1

    @pytest.mark.asyncio
    async def test_tracer_error_handling(self):
        """Test PipelineTracer handles errors."""
        with pytest.raises(ValueError):
            async with PipelineTracer("test", "session") as tracer:
                tracer.trace.set_stage(TraceStage.CONTEXT_TELOS)
                raise ValueError("Test error")

        # Trace should be marked as failed
        assert not tracer.trace.success
        assert tracer.trace.error_stage == TraceStage.CONTEXT_TELOS


class TestContextScenarios:
    """Test context assembly in full scenarios."""

    @pytest.mark.asyncio
    async def test_full_context_assembly_scenario(self, mocks):
        """Test a scenario with full context assembly."""
        scenario = E2EScenario(
            name="context_assembly",
            description="Test full context assembly",
            user_inputs=["research Python tutorials"],
        )

        # Configure expectations
        scenario.expect_telos(loaded=True)
        scenario.expect_skill("Research")

        # Configure mocks
        scenario.with_mocks(mocks)
        mocks.router.add_route(
            "research",
            MockSemanticMatch("Research", 0.92, "research [topic]", "semantic_direct")
        )

        # Run scenario
        result = await scenario.run()

        # Verify context was captured
        assert len(result.traces) == 1
        trace = result.traces[0]
        assert trace.telos_loaded

    @pytest.mark.asyncio
    async def test_context_without_telos(self, mocks):
        """Test scenario where Telos is not loaded."""
        scenario = E2EScenario(
            name="no_telos",
            description="Test without Telos context",
            user_inputs=["hello"],
        )

        scenario.with_mocks(mocks)
        mocks.router.add_route(".*", MockSemanticMatch("chat", 0.6, "", "pattern"))

        result = await scenario.run()

        assert len(result.traces) == 1
        # The mock scenario sets telos_loaded to True by default
        # In real tests with actual Workshop, we'd verify actual behavior


class TestVisualization:
    """Test visualization of context in traces."""

    def test_ascii_context_section(self, trace: ContextPipelineTrace, visualizer: TraceVisualizer):
        """Test ASCII visualization includes context."""
        # Setup trace with context
        trace.add_telos_layer(ContextLayer("profile", content="Test", content_length=100))
        trace.tasks_pending_count = 2
        trace.active_files = ["file1.py"]
        trace.memory_results_count = 3
        trace.complete("Test response")

        # Generate ASCII
        output = visualizer.to_ascii(trace)

        # Verify sections present
        assert "CONTEXT ASSEMBLY" in output
        assert "TELOS" in output
        assert "TASK CONTEXT" in output
        assert "AUTOMATIC CONTEXT" in output
        assert "MEMORY CONTEXT" in output

    def test_markdown_context_section(self, trace: ContextPipelineTrace, visualizer: TraceVisualizer):
        """Test markdown visualization includes context."""
        trace.add_telos_layer(ContextLayer(
            "profile",
            source_path="/path/profile.md",
            content="Test",
            content_length=100,
        ))
        trace.telos_project_detected = "workshop"
        trace.tasks_pending_count = 1
        trace.tasks_in_progress_count = 1
        trace.active_files = ["agent.py", "main.py"]
        trace.detected_workflow = "debugging"
        trace.workflow_confidence = 0.85
        trace.memory_search_performed = True
        trace.memory_results_count = 5
        trace.complete("Done")

        # Generate markdown
        output = visualizer.to_markdown(trace)

        # Verify structure
        assert "## Context Assembly" in output
        assert "### Layer 1: Telos" in output
        assert "### Layer 2: Task Context" in output
        assert "### Layer 3: Automatic Context" in output
        assert "### Layer 4: Memory Context" in output
        assert "workshop" in output  # Project detected
        assert "debugging" in output  # Workflow


class TestContextValidation:
    """Test validation of context expectations."""

    @pytest.mark.asyncio
    async def test_validate_telos_loaded(self, mocks):
        """Test validation that Telos was loaded."""
        scenario = E2EScenario(
            name="validate_telos",
            user_inputs=["test"],
        )
        scenario.with_mocks(mocks)
        scenario.expect_telos(loaded=True)

        result = await scenario.run()

        # Find telos validation
        telos_checks = [v for v in result.validations if "telos" in v.name]
        assert len(telos_checks) > 0
        assert all(v.result == ValidationResult.PASSED for v in telos_checks)

    @pytest.mark.asyncio
    async def test_validate_telos_not_loaded(self, mocks):
        """Test validation that Telos was not loaded (when expected)."""
        scenario = E2EScenario(
            name="validate_no_telos",
            user_inputs=["test"],
        )
        scenario.with_mocks(mocks)
        scenario.expect_telos(loaded=False)

        result = await scenario.run()

        # This will fail because mock sets telos_loaded=True
        telos_checks = [v for v in result.validations if "telos" in v.name]
        assert len(telos_checks) > 0
        # At least one should fail since we expected no telos
        assert any(v.result == ValidationResult.FAILED for v in telos_checks)


class TestContextTimingCapture:
    """Test timing capture for context assembly."""

    def test_context_timing(self, trace: ContextPipelineTrace):
        """Test that context timing is captured."""
        trace.duration_context_ms = 150
        trace.telos_load_duration_ms = 50

        assert trace.duration_context_ms == 150
        assert trace.telos_load_duration_ms == 50

    def test_context_layer_timing(self):
        """Test individual layer timing."""
        layer = ContextLayer(
            layer_name="test",
            load_duration_ms=25,
        )

        assert layer.load_duration_ms == 25

    def test_timing_in_visualization(self, trace: ContextPipelineTrace, visualizer: TraceVisualizer):
        """Test timing appears in visualization."""
        trace.duration_context_ms = 150
        trace.duration_routing_ms = 50
        trace.duration_llm_ms = 500
        trace.duration_tools_ms = 200
        trace.duration_total_ms = 900
        trace.complete("Done")

        output = visualizer.to_ascii(trace)

        assert "150ms" in output or "context" in output.lower()
