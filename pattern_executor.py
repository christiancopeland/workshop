"""
Pattern Executor - Execute Fabric-style patterns on input.

Patterns are stateless prompt templates that transform input to output.
They differ from Skills (which have tools) and Agents (which have memory).

This is inspired by Daniel Miessler's Fabric project, where patterns are
composable, single-purpose prompts that can be chained like Unix pipes:

    content | extract_wisdom | create_summary

Key concepts:
- Patterns are pure text transformations (no tools, no state)
- Each pattern has a single purpose (extract, analyze, create, improve, transform)
- Patterns can be chained via PatternPipeline
- Patterns use the same LLM as skill execution (phi4:14b by default)

Directory structure:
    ~/.workshop/patterns/
    ├── _registry.json           # Pattern metadata
    ├── analyze/
    │   └── analyze_claims/
    │       └── system.md
    ├── extract/
    │   └── extract_wisdom/
    │       └── system.md
    └── ...

Usage:
    executor = PatternExecutor()
    await executor.initialize()

    result = await executor.execute("extract_wisdom", article_text)
    print(result.output)

    # Or chain patterns:
    pipeline = PatternPipeline(executor)
    result = await pipeline.execute(
        ["extract_wisdom", "create_summary"],
        article_text
    )
"""

import asyncio
import aiohttp
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from logger import get_logger
from config import Config

# Dashboard integration
try:
    import dashboard_integration as dash
except ImportError:
    dash = None

log = get_logger("pattern_executor")


@dataclass
class PatternInfo:
    """Metadata about a pattern from the registry."""
    name: str
    category: str
    purpose: str
    input_type: str = "text"
    output_type: str = "markdown"
    triggers: List[str] = field(default_factory=list)
    model_preference: str = "phi4:14b"
    token_estimate: int = 1500


@dataclass
class PatternResult:
    """Result of pattern execution."""
    pattern_name: str
    output: str
    model_used: str
    tokens_used: int
    duration_ms: int
    success: bool = True
    error: Optional[str] = None


class PatternRegistry:
    """
    Registry for discovering and loading patterns.

    Patterns are stored in ~/.workshop/patterns/ with metadata in _registry.json.
    """

    def __init__(self, patterns_dir: Path = None):
        self.patterns_dir = patterns_dir or Path.home() / ".workshop" / "patterns"
        self._registry: Dict[str, PatternInfo] = {}
        self._initialized = False

    async def initialize(self):
        """Load the pattern registry."""
        if self._initialized:
            return

        await self._load_registry()
        self._initialized = True
        log.info(f"PatternRegistry initialized with {len(self._registry)} patterns")

    async def _load_registry(self):
        """Load patterns from _registry.json."""
        registry_path = self.patterns_dir / "_registry.json"

        if not registry_path.exists():
            log.warning(f"Pattern registry not found: {registry_path}")
            return

        try:
            data = json.loads(registry_path.read_text())
            patterns = data.get("patterns", {})

            for name, info in patterns.items():
                self._registry[name] = PatternInfo(
                    name=name,
                    category=info.get("category", "other"),
                    purpose=info.get("purpose", ""),
                    input_type=info.get("input_type", "text"),
                    output_type=info.get("output_type", "markdown"),
                    triggers=info.get("triggers", []),
                    model_preference=info.get("model_preference", "phi4:14b"),
                    token_estimate=info.get("token_estimate", 1500),
                )

        except Exception as e:
            log.error(f"Failed to load pattern registry: {e}")

    def list_patterns(self, category: str = None) -> List[str]:
        """List available patterns, optionally filtered by category."""
        if category:
            return [name for name, info in self._registry.items()
                    if info.category == category]
        return list(self._registry.keys())

    def list_categories(self) -> List[str]:
        """List available pattern categories."""
        return list(set(info.category for info in self._registry.values()))

    def get_pattern_info(self, pattern_name: str) -> Optional[PatternInfo]:
        """Get metadata for a pattern."""
        return self._registry.get(pattern_name)

    def get_pattern_prompt(self, pattern_name: str) -> Optional[str]:
        """Load the system.md content for a pattern."""
        info = self._registry.get(pattern_name)
        if not info:
            return None

        # Check category subdirectory first
        category_path = self.patterns_dir / info.category / pattern_name / "system.md"
        if category_path.exists():
            return category_path.read_text()

        # Check flat structure
        flat_path = self.patterns_dir / pattern_name / "system.md"
        if flat_path.exists():
            return flat_path.read_text()

        log.warning(f"Pattern prompt not found: {pattern_name}")
        return None

    def match_pattern(self, user_input: str) -> Optional[str]:
        """
        Try to match user input to a pattern based on triggers.

        Returns pattern name if a trigger matches, None otherwise.
        """
        user_lower = user_input.lower().strip()

        for name, info in self._registry.items():
            for trigger in info.triggers:
                if trigger.lower() in user_lower:
                    return name

        return None


class PatternExecutor:
    """
    Execute Fabric-style patterns on input text.

    Patterns are stateless text transformations - they take input and produce
    output without any tool calls or state management.
    """

    def __init__(
        self,
        registry: PatternRegistry = None,
        ollama_url: str = None,
        default_model: str = None,
    ):
        self.registry = registry or PatternRegistry()
        self.ollama_url = ollama_url or Config.OLLAMA_URL
        self.default_model = default_model or "phi4:14b"
        self._initialized = False

    async def initialize(self):
        """Initialize the executor and registry."""
        if self._initialized:
            return

        await self.registry.initialize()
        self._initialized = True
        log.info(f"PatternExecutor initialized with model: {self.default_model}")

    def list_patterns(self, category: str = None) -> List[str]:
        """List available patterns."""
        return self.registry.list_patterns(category)

    def list_categories(self) -> List[str]:
        """List pattern categories."""
        return self.registry.list_categories()

    def get_pattern_info(self, pattern_name: str) -> Optional[PatternInfo]:
        """Get pattern metadata."""
        return self.registry.get_pattern_info(pattern_name)

    async def execute(
        self,
        pattern_name: str,
        input_text: str,
        model: str = None,
        trace_id: str = None,
        stage: int = 1,
        total_stages: int = 1,
    ) -> PatternResult:
        """
        Execute a pattern on input text.

        Args:
            pattern_name: Name of the pattern to execute
            input_text: The text to transform
            model: Optional model override
            trace_id: Optional trace ID for dashboard events
            stage: Current stage in pipeline (for dashboard)
            total_stages: Total stages in pipeline (for dashboard)

        Returns:
            PatternResult with the transformed output
        """
        if not self._initialized:
            await self.initialize()

        # Get pattern info and prompt
        info = self.registry.get_pattern_info(pattern_name)
        if not info:
            return PatternResult(
                pattern_name=pattern_name,
                output="",
                model_used="",
                tokens_used=0,
                duration_ms=0,
                success=False,
                error=f"Unknown pattern: {pattern_name}",
            )

        prompt = self.registry.get_pattern_prompt(pattern_name)
        if not prompt:
            return PatternResult(
                pattern_name=pattern_name,
                output="",
                model_used="",
                tokens_used=0,
                duration_ms=0,
                success=False,
                error=f"Pattern prompt not found: {pattern_name}",
            )

        # Use pattern's preferred model or default
        use_model = model or info.model_preference or self.default_model

        # Build the full prompt with input
        full_prompt = self._inject_input(prompt, input_text)

        log.info(f"Executing pattern '{pattern_name}' with {use_model} (stage {stage}/{total_stages})")

        # Emit dashboard events - pattern execution starting
        if dash:
            await dash.pattern_executing(pattern_name, stage, total_stages, trace_id)
            # Create messages list for dashboard display
            pattern_messages = [{"role": "user", "content": full_prompt}]
            await dash.llm_calling(use_model, 1, messages=pattern_messages, system_prompt=None, trace_id=trace_id)

        start_time = time.time()

        try:
            output = await self._call_llm(full_prompt, use_model)
            duration_ms = int((time.time() - start_time) * 1000)

            # Emit completion events
            if dash:
                await dash.llm_complete(len(output), duration_ms, 0, trace_id)
                await dash.pattern_complete(pattern_name, len(output), duration_ms, stage, trace_id)

            return PatternResult(
                pattern_name=pattern_name,
                output=output,
                model_used=use_model,
                tokens_used=len(output.split()),  # Rough estimate
                duration_ms=duration_ms,
                success=True,
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            log.error(f"Pattern execution failed: {e}")

            if dash:
                await dash.error(str(e), "pattern_execution", trace_id)

            return PatternResult(
                pattern_name=pattern_name,
                output="",
                model_used=use_model,
                tokens_used=0,
                duration_ms=duration_ms,
                success=False,
                error=str(e),
            )

    def _inject_input(self, prompt: str, input_text: str) -> str:
        """Inject input text into the pattern prompt."""
        # Replace INPUT: placeholder
        if "INPUT:" in prompt:
            # Find the INPUT: section and append the text after it
            parts = prompt.split("INPUT:")
            if len(parts) == 2:
                return parts[0] + "INPUT:\n\n" + input_text

        # If no INPUT: marker, append at the end
        return prompt + "\n\n# INPUT\n\n" + input_text

    async def _call_llm(self, prompt: str, model: str) -> str:
        """Call the LLM with the pattern prompt."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 4096,  # Allow long outputs for extraction patterns
                }
            }

            async with session.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),  # 5 min for long content
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"LLM returned {response.status}: {error_text}")

                result = await response.json()
                return result.get("message", {}).get("content", "")


class PatternPipeline:
    """
    Execute a chain of patterns, piping output to input.

    This enables Unix-style composition:
        content | extract_wisdom | create_summary

    Usage:
        pipeline = PatternPipeline(executor)
        result = await pipeline.execute(
            ["extract_wisdom", "create_summary"],
            article_text
        )
    """

    def __init__(self, executor: PatternExecutor):
        self.executor = executor

    async def execute(
        self,
        patterns: List[str],
        initial_input: str,
        model: str = None,
        trace_id: str = None,
    ) -> PatternResult:
        """
        Execute patterns in sequence, piping output to input.

        Args:
            patterns: List of pattern names to execute in order
            initial_input: The starting text
            model: Optional model override (applies to all patterns)
            trace_id: Optional trace ID for dashboard events

        Returns:
            PatternResult with the final output and aggregate stats
        """
        if not patterns:
            return PatternResult(
                pattern_name="empty_pipeline",
                output=initial_input,
                model_used="",
                tokens_used=0,
                duration_ms=0,
                success=True,
            )

        # Emit pipeline started event
        if dash:
            await dash.pipeline_detected(patterns, trace_id)

        current_input = initial_input
        total_tokens = 0
        total_duration = 0
        models_used = []
        total_stages = len(patterns)

        for i, pattern_name in enumerate(patterns):
            stage = i + 1
            log.info(f"Pipeline stage {stage}/{total_stages}: {pattern_name}")

            result = await self.executor.execute(
                pattern_name,
                current_input,
                model=model,
                trace_id=trace_id,
                stage=stage,
                total_stages=total_stages,
            )

            if not result.success:
                if dash:
                    await dash.error(f"Pipeline failed at stage {stage}: {result.error}", "pipeline", trace_id)

                return PatternResult(
                    pattern_name=" | ".join(patterns),
                    output="",
                    model_used=", ".join(models_used) if models_used else "",
                    tokens_used=total_tokens,
                    duration_ms=total_duration,
                    success=False,
                    error=f"Pipeline failed at '{pattern_name}': {result.error}",
                )

            current_input = result.output
            total_tokens += result.tokens_used
            total_duration += result.duration_ms
            models_used.append(result.model_used)

        # Emit pipeline complete event
        if dash:
            await dash.pipeline_complete(patterns, total_duration, trace_id)

        return PatternResult(
            pattern_name=" | ".join(patterns),
            output=current_input,
            model_used=", ".join(set(models_used)),
            tokens_used=total_tokens,
            duration_ms=total_duration,
            success=True,
        )

    @staticmethod
    def parse_pipeline_syntax(text: str) -> Optional[List[str]]:
        """
        Parse pipeline syntax from user input.

        Recognizes:
        - "extract_wisdom | create_summary" (pipe syntax)
        - "extract wisdom, then summarize" (natural language)
        - "first extract ideas, then create summary" (natural language)

        Returns list of pattern names or None if not a pipeline.
        """
        text = text.lower().strip()

        # Pipe syntax: "pattern1 | pattern2"
        if " | " in text:
            parts = [p.strip().replace(" ", "_") for p in text.split("|")]
            if len(parts) >= 2:
                return parts

        # Natural language: "X, then Y"
        import re
        then_match = re.match(r"(.+?),?\s+then\s+(.+)", text)
        if then_match:
            patterns = []
            for part in [then_match.group(1), then_match.group(2)]:
                # Try to extract pattern name
                pattern = part.strip().replace(" ", "_")
                patterns.append(pattern)
            return patterns if len(patterns) >= 2 else None

        # Natural language: "first X, then Y"
        first_match = re.match(r"first\s+(.+?),?\s+then\s+(.+)", text)
        if first_match:
            patterns = []
            for part in [first_match.group(1), first_match.group(2)]:
                pattern = part.strip().replace(" ", "_")
                patterns.append(pattern)
            return patterns if len(patterns) >= 2 else None

        return None


# Convenience functions for common use cases

async def apply_pattern(
    pattern_name: str,
    input_text: str,
    model: str = None,
) -> str:
    """
    Quick helper to apply a single pattern.

    Usage:
        summary = await apply_pattern("create_summary", article_text)
    """
    executor = PatternExecutor()
    await executor.initialize()
    result = await executor.execute(pattern_name, input_text, model=model)

    if result.success:
        return result.output
    else:
        raise RuntimeError(f"Pattern failed: {result.error}")


async def apply_pipeline(
    patterns: List[str],
    input_text: str,
    model: str = None,
) -> str:
    """
    Quick helper to apply a pattern pipeline.

    Usage:
        result = await apply_pipeline(
            ["extract_wisdom", "create_summary"],
            article_text
        )
    """
    executor = PatternExecutor()
    await executor.initialize()
    pipeline = PatternPipeline(executor)
    result = await pipeline.execute(patterns, input_text, model=model)

    if result.success:
        return result.output
    else:
        raise RuntimeError(f"Pipeline failed: {result.error}")
