"""
Pattern Loader - PAI-style prompt template management
Implements standardized prompt templates with variable substitution
"""

import re
from pathlib import Path
from typing import Dict, Any, Optional, List


class PatternLoader:
    """
    Load and process PAI-style prompt templates with variable substitution.

    Patterns follow the structure:
    ~/.workshop/patterns/
    ├── base/
    │   ├── system.md       # Core system prompt
    │   └── voice_system.md # Voice-optimized variant
    ├── tools/
    │   └── [tool_name]/
    │       ├── system.md   # Tool-specific instructions
    │       └── voice.md    # Voice variant
    └── workflows/
        └── [workflow_name].md
    """

    # Sentinel token for secure input substitution
    INPUT_SENTINEL = "__WORKSHOP_INPUT_SENTINEL__"

    def __init__(self, patterns_dir: Optional[Path] = None):
        """Initialize pattern loader with patterns directory"""
        if patterns_dir is None:
            # Default to .workshop/patterns in the project root
            patterns_dir = Path(__file__).parent / ".workshop" / "patterns"

        self.patterns_dir = Path(patterns_dir)
        self.patterns_dir.mkdir(parents=True, exist_ok=True)

        # Ensure base directories exist
        (self.patterns_dir / "base").mkdir(exist_ok=True)
        (self.patterns_dir / "tools").mkdir(exist_ok=True)
        (self.patterns_dir / "workflows").mkdir(exist_ok=True)

    def load_pattern(
        self,
        pattern_name: str,
        mode: str = "text",
        **variables
    ) -> str:
        """
        Load a pattern template and substitute variables.

        Args:
            pattern_name: Pattern path like "base/system" or "tools/read_file"
            mode: "text" or "voice" - determines which template file to load
            **variables: Variables to substitute in template

        Returns:
            Processed template string with variables substituted

        Examples:
            >>> loader.load_pattern("base/system", mode="voice", user_name="Alice")
            >>> loader.load_pattern("tools/read_file", tool_roster="...")
        """
        # Determine which template file to load based on mode
        # Voice mode tries voice-specific patterns first, then falls back to standard
        if mode == "voice":
            template_filenames = ["voice_system.md", "voice.md", "system.md"]
        else:
            template_filenames = ["system.md"]

        # Parse pattern path - handles "base/system" format
        # Split into directory and base name parts
        if "/" in pattern_name:
            parts = pattern_name.rsplit("/", 1)
            pattern_dir = self.patterns_dir / parts[0]
            pattern_base = parts[1]
        else:
            pattern_dir = self.patterns_dir
            pattern_base = pattern_name

        # Try each template filename in order
        template_content = None
        tried_paths = []
        for filename in template_filenames:
            # For voice mode, replace "system.md" part with voice variant
            # e.g., "system" base + "voice_system.md" filename -> "voice_system.md"
            if filename.startswith("voice_") and pattern_base == "system":
                test_path = pattern_dir / filename
            elif filename == "system.md":
                test_path = pattern_dir / f"{pattern_base}.md"
            else:
                test_path = pattern_dir / filename

            tried_paths.append(str(test_path))

            if test_path.exists() and test_path.is_file():
                template_content = test_path.read_text()
                break

        if template_content is None:
            raise FileNotFoundError(
                f"Pattern '{pattern_name}' not found. Tried: {tried_paths}"
            )

        # Substitute variables
        return self.substitute_variables(template_content, **variables)

    def substitute_variables(self, template: str, **variables) -> str:
        """
        Substitute {{variable}} placeholders with security.

        Uses sentinel token approach to prevent prompt injection:
        1. Replace {{input}} with sentinel token
        2. Process all other variables
        3. Replace sentinel with escaped user input

        Args:
            template: Template string with {{variable}} placeholders
            **variables: Variable name -> value mappings

        Returns:
            Template with variables substituted
        """
        # Step 1: Replace {{input}} with sentinel (if present)
        result = template.replace("{{input}}", self.INPUT_SENTINEL)

        # Step 2: Process all other variables
        for key, value in variables.items():
            if key == "input":
                continue  # Handle input separately for security

            placeholder = f"{{{{{key}}}}}"

            # Convert value to string if needed
            if value is None:
                value_str = ""
            elif isinstance(value, (list, dict)):
                # For complex types, use simple string representation
                value_str = str(value)
            else:
                value_str = str(value)

            result = result.replace(placeholder, value_str)

        # Step 3: Replace sentinel with escaped user input
        if "input" in variables:
            escaped_input = self._escape_for_llm(variables["input"])
            result = result.replace(self.INPUT_SENTINEL, escaped_input)
        else:
            # Remove sentinel if no input was provided
            result = result.replace(self.INPUT_SENTINEL, "")

        return result

    def _escape_for_llm(self, text: str) -> str:
        """
        Escape user input to prevent prompt injection.

        This is a basic implementation - could be enhanced with more
        sophisticated prompt injection detection.
        """
        if not isinstance(text, str):
            text = str(text)

        # For now, we treat input as literal text
        # In the future, could add detection for:
        # - Prompt breaking sequences
        # - System message injection attempts
        # - Tool call injection attempts

        return text

    def load_tool_pattern(
        self,
        tool_name: str,
        mode: str = "text"
    ) -> str:
        """
        Load tool-specific prompt template.

        Args:
            tool_name: Name of the tool (e.g., "read_file")
            mode: "text" or "voice"

        Returns:
            Tool prompt template
        """
        return self.load_pattern(f"tools/{tool_name}", mode=mode)

    def get_available_patterns(self) -> Dict[str, List[str]]:
        """
        List all available patterns.

        Returns:
            Dictionary with pattern categories and their available patterns
        """
        patterns = {
            "base": [],
            "tools": [],
            "workflows": []
        }

        # Scan base patterns
        base_dir = self.patterns_dir / "base"
        if base_dir.exists():
            patterns["base"] = [
                f.stem for f in base_dir.glob("*.md")
            ]

        # Scan tool patterns (directories)
        tools_dir = self.patterns_dir / "tools"
        if tools_dir.exists():
            patterns["tools"] = [
                d.name for d in tools_dir.iterdir() if d.is_dir()
            ]

        # Scan workflow patterns
        workflows_dir = self.patterns_dir / "workflows"
        if workflows_dir.exists():
            patterns["workflows"] = [
                f.stem for f in workflows_dir.glob("*.md")
            ]

        return patterns

    def validate_pattern(self, pattern_content: str) -> Dict[str, Any]:
        """
        Validate that a pattern follows PAI format standards.

        Checks for required sections:
        - IDENTITY and PURPOSE
        - Other standard sections

        Returns:
            Validation results with any warnings or errors
        """
        sections = {
            "has_identity": bool(re.search(r"#\s*IDENTITY", pattern_content, re.IGNORECASE)),
            "has_purpose": bool(re.search(r"PURPOSE", pattern_content, re.IGNORECASE)),
            "has_steps": bool(re.search(r"#\s*STEPS", pattern_content, re.IGNORECASE)),
            "has_output": bool(re.search(r"#\s*OUTPUT", pattern_content, re.IGNORECASE)),
        }

        warnings = []
        if not sections["has_identity"]:
            warnings.append("Missing 'IDENTITY' section")
        if not sections["has_purpose"]:
            warnings.append("Missing 'PURPOSE' section")

        return {
            "valid": sections["has_identity"] and sections["has_purpose"],
            "sections": sections,
            "warnings": warnings
        }


class PromptPrimitives:
    """
    Composable prompt building blocks (PAI pattern).

    These primitives enable modular prompt construction and achieved
    65% token reduction in Miessler's system.
    """

    @staticmethod
    def roster(items: List[str], title: str = "Available", format: str = "markdown") -> str:
        """
        Generate formatted list of items (tools, skills, etc.).

        Args:
            items: List of items to display
            title: Section title
            format: "markdown", "numbered", or "plain"

        Returns:
            Formatted roster string
        """
        if not items:
            return f"## {title}\nNone available"

        if format == "numbered":
            lines = [f"{i+1}. {item}" for i, item in enumerate(items)]
        elif format == "plain":
            lines = [item for item in items]
        else:  # markdown (default)
            lines = [f"- {item}" for item in items]

        return f"## {title}\n" + "\n".join(lines)

    @staticmethod
    def voice(mode: str = "concise", context: Optional[str] = None) -> str:
        """
        Communication style specification for different voice modes.

        Args:
            mode: "concise", "detailed", or "debug"
            context: Optional context about why this mode is selected

        Returns:
            Voice style instructions
        """
        voices = {
            "concise": """Be brief and actionable. User is hands-free.
- Responses under 3 sentences when possible
- Confirm actions before executing
- Summarize long outputs, don't recite everything""",

            "detailed": """Provide thorough explanations.
- Include examples and reasoning
- Explain what you're doing and why
- Offer follow-up options""",

            "debug": """Include technical details.
- Show file paths and line numbers
- Mention what tools you're using
- Explain errors with full context"""
        }

        style = voices.get(mode, voices["concise"])

        if context:
            return f"## Communication Style\n{style}\n\nContext: {context}"

        return f"## Communication Style\n{style}"

    @staticmethod
    def structure(format_type: str, fields: Optional[List[str]] = None) -> str:
        """
        Response format specification.

        Args:
            format_type: "json", "markdown", "yaml", or "plain"
            fields: Optional list of expected fields

        Returns:
            Format instructions
        """
        formats = {
            "json": "Return response as valid JSON object",
            "markdown": "Use markdown formatting with headers and lists",
            "yaml": "Return response as valid YAML",
            "plain": "Return plain text response, no formatting"
        }

        instruction = formats.get(format_type, formats["plain"])

        if fields:
            fields_str = ", ".join(fields)
            return f"## Output Format\n{instruction}\n\nExpected fields: {fields_str}"

        return f"## Output Format\n{instruction}"

    @staticmethod
    def briefing(context: Dict[str, Any], include_keys: Optional[List[str]] = None) -> str:
        """
        Context and background information.

        Args:
            context: Dictionary of context information
            include_keys: Optional list of keys to include (default: all)

        Returns:
            Formatted briefing section
        """
        if include_keys:
            context = {k: v for k, v in context.items() if k in include_keys}

        lines = ["## Context"]
        for key, value in context.items():
            # Convert key to title case
            display_key = key.replace("_", " ").title()
            lines.append(f"**{display_key}**: {value}")

        return "\n".join(lines)

    @staticmethod
    def gate(condition: str, then_action: str, else_action: Optional[str] = None) -> str:
        """
        Conditional logic for prompts.

        Args:
            condition: The condition to check
            then_action: Action if condition is true
            else_action: Optional action if condition is false

        Returns:
            Formatted conditional instruction
        """
        if else_action:
            return f"IF {condition} THEN {then_action}\nELSE {else_action}"

        return f"IF {condition} THEN {then_action}"
