"""
Workshop Skill Registry
PAI-inspired Skills architecture for organizing and routing tools
"""

import re
import asyncio
import inspect
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass

from logger import get_logger

# Import pattern tools for integration
try:
    from pattern_tools import PATTERN_TOOLS, get_pattern_tool
    PATTERNS_AVAILABLE = True
except ImportError:
    PATTERNS_AVAILABLE = False
    PATTERN_TOOLS = {}
    get_pattern_tool = None

log = get_logger("skill_registry")


@dataclass
class ToolInfo:
    """Information about a tool"""
    name: str
    func: Callable  # Always async
    description: str
    signature: str
    examples: List[str]
    skill_name: str


@dataclass
class IntentMatch:
    """
    Result of intent routing - contains everything needed to execute a request.

    This dataclass captures the full routing decision including:
    - Which skill matched
    - Which specific tool or workflow to use
    - Extracted context from the query (file paths, symbols, etc.)
    - The pattern that matched (for debugging)
    - Confidence level of the match
    """
    skill: Optional['Skill']
    skill_name: str
    tool_name: Optional[str] = None
    workflow_name: Optional[str] = None
    matched_pattern: Optional[str] = None
    extracted_args: Optional[Dict[str, Any]] = None  # Context extracted from query
    confidence: float = 0.0  # 0.0 to 1.0
    match_type: str = "none"  # "pattern", "keyword", "workflow", "none"

    @property
    def matched(self) -> bool:
        """Whether any skill/tool/workflow matched"""
        return self.skill is not None or self.workflow_name is not None

    def __repr__(self):
        if self.workflow_name:
            return f"IntentMatch(workflow={self.workflow_name}, skill={self.skill_name})"
        elif self.tool_name:
            return f"IntentMatch(tool={self.tool_name}, skill={self.skill_name})"
        elif self.skill:
            return f"IntentMatch(skill={self.skill_name})"
        return "IntentMatch(no match)"


@dataclass
class WorkflowInfo:
    """
    Information about a multi-step workflow.

    Workflows are step-by-step procedures that orchestrate multiple tools
    to accomplish complex tasks. They provide procedural guidance so the
    model knows exactly how to handle multi-step requests.
    """
    name: str
    purpose: str
    triggers: List[str]  # Phrases that should trigger this workflow
    steps: str  # Full markdown content with step-by-step instructions
    skill_name: str


class Skill:
    """
    Represents a self-contained skill package.

    A Skill is a collection of related tools with routing logic,
    documentation, workflows for multi-step procedures, and optional
    prompt templates.

    Directory structure:
        Skills/[SkillName]/
        ├── SKILL.md          # Routing info and documentation
        ├── __init__.py       # Optional skill initialization
        ├── tools/            # Tool implementations
        │   ├── __init__.py
        │   ├── tool1.py
        │   └── tool2.py
        ├── Workflows/        # Multi-step procedures (NEW)
        │   ├── ExploreProject.md
        │   └── InvestigateChanges.md
        └── prompts/          # Optional prompt templates
            ├── tool1.md
            └── tool2.md
    """

    def __init__(
        self,
        skill_dir: Path,
        dependencies: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize skill from directory.

        Args:
            skill_dir: Path to skill directory
            dependencies: Dict of injected dependencies (memory, config, etc.)
        """
        self.name = skill_dir.name
        self.skill_dir = skill_dir
        self.dependencies = dependencies or {}

        log.info(f"Loading skill: {self.name}")

        # Load SKILL.md routing info
        self.routing_info = self._load_routing_info()

        # Load tools from tools/ subdirectory
        self.tools: Dict[str, ToolInfo] = self._load_tools()

        # Load pattern tools declared in SKILL.md (Fabric-style patterns as tools)
        self._load_pattern_tools()

        # Load workflows from Workflows/ subdirectory (NEW)
        self.workflows: Dict[str, WorkflowInfo] = self._load_workflows()

        workflow_count = len(self.workflows)
        pattern_count = sum(1 for t in self.tools.values() if getattr(t, 'is_pattern', False))
        if workflow_count > 0 or pattern_count > 0:
            log.info(f"Skill {self.name} loaded with {len(self.tools)} tools ({pattern_count} patterns), {workflow_count} workflows")
        else:
            log.info(f"Skill {self.name} loaded with {len(self.tools)} tools")

    def _load_routing_info(self) -> Dict[str, Any]:
        """
        Parse SKILL.md for routing logic.

        Expected sections:
        - ## Purpose
        - ## User Intent Patterns (regex patterns, one per line starting with "- ")
        - ## Keywords
        - ## Priority (HIGH/MEDIUM/LOW)
        """
        skill_md = self.skill_dir / "SKILL.md"

        if not skill_md.exists():
            log.warning(f"No SKILL.md found for {self.name}")
            return {
                "purpose": "",
                "intent_patterns": [],
                "keywords": [],
                "priority": "MEDIUM"
            }

        try:
            content = skill_md.read_text()

            return {
                "purpose": self._extract_section(content, "Purpose"),
                "intent_patterns": self._extract_intent_patterns(content),
                "keywords": self._extract_keywords(content),
                "priority": self._extract_priority(content),
            }
        except Exception as e:
            log.error(f"Error parsing SKILL.md for {self.name}: {e}")
            return {
                "purpose": "",
                "intent_patterns": [],
                "keywords": [],
                "priority": "MEDIUM"
            }

    def _extract_section(self, content: str, section_name: str) -> str:
        """Extract content of a markdown section"""
        # Match ## Section Name followed by content until next ## or end
        pattern = rf"##\s+{section_name}\s*\n(.*?)(?=\n##|\Z)"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _extract_intent_patterns(self, content: str) -> List[str]:
        """
        Extract user intent patterns from SKILL.md.

        Patterns are under "## User Intent Patterns" section,
        formatted as markdown list items with quoted strings:
        - "pattern1"
        - "pattern2"
        """
        section = self._extract_section(content, "User Intent Patterns")
        if not section:
            return []

        # Match lines starting with - " and extract the quoted pattern
        pattern = r'^[\-\*]\s+"(.+?)"'
        patterns = re.findall(pattern, section, re.MULTILINE)

        log.debug(f"Extracted {len(patterns)} intent patterns for {self.name}")
        return patterns

    def _extract_keywords(self, content: str) -> List[str]:
        """
        Extract keywords from SKILL.md.

        Keywords are under "## Keywords" section as comma-separated list.
        """
        section = self._extract_section(content, "Keywords")
        if not section:
            return []

        # Split by comma and clean up
        keywords = [k.strip() for k in section.split(',')]
        keywords = [k for k in keywords if k and not k.startswith('#')]

        return keywords

    def _extract_priority(self, content: str) -> str:
        """
        Extract priority from SKILL.md.

        Priority is under "## Priority" section: HIGH, MEDIUM, or LOW
        """
        section = self._extract_section(content, "Priority")
        if not section:
            return "MEDIUM"

        priority = section.strip().upper()
        if priority in ["HIGH", "MEDIUM", "LOW"]:
            return priority

        return "MEDIUM"

    def _load_tools(self) -> Dict[str, ToolInfo]:
        """
        Load tool functions from tools/ subdirectory.

        Each .py file should contain:
        1. An async function matching the filename (e.g., read_file.py -> async def read_file)
        2. Optionally: TOOL_DESCRIPTION, TOOL_SIGNATURE, TOOL_EXAMPLES module constants

        Sync functions are auto-wrapped to async.
        """
        tools = {}
        tools_dir = self.skill_dir / "tools"

        if not tools_dir.exists():
            log.warning(f"No tools/ directory found for {self.name}")
            return tools

        # Ensure tools/ is a package
        init_file = tools_dir / "__init__.py"
        if not init_file.exists():
            init_file.touch()

        for tool_file in tools_dir.glob("*.py"):
            if tool_file.name.startswith("_"):
                continue  # Skip __init__.py and private files

            try:
                # Import tool module
                module = self._import_tool_module(tool_file)

                # Find the main tool function (matches filename)
                tool_name = tool_file.stem
                if not hasattr(module, tool_name):
                    log.warning(
                        f"Tool file {tool_file} doesn't have function {tool_name}, skipping"
                    )
                    continue

                func = getattr(module, tool_name)

                # Auto-wrap sync functions to async
                if not inspect.iscoroutinefunction(func):
                    log.debug(f"Wrapping sync function {tool_name} to async")
                    func = self._make_async(func)

                # Get metadata from module
                description = getattr(
                    module,
                    "TOOL_DESCRIPTION",
                    func.__doc__ or f"{tool_name} tool"
                )
                signature = getattr(
                    module,
                    "TOOL_SIGNATURE",
                    self._generate_signature(func)
                )
                examples = getattr(module, "TOOL_EXAMPLES", [])

                tools[tool_name] = ToolInfo(
                    name=tool_name,
                    func=func,
                    description=description,
                    signature=signature,
                    examples=examples,
                    skill_name=self.name
                )

                log.debug(f"Loaded tool: {tool_name}")

            except Exception as e:
                log.error(f"Error loading tool from {tool_file}: {e}")
                continue

        return tools

    def _load_pattern_tools(self):
        """
        Load pattern tools declared in SKILL.md's Available Tools section.

        Pattern tools are Fabric-style text transformations that are defined
        in ~/.workshop/patterns/ but can be used as tools within skills.

        This allows the LLM to naturally use patterns like extract_wisdom()
        just like any other tool, without requiring special phrases.

        Pattern tools are identified by checking if their name exists in
        the global PATTERN_TOOLS registry.
        """
        if not PATTERNS_AVAILABLE:
            return

        # Get the Available Tools section from SKILL.md
        skill_md = self.skill_dir / "SKILL.md"
        if not skill_md.exists():
            return

        content = skill_md.read_text()
        tools_section = self._extract_section(content, "Available Tools")

        if not tools_section:
            return

        # Parse tool names from the section
        # Matches: - tool_name(args): description
        # or: - **tool_name**(args): description
        tool_pattern = r'-\s+\*?\*?(\w+)\*?\*?\s*\([^)]*\)'

        for match in re.finditer(tool_pattern, tools_section):
            tool_name = match.group(1)

            # Check if this is a pattern tool
            if tool_name in PATTERN_TOOLS:
                # Don't add if already loaded from tools/ directory
                if tool_name in self.tools:
                    log.debug(f"Pattern tool {tool_name} already loaded from tools/, skipping")
                    continue

                func = PATTERN_TOOLS[tool_name]
                description = func.__doc__.strip().split('\n')[0] if func.__doc__ else f"Apply {tool_name} pattern"

                # Create ToolInfo for the pattern
                tool_info = ToolInfo(
                    name=tool_name,
                    func=func,
                    description=description,
                    signature=f"{tool_name}(text: str) -> str",
                    examples=[f'{tool_name}("Your text here")'],
                    skill_name=self.name
                )
                # Mark it as a pattern tool
                tool_info.is_pattern = True

                self.tools[tool_name] = tool_info
                log.debug(f"Loaded pattern tool: {tool_name}")

    def _load_workflows(self) -> Dict[str, WorkflowInfo]:
        """
        Load workflow definitions from Workflows/ subdirectory.

        Workflows are Markdown files that define multi-step procedures
        for accomplishing complex tasks. Each workflow specifies:
        - Purpose: What the workflow accomplishes
        - Triggers: Phrases that should activate this workflow
        - Steps: The sequence of tools to call

        Returns:
            Dict mapping workflow name to WorkflowInfo
        """
        workflows = {}
        workflows_dir = self.skill_dir / "Workflows"

        if not workflows_dir.exists():
            return workflows

        for workflow_file in workflows_dir.glob("*.md"):
            try:
                content = workflow_file.read_text()
                name = workflow_file.stem

                # Parse purpose from ## Purpose section
                purpose = self._extract_section(content, "Purpose")
                if not purpose:
                    # Try alternative header "## What This Does"
                    purpose = self._extract_section(content, "What This Does")

                # Parse triggers from ## Triggers section
                triggers_section = self._extract_section(content, "Triggers")
                triggers = self._parse_list_items(triggers_section)

                workflows[name] = WorkflowInfo(
                    name=name,
                    purpose=purpose.strip() if purpose else f"{name} workflow",
                    triggers=triggers,
                    steps=content,  # Full content for system prompt
                    skill_name=self.name
                )

                log.debug(f"Loaded workflow: {name} ({len(triggers)} triggers)")

            except Exception as e:
                log.error(f"Error loading workflow {workflow_file}: {e}")
                continue

        return workflows

    def _parse_list_items(self, section: str) -> List[str]:
        """
        Parse markdown list items from a section.

        Handles both:
        - "item text"
        - item text

        Returns list of items with quotes stripped.
        """
        if not section:
            return []

        items = []
        for line in section.split('\n'):
            line = line.strip()
            # Match lines starting with - or *
            if line.startswith('-') or line.startswith('*'):
                # Extract the text after the bullet
                item = line[1:].strip()
                # Remove surrounding quotes if present
                if item.startswith('"') and item.endswith('"'):
                    item = item[1:-1]
                elif item.startswith("'") and item.endswith("'"):
                    item = item[1:-1]
                if item:
                    items.append(item)

        return items

    def _import_tool_module(self, tool_file: Path):
        """Import a tool module from file path"""
        module_name = f"Skills.{self.name}.tools.{tool_file.stem}"

        spec = importlib.util.spec_from_file_location(module_name, tool_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for {tool_file}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module

    def _make_async(self, func: Callable) -> Callable:
        """Wrap a sync function to make it async"""
        async def async_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

        # Preserve function metadata
        async_wrapper.__name__ = func.__name__
        async_wrapper.__doc__ = func.__doc__

        return async_wrapper

    def _generate_signature(self, func: Callable) -> str:
        """Generate function signature string from callable"""
        try:
            sig = inspect.signature(func)
            return f"{func.__name__}{sig}"
        except Exception:
            return f"{func.__name__}(...)"

    def matches_intent(self, user_input: str) -> bool:
        """
        Check if user input matches this skill's intent patterns.

        Args:
            user_input: User's natural language input

        Returns:
            True if any intent pattern matches
        """
        # Check regex patterns
        for pattern in self.routing_info.get("intent_patterns", []):
            try:
                if re.search(pattern, user_input, re.IGNORECASE):
                    log.debug(f"Matched pattern '{pattern}' in {self.name}")
                    return True
            except re.error as e:
                log.warning(f"Invalid regex pattern '{pattern}' in {self.name}: {e}")
                continue

        # Check keywords (fuzzy match)
        input_lower = user_input.lower()
        keywords = self.routing_info.get("keywords", [])
        for keyword in keywords:
            if keyword.lower() in input_lower:
                log.debug(f"Matched keyword '{keyword}' in {self.name}")
                return True

        return False

    async def execute_tool(self, tool_name: str, args: Dict) -> Any:
        """
        Execute a tool with dependency injection.

        Args:
            tool_name: Name of tool to execute
            args: Arguments to pass to tool

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool not found
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found in skill {self.name}")

        tool_info = self.tools[tool_name]

        # Inject dependencies as _deps argument
        # Tools can access via: deps = kwargs.pop('_deps', {})
        try:
            result = await tool_info.func(_deps=self.dependencies, **args)
            return result
        except TypeError as e:
            # Handle case where function doesn't accept _deps
            if "_deps" in str(e):
                # Try without _deps
                result = await tool_info.func(**args)
                return result
            else:
                raise

    def get_tool_info(self, tool_name: str) -> Optional[ToolInfo]:
        """Get information about a specific tool"""
        return self.tools.get(tool_name)

    def list_tools(self) -> List[str]:
        """List all tool names in this skill"""
        return list(self.tools.keys())


class SkillRegistry:
    """
    Registry that manages all Skills and routes requests.

    Replaces the old ToolRegistry with a Skills-based architecture
    while maintaining interface compatibility.
    """

    # Argument aliases for LLM flexibility
    # LLMs often use different names for the same concept
    ARG_ALIASES = {
        "directory": "path",
        "dir": "path",
        "folder": "path",
        "filepath": "path",
        "file_path": "path",
        "file": "path",
        "filename": "path",
        "location": "path",
        "text": "content",
        "message": "content",
        "body": "content",
        "data": "content",
        "search": "query",
        "term": "query",
        "q": "query",
        "expr": "expression",
        "math": "expression",
        "name": "title",
        "cmd": "command",
        # Research persistence aliases
        "notes": "summary",           # LLM often calls archive_source with notes instead of summary
        "description": "summary",     # Another common alias
    }

    def __init__(
        self,
        skills_dir: Path,
        dependencies: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize registry and load all skills.

        Args:
            skills_dir: Path to .workshop/Skills directory
            dependencies: Dependencies to inject into all skills
        """
        self.skills_dir = Path(skills_dir)
        self.dependencies = dependencies or {}
        self.skills: Dict[str, Skill] = {}

        log.info(f"Initializing SkillRegistry from {self.skills_dir}")

        # Create skills directory if it doesn't exist
        self.skills_dir.mkdir(parents=True, exist_ok=True)

        # Load all skills
        self._load_all_skills()

        log.info(
            f"SkillRegistry initialized with {len(self.skills)} skills, "
            f"{len(self.list_all_tools())} total tools"
        )

    def _load_all_skills(self):
        """Discover and load all skills from Skills/ directory"""
        if not self.skills_dir.exists():
            log.warning(f"Skills directory not found: {self.skills_dir}")
            return

        for skill_dir in self.skills_dir.iterdir():
            # Skip non-directories and private directories
            if not skill_dir.is_dir() or skill_dir.name.startswith("_"):
                continue

            try:
                skill = Skill(skill_dir, dependencies=self.dependencies)
                self.skills[skill.name] = skill
                log.info(f"Loaded skill: {skill.name}")
            except Exception as e:
                log.error(f"Error loading skill from {skill_dir}: {e}")
                continue

    def route_request(self, user_input: str) -> Optional[Skill]:
        """
        Determine which skill should handle this request.

        Algorithm:
        1. Sort skills by priority (HIGH > MEDIUM > LOW)
        2. For each skill in priority order:
           a. Check if any intent pattern matches
           b. If match found, return that skill
        3. If no matches, return None

        Args:
            user_input: User's natural language input

        Returns:
            Matching Skill or None
        """
        # Sort by priority
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "": 3}

        sorted_skills = sorted(
            self.skills.values(),
            key=lambda s: priority_order.get(
                s.routing_info.get("priority", ""), 3
            )
        )

        # Check each skill in priority order
        for skill in sorted_skills:
            if skill.matches_intent(user_input):
                log.info(f"Routed request to skill: {skill.name}")
                return skill

        log.debug("No skill matched user input")
        return None

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get skill by name"""
        return self.skills.get(name)

    def get_tool(self, tool_name: str) -> Optional[ToolInfo]:
        """
        Find tool by name across all skills.

        If multiple skills have the same tool name, returns the first match.
        """
        for skill in self.skills.values():
            tool_info = skill.get_tool_info(tool_name)
            if tool_info:
                return tool_info

        return None

    async def execute(self, tool_name: str, args: Dict) -> Any:
        """
        Execute a tool with signature-aware argument normalization.

        This method maintains compatibility with the old ToolRegistry interface.
        Argument aliasing (e.g., directory -> path) is only applied when the
        target function's signature accepts the aliased name.

        Args:
            tool_name: Name of tool to execute
            args: Tool arguments (will be normalized based on function signature)

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool not found
        """
        log.info(f"Executing tool: {tool_name}")
        log.debug(f"  Original args: {args}")

        # Find the tool across all skills
        tool_info = self.get_tool(tool_name)
        if not tool_info:
            log.error(f"Unknown tool: {tool_name}")
            raise ValueError(f"Unknown tool: {tool_name}")

        # Get the skill that owns this tool
        skill = self.skills[tool_info.skill_name]

        # Normalize argument names using signature-aware aliasing
        # This ensures aliases like directory->path only apply when the
        # target function actually accepts 'path' as a parameter
        normalized_args = self._normalize_args_for_function(args, tool_info.func)

        if args != normalized_args:
            log.debug(f"  Normalized args: {normalized_args}")

        # === TELEMETRY: Update current trace with normalized args ===
        try:
            from telemetry import get_current_trace
            trace = get_current_trace()
            if trace and trace.tool_calls:
                # Update the most recent tool call with normalized args
                current_tool_trace = trace.tool_calls[-1]
                if current_tool_trace.tool_name == tool_name:
                    current_tool_trace.args_normalized = normalized_args
                    current_tool_trace.dependencies_available = list(self.dependencies.keys())
        except Exception:
            pass  # Don't let telemetry errors break tool execution

        try:
            # Execute through the skill (handles dependency injection)
            result = await skill.execute_tool(tool_name, normalized_args)
            log.debug(f"  Result: {str(result)[:200]}...")
            return result

        except TypeError as e:
            # Handle missing/extra arguments
            log.error(f"  Argument error: {e}")
            return f"Error: {e}"
        except Exception as e:
            log.error(f"  Execution error: {e}")
            return f"Error: {e}"

    def _normalize_args(self, args: Dict) -> Dict:
        """Normalize argument names using aliases (legacy method, signature-unaware)"""
        normalized = {}
        for key, value in args.items():
            canonical = self.ARG_ALIASES.get(key.lower(), key)
            normalized[canonical] = value
        return normalized

    def _normalize_args_for_function(self, args: Dict, func: Callable) -> Dict:
        """
        Normalize arguments while respecting the target function's signature.

        Only applies aliases if the target function accepts the aliased parameter name.
        This prevents breaking functions that expect specific parameter names like
        `file_path` instead of the aliased `path`.

        Args:
            args: Original arguments dict
            func: Target function to inspect

        Returns:
            Normalized arguments dict that matches function signature
        """
        # Get valid parameter names from function signature
        try:
            sig = inspect.signature(func)
            valid_params = {p.name for p in sig.parameters.values() if p.name != '_deps'}
        except (ValueError, TypeError):
            # If we can't inspect the signature, fall back to basic normalization
            log.debug(f"Could not inspect signature of {func.__name__}, using basic normalization")
            return self._normalize_args(args)

        normalized = {}
        for key, value in args.items():
            key_lower = key.lower()

            # If the original key matches a valid param, use it as-is
            if key in valid_params:
                normalized[key] = value
                continue

            # Try aliasing
            canonical = self.ARG_ALIASES.get(key_lower, key)

            # Only use alias if the canonical name is a valid param
            if canonical in valid_params:
                normalized[canonical] = value
                log.debug(f"Aliased argument '{key}' -> '{canonical}'")
            elif key in valid_params:
                # Original key is valid (case-sensitive check already done above)
                normalized[key] = value
            else:
                # Neither works - keep original and log warning
                # This lets Python raise a clear error if the param is truly wrong
                normalized[key] = value
                log.warning(
                    f"Argument '{key}' (aliased to '{canonical}') not in function "
                    f"{func.__name__} signature: {valid_params}"
                )

        return normalized

    def list_all_tools(self) -> Dict[str, ToolInfo]:
        """Get flattened dict of all tools across all skills"""
        all_tools = {}
        for skill in self.skills.values():
            for tool_name, tool_info in skill.tools.items():
                # If duplicate tool names exist, first one wins
                if tool_name not in all_tools:
                    all_tools[tool_name] = tool_info
                else:
                    log.warning(
                        f"Duplicate tool name '{tool_name}' in skills "
                        f"{all_tools[tool_name].skill_name} and {tool_info.skill_name}. "
                        f"Using {all_tools[tool_name].skill_name} version."
                    )

        return all_tools

    def get_all_tools(self) -> Dict[str, dict]:
        """
        Get all tools in format compatible with Agent.

        Returns dict matching old ToolRegistry.get_all_tools() format:
        {
            "tool_name": {
                "description": "...",
                "signature": "...",
                "examples": [...],
            }
        }
        """
        all_tools = self.list_all_tools()

        # Convert to old format
        result = {}
        for tool_name, tool_info in all_tools.items():
            result[tool_name] = {
                "description": tool_info.description,
                "signature": tool_info.signature,
                "examples": tool_info.examples,
            }

        return result

    def route_by_intent(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> IntentMatch:
        """
        Route user input to the appropriate skill, tool, or workflow.

        This is the main routing function that:
        1. Resolves ambiguous context references ("here", "current directory")
        2. Checks workflow triggers first (multi-step procedures)
        3. Matches against skill intent patterns
        4. Extracts relevant arguments from the query
        5. Returns detailed routing information

        Args:
            user_input: User's natural language input
            context: Runtime context dict with keys like:
                - working_directory: Current working directory
                - active_project: Currently active project path
                - recent_files: Recently accessed files

        Returns:
            IntentMatch with routing decision and extracted context
        """
        context = context or {}
        working_dir = context.get("working_directory", ".")

        # Normalize user input
        input_lower = user_input.lower().strip()

        # Step 1: Check for workflow matches first (highest priority)
        workflow_match = self._match_workflow(user_input)
        if workflow_match.matched:
            log.info(f"Matched workflow: {workflow_match.workflow_name}")
            return workflow_match

        # Step 2: Check skill patterns with context extraction
        best_match = IntentMatch(skill=None, skill_name="")

        # Sort skills by priority
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        sorted_skills = sorted(
            self.skills.values(),
            key=lambda s: priority_order.get(
                s.routing_info.get("priority", "MEDIUM"), 1
            )
        )

        for skill in sorted_skills:
            match = self._match_skill_patterns(skill, user_input, context)
            if match.matched and match.confidence > best_match.confidence:
                best_match = match

        if best_match.matched:
            log.info(
                f"Matched skill: {best_match.skill_name}, "
                f"tool: {best_match.tool_name}, "
                f"confidence: {best_match.confidence:.2f}"
            )
            return best_match

        # Step 3: Fall back to keyword matching
        for skill in sorted_skills:
            keywords = skill.routing_info.get("keywords", [])
            for keyword in keywords:
                if keyword.lower() in input_lower:
                    log.info(f"Matched skill by keyword: {skill.name} ({keyword})")
                    return IntentMatch(
                        skill=skill,
                        skill_name=skill.name,
                        matched_pattern=f"keyword:{keyword}",
                        confidence=0.5,
                        match_type="keyword"
                    )

        # No match found
        log.debug(f"No intent match for: {user_input[:50]}...")
        return IntentMatch(skill=None, skill_name="")

    def _match_workflow(self, user_input: str) -> IntentMatch:
        """
        Check if user input matches any workflow trigger.

        Workflows are checked first because they represent
        high-level user intents that span multiple tools.
        """
        input_lower = user_input.lower()

        for skill in self.skills.values():
            for workflow_name, workflow in skill.workflows.items():
                for trigger in workflow.triggers:
                    # Simple substring match for triggers
                    if trigger.lower() in input_lower:
                        log.debug(f"Matched workflow trigger: {trigger} -> {workflow_name}")
                        return IntentMatch(
                            skill=skill,
                            skill_name=skill.name,
                            workflow_name=workflow_name,
                            matched_pattern=f"workflow_trigger:{trigger}",
                            confidence=0.9,  # Workflows are high confidence
                            match_type="workflow"
                        )

        return IntentMatch(skill=None, skill_name="")

    def _match_skill_patterns(
        self,
        skill: 'Skill',
        user_input: str,
        context: Dict[str, Any]
    ) -> IntentMatch:
        """
        Match user input against a skill's intent patterns.

        Also extracts any captured groups as arguments.
        """
        patterns = skill.routing_info.get("intent_patterns", [])

        for pattern in patterns:
            try:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    # Extract captured groups as potential arguments
                    extracted_args = {}
                    groups = match.groups()

                    if groups:
                        # Try to intelligently name the extracted args
                        extracted_args = self._extract_args_from_match(
                            pattern, groups, context
                        )

                    # Determine confidence based on match quality
                    # Full match = higher confidence
                    match_length = match.end() - match.start()
                    input_length = len(user_input)
                    confidence = 0.6 + (0.4 * match_length / input_length)

                    return IntentMatch(
                        skill=skill,
                        skill_name=skill.name,
                        matched_pattern=pattern,
                        extracted_args=extracted_args if extracted_args else None,
                        confidence=min(confidence, 0.95),
                        match_type="pattern"
                    )

            except re.error as e:
                log.warning(f"Invalid regex pattern '{pattern}' in {skill.name}: {e}")
                continue

        return IntentMatch(skill=None, skill_name="")

    def _extract_args_from_match(
        self,
        pattern: str,
        groups: tuple,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract named arguments from regex match groups.

        Uses heuristics to determine what each captured group represents:
        - Paths (contain / or ~)
        - Symbols (single word, CamelCase or snake_case)
        - Queries (longer text)
        """
        args = {}
        working_dir = context.get("working_directory", ".")

        for i, value in enumerate(groups):
            if not value:
                continue

            value = value.strip()

            # Resolve ambiguous path references
            resolved_value = self._resolve_context_reference(value, context)
            if resolved_value != value:
                # It was a context reference that got resolved
                args["path"] = resolved_value
                continue

            # Determine argument type by content
            if "/" in value or value.startswith("~") or value.startswith("."):
                # Looks like a path
                args["path"] = value
            elif re.match(r'^[A-Z][a-zA-Z0-9]*$', value):
                # CamelCase - likely a class/type name
                args["symbol"] = value
            elif re.match(r'^[a-z_][a-z0-9_]*$', value):
                # snake_case - likely a function/variable name
                args["symbol"] = value
            elif len(value.split()) > 2:
                # Multiple words - likely a search query
                args["query"] = value
            else:
                # Generic captured text
                args[f"arg{i}"] = value

        return args

    def _resolve_context_reference(
        self,
        value: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Resolve ambiguous context references to concrete paths.

        Handles:
        - "here", "current directory", "this folder" -> working_directory
        - "the project", "this project" -> active_project
        - "recently", "recent" -> triggers recent files lookup
        """
        value_lower = value.lower().strip()
        working_dir = context.get("working_directory", ".")
        active_project = context.get("active_project", working_dir)

        # Current directory references
        current_dir_refs = [
            "here", "current directory", "this directory",
            "this folder", "current folder", "cwd", "pwd",
            ".", "current", "this"
        ]
        if value_lower in current_dir_refs:
            return working_dir

        # Project references
        project_refs = [
            "the project", "this project", "project",
            "project directory", "project folder"
        ]
        if value_lower in project_refs:
            return active_project

        # Home directory
        if value_lower in ["home", "~", "home directory"]:
            return str(Path.home())

        return value

    def resolve_query_context(
        self,
        user_input: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Enrich user query by resolving ambiguous references.

        This is used to transform queries like:
        "what files are here?" -> "what files are in /home/user/project?"

        Args:
            user_input: Original user query
            context: Runtime context with working_directory, etc.

        Returns:
            Query with resolved references
        """
        result = user_input
        working_dir = context.get("working_directory", ".")

        # Patterns to replace with working directory
        replacements = [
            (r'\bhere\b', working_dir),
            (r'\bcurrent directory\b', working_dir),
            (r'\bthis directory\b', working_dir),
            (r'\bthis folder\b', working_dir),
            (r'\bcurrent folder\b', working_dir),
        ]

        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        return result

    def get_workflow(self, workflow_name: str) -> Optional[WorkflowInfo]:
        """
        Get a workflow by name from any skill.
        """
        for skill in self.skills.values():
            if workflow_name in skill.workflows:
                return skill.workflows[workflow_name]
        return None

    def list_all_workflows(self) -> Dict[str, WorkflowInfo]:
        """
        Get all workflows across all skills.
        """
        workflows = {}
        for skill in self.skills.values():
            for name, workflow in skill.workflows.items():
                if name not in workflows:
                    workflows[name] = workflow
        return workflows

    def generate_capabilities_manifest(self, write_to_file: bool = True) -> str:
        """
        Generate a markdown manifest of all capabilities for context injection.

        This manifest describes all available skills, tools, and workflows,
        making it easier for the agent to understand what it can do.

        Args:
            write_to_file: If True, writes to ~/.workshop/CAPABILITIES.md

        Returns:
            Markdown string describing all capabilities
        """
        lines = [
            "# Workshop Capabilities Manifest",
            "",
            "This document describes all available skills, tools, and workflows.",
            "Auto-generated at startup - do not edit manually.",
            ""
        ]

        # Sort skills by priority for display
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        sorted_skills = sorted(
            self.skills.values(),
            key=lambda s: priority_order.get(
                s.routing_info.get("priority", "MEDIUM"), 1
            )
        )

        for skill in sorted_skills:
            skill_priority = skill.routing_info.get("priority", "MEDIUM")
            lines.append(f"## {skill.name}")
            lines.append("")

            # Purpose
            purpose = skill.routing_info.get("purpose", "No description available")
            if purpose:
                lines.append(f"**Purpose:** {purpose[:200]}")
                lines.append("")

            # Priority badge
            lines.append(f"**Priority:** {skill_priority}")
            lines.append("")

            # List tools
            if skill.tools:
                lines.append("### Tools")
                lines.append("")
                for tool_name, tool_info in skill.tools.items():
                    desc = tool_info.description
                    # Truncate long descriptions
                    if len(desc) > 120:
                        desc = desc[:117] + "..."
                    # Mark pattern tools
                    is_pattern = getattr(tool_info, 'is_pattern', False)
                    pattern_tag = " _(pattern)_" if is_pattern else ""
                    lines.append(f"- **{tool_name}**{pattern_tag}: {desc}")
                lines.append("")

            # List workflows
            if skill.workflows:
                lines.append("### Workflows")
                lines.append("")
                for workflow_name, workflow_info in skill.workflows.items():
                    lines.append(f"- **{workflow_name}**: {workflow_info.purpose[:100]}")
                lines.append("")

            # Keywords for routing
            keywords = skill.routing_info.get("keywords", [])
            if keywords:
                lines.append(f"**Keywords:** {', '.join(keywords[:10])}")
                lines.append("")

        # Summary section
        lines.append("---")
        lines.append("")
        lines.append("## Summary")
        lines.append("")

        total_tools = len(self.list_all_tools())
        total_workflows = len(self.list_all_workflows())
        total_skills = len(self.skills)

        lines.append(f"- **Skills:** {total_skills}")
        lines.append(f"- **Tools:** {total_tools}")
        lines.append(f"- **Workflows:** {total_workflows}")

        manifest = "\n".join(lines)

        # Write to file if requested
        if write_to_file:
            manifest_path = Path.home() / ".workshop" / "CAPABILITIES.md"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(manifest)
            log.info(f"Capabilities manifest written to {manifest_path}")

        return manifest

    def get_capabilities_summary(self) -> str:
        """
        Get a condensed capabilities summary for context injection.

        Returns a shorter version suitable for system prompts, listing
        just tool names grouped by skill with one-line descriptions.

        Returns:
            Condensed markdown string for system prompt injection
        """
        lines = [
            "## Available Capabilities",
            "",
            "You have access to the following tools organized by skill:",
            ""
        ]

        # Sort skills by priority
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        sorted_skills = sorted(
            self.skills.values(),
            key=lambda s: priority_order.get(
                s.routing_info.get("priority", "MEDIUM"), 1
            )
        )

        for skill in sorted_skills:
            if not skill.tools:
                continue

            lines.append(f"### {skill.name}")

            # List tools with short descriptions
            for tool_name, tool_info in skill.tools.items():
                # Get first sentence of description
                desc = tool_info.description.split('.')[0]
                if len(desc) > 80:
                    desc = desc[:77] + "..."
                lines.append(f"- `{tool_name}`: {desc}")

            lines.append("")

        # Add workflows if any
        all_workflows = self.list_all_workflows()
        if all_workflows:
            lines.append("### Workflows (Multi-step Procedures)")
            for name, workflow in all_workflows.items():
                lines.append(f"- `{name}`: {workflow.purpose[:60]}")
            lines.append("")

        return "\n".join(lines)
