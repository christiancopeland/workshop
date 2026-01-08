"""
Fabric-Style Two-Stage Router for Workshop.

This module implements a two-stage routing system inspired by Daniel Miessler's Fabric:
1. Stage 1 (Router): Fast intent classification → outputs skill name, pattern name, or pipeline
2. Stage 2 (Executor): Execute skill's system.md OR pattern's system.md

The key insight: Don't try to extract arguments with regex. Let the LLM always do it,
but constrain it to only the matched skill's tools (3-5 tools, not 100+).

NEW: Pattern Support (Fabric-style atomic prompts)
- Patterns are stateless text transformations (no tools)
- Patterns can be chained: "extract_wisdom | create_summary"
- Patterns are detected before skills for text transformation requests

Architecture:
    User Input
        │
        ▼
    ┌─────────────────────────────────────┐
    │ PATTERN DETECTION                   │
    │ - Check for pattern triggers        │
    │ - Detect pipeline syntax            │
    │ - Returns: pattern | pipeline | None│
    └─────────────────────────────────────┘
        │ (if None)
        ▼
    ┌─────────────────────────────────────┐
    │ SEMANTIC PRE-FILTER (optional)      │
    │ - Embedding similarity              │
    │ - Returns top candidates            │
    └─────────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────┐
    │ ROUTER PROMPT (phi3:mini)           │
    │ - Classifies intent                 │
    │ - Outputs: skill_name | chat | clarify │
    └─────────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────┐
    │ SKILL EXECUTOR (phi4:14b)           │
    │ - Loads skill's system.md           │
    │ - Exposes ONLY skill's tools        │
    │ - LLM extracts args and calls tools │
    └─────────────────────────────────────┘
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from logger import get_logger
from config import Config

# Claude Code integration for low-confidence routing fallback
# Phase 4: Use shared bridge for session continuity
try:
    from claude_bridge import ClaudeCodeBridge, get_claude_bridge
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    ClaudeCodeBridge = None
    get_claude_bridge = None

# Import pattern registry for pattern routing
try:
    from pattern_executor import PatternRegistry, PatternPipeline
    PATTERNS_AVAILABLE = True
except ImportError:
    PATTERNS_AVAILABLE = False
    PatternRegistry = None
    PatternPipeline = None

# Dashboard integration for pattern events
try:
    import dashboard_integration as dash
except ImportError:
    dash = None

log = get_logger("router")


@dataclass
class RoutingResult:
    """Result of the routing decision."""
    skill_name: str  # The matched skill, "chat", or "clarify"
    method: str  # "semantic_direct", "router_prompt", "pattern", "pipeline", "fallback"
    confidence: float
    matched_utterance: str = ""
    candidates: List[str] = field(default_factory=list)  # Top candidates considered

    # Pattern routing fields
    pattern_name: Optional[str] = None  # Single pattern to execute
    pipeline: Optional[List[str]] = None  # List of patterns to chain

    @property
    def needs_clarification(self) -> bool:
        return self.skill_name == "clarify"

    @property
    def is_chat(self) -> bool:
        return self.skill_name == "chat"

    @property
    def is_skill(self) -> bool:
        return self.skill_name not in ("chat", "clarify", None, "") and not self.is_pattern

    @property
    def is_pattern(self) -> bool:
        return self.pattern_name is not None or self.pipeline is not None

    @property
    def is_pipeline(self) -> bool:
        return self.pipeline is not None and len(self.pipeline) > 1


@dataclass
class SkillInfo:
    """Information about a loaded skill."""
    name: str
    purpose: str
    system_prompt: str  # Content of system.md
    tools: List[str]  # Tool names available in this skill
    keywords: List[str]


class PromptRouter:
    """
    Fabric-style two-stage routing for Workshop.

    Stage 1: Route to skill (fast)
    Stage 2: Execute skill with constrained tools (in SkillExecutor)

    Usage:
        router = PromptRouter()
        await router.initialize()

        result = await router.route("research Daniel Miessler PAI")
        # RoutingResult(skill_name="Research", method="router_prompt", confidence=0.8)
    """

    # Thresholds for semantic pre-filtering
    SEMANTIC_BYPASS_THRESHOLD = 0.85  # Skip router, go direct to skill
    SEMANTIC_CONFIRM_THRESHOLD = 0.45  # Router confirms from top candidates

    def __init__(
        self,
        skills_dir: Path = None,
        agents_dir: Path = None,
        ollama_url: str = None,
        router_model: str = None,
        enable_patterns: bool = True,
        use_claude: bool = True,  # Use Claude Code for low-confidence fallback
    ):
        self.skills_dir = skills_dir or Path(".workshop/Skills")
        self.agents_dir = agents_dir or Path.home() / ".workshop" / "agents"
        self.ollama_url = ollama_url or Config.OLLAMA_URL
        self.router_model = router_model or Config.ROUTER_MODEL
        self.enable_patterns = enable_patterns and PATTERNS_AVAILABLE
        self.use_claude = use_claude and CLAUDE_AVAILABLE

        # Loaded skills
        self.skills: Dict[str, SkillInfo] = {}

        # Router prompt template (loaded from agents/router.md)
        self.router_prompt_template: str = ""

        # Semantic router (optional, for pre-filtering)
        self._semantic_router = None

        # Pattern registry (for Fabric-style patterns)
        self._pattern_registry: Optional[PatternRegistry] = None

        # Claude Code bridge for low-confidence routing (Phase 3 migration)
        # Phase 4: Use shared bridge for session continuity
        self._claude_bridge: Optional[ClaudeCodeBridge] = None
        if self.use_claude:
            try:
                # Use shared bridge instead of creating a new instance
                self._claude_bridge = get_claude_bridge(timeout_seconds=180)  # Match skill_executor timeout
                log.info("PromptRouter: Claude Code bridge initialized (shared instance)")
            except Exception as e:
                log.warning(f"Failed to initialize Claude Code for router: {e}")
                self.use_claude = False

        self._initialized = False

    async def initialize(self, semantic_router=None):
        """
        Initialize the router.

        Args:
            semantic_router: Optional SemanticRouter instance for pre-filtering
        """
        if self._initialized:
            return

        # Load skills from .workshop/Skills/
        self._load_skills()

        # Load router prompt from ~/.workshop/agents/router.md
        self._load_router_prompt()

        # Store semantic router reference if provided
        self._semantic_router = semantic_router

        # Initialize pattern registry if patterns are enabled
        if self.enable_patterns and PatternRegistry:
            self._pattern_registry = PatternRegistry()
            await self._pattern_registry.initialize()
            log.info(f"Pattern registry loaded with {len(self._pattern_registry.list_patterns())} patterns")

        self._initialized = True
        log.info(f"PromptRouter initialized with {len(self.skills)} skills")

    def _load_skills(self):
        """Load skill information from .workshop/Skills/ directories."""
        if not self.skills_dir.exists():
            log.warning(f"Skills directory not found: {self.skills_dir}")
            return

        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue

            skill_md = skill_dir / "SKILL.md"
            system_md = skill_dir / "system.md"

            if not skill_md.exists():
                continue

            try:
                skill_info = self._parse_skill(skill_dir.name, skill_md, system_md)
                self.skills[skill_info.name] = skill_info
                log.debug(f"Loaded skill: {skill_info.name} ({len(skill_info.tools)} tools)")
            except Exception as e:
                log.error(f"Failed to load skill {skill_dir.name}: {e}")

    def _parse_skill(self, name: str, skill_md: Path, system_md: Path) -> SkillInfo:
        """Parse skill metadata and system prompt."""
        import re

        content = skill_md.read_text()

        # Extract purpose
        purpose = ""
        purpose_section = re.search(r'## Purpose\s*\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
        if purpose_section:
            # Get first line/sentence of purpose
            purpose = purpose_section.group(1).strip().split('\n')[0]

        # Extract tools
        tools = []
        # Match ## Tools or ## Available Tools section, stopping at next ## (but not ###)
        tools_section = re.search(r'## (?:Available )?Tools?\s*\n(.+?)(?=\n## [^#]|\Z)', content, re.DOTALL)
        if tools_section:
            for line in tools_section.group(1).split('\n'):
                line = line.strip()
                if line.startswith('-'):
                    # Extract tool name (first word or before parenthesis)
                    tool_line = line[1:].strip()
                    tool_name = tool_line.split('(')[0].split(':')[0].strip().strip('*')
                    if tool_name:
                        tools.append(tool_name)

        # Extract keywords
        keywords = []
        keywords_section = re.search(r'## Keywords?\s*\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
        if keywords_section:
            kw_text = keywords_section.group(1).strip()
            keywords = [k.strip() for k in kw_text.replace('\n', ',').split(',') if k.strip()]

        # Load system prompt (Fabric-style)
        system_prompt = ""
        if system_md.exists():
            system_prompt = system_md.read_text()
        else:
            # Fallback: generate basic system prompt from SKILL.md
            system_prompt = self._generate_fallback_system_prompt(name, purpose, tools)

        return SkillInfo(
            name=name,
            purpose=purpose,
            system_prompt=system_prompt,
            tools=tools,
            keywords=keywords,
        )

    def _generate_fallback_system_prompt(self, name: str, purpose: str, tools: List[str]) -> str:
        """Generate a basic system prompt if system.md doesn't exist."""
        tools_list = "\n".join(f"- {tool}" for tool in tools)

        return f"""# IDENTITY and PURPOSE

You are Workshop's {name} specialist. {purpose}

Take a deep breath and think step by step about how to accomplish this task.

# AVAILABLE TOOLS

You have access to these tools ONLY:
{tools_list}

# STEPS

1. Understand what the user wants
2. Choose the appropriate tool
3. Extract the correct arguments from the user's message
4. Call the tool
5. Present results clearly

# OUTPUT INSTRUCTIONS

- Always use a tool - never make up information
- Extract clean arguments from the user's message
- Do not apologize or add unnecessary caveats

# INPUT

{{{{user_input}}}}
"""

    def _load_router_prompt(self):
        """Load the router prompt template from ~/.workshop/agents/router.md."""
        router_md = self.agents_dir / "router.md"

        if router_md.exists():
            content = router_md.read_text()

            # Skip YAML frontmatter if present
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    content = parts[2].strip()

            self.router_prompt_template = content
            log.debug("Loaded router prompt from router.md")
        else:
            # Use built-in router prompt
            self.router_prompt_template = self._get_builtin_router_prompt()
            log.debug("Using built-in router prompt")

    def _get_builtin_router_prompt(self) -> str:
        """Built-in router prompt (Fabric-style)."""
        return """# IDENTITY

You are Workshop's intent classifier. Your ONLY job is to output the skill name that best matches the user's request.

# AVAILABLE SKILLS

{{skill_list}}

# RULES

- Output ONLY the skill name, nothing else
- If the request is conversational (greetings, thanks, chitchat), output: chat
- If the request is ambiguous or unclear, output: clarify
- When multiple skills could match, choose the most specific one
- Research/search/lookup → Research
- Files/directories/code → FileOperations or ContextIntelligence
- Remember/recall/notes → Memory
- Tasks/progress/todo → TaskManagement
- Arduino/ESP32/compile/upload → Arduino

# EXAMPLES

User: "search for Python tutorials"
Output: Research

User: "research Daniel Miessler's PAI concept"
Output: Research

User: "what files are in the src directory"
Output: FileOperations

User: "where is the Agent class defined"
Output: ContextIntelligence

User: "remember that the API key is xyz"
Output: Memory

User: "what are you working on"
Output: TaskManagement

User: "compile the blink sketch"
Output: Arduino

User: "hello how are you"
Output: chat

User: "thanks that's helpful"
Output: chat

User: "do the thing"
Output: clarify

# INPUT

{{user_input}}"""

    def _build_router_prompt(self, user_input: str, candidates: List[str] = None, context: Dict[str, Any] = None) -> str:
        """
        Build the router prompt with skill list and user input.

        Args:
            user_input: The user's message
            candidates: Optional list of candidate skills (from semantic pre-filter)
            context: Optional context dict with task_context, working_directory, etc.
        """
        # Determine which skills to include
        if candidates:
            skills_to_show = [self.skills[name] for name in candidates if name in self.skills]
        else:
            skills_to_show = list(self.skills.values())

        # Build skill list
        skill_list = "\n".join(
            f"- {skill.name}: {skill.purpose}"
            for skill in skills_to_show
        )

        # Always include chat and clarify as options
        skill_list += "\n- chat: Casual conversation, greetings, thanks"
        skill_list += "\n- clarify: Request is unclear or ambiguous"

        # Build prompt
        prompt = self.router_prompt_template
        prompt = prompt.replace("{{skill_list}}", skill_list)

        # Inject task context if available - helps with "continue", "go on", etc.
        # IMPORTANT: Frame context so the router doesn't hallucinate task names as skill names
        enriched_input = user_input
        if context and context.get("task_context"):
            # Build the valid outputs reminder
            valid_skills = list(self.skills.keys()) + ["chat", "clarify"]
            valid_outputs_str = ", ".join(valid_skills)

            enriched_input = f"""[BACKGROUND CONTEXT - for understanding only, do NOT output these as skill names]
{context['task_context']}

[USER REQUEST]
{user_input}

[CRITICAL REMINDER: Your output must be EXACTLY one of: {valid_outputs_str}]"""

        prompt = prompt.replace("{{user_input}}", enriched_input)

        return prompt

    async def _detect_pattern(self, user_input: str, trace_id: str = None) -> Optional[RoutingResult]:
        """
        Detect if the user input matches a pattern or pipeline.

        Checks:
        1. Explicit pipeline syntax: "extract_wisdom | create_summary"
        2. Natural language pipeline: "extract wisdom, then summarize"
        3. Pattern triggers: "summarize this", "extract the wisdom"

        Returns:
            RoutingResult if pattern detected, None otherwise
        """
        if not self._pattern_registry:
            return None

        # Check for pipeline syntax first
        if PatternPipeline:
            pipeline = PatternPipeline.parse_pipeline_syntax(user_input)
            if pipeline:
                # Validate all patterns exist
                valid_pipeline = []
                for pattern_name in pipeline:
                    if self._pattern_registry.get_pattern_info(pattern_name):
                        valid_pipeline.append(pattern_name)
                    else:
                        # Try without underscores
                        alt_name = pattern_name.replace("_", "")
                        for actual_name in self._pattern_registry.list_patterns():
                            if actual_name.replace("_", "") == alt_name:
                                valid_pipeline.append(actual_name)
                                break

                if len(valid_pipeline) >= 2:
                    log.info(f"Detected pipeline: {valid_pipeline}")

                    # Emit dashboard event
                    if dash:
                        await dash.pipeline_detected(valid_pipeline, trace_id)

                    return RoutingResult(
                        skill_name="pattern",
                        method="pipeline",
                        confidence=0.9,
                        pipeline=valid_pipeline,
                    )

        # Check for single pattern triggers
        matched_pattern = self._pattern_registry.match_pattern(user_input)
        if matched_pattern:
            log.info(f"Detected pattern trigger: {matched_pattern}")

            # Emit dashboard event
            if dash:
                await dash.pattern_detected(matched_pattern, "trigger", trace_id)

            return RoutingResult(
                skill_name="pattern",
                method="pattern",
                confidence=0.85,
                pattern_name=matched_pattern,
            )

        return None

    async def route(self, user_input: str, context: Dict[str, Any] = None) -> RoutingResult:
        """
        Route user input to the appropriate skill or pattern.

        Phase 3 Migration: Semantic-first routing with Claude fallback.

        Routing strategy:
        1. Pattern detection → check for pattern triggers or pipeline syntax
        2. Semantic routing (embeddings only, no LLM):
           - High confidence (≥0.85): Direct to skill
           - Medium confidence (0.45-0.85): Trust semantic match (no LLM confirmation)
           - Low confidence (<0.45): Claude fallback for clarification
        3. Fallback to chat if nothing matches

        Args:
            user_input: The user's message
            context: Optional context (working directory, active project, etc.)

        Returns:
            RoutingResult with skill_name and metadata
        """
        if not self._initialized:
            await self.initialize()

        # Extract trace_id from context if available
        trace_id = context.get("trace_id") if context else None

        # Stage 0: Pattern detection (Fabric-style patterns)
        if self.enable_patterns:
            pattern_result = await self._detect_pattern(user_input, trace_id)
            if pattern_result:
                return pattern_result

        # Stage 0.5: Active skill continuation (auto-continue bypass)
        # When auto-continue sends "Continue.", honor the active_skill from context
        # This prevents re-routing ambiguous continuation prompts
        if context and context.get("active_skill"):
            active_skill = context["active_skill"]
            if active_skill in self.skills:
                # Check if there are pending tasks (from task_context)
                has_pending = "pending" in context.get("task_context", "").lower()
                if has_pending or user_input.strip().lower() in ("continue", "continue."):
                    log.info(f"Continuing active skill: {active_skill} (bypass routing)")
                    return RoutingResult(
                        skill_name=active_skill,
                        method="active_skill_continue",
                        confidence=0.95,
                    )

        # Stage 1: Semantic routing (embeddings only - Phase 3 migration)
        if self._semantic_router:
            try:
                semantic_match = await self._semantic_router.route(user_input)

                if semantic_match.confidence >= self.SEMANTIC_BYPASS_THRESHOLD:
                    # Very high confidence - direct routing
                    log.info(f"Semantic routing (high): {semantic_match.skill_name} ({semantic_match.confidence:.3f})")
                    return RoutingResult(
                        skill_name=semantic_match.skill_name,
                        method="semantic_direct",
                        confidence=semantic_match.confidence,
                        matched_utterance=semantic_match.matched_utterance,
                    )

                elif semantic_match.confidence >= self.SEMANTIC_CONFIRM_THRESHOLD:
                    # Medium confidence - TRUST semantic match (Phase 3: no LLM confirmation)
                    log.info(f"Semantic routing (medium): {semantic_match.skill_name} ({semantic_match.confidence:.3f})")
                    return RoutingResult(
                        skill_name=semantic_match.skill_name,
                        method="semantic_trusted",
                        confidence=semantic_match.confidence,
                        matched_utterance=semantic_match.matched_utterance,
                    )

                else:
                    # Low confidence - need clarification or fallback
                    log.info(f"Semantic routing (low): {semantic_match.confidence:.3f} - using fallback")

            except Exception as e:
                log.warning(f"Semantic routing failed: {e}")

        # Stage 2: Fallback for low-confidence or no semantic match
        # Check for obvious chat patterns first (fast path)
        if self._is_chat_request(user_input):
            return RoutingResult(
                skill_name="chat",
                method="pattern_chat",
                confidence=0.8,
            )

        # Use Claude for ambiguous routing (Phase 3: replaces Ollama phi3:mini)
        if self.use_claude and self._claude_bridge:
            skill_name = await self._run_claude_router(user_input, context)
            return RoutingResult(
                skill_name=skill_name,
                method="claude_fallback",
                confidence=0.7 if skill_name not in ("chat", "clarify") else 0.5,
            )

        # Final fallback: request clarification
        return RoutingResult(
            skill_name="clarify",
            method="fallback",
            confidence=0.3,
        )

    def _is_chat_request(self, user_input: str) -> bool:
        """Quick check for obvious chat/conversational patterns."""
        chat_patterns = [
            r'^(hi|hello|hey|greetings|good (morning|afternoon|evening))',
            r'^(thanks?|thank you|thx|ty)',
            r'^(bye|goodbye|see you|later)',
            r'^(how are you|what\'?s up|sup)',
            r'^(ok|okay|sure|got it|understood)',
        ]
        user_lower = user_input.lower().strip()
        for pattern in chat_patterns:
            if re.match(pattern, user_lower, re.IGNORECASE):
                return True
        return False

    async def _run_claude_router(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """
        Use Claude Code for intent classification (Phase 3 migration).

        This replaces the Ollama phi3:mini router for low-confidence cases.
        Only called when semantic routing doesn't produce a confident match.

        Args:
            user_input: The user's message
            context: Optional context dict with task_context, etc.

        Returns:
            Skill name, "chat", or "clarify"
        """
        if not self._claude_bridge:
            return "clarify"

        # Build skill list for the prompt
        skill_list = "\n".join(
            f"- {skill.name}: {skill.purpose}"
            for skill in self.skills.values()
        )

        # Build a concise routing prompt for Claude
        router_prompt = f"""You are a routing assistant. Classify the user's request to one of these skills:

{skill_list}
- chat: Casual conversation, greetings, thanks, chitchat
- clarify: Request is unclear or ambiguous

Rules:
- Output ONLY the skill name, nothing else
- Be precise: "research X" → Research, "list files" → FileOperations, etc.

User request: "{user_input}"

Output:"""

        try:
            messages = [{"role": "user", "content": router_prompt}]
            result = await self._claude_bridge.query(
                messages=messages,
                system_prompt="You are a concise intent classifier. Output only a single word: the skill name.",
                disable_native_tools=True,
                max_turns=1,
            )

            raw_response = result.get("content", "").strip()
            skill_name = self._parse_router_response(raw_response, context)

            log.info(f"Claude router: '{user_input[:50]}...' → {skill_name}")
            return skill_name

        except asyncio.TimeoutError:
            log.error("Claude router timed out")
            return "clarify"
        except Exception as e:
            log.error(f"Claude router error: {e}")
            return "clarify"

    # DEPRECATED: Ollama-based routing (kept for backward compatibility)
    # Use _run_claude_router instead (Phase 3 migration)
    async def _run_router_prompt_ollama(self, user_input: str, candidates: List[str] = None, context: Dict[str, Any] = None) -> str:
        """
        [DEPRECATED] Execute the router prompt via Ollama.

        This method is deprecated as of Phase 3 migration. Use _run_claude_router instead.
        Kept for backward compatibility if use_claude=False.
        """
        # Import aiohttp only if needed (Ollama fallback)
        import aiohttp

        prompt = self._build_router_prompt(user_input, candidates, context)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.router_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_predict": 20,
                            "temperature": 0.1,
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        log.error(f"Router prompt failed: {response.status}")
                        return "chat"

                    result = await response.json()
                    raw_response = result.get("response", "").strip()
                    skill_name = self._parse_router_response(raw_response, context)
                    log.info(f"Ollama router: '{user_input[:50]}...' → {skill_name}")
                    return skill_name

        except asyncio.TimeoutError:
            log.error("Router prompt timed out")
            return "chat"
        except Exception as e:
            log.error(f"Router prompt error: {e}")
            return "chat"

    def _parse_router_response(self, response: str, context: Dict[str, Any] = None) -> str:
        """
        Parse the router's response to extract skill name.

        The router should output just the skill name, but may include extra text.
        If parsing fails and context contains an active_skill, fall back to that.
        """
        response = response.strip()

        # Handle common formats
        # "Research" → Research
        # "Output: Research" → Research
        # "The best match is Research" → Research

        # First, check if any skill name appears in the response
        for skill_name in self.skills.keys():
            if skill_name.lower() in response.lower():
                return skill_name

        # Check for special responses
        if "chat" in response.lower():
            return "chat"
        if "clarify" in response.lower():
            return "clarify"

        # Try to extract first word as skill name
        first_word = response.split()[0] if response.split() else ""
        first_word = first_word.strip('.:,')

        if first_word in self.skills:
            return first_word

        # Try case-insensitive match
        for skill_name in self.skills.keys():
            if skill_name.lower() == first_word.lower():
                return skill_name

        # CONTEXT-DRIVEN FALLBACK: If we have an active skill from context, use it
        # This handles cases where the router hallucinates task names or other invalid outputs
        if context and context.get("active_skill"):
            active_skill = context["active_skill"]
            if active_skill in self.skills:
                log.info(f"Router parse failed, falling back to active skill: {active_skill}")
                return active_skill

        # Final fallback
        log.warning(f"Could not parse router response: '{response}'")
        return "chat"

    def get_skill(self, skill_name: str) -> Optional[SkillInfo]:
        """Get skill info by name."""
        return self.skills.get(skill_name)

    def get_skill_tools(self, skill_name: str) -> List[str]:
        """Get list of tool names for a skill."""
        skill = self.skills.get(skill_name)
        return skill.tools if skill else []

    def get_skill_prompt(self, skill_name: str) -> str:
        """Get the system prompt for a skill."""
        skill = self.skills.get(skill_name)
        return skill.system_prompt if skill else ""
