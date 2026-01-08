"""
Pattern Tools - Exposes Fabric-style patterns as callable tools.

This module bridges the gap between patterns (text transformations) and the
tool system, allowing patterns to be used automatically by the LLM during
skill execution, just like any other tool.

The key insight: Patterns ARE tools - they just happen to be text-in, text-out
transformations defined by specialized prompts rather than Python functions.

Usage:
    # Register pattern tools with a skill
    from pattern_tools import get_pattern_tools, apply_pattern

    # Get tool definitions for a skill's SKILL.md
    tools = get_pattern_tools(["extract_wisdom", "create_summary"])

    # Or use directly in Python
    result = await apply_pattern("extract_wisdom", article_text)
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path

from logger import get_logger
from pattern_executor import PatternExecutor, PatternRegistry, PatternResult

log = get_logger("pattern_tools")

# Singleton executor for efficiency
_executor: Optional[PatternExecutor] = None


async def get_executor() -> PatternExecutor:
    """Get or create the singleton pattern executor."""
    global _executor
    if _executor is None:
        _executor = PatternExecutor()
        await _executor.initialize()
    return _executor


async def apply_pattern(
    pattern_name: str,
    input_text: str,
    model: str = None,
) -> str:
    """
    Apply a pattern to input text and return the result.

    This is the main tool function that gets registered with skills.

    Args:
        pattern_name: Name of the pattern (e.g., "extract_wisdom")
        input_text: The text to transform
        model: Optional model override

    Returns:
        The transformed text

    Raises:
        ValueError: If pattern doesn't exist or execution fails
    """
    executor = await get_executor()
    result = await executor.execute(pattern_name, input_text, model=model)

    if not result.success:
        raise ValueError(f"Pattern '{pattern_name}' failed: {result.error}")

    return result.output


# === Pattern Tool Factories ===
# These create tool functions for specific patterns

async def extract_wisdom(text: str) -> str:
    """
    Extract key insights and wisdom from text.

    Identifies surprising, actionable insights that change how you think.
    Returns structured markdown with ideas, insights, quotes, and takeaways.
    """
    return await apply_pattern("extract_wisdom", text)


async def extract_ideas(text: str) -> str:
    """
    Extract distinct ideas and concepts from text.

    Separates compound ideas into atomic components and organizes by theme.
    Returns a clean, categorized list of ideas.
    """
    return await apply_pattern("extract_ideas", text)


async def extract_questions(text: str) -> str:
    """
    Generate thoughtful questions from text for further exploration.

    Creates foundational, exploratory, critical, and research questions.
    """
    return await apply_pattern("extract_questions", text)


async def analyze_claims(text: str) -> str:
    """
    Evaluate claims for truthfulness, logic, and evidential support.

    Rates each claim's strength and identifies logical fallacies.
    Returns structured analysis with overall credibility rating.
    """
    return await apply_pattern("analyze_claims", text)


async def analyze_paper(text: str) -> str:
    """
    Evaluate research paper for methodology, rigor, and findings.

    Assesses study design, statistical analysis, and transparency.
    Returns quality ratings and overall grade.
    """
    return await apply_pattern("analyze_paper", text)


async def explain_code(code: str) -> str:
    """
    Explain what code does in plain language.

    Covers purpose, mechanism, data flow, and key implementation details.
    """
    return await apply_pattern("explain_code", code)


async def create_summary(text: str) -> str:
    """
    Create a concise, well-structured summary of text.

    Returns one-sentence summary, main points, and key takeaways.
    """
    return await apply_pattern("create_summary", text)


async def create_synthesis(texts: str) -> str:
    """
    Synthesize multiple pieces of information into a coherent whole.

    Finds common themes, unique contributions, and tensions.
    Returns integrated understanding with open questions.
    """
    return await apply_pattern("create_synthesis", texts)


async def improve_writing(text: str) -> str:
    """
    Improve clarity, flow, and impact of written content.

    Preserves author's voice while enhancing readability.
    Returns improved version with list of changes made.
    """
    return await apply_pattern("improve_writing", text)


async def clean_text(text: str) -> str:
    """
    Clean and normalize messy text (OCR, transcripts, etc.).

    Fixes formatting, line breaks, and obvious errors.
    Preserves meaning while improving readability.
    """
    return await apply_pattern("clean_text", text)


# === NEW EVALUATION PATTERNS ===

async def evaluate_search_results(text: str) -> str:
    """
    Evaluate search results for relevance, freshness, and authority.

    Rates each result against the query intent and provides a quality grade.
    Identifies top results and recommends follow-up searches.
    """
    return await apply_pattern("evaluate_search_results", text)


async def extract_source_credibility(text: str) -> str:
    """
    Assess source credibility, authority, and potential bias.

    Evaluates expertise signals, transparency, and content quality.
    Returns credibility rating with specific evidence.
    """
    return await apply_pattern("extract_source_credibility", text)


async def identify_information_gaps(text: str) -> str:
    """
    Find what's missing from content that should be covered.

    Compares actual coverage against expected comprehensive treatment.
    Prioritizes gaps by importance and suggests how to fill them.
    """
    return await apply_pattern("identify_information_gaps", text)


async def extract_search_refinements(text: str) -> str:
    """
    Generate better search queries from initial results.

    Analyzes what's working and missing in current results.
    Provides refined queries with search operators and alternative strategies.
    """
    return await apply_pattern("extract_search_refinements", text)


# === NEW EXTRACT PATTERNS ===

async def extract_implementation_steps(text: str) -> str:
    """
    Transform documentation into actionable implementation steps.

    Extracts prerequisites, steps, verification points, and gotchas.
    Returns a checklist-ready implementation guide.
    """
    return await apply_pattern("extract_implementation_steps", text)


# === NEW ANALYZE PATTERNS ===

async def analyze_tradeoffs(text: str) -> str:
    """
    Evaluate options by examining costs, benefits, and risks.

    Provides structured comparison with second-order effects.
    Returns conditional recommendations based on priorities.
    """
    return await apply_pattern("analyze_tradeoffs", text)


# === NEW COMPARE PATTERNS ===

async def compare_technologies(text: str) -> str:
    """
    Provide objective comparison of technical options.

    Evaluates performance, DX, ecosystem, and operational factors.
    Returns use-case-specific recommendations.
    """
    return await apply_pattern("compare_technologies", text)


# === NEW CREATE PATTERNS ===

async def create_learning_path(text: str) -> str:
    """
    Design structured learning journey for a technical topic.

    Creates phased curriculum with resources, milestones, and projects.
    Returns actionable learning roadmap with time estimates.
    """
    return await apply_pattern("create_learning_path", text)


async def create_decision_doc(text: str) -> str:
    """
    Create comprehensive decision document.

    Captures context, options considered, reasoning, and implications.
    Returns documented decision with review triggers.
    """
    return await apply_pattern("create_decision_doc", text)


# === Tool Registration ===

# Map of pattern names to tool functions
PATTERN_TOOLS: Dict[str, Callable] = {
    # Original patterns
    "extract_wisdom": extract_wisdom,
    "extract_ideas": extract_ideas,
    "extract_questions": extract_questions,
    "analyze_claims": analyze_claims,
    "analyze_paper": analyze_paper,
    "explain_code": explain_code,
    "create_summary": create_summary,
    "create_synthesis": create_synthesis,
    "improve_writing": improve_writing,
    "clean_text": clean_text,
    # New evaluation patterns
    "evaluate_search_results": evaluate_search_results,
    "extract_source_credibility": extract_source_credibility,
    "identify_information_gaps": identify_information_gaps,
    "extract_search_refinements": extract_search_refinements,
    # New extract patterns
    "extract_implementation_steps": extract_implementation_steps,
    # New analyze patterns
    "analyze_tradeoffs": analyze_tradeoffs,
    # New compare patterns
    "compare_technologies": compare_technologies,
    # New create patterns
    "create_learning_path": create_learning_path,
    "create_decision_doc": create_decision_doc,
}


def get_pattern_tool(pattern_name: str) -> Optional[Callable]:
    """Get the tool function for a pattern."""
    return PATTERN_TOOLS.get(pattern_name)


def get_all_pattern_tools() -> Dict[str, Callable]:
    """Get all available pattern tools."""
    return PATTERN_TOOLS.copy()


def get_pattern_tool_descriptions() -> Dict[str, str]:
    """Get descriptions for all pattern tools (from docstrings)."""
    return {
        name: func.__doc__.strip().split('\n')[0] if func.__doc__ else f"Apply {name} pattern"
        for name, func in PATTERN_TOOLS.items()
    }


def get_pattern_tools_for_skill(patterns: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Get tool definitions for a list of patterns (for SKILL.md format).

    Returns dict with tool info suitable for skill registration.
    """
    tools = {}
    for pattern_name in patterns:
        if pattern_name in PATTERN_TOOLS:
            func = PATTERN_TOOLS[pattern_name]
            tools[pattern_name] = {
                "function": func,
                "description": func.__doc__.strip().split('\n')[0] if func.__doc__ else "",
                "signature": f"{pattern_name}(text: str) -> str",
                "is_pattern": True,
            }
    return tools


# === Skill Integration Helpers ===

def generate_skill_tools_section(patterns: List[str]) -> str:
    """
    Generate the ## Available Tools section content for a SKILL.md file.

    This makes it easy to add pattern tools to existing skills.
    """
    lines = []
    for pattern_name in patterns:
        if pattern_name in PATTERN_TOOLS:
            func = PATTERN_TOOLS[pattern_name]
            desc = func.__doc__.strip().split('\n')[0] if func.__doc__ else f"Apply {pattern_name}"
            lines.append(f"- **{pattern_name}**(text): {desc}")
    return "\n".join(lines)


def generate_system_prompt_section(patterns: List[str]) -> str:
    """
    Generate tool descriptions for a skill's system.md prompt.

    This helps the LLM understand when to use each pattern tool.
    """
    lines = ["## Pattern Tools", ""]
    lines.append("These tools transform text using specialized prompts:")
    lines.append("")

    for pattern_name in patterns:
        if pattern_name in PATTERN_TOOLS:
            func = PATTERN_TOOLS[pattern_name]
            if func.__doc__:
                # Get first paragraph of docstring
                desc = func.__doc__.strip().split('\n\n')[0].replace('\n', ' ')
                lines.append(f"- **{pattern_name}(text)**: {desc}")

    lines.append("")
    lines.append("Use these when you need to analyze, extract, or transform text content.")

    return "\n".join(lines)
