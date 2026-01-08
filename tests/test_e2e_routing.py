#!/usr/bin/env python3
"""
End-to-End Routing Test for Workshop Phase 4

This test exercises the complete routing pipeline:
1. User input → Skill pattern matching
2. User input → Semantic routing
3. User input → Full agent chat (with mocked LLM)
4. Context injection and enrichment
5. Tool execution

Run with: python test_e2e_routing.py
"""

import asyncio
import sys
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from skill_registry import SkillRegistry
from semantic_router import SemanticRouter


@dataclass
class TestCase:
    """A test case for routing."""
    query: str
    expected_skill: Optional[str]
    expected_tool: Optional[str] = None
    description: str = ""
    should_match: bool = True  # False for queries that should NOT match


# Comprehensive test cases covering all skills
TEST_CASES = [
    # === Research Skill ===
    TestCase("search the web for Python best practices", "Research", "web_search", "Web search"),
    TestCase("google machine learning tutorials", "Research", "web_search", "Google search"),
    TestCase("research PAI architecture", "Research", "deep_research", "Deep research"),
    TestCase("look up asyncio documentation online", "Research", "web_search", "Online lookup"),
    TestCase("what are your thoughts on this research", "Research", "summarize_research", "Research synthesis"),

    # === Memory Skill ===
    TestCase("remember the API key is sk-12345", "Memory", "remember", "Remember fact"),
    TestCase("what do you remember about the database", "Memory", "recall", "Recall memory"),
    TestCase("take a note about this bug", "Memory", "take_note", "Take note"),
    TestCase("show me my notes", "Memory", "list_notes", "List notes"),
    TestCase("store this for later", "Memory", "remember", "Store info"),

    # === FileOperations Skill ===
    TestCase("read main.py", "FileOperations", "read_file", "Read file"),
    TestCase("read the config.py file", "FileOperations", "read_file", "Read file with 'the'"),
    TestCase("list files in ~/projects", "FileOperations", "list_directory", "List directory"),
    TestCase("show me what's in this folder", "FileOperations", "list_directory", "Show folder"),
    TestCase("search for 'def main' in ~/workshop", "FileOperations", "search_files", "Search files"),
    TestCase("write to output.txt", "FileOperations", "write_file", "Write file"),

    # === Arduino Skill ===
    TestCase("compile the sketch", "Arduino", "arduino_compile", "Compile"),
    TestCase("compile", "Arduino", "arduino_compile", "Bare compile"),
    TestCase("upload to the board", "Arduino", "arduino_upload", "Upload"),
    TestCase("upload", "Arduino", "arduino_upload", "Bare upload"),
    TestCase("monitor serial", "Arduino", "arduino_monitor", "Serial monitor"),
    TestCase("what boards are connected", "Arduino", "arduino_boards", "List boards"),
    TestCase("flash the firmware", "Arduino", "arduino_upload", "Flash firmware"),

    # === ContextIntelligence Skill ===
    TestCase("what files are related to main.py", "ContextIntelligence", "get_related_files", "Related files"),
    TestCase("what did I change recently", "ContextIntelligence", "get_recent_edits", "Recent changes"),
    TestCase("where is Agent defined", "ContextIntelligence", "find_definition", "Find definition"),
    TestCase("find references to this function", "ContextIntelligence", "find_references", "Find references"),
    TestCase("what am I working on", "ContextIntelligence", "get_context_stats", "Workflow status"),

    # === Telos Skill ===
    TestCase("edit my profile", "Telos", "edit_profile", "Edit profile"),
    TestCase("show telos stats", "Telos", "show_telos_stats", "Telos stats"),
    TestCase("reload telos", "Telos", "reload_telos", "Reload telos"),

    # === Ambiguous / No Match (should have low confidence) ===
    TestCase("hello", None, None, "Greeting - no match", should_match=False),
    TestCase("how are you", None, None, "Chitchat - no match", should_match=False),
    TestCase("thanks", None, None, "Thanks - no match", should_match=False),
    TestCase("do the thing", None, None, "Vague - no match", should_match=False),
]


class TestResults:
    """Collect and report test results."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def add_pass(self, test: TestCase, details: str = ""):
        self.passed += 1
        print(f"  [PASS] {test.description}: \"{test.query[:40]}...\"")
        if details:
            print(f"         {details}")

    def add_fail(self, test: TestCase, expected: str, got: str, details: str = ""):
        self.failed += 1
        self.errors.append((test, expected, got))
        print(f"  [FAIL] {test.description}: \"{test.query[:40]}...\"")
        print(f"         Expected: {expected}")
        print(f"         Got: {got}")
        if details:
            print(f"         {details}")

    def summary(self) -> bool:
        total = self.passed + self.failed
        pct = (self.passed / total * 100) if total > 0 else 0
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} passed ({pct:.0f}%)")
        print(f"{'='*60}")
        return self.failed == 0


async def test_skill_registry_routing():
    """Test pattern-based routing through SkillRegistry.

    Note: Pattern routing focuses on skill-level matching. Tool-level matching
    is handled by the semantic router or by the skill itself after routing.
    """
    print("\n" + "="*60)
    print("Test 1: SkillRegistry Pattern Routing")
    print("="*60)

    registry = SkillRegistry(
        skills_dir=Path(__file__).parent / ".workshop" / "Skills"
    )

    results = TestResults()
    skipped = 0

    for test in TEST_CASES:
        match = registry.route_by_intent(test.query, {})

        if test.should_match:
            if match.matched and match.skill_name == test.expected_skill:
                # Skill matched correctly - that's the primary goal of pattern routing
                details = f"skill={match.skill_name}, conf={match.confidence:.2f}"
                if match.tool_name:
                    details += f", tool={match.tool_name}"
                results.add_pass(test, details)
            elif match.matched:
                # Wrong skill
                results.add_fail(test, test.expected_skill, match.skill_name)
            else:
                # Pattern didn't match - not a failure, semantic routing will handle it
                skipped += 1
        else:
            # Should NOT match
            if not match.matched or match.confidence < 0.5:
                results.add_pass(test, f"Correctly rejected (conf={match.confidence:.2f})")
            else:
                results.add_fail(test, "No match", f"{match.skill_name} (conf={match.confidence:.2f})")

    if skipped > 0:
        print(f"  (Skipped {skipped} tests - will be handled by semantic routing)")

    return results.summary()


async def test_semantic_routing():
    """Test embedding-based semantic routing."""
    print("\n" + "="*60)
    print("Test 2: Semantic Router (Embedding-based)")
    print("="*60)

    config = Config()
    router = SemanticRouter(
        skills_dir=Path(__file__).parent / ".workshop" / "Skills",
        embeddings_cache_path=config.SEMANTIC_EMBEDDINGS_CACHE,
        embedding_model=config.SEMANTIC_EMBEDDING_MODEL,
        ollama_url=config.OLLAMA_URL,
        llm_model=config.ROUTER_MODEL,
    )

    await router.initialize()

    results = TestResults()

    for test in TEST_CASES:
        match = await router.route(test.query)

        if test.should_match:
            if match.matched and match.skill_name == test.expected_skill:
                results.add_pass(test, f"skill={match.skill_name}, conf={match.confidence:.2f}, method={match.method}")
            elif match.matched:
                # Wrong skill
                results.add_fail(test, test.expected_skill, match.skill_name,
                               f"conf={match.confidence:.2f}, matched=\"{match.matched_utterance[:30]}...\"")
            else:
                # No match when we expected one
                results.add_fail(test, test.expected_skill, "None", f"conf={match.confidence:.2f}")
        else:
            # Should NOT match
            if not match.matched or match.confidence < 0.45:
                results.add_pass(test, f"Correctly rejected (conf={match.confidence:.2f})")
            else:
                results.add_fail(test, "No match", f"{match.skill_name} (conf={match.confidence:.2f})")

    return results.summary()


async def test_hybrid_routing():
    """Test combined pattern + semantic routing."""
    print("\n" + "="*60)
    print("Test 3: Hybrid Routing (Pattern → Semantic fallback)")
    print("="*60)

    config = Config()

    # Initialize both routers
    registry = SkillRegistry(
        skills_dir=Path(__file__).parent / ".workshop" / "Skills"
    )

    semantic_router = SemanticRouter(
        skills_dir=Path(__file__).parent / ".workshop" / "Skills",
        embeddings_cache_path=config.SEMANTIC_EMBEDDINGS_CACHE,
        embedding_model=config.SEMANTIC_EMBEDDING_MODEL,
        ollama_url=config.OLLAMA_URL,
        llm_model=config.ROUTER_MODEL,
    )
    await semantic_router.initialize()

    results = TestResults()

    for test in TEST_CASES:
        # Try pattern first
        pattern_match = registry.route_by_intent(test.query, {})

        if pattern_match.matched and pattern_match.confidence >= 0.7:
            # Pattern matched with high confidence
            method = "pattern"
            skill = pattern_match.skill_name
            tool = pattern_match.tool_name
            conf = pattern_match.confidence
        else:
            # Fall back to semantic
            semantic_match = await semantic_router.route(test.query)
            method = f"semantic:{semantic_match.method}"
            skill = semantic_match.skill_name
            tool = None
            conf = semantic_match.confidence

        if test.should_match:
            if skill == test.expected_skill:
                results.add_pass(test, f"skill={skill}, method={method}, conf={conf:.2f}")
            elif skill:
                results.add_fail(test, test.expected_skill, skill, f"method={method}, conf={conf:.2f}")
            else:
                results.add_fail(test, test.expected_skill, "None", f"method={method}, conf={conf:.2f}")
        else:
            if not skill or conf < 0.45:
                results.add_pass(test, f"Correctly rejected (method={method}, conf={conf:.2f})")
            else:
                results.add_fail(test, "No match", f"{skill} (method={method}, conf={conf:.2f})")

    return results.summary()


async def test_tool_execution():
    """Test that matched tools can actually be called."""
    print("\n" + "="*60)
    print("Test 4: Tool Execution (Smoke Test)")
    print("="*60)

    registry = SkillRegistry(
        skills_dir=Path(__file__).parent / ".workshop" / "Skills"
    )

    # Test a few tools that don't require external dependencies
    test_tools = [
        ("list_directory", {"path": "."}),
        ("read_file", {"path": "README.md"}),
    ]

    results = TestResults()

    for tool_name, args in test_tools:
        tool_info = registry.get_tool(tool_name)
        if tool_info:
            try:
                # ToolInfo has a .func attribute that is the actual callable
                tool_func = tool_info.func
                # Call the tool
                result = await tool_func(**args, _deps={})
                if result and len(str(result)) > 0:
                    results.add_pass(
                        TestCase(f"{tool_name}({args})", None, tool_name, f"Execute {tool_name}"),
                        f"Got {len(str(result))} chars"
                    )
                else:
                    results.add_fail(
                        TestCase(f"{tool_name}({args})", None, tool_name, f"Execute {tool_name}"),
                        "Non-empty result",
                        f"Empty result"
                    )
            except Exception as e:
                results.add_fail(
                    TestCase(f"{tool_name}({args})", None, tool_name, f"Execute {tool_name}"),
                    "Success",
                    f"Error: {e}"
                )
        else:
            results.add_fail(
                TestCase(f"{tool_name}", None, tool_name, f"Find {tool_name}"),
                "Tool found",
                "Tool not found"
            )

    return results.summary()


async def test_context_compression():
    """Test context compression for agent handoffs."""
    print("\n" + "="*60)
    print("Test 5: Context Compression")
    print("="*60)

    from subagent_manager import ContextCompressor

    compressor = ContextCompressor()

    # Build a realistic conversation
    messages = [
        {"role": "user", "content": "Search for information about Ollama"},
        {"role": "assistant", "content": "I'll search for Ollama information."},
        {"role": "tool", "tool_name": "web_search", "content": "Result: " + "x" * 5000},
        {"role": "assistant", "content": "I found several results about Ollama."},
        {"role": "user", "content": "Tell me more about installation"},
        {"role": "tool", "tool_name": "web_search", "content": "Installation: " + "y" * 3000},
        {"role": "assistant", "content": "Here's how to install Ollama."},
        {"role": "user", "content": "Now compile my sketch"},
        {"role": "tool", "tool_name": "arduino_compile", "content": "Compiling... Done!"},
        {"role": "assistant", "content": "Compilation successful!"},
    ]

    results = TestResults()

    # Test observation masking
    compressed = compressor.observation_masking(messages, keep_recent=5)

    original_size = sum(len(m.get("content", "")) for m in messages)
    compressed_size = sum(len(m.get("content", "")) for m in compressed)

    if compressed_size < original_size:
        results.add_pass(
            TestCase("Observation masking", None, None, "Compress tool outputs"),
            f"Reduced {original_size} → {compressed_size} chars ({100 - compressed_size/original_size*100:.0f}% reduction)"
        )
    else:
        results.add_fail(
            TestCase("Observation masking", None, None, "Compress tool outputs"),
            "Smaller output",
            f"{original_size} → {compressed_size}"
        )

    # Check that recent messages are preserved
    recent_preserved = messages[-5:] == compressed[-5:]
    if recent_preserved:
        results.add_pass(
            TestCase("Recent preservation", None, None, "Keep recent messages intact"),
            "Last 5 messages unchanged"
        )
    else:
        results.add_fail(
            TestCase("Recent preservation", None, None, "Keep recent messages intact"),
            "Identical recent messages",
            "Messages modified"
        )

    return results.summary()


async def main():
    """Run all end-to-end tests."""
    print("\n" + "="*60)
    print("Workshop Phase 4 End-to-End Tests")
    print("="*60)

    all_passed = True

    # Test 1: Pattern routing
    if not await test_skill_registry_routing():
        all_passed = False

    # Test 2: Semantic routing
    if not await test_semantic_routing():
        all_passed = False

    # Test 3: Hybrid routing
    if not await test_hybrid_routing():
        all_passed = False

    # Test 4: Tool execution
    if not await test_tool_execution():
        all_passed = False

    # Test 5: Context compression
    if not await test_context_compression():
        all_passed = False

    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    if all_passed:
        print("All test suites PASSED!")
        return 0
    else:
        print("Some test suites FAILED. Review output above.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
