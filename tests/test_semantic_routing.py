#!/usr/bin/env python3
"""
Test script for Phase 4 Semantic Routing integration.

This script tests:
1. SemanticRouter initialization and embedding generation
2. Semantic similarity matching
3. LLM confirmation for ambiguous cases
4. Integration with existing SkillRegistry patterns
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from semantic_router import SemanticRouter, SemanticMatch


async def test_semantic_router():
    """Test the SemanticRouter in isolation."""
    print("=" * 60)
    print("Testing SemanticRouter")
    print("=" * 60)

    config = Config()
    router = SemanticRouter(
        skills_dir=Path(__file__).parent / ".workshop" / "Skills",
        embeddings_cache_path=config.SEMANTIC_EMBEDDINGS_CACHE,
        embedding_model=config.SEMANTIC_EMBEDDING_MODEL,
        ollama_url=config.OLLAMA_URL,
        llm_model=config.ROUTER_MODEL,
    )

    print(f"\nInitializing router...")
    print(f"  Skills dir: {router.skills_dir}")
    print(f"  Cache path: {router.embeddings_cache_path}")
    print(f"  Embedding model: {router.embedding_model}")

    # Force rebuild to test embedding generation
    await router.initialize(force_rebuild=True)

    print(f"\nLoaded {len(router.skill_embeddings)} skills:")
    for name, emb in router.skill_embeddings.items():
        print(f"  - {name}: {len(emb.utterances)} utterances, {len(emb.tools)} tools")

    # Test queries
    test_queries = [
        # Research
        ("search the web for Ollama documentation", "Research"),
        ("do some research on PAI architecture", "Research"),
        ("look up Python asyncio", "Research"),

        # Memory
        ("remember that the API key is abc123", "Memory"),
        ("what do you remember about the project", "Memory"),
        ("take a note about this bug", "Memory"),

        # FileOperations
        ("read the config.py file", "FileOperations"),
        ("show me what's in the data folder", "FileOperations"),
        ("search for 'def main' in the project", "FileOperations"),

        # Arduino
        ("compile the sketch", "Arduino"),
        ("upload to the board", "Arduino"),
        ("what boards are connected", "Arduino"),

        # ContextIntelligence
        ("what files are related to main.py", "ContextIntelligence"),
        ("what did I change recently", "ContextIntelligence"),
        ("where is Agent defined", "ContextIntelligence"),

        # Ambiguous (should be lower confidence)
        ("hello", None),
        ("what's up", None),
        ("do the thing", None),
    ]

    print("\n" + "=" * 60)
    print("Testing Query Routing")
    print("=" * 60)

    correct = 0
    total = 0

    for query, expected_skill in test_queries:
        match = await router.route(query)

        # Check if match is correct
        if expected_skill is None:
            is_correct = match.confidence < 0.45 or match.skill_name is None
        else:
            is_correct = match.skill_name == expected_skill

        status = "[PASS]" if is_correct else "[FAIL]"
        if is_correct:
            correct += 1
        total += 1

        print(f"\n{status} Query: \"{query}\"")
        print(f"  Expected: {expected_skill or 'low confidence'}")
        print(f"  Got: {match.skill_name} (confidence={match.confidence:.2f}, method={match.method})")
        if match.matched_utterance:
            print(f"  Matched: \"{match.matched_utterance[:50]}...\"")

    print("\n" + "=" * 60)
    print(f"Results: {correct}/{total} tests passed ({100*correct/total:.0f}%)")
    print("=" * 60)

    return correct == total


async def test_skill_registry_integration():
    """Test that semantic routing integrates with SkillRegistry."""
    print("\n" + "=" * 60)
    print("Testing SkillRegistry Integration")
    print("=" * 60)

    try:
        from skill_registry import SkillRegistry

        registry = SkillRegistry(
            skills_dir=Path(__file__).parent / ".workshop" / "Skills"
        )

        print(f"\nLoaded {len(registry.skills)} skills from SkillRegistry:")
        for name in registry.skills:
            print(f"  - {name}")

        # Test that skill patterns still work
        test_queries = [
            "search the web for Python tutorials",
            "remember the password is secret123",
            "compile",
        ]

        print("\nTesting pattern-based routing:")
        for query in test_queries:
            match = registry.route_by_intent(query, {})
            if match.matched:
                print(f"  \"{query}\" -> {match.skill_name} (confidence={match.confidence:.2f})")
            else:
                print(f"  \"{query}\" -> No pattern match")

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_context_compression():
    """Test the ContextCompressor for agent handoffs."""
    print("\n" + "=" * 60)
    print("Testing Context Compression")
    print("=" * 60)

    from subagent_manager import ContextCompressor

    compressor = ContextCompressor()

    # Sample conversation with tool outputs
    messages = [
        {"role": "user", "content": "Search for Ollama documentation"},
        {"role": "assistant", "content": "I'll search the web for Ollama documentation."},
        {"role": "tool", "tool_name": "web_search", "content": "Result 1: Ollama is a local LLM runner..." * 50},
        {"role": "assistant", "content": "I found several resources about Ollama."},
        {"role": "user", "content": "Tell me more about installation"},
        {"role": "tool", "tool_name": "web_search", "content": "Installation steps: 1. Download..." * 30},
        {"role": "assistant", "content": "Here are the installation steps..."},
        {"role": "user", "content": "Now compile my Arduino sketch"},
        {"role": "tool", "tool_name": "arduino_compile", "content": "Compiling... Success!"},
        {"role": "assistant", "content": "Compilation successful!"},
    ]

    print(f"\nOriginal: {len(messages)} messages")

    # Test observation masking
    compressed = compressor.observation_masking(messages, keep_recent=5)
    print(f"After masking: {len(compressed)} messages")

    # Check that tool outputs were compressed
    compressed_tool_content = sum(len(m.get('content', '')) for m in compressed if m.get('role') == 'tool')
    original_tool_content = sum(len(m.get('content', '')) for m in messages if m.get('role') == 'tool')

    print(f"Tool content: {original_tool_content} -> {compressed_tool_content} chars")

    return compressed_tool_content < original_tool_content


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Phase 4 Semantic Routing Integration Tests")
    print("=" * 60)

    results = []

    # Test 1: SemanticRouter
    try:
        results.append(("SemanticRouter", await test_semantic_router()))
    except Exception as e:
        print(f"SemanticRouter test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("SemanticRouter", False))

    # Test 2: SkillRegistry integration
    try:
        results.append(("SkillRegistry Integration", await test_skill_registry_integration()))
    except Exception as e:
        print(f"SkillRegistry test failed: {e}")
        results.append(("SkillRegistry Integration", False))

    # Test 3: Context compression
    try:
        results.append(("Context Compression", await test_context_compression()))
    except Exception as e:
        print(f"Context compression test failed: {e}")
        results.append(("Context Compression", False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed. Check output above for details.")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
