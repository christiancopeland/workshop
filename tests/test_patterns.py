#!/usr/bin/env python3
"""
Test script for the Fabric-style pattern system.

Tests:
1. Pattern registry loading
2. Pattern detection (single and pipeline)
3. Pattern execution
4. Pipeline execution
"""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from pattern_executor import PatternRegistry, PatternExecutor, PatternPipeline


async def test_registry():
    """Test pattern registry loading."""
    print("\n=== Testing Pattern Registry ===")

    registry = PatternRegistry()
    await registry.initialize()

    patterns = registry.list_patterns()
    print(f"Loaded {len(patterns)} patterns:")
    for p in patterns:
        info = registry.get_pattern_info(p)
        print(f"  - {p}: {info.purpose[:50]}...")

    categories = registry.list_categories()
    print(f"\nCategories: {categories}")

    # Test pattern matching
    test_inputs = [
        "extract wisdom from this",
        "summarize this article",
        "analyze the claims here",
        "random unrelated text",
    ]

    print("\nPattern matching tests:")
    for text in test_inputs:
        match = registry.match_pattern(text)
        print(f"  '{text}' → {match or 'No match'}")

    return len(patterns) > 0


async def test_pipeline_syntax():
    """Test pipeline syntax detection."""
    print("\n=== Testing Pipeline Syntax ===")

    test_cases = [
        ("extract_wisdom | create_summary", True),
        ("extract wisdom, then summarize", True),
        ("first extract ideas, then create summary", True),
        ("just a normal request", False),
        ("summarize this", False),
    ]

    all_passed = True
    for text, should_detect in test_cases:
        result = PatternPipeline.parse_pipeline_syntax(text)
        detected = result is not None
        status = "✓" if detected == should_detect else "✗"
        print(f"  {status} '{text}' → {result}")
        if detected != should_detect:
            all_passed = False

    return all_passed


async def test_executor():
    """Test pattern execution (requires Ollama)."""
    print("\n=== Testing Pattern Executor ===")

    # Check if Ollama is available
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    print("  Skipping: Ollama not available")
                    return True
    except:
        print("  Skipping: Ollama not available")
        return True

    executor = PatternExecutor()
    await executor.initialize()

    # Test with a short input
    test_input = """
    The key insight from studying complex systems is that emergence occurs when
    simple rules interact at scale. This is why ant colonies can solve complex
    problems that no individual ant understands. The same principle applies to
    neural networks and markets.
    """

    print(f"  Executing 'create_summary' pattern...")
    result = await executor.execute("create_summary", test_input)

    if result.success:
        print(f"  ✓ Pattern executed successfully in {result.duration_ms}ms")
        print(f"  Output preview: {result.output[:200]}...")
        return True
    else:
        print(f"  ✗ Pattern failed: {result.error}")
        return False


async def test_router_integration():
    """Test router pattern detection."""
    print("\n=== Testing Router Integration ===")

    from router import PromptRouter

    router = PromptRouter(enable_patterns=True)
    await router.initialize()

    test_cases = [
        ("extract wisdom from this article", "pattern"),
        ("summarize this", "pattern"),
        ("research quantum computing", None),  # Should NOT match pattern
        ("hello how are you", None),  # Should NOT match pattern
    ]

    all_passed = True
    for text, expected_type in test_cases:
        result = await router.route(text)

        if expected_type == "pattern":
            is_correct = result.is_pattern
        else:
            is_correct = not result.is_pattern

        status = "✓" if is_correct else "✗"
        print(f"  {status} '{text}' → {result.method} (pattern={result.is_pattern})")

        if not is_correct:
            all_passed = False

    return all_passed


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Fabric-Style Pattern System Tests")
    print("=" * 60)

    results = []

    # Test 1: Registry
    results.append(("Registry Loading", await test_registry()))

    # Test 2: Pipeline syntax
    results.append(("Pipeline Syntax", await test_pipeline_syntax()))

    # Test 3: Executor (optional - needs Ollama)
    results.append(("Pattern Executor", await test_executor()))

    # Test 4: Router integration
    results.append(("Router Integration", await test_router_integration()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
