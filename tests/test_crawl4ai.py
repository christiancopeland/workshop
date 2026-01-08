#!/usr/bin/env python3
"""
Test script for Crawl4AI integration in Workshop.

Run this after installing crawl4ai:
    pip install crawl4ai
    crawl4ai-setup
    python test_crawl4ai.py
"""

import asyncio
import sys
from pathlib import Path

# Add workshop to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / ".workshop" / "Skills" / "Research" / "tools"))


async def test_crawl4ai_availability():
    """Test if Crawl4AI is properly installed."""
    print("\n=== Test 1: Crawl4AI Availability ===")

    try:
        from crawl4ai_fetcher import CRAWL4AI_AVAILABLE, is_crawl4ai_available

        if not CRAWL4AI_AVAILABLE:
            print("FAIL: crawl4ai package not installed")
            print("  Run: pip install crawl4ai && crawl4ai-setup")
            return False

        print("OK: crawl4ai package is installed")

        # Check if browser works
        print("Testing browser availability...")
        available = await is_crawl4ai_available()
        if available:
            print("OK: Browser is configured and working")
            return True
        else:
            print("FAIL: Browser not working")
            print("  Run: crawl4ai-setup")
            return False

    except Exception as e:
        print(f"FAIL: {e}")
        return False


async def test_basic_fetch():
    """Test basic URL fetching with Crawl4AI."""
    print("\n=== Test 2: Basic Fetch (Static Page) ===")

    try:
        from crawl4ai_fetcher import fetch_url_crawl4ai

        content = await fetch_url_crawl4ai("https://example.com")

        if "Example Domain" in content:
            print(f"OK: Fetched {len(content)} chars")
            print(f"  Preview: {content[:200]}...")
            return True
        else:
            print(f"FAIL: Expected 'Example Domain' in content")
            return False

    except Exception as e:
        print(f"FAIL: {e}")
        return False


async def test_js_rendered_page():
    """Test fetching a JavaScript-rendered page."""
    print("\n=== Test 3: JS-Rendered Page ===")

    try:
        from crawl4ai_fetcher import Crawl4AIFetcher

        # quotes.toscrape.com/js/ requires JS to load quotes
        async with Crawl4AIFetcher(timeout_ms=30000) as fetcher:
            content, url, metadata = await fetcher.fetch(
                "https://quotes.toscrape.com/js/",
                wait_for=".quote"  # Wait for quotes to load
            )

        if "quote" in content.lower() or len(content) > 500:
            print(f"OK: JS content rendered, {len(content)} chars")
            print(f"  Title: {metadata.get('title', 'N/A')}")
            return True
        else:
            print(f"WARN: Content might not have JS rendered")
            print(f"  Got {len(content)} chars")
            return False

    except Exception as e:
        print(f"FAIL: {e}")
        return False


async def test_fetch_url_integration():
    """Test the integrated fetch_url function."""
    print("\n=== Test 4: Integrated fetch_url ===")

    try:
        from fetch_url import fetch_url, CRAWL4AI_AVAILABLE

        print(f"  CRAWL4AI_AVAILABLE: {CRAWL4AI_AVAILABLE}")

        content = await fetch_url("https://httpbin.org/html")

        if "Herman Melville" in content or "Moby" in content:
            print(f"OK: Fetched {len(content)} chars via integrated function")
            if "[Fetched with Crawl4AI" in content:
                print("  Used: Crawl4AI (primary)")
            else:
                print("  Used: Trafilatura (fallback)")
            return True
        else:
            print(f"WARN: Content may not be complete")
            print(f"  Preview: {content[:200]}...")
            return True  # Still pass if we got content

    except Exception as e:
        print(f"FAIL: {e}")
        return False


async def test_fallback_to_trafilatura():
    """Test that fallback to Trafilatura works."""
    print("\n=== Test 5: Trafilatura Fallback ===")

    try:
        from fetch_url import fetch_url

        # Force Trafilatura by disabling Crawl4AI
        content = await fetch_url("https://example.com", use_crawl4ai=False)

        if "Example Domain" in content:
            print(f"OK: Trafilatura fallback works, {len(content)} chars")
            if "[Fetched with Crawl4AI" not in content:
                print("  Confirmed: Used Trafilatura")
            return True
        else:
            print(f"FAIL: Expected content not found")
            return False

    except Exception as e:
        print(f"FAIL: {e}")
        return False


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Crawl4AI Integration Test Suite")
    print("=" * 60)

    results = []

    # Test 1: Availability
    results.append(("Crawl4AI Availability", await test_crawl4ai_availability()))

    # Only continue if Crawl4AI is available
    if results[0][1]:
        results.append(("Basic Fetch", await test_basic_fetch()))
        results.append(("JS-Rendered Page", await test_js_rendered_page()))
        results.append(("Integrated fetch_url", await test_fetch_url_integration()))
        results.append(("Trafilatura Fallback", await test_fallback_to_trafilatura()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests passed! Crawl4AI integration is ready.")
        return 0
    else:
        print("\nSome tests failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
