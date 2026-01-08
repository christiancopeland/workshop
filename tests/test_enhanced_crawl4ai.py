#!/usr/bin/env python3
"""
Test script for enhanced Crawl4AI fetcher capabilities.

Tests:
1. Basic fetch with stealth mode
2. Domain configuration
3. Deep crawling (if available)
4. Parallel fetching
5. LLM extraction (if available)
"""

import asyncio
import sys
sys.path.insert(0, '.workshop/Skills/Research/tools')

from crawl4ai_fetcher import (
    Crawl4AIFetcher,
    CRAWL4AI_AVAILABLE,
    DEEP_CRAWL_AVAILABLE,
    UNDETECTED_AVAILABLE,
    LLM_EXTRACTION_AVAILABLE,
    get_domain_config,
    DOMAIN_CONFIGS,
    fetch_with_stealth,
    deep_crawl_url,
)


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


async def test_feature_detection():
    """Test that features are properly detected."""
    print_section("Feature Detection")

    print(f"  Crawl4AI Available:      {CRAWL4AI_AVAILABLE}")
    print(f"  Deep Crawl Available:    {DEEP_CRAWL_AVAILABLE}")
    print(f"  Undetected Available:    {UNDETECTED_AVAILABLE}")
    print(f"  LLM Extraction Available: {LLM_EXTRACTION_AVAILABLE}")

    return CRAWL4AI_AVAILABLE


async def test_domain_config():
    """Test domain configuration system."""
    print_section("Domain Configuration")

    test_urls = [
        "https://medium.com/article",
        "https://github.com/user/repo",
        "https://linkedin.com/profile",
        "https://example.com/page",
        "https://news.ycombinator.com",
    ]

    for url in test_urls:
        config = get_domain_config(url)
        if config:
            status = "BLOCKED" if config.blocked else "configured"
            print(f"  {url}")
            print(f"    -> {status}")
            if config.wait_for:
                print(f"       wait_for: {config.wait_for}")
            if config.scroll_first:
                print(f"       scroll_first: True")
        else:
            print(f"  {url}")
            print(f"    -> default config")

    print(f"\n  Total configured domains: {len(DOMAIN_CONFIGS)}")


async def test_basic_fetch():
    """Test basic fetch with stealth mode."""
    print_section("Basic Fetch with Stealth Mode")

    if not CRAWL4AI_AVAILABLE:
        print("  SKIPPED: crawl4ai not available")
        return

    url = "https://httpbin.org/headers"

    try:
        async with Crawl4AIFetcher(
            timeout_ms=30000,
            stealth_mode=True,
            verbose=False
        ) as fetcher:
            content, final_url, metadata = await fetcher.fetch(url)

            print(f"  URL: {url}")
            print(f"  Final URL: {final_url}")
            print(f"  Content length: {len(content)} chars")
            print(f"  Title: {metadata.get('title', 'N/A')}")
            print(f"  Status: SUCCESS")

            # Check if we got content
            if len(content) > 100:
                print(f"  Preview: {content[:200]}...")

    except Exception as e:
        print(f"  ERROR: {e}")


async def test_parallel_fetch():
    """Test parallel fetching of multiple URLs."""
    print_section("Parallel Fetch (fetch_many)")

    if not CRAWL4AI_AVAILABLE:
        print("  SKIPPED: crawl4ai not available")
        return

    urls = [
        "https://example.com",
        "https://httpbin.org/html",
        "https://linkedin.com/company/test",  # Should be blocked
    ]

    try:
        async with Crawl4AIFetcher(timeout_ms=30000) as fetcher:
            results = await fetcher.fetch_many(urls, max_concurrent=3)

            for result in results:
                status = "SUCCESS" if result["success"] else "FAILED"
                content_len = len(result.get("content", ""))
                error = result.get("error", "")

                print(f"  {result['url']}")
                print(f"    -> {status} ({content_len} chars)")
                if error:
                    print(f"       Error: {error}")

    except Exception as e:
        print(f"  ERROR: {e}")


async def test_deep_crawl():
    """Test deep crawling functionality."""
    print_section("Deep Crawl")

    if not CRAWL4AI_AVAILABLE:
        print("  SKIPPED: crawl4ai not available")
        return

    if not DEEP_CRAWL_AVAILABLE:
        print("  SKIPPED: deep crawling features not available")
        print("  (This requires a newer version of crawl4ai)")
        return

    url = "https://docs.python.org/3/library/asyncio.html"
    keywords = ["async", "await", "coroutine"]

    try:
        async with Crawl4AIFetcher(timeout_ms=60000) as fetcher:
            results = await fetcher.deep_crawl(
                url=url,
                keywords=keywords,
                max_depth=1,
                max_pages=5,
                strategy="bfs"
            )

            print(f"  Seed URL: {url}")
            print(f"  Keywords: {keywords}")
            print(f"  Pages crawled: {len(results)}")

            for i, result in enumerate(results[:5]):
                depth = result.get("depth", 0)
                score = result.get("score", 0)
                content_len = len(result.get("content", ""))
                print(f"  [{i+1}] depth={depth} score={score:.2f} ({content_len} chars)")
                print(f"      {result['url'][:60]}...")

    except Exception as e:
        print(f"  ERROR: {e}")


async def test_stealth_fetch():
    """Test maximum stealth fetch."""
    print_section("Maximum Stealth Fetch")

    if not CRAWL4AI_AVAILABLE:
        print("  SKIPPED: crawl4ai not available")
        return

    url = "https://httpbin.org/headers"

    try:
        content, final_url, metadata = await fetch_with_stealth(
            url=url,
            timeout=30,
            use_undetected=UNDETECTED_AVAILABLE
        )

        print(f"  URL: {url}")
        print(f"  Undetected mode: {UNDETECTED_AVAILABLE}")
        print(f"  Content length: {len(content)} chars")
        print(f"  Status: SUCCESS")

    except Exception as e:
        print(f"  ERROR: {e}")


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print(" Enhanced Crawl4AI Fetcher Test Suite")
    print("="*60)

    # Run tests
    available = await test_feature_detection()
    await test_domain_config()

    if available:
        await test_basic_fetch()
        await test_parallel_fetch()
        await test_deep_crawl()
        await test_stealth_fetch()
    else:
        print("\n  SKIPPING fetch tests: crawl4ai not available")
        print("  Install with: pip install crawl4ai && crawl4ai-setup")

    print_section("Test Complete")


if __name__ == "__main__":
    asyncio.run(main())
