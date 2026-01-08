"""
Research Tools - URL validation, metadata extraction, and source quality utilities.

These tools help Workshop evaluate and work with web sources more effectively,
supporting the ValidatedResearch workflow and other research capabilities.

Usage:
    from research_tools import validate_url, extract_page_metadata, check_source_reputation

    # Check if a URL is accessible
    result = await validate_url("https://example.com/article")

    # Get page metadata
    metadata = await extract_page_metadata("https://example.com/article")

    # Check domain reputation
    reputation = check_source_reputation("example.com")
"""

import asyncio
import aiohttp
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlparse
import re

from logger import get_logger

log = get_logger("research_tools")

# Known high-quality domains by category
DOMAIN_REPUTATION: Dict[str, Dict[str, Any]] = {
    # Official documentation
    "docs.python.org": {"tier": "high", "type": "official_docs", "bias": "neutral"},
    "developer.mozilla.org": {"tier": "high", "type": "official_docs", "bias": "neutral"},
    "docs.microsoft.com": {"tier": "high", "type": "official_docs", "bias": "vendor"},
    "learn.microsoft.com": {"tier": "high", "type": "official_docs", "bias": "vendor"},
    "cloud.google.com": {"tier": "high", "type": "official_docs", "bias": "vendor"},
    "docs.aws.amazon.com": {"tier": "high", "type": "official_docs", "bias": "vendor"},
    "kubernetes.io": {"tier": "high", "type": "official_docs", "bias": "neutral"},
    "docs.docker.com": {"tier": "high", "type": "official_docs", "bias": "vendor"},
    "reactjs.org": {"tier": "high", "type": "official_docs", "bias": "neutral"},
    "vuejs.org": {"tier": "high", "type": "official_docs", "bias": "neutral"},
    "rust-lang.org": {"tier": "high", "type": "official_docs", "bias": "neutral"},
    "go.dev": {"tier": "high", "type": "official_docs", "bias": "vendor"},
    "nodejs.org": {"tier": "high", "type": "official_docs", "bias": "neutral"},

    # Academic/Research
    "arxiv.org": {"tier": "high", "type": "academic", "bias": "neutral"},
    "scholar.google.com": {"tier": "high", "type": "academic", "bias": "neutral"},
    "acm.org": {"tier": "high", "type": "academic", "bias": "neutral"},
    "ieee.org": {"tier": "high", "type": "academic", "bias": "neutral"},
    "nature.com": {"tier": "high", "type": "academic", "bias": "neutral"},
    "sciencedirect.com": {"tier": "high", "type": "academic", "bias": "neutral"},

    # Developer platforms
    "github.com": {"tier": "high", "type": "code_platform", "bias": "neutral"},
    "gitlab.com": {"tier": "high", "type": "code_platform", "bias": "neutral"},
    "stackoverflow.com": {"tier": "medium", "type": "community", "bias": "neutral"},
    "stackexchange.com": {"tier": "medium", "type": "community", "bias": "neutral"},

    # Tech news/blogs
    "hacker-news.firebaseio.com": {"tier": "medium", "type": "aggregator", "bias": "tech_optimist"},
    "news.ycombinator.com": {"tier": "medium", "type": "aggregator", "bias": "tech_optimist"},
    "medium.com": {"tier": "low", "type": "blog_platform", "bias": "varies"},
    "dev.to": {"tier": "medium", "type": "blog_platform", "bias": "neutral"},
    "hashnode.com": {"tier": "medium", "type": "blog_platform", "bias": "neutral"},
    "techcrunch.com": {"tier": "medium", "type": "news", "bias": "startup_positive"},
    "arstechnica.com": {"tier": "medium", "type": "news", "bias": "neutral"},
    "wired.com": {"tier": "medium", "type": "news", "bias": "tech_optimist"},
    "theverge.com": {"tier": "medium", "type": "news", "bias": "consumer_tech"},

    # Reference
    "wikipedia.org": {"tier": "medium", "type": "encyclopedia", "bias": "neutral"},
    "en.wikipedia.org": {"tier": "medium", "type": "encyclopedia", "bias": "neutral"},

    # Lower quality / caution
    "w3schools.com": {"tier": "low", "type": "tutorial", "bias": "neutral", "note": "Sometimes outdated"},
    "geeksforgeeks.org": {"tier": "low", "type": "tutorial", "bias": "neutral", "note": "Variable quality"},
    "tutorialspoint.com": {"tier": "low", "type": "tutorial", "bias": "neutral", "note": "Often outdated"},
}

# Content type indicators
CONTENT_INDICATORS = {
    "documentation": ["docs", "documentation", "reference", "api", "guide"],
    "tutorial": ["tutorial", "how-to", "getting-started", "learn", "course"],
    "blog": ["blog", "post", "article", "thoughts", "opinion"],
    "news": ["news", "announce", "release", "update"],
    "discussion": ["discuss", "forum", "community", "question", "answer"],
    "code": ["github.com", "gitlab.com", "gist", "code", "repository"],
}


@dataclass
class URLValidationResult:
    """Result of URL validation."""
    url: str
    is_valid: bool
    status_code: Optional[int] = None
    is_accessible: bool = False
    is_paywalled: bool = False
    redirect_url: Optional[str] = None
    content_type: Optional[str] = None
    error: Optional[str] = None
    response_time_ms: Optional[float] = None


@dataclass
class PageMetadata:
    """Extracted metadata from a web page."""
    url: str
    title: Optional[str] = None
    description: Optional[str] = None
    author: Optional[str] = None
    publish_date: Optional[str] = None
    modified_date: Optional[str] = None
    domain: Optional[str] = None
    content_type: Optional[str] = None
    word_count: Optional[int] = None
    has_paywall: bool = False
    language: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    og_image: Optional[str] = None
    canonical_url: Optional[str] = None


@dataclass
class SourceReputation:
    """Reputation assessment for a domain."""
    domain: str
    tier: str  # high, medium, low, unknown
    source_type: str  # official_docs, academic, blog_platform, etc.
    bias: str  # neutral, vendor, varies, etc.
    notes: Optional[str] = None
    is_known: bool = False


async def validate_url(
    url: str,
    timeout: int = 10,
    follow_redirects: bool = True
) -> URLValidationResult:
    """
    Validate a URL by checking if it's accessible.

    Args:
        url: The URL to validate
        timeout: Request timeout in seconds
        follow_redirects: Whether to follow redirects

    Returns:
        URLValidationResult with accessibility info
    """
    result = URLValidationResult(url=url, is_valid=False)

    # Basic URL validation
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            result.error = "Invalid URL format"
            return result
        result.is_valid = True
    except Exception as e:
        result.error = f"URL parsing error: {str(e)}"
        return result

    # Try to access the URL
    try:
        start_time = datetime.now()

        async with aiohttp.ClientSession() as session:
            async with session.head(
                url,
                timeout=aiohttp.ClientTimeout(total=timeout),
                allow_redirects=follow_redirects,
                headers={"User-Agent": "Workshop-Research-Bot/1.0"}
            ) as response:
                end_time = datetime.now()
                result.response_time_ms = (end_time - start_time).total_seconds() * 1000
                result.status_code = response.status
                result.content_type = response.headers.get("Content-Type", "").split(";")[0]

                # Check for redirects
                if response.history:
                    result.redirect_url = str(response.url)

                # Assess accessibility
                if response.status == 200:
                    result.is_accessible = True
                elif response.status == 403:
                    result.is_accessible = False
                    result.error = "Access forbidden (possibly paywalled)"
                    result.is_paywalled = True
                elif response.status == 404:
                    result.is_accessible = False
                    result.error = "Page not found"
                elif response.status >= 500:
                    result.is_accessible = False
                    result.error = f"Server error ({response.status})"
                else:
                    result.is_accessible = response.status < 400

    except asyncio.TimeoutError:
        result.error = f"Request timed out after {timeout}s"
    except aiohttp.ClientError as e:
        result.error = f"Connection error: {str(e)}"
    except Exception as e:
        result.error = f"Unexpected error: {str(e)}"

    return result


async def extract_page_metadata(
    url: str,
    timeout: int = 15
) -> PageMetadata:
    """
    Extract metadata from a web page.

    Args:
        url: The URL to extract metadata from
        timeout: Request timeout in seconds

    Returns:
        PageMetadata with extracted information
    """
    parsed_url = urlparse(url)
    metadata = PageMetadata(url=url, domain=parsed_url.netloc)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=timeout),
                headers={"User-Agent": "Workshop-Research-Bot/1.0"}
            ) as response:
                if response.status != 200:
                    return metadata

                content_type = response.headers.get("Content-Type", "")
                if "text/html" not in content_type:
                    metadata.content_type = content_type.split(";")[0]
                    return metadata

                html = await response.text()

                # Extract title
                title_match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
                if title_match:
                    metadata.title = title_match.group(1).strip()

                # Extract meta tags
                meta_patterns = {
                    "description": r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
                    "author": r'<meta[^>]+name=["\']author["\'][^>]+content=["\']([^"\']+)["\']',
                    "keywords": r'<meta[^>]+name=["\']keywords["\'][^>]+content=["\']([^"\']+)["\']',
                    "publish_date": r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)["\']',
                    "modified_date": r'<meta[^>]+property=["\']article:modified_time["\'][^>]+content=["\']([^"\']+)["\']',
                    "og_image": r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
                    "canonical": r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)["\']',
                    "language": r'<html[^>]+lang=["\']([^"\']+)["\']',
                }

                for field, pattern in meta_patterns.items():
                    match = re.search(pattern, html, re.IGNORECASE)
                    if match:
                        value = match.group(1).strip()
                        if field == "keywords":
                            metadata.keywords = [k.strip() for k in value.split(",")]
                        elif field == "canonical":
                            metadata.canonical_url = value
                        elif field == "language":
                            metadata.language = value
                        else:
                            setattr(metadata, field, value)

                # Estimate word count (rough)
                # Remove scripts and styles, then count words
                text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r"<[^>]+>", " ", text)
                words = text.split()
                metadata.word_count = len(words)

                # Check for paywall indicators
                paywall_indicators = [
                    "paywall", "subscribe to read", "subscriber-only",
                    "premium content", "unlock this article", "sign in to read"
                ]
                html_lower = html.lower()
                metadata.has_paywall = any(indicator in html_lower for indicator in paywall_indicators)

                # Determine content type
                for content_type, indicators in CONTENT_INDICATORS.items():
                    if any(ind in url.lower() or ind in (metadata.title or "").lower()
                           for ind in indicators):
                        metadata.content_type = content_type
                        break

    except Exception as e:
        log.warning(f"Error extracting metadata from {url}: {e}")

    return metadata


def check_source_reputation(domain: str) -> SourceReputation:
    """
    Check the reputation of a domain.

    Args:
        domain: The domain to check (e.g., "github.com")

    Returns:
        SourceReputation with tier and bias information
    """
    # Clean up domain
    domain = domain.lower().strip()
    if domain.startswith("www."):
        domain = domain[4:]

    # Check direct match
    if domain in DOMAIN_REPUTATION:
        info = DOMAIN_REPUTATION[domain]
        return SourceReputation(
            domain=domain,
            tier=info["tier"],
            source_type=info["type"],
            bias=info.get("bias", "unknown"),
            notes=info.get("note"),
            is_known=True
        )

    # Check parent domain
    parts = domain.split(".")
    if len(parts) > 2:
        parent = ".".join(parts[-2:])
        if parent in DOMAIN_REPUTATION:
            info = DOMAIN_REPUTATION[parent]
            return SourceReputation(
                domain=domain,
                tier=info["tier"],
                source_type=info["type"],
                bias=info.get("bias", "unknown"),
                notes=info.get("note"),
                is_known=True
            )

    # Unknown domain - try to infer
    inferred_type = "unknown"
    if "gov" in domain:
        inferred_type = "government"
    elif "edu" in domain:
        inferred_type = "educational"
    elif "blog" in domain:
        inferred_type = "blog"

    return SourceReputation(
        domain=domain,
        tier="unknown",
        source_type=inferred_type,
        bias="unknown",
        is_known=False
    )


def assess_source_quality(
    metadata: PageMetadata,
    reputation: SourceReputation
) -> Dict[str, Any]:
    """
    Combine metadata and reputation into an overall quality assessment.

    Args:
        metadata: Extracted page metadata
        reputation: Domain reputation

    Returns:
        Dict with quality scores and assessment
    """
    scores = {
        "authority": 0,
        "freshness": 0,
        "depth": 0,
        "accessibility": 0,
    }

    # Authority score based on reputation tier
    tier_scores = {"high": 9, "medium": 6, "low": 3, "unknown": 4}
    scores["authority"] = tier_scores.get(reputation.tier, 4)

    # Adjust for source type
    type_bonuses = {
        "official_docs": 2,
        "academic": 2,
        "code_platform": 1,
        "encyclopedia": 0,
        "community": 0,
        "blog_platform": -1,
        "tutorial": -1,
    }
    scores["authority"] = min(10, scores["authority"] + type_bonuses.get(reputation.source_type, 0))

    # Freshness score based on dates
    if metadata.publish_date:
        try:
            # Try to parse and check age
            # This is simplified - real implementation would parse various date formats
            if "2024" in metadata.publish_date or "2025" in metadata.publish_date:
                scores["freshness"] = 9
            elif "2023" in metadata.publish_date:
                scores["freshness"] = 7
            elif "2022" in metadata.publish_date:
                scores["freshness"] = 5
            else:
                scores["freshness"] = 3
        except:
            scores["freshness"] = 5  # Unknown
    else:
        scores["freshness"] = 5  # Unknown

    # Depth score based on word count
    if metadata.word_count:
        if metadata.word_count > 2000:
            scores["depth"] = 9
        elif metadata.word_count > 1000:
            scores["depth"] = 7
        elif metadata.word_count > 500:
            scores["depth"] = 5
        else:
            scores["depth"] = 3
    else:
        scores["depth"] = 5  # Unknown

    # Accessibility score
    scores["accessibility"] = 2 if metadata.has_paywall else 10

    # Overall score (weighted average)
    weights = {"authority": 0.4, "freshness": 0.2, "depth": 0.25, "accessibility": 0.15}
    overall = sum(scores[k] * weights[k] for k in scores)

    return {
        "scores": scores,
        "overall": round(overall, 1),
        "reputation": reputation,
        "metadata": metadata,
        "warnings": _generate_warnings(metadata, reputation),
    }


def _generate_warnings(metadata: PageMetadata, reputation: SourceReputation) -> List[str]:
    """Generate warnings about potential quality issues."""
    warnings = []

    if metadata.has_paywall:
        warnings.append("Content may be behind a paywall")

    if reputation.bias == "vendor":
        warnings.append(f"Source may have vendor bias ({reputation.domain})")
    elif reputation.bias == "varies":
        warnings.append("Content quality varies on this platform")

    if reputation.tier == "low":
        warnings.append("Source has lower reliability rating")

    if reputation.notes:
        warnings.append(reputation.notes)

    if metadata.word_count and metadata.word_count < 300:
        warnings.append("Very short content - may lack depth")

    return warnings


# === Convenience functions for skill integration ===

async def quick_source_check(url: str) -> Dict[str, Any]:
    """
    Quick assessment of a source URL.

    Returns a dict with essential quality indicators.
    """
    parsed = urlparse(url)
    reputation = check_source_reputation(parsed.netloc)

    # Try to validate and get metadata
    validation = await validate_url(url)

    if validation.is_accessible:
        metadata = await extract_page_metadata(url)
        assessment = assess_source_quality(metadata, reputation)
        return {
            "url": url,
            "accessible": True,
            "quality_score": assessment["overall"],
            "tier": reputation.tier,
            "source_type": reputation.source_type,
            "warnings": assessment["warnings"],
            "title": metadata.title,
            "word_count": metadata.word_count,
        }
    else:
        return {
            "url": url,
            "accessible": False,
            "error": validation.error,
            "tier": reputation.tier,
            "source_type": reputation.source_type,
        }


async def batch_check_sources(urls: List[str]) -> List[Dict[str, Any]]:
    """
    Check multiple sources in parallel.

    Returns list of quick assessments for all URLs.
    """
    tasks = [quick_source_check(url) for url in urls]
    return await asyncio.gather(*tasks)


def format_source_assessment(assessment: Dict[str, Any]) -> str:
    """
    Format a source assessment as readable text.
    """
    lines = []
    lines.append(f"**URL**: {assessment['url']}")

    if assessment.get('accessible'):
        lines.append(f"**Quality Score**: {assessment['quality_score']}/10")
        lines.append(f"**Tier**: {assessment['tier'].title()}")
        lines.append(f"**Type**: {assessment['source_type'].replace('_', ' ').title()}")
        if assessment.get('title'):
            lines.append(f"**Title**: {assessment['title']}")
        if assessment.get('word_count'):
            lines.append(f"**Word Count**: ~{assessment['word_count']}")
        if assessment.get('warnings'):
            lines.append("**Warnings**:")
            for w in assessment['warnings']:
                lines.append(f"  - {w}")
    else:
        lines.append(f"**Accessible**: No")
        lines.append(f"**Error**: {assessment.get('error', 'Unknown')}")

    return "\n".join(lines)
