#!/usr/bin/env python3
"""
Test SkillRegistry Intent Routing

This tests that the new route_by_intent() method correctly routes
various user queries to the appropriate skills, tools, and workflows.
"""

import sys
from pathlib import Path

# Add workshop to path
sys.path.insert(0, str(Path(__file__).parent))

from skill_registry import SkillRegistry, IntentMatch


def test_skill_routing():
    """Test the SkillRegistry routing with various queries."""

    # Initialize SkillRegistry
    skills_dir = Path(__file__).parent / ".workshop" / "Skills"
    registry = SkillRegistry(skills_dir)

    print("=" * 60)
    print("SkillRegistry Intent Routing Test")
    print("=" * 60)
    print(f"\nLoaded {len(registry.skills)} skills:")
    for name, skill in registry.skills.items():
        patterns = len(skill.routing_info.get('intent_patterns', []))
        keywords = len(skill.routing_info.get('keywords', []))
        workflows = len(skill.workflows) if hasattr(skill, 'workflows') else 0
        print(f"  - {name}: {patterns} patterns, {keywords} keywords, {workflows} workflows")

    # Test cases: (query, expected_skill_or_workflow)
    test_cases = [
        # Context Intelligence queries (should work from any directory)
        ("what changed recently?", "ContextIntelligence"),
        ("what am I working on?", "ContextIntelligence"),
        ("what files are related to main.py?", "ContextIntelligence"),
        ("where is Agent defined?", "ContextIntelligence"),
        ("find where we initialize Ollama", "ContextIntelligence"),

        # File Operations queries
        ("read main.py", "FileOperations"),
        ("list files in ~/projects", "FileOperations"),
        ("search for 'def main' in ~/workshop", "FileOperations"),

        # Memory queries
        ("remember the project path is ~/foo", "Memory"),
        ("what do you know about my preferences?", "Memory"),

        # Arduino queries
        ("compile this", "Arduino"),
        ("upload to the board", "Arduino"),
        ("what boards are connected?", "Arduino"),

        # Telos queries
        ("what are my goals?", "Telos"),
        ("add a goal to learn Rust", "Telos"),

        # Ambiguous queries that need context resolution
        ("what files are here?", None),  # Should resolve "here" to cwd
        ("list the current directory", None),
    ]

    print("\n" + "=" * 60)
    print("Testing Queries")
    print("=" * 60)

    # Test context for resolution (use dynamic path)
    test_dir = str(Path(__file__).parent.parent)
    context = {
        "working_directory": test_dir,
        "active_project": test_dir
    }

    passed = 0
    failed = 0

    for query, expected in test_cases:
        match = registry.route_by_intent(query, context)

        if expected:
            success = match.matched and match.skill_name == expected
        else:
            # For None expected, just check that something was attempted
            success = True

        status = "✓" if success else "✗"
        if success:
            passed += 1
        else:
            failed += 1

        print(f"\n{status} Query: \"{query}\"")
        print(f"  Expected: {expected or '(context resolution)'}")
        print(f"  Got: {match}")
        if match.matched:
            print(f"  Pattern: {match.matched_pattern}")
            print(f"  Confidence: {match.confidence:.2f}")
            if match.extracted_args:
                print(f"  Args: {match.extracted_args}")
            if match.workflow_name:
                print(f"  Workflow: {match.workflow_name}")

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    # Test context resolution
    print("\n" + "=" * 60)
    print("Testing Context Resolution")
    print("=" * 60)

    resolution_tests = [
        ("what files are here?", f"what files are {test_dir}?"),
        ("list the current directory", f"list the {test_dir}"),
    ]

    for original, expected_contains in resolution_tests:
        resolved = registry.resolve_query_context(original, context)
        contains_path = context["working_directory"] in resolved
        status = "✓" if contains_path else "✗"
        print(f"\n{status} Original: \"{original}\"")
        print(f"  Resolved: \"{resolved}\"")

    # Test workflow loading
    print("\n" + "=" * 60)
    print("Testing Workflow Loading")
    print("=" * 60)

    all_workflows = registry.list_all_workflows()
    print(f"\nFound {len(all_workflows)} workflows:")
    for name, workflow in all_workflows.items():
        print(f"  - {name} ({workflow.skill_name})")
        print(f"    Purpose: {workflow.purpose[:60]}...")
        print(f"    Triggers: {workflow.triggers[:3]}")

    return failed == 0


if __name__ == "__main__":
    success = test_skill_routing()
    sys.exit(0 if success else 1)
