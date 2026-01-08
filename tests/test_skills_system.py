"""
Comprehensive unit tests for Skills system

Run with: python test_skills_system.py
"""

import asyncio
from pathlib import Path
from skill_registry import SkillRegistry, Skill, ToolInfo

def test_skill_registry_initialization():
    """Test SkillRegistry loads all skills"""
    print("=" * 60)
    print("TEST: SkillRegistry Initialization")
    print("=" * 60)

    skills_dir = Path(__file__).parent / ".workshop" / "Skills"
    registry = SkillRegistry(skills_dir)

    # Check skills loaded
    expected_skills = ["FileOperations", "Memory", "ContextIntelligence", "Research", "VisualConstructs"]
    loaded_skills = list(registry.skills.keys())

    print(f"\nExpected skills: {expected_skills}")
    print(f"Loaded skills:   {loaded_skills}")

    for skill_name in expected_skills:
        if skill_name in loaded_skills:
            print(f"✓ {skill_name} loaded")
        else:
            print(f"✗ {skill_name} NOT loaded")

    print(f"\nTotal tools loaded: {len(registry.list_all_tools())}")

    return len(loaded_skills) >= 4  # At least 4/5 skills


def test_skill_routing():
    """Test skill routing with intent patterns"""
    print("\n" + "=" * 60)
    print("TEST: Skill Routing")
    print("=" * 60)

    skills_dir = Path(__file__).parent / ".workshop" / "Skills"
    registry = SkillRegistry(skills_dir)

    test_cases = [
        ("read main.py", "FileOperations"),
        ("remember the API key is xyz", "Memory"),
        ("what files are related to agent.py", "ContextIntelligence"),
        ("search the web for Python tutorials", "Research"),
        ("show me config.json", "VisualConstructs"),
    ]

    results = []
    for user_input, expected_skill in test_cases:
        skill = registry.route_request(user_input)
        actual_skill = skill.name if skill else None

        match = actual_skill == expected_skill
        status = "✓" if match else "✗"

        print(f"\n{status} Input: '{user_input}'")
        print(f"  Expected: {expected_skill}")
        print(f"  Got:      {actual_skill}")

        results.append(match)

    return all(results)


def test_tool_loading():
    """Test that all expected tools are loaded"""
    print("\n" + "=" * 60)
    print("TEST: Tool Loading")
    print("=" * 60)

    skills_dir = Path(__file__).parent / ".workshop" / "Skills"
    registry = SkillRegistry(skills_dir)

    # Expected tools by skill
    expected_tools = {
        "FileOperations": ["read_file", "write_file", "list_directory", "search_files"],
        "Memory": ["remember", "recall", "take_note", "list_notes"],
        "ContextIntelligence": [
            "get_file_content", "search_project_files", "find_definition",
            "find_references", "get_related_files", "get_recent_edits",
            "search_web_docs", "get_context_stats"
        ],
        "Research": ["web_search"],
        "VisualConstructs": [
            "show_file", "show_directory", "create_note",
            "close_construct", "close_all_constructs", "list_constructs"
        ]
    }

    all_good = True

    for skill_name, tools in expected_tools.items():
        skill = registry.get_skill(skill_name)

        if not skill:
            print(f"\n✗ Skill {skill_name} not found")
            all_good = False
            continue

        print(f"\n{skill_name}:")
        for tool_name in tools:
            if tool_name in skill.tools:
                print(f"  ✓ {tool_name}")
            else:
                print(f"  ✗ {tool_name} NOT loaded")
                all_good = False

    return all_good


def test_argument_normalization():
    """Test argument alias normalization"""
    print("\n" + "=" * 60)
    print("TEST: Argument Normalization")
    print("=" * 60)

    skills_dir = Path(__file__).parent / ".workshop" / "Skills"
    registry = SkillRegistry(skills_dir)

    test_cases = [
        ({"file": "test.txt"}, {"path": "test.txt"}),
        ({"directory": "~/projects"}, {"path": "~/projects"}),
        ({"filepath": "config.json"}, {"path": "config.json"}),
        ({"text": "hello"}, {"content": "hello"}),
        ({"message": "hello"}, {"content": "hello"}),
        ({"search": "query"}, {"query": "query"}),
    ]

    results = []
    for input_args, expected_output in test_cases:
        output = registry._normalize_args(input_args)
        match = output == expected_output

        status = "✓" if match else "✗"
        print(f"{status} {input_args} → {output}")

        results.append(match)

    return all(results)


async def test_tool_execution():
    """Test tool execution with dependencies"""
    print("\n" + "=" * 60)
    print("TEST: Tool Execution")
    print("=" * 60)

    # Create mock dependencies
    class MockConfig:
        def is_path_allowed(self, path):
            return True

    skills_dir = Path(__file__).parent / ".workshop" / "Skills"
    registry = SkillRegistry(
        skills_dir,
        dependencies={"config": MockConfig()}
    )

    # Test get_current_time (no dependencies needed)
    print("\nTesting get_current_time tool:")
    try:
        # This tool might not exist - that's ok
        result = await registry.execute("get_current_time", {})
        print(f"✓ Result: {result}")
        return_val = True
    except ValueError:
        print("✗ get_current_time tool not found (might be in project_tools.py)")
        return_val = False

    # Test tool lookup
    print("\nTesting tool lookup:")
    tool_info = registry.get_tool("read_file")
    if tool_info:
        print(f"✓ Found read_file tool")
        print(f"  Skill: {tool_info.skill_name}")
        print(f"  Description: {tool_info.description[:50]}...")
    else:
        print("✗ read_file tool not found")
        return_val = False

    return return_val


def test_skill_md_parsing():
    """Test SKILL.md parsing"""
    print("\n" + "=" * 60)
    print("TEST: SKILL.md Parsing")
    print("=" * 60)

    skills_dir = Path(__file__).parent / ".workshop" / "Skills"
    file_ops_skill = Skill(skills_dir / "FileOperations")

    print(f"\nSkill: {file_ops_skill.name}")
    print(f"Priority: {file_ops_skill.routing_info.get('priority')}")
    print(f"Intent patterns: {len(file_ops_skill.routing_info.get('intent_patterns', []))}")
    print(f"Keywords: {file_ops_skill.routing_info.get('keywords')}")

    # Check that patterns exist
    patterns = file_ops_skill.routing_info.get('intent_patterns', [])
    keywords = file_ops_skill.routing_info.get('keywords', [])

    has_patterns = len(patterns) > 0
    has_keywords = len(keywords) > 0
    has_priority = file_ops_skill.routing_info.get('priority') in ['HIGH', 'MEDIUM', 'LOW']

    print(f"\n✓ Has intent patterns: {has_patterns}")
    print(f"✓ Has keywords: {has_keywords}")
    print(f"✓ Has priority: {has_priority}")

    return has_patterns and has_keywords and has_priority


def test_get_all_tools_compatibility():
    """Test get_all_tools() backwards compatibility"""
    print("\n" + "=" * 60)
    print("TEST: Backwards Compatibility (get_all_tools)")
    print("=" * 60)

    skills_dir = Path(__file__).parent / ".workshop" / "Skills"
    registry = SkillRegistry(skills_dir)

    # This method needs to work for Agent compatibility
    all_tools = registry.get_all_tools()

    print(f"\nTotal tools: {len(all_tools)}")

    # Check format matches old ToolRegistry
    if all_tools:
        first_tool = list(all_tools.values())[0]
        has_description = "description" in first_tool
        has_signature = "signature" in first_tool

        print(f"✓ Has 'description' field: {has_description}")
        print(f"✓ Has 'signature' field: {has_signature}")

        return has_description and has_signature

    return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print(" WORKSHOP SKILLS SYSTEM - COMPREHENSIVE UNIT TESTS")
    print("=" * 70)

    tests = [
        ("SkillRegistry Initialization", test_skill_registry_initialization),
        ("Skill Routing", test_skill_routing),
        ("Tool Loading", test_tool_loading),
        ("Argument Normalization", test_argument_normalization),
        ("SKILL.md Parsing", test_skill_md_parsing),
        ("Backwards Compatibility", test_get_all_tools_compatibility),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ TEST FAILED: {test_name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Async test
    print("\n" + "=" * 60)
    try:
        result = asyncio.run(test_tool_execution())
        results.append(("Tool Execution", result))
    except Exception as e:
        print(f"\n✗ TEST FAILED: Tool Execution")
        print(f"  Error: {e}")
        results.append(("Tool Execution", False))

    # Summary
    print("\n" + "=" * 70)
    print(" TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")

    print("\n" + "=" * 70)
    print(f" TOTAL: {passed}/{total} tests passed")
    print("=" * 70)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
