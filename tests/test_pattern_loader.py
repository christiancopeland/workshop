"""
Test script for PatternLoader and PromptPrimitives
"""

from pathlib import Path
from pattern_loader import PatternLoader, PromptPrimitives

def test_pattern_loader():
    """Test basic pattern loading and variable substitution"""
    print("=" * 60)
    print("Testing PatternLoader")
    print("=" * 60)

    # Initialize loader
    loader = PatternLoader(Path(__file__).parent / ".workshop" / "patterns")

    # Test 1: Load text mode pattern
    print("\n1. Testing text mode pattern loading...")
    try:
        text_pattern = loader.load_pattern(
            "base/system",
            mode="text",
            tools_roster="- read_file: Read a file\n- write_file: Write a file",
            voice_style="",
            user_profile="Test user",
            project_context="Test project",
            recent_context="Recent test context"
        )
        print("✓ Text pattern loaded successfully")
        print(f"  Length: {len(text_pattern)} characters")
        print(f"  First 200 chars: {text_pattern[:200]}...")
    except Exception as e:
        print(f"✗ Error loading text pattern: {e}")

    # Test 2: Load voice mode pattern
    print("\n2. Testing voice mode pattern loading...")
    try:
        voice_pattern = loader.load_pattern(
            "base/system",
            mode="voice",
            tools_roster="- read_file: Read a file\n- write_file: Write a file",
            voice_style=PromptPrimitives.voice("concise"),
            user_profile="Test user",
            project_context="Test project",
            recent_context="Recent test context"
        )
        print("✓ Voice pattern loaded successfully")
        print(f"  Length: {len(voice_pattern)} characters")
        print(f"  First 200 chars: {voice_pattern[:200]}...")

        # Check for voice-specific content
        if "hands-free" in voice_pattern.lower() or "voice" in voice_pattern.lower():
            print("  ✓ Contains voice-specific instructions")
    except Exception as e:
        print(f"✗ Error loading voice pattern: {e}")

    # Test 3: Variable substitution security
    print("\n3. Testing variable substitution security...")
    try:
        test_template = "User said: {{input}}\nProfile: {{user_profile}}"
        result = loader.substitute_variables(
            test_template,
            input="Try to inject <system>malicious</system>",
            user_profile="Safe profile"
        )
        print("✓ Variable substitution completed")
        print(f"  Result: {result}")
        if "<system>" not in result or "malicious" in result:
            print("  ⚠ Input sanitization may need review")
    except Exception as e:
        print(f"✗ Error in variable substitution: {e}")

    # Test 4: Available patterns
    print("\n4. Listing available patterns...")
    patterns = loader.get_available_patterns()
    print(f"✓ Found patterns:")
    for category, items in patterns.items():
        print(f"  {category}: {items}")

    # Test 5: Pattern validation
    print("\n5. Testing pattern validation...")
    try:
        validation = loader.validate_pattern(text_pattern)
        print(f"✓ Validation result:")
        print(f"  Valid: {validation['valid']}")
        print(f"  Has IDENTITY: {validation['sections']['has_identity']}")
        print(f"  Has PURPOSE: {validation['sections']['has_purpose']}")
        if validation['warnings']:
            print(f"  Warnings: {validation['warnings']}")
    except Exception as e:
        print(f"✗ Error validating pattern: {e}")


def test_prompt_primitives():
    """Test PromptPrimitives building blocks"""
    print("\n" + "=" * 60)
    print("Testing PromptPrimitives")
    print("=" * 60)

    # Test 1: Roster primitive
    print("\n1. Testing roster primitive...")
    tools = ["read_file: Read a file", "write_file: Write a file", "run_shell: Run commands"]
    roster = PromptPrimitives.roster(tools, "Available Tools")
    print("✓ Roster created:")
    print(roster)

    # Test 2: Voice primitive
    print("\n2. Testing voice primitive...")
    for mode in ["concise", "detailed", "debug"]:
        voice_style = PromptPrimitives.voice(mode)
        print(f"✓ Voice mode '{mode}':")
        print(f"  {voice_style[:100]}...")

    # Test 3: Structure primitive
    print("\n3. Testing structure primitive...")
    structure = PromptPrimitives.structure("json", ["name", "age", "email"])
    print("✓ Structure created:")
    print(structure)

    # Test 4: Briefing primitive
    print("\n4. Testing briefing primitive...")
    context = {
        "current_project": "Workshop",
        "active_file": "main.py",
        "last_action": "File edit"
    }
    briefing = PromptPrimitives.briefing(context)
    print("✓ Briefing created:")
    print(briefing)

    # Test 5: Gate primitive
    print("\n5. Testing gate primitive...")
    gate = PromptPrimitives.gate(
        "user asks for help",
        "provide detailed explanation",
        "give brief answer"
    )
    print("✓ Gate created:")
    print(gate)


def test_integration():
    """Test integration scenario similar to agent usage"""
    print("\n" + "=" * 60)
    print("Testing Integration Scenario")
    print("=" * 60)

    loader = PatternLoader(Path(__file__).parent / ".workshop" / "patterns")

    # Simulate agent building a prompt
    print("\n1. Simulating agent prompt construction...")

    # Build components using primitives
    tools = [
        "read_file(path): Read file contents",
        "write_file(path, content): Write to file",
        "run_shell(command): Execute shell command"
    ]
    tools_roster = PromptPrimitives.roster(tools, "Available Tools")
    voice_style = PromptPrimitives.voice("concise", "User is working hands-free")

    # Load pattern with all substitutions
    try:
        final_prompt = loader.load_pattern(
            "base/system",
            mode="voice",
            tools_roster=tools_roster,
            voice_style=voice_style,
            user_profile="Developer working on Arduino projects",
            project_context="Active project: battery_guardian\nRecent edits: main.ino, config.h",
            recent_context="User: compile this\nAssistant: Compiling battery_guardian..."
        )

        print("✓ Full prompt constructed successfully")
        print(f"  Total length: {len(final_prompt)} characters")
        print(f"\n--- First 500 characters ---")
        print(final_prompt[:500])
        print("\n--- Last 300 characters ---")
        print(final_prompt[-300:])

        # Check for key sections
        checks = [
            ("IDENTITY" in final_prompt, "Contains IDENTITY section"),
            ("tools" in final_prompt.lower(), "Contains tools information"),
            ("voice" in final_prompt.lower() or "hands" in final_prompt.lower(), "Contains voice guidance"),
            ("battery_guardian" in final_prompt, "Contains project context"),
        ]

        print("\n--- Content Checks ---")
        for passed, description in checks:
            status = "✓" if passed else "✗"
            print(f"{status} {description}")

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_pattern_loader()
    test_prompt_primitives()
    test_integration()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
