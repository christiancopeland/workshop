#!/usr/bin/env python3
"""
Test script for subagent system.

Tests:
1. Model unloading/loading
2. Context snapshot creation
3. Research summarizer subagent execution
"""

import asyncio
import sys
from pathlib import Path

# Add workshop to path
sys.path.insert(0, str(Path(__file__).parent))

from subagent_manager import SubagentManager, ContextSnapshot


async def test_model_management():
    """Test model loading and unloading."""
    print("\n=== Testing Model Management ===")

    manager = SubagentManager()

    # Check what's loaded
    loaded = await manager.get_loaded_models()
    print(f"Currently loaded: {[m.get('name', m) for m in loaded]}")

    # Ensure primary model is loaded
    print("\nLoading primary model (llama3-groq-tool-use:8b)...")
    success = await manager.ensure_model_loaded("llama3-groq-tool-use:8b")
    print(f"Primary model loaded: {success}")

    # Unload it
    print("\nUnloading primary model...")
    await manager.unload_model("llama3-groq-tool-use:8b")

    # Load phi4
    print("\nLoading phi4:14b...")
    success = await manager.ensure_model_loaded("phi4:14b")
    print(f"phi4 loaded: {success}")

    # Unload phi4
    print("\nUnloading phi4:14b...")
    await manager.unload_model("phi4:14b")

    print("\n✅ Model management test complete")


async def test_context_snapshot():
    """Test context snapshot creation and loading."""
    print("\n=== Testing Context Snapshot ===")

    manager = SubagentManager()

    # Create a snapshot
    snapshot = manager.create_snapshot(
        primary_model="llama3-groq-tool-use:8b",
        conversation_summary="User researched PAI concepts and wants a summary",
        last_user_message="Summarize the PAI research",
        pending_task="Synthesize research findings",
        telos_context={
            "profile_summary": "Developer working on Workshop",
            "active_project": "workshop",
            "goals": ["Integrate PAI patterns", "Implement subagents"]
        },
        research_context={
            "topic": "Daniel Miessler PAI Concept",
            "source_count": 12,
            "key_findings": "- System over intelligence\n- File-based context\n- Skills architecture"
        }
    )

    print(f"Snapshot created: {snapshot.snapshot_id}")
    print(f"  Primary model: {snapshot.primary_model}")
    print(f"  Research topic: {snapshot.research_topic}")
    print(f"  Source count: {snapshot.research_source_count}")

    # Check it was saved
    snapshot_path = manager.snapshot_dir / f"{snapshot.snapshot_id}.json"
    print(f"  Saved to: {snapshot_path}")
    print(f"  File exists: {snapshot_path.exists()}")

    # Load it back
    loaded = ContextSnapshot.load(snapshot_path)
    print(f"\nLoaded snapshot: {loaded.snapshot_id}")
    print(f"  Matches: {loaded.snapshot_id == snapshot.snapshot_id}")

    print("\n✅ Context snapshot test complete")


async def test_subagent_execution():
    """Test full subagent execution with research summarizer."""
    print("\n=== Testing Subagent Execution ===")

    manager = SubagentManager()

    # Create mock research data
    mock_research = {
        "topic": "Daniel Miessler PAI Concept",
        "original_query": "What is PAI?",
        "sources": [
            {
                "title": "Building a Personal AI Infrastructure",
                "url": "https://danielmiessler.com/blog/personal-ai-infrastructure",
                "snippet": "How I built my own unified, modular Agentic AI system named Kai",
                "key_points": [
                    "System over intelligence - orchestration matters more than raw AI capability",
                    "File-based context using markdown is powerful",
                    "Solve once, reuse forever follows UNIX philosophy"
                ]
            },
            {
                "title": "PAI GitHub Repository",
                "url": "https://github.com/danielmiessler/Personal_AI_Infrastructure",
                "snippet": "Open-source template for building your own AI-powered operating system",
                "key_points": [
                    "Skills, Agents, Hooks are the main components",
                    "Platform-independent design enables future migration",
                    "Context is loaded from markdown files"
                ]
            }
        ],
        "queries_executed": ["PAI architecture", "PAI implementation"]
    }

    print("Spawning research-summarizer subagent...")
    print("  This will: unload primary → load phi4 → summarize → unload phi4 → reload primary")
    print("  Expected time: 30-60 seconds\n")

    result = await manager.spawn_subagent(
        agent_name="research-summarizer",
        task="Synthesize the PAI research findings into actionable insights for the Workshop project",
        input_data={
            "research_platform": mock_research,
            "focus_areas": ["architecture", "context management", "skills system"]
        },
        primary_model="llama3-groq-tool-use:8b",
        research_context={
            "topic": mock_research["topic"],
            "source_count": len(mock_research["sources"]),
            "key_findings": "\n".join([
                f"- {point}"
                for s in mock_research["sources"]
                for point in s.get("key_points", [])[:2]
            ])
        }
    )

    print(f"\nSubagent result:")
    print(f"  Success: {result.get('success')}")
    print(f"  Agent: {result.get('agent')}")
    print(f"  Model: {result.get('model')}")
    print(f"  Duration: {result.get('duration_ms')}ms")
    print(f"  Snapshot ID: {result.get('snapshot_id')}")

    if result.get('output'):
        print(f"\n--- Subagent Output ({len(result['output'])} chars) ---")
        print(result['output'][:1500])
        if len(result['output']) > 1500:
            print(f"\n... [truncated, {len(result['output']) - 1500} more chars]")
    else:
        print(f"\nError: {result.get('error')}")

    print("\n✅ Subagent execution test complete")


async def main():
    print("=" * 60)
    print("Workshop Subagent System Test")
    print("=" * 60)

    try:
        # Test 1: Model management
        await test_model_management()

        # Test 2: Context snapshots
        await test_context_snapshot()

        # Test 3: Full subagent execution
        await test_subagent_execution()

        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
