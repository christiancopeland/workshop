#!/usr/bin/env python3
"""
Test script for native Ollama tool calling.
Verifies that Workshop correctly uses Ollama's native function calling API.
"""

import asyncio
import json
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from agent import Agent
from memory import MemorySystem
from skill_registry import SkillRegistry


async def test_native_tools():
    """Test that native tool calling works"""
    print("=" * 60)
    print("Testing Native Ollama Tool Calling")
    print("=" * 60)

    # Initialize components
    config = Config()
    print(f"\n1. Using model: {config.MODEL}")
    print(f"   Ollama URL: {config.OLLAMA_URL}")

    print("\n2. Initializing memory system...")
    memory = MemorySystem(
        chroma_path=config.CHROMA_PATH,
        sqlite_path=config.SQLITE_PATH
    )

    print("\n3. Loading skills...")
    tools = SkillRegistry(
        skills_dir=Path(__file__).parent / ".workshop" / "Skills",
        dependencies={"memory": memory, "config": config}
    )
    print(f"   Loaded {len(tools.skills)} skills with {len(tools.list_all_tools())} tools")

    print("\n4. Initializing agent...")
    agent = Agent(
        model=config.MODEL,
        tools=tools,
        memory=memory,
        ollama_url=config.OLLAMA_URL
    )

    # Test building native tools
    print("\n5. Testing native tool format...")
    native_tools = agent._build_native_tools()
    print(f"   Built {len(native_tools)} native tool definitions")

    # Print a sample tool
    if native_tools:
        sample = native_tools[0]
        print(f"\n   Sample tool: {json.dumps(sample, indent=2)[:500]}...")

    # Test queries that should trigger tool calls
    test_queries = [
        "What files are in the current directory?",
        "What changed recently?",
    ]

    print("\n6. Testing tool-calling queries...")
    for query in test_queries:
        print(f"\n   Query: {query}")
        print("   " + "-" * 40)

        try:
            response = await agent.chat(query)
            # Truncate long responses
            if len(response) > 500:
                response = response[:500] + "..."
            print(f"   Response: {response}")
        except Exception as e:
            print(f"   ERROR: {e}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_native_tools())
