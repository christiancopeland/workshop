#!/usr/bin/env python3
"""
Test Agent with SkillRegistry Routing

This tests the end-to-end flow: user query -> routing -> enrichment -> LLM
"""

import asyncio
import sys
from pathlib import Path

# Add workshop to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_agent_routing():
    """Test the Agent's SkillRegistry integration."""
    from agent import Agent
    from skill_registry import SkillRegistry
    from memory import MemorySystem
    from config import Config

    config = Config()

    # Initialize components
    skills_dir = Path(__file__).parent / ".workshop" / "Skills"
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    registry = SkillRegistry(skills_dir)
    memory = MemorySystem(
        chroma_path=data_dir / "chromadb",
        sqlite_path=data_dir / "workshop.db"
    )

    # Initialize agent
    agent = Agent(
        model=config.MODEL,
        tools=registry,
        memory=memory,
        ollama_url=config.OLLAMA_URL,
        voice_mode=False
    )

    print("=" * 60)
    print("Agent Routing Integration Test")
    print("=" * 60)

    # Test queries that should trigger different behaviors
    test_queries = [
        # Should trigger ContextIntelligence skill
        "what changed recently?",

        # Should trigger ExploreProject workflow
        "what files are here?",

        # Should trigger FileOperations with context resolution
        "list the current directory",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("=" * 60)

        try:
            response = await agent.chat(query)
            print(f"\nResponse:\n{response[:500]}...")
        except Exception as e:
            print(f"\nError: {e}")

        print()


if __name__ == "__main__":
    asyncio.run(test_agent_routing())
