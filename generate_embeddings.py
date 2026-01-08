#!/usr/bin/env python3
"""
Generate semantic embeddings for Workshop skills.

This script processes all SKILL.md files and creates a cached embeddings file
that the SemanticRouter uses for fast intent matching.

Usage:
    python generate_embeddings.py [--force]

Options:
    --force: Regenerate embeddings even if cache exists
"""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from semantic_router import SemanticRouter
from config import Config


async def main():
    force = "--force" in sys.argv

    config = Config()
    skills_dir = Path(__file__).parent / ".workshop" / "Skills"
    cache_path = config.SEMANTIC_EMBEDDINGS_CACHE

    print(f"Skills directory: {skills_dir}")
    print(f"Cache path: {cache_path}")
    print(f"Force rebuild: {force}")
    print()

    router = SemanticRouter(
        skills_dir=skills_dir,
        embeddings_cache_path=cache_path,
        embedding_model=config.SEMANTIC_EMBEDDING_MODEL,
    )

    print("Initializing semantic router...")
    await router.initialize(force_rebuild=force)

    print()
    print("=" * 50)
    print("Skill Embeddings Summary")
    print("=" * 50)

    for skill_name in router.list_skills():
        skill = router.get_skill_info(skill_name)
        print(f"\n{skill_name}:")
        print(f"  Purpose: {skill.purpose[:80]}...")
        print(f"  Utterances: {len(skill.utterances)}")
        print(f"  Tools: {skill.tools}")
        print(f"  Keywords: {skill.keywords[:5]}...")

    print()
    print(f"Total skills: {len(router.list_skills())}")
    print(f"Total utterances: {len(router._utterance_to_skill)}")
    print(f"Embeddings saved to: {cache_path}")


if __name__ == "__main__":
    asyncio.run(main())
