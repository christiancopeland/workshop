"""
Semantic Router - Embedding-based intent routing for Workshop.

This module provides a two-stage routing system:
1. Fast embedding similarity matching (runs on CPU)
2. LLM confirmation for ambiguous cases

The semantic router complements the existing pattern-based routing in SkillRegistry,
providing a fallback for queries that don't match explicit patterns but are
semantically similar to known intents.

Architecture:
- Embeddings run on CPU (~1.2GB RAM for all-MiniLM-L6-v2)
- GPU reserved for Ollama LLM inference
- Pre-computed skill embeddings loaded at startup
- Dynamic query embedding at runtime
"""

import json
import asyncio
import aiohttp
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

from logger import get_logger

log = get_logger("semantic_router")

# Lazy load sentence-transformers to avoid startup cost if not needed
_encoder = None
_encoder_model = None


def get_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Lazy-load the sentence transformer encoder.
    Runs on CPU to preserve GPU VRAM for Ollama.
    """
    global _encoder, _encoder_model

    if _encoder is None or _encoder_model != model_name:
        try:
            from sentence_transformers import SentenceTransformer
            import torch

            # Force CPU to preserve GPU VRAM for Ollama
            device = "cpu"
            log.info(f"Loading embedding model: {model_name} (device={device})")

            _encoder = SentenceTransformer(model_name, device=device)
            _encoder_model = model_name

            log.info(f"Embedding model loaded: {_encoder.get_sentence_embedding_dimension()}d vectors")

        except ImportError:
            log.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise

    return _encoder


@dataclass
class SemanticMatch:
    """Result of semantic routing."""
    skill_name: str
    tool_name: Optional[str] = None
    similarity_score: float = 0.0
    matched_utterance: str = ""
    method: str = "semantic"  # "semantic", "llm_confirmed", "clarify"
    confidence: float = 0.0

    @property
    def matched(self) -> bool:
        return self.skill_name is not None and self.confidence > 0

    def __repr__(self):
        return (f"SemanticMatch(skill={self.skill_name}, tool={self.tool_name}, "
                f"score={self.similarity_score:.3f}, confidence={self.confidence:.3f}, "
                f"method={self.method})")


@dataclass
class SkillEmbedding:
    """Pre-computed embeddings for a skill."""
    skill_name: str
    purpose: str
    purpose_embedding: Optional[np.ndarray] = None
    utterances: List[str] = field(default_factory=list)
    utterance_embeddings: Optional[np.ndarray] = None  # Shape: (num_utterances, embedding_dim)
    tools: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Serialize for JSON storage (embeddings as lists)."""
        return {
            "skill_name": self.skill_name,
            "purpose": self.purpose,
            "purpose_embedding": self.purpose_embedding.tolist() if self.purpose_embedding is not None else None,
            "utterances": self.utterances,
            "utterance_embeddings": self.utterance_embeddings.tolist() if self.utterance_embeddings is not None else None,
            "tools": self.tools,
            "keywords": self.keywords,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SkillEmbedding':
        """Deserialize from JSON."""
        return cls(
            skill_name=data["skill_name"],
            purpose=data["purpose"],
            purpose_embedding=np.array(data["purpose_embedding"]) if data.get("purpose_embedding") else None,
            utterances=data.get("utterances", []),
            utterance_embeddings=np.array(data["utterance_embeddings"]) if data.get("utterance_embeddings") else None,
            tools=data.get("tools", []),
            keywords=data.get("keywords", []),
        )


class SemanticRouter:
    """
    Embedding-based semantic routing for Workshop skills.

    Two-stage routing:
    1. High confidence (>0.85): Direct routing based on embedding similarity
    2. Medium confidence (0.45-0.85): LLM confirmation
    3. Low confidence (<0.45): Request clarification

    Usage:
        router = SemanticRouter(skills_dir=Path(".workshop/Skills"))
        await router.initialize()

        match = await router.route("schedule a meeting for tomorrow")
        # SemanticMatch(skill="Calendar", confidence=0.92, method="semantic")
    """

    # Routing thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.85
    MEDIUM_CONFIDENCE_THRESHOLD = 0.45

    def __init__(
        self,
        skills_dir: Path = None,
        embeddings_cache_path: Path = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        ollama_url: str = "http://localhost:11434",
        llm_model: str = "phi3:mini",  # Small model for confirmation
    ):
        self.skills_dir = skills_dir or Path(".workshop/Skills")
        self.embeddings_cache_path = embeddings_cache_path or Path("data/skill_embeddings.json")
        self.embedding_model = embedding_model
        self.ollama_url = ollama_url
        self.llm_model = llm_model

        # Skill embeddings indexed by name
        self.skill_embeddings: Dict[str, SkillEmbedding] = {}

        # Pre-computed matrix of all utterance embeddings for fast similarity
        self._all_utterance_embeddings: Optional[np.ndarray] = None
        self._utterance_to_skill: List[Tuple[str, str, int]] = []  # (skill_name, utterance, idx)

        self._initialized = False

    async def initialize(self, force_rebuild: bool = False):
        """
        Initialize the semantic router.

        Loads pre-computed embeddings from cache or generates them from SKILL.md files.
        """
        if self._initialized and not force_rebuild:
            return

        # Try loading from cache first
        if not force_rebuild and self.embeddings_cache_path.exists():
            try:
                self._load_embeddings_cache()
                self._build_utterance_matrix()
                self._initialized = True
                log.info(f"Loaded {len(self.skill_embeddings)} skill embeddings from cache")
                return
            except Exception as e:
                log.warning(f"Failed to load embeddings cache: {e}")

        # Generate embeddings from SKILL.md files
        await self._generate_embeddings()
        self._build_utterance_matrix()
        self._save_embeddings_cache()
        self._initialized = True
        log.info(f"Generated embeddings for {len(self.skill_embeddings)} skills")

    def _load_embeddings_cache(self):
        """Load pre-computed embeddings from JSON cache."""
        with open(self.embeddings_cache_path, 'r') as f:
            data = json.load(f)

        self.skill_embeddings = {
            name: SkillEmbedding.from_dict(emb_data)
            for name, emb_data in data.get("skills", {}).items()
        }

        log.debug(f"Loaded embeddings for skills: {list(self.skill_embeddings.keys())}")

    def _save_embeddings_cache(self):
        """Save computed embeddings to JSON cache."""
        self.embeddings_cache_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "generated_at": datetime.now().isoformat(),
            "embedding_model": self.embedding_model,
            "skills": {
                name: emb.to_dict()
                for name, emb in self.skill_embeddings.items()
            }
        }

        with open(self.embeddings_cache_path, 'w') as f:
            json.dump(data, f, indent=2)

        log.info(f"Saved embeddings cache to {self.embeddings_cache_path}")

    async def _generate_embeddings(self):
        """Generate embeddings from SKILL.md files in skills directory."""
        encoder = get_encoder(self.embedding_model)

        if not self.skills_dir.exists():
            log.warning(f"Skills directory not found: {self.skills_dir}")
            return

        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue

            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue

            try:
                skill_emb = self._parse_skill_md(skill_md)

                # Generate embeddings
                if skill_emb.purpose:
                    skill_emb.purpose_embedding = encoder.encode(
                        skill_emb.purpose,
                        convert_to_numpy=True
                    )

                if skill_emb.utterances:
                    skill_emb.utterance_embeddings = encoder.encode(
                        skill_emb.utterances,
                        convert_to_numpy=True
                    )

                self.skill_embeddings[skill_emb.skill_name] = skill_emb
                log.debug(f"Generated embeddings for skill: {skill_emb.skill_name} "
                         f"({len(skill_emb.utterances)} utterances)")

            except Exception as e:
                log.error(f"Failed to process {skill_md}: {e}")

    def _parse_skill_md(self, skill_md: Path) -> SkillEmbedding:
        """Parse a SKILL.md file to extract routing information."""
        content = skill_md.read_text()
        skill_name = skill_md.parent.name

        # Extract purpose (first paragraph or ## Purpose section)
        purpose = ""
        purpose_match = None

        # Try ## Purpose section first
        import re
        purpose_section = re.search(r'## Purpose\s*\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
        if purpose_section:
            purpose = purpose_section.group(1).strip()
        else:
            # Fall back to first paragraph
            lines = content.split('\n')
            for line in lines:
                if line.strip() and not line.startswith('#'):
                    purpose = line.strip()
                    break

        # Extract user intent patterns as utterances
        utterances = []
        intent_section = re.search(r'## User Intent Patterns?\s*\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
        if intent_section:
            for line in intent_section.group(1).split('\n'):
                line = line.strip()
                if line.startswith('-'):
                    # Extract pattern, convert to natural utterance
                    pattern = line[1:].strip().strip('"\'`')
                    # Skip if it's a complex regex
                    if not any(c in pattern for c in ['^', '$', '\\s', '\\w', '(?:', '[', ']']):
                        utterances.append(pattern)

        # Also extract from ## Examples or ## Sample Queries
        examples_section = re.search(r'## (?:Examples?|Sample Queries?)\s*\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
        if examples_section:
            for line in examples_section.group(1).split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('*'):
                    example = line[1:].strip().strip('"\'`')
                    if example and example not in utterances:
                        utterances.append(example)

        # Extract from ## Semantic Utterances section (primary source for embeddings)
        semantic_section = re.search(r'## Semantic Utterances?\s*\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
        if semantic_section:
            for line in semantic_section.group(1).split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('*'):
                    utterance = line[1:].strip().strip('"\'`')
                    # Skip header lines
                    if utterance and not utterance.startswith('These') and utterance not in utterances:
                        utterances.append(utterance)

        # Generate synthetic utterances from purpose if none found
        if not utterances and purpose:
            # Create variations based on purpose
            utterances = [
                purpose,
                f"I want to {purpose.lower()}",
                f"Help me {purpose.lower()}",
                f"Can you {purpose.lower()}",
            ]

        # Extract tools
        tools = []
        tools_section = re.search(r'## (?:Available )?Tools?\s*\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
        if tools_section:
            for line in tools_section.group(1).split('\n'):
                line = line.strip()
                if line.startswith('-'):
                    # Extract tool name (usually first word or before parenthesis)
                    tool_line = line[1:].strip()
                    tool_name = tool_line.split('(')[0].split(':')[0].strip()
                    if tool_name:
                        tools.append(tool_name)

        # Extract keywords
        keywords = []
        keywords_section = re.search(r'## Keywords?\s*\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
        if keywords_section:
            kw_text = keywords_section.group(1).strip()
            keywords = [k.strip() for k in kw_text.replace('\n', ',').split(',') if k.strip()]

        return SkillEmbedding(
            skill_name=skill_name,
            purpose=purpose,
            utterances=utterances,
            tools=tools,
            keywords=keywords,
        )

    def _build_utterance_matrix(self):
        """Build a single matrix of all utterance embeddings for fast similarity."""
        all_embeddings = []
        self._utterance_to_skill = []

        for skill_name, skill_emb in self.skill_embeddings.items():
            if skill_emb.utterance_embeddings is not None:
                for i, utterance in enumerate(skill_emb.utterances):
                    all_embeddings.append(skill_emb.utterance_embeddings[i])
                    self._utterance_to_skill.append((skill_name, utterance, i))

        if all_embeddings:
            self._all_utterance_embeddings = np.stack(all_embeddings)
            log.debug(f"Built utterance matrix: {self._all_utterance_embeddings.shape}")
        else:
            self._all_utterance_embeddings = None

    def _compute_similarities(self, query_embedding: np.ndarray) -> List[Tuple[str, str, float]]:
        """
        Compute similarity between query and all skill utterances.

        Returns list of (skill_name, utterance, similarity) sorted by similarity descending.
        """
        if self._all_utterance_embeddings is None or len(self._utterance_to_skill) == 0:
            return []

        # Cosine similarity via dot product (embeddings are normalized)
        # Shape: (num_utterances,)
        similarities = np.dot(self._all_utterance_embeddings, query_embedding)

        # Build results
        results = []
        for i, (skill_name, utterance, _) in enumerate(self._utterance_to_skill):
            results.append((skill_name, utterance, float(similarities[i])))

        # Sort by similarity descending
        results.sort(key=lambda x: x[2], reverse=True)

        return results

    async def route(
        self,
        query: str,
        context: Dict[str, Any] = None,
        available_skills: List[str] = None,
    ) -> SemanticMatch:
        """
        Route a query to the most appropriate skill using semantic similarity.

        Args:
            query: User's natural language query
            context: Optional context dict (working_directory, active_project, etc.)
            available_skills: Optional list of skill names to restrict routing to

        Returns:
            SemanticMatch with skill_name, confidence, and routing method
        """
        if not self._initialized:
            await self.initialize()

        if not self.skill_embeddings:
            return SemanticMatch(skill_name=None, confidence=0.0, method="no_skills")

        # Encode query
        encoder = get_encoder(self.embedding_model)
        query_embedding = encoder.encode(query, convert_to_numpy=True)

        # Compute similarities to all utterances
        similarities = self._compute_similarities(query_embedding)

        if not similarities:
            return SemanticMatch(skill_name=None, confidence=0.0, method="no_matches")

        # Filter to available skills if specified
        if available_skills:
            similarities = [s for s in similarities if s[0] in available_skills]
            if not similarities:
                return SemanticMatch(skill_name=None, confidence=0.0, method="filtered_out")

        # Get best match
        best_skill, best_utterance, best_score = similarities[0]

        # High confidence: direct routing
        if best_score >= self.HIGH_CONFIDENCE_THRESHOLD:
            log.info(f"Semantic routing (high confidence): {best_skill} ({best_score:.3f})")
            return SemanticMatch(
                skill_name=best_skill,
                similarity_score=best_score,
                matched_utterance=best_utterance,
                method="semantic",
                confidence=best_score,
            )

        # Medium confidence: Return match for PromptRouter to confirm
        # NOTE: We skip LLM confirmation here because the PromptRouter already does
        # LLM-based routing in Stage 1. Double LLM calls add ~2s+ latency.
        if best_score >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            log.info(f"Semantic routing (medium confidence): {best_skill} ({best_score:.3f}) - returning for router confirmation")
            return SemanticMatch(
                skill_name=best_skill,
                similarity_score=best_score,
                matched_utterance=best_utterance,
                method="semantic_medium",
                confidence=best_score,
            )

        # Low confidence: request clarification
        log.info(f"Semantic routing (low confidence): best was {best_skill} ({best_score:.3f}) - needs clarification")
        return SemanticMatch(
            skill_name=None,
            similarity_score=best_score,
            method="clarify",
            confidence=best_score,
        )

    async def _llm_confirm(
        self,
        query: str,
        candidates: List[Tuple[str, str, float]]
    ) -> Optional[Tuple[str, str, float]]:
        """
        Use LLM to confirm or correct the semantic routing.

        Args:
            query: Original user query
            candidates: Top candidates as (skill_name, utterance, similarity)

        Returns:
            Confirmed candidate tuple or None if no good match
        """
        candidate_list = "\n".join([
            f"{i+1}. {skill} (example: \"{utterance}\")"
            for i, (skill, utterance, _) in enumerate(candidates)
        ])

        prompt = f"""User request: "{query}"

Top matching skills based on semantic similarity:
{candidate_list}

Which skill best matches the user's intent? Reply with ONLY the number (1, 2, or 3), or NONE if none are appropriate."""

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.llm_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_predict": 10,
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        response = data.get("response", "").strip().upper()

                        # Parse response
                        if "NONE" in response:
                            return None

                        for i, candidate in enumerate(candidates):
                            if str(i + 1) in response:
                                log.debug(f"LLM confirmed candidate {i+1}: {candidate[0]}")
                                return candidate

                        # Default to first if LLM didn't give clear answer
                        return candidates[0]
                    else:
                        log.warning(f"LLM confirmation failed: {resp.status}")
                        return candidates[0]  # Fall back to best semantic match

        except Exception as e:
            log.warning(f"LLM confirmation error: {e}")
            return candidates[0]  # Fall back to best semantic match

    def get_skill_info(self, skill_name: str) -> Optional[SkillEmbedding]:
        """Get embedding info for a specific skill."""
        return self.skill_embeddings.get(skill_name)

    def list_skills(self) -> List[str]:
        """List all skills with embeddings."""
        return list(self.skill_embeddings.keys())

    async def add_skill_utterances(self, skill_name: str, utterances: List[str]):
        """
        Dynamically add utterances to a skill's embeddings.
        Useful for learning from user interactions.
        """
        if skill_name not in self.skill_embeddings:
            log.warning(f"Skill not found: {skill_name}")
            return

        encoder = get_encoder(self.embedding_model)
        skill_emb = self.skill_embeddings[skill_name]

        # Encode new utterances
        new_embeddings = encoder.encode(utterances, convert_to_numpy=True)

        # Append to existing
        skill_emb.utterances.extend(utterances)
        if skill_emb.utterance_embeddings is not None:
            skill_emb.utterance_embeddings = np.vstack([
                skill_emb.utterance_embeddings,
                new_embeddings
            ])
        else:
            skill_emb.utterance_embeddings = new_embeddings

        # Rebuild matrix
        self._build_utterance_matrix()

        log.info(f"Added {len(utterances)} utterances to {skill_name}")


class HybridRouter:
    """
    Combines pattern-based routing (SkillRegistry) with semantic routing.

    Routing priority:
    1. Workflow triggers (highest priority, from SkillRegistry)
    2. High-confidence pattern matches (from SkillRegistry)
    3. Semantic similarity matches (from SemanticRouter)
    4. LLM-confirmed semantic matches
    5. LLM decision (fallback)
    """

    def __init__(
        self,
        skill_registry: "SkillRegistry",
        semantic_router: SemanticRouter,
        pattern_confidence_threshold: float = 0.7,
        semantic_confidence_threshold: float = 0.6,
    ):
        self.skill_registry = skill_registry
        self.semantic_router = semantic_router
        self.pattern_threshold = pattern_confidence_threshold
        self.semantic_threshold = semantic_confidence_threshold

    async def route(
        self,
        query: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Route query through hybrid pattern + semantic pipeline.

        Returns:
            Dict with:
                - method: "pattern" | "semantic" | "llm_confirmed" | "llm_fallback"
                - skill_name: matched skill or None
                - tool_name: matched tool or None
                - confidence: routing confidence
                - pattern_match: IntentMatch if pattern-based
                - semantic_match: SemanticMatch if semantic-based
        """
        result = {
            "method": "llm_fallback",
            "skill_name": None,
            "tool_name": None,
            "confidence": 0.0,
            "pattern_match": None,
            "semantic_match": None,
        }

        # Try pattern-based routing first
        if hasattr(self.skill_registry, 'route_by_intent'):
            pattern_match = self.skill_registry.route_by_intent(query, context or {})

            if pattern_match and pattern_match.matched:
                result["pattern_match"] = pattern_match

                # High confidence pattern match - use it
                if pattern_match.confidence >= self.pattern_threshold:
                    result["method"] = "pattern"
                    result["skill_name"] = pattern_match.skill_name
                    result["tool_name"] = pattern_match.tool_name
                    result["confidence"] = pattern_match.confidence
                    return result

        # Try semantic routing
        semantic_match = await self.semantic_router.route(
            query,
            context=context,
            available_skills=list(self.skill_registry.skills.keys()) if self.skill_registry else None
        )

        result["semantic_match"] = semantic_match

        if semantic_match.matched and semantic_match.confidence >= self.semantic_threshold:
            result["method"] = semantic_match.method
            result["skill_name"] = semantic_match.skill_name
            result["confidence"] = semantic_match.confidence
            return result

        # No confident match - will fall back to LLM
        return result


# Convenience function for quick routing
async def quick_route(
    query: str,
    skills_dir: Path = None,
    ollama_url: str = "http://localhost:11434",
) -> SemanticMatch:
    """
    Quick semantic routing without full initialization.
    Useful for one-off routing checks.
    """
    router = SemanticRouter(
        skills_dir=skills_dir,
        ollama_url=ollama_url,
    )
    await router.initialize()
    return await router.route(query)
