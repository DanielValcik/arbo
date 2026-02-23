"""Semantic market graph using e5-large-v2 + Chroma DB (PM-205).

Encodes market questions as embeddings, stores in Chroma for similarity
search, and classifies relationships via Gemini LLM.

Relationship types:
  SUBSET:      Child market is a subset of parent (e.g., "Arsenal wins EPL" ⊂ "Top 4 finish")
  MUTEX:       Markets are mutually exclusive (only one can resolve YES)
  IMPLICATION: If A then B (e.g., "Arsenal wins EPL" → "Arsenal qualifies CL")
  TEMPORAL:    Same question at different time horizons
  NONE:        No meaningful relationship

See brief Layer 3 for full specification.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from functools import partial
from typing import Any

from arbo.utils.logger import get_logger

logger = get_logger("market_graph")

# Similarity threshold for considering a pair
SIMILARITY_THRESHOLD = 0.75

# Refresh interval (24 hours)
REFRESH_INTERVAL_S = 86400

# Top-K similar markets to check per market
DEFAULT_TOP_K = 10


class RelationType(Enum):
    """Type of relationship between two markets."""

    SUBSET = "SUBSET"
    MUTEX = "MUTEX"
    IMPLICATION = "IMPLICATION"
    TEMPORAL = "TEMPORAL"
    NONE = "NONE"


@dataclass
class MarketRelationship:
    """A classified relationship between two markets."""

    source_condition_id: str
    target_condition_id: str
    source_question: str
    target_question: str
    similarity_score: float
    relation_type: RelationType
    llm_confidence: float
    classified_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class SemanticMarketGraph:
    """Semantic graph of market relationships.

    Uses e5-large-v2 embeddings for similarity search and Gemini LLM for
    relationship classification. Stores embeddings in Chroma DB (in-memory).

    Features:
    - Encode market questions as 768-dim embeddings
    - Chroma similarity search with configurable threshold
    - LLM-based relationship classification
    - 24h refresh interval
    """

    def __init__(
        self,
        discovery: Any = None,
        gemini: Any = None,
    ) -> None:
        self._discovery = discovery
        self._gemini = gemini
        self._model: Any = None
        self._collection: Any = None
        self._relationships: list[MarketRelationship] = []
        self._last_refresh: float = 0
        self._question_map: dict[str, str] = {}  # condition_id → question

    async def initialize(self) -> None:
        """Lazy-load the embedding model and Chroma collection."""
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("intfloat/e5-large-v2")
            logger.info("embedding_model_loaded", model="e5-large-v2")
        except ImportError:
            logger.warning("sentence_transformers_not_available")
            self._model = None

        try:
            import chromadb

            client = chromadb.Client()
            self._collection = client.get_or_create_collection(
                name="polymarket_graph",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("chroma_collection_initialized")
        except ImportError:
            logger.warning("chromadb_not_available")
            self._collection = None

    async def close(self) -> None:
        """Clean up resources."""
        self._model = None
        self._collection = None

    async def build_graph(self, markets: list[Any]) -> int:
        """Build or rebuild the market graph from a list of markets.

        Args:
            markets: List of GammaMarket objects with question and condition_id.

        Returns:
            Number of relationships found.
        """
        if not self._model or not self._collection:
            logger.warning("graph_build_skipped", reason="model or collection not available")
            return 0

        # Encode and store all questions
        questions = []
        ids = []
        for m in markets:
            cid = m.condition_id if hasattr(m, "condition_id") else m.get("condition_id", "")
            question = m.question if hasattr(m, "question") else m.get("question", "")
            if cid and question:
                questions.append(f"query: {question}")
                ids.append(cid)
                self._question_map[cid] = question

        if not questions:
            return 0

        # Encode all questions (CPU-bound — run in thread to avoid blocking event loop)
        embeddings = await asyncio.to_thread(
            partial(self._model.encode, questions, show_progress_bar=False)
        )

        # Upsert into Chroma in batches (max ~5000 per batch due to Chroma limit)
        embedding_lists = [e.tolist() for e in embeddings]
        batch_size = 5000
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            await asyncio.to_thread(
                self._collection.upsert,
                ids=ids[start:end],
                embeddings=embedding_lists[start:end],
                documents=questions[start:end],
            )

        logger.info("graph_embeddings_stored", count=len(ids))

        # Find similar pairs and classify
        self._relationships.clear()
        for i, cid in enumerate(ids):
            results = self._collection.query(
                query_embeddings=[embeddings[i].tolist()],
                n_results=min(DEFAULT_TOP_K + 1, len(ids)),
            )

            if not results or not results["ids"]:
                continue

            for j, target_id in enumerate(results["ids"][0]):
                if target_id == cid:
                    continue

                distance = results["distances"][0][j] if results.get("distances") else 1.0
                similarity = 1.0 - distance

                if similarity < SIMILARITY_THRESHOLD:
                    continue

                source_q = self._question_map.get(cid, "")
                target_q = self._question_map.get(target_id, "")

                rel = await self._classify_relationship(source_q, target_q)

                self._relationships.append(
                    MarketRelationship(
                        source_condition_id=cid,
                        target_condition_id=target_id,
                        source_question=source_q,
                        target_question=target_q,
                        similarity_score=similarity,
                        relation_type=rel[0],
                        llm_confidence=rel[1],
                    )
                )

        self._last_refresh = time.monotonic()

        logger.info(
            "graph_built",
            markets=len(ids),
            relationships=len(self._relationships),
        )

        return len(self._relationships)

    async def refresh_if_stale(self) -> None:
        """Refresh graph if 24h interval has elapsed."""
        now = time.monotonic()
        if now - self._last_refresh < REFRESH_INTERVAL_S:
            return

        if self._discovery:
            markets = self._discovery.get_all()
            await self.build_graph(markets)

    def find_similar(
        self, condition_id: str, top_k: int = DEFAULT_TOP_K
    ) -> list[MarketRelationship]:
        """Find similar markets to a given condition_id.

        Args:
            condition_id: The source market condition ID.
            top_k: Maximum number of results.

        Returns:
            List of relationships sorted by similarity score descending.
        """
        matches = [
            r
            for r in self._relationships
            if r.source_condition_id == condition_id or r.target_condition_id == condition_id
        ]
        matches.sort(key=lambda r: r.similarity_score, reverse=True)
        return matches[:top_k]

    def get_all_relationships(self) -> list[MarketRelationship]:
        """Get all classified relationships."""
        return list(self._relationships)

    @property
    def stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        type_counts: dict[str, int] = {}
        for r in self._relationships:
            key = r.relation_type.value
            type_counts[key] = type_counts.get(key, 0) + 1

        return {
            "total_markets": len(self._question_map),
            "total_relationships": len(self._relationships),
            "by_type": type_counts,
        }

    async def _classify_relationship(
        self, question_a: str, question_b: str
    ) -> tuple[RelationType, float]:
        """Classify relationship between two markets via LLM.

        Falls back to NONE with 0.0 confidence if LLM unavailable.
        """
        if not self._gemini:
            return (RelationType.NONE, 0.0)

        prompt = (
            "Classify the relationship between these two prediction market questions.\n\n"
            f"Question A: {question_a}\n"
            f"Question B: {question_b}\n\n"
            "Respond with JSON:\n"
            '{"relationship": "SUBSET|MUTEX|IMPLICATION|TEMPORAL|NONE", '
            '"confidence": 0.0-1.0, "reasoning": "..."}\n\n'
            "SUBSET: A is a subset of B (if A true, then B true)\n"
            "MUTEX: A and B cannot both be true\n"
            "IMPLICATION: A implies B (but B does not imply A)\n"
            "TEMPORAL: Same question at different time horizons\n"
            "NONE: No meaningful relationship"
        )

        try:
            prediction = await self._gemini.predict(
                question=prompt,
                current_price=0.5,
                category="semantic_analysis",
            )

            if prediction is None:
                return (RelationType.NONE, 0.0)

            # Parse the reasoning field for relationship type
            reasoning = prediction.reasoning.upper()
            for rel_type in RelationType:
                if rel_type.value in reasoning:
                    return (rel_type, prediction.confidence)

            return (RelationType.NONE, prediction.confidence)

        except Exception as e:
            logger.debug("classification_error", error=str(e))
            return (RelationType.NONE, 0.0)
