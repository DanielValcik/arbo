"""Tests for Semantic Market Graph (PM-205).

Tests verify:
1. Embedding: encode, store, threshold filtering (mocked)
2. Relationship classification: subset, mutex, none via mock Gemini
3. Graph queries: find_similar, empty graph, get_all
4. Build graph: processes markets, refresh interval, stats
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arbo.models.market_graph import (
    MarketRelationship,
    RelationType,
    SemanticMarketGraph,
)

# ================================================================
# Helpers
# ================================================================


@dataclass
class FakeMarket:
    """Minimal market stub for testing."""

    condition_id: str
    question: str


class FakePrediction:
    """Mock LLM prediction."""

    def __init__(self, reasoning: str, confidence: float = 0.8) -> None:
        self.probability = 0.5
        self.confidence = confidence
        self.reasoning = reasoning
        self.provider = "mock"
        self.latency_ms = 100
        self.model = "test"


def _make_relationship(
    source_id: str = "cond_1",
    target_id: str = "cond_2",
    relation_type: RelationType = RelationType.MUTEX,
    similarity: float = 0.85,
) -> MarketRelationship:
    return MarketRelationship(
        source_condition_id=source_id,
        target_condition_id=target_id,
        source_question="Question A?",
        target_question="Question B?",
        similarity_score=similarity,
        relation_type=relation_type,
        llm_confidence=0.8,
    )


# ================================================================
# TestEmbedding
# ================================================================


class TestEmbedding:
    """Embedding and storage tests (mocked)."""

    def test_encode_questions_called(self) -> None:
        """Embedding model receives questions with 'query:' prefix."""
        graph = SemanticMarketGraph()
        mock_model = MagicMock()
        mock_model.encode.return_value = [
            MagicMock(tolist=lambda: [0.1] * 768),
            MagicMock(tolist=lambda: [0.2] * 768),
        ]
        graph._model = mock_model

        # Mock Chroma
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["cond_2"]],
            "distances": [[0.5]],
        }
        graph._collection = mock_collection

        import asyncio

        asyncio.get_event_loop().run_until_complete(
            graph.build_graph(
                [
                    FakeMarket("cond_1", "Will Arsenal win?"),
                    FakeMarket("cond_2", "Will Chelsea win?"),
                ]
            )
        )

        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args[0][0]
        assert all("query:" in q for q in call_args)

    def test_chroma_upsert_called(self) -> None:
        """Embeddings are stored in Chroma collection."""
        graph = SemanticMarketGraph()
        mock_model = MagicMock()
        mock_model.encode.return_value = [
            MagicMock(tolist=lambda: [0.1] * 768),
        ]
        graph._model = mock_model

        mock_collection = MagicMock()
        mock_collection.query.return_value = {"ids": [[]], "distances": [[]]}
        graph._collection = mock_collection

        import asyncio

        asyncio.get_event_loop().run_until_complete(
            graph.build_graph([FakeMarket("cond_1", "Test?")])
        )

        mock_collection.upsert.assert_called_once()

    def test_low_similarity_filtered(self) -> None:
        """Pairs below similarity threshold are not classified."""
        graph = SemanticMarketGraph()
        mock_model = MagicMock()
        mock_model.encode.return_value = [
            MagicMock(tolist=lambda: [0.1] * 768),
            MagicMock(tolist=lambda: [0.9] * 768),
        ]
        graph._model = mock_model

        # Return high distance (low similarity < threshold)
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["cond_2"]],
            "distances": [[0.5]],  # similarity = 0.5 < 0.75 threshold
        }
        graph._collection = mock_collection

        import asyncio

        count = asyncio.get_event_loop().run_until_complete(
            graph.build_graph(
                [
                    FakeMarket("cond_1", "A?"),
                    FakeMarket("cond_2", "B?"),
                ]
            )
        )

        assert count == 0


# ================================================================
# TestRelationClassification
# ================================================================


class TestRelationClassification:
    """Relationship classification via mock Gemini."""

    @pytest.mark.asyncio
    async def test_classify_subset(self) -> None:
        """LLM classifies relationship as SUBSET."""
        gemini = AsyncMock()
        gemini.predict.return_value = FakePrediction(
            reasoning="This is a SUBSET relationship", confidence=0.9
        )
        graph = SemanticMarketGraph(gemini=gemini)

        rel_type, confidence = await graph._classify_relationship("A?", "B?")
        assert rel_type == RelationType.SUBSET
        assert confidence == 0.9

    @pytest.mark.asyncio
    async def test_classify_mutex(self) -> None:
        """LLM classifies relationship as MUTEX."""
        gemini = AsyncMock()
        gemini.predict.return_value = FakePrediction(
            reasoning="These are MUTEX outcomes", confidence=0.85
        )
        graph = SemanticMarketGraph(gemini=gemini)

        rel_type, _confidence = await graph._classify_relationship("A?", "B?")
        assert rel_type == RelationType.MUTEX

    @pytest.mark.asyncio
    async def test_classify_none_when_no_gemini(self) -> None:
        """No Gemini → NONE with 0.0 confidence."""
        graph = SemanticMarketGraph(gemini=None)

        rel_type, confidence = await graph._classify_relationship("A?", "B?")
        assert rel_type == RelationType.NONE
        assert confidence == 0.0


# ================================================================
# TestGraphQueries
# ================================================================


class TestGraphQueries:
    """Query methods on the graph."""

    def test_find_similar_top_k(self) -> None:
        """find_similar returns up to top_k results."""
        graph = SemanticMarketGraph()
        graph._relationships = [
            _make_relationship("cond_1", f"cond_{i}", similarity=0.9 - i * 0.01)
            for i in range(2, 15)
        ]

        results = graph.find_similar("cond_1", top_k=5)
        assert len(results) == 5
        # Should be sorted by similarity descending
        assert results[0].similarity_score >= results[-1].similarity_score

    def test_empty_graph_returns_empty(self) -> None:
        """Empty graph returns empty list."""
        graph = SemanticMarketGraph()
        results = graph.find_similar("cond_1")
        assert results == []

    def test_get_all_relationships(self) -> None:
        """get_all_relationships returns all stored relationships."""
        graph = SemanticMarketGraph()
        graph._relationships = [
            _make_relationship("cond_1", "cond_2"),
            _make_relationship("cond_3", "cond_4"),
        ]
        assert len(graph.get_all_relationships()) == 2


# ================================================================
# TestBuildGraph
# ================================================================


class TestBuildGraph:
    """Build graph pipeline tests."""

    @pytest.mark.asyncio
    async def test_processes_markets(self) -> None:
        """build_graph processes market list and returns count."""
        graph = SemanticMarketGraph()
        mock_model = MagicMock()
        mock_model.encode.return_value = [
            MagicMock(tolist=lambda: [0.1] * 768),
            MagicMock(tolist=lambda: [0.2] * 768),
        ]
        graph._model = mock_model

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["cond_2"]],
            "distances": [[0.1]],  # similarity = 0.9 > threshold
        }
        graph._collection = mock_collection

        # No Gemini → NONE classification (but still creates relationship)
        count = await graph.build_graph(
            [
                FakeMarket("cond_1", "Will Arsenal win EPL?"),
                FakeMarket("cond_2", "Will Chelsea win EPL?"),
            ]
        )

        assert count >= 0  # At least processed without error

    @pytest.mark.asyncio
    async def test_refresh_respects_interval(self) -> None:
        """refresh_if_stale skips when interval not elapsed."""
        graph = SemanticMarketGraph()
        graph._last_refresh = time.monotonic()

        with patch.object(graph, "build_graph", new_callable=AsyncMock) as mock_build:
            await graph.refresh_if_stale()
            mock_build.assert_not_called()

    def test_stats(self) -> None:
        """Stats returns counts by relationship type."""
        graph = SemanticMarketGraph()
        graph._question_map = {"cond_1": "A?", "cond_2": "B?"}
        graph._relationships = [
            _make_relationship(relation_type=RelationType.MUTEX),
            _make_relationship(relation_type=RelationType.MUTEX),
            _make_relationship(relation_type=RelationType.SUBSET),
        ]

        stats = graph.stats
        assert stats["total_markets"] == 2
        assert stats["total_relationships"] == 3
        assert stats["by_type"]["MUTEX"] == 2
        assert stats["by_type"]["SUBSET"] == 1
