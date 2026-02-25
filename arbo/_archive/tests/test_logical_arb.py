"""Tests for Logical Arb Scanner (PM-206).

Tests verify:
1. NegRisk sum checks: in range, below 0.97, above 1.03
2. Pair analysis: mutex, subset, implication violations, under threshold
3. Scan cycle: full scan, rate limit
4. Signal format: layer=5, details contain reasoning
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from arbo.core.scanner import SignalDirection
from arbo.models.market_graph import MarketRelationship, RelationType, SemanticMarketGraph
from arbo.strategies.logical_arb import LogicalArbScanner, PricingViolation

# ================================================================
# Helpers
# ================================================================


@dataclass
class FakeMarket:
    """Minimal market stub."""

    condition_id: str
    slug: str
    price_yes: Decimal | None
    token_id_yes: str = "tok_yes"
    neg_risk: bool = True


def _make_relationship(
    source_id: str = "cond_1",
    target_id: str = "cond_2",
    relation_type: RelationType = RelationType.MUTEX,
    similarity: float = 0.85,
) -> MarketRelationship:
    return MarketRelationship(
        source_condition_id=source_id,
        target_condition_id=target_id,
        source_question="Q1?",
        target_question="Q2?",
        similarity_score=similarity,
        relation_type=relation_type,
        llm_confidence=0.8,
    )


def _make_discovery(markets: list[FakeMarket]) -> MagicMock:
    disc = MagicMock()
    market_map = {m.condition_id: m for m in markets}
    disc.get_by_condition_id = lambda cid: market_map.get(cid)
    disc.get_negrisk_events.return_value = [m for m in markets if m.neg_risk]
    return disc


# ================================================================
# TestNegRiskCheck
# ================================================================


class TestNegRiskCheck:
    """NegRisk sum of YES prices check."""

    def test_sum_in_range(self) -> None:
        """Sum within [0.97, 1.03] → no signal."""
        graph = MagicMock(spec=SemanticMarketGraph)
        graph.get_all_relationships.return_value = []

        markets = [
            FakeMarket("c1", "who-wins-election-yes", Decimal("0.50")),
            FakeMarket("c2", "who-wins-election-no", Decimal("0.49")),
        ]
        disc = _make_discovery(markets)
        scanner = LogicalArbScanner(market_graph=graph, discovery=disc)

        signals = scanner._check_negrisk_sum(disc.get_negrisk_events())
        assert len(signals) == 0

    def test_sum_below_097(self) -> None:
        """Sum < 0.97 → BUY signal (underpriced)."""
        graph = MagicMock(spec=SemanticMarketGraph)
        markets = [
            FakeMarket("c1", "who-wins-election-yes", Decimal("0.40")),
            FakeMarket("c2", "who-wins-election-no", Decimal("0.50")),
        ]
        disc = _make_discovery(markets)
        scanner = LogicalArbScanner(market_graph=graph, discovery=disc)

        signals = scanner._check_negrisk_sum(disc.get_negrisk_events())
        assert len(signals) == 1
        assert signals[0].direction == SignalDirection.BUY_YES

    def test_sum_above_103(self) -> None:
        """Sum > 1.03 → SELL signal (overpriced)."""
        graph = MagicMock(spec=SemanticMarketGraph)
        markets = [
            FakeMarket("c1", "who-wins-election-yes", Decimal("0.60")),
            FakeMarket("c2", "who-wins-election-no", Decimal("0.50")),
        ]
        disc = _make_discovery(markets)
        scanner = LogicalArbScanner(market_graph=graph, discovery=disc)

        signals = scanner._check_negrisk_sum(disc.get_negrisk_events())
        assert len(signals) == 1
        assert signals[0].direction == SignalDirection.SELL_YES


# ================================================================
# TestPairAnalysis
# ================================================================


class TestPairAnalysis:
    """Pair analysis for pricing violations."""

    @pytest.mark.asyncio
    async def test_mutex_violation(self) -> None:
        """MUTEX pair with sum > 1.03 → violation."""
        graph = MagicMock(spec=SemanticMarketGraph)
        markets = [
            FakeMarket("c1", "slug1", Decimal("0.60")),
            FakeMarket("c2", "slug2", Decimal("0.50")),
        ]
        disc = _make_discovery(markets)
        scanner = LogicalArbScanner(market_graph=graph, discovery=disc)

        rel = _make_relationship("c1", "c2", RelationType.MUTEX)
        violation = await scanner._analyze_pair(rel)
        assert violation is not None
        assert violation.relation_type == RelationType.MUTEX
        assert "MUTEX" in violation.reasoning

    @pytest.mark.asyncio
    async def test_subset_violation(self) -> None:
        """SUBSET pair with child > parent → violation."""
        graph = MagicMock(spec=SemanticMarketGraph)
        markets = [
            FakeMarket("c1", "slug1", Decimal("0.70")),  # child
            FakeMarket("c2", "slug2", Decimal("0.50")),  # parent
        ]
        disc = _make_discovery(markets)
        scanner = LogicalArbScanner(market_graph=graph, discovery=disc)

        rel = _make_relationship("c1", "c2", RelationType.SUBSET)
        violation = await scanner._analyze_pair(rel)
        assert violation is not None
        assert violation.relation_type == RelationType.SUBSET

    @pytest.mark.asyncio
    async def test_implication_violation(self) -> None:
        """IMPLICATION pair with P(B) < P(A) → violation."""
        graph = MagicMock(spec=SemanticMarketGraph)
        markets = [
            FakeMarket("c1", "slug1", Decimal("0.70")),  # A (higher)
            FakeMarket("c2", "slug2", Decimal("0.50")),  # B (lower, should be >= A)
        ]
        disc = _make_discovery(markets)
        scanner = LogicalArbScanner(market_graph=graph, discovery=disc)

        rel = _make_relationship("c1", "c2", RelationType.IMPLICATION)
        violation = await scanner._analyze_pair(rel)
        assert violation is not None
        assert "IMPLICATION" in violation.reasoning

    @pytest.mark.asyncio
    async def test_under_threshold_no_violation(self) -> None:
        """Pair with violation < 3% → no violation."""
        graph = MagicMock(spec=SemanticMarketGraph)
        markets = [
            FakeMarket("c1", "slug1", Decimal("0.51")),
            FakeMarket("c2", "slug2", Decimal("0.50")),
        ]
        disc = _make_discovery(markets)
        scanner = LogicalArbScanner(market_graph=graph, discovery=disc)

        rel = _make_relationship("c1", "c2", RelationType.MUTEX)
        violation = await scanner._analyze_pair(rel)
        assert violation is None


# ================================================================
# TestScanCycle
# ================================================================


class TestScanCycle:
    """Full scan cycle tests."""

    @pytest.mark.asyncio
    async def test_full_scan_with_violations(self) -> None:
        """Full scan detects violations across pairs and NegRisk."""
        graph = MagicMock(spec=SemanticMarketGraph)
        graph.get_all_relationships.return_value = [
            _make_relationship("c1", "c2", RelationType.MUTEX),
        ]

        markets = [
            FakeMarket("c1", "event-a-1", Decimal("0.60")),
            FakeMarket("c2", "event-a-2", Decimal("0.50")),
        ]
        disc = _make_discovery(markets)
        scanner = LogicalArbScanner(market_graph=graph, discovery=disc)

        signals = await scanner.scan()
        # Should detect MUTEX violation + NegRisk sum
        assert len(signals) >= 1
        assert all(s.layer == 5 for s in signals)

    @pytest.mark.asyncio
    async def test_rate_limit_respected(self) -> None:
        """Scanner respects rate limit for LLM calls."""
        graph = MagicMock(spec=SemanticMarketGraph)
        graph.get_all_relationships.return_value = []

        scanner = LogicalArbScanner(market_graph=graph)
        stats = scanner.stats
        assert "rate_limiter_remaining" in stats


# ================================================================
# TestSignalFormat
# ================================================================


class TestSignalFormat:
    """Signal output format."""

    def test_signal_layer_5(self) -> None:
        """Violation signals have layer=5."""
        graph = MagicMock(spec=SemanticMarketGraph)
        scanner = LogicalArbScanner(market_graph=graph)

        violation = PricingViolation(
            source_condition_id="c1",
            target_condition_id="c2",
            relation_type=RelationType.MUTEX,
            violation_pct=Decimal("0.10"),
            source_price=Decimal("0.60"),
            target_price=Decimal("0.50"),
            reasoning="MUTEX violation",
        )
        signal = scanner._violation_to_signal(violation)
        assert signal.layer == 5

    def test_signal_details_contain_reasoning(self) -> None:
        """Signal details contain reasoning and violation info."""
        graph = MagicMock(spec=SemanticMarketGraph)
        scanner = LogicalArbScanner(market_graph=graph)

        violation = PricingViolation(
            source_condition_id="c1",
            target_condition_id="c2",
            relation_type=RelationType.SUBSET,
            violation_pct=Decimal("0.15"),
            source_price=Decimal("0.65"),
            target_price=Decimal("0.50"),
            reasoning="SUBSET violation: child > parent",
        )
        signal = scanner._violation_to_signal(violation)
        assert "reasoning" in signal.details
        assert signal.details["relation_type"] == "SUBSET"
