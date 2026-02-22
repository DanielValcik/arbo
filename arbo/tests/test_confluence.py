"""Tests for Confluence Scoring Engine (PM-209).

Tests verify:
1. Scoring: 1 layer=1pt, 2 layers=2pt, 3 layers=3pt, max 5, duplicate layer counted once
2. Position sizing: score 0-1 no trade, score 2 standard, score 3+ double capped
3. Edge threshold: value edge <5% not counted, >5% counted
4. Risk integration: approved returns opportunity, rejected filters out
5. Full pipeline: get_tradeable with mixed signals, audit trail
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from arbo.core.confluence import (
    ConfluenceScorer,
)
from arbo.core.risk_manager import RiskManager
from arbo.core.scanner import Signal, SignalDirection

# ================================================================
# Helpers
# ================================================================


def _make_signal(
    layer: int = 4,
    market_condition_id: str = "cond_1",
    token_id: str = "tok_1",
    direction: SignalDirection = SignalDirection.BUY_YES,
    edge: Decimal = Decimal("0.06"),
    confidence: Decimal = Decimal("0.7"),
) -> Signal:
    return Signal(
        layer=layer,
        market_condition_id=market_condition_id,
        token_id=token_id,
        direction=direction,
        edge=edge,
        confidence=confidence,
    )


@pytest.fixture
def risk_manager() -> RiskManager:
    RiskManager.reset()
    return RiskManager(capital=Decimal("2000"))


@pytest.fixture
def scorer(risk_manager: RiskManager) -> ConfluenceScorer:
    return ConfluenceScorer(risk_manager=risk_manager, capital=Decimal("2000"))


# ================================================================
# TestScoring
# ================================================================


class TestScoring:
    """Confluence score computation."""

    def test_one_layer_one_point(self, scorer: ConfluenceScorer) -> None:
        """Single layer signal → score 1."""
        signals = [_make_signal(layer=4)]
        opps = scorer.score_signals(signals)
        assert opps[0].score == 1

    def test_two_layers_two_points(self, scorer: ConfluenceScorer) -> None:
        """Signals from 2 different layers → score 2."""
        signals = [
            _make_signal(layer=4, market_condition_id="cond_1"),
            _make_signal(layer=7, market_condition_id="cond_1"),
        ]
        opps = scorer.score_signals(signals)
        assert opps[0].score == 2

    def test_three_layers_three_points(self, scorer: ConfluenceScorer) -> None:
        """Signals from 3 layers → score 3."""
        signals = [
            _make_signal(layer=4, market_condition_id="cond_1"),
            _make_signal(layer=7, market_condition_id="cond_1"),
            _make_signal(layer=5, market_condition_id="cond_1"),
        ]
        opps = scorer.score_signals(signals)
        assert opps[0].score == 3

    def test_max_score_5(self, scorer: ConfluenceScorer) -> None:
        """Score capped at 5 even with all layers present."""
        signals = [
            _make_signal(layer=2, market_condition_id="cond_1", edge=Decimal("0.10")),
            _make_signal(layer=4, market_condition_id="cond_1"),
            _make_signal(layer=5, market_condition_id="cond_1"),
            _make_signal(layer=7, market_condition_id="cond_1"),
            _make_signal(layer=8, market_condition_id="cond_1"),
        ]
        opps = scorer.score_signals(signals)
        assert opps[0].score == 5

    def test_duplicate_layer_counted_once(self, scorer: ConfluenceScorer) -> None:
        """Two signals from same layer → counted once."""
        signals = [
            _make_signal(layer=4, market_condition_id="cond_1", token_id="tok_1"),
            _make_signal(layer=4, market_condition_id="cond_1", token_id="tok_2"),
        ]
        opps = scorer.score_signals(signals)
        assert opps[0].score == 1

    def test_non_scoring_layer_not_counted(self, scorer: ConfluenceScorer) -> None:
        """Layer 1 and Layer 3 are not in SCORING_LAYERS and don't count."""
        signals = [
            _make_signal(layer=1, market_condition_id="cond_1"),
            _make_signal(layer=3, market_condition_id="cond_1"),
        ]
        opps = scorer.score_signals(signals)
        assert opps[0].score == 0


# ================================================================
# TestPositionSizing
# ================================================================


class TestPositionSizing:
    """Position sizing based on confluence score."""

    def test_score_0_no_trade(self, scorer: ConfluenceScorer) -> None:
        """Score 0 → 0% size (no trade)."""
        signals = [_make_signal(layer=1)]  # Layer 1 not in scoring layers
        opps = scorer.score_signals(signals)
        assert opps[0].recommended_size == Decimal("0")

    def test_score_1_standard_size_diagnostic(self, scorer: ConfluenceScorer) -> None:
        """Score 1 → standard size (min_score=1 diagnostic mode)."""
        signals = [_make_signal(layer=4)]
        opps = scorer.score_signals(signals)
        assert opps[0].position_size_pct == Decimal("0.025")
        assert opps[0].recommended_size == Decimal("50.00")

    def test_score_2_double_size(self, scorer: ConfluenceScorer) -> None:
        """Score 2 → double size (5% capital = $100) when min_score=1."""
        signals = [
            _make_signal(layer=4, market_condition_id="cond_1"),
            _make_signal(layer=7, market_condition_id="cond_1"),
        ]
        opps = scorer.score_signals(signals)
        assert opps[0].position_size_pct == Decimal("0.05")
        assert opps[0].recommended_size == Decimal("100.00")

    def test_score_3_double_capped(self, scorer: ConfluenceScorer) -> None:
        """Score 3+ → 5% capital = $100 (hard cap)."""
        signals = [
            _make_signal(layer=4, market_condition_id="cond_1"),
            _make_signal(layer=5, market_condition_id="cond_1"),
            _make_signal(layer=7, market_condition_id="cond_1"),
        ]
        opps = scorer.score_signals(signals)
        assert opps[0].position_size_pct == Decimal("0.05")
        assert opps[0].recommended_size == Decimal("100.00")


# ================================================================
# TestEdgeThreshold
# ================================================================


class TestEdgeThreshold:
    """Layer 2 edge threshold for confluence scoring."""

    def test_value_edge_below_threshold_not_counted(self, scorer: ConfluenceScorer) -> None:
        """Layer 2 with edge < 5% → not counted in score."""
        signals = [
            _make_signal(layer=2, market_condition_id="cond_1", edge=Decimal("0.04")),
            _make_signal(layer=4, market_condition_id="cond_1"),
        ]
        opps = scorer.score_signals(signals)
        # Only Layer 4 counts
        assert opps[0].score == 1
        assert 2 not in opps[0].contributing_layers

    def test_value_edge_above_threshold_counted(self, scorer: ConfluenceScorer) -> None:
        """Layer 2 with edge > 5% → counted in score."""
        signals = [
            _make_signal(layer=2, market_condition_id="cond_1", edge=Decimal("0.06")),
            _make_signal(layer=4, market_condition_id="cond_1"),
        ]
        opps = scorer.score_signals(signals)
        assert opps[0].score == 2
        assert 2 in opps[0].contributing_layers


# ================================================================
# TestRiskIntegration
# ================================================================


class TestRiskIntegration:
    """Integration with RiskManager."""

    def test_approved_returns_opportunity(self, scorer: ConfluenceScorer) -> None:
        """Risk-approved opportunity is returned."""
        signals = [
            _make_signal(layer=4, market_condition_id="cond_1"),
            _make_signal(layer=7, market_condition_id="cond_1"),
        ]
        tradeable = scorer.get_tradeable(signals, {"cond_1": "crypto"})
        assert len(tradeable) == 1
        assert tradeable[0].score == 2

    def test_rejected_filters_out(self, risk_manager: RiskManager) -> None:
        """Risk-rejected opportunity is filtered out."""
        # Trigger shutdown to reject all trades
        risk_manager._trigger_shutdown("test shutdown")
        scorer = ConfluenceScorer(risk_manager=risk_manager, capital=Decimal("2000"))

        signals = [
            _make_signal(layer=4, market_condition_id="cond_1"),
            _make_signal(layer=7, market_condition_id="cond_1"),
        ]
        tradeable = scorer.get_tradeable(signals, {"cond_1": "crypto"})
        assert len(tradeable) == 0


# ================================================================
# TestGetTradeable
# ================================================================


class TestGetTradeable:
    """Full get_tradeable pipeline."""

    def test_mixed_signals_pipeline(self, scorer: ConfluenceScorer) -> None:
        """Mixed signals: score >= 1 tradeable, score 0 not (diagnostic mode)."""
        signals = [
            # Market 1: score 2 (tradeable)
            _make_signal(layer=4, market_condition_id="cond_1"),
            _make_signal(layer=7, market_condition_id="cond_1"),
            # Market 2: score 1 (tradeable in diagnostic mode)
            _make_signal(layer=5, market_condition_id="cond_2"),
        ]
        tradeable = scorer.get_tradeable(signals, {"cond_1": "crypto", "cond_2": "politics"})
        assert len(tradeable) == 2
        tradeable_ids = {o.market_condition_id for o in tradeable}
        assert "cond_1" in tradeable_ids
        assert "cond_2" in tradeable_ids

    def test_stats_tracked(self, scorer: ConfluenceScorer) -> None:
        """Scorer stats are updated after pipeline runs."""
        signals = [
            _make_signal(layer=4, market_condition_id="cond_1"),
            _make_signal(layer=7, market_condition_id="cond_1"),
        ]
        scorer.get_tradeable(signals, {"cond_1": "crypto"})
        stats = scorer.stats
        assert stats["total_scored"] > 0
        assert stats["total_tradeable"] > 0
