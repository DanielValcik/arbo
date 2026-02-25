"""Tests for NegRisk Arb Monitor (PM-203).

Tests verify:
1. NegRisk grouping: group by event, single market not grouped
2. Price sum check: in range, below 0.97, above 1.03
3. Monitoring: scan returns opportunities, history accumulates, no signals emitted
"""

from __future__ import annotations

from decimal import Decimal

from arbo.connectors.market_discovery import GammaMarket, MarketDiscovery
from arbo.strategies.arb_monitor import NegRiskArbMonitor, NegRiskOpportunity

# ================================================================
# Helpers
# ================================================================


def _make_negrisk_market(
    condition_id: str = "cond_1",
    question: str = "Who wins?",
    slug: str = "who-wins-election-yes",
    outcome_prices: list[str] | None = None,
) -> GammaMarket:
    raw = {
        "conditionId": condition_id,
        "question": question,
        "slug": slug,
        "outcomes": ["Yes", "No"],
        "outcomePrices": outcome_prices or ["0.50", "0.50"],
        "clobTokenIds": ["tok_yes", "tok_no"],
        "volume": "100000",
        "volume24hr": "5000",
        "liquidity": "10000",
        "active": True,
        "closed": False,
        "feesEnabled": False,
        "enableNegRisk": True,
        "tags": [],
    }
    return GammaMarket(raw)


def _make_discovery(markets: list[GammaMarket]) -> MarketDiscovery:
    disc = MarketDiscovery()
    disc._markets = {m.condition_id: m for m in markets}
    return disc


# ================================================================
# TestNegRiskGrouping
# ================================================================


class TestNegRiskGrouping:
    """NegRisk event grouping."""

    def test_group_by_event(self) -> None:
        """Markets with same slug prefix are grouped together."""
        markets = [
            _make_negrisk_market("c1", slug="us-election-2026-trump"),
            _make_negrisk_market("c2", slug="us-election-2026-biden"),
        ]
        disc = _make_discovery(markets)
        monitor = NegRiskArbMonitor(disc)

        negrisk = disc.get_negrisk_events()
        grouped = monitor._group_negrisk_events(negrisk)
        # Both should be in "us-election-2026" group
        assert len(grouped) == 1
        group = next(iter(grouped.values()))
        assert len(group) == 2

    def test_single_market_not_grouped(self) -> None:
        """Single market in a group is excluded (need ≥2 for comparison)."""
        markets = [
            _make_negrisk_market("c1", slug="unique-event-market-a"),
            _make_negrisk_market("c2", slug="different-event-market-b"),
        ]
        disc = _make_discovery(markets)
        monitor = NegRiskArbMonitor(disc)

        negrisk = disc.get_negrisk_events()
        grouped = monitor._group_negrisk_events(negrisk)
        # Each is in its own group, filtered out since < 2
        assert len(grouped) == 0


# ================================================================
# TestPriceSumCheck
# ================================================================


class TestPriceSumCheck:
    """Price sum deviation check."""

    def test_in_range(self) -> None:
        """Sum in [0.97, 1.03] → no opportunity."""
        markets = [
            _make_negrisk_market(
                "c1", slug="event-abc-group-option1", outcome_prices=["0.50", "0.50"]
            ),
            _make_negrisk_market(
                "c2", slug="event-abc-group-option2", outcome_prices=["0.49", "0.51"]
            ),
        ]
        disc = _make_discovery(markets)
        monitor = NegRiskArbMonitor(disc)

        grouped = monitor._group_negrisk_events(disc.get_negrisk_events())
        for event_id, group in grouped.items():
            opp = monitor._check_event(event_id, group)
            assert opp is None

    def test_below_097(self) -> None:
        """Sum < 0.97 → opportunity detected, is_under=True."""
        markets = [
            _make_negrisk_market(
                "c1", slug="event-abc-group-option1", outcome_prices=["0.40", "0.60"]
            ),
            _make_negrisk_market(
                "c2", slug="event-abc-group-option2", outcome_prices=["0.45", "0.55"]
            ),
        ]
        disc = _make_discovery(markets)
        monitor = NegRiskArbMonitor(disc)

        grouped = monitor._group_negrisk_events(disc.get_negrisk_events())
        for event_id, group in grouped.items():
            opp = monitor._check_event(event_id, group)
            assert opp is not None
            assert opp.is_under is True
            assert opp.price_sum == Decimal("0.85")  # 0.40 + 0.45

    def test_above_103(self) -> None:
        """Sum > 1.03 → opportunity detected, is_under=False."""
        markets = [
            _make_negrisk_market(
                "c1", slug="event-abc-group-option1", outcome_prices=["0.60", "0.40"]
            ),
            _make_negrisk_market(
                "c2", slug="event-abc-group-option2", outcome_prices=["0.55", "0.45"]
            ),
        ]
        disc = _make_discovery(markets)
        monitor = NegRiskArbMonitor(disc)

        grouped = monitor._group_negrisk_events(disc.get_negrisk_events())
        for event_id, group in grouped.items():
            opp = monitor._check_event(event_id, group)
            assert opp is not None
            assert opp.is_under is False
            assert opp.price_sum == Decimal("1.15")  # 0.60 + 0.55


# ================================================================
# TestMonitoring
# ================================================================


class TestMonitoring:
    """Monitoring-only behavior."""

    def test_scan_returns_opportunities(self) -> None:
        """Scan returns NegRiskOpportunity objects (not signals)."""
        markets = [
            _make_negrisk_market(
                "c1", slug="event-abc-group-option1", outcome_prices=["0.60", "0.40"]
            ),
            _make_negrisk_market(
                "c2", slug="event-abc-group-option2", outcome_prices=["0.55", "0.45"]
            ),
        ]
        disc = _make_discovery(markets)
        monitor = NegRiskArbMonitor(disc)

        opps = monitor.scan()
        assert len(opps) >= 1
        assert isinstance(opps[0], NegRiskOpportunity)

    def test_history_accumulates(self) -> None:
        """History grows with each scan."""
        markets = [
            _make_negrisk_market(
                "c1", slug="event-abc-group-option1", outcome_prices=["0.60", "0.40"]
            ),
            _make_negrisk_market(
                "c2", slug="event-abc-group-option2", outcome_prices=["0.55", "0.45"]
            ),
        ]
        disc = _make_discovery(markets)
        monitor = NegRiskArbMonitor(disc)

        monitor.scan()
        monitor.scan()
        history = monitor.get_history()
        assert len(history) >= 2

    def test_no_signals_emitted(self) -> None:
        """Monitor does NOT produce Signal objects — only NegRiskOpportunity."""
        markets = [
            _make_negrisk_market(
                "c1", slug="event-abc-group-option1", outcome_prices=["0.60", "0.40"]
            ),
            _make_negrisk_market(
                "c2", slug="event-abc-group-option2", outcome_prices=["0.55", "0.45"]
            ),
        ]
        disc = _make_discovery(markets)
        monitor = NegRiskArbMonitor(disc)

        opps = monitor.scan()
        for opp in opps:
            assert isinstance(opp, NegRiskOpportunity)
            # Verify it's NOT a Signal
            assert not hasattr(opp, "layer")
