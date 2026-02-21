"""Tests for Live Sports Latency Scanner (PM-208).

Tests verify:
1. Live score parsing: completed, in-progress, no events
2. Outcome detection: completed, large lead, close game
3. Fee check: below threshold, above threshold
4. Poll cycle: extreme prob signal, no events, quota respected
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arbo.connectors.market_discovery import GammaMarket, MarketDiscovery
from arbo.connectors.odds_api_client import OddsApiClient
from arbo.core.scanner import SignalDirection
from arbo.strategies.sports_latency import LiveScore, SportsLatencyScanner

# ================================================================
# Helpers
# ================================================================


def _make_score(
    home_team: str = "Arsenal",
    away_team: str = "Chelsea",
    home_score: int = 0,
    away_score: int = 0,
    status: str = "in_progress",
    elapsed: int = 45,
) -> LiveScore:
    return LiveScore(
        sport="soccer",
        event_id="evt_1",
        home_team=home_team,
        away_team=away_team,
        home_score=home_score,
        away_score=away_score,
        status=status,
        elapsed_minutes=elapsed,
        source="test",
    )


def _make_market(
    condition_id: str = "cond_1",
    question: str = "Will Arsenal win?",
    outcome_prices: list[str] | None = None,
    fee_enabled: bool = False,
) -> GammaMarket:
    raw = {
        "conditionId": condition_id,
        "question": question,
        "slug": "arsenal-win",
        "outcomes": ["Yes", "No"],
        "outcomePrices": outcome_prices or ["0.50", "0.50"],
        "clobTokenIds": ["tok_yes", "tok_no"],
        "volume": "100000",
        "volume24hr": "5000",
        "liquidity": "10000",
        "active": True,
        "closed": False,
        "feesEnabled": fee_enabled,
        "enableNegRisk": False,
        "tags": [{"label": "soccer"}],
    }
    return GammaMarket(raw)


def _make_odds_client(remaining_quota: int | None = 400) -> OddsApiClient:
    client = MagicMock(spec=OddsApiClient)
    client.remaining_quota = remaining_quota
    client.close = AsyncMock()
    client.get_odds = AsyncMock(return_value=[])
    return client


# ================================================================
# TestLiveScoreParsing
# ================================================================


class TestLiveScoreParsing:
    """Live score creation and field access."""

    def test_completed_game(self) -> None:
        """Completed game has correct status."""
        score = _make_score(status="completed", home_score=2, away_score=1)
        assert score.status == "completed"
        assert score.home_score == 2

    def test_in_progress_game(self) -> None:
        """In-progress game stores elapsed time."""
        score = _make_score(status="in_progress", elapsed=67)
        assert score.status == "in_progress"
        assert score.elapsed_minutes == 67

    def test_not_started_game(self) -> None:
        """Not-started game defaults."""
        score = _make_score(status="not_started", elapsed=0)
        assert score.status == "not_started"
        assert score.elapsed_minutes == 0


# ================================================================
# TestOutcomeDetection
# ================================================================


class TestOutcomeDetection:
    """Outcome event detection from live scores."""

    def test_completed_detected(self) -> None:
        """Completed game → 'completed' event."""
        disc = MarketDiscovery()
        odds = _make_odds_client()
        scanner = SportsLatencyScanner(discovery=disc, odds_client=odds)

        score = _make_score(status="completed", home_score=2, away_score=1)
        event = scanner._detect_outcome_event(score)
        assert event == "completed"

    def test_large_lead_detected(self) -> None:
        """3+ goal lead with 60+ minutes → 'large_lead' event."""
        disc = MarketDiscovery()
        odds = _make_odds_client()
        scanner = SportsLatencyScanner(discovery=disc, odds_client=odds)

        score = _make_score(home_score=4, away_score=1, elapsed=65)
        event = scanner._detect_outcome_event(score)
        assert event == "large_lead"

    def test_close_game_no_event(self) -> None:
        """Close game (1-0 at 50 min) → no event."""
        disc = MarketDiscovery()
        odds = _make_odds_client()
        scanner = SportsLatencyScanner(discovery=disc, odds_client=odds)

        score = _make_score(home_score=1, away_score=0, elapsed=50)
        event = scanner._detect_outcome_event(score)
        assert event is None


# ================================================================
# TestFeeCheck
# ================================================================


class TestFeeCheck:
    """Fee favorability check."""

    def test_below_threshold_passes(self) -> None:
        """Low fee at extreme price passes check."""
        disc = MarketDiscovery()
        odds = _make_odds_client()
        scanner = SportsLatencyScanner(discovery=disc, odds_client=odds)

        # At p=0.95, fee = 0.95 * 0.05 * 0.0315 = ~0.0015, fee_pct = ~0.16%
        assert scanner._fee_check(Decimal("0.95"), fee_enabled=True) is True

    def test_above_threshold_fails(self) -> None:
        """High fee at midpoint price fails check."""
        disc = MarketDiscovery()
        odds = _make_odds_client()
        scanner = SportsLatencyScanner(discovery=disc, odds_client=odds)

        # At p=0.50, fee = 0.50 * 0.50 * 0.0315 = ~0.008, fee_pct = ~1.6%
        assert scanner._fee_check(Decimal("0.50"), fee_enabled=True) is False

    def test_no_fee_always_passes(self) -> None:
        """Non-fee market always passes."""
        disc = MarketDiscovery()
        odds = _make_odds_client()
        scanner = SportsLatencyScanner(discovery=disc, odds_client=odds)

        assert scanner._fee_check(Decimal("0.50"), fee_enabled=False) is True


# ================================================================
# TestPollCycle
# ================================================================


class TestPollCycle:
    """Full poll cycle tests."""

    @pytest.mark.asyncio
    async def test_extreme_prob_signal(self) -> None:
        """Completed game with matched market → Layer 9 signal."""
        market = _make_market(question="Will Arsenal win against Chelsea?")
        disc = MarketDiscovery()
        disc._markets = {market.condition_id: market}

        odds = _make_odds_client()
        scanner = SportsLatencyScanner(discovery=disc, odds_client=odds)

        # Mock _fetch_live_scores to return completed game
        completed_score = _make_score(
            home_team="Arsenal",
            away_team="Chelsea",
            home_score=2,
            away_score=0,
            status="completed",
        )

        with patch.object(scanner, "_fetch_live_scores", return_value=[completed_score]):
            signals = await scanner.poll_cycle()
            assert len(signals) == 1
            assert signals[0].layer == 9
            assert signals[0].direction == SignalDirection.BUY_YES

    @pytest.mark.asyncio
    async def test_no_events_no_signal(self) -> None:
        """Live games without decisive events → no signals."""
        market = _make_market(question="Will Arsenal win?")
        disc = MarketDiscovery()
        disc._markets = {market.condition_id: market}

        odds = _make_odds_client()
        scanner = SportsLatencyScanner(discovery=disc, odds_client=odds)

        # Close game, no decisive event
        close_score = _make_score(home_score=1, away_score=0, elapsed=50)

        with patch.object(scanner, "_fetch_live_scores", return_value=[close_score]):
            signals = await scanner.poll_cycle()
            assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_quota_respected(self) -> None:
        """Low quota → no API calls, empty signals."""
        disc = MarketDiscovery()
        odds = _make_odds_client(remaining_quota=10)  # Below min_remaining_quota (50)
        scanner = SportsLatencyScanner(discovery=disc, odds_client=odds)

        signals = await scanner.poll_cycle()
        assert len(signals) == 0
