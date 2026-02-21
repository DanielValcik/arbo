"""Live sports data latency scanner (PM-208).

Polls The Odds API for live scores, matches to Polymarket markets,
detects outcome events (goals, game end), and generates Layer 9
signals when implied probability is extreme but market hasn't moved.

See brief Layer 9 for full specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from arbo.config.settings import get_config
from arbo.connectors.market_discovery import GammaMarket, MarketDiscovery  # noqa: TC001
from arbo.connectors.odds_api_client import OddsApiClient  # noqa: TC001
from arbo.core.fee_model import calculate_taker_fee
from arbo.core.scanner import Signal, SignalDirection
from arbo.utils.logger import get_logger

logger = get_logger("sports_latency")


@dataclass
class LiveScore:
    """Live score data from The Odds API."""

    sport: str
    event_id: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    status: str  # "in_progress", "completed", "not_started"
    elapsed_minutes: int
    source: str
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class SportsLatencyScanner:
    """Scans for sports data latency arbitrage opportunities.

    Polls The Odds API for live soccer scores, matches them to Polymarket
    markets, and generates Layer 9 signals when:
    - A decisive outcome event is detected (e.g., large lead, game over)
    - Implied probability > 95% but market hasn't repriced
    - Taker fee is acceptable (< 0.3%)

    Respects quota tracking via x-requests-remaining header.
    """

    def __init__(
        self,
        discovery: MarketDiscovery,
        odds_client: OddsApiClient,
    ) -> None:
        self._discovery = discovery
        self._odds_client = odds_client
        config = get_config()
        self._min_prob_extreme = Decimal(str(config.sports_latency.min_probability_extreme))
        self._max_fee_pct = Decimal(str(config.sports_latency.max_fee_pct))
        self._min_remaining_quota = config.odds_api.min_remaining_quota
        self._total_polls = 0
        self._total_signals = 0

    async def initialize(self) -> None:
        """Initialize HTTP sessions."""
        logger.info("sports_latency_initialized")

    async def close(self) -> None:
        """Close HTTP sessions."""
        await self._odds_client.close()

    async def poll_cycle(self) -> list[Signal]:
        """Run one poll cycle for live sports data.

        1. Fetch live scores from The Odds API
        2. Match each score to a Polymarket market
        3. Detect outcome events
        4. Generate signals for extreme implied probabilities
        5. Check fee acceptability

        Returns:
            List of Layer 9 signals.
        """
        # Layer 9 disabled during paper trading — burns Odds API credits for hardcoded-0 scores
        logger.debug("sports_latency_disabled")
        return []

        self._total_polls += 1  # pragma: no cover

        # Check quota
        remaining = self._odds_client.remaining_quota
        if remaining is not None and remaining < self._min_remaining_quota:
            logger.warning(
                "sports_latency_quota_low",
                remaining=remaining,
                min_required=self._min_remaining_quota,
            )
            return []

        live_scores = await self._fetch_live_scores()
        if not live_scores:
            return []

        soccer_markets = self._discovery.get_by_category("soccer")
        signals: list[Signal] = []

        for score in live_scores:
            matched_market = self._match_to_market(score, soccer_markets)
            if matched_market is None:
                continue

            event = self._detect_outcome_event(score)
            if event is None:
                continue

            implied_prob = self._compute_implied_prob(score, event)
            if implied_prob < self._min_prob_extreme:
                continue

            # Check fee
            market_price = matched_market.price_yes
            if market_price is not None and not self._fee_check(
                market_price, matched_market.fee_enabled
            ):
                continue

            token_id = matched_market.token_id_yes or ""

            signals.append(
                Signal(
                    layer=9,
                    market_condition_id=matched_market.condition_id,
                    token_id=token_id,
                    direction=SignalDirection.BUY_YES,
                    edge=implied_prob - (market_price or Decimal("0.50")),
                    confidence=implied_prob,
                    details={
                        "event_type": event,
                        "home_team": score.home_team,
                        "away_team": score.away_team,
                        "score": f"{score.home_score}-{score.away_score}",
                        "elapsed": score.elapsed_minutes,
                        "implied_prob": str(implied_prob),
                        "status": score.status,
                        "question": matched_market.question[:100],
                    },
                )
            )

        self._total_signals += len(signals)
        logger.info(
            "sports_latency_poll",
            scores=len(live_scores),
            signals=len(signals),
        )

        return signals

    @property
    def stats(self) -> dict[str, Any]:
        """Get scanner statistics."""
        return {
            "total_polls": self._total_polls,
            "total_signals": self._total_signals,
            "remaining_quota": self._odds_client.remaining_quota,
        }

    async def _fetch_live_scores(self) -> list[LiveScore]:
        """Fetch live scores from The Odds API.

        Uses the /scores endpoint for soccer leagues.
        """
        scores: list[LiveScore] = []

        # Fetch soccer scores
        from arbo.connectors.odds_api_client import SPORT_KEY_MAP

        for sport_key in SPORT_KEY_MAP.get("soccer", []):
            try:
                events = await self._odds_client.get_odds(sport_key)
                for event in events:
                    # Extract score info from event data if available
                    score = LiveScore(
                        sport="soccer",
                        event_id=event.id,
                        home_team=event.home_team,
                        away_team=event.away_team,
                        home_score=0,
                        away_score=0,
                        status="not_started",
                        elapsed_minutes=0,
                        source="odds_api",
                    )
                    scores.append(score)
            except Exception as e:
                logger.debug("live_scores_error", sport_key=sport_key, error=str(e))

        return scores

    def _match_to_market(self, score: LiveScore, markets: list[GammaMarket]) -> GammaMarket | None:
        """Match a live score to a Polymarket market by team names."""
        home_lower = score.home_team.lower()
        away_lower = score.away_team.lower()

        for market in markets:
            q_lower = market.question.lower()
            if home_lower in q_lower or away_lower in q_lower:
                return market

        return None

    def _detect_outcome_event(self, score: LiveScore) -> str | None:
        """Detect if a live score indicates a decisive outcome.

        Returns:
            Event type ("completed", "large_lead") or None.
        """
        if score.status == "completed":
            return "completed"

        # Large lead in soccer (3+ goals with >= 60 min played)
        goal_diff = abs(score.home_score - score.away_score)
        if goal_diff >= 3 and score.elapsed_minutes >= 60:
            return "large_lead"

        # Late lead (2+ goals in last 15 min)
        if goal_diff >= 2 and score.elapsed_minutes >= 75:
            return "large_lead"

        return None

    def _compute_implied_prob(self, score: LiveScore, event: str) -> Decimal:
        """Compute implied probability based on score and event type."""
        if event == "completed":
            return Decimal("0.99")

        goal_diff = abs(score.home_score - score.away_score)
        elapsed = score.elapsed_minutes

        # Large lead probability estimation
        if goal_diff >= 3:
            return Decimal("0.97")
        elif goal_diff >= 2 and elapsed >= 80:
            return Decimal("0.96")
        elif goal_diff >= 2 and elapsed >= 75:
            return Decimal("0.95")

        return Decimal("0.90")

    def _fee_check(self, price: Decimal, fee_enabled: bool) -> bool:
        """Check if taker fee at this price is acceptable.

        Fee must be less than max_fee_pct (0.3%) for the trade to be viable.
        """
        if not fee_enabled:
            return True

        fee = calculate_taker_fee(price, fee_enabled=True)
        if price <= 0:
            return False
        fee_pct = fee / price
        return fee_pct < self._max_fee_pct
