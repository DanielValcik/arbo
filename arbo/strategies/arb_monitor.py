"""NegRisk arbitrage monitor — monitoring only (PM-203).

Monitors NegRisk multi-outcome events for pricing violations (sum of YES
prices outside [0.97, 1.03]). Does NOT generate signals or auto-execute.
Logs opportunities for CEO review.

See brief Layer 3 for full specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from arbo.connectors.market_discovery import GammaMarket, MarketDiscovery  # noqa: TC001
from arbo.utils.logger import get_logger

logger = get_logger("arb_monitor")


@dataclass
class NegRiskOpportunity:
    """A detected NegRisk pricing deviation."""

    event_id: str
    condition_ids: list[str]
    questions: list[str]
    yes_prices: list[Decimal]
    price_sum: Decimal
    deviation: Decimal
    is_under: bool  # True if sum < 0.97 (buy opportunity), False if > 1.03
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class NegRiskArbMonitor:
    """Monitors NegRisk events for pricing deviations.

    MONITORING ONLY — no Signal emitted, no auto-execution.
    Logs opportunities to in-memory history and structlog for CEO review.

    Sum of YES prices per NegRisk event group should be ~1.00.
    Alert if outside [0.97, 1.03] window.
    """

    def __init__(self, discovery: MarketDiscovery) -> None:
        self._discovery = discovery
        self._history: list[NegRiskOpportunity] = []
        self._total_scans = 0

    def scan(self) -> list[NegRiskOpportunity]:
        """Scan NegRisk markets for pricing deviations.

        Groups markets by event (slug prefix), sums YES prices per group,
        and reports any deviations outside [0.97, 1.03].

        Returns:
            List of detected opportunities (NOT signals — monitoring only).
        """
        self._total_scans += 1
        negrisk_markets = self._discovery.get_negrisk_events()

        if not negrisk_markets:
            return []

        grouped = self._group_negrisk_events(negrisk_markets)
        opportunities: list[NegRiskOpportunity] = []

        for event_id, markets in grouped.items():
            opp = self._check_event(event_id, markets)
            if opp is not None:
                opportunities.append(opp)
                self.log_opportunity(opp)

        logger.info(
            "arb_monitor_scan",
            events_checked=len(grouped),
            opportunities=len(opportunities),
        )

        return opportunities

    def log_opportunity(self, opp: NegRiskOpportunity) -> None:
        """Log opportunity for CEO review and store in history."""
        self._history.append(opp)

        logger.info(
            "negrisk_opportunity",
            event_id=opp.event_id,
            price_sum=str(opp.price_sum),
            deviation=str(opp.deviation),
            is_under=opp.is_under,
            markets=len(opp.condition_ids),
        )

    def get_history(self) -> list[NegRiskOpportunity]:
        """Get all detected opportunities."""
        return list(self._history)

    @property
    def stats(self) -> dict[str, Any]:
        """Get monitor statistics."""
        return {
            "total_scans": self._total_scans,
            "total_opportunities": len(self._history),
        }

    def _group_negrisk_events(self, markets: list[GammaMarket]) -> dict[str, list[GammaMarket]]:
        """Group NegRisk markets by event.

        Uses slug prefix (first 3 hyphen-separated segments) as event ID.
        Single-market groups are excluded (need ≥2 markets for comparison).
        """
        groups: dict[str, list[GammaMarket]] = {}

        for market in markets:
            slug = market.slug
            if not slug:
                continue

            # Use first 3 segments of slug as event grouping key
            parts = slug.split("-")
            prefix = "-".join(parts[:3]) if len(parts) >= 3 else slug

            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append(market)

        # Filter out single-market groups
        return {k: v for k, v in groups.items() if len(v) >= 2}

    def _check_event(self, event_id: str, markets: list[GammaMarket]) -> NegRiskOpportunity | None:
        """Check if a NegRisk event has a pricing deviation.

        Sum of YES prices should be within [0.97, 1.03].
        """
        yes_prices: list[Decimal] = []
        condition_ids: list[str] = []
        questions: list[str] = []

        for market in markets:
            price = market.price_yes
            if price is not None:
                yes_prices.append(price)
                condition_ids.append(market.condition_id)
                questions.append(market.question[:80])

        if not yes_prices:
            return None

        price_sum = sum(yes_prices, Decimal("0"))
        deviation = abs(Decimal("1") - price_sum)

        # Check if outside [0.97, 1.03]
        if Decimal("0.97") <= price_sum <= Decimal("1.03"):
            return None

        is_under = price_sum < Decimal("0.97")

        return NegRiskOpportunity(
            event_id=event_id,
            condition_ids=condition_ids,
            questions=questions,
            yes_prices=yes_prices,
            price_sum=price_sum,
            deviation=deviation,
            is_under=is_under,
        )
