"""Logical/combinatorial arbitrage scanner (PM-206).

Identifies pricing inconsistencies between related markets using the
semantic market graph and LLM confirmation. Also checks NegRisk events
for sum-of-YES violations.

See brief Layer 5 for full specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from arbo.agents.gemini_agent import RateLimiter
from arbo.config.settings import get_config
from arbo.connectors.market_discovery import MarketDiscovery  # noqa: TC001
from arbo.core.scanner import Signal, SignalDirection
from arbo.models.market_graph import MarketRelationship, RelationType, SemanticMarketGraph
from arbo.utils.logger import get_logger

logger = get_logger("logical_arb")


@dataclass
class PricingViolation:
    """A detected pricing inconsistency between related markets."""

    source_condition_id: str
    target_condition_id: str
    relation_type: RelationType
    violation_pct: Decimal
    source_price: Decimal
    target_price: Decimal
    reasoning: str
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class LogicalArbScanner:
    """Scans for logical/combinatorial arbitrage opportunities.

    Uses the semantic market graph to find related market pairs, then
    checks for pricing violations based on relationship type:
    - MUTEX: sum of YES prices should be ≤ 1.0
    - SUBSET: child price should be ≤ parent price
    - IMPLICATION: P(B) should be ≥ P(A) when A→B

    Also checks NegRisk multi-outcome events for sum violations.

    Rate limited to max_llm_calls_per_hour for Gemini confirmations.
    """

    def __init__(
        self,
        market_graph: SemanticMarketGraph,
        gemini: Any = None,
        discovery: MarketDiscovery | None = None,
    ) -> None:
        self._market_graph = market_graph
        self._gemini = gemini
        self._discovery = discovery
        config = get_config()
        self._min_violation = Decimal(str(config.logical_arb.min_pricing_violation))
        self._negrisk_threshold = Decimal(str(config.logical_arb.negrisk_sum_threshold))
        self._rate_limiter = RateLimiter(max_calls=config.logical_arb.max_llm_calls_per_hour)
        self._total_scans = 0
        self._total_violations = 0

    async def scan(self) -> list[Signal]:
        """Run full logical arb scan.

        1. Get relationships from market graph (type != NONE)
        2. Analyze each pair for pricing violations
        3. Check NegRisk events for sum violations
        4. Convert violations to Layer 5 signals

        Returns:
            List of Layer 5 signals.
        """
        self._total_scans += 1
        signals: list[Signal] = []

        # Scan related market pairs
        relationships = self._market_graph.get_all_relationships()
        # Sort by similarity descending (prioritize strongest pairs)
        relationships.sort(key=lambda r: r.similarity_score, reverse=True)

        for rel in relationships:
            if rel.relation_type == RelationType.NONE:
                continue

            violation = await self._analyze_pair(rel)
            if violation:
                self._total_violations += 1
                signals.append(self._violation_to_signal(violation))

        # NegRisk sum check
        if self._discovery:
            negrisk_signals = self._check_negrisk_sum(self._discovery.get_negrisk_events())
            signals.extend(negrisk_signals)

        logger.info(
            "logical_arb_scan",
            pairs_checked=len(relationships),
            violations=len(signals),
        )

        return signals

    @property
    def stats(self) -> dict[str, Any]:
        """Get scanner statistics."""
        return {
            "total_scans": self._total_scans,
            "total_violations": self._total_violations,
            "rate_limiter_remaining": self._rate_limiter.remaining(),
        }

    async def _analyze_pair(self, rel: MarketRelationship) -> PricingViolation | None:
        """Analyze a market pair for pricing violations based on relationship type.

        Args:
            rel: The relationship between two markets.

        Returns:
            PricingViolation if found, None otherwise.
        """
        source_price = self._get_market_price(rel.source_condition_id)
        target_price = self._get_market_price(rel.target_condition_id)

        if source_price is None or target_price is None:
            return None

        violation_pct = Decimal("0")
        reasoning = ""

        if rel.relation_type == RelationType.MUTEX:
            # MUTEX: sum of YES prices should be ≤ 1.0
            price_sum = source_price + target_price
            if price_sum > Decimal("1") + self._min_violation:
                violation_pct = price_sum - Decimal("1")
                reasoning = f"MUTEX violation: sum={price_sum} > 1.0"

        elif rel.relation_type == RelationType.SUBSET:
            # SUBSET: child price ≤ parent price
            if source_price > target_price + self._min_violation:
                violation_pct = source_price - target_price
                reasoning = f"SUBSET violation: child={source_price} > parent={target_price}"

        elif rel.relation_type == RelationType.IMPLICATION:
            # IMPLICATION: P(B) ≥ P(A) when A→B
            if target_price < source_price - self._min_violation:
                violation_pct = source_price - target_price
                reasoning = f"IMPLICATION violation: P(B)={target_price} < P(A)={source_price}"

        if violation_pct < self._min_violation:
            return None

        # Confirm via Gemini if available and rate limit allows
        if self._gemini and self._rate_limiter.allow():
            confirmed = await self._confirm_via_llm(rel, reasoning)
            if not confirmed:
                return None

        return PricingViolation(
            source_condition_id=rel.source_condition_id,
            target_condition_id=rel.target_condition_id,
            relation_type=rel.relation_type,
            violation_pct=violation_pct,
            source_price=source_price,
            target_price=target_price,
            reasoning=reasoning,
        )

    def _check_negrisk_sum(self, markets: list[Any]) -> list[Signal]:
        """Check NegRisk multi-outcome events for sum violations.

        Sum of YES prices should be in [0.97, 1.03]. Signals emitted
        when outside this range.
        """
        signals: list[Signal] = []

        # Group by slug prefix (first 2 segments) as proxy for event grouping
        events: dict[str, list[Any]] = {}
        for m in markets:
            slug = m.slug if hasattr(m, "slug") else ""
            prefix = "-".join(slug.split("-")[:3]) if slug else ""
            if prefix:
                if prefix not in events:
                    events[prefix] = []
                events[prefix].append(m)

        for event_id, event_markets in events.items():
            if len(event_markets) < 2:
                continue

            yes_prices = []
            for m in event_markets:
                price = m.price_yes if hasattr(m, "price_yes") else None
                if price is not None:
                    yes_prices.append(price)

            if not yes_prices:
                continue

            price_sum = sum(yes_prices, Decimal("0"))
            lower = Decimal("1") - self._negrisk_threshold
            upper = Decimal("1") + self._negrisk_threshold

            if lower <= price_sum <= upper:
                continue

            is_under = price_sum < lower
            deviation = abs(Decimal("1") - price_sum)
            token_id = (
                event_markets[0].token_id_yes if hasattr(event_markets[0], "token_id_yes") else ""
            )

            signals.append(
                Signal(
                    layer=5,
                    market_condition_id=(
                        event_markets[0].condition_id
                        if hasattr(event_markets[0], "condition_id")
                        else ""
                    ),
                    token_id=token_id or "",
                    direction=SignalDirection.BUY_YES if is_under else SignalDirection.SELL_YES,
                    edge=deviation,
                    confidence=Decimal("0.80"),
                    details={
                        "negrisk_sum": str(price_sum),
                        "deviation": str(deviation),
                        "event_id": event_id,
                        "market_count": len(event_markets),
                    },
                )
            )

        return signals

    def _violation_to_signal(self, violation: PricingViolation) -> Signal:
        """Convert a pricing violation to a Layer 5 signal."""
        return Signal(
            layer=5,
            market_condition_id=violation.source_condition_id,
            token_id="",  # Needs resolution at execution time
            direction=SignalDirection.BUY_YES,
            edge=violation.violation_pct,
            confidence=Decimal("0.75"),
            details={
                "relation_type": violation.relation_type.value,
                "violation_pct": str(violation.violation_pct),
                "source_price": str(violation.source_price),
                "target_price": str(violation.target_price),
                "target_condition_id": violation.target_condition_id,
                "reasoning": violation.reasoning,
            },
        )

    def _get_market_price(self, condition_id: str) -> Decimal | None:
        """Get current YES price for a market by condition_id."""
        if not self._discovery:
            return None

        market = self._discovery.get_by_condition_id(condition_id)
        if market is None:
            return None

        return market.price_yes

    async def _confirm_via_llm(self, rel: MarketRelationship, reasoning: str) -> bool:
        """Confirm pricing violation via LLM analysis."""
        if not self._gemini:
            return True

        try:
            prediction = await self._gemini.predict(
                question=(
                    f"Confirm this pricing violation:\n"
                    f"Market A: {rel.source_question}\n"
                    f"Market B: {rel.target_question}\n"
                    f"Relationship: {rel.relation_type.value}\n"
                    f"Analysis: {reasoning}\n\n"
                    f"Is this a genuine pricing inconsistency? "
                    f"Reply with probability 1.0 for yes, 0.0 for no."
                ),
                current_price=0.5,
                category="logical_arb",
            )

            return bool(prediction and prediction.probability > 0.5)

        except Exception as e:
            logger.debug("llm_confirm_error", error=str(e))
            return True  # Default to accepting if LLM fails
