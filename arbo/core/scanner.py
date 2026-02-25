"""Opportunity scanner and signal DTOs.

Produces unified Signal objects for the paper trading engine.
Strategy-specific subtypes (ThetaDecaySignal, etc.) carry extra
fields without breaking the universal Signal interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from arbo.connectors.market_discovery import GammaMarket  # noqa: TC001 — used in method signatures
from arbo.core.fee_model import calculate_taker_fee, is_fee_favorable
from arbo.utils.logger import get_logger

logger = get_logger("scanner")


class SignalDirection(Enum):
    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"
    SELL_YES = "SELL_YES"
    SELL_NO = "SELL_NO"


@dataclass
class Signal:
    """Unified trading signal from any strategy.

    Attributes:
        layer: Legacy layer number (1-9). Kept for backward compatibility.
        strategy: Strategy identifier ("A", "B", "C", or "" for legacy).
        market_condition_id: Polymarket condition ID.
        token_id: CLOB token ID to trade.
        direction: Trade direction (BUY_YES, BUY_NO, etc.).
        edge: Estimated edge as decimal (e.g. 0.05 = 5%).
        confidence: Confidence score 0-1.
        confluence_score: Number of sources confirming (0-5).
        details: Extra context per strategy.
        detected_at: UTC timestamp.
    """

    layer: int
    market_condition_id: str
    token_id: str
    direction: SignalDirection
    edge: Decimal
    confidence: Decimal
    strategy: str = ""
    confluence_score: int = 0
    details: dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_db_dict(self) -> dict[str, Any]:
        """Convert to dict for DB insertion."""
        return {
            "layer": self.layer,
            "strategy": self.strategy,
            "market_condition_id": self.market_condition_id,
            "direction": self.direction.value,
            "edge": float(self.edge),
            "confidence": float(self.confidence),
            "confluence_score": self.confluence_score,
            "details": self.details,
            "detected_at": self.detected_at,
        }


@dataclass
class ThetaDecaySignal(Signal):
    """Strategy A signal: theta decay on longshot markets.

    Extends Signal with peak optimism detection data.
    """

    z_score: float = 0.0
    taker_ratio: float = 0.0

    def to_db_dict(self) -> dict[str, Any]:
        """Convert to dict with theta-specific fields in details."""
        d = super().to_db_dict()
        d["details"] = {
            **d.get("details", {}),
            "z_score": self.z_score,
            "taker_ratio": self.taker_ratio,
        }
        return d


class OpportunityScanner:
    """Scans markets for trading opportunities across all layers.

    Active layers in Sprint 1:
    - Layer 1 (MM): Wide spread markets with sufficient volume
    - Layer 2 (Value): Pinnacle vs Polymarket divergence (placeholder until PM-003)
    - Layer 3 (Arb): NegRisk markets with pricing violations

    Placeholder layers (Sprint 2+):
    - Layer 4 (Whale): Top wallet tracking
    - Layer 6 (Crypto): 15-min crypto market arbitrage
    """

    def __init__(
        self,
        mm_min_spread: Decimal = Decimal("0.04"),
        mm_min_volume_24h: Decimal = Decimal("1000"),
        mm_max_volume_24h: Decimal = Decimal("50000"),
        negrisk_sum_threshold: Decimal = Decimal("0.03"),
        value_edge_threshold: Decimal = Decimal("0.03"),
    ) -> None:
        self._mm_min_spread = mm_min_spread
        self._mm_min_volume_24h = mm_min_volume_24h
        self._mm_max_volume_24h = mm_max_volume_24h
        self._negrisk_sum_threshold = negrisk_sum_threshold
        self._value_edge_threshold = value_edge_threshold

    def scan_all(self, markets: list[GammaMarket]) -> list[Signal]:
        """Run all active layer scans on given markets.

        Args:
            markets: List of discovered markets to scan.

        Returns:
            Deduplicated list of signals from all layers.
        """
        signals: list[Signal] = []
        signals.extend(self.scan_layer1_mm(markets))
        signals.extend(self.scan_layer3_arb(markets))
        signals.extend(self.scan_layer6_crypto(markets))

        # Build source breakdown for logging
        source_counts: dict[str, int] = {}
        for s in signals:
            key = s.strategy if s.strategy else f"L{s.layer}"
            source_counts[key] = source_counts.get(key, 0) + 1

        logger.info(
            "scan_complete",
            total_markets=len(markets),
            signals_found=len(signals),
            by_source=source_counts if signals else {},
        )

        return signals

    def scan_layer1_mm(self, markets: list[GammaMarket]) -> list[Signal]:
        """Layer 1: Market Making — find wide-spread markets.

        Criteria:
        - Spread > min_spread (default 4%)
        - Volume $1K-$50K/day (sweet spot for retail MM)
        - Prefer fee-enabled markets (maker rebates)

        Returns:
            List of Layer 1 signals.
        """
        signals: list[Signal] = []

        for market in markets:
            spread = market.spread
            if spread is None:
                continue

            if spread < self._mm_min_spread:
                continue

            if market.volume_24h < self._mm_min_volume_24h:
                continue

            if market.volume_24h > self._mm_max_volume_24h:
                continue

            if not market.token_id_yes:
                continue

            # Edge is half the spread (capturing both sides)
            edge = spread / 2
            confidence = Decimal("0.6")  # MM confidence is moderate

            # Boost confidence for fee-enabled markets (maker rebates)
            if market.fee_enabled:
                confidence = Decimal("0.7")

            signals.append(
                Signal(
                    layer=1,
                    market_condition_id=market.condition_id,
                    token_id=market.token_id_yes,
                    direction=SignalDirection.BUY_YES,
                    edge=edge,
                    confidence=confidence,
                    details={
                        "spread": str(spread),
                        "volume_24h": str(market.volume_24h),
                        "fee_enabled": market.fee_enabled,
                        "question": market.question[:100],
                    },
                )
            )

        logger.debug("layer1_mm_scan", candidates=len(signals))
        return signals

    def scan_layer2_value(
        self,
        markets: list[GammaMarket],
        pinnacle_odds: dict[str, Decimal] | None = None,
    ) -> list[Signal]:
        """Layer 2: Value Betting — Pinnacle vs Polymarket divergence.

        Requires PM-003 (The Odds API) for Pinnacle odds.
        Currently a placeholder that logs when called without odds data.

        Args:
            markets: Markets to scan.
            pinnacle_odds: Mapping of condition_id → Pinnacle implied probability.

        Returns:
            List of Layer 2 signals.
        """
        if pinnacle_odds is None:
            logger.debug("layer2_value_scan_skipped", reason="no pinnacle odds available")
            return []

        signals: list[Signal] = []

        for market in markets:
            if market.condition_id not in pinnacle_odds:
                continue

            if market.price_yes is None or not market.token_id_yes:
                continue

            pinnacle_prob = pinnacle_odds[market.condition_id]
            poly_price = market.price_yes

            # Calculate edge after fee
            fee = calculate_taker_fee(poly_price, market.fee_enabled)
            raw_edge = abs(pinnacle_prob - poly_price)
            edge = raw_edge - fee

            if edge < self._value_edge_threshold:
                continue

            # Direction: if pinnacle says higher than poly, BUY YES
            if pinnacle_prob > poly_price:
                direction = SignalDirection.BUY_YES
                token_id = market.token_id_yes
            else:
                direction = SignalDirection.BUY_NO
                token_id = market.token_id_no or market.token_id_yes

            signals.append(
                Signal(
                    layer=2,
                    market_condition_id=market.condition_id,
                    token_id=token_id,
                    direction=direction,
                    edge=edge,
                    confidence=min(Decimal("0.9"), Decimal("0.5") + edge * 5),
                    details={
                        "pinnacle_prob": str(pinnacle_prob),
                        "poly_price": str(poly_price),
                        "raw_edge": str(raw_edge),
                        "fee": str(fee),
                        "question": market.question[:100],
                    },
                )
            )

        logger.debug("layer2_value_scan", candidates=len(signals))
        return signals

    def scan_layer3_arb(self, markets: list[GammaMarket]) -> list[Signal]:
        """Layer 3: Logical Arbitrage — NegRisk pricing violations.

        Finds NegRisk multi-outcome markets where sum of YES prices
        deviates from $1.00 by more than threshold (default 3%).

        Returns:
            List of Layer 3 signals.
        """
        signals: list[Signal] = []

        for market in markets:
            if not market.neg_risk:
                continue

            yes_price = market.price_yes
            no_price = market.price_no

            if yes_price is None or no_price is None:
                continue

            price_sum = yes_price + no_price
            deviation = abs(Decimal("1") - price_sum)

            if deviation < self._negrisk_sum_threshold:
                continue

            if not market.token_id_yes:
                continue

            # If sum < 0.97, buy all outcomes (arb profit)
            # If sum > 1.03, sell all outcomes
            if price_sum < Decimal("1"):
                direction = SignalDirection.BUY_YES
                edge = Decimal("1") - price_sum
            else:
                direction = SignalDirection.SELL_YES
                edge = price_sum - Decimal("1")

            signals.append(
                Signal(
                    layer=3,
                    market_condition_id=market.condition_id,
                    token_id=market.token_id_yes,
                    direction=direction,
                    edge=edge,
                    confidence=Decimal("0.85"),  # Arb is high confidence
                    details={
                        "price_sum": str(price_sum),
                        "deviation": str(deviation),
                        "neg_risk": True,
                        "question": market.question[:100],
                    },
                )
            )

        logger.debug("layer3_arb_scan", candidates=len(signals))
        return signals

    def scan_layer6_crypto(self, markets: list[GammaMarket]) -> list[Signal]:
        """Layer 6: Temporal Crypto Arb — 15-min crypto markets.

        Discovers 15-minute crypto resolution markets. Full implementation
        requires real-time spot price feed (Sprint 2+). Currently detects
        candidate markets and checks fee favorability.

        Returns:
            List of Layer 6 candidate signals.
        """
        signals: list[Signal] = []

        for market in markets:
            if market.category != "crypto":
                continue

            if not market.fee_enabled:
                continue

            # Look for short-duration markets
            q_lower = market.question.lower()
            if not ("15" in q_lower or "minute" in q_lower or "hour" in q_lower):
                continue

            if market.price_yes is None or not market.token_id_yes:
                continue

            # Check fee favorability (need extreme prices for latency arb)
            if not is_fee_favorable(market.price_yes, market.fee_enabled):
                continue

            signals.append(
                Signal(
                    layer=6,
                    market_condition_id=market.condition_id,
                    token_id=market.token_id_yes,
                    direction=SignalDirection.BUY_YES,
                    edge=Decimal("0"),  # Unknown until spot price comparison
                    confidence=Decimal("0.3"),  # Low without spot data
                    details={
                        "price_yes": str(market.price_yes),
                        "fee_favorable": True,
                        "question": market.question[:100],
                    },
                )
            )

        logger.debug("layer6_crypto_scan", candidates=len(signals))
        return signals

    def compute_confluence(self, signals: list[Signal]) -> list[Signal]:
        """Compute confluence scores across sources for same market.

        Markets with signals from multiple layers/strategies get higher scores.
        Score = number of distinct sources detecting an opportunity.

        Args:
            signals: Raw signals from all sources.

        Returns:
            Same signals with updated confluence_score.
        """
        # Group by market condition ID — count distinct sources (layer or strategy)
        market_sources: dict[str, set[str]] = {}
        for sig in signals:
            mid = sig.market_condition_id
            if mid not in market_sources:
                market_sources[mid] = set()
            # Use strategy as source key if set, else fall back to layer number
            source_key = sig.strategy if sig.strategy else f"L{sig.layer}"
            market_sources[mid].add(source_key)

        # Update confluence scores
        for sig in signals:
            sig.confluence_score = len(market_sources[sig.market_condition_id])

        return signals
