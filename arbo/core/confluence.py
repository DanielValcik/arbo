"""Confluence scoring engine — central decision mechanism (PM-209).

Aggregates signals from all 9 layers, computes confluence scores per market,
sizes positions based on score, and integrates with RiskManager for pre-trade
validation.

Scoring table (from brief):
  Whale buys position (Layer 4)     → +1
  Value model edge > 5% (Layer 2)   → +1
  News/sentiment event (Layer 8)    → +1
  Order flow spike (Layer 7)        → +1
  Logical inconsistency (Layer 5)   → +1

Execution:
  Score 0-1: NO TRADE
  Score 2:   Standard size (2.5% capital)
  Score 3+:  Double size (5% capital — hard cap)

See brief Section 4 for full specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from arbo.config.settings import get_config
from arbo.core.risk_manager import RiskManager, TradeRequest
from arbo.core.scanner import Signal, SignalDirection
from arbo.utils.logger import get_logger

logger = get_logger("confluence")

# Edge threshold for Layer 2 to count as a contributing signal
VALUE_EDGE_THRESHOLD = Decimal("0.05")

# Layers that contribute to confluence score
SCORING_LAYERS = {2, 4, 5, 7, 8}


@dataclass
class ScoredOpportunity:
    """A market opportunity with confluence score and position sizing.

    Attributes:
        market_condition_id: Polymarket condition ID.
        token_id: CLOB token ID to trade.
        direction: Recommended trade direction.
        score: Confluence score (0-5).
        signals: Contributing signals.
        contributing_layers: Set of layer numbers that contributed.
        position_size_pct: Position size as percentage of capital.
        recommended_size: Absolute USDC position size.
        best_edge: Highest edge among contributing signals.
        scored_at: UTC timestamp.
    """

    market_condition_id: str
    token_id: str
    direction: SignalDirection
    score: int
    signals: list[Signal]
    contributing_layers: set[int]
    position_size_pct: Decimal
    recommended_size: Decimal
    best_edge: Decimal
    scored_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class ConfluenceScorer:
    """Central confluence scoring engine.

    Aggregates signals from all layers, computes confluence scores,
    determines position sizes, and validates through RiskManager.

    Replaces OpportunityScanner.compute_confluence() as the authoritative scorer.
    """

    def __init__(self, risk_manager: RiskManager, capital: Decimal) -> None:
        self._risk_manager = risk_manager
        self._capital = capital
        config = get_config()
        self._min_score = config.confluence.min_score
        self._standard_size_pct = Decimal(str(config.confluence.standard_size_pct))
        self._double_size_pct = Decimal(str(config.confluence.double_size_pct))
        self._total_scored = 0
        self._total_tradeable = 0
        self._total_rejected = 0

    def score_signals(self, signals: list[Signal]) -> list[ScoredOpportunity]:
        """Score a batch of signals by grouping per market and computing confluence.

        Args:
            signals: Raw signals from all layers.

        Returns:
            List of scored opportunities (including score 0-1 which are no-trade).
        """
        grouped = self._group_by_market(signals)
        opportunities: list[ScoredOpportunity] = []

        for market_id, market_signals in grouped.items():
            score, contributing = self._compute_score(market_signals)
            direction = self._determine_direction(market_signals)
            size_pct = self._compute_position_size(score)
            recommended_size = (self._capital * size_pct).quantize(Decimal("0.01"))

            # Find best edge among signals
            best_edge = max((s.edge for s in market_signals), default=Decimal("0"))

            # Pick token_id from first signal
            token_id = market_signals[0].token_id

            opp = ScoredOpportunity(
                market_condition_id=market_id,
                token_id=token_id,
                direction=direction,
                score=score,
                signals=market_signals,
                contributing_layers=contributing,
                position_size_pct=size_pct,
                recommended_size=recommended_size,
                best_edge=best_edge,
            )
            opportunities.append(opp)
            self._total_scored += 1

        logger.info(
            "confluence_scored",
            markets=len(opportunities),
            by_score={str(s): sum(1 for o in opportunities if o.score == s) for s in range(6)},
        )

        return opportunities

    def get_tradeable(
        self,
        signals: list[Signal],
        market_category_map: dict[str, str] | None = None,
    ) -> list[ScoredOpportunity]:
        """Full pipeline: score → filter → risk check.

        Args:
            signals: Raw signals from all layers.
            market_category_map: Mapping of condition_id → category for risk checks.

        Returns:
            List of risk-approved, tradeable opportunities (score ≥ min_score).
        """
        if market_category_map is None:
            market_category_map = {}

        opportunities = self.score_signals(signals)

        # Build price map from signal details
        price_map: dict[str, Decimal] = {}
        for sig in signals:
            if sig.market_condition_id not in price_map:
                poly_price = sig.details.get("poly_price")
                if poly_price is not None:
                    price_map[sig.market_condition_id] = Decimal(str(poly_price))

        tradeable: list[ScoredOpportunity] = []
        for opp in opportunities:
            if opp.score < self._min_score:
                continue

            # Tag diagnostic mode for score-1 trades (min_score temporarily lowered)
            diagnostic_mode = opp.score == 1
            if diagnostic_mode:
                logger.info(
                    "confluence_tradeable_diagnostic",
                    market_id=opp.market_condition_id,
                    score=opp.score,
                    layers=sorted(opp.contributing_layers),
                    edge=str(opp.best_edge),
                )

            category = market_category_map.get(opp.market_condition_id, "other")
            price = price_map.get(opp.market_condition_id)
            checked = self._apply_risk_check(opp, category, price=price)
            if checked is not None:
                tradeable.append(checked)
                self._total_tradeable += 1
            else:
                self._total_rejected += 1

        logger.info(
            "confluence_tradeable",
            scored=len(opportunities),
            tradeable=len(tradeable),
            rejected=self._total_rejected,
        )

        return tradeable

    @property
    def stats(self) -> dict[str, Any]:
        """Get confluence scorer statistics."""
        return {
            "total_scored": self._total_scored,
            "total_tradeable": self._total_tradeable,
            "total_rejected": self._total_rejected,
        }

    def _group_by_market(self, signals: list[Signal]) -> dict[str, list[Signal]]:
        """Group signals by market_condition_id."""
        grouped: dict[str, list[Signal]] = {}
        for sig in signals:
            if sig.market_condition_id not in grouped:
                grouped[sig.market_condition_id] = []
            grouped[sig.market_condition_id].append(sig)
        return grouped

    def _compute_score(self, signals: list[Signal]) -> tuple[int, set[int]]:
        """Compute confluence score for a set of signals from one market.

        Each SCORING_LAYER counts at most once. Layer 2 only counts if edge > 5%.
        Maximum score = 5 (one per scoring layer).

        Returns:
            Tuple of (score, set of contributing layers).
        """
        contributing: set[int] = set()

        for sig in signals:
            if sig.layer not in SCORING_LAYERS:
                continue

            # Layer 2: only counts if edge > 5%
            if sig.layer == 2 and sig.edge < VALUE_EDGE_THRESHOLD:
                continue

            contributing.add(sig.layer)

        score = min(len(contributing), 5)
        return score, contributing

    def _determine_direction(self, signals: list[Signal]) -> SignalDirection:
        """Determine trade direction from majority of contributing signals.

        Counts BUY_YES + SELL_NO as one direction, BUY_NO + SELL_YES as other.
        Returns the majority direction.
        """
        buy_yes_count = 0
        buy_no_count = 0

        for sig in signals:
            if sig.direction in (SignalDirection.BUY_YES, SignalDirection.SELL_NO):
                buy_yes_count += 1
            else:
                buy_no_count += 1

        return SignalDirection.BUY_YES if buy_yes_count >= buy_no_count else SignalDirection.BUY_NO

    def _compute_position_size(self, score: int) -> Decimal:
        """Compute position size percentage based on confluence score.

        Score 0-1: 0% (NO TRADE)
        Score 2:   standard_size_pct (2.5%)
        Score 3+:  double_size_pct (5% hard cap)
        """
        if score < self._min_score:
            return Decimal("0")
        elif score == self._min_score:
            return self._standard_size_pct
        else:
            return self._double_size_pct

    def _apply_risk_check(
        self,
        opp: ScoredOpportunity,
        category: str,
        price: Decimal | None = None,
    ) -> ScoredOpportunity | None:
        """Validate opportunity through RiskManager.pre_trade_check().

        Args:
            opp: Scored opportunity to validate.
            category: Market category for concentration checks.
            price: Real market price. Falls back to signal details or best_edge estimate.

        Returns:
            The opportunity if approved, None if rejected.
        """
        side = (
            "BUY" if opp.direction in (SignalDirection.BUY_YES, SignalDirection.BUY_NO) else "SELL"
        )

        # Resolve real price: explicit > signal details > estimate from edge
        if price is None and opp.signals:
            poly_price = opp.signals[0].details.get("poly_price")
            if poly_price is not None:
                price = Decimal(str(poly_price))
        if price is None:
            # Last resort: estimate from best_edge (still better than hardcoded 0.50)
            price = max(Decimal("0.01"), Decimal("0.50") - opp.best_edge)

        request = TradeRequest(
            market_id=opp.market_condition_id,
            token_id=opp.token_id,
            side=side,
            price=price,
            size=opp.recommended_size,
            layer=min(opp.contributing_layers) if opp.contributing_layers else 0,
            market_category=category,
            confluence_score=opp.score,
        )

        decision = self._risk_manager.pre_trade_check(request)

        if decision.approved:
            logger.info(
                "confluence_approved",
                market_id=opp.market_condition_id,
                score=opp.score,
                layers=sorted(opp.contributing_layers),
                size=str(opp.recommended_size),
            )
            return opp

        logger.info(
            "confluence_rejected",
            market_id=opp.market_condition_id,
            score=opp.score,
            reason=decision.reason,
        )
        return None
