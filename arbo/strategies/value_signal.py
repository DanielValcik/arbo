"""Value signal generator (PM-102).

Orchestrates the Layer 2 value betting pipeline:
1. Get Polymarket soccer markets (MarketDiscovery)
2. Get Pinnacle odds — h2h (match-level) + outrights (seasonal)
3. Match markets via EventMatcher
4. For each matched pair: estimate model probability & edge
5. If edge > threshold → generate Signal for paper trading

Scan interval: 5 minutes (config: value_model.scan_interval_s).
Acceptance: generates ≥3 valid signals/24h with edge >3%.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from arbo.connectors.event_matcher import EventMatcher, MatchedPair  # noqa: TC001
from arbo.connectors.market_discovery import GammaMarket, MarketDiscovery  # noqa: TC001
from arbo.connectors.odds_api_client import OddsApiClient  # noqa: TC001
from arbo.core.fee_model import calculate_taker_fee
from arbo.core.scanner import Signal, SignalDirection
from arbo.models.feature_engineering import MarketFeatures
from arbo.models.xgboost_value import ValueModel  # noqa: TC001
from arbo.utils.logger import get_logger

logger = get_logger("value_signal")


def build_market_features(
    pair: MatchedPair,
    market: GammaMarket,
) -> MarketFeatures:
    """Build MarketFeatures from a matched pair for ValueModel input.

    Args:
        pair: Matched Polymarket-Pinnacle pair.
        market: The Polymarket GammaMarket.

    Returns:
        MarketFeatures ready for model prediction.
    """
    pinnacle_prob = float(pair.pinnacle_prob) if pair.pinnacle_prob is not None else None
    poly_mid = float(market.price_yes) if market.price_yes is not None else None

    # Time to event (hours) — from end_date if available
    time_to_event_hours: float | None = None
    if market.end_date:
        try:
            end_dt = datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
            delta = end_dt - datetime.now(UTC)
            time_to_event_hours = max(0.0, delta.total_seconds() / 3600)
        except (ValueError, TypeError):
            pass

    return MarketFeatures(
        pinnacle_prob=pinnacle_prob,
        polymarket_mid=poly_mid,
        time_to_event_hours=time_to_event_hours,
        category=market.category,
        volume_24h=float(market.volume_24h),
        volume_30d_avg=float(market.volume) / 30 if market.volume > 0 else 0.0,
        liquidity=float(market.liquidity),
        spread=float(market.spread) if market.spread is not None else None,
        fee_enabled=market.fee_enabled,
    )


def evaluate_pair(
    pair: MatchedPair,
    edge_threshold: float = 0.03,
    value_model: ValueModel | None = None,
) -> Signal | None:
    """Evaluate a single matched pair for value signal generation.

    If a trained ValueModel is available, uses calibrated model probability.
    Otherwise falls back to raw Pinnacle implied probability.

    Args:
        pair: Matched Polymarket-Pinnacle pair.
        edge_threshold: Minimum edge to emit a signal (PM-101: 0.03).
        value_model: Optional trained XGBoost model.

    Returns:
        Signal if edge exceeds threshold, else None.
    """
    market = pair.polymarket

    if pair.pinnacle_prob is None:
        return None

    if market.price_yes is None or not market.token_id_yes:
        return None

    poly_price = market.price_yes
    pinnacle_prob = pair.pinnacle_prob

    # Determine model probability
    if value_model is not None and value_model.is_trained:
        features = build_market_features(pair, market)
        model_prob = Decimal(str(value_model.predict_single(features)))
    else:
        # Fallback: use Pinnacle implied prob directly (still sharper than Poly)
        model_prob = pinnacle_prob

    # Calculate edge after fee
    fee = calculate_taker_fee(poly_price, market.fee_enabled)
    raw_edge = abs(model_prob - poly_price)
    edge = raw_edge - fee

    if edge < Decimal(str(edge_threshold)):
        return None

    # Direction: model says higher → BUY YES, lower → BUY NO
    if model_prob > poly_price:
        direction = SignalDirection.BUY_YES
        token_id = market.token_id_yes
    else:
        direction = SignalDirection.BUY_NO
        token_id = market.token_id_no or market.token_id_yes

    # Confidence scales with edge magnitude
    confidence = min(Decimal("0.9"), Decimal("0.5") + edge * 5)

    return Signal(
        layer=2,
        market_condition_id=market.condition_id,
        token_id=token_id,
        direction=direction,
        edge=edge,
        confidence=confidence,
        details={
            "pinnacle_prob": str(pinnacle_prob),
            "model_prob": str(model_prob),
            "poly_price": str(poly_price),
            "raw_edge": str(raw_edge),
            "fee": str(fee),
            "match_type": pair.match_type,
            "match_score": round(pair.match_score, 3),
            "question": market.question[:100],
            "outright_team": pair.outright_team,
            "sport_key": pair.sport_key,
            "used_model": value_model is not None and value_model.is_trained,
        },
    )


class ValueSignalGenerator:
    """Generates Layer 2 value signals from Pinnacle-Polymarket divergence.

    Combines:
    - Match-level matching (h2h odds from Odds API)
    - Seasonal matching (outright/futures odds)
    - XGBoost value model predictions (when trained)
    - Edge calculation with fee deduction
    """

    def __init__(
        self,
        discovery: MarketDiscovery,
        odds_client: OddsApiClient,
        matcher: EventMatcher,
        value_model: ValueModel | None = None,
        edge_threshold: float = 0.03,
    ) -> None:
        self._discovery = discovery
        self._odds_client = odds_client
        self._matcher = matcher
        self._value_model = value_model
        self._edge_threshold = edge_threshold
        self._scan_count = 0
        self._total_signals = 0

    @property
    def scan_count(self) -> int:
        """Number of scans completed."""
        return self._scan_count

    @property
    def total_signals(self) -> int:
        """Total signals generated across all scans."""
        return self._total_signals

    async def scan(self) -> list[Signal]:
        """Run a single value scan cycle.

        1. Get Polymarket soccer markets from discovery cache
        2. Fetch Pinnacle match-level + outright odds
        3. Match markets (both match-level and seasonal)
        4. Evaluate each pair for value signals

        Returns:
            List of signals with edge > threshold.
        """
        self._scan_count += 1

        # 1. Get soccer markets
        soccer_markets = self._discovery.get_by_category("soccer")
        if not soccer_markets:
            logger.info("value_scan_no_soccer_markets")
            return []

        # 2. Fetch Pinnacle odds (match-level + outrights)
        match_events = await self._odds_client.get_soccer_odds()
        outright_odds = await self._odds_client.get_all_soccer_outrights()

        # 3. Match markets
        match_pairs = self._matcher.match_markets(soccer_markets, match_events)
        seasonal_pairs = self._matcher.match_seasonal_markets(soccer_markets, outright_odds)

        # Fallback: if no outright odds, derive from match-level events
        # (The Odds API doesn't support league outrights for domestic leagues)
        derived_pairs: list[MatchedPair] = []
        if not seasonal_pairs and match_events:
            derived_pairs = self._matcher.match_seasonal_via_match_odds(
                soccer_markets, match_events
            )

        all_pairs = match_pairs + seasonal_pairs + derived_pairs

        if not all_pairs:
            logger.info(
                "value_scan_no_pairs",
                soccer_markets=len(soccer_markets),
                match_events=len(match_events),
                outright_leagues=len(outright_odds),
            )
            return []

        # 4. Evaluate each pair
        signals: list[Signal] = []
        for pair in all_pairs:
            signal = evaluate_pair(pair, self._edge_threshold, self._value_model)
            if signal is not None:
                signals.append(signal)

        self._total_signals += len(signals)

        logger.info(
            "value_scan_complete",
            scan_number=self._scan_count,
            soccer_markets=len(soccer_markets),
            match_pairs=len(match_pairs),
            seasonal_pairs=len(seasonal_pairs),
            derived_pairs=len(derived_pairs),
            signals=len(signals),
            total_signals=self._total_signals,
        )

        return signals
