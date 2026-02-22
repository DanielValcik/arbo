"""Value signal generator (PM-102 + PM-404).

Multi-category value betting pipeline:
1. Soccer: Polymarket ↔ Pinnacle odds via EventMatcher + XGBoost soccer model
2. Crypto: Polymarket crypto markets + Binance features + XGBoost crypto model
3. Politics: Polymarket politics markets + GeminiAgent LLM probability

Scan interval: 5 minutes (config: value_model.scan_interval_s).
Acceptance: generates ≥3 valid signals/24h with edge >3%.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from arbo.connectors.event_matcher import EventMatcher, MatchedPair  # noqa: TC001
from arbo.connectors.market_discovery import (
    GammaMarket,
    MarketDiscovery,
    categorize_crypto_market,
)
from arbo.connectors.odds_api_client import OddsApiClient  # noqa: TC001
from arbo.core.fee_model import calculate_taker_fee
from arbo.core.scanner import Signal, SignalDirection
from arbo.models.feature_engineering import MarketFeatures
from arbo.models.xgboost_value import ValueModel  # noqa: TC001
from arbo.utils.logger import get_logger

if TYPE_CHECKING:
    from arbo.agents.gemini_agent import GeminiAgent
    from arbo.connectors.binance_client import BinanceClient
    from arbo.models.xgboost_crypto import CryptoValueModel

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
    """Generates Layer 2 value signals across soccer, crypto, and politics.

    Soccer: Pinnacle-Polymarket divergence via EventMatcher + XGBoost.
    Crypto: Binance features + XGBoost crypto model.
    Politics: GeminiAgent LLM probability estimation.
    """

    def __init__(
        self,
        discovery: MarketDiscovery,
        odds_client: OddsApiClient,
        matcher: EventMatcher,
        value_model: ValueModel | None = None,
        edge_threshold: float = 0.03,
        crypto_model: CryptoValueModel | None = None,
        binance_client: BinanceClient | None = None,
        gemini: GeminiAgent | None = None,
        crypto_edge_threshold: float = 0.03,
        politics_edge_threshold: float = 0.04,
        max_politics_per_scan: int = 10,
    ) -> None:
        self._discovery = discovery
        self._odds_client = odds_client
        self._matcher = matcher
        self._value_model = value_model
        self._edge_threshold = edge_threshold
        self._crypto_model = crypto_model
        self._binance_client = binance_client
        self._gemini = gemini
        self._crypto_edge_threshold = crypto_edge_threshold
        self._politics_edge_threshold = politics_edge_threshold
        self._max_politics_per_scan = max_politics_per_scan
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
        """Run a multi-category value scan cycle.

        Scans soccer, crypto, and politics markets for value signals.

        Returns:
            List of signals with edge > threshold across all categories.
        """
        self._scan_count += 1

        signals: list[Signal] = []
        signals.extend(await self._scan_soccer())
        signals.extend(await self._scan_crypto())
        signals.extend(await self._scan_politics())

        self._total_signals += len(signals)

        logger.info(
            "value_scan_complete",
            scan_number=self._scan_count,
            signals=len(signals),
            total_signals=self._total_signals,
        )

        return signals

    async def _scan_soccer(self) -> list[Signal]:
        """Soccer value scan (original logic).

        1. Get Polymarket soccer markets from discovery cache
        2. Fetch Pinnacle match-level + outright odds
        3. Match markets (both match-level and seasonal)
        4. Evaluate each pair for value signals
        """
        soccer_markets = self._discovery.get_by_category("soccer")
        if not soccer_markets:
            logger.debug("value_scan_no_soccer_markets")
            return []

        match_events = await self._odds_client.get_soccer_odds()
        outright_odds = await self._odds_client.get_all_soccer_outrights()

        match_pairs = self._matcher.match_markets(soccer_markets, match_events)
        seasonal_pairs = self._matcher.match_seasonal_markets(soccer_markets, outright_odds)

        derived_pairs: list[MatchedPair] = []
        if not seasonal_pairs and match_events:
            derived_pairs = self._matcher.match_seasonal_via_match_odds(
                soccer_markets, match_events
            )

        all_pairs = match_pairs + seasonal_pairs + derived_pairs

        if not all_pairs:
            logger.debug(
                "value_scan_no_soccer_pairs",
                soccer_markets=len(soccer_markets),
                match_events=len(match_events),
            )
            return []

        signals: list[Signal] = []
        for pair in all_pairs:
            signal = evaluate_pair(pair, self._edge_threshold, self._value_model)
            if signal is not None:
                signals.append(signal)

        logger.info(
            "soccer_scan_done",
            markets=len(soccer_markets),
            pairs=len(all_pairs),
            signals=len(signals),
        )
        return signals

    async def _scan_crypto(self) -> list[Signal]:
        """Crypto value scan using Binance features + XGBoost crypto model.

        1. Get crypto markets from discovery
        2. Parse each market for asset/strike/expiry
        3. Fetch Binance ticker + features
        4. Predict via crypto model
        5. Emit signal if edge > threshold
        """
        if self._crypto_model is None or not self._crypto_model.is_trained:
            return []
        if self._binance_client is None:
            return []

        crypto_markets = self._discovery.get_crypto_markets()
        if not crypto_markets:
            logger.debug("value_scan_no_crypto_markets")
            return []

        signals: list[Signal] = []
        for market in crypto_markets:
            signal = await self._evaluate_crypto_market(market)
            if signal is not None:
                signals.append(signal)

        logger.info("crypto_scan_done", markets=len(crypto_markets), signals=len(signals))
        return signals

    async def _evaluate_crypto_market(self, market: GammaMarket) -> Signal | None:
        """Evaluate a single crypto market for value signal."""
        info = categorize_crypto_market(market.question)
        if info is None or info.is_5min:
            return None

        if market.price_yes is None or not market.token_id_yes:
            return None

        # Fetch Binance ticker for current spot price
        try:
            ticker = await self._binance_client.get_ticker_24h(info.symbol)  # type: ignore[union-attr]
        except Exception as e:
            logger.debug("crypto_ticker_failed", symbol=info.symbol, error=str(e))
            return None

        spot = ticker.last_price
        strike = float(info.strike)
        if strike <= 0:
            return None

        # Time to expiry
        time_to_expiry: float | None = None
        if market.end_date:
            try:
                end_dt = datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
                delta = end_dt - datetime.now(UTC)
                time_to_expiry = max(0.0, delta.total_seconds() / 3600)
            except (ValueError, TypeError):
                pass

        from arbo.models.crypto_features import (
            CryptoFeatures,
            compute_distance_pct,
            compute_spot_vs_strike,
        )

        features = CryptoFeatures(
            spot_vs_strike=compute_spot_vs_strike(spot, strike),
            time_to_expiry=time_to_expiry,
            volatility_24h=None,  # Would need OHLCV — use ticker approx
            volatility_7d=None,
            volume_24h_log=math.log1p(ticker.quote_volume) if ticker.quote_volume > 0 else None,
            volume_trend=None,
            funding_rate=None,
            rsi_14=None,
            momentum_24h=ticker.price_change_pct / 100 if ticker.price_change_pct else None,
            distance_pct=compute_distance_pct(spot, strike),
            polymarket_mid=float(market.price_yes),
        )

        model_prob = self._crypto_model.predict_single(features)  # type: ignore[union-attr]
        poly_price = market.price_yes
        fee = calculate_taker_fee(poly_price, market.fee_enabled)
        raw_edge = abs(Decimal(str(model_prob)) - poly_price)
        edge = raw_edge - fee

        if edge < Decimal(str(self._crypto_edge_threshold)):
            return None

        if Decimal(str(model_prob)) > poly_price:
            direction = SignalDirection.BUY_YES
            token_id = market.token_id_yes
        else:
            direction = SignalDirection.BUY_NO
            token_id = market.token_id_no or market.token_id_yes

        confidence = min(Decimal("0.9"), Decimal("0.5") + edge * 5)

        return Signal(
            layer=2,
            market_condition_id=market.condition_id,
            token_id=token_id,
            direction=direction,
            edge=edge,
            confidence=confidence,
            details={
                "model_prob": str(model_prob),
                "poly_price": str(poly_price),
                "raw_edge": str(raw_edge),
                "fee": str(fee),
                "spot_price": str(spot),
                "strike_price": str(strike),
                "asset": info.asset,
                "question": market.question[:100],
                "category": "crypto",
                "used_model": True,
            },
        )

    async def _scan_politics(self) -> list[Signal]:
        """Politics value scan using GeminiAgent LLM probability.

        1. Get politics markets from discovery
        2. For each (up to max_politics_per_scan): call gemini.predict()
        3. Emit signal if edge > threshold
        """
        if self._gemini is None:
            return []

        politics_markets = self._discovery.get_politics_markets()
        if not politics_markets:
            logger.debug("value_scan_no_politics_markets")
            return []

        # Cap to max per scan to control LLM costs
        markets_to_scan = politics_markets[: self._max_politics_per_scan]

        signals: list[Signal] = []
        for market in markets_to_scan:
            signal = await self._evaluate_politics_market(market)
            if signal is not None:
                signals.append(signal)

        logger.info(
            "politics_scan_done",
            markets=len(markets_to_scan),
            signals=len(signals),
        )
        return signals

    async def _evaluate_politics_market(self, market: GammaMarket) -> Signal | None:
        """Evaluate a single politics market via GeminiAgent."""
        if market.price_yes is None or not market.token_id_yes:
            return None

        poly_price = market.price_yes

        try:
            prediction = await self._gemini.predict(  # type: ignore[union-attr]
                question=market.question,
                current_price=float(poly_price),
                category="politics",
                volume_24h=float(market.volume_24h),
            )
        except Exception as e:
            logger.debug("politics_llm_failed", error=str(e))
            return None

        if prediction is None:
            return None

        model_prob = Decimal(str(prediction.probability))
        fee = calculate_taker_fee(poly_price, market.fee_enabled)
        raw_edge = abs(model_prob - poly_price)
        edge = raw_edge - fee

        if edge < Decimal(str(self._politics_edge_threshold)):
            return None

        if model_prob > poly_price:
            direction = SignalDirection.BUY_YES
            token_id = market.token_id_yes
        else:
            direction = SignalDirection.BUY_NO
            token_id = market.token_id_no or market.token_id_yes

        confidence = min(
            Decimal("0.9"),
            Decimal(str(prediction.confidence)) * (Decimal("0.5") + edge * 3),
        )

        return Signal(
            layer=2,
            market_condition_id=market.condition_id,
            token_id=token_id,
            direction=direction,
            edge=edge,
            confidence=confidence,
            details={
                "model_prob": str(model_prob),
                "poly_price": str(poly_price),
                "raw_edge": str(raw_edge),
                "fee": str(fee),
                "llm_reasoning": prediction.reasoning[:200],
                "llm_provider": prediction.provider,
                "llm_confidence": str(prediction.confidence),
                "question": market.question[:100],
                "category": "politics",
                "used_model": False,
            },
        )
