"""Crypto Price Scanner — Discover and scan crypto price prediction markets.

Scans Polymarket crypto price markets ("Will BTC be above $X on date?"),
estimates probability using the volatility model, and produces trading
signals with calculated edge.

Analogous to weather_scanner.py but for crypto price markets.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from arbo.connectors.market_discovery import CryptoMarketInfo, categorize_crypto_market
from arbo.models.volatility_model import VolatilityEstimator, estimate_crypto_prob
from arbo.utils.logger import get_logger

logger = get_logger("crypto_price_scanner")


@dataclass
class CryptoSignal:
    """A crypto price trading signal."""

    # Market info
    condition_id: str
    question: str
    token_id: str
    token_id_no: str | None
    asset: str  # BTC, ETH
    symbol: str  # BTCUSDT
    strike: Decimal
    direction: str  # "above" or "below"
    market_type: str  # "daily_above" or "monthly_hit"
    expiry: datetime | None
    neg_risk: bool  # Always False for crypto
    fee_enabled: bool
    volume_24h: float
    liquidity: float

    # Signal
    model_prob: float
    edge: float
    current_exchange_price: float
    market_price: float  # Polymarket YES price
    sigma_hourly: float
    hours_to_expiry: float

    @property
    def signal_direction(self) -> str:
        """BUY_YES if model_prob > market_price, BUY_NO otherwise."""
        return "BUY_YES" if self.edge > 0 else "BUY_NO"


def parse_crypto_market(
    market: Any,
) -> tuple[CryptoMarketInfo | None, str, str | None, str | None, bool, float, float]:
    """Parse a GammaMarket into CryptoMarketInfo + metadata.

    Returns:
        (info, question, token_id_yes, token_id_no, fee_enabled, volume, liquidity)
        or (None, ...) if not parseable.
    """
    question = market.question
    info = categorize_crypto_market(question)
    if info is None:
        return None, question, None, None, False, 0, 0

    token_id_yes = market.clob_token_ids[0] if market.clob_token_ids else None
    token_id_no = market.clob_token_ids[1] if len(market.clob_token_ids) > 1 else None
    fee_enabled = market.fee_enabled
    volume = float(market.raw.get("volume24hr", 0) or 0) if hasattr(market, "raw") else 0
    liquidity = float(market.raw.get("liquidityClob", 0) or 0) if hasattr(market, "raw") else 0

    return info, question, token_id_yes, token_id_no, fee_enabled, volume, liquidity


def classify_market_type(question: str) -> str:
    """Classify market question as daily_above or monthly_hit."""
    q = question.lower()
    if "what price" in q and "hit" in q:
        return "monthly_hit"
    return "daily_above"


def compute_hours_to_expiry(expiry: datetime | None) -> float:
    """Compute hours until market resolution."""
    if expiry is None:
        return 24.0  # Default assumption
    now = datetime.now(timezone.utc)
    # Ensure expiry is timezone-aware
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    delta = expiry - now
    hours = delta.total_seconds() / 3600
    return max(hours, 0.01)


def scan_crypto_markets(
    markets: list[Any],
    exchange_prices: dict[str, float],
    vol_estimator: VolatilityEstimator,
    min_edge: float = 0.01,
    current_time: float = 0.0,
) -> list[CryptoSignal]:
    """Scan all crypto price markets and produce trading signals.

    Args:
        markets: List of GammaMarket objects (from market discovery).
        exchange_prices: Current exchange prices {symbol → price}.
        vol_estimator: Volatility estimator with loaded price history.
        min_edge: Minimum absolute edge to include in results.
        current_time: Current time for sigma cache.

    Returns:
        List of CryptoSignal sorted by absolute edge (descending).
    """
    signals: list[CryptoSignal] = []

    for market in markets:
        # Only process crypto markets
        if not hasattr(market, "category") or market.category != "crypto":
            continue

        info, question, token_id_yes, token_id_no, fee_enabled, volume, liquidity = (
            parse_crypto_market(market)
        )
        if info is None or token_id_yes is None:
            continue

        # Skip 5-minute markets (too fast for our strategy)
        if info.is_5min:
            continue

        # ── MARKET TYPE FILTER ──
        # Trade: daily "above" + daily "hit/reach" (barrier options)
        # Both have /price endpoint liquidity (RFQM, not orderbook)
        # Skip: weekly "Will the price of...", up/down, between, dip
        q_lower = question.lower()

        # Reject wrong market types
        if any(kw in q_lower for kw in [
            "dip to", "between", "up or down",
            "volatility index",  # Not a price market
        ]):
            continue

        # Must be "above/below" or "reach/hit"
        is_above = "above" in q_lower or "below" in q_lower
        is_hit = "reach" in q_lower or "hit" in q_lower
        if not is_above and not is_hit:
            continue

        # ── EXCHANGE PRICE + ATM FILTER ──
        exchange_price = exchange_prices.get(info.symbol)
        if exchange_price is None or exchange_price <= 0:
            continue

        # ATM filter: only trade strikes within ±5% of current price
        strike_val = float(info.strike)
        if strike_val <= 0:
            continue
        distance_pct = abs(strike_val - exchange_price) / exchange_price
        if distance_pct > 0.05:
            continue  # Too far from money — OTM/deep ITM, thin books

        # Compute volatility
        sigma_hourly = vol_estimator.get_sigma(info.symbol, current_time)

        # Compute hours to expiry
        hours_to_expiry = compute_hours_to_expiry(info.expiry)

        market_type = "monthly_hit" if is_hit else "daily_above"

        # Compute model probability
        model_prob = estimate_crypto_prob(
            current_price=exchange_price,
            strike=float(info.strike),
            hours_to_expiry=hours_to_expiry,
            sigma_per_hour=sigma_hourly,
            market_type=market_type,
            direction=info.direction,
        )

        # Get market price (Polymarket YES price)
        market_price = float(market.price_yes) if hasattr(market, "price_yes") else 0
        if market_price <= 0 or market_price >= 1:
            continue

        # Calculate edge
        edge = model_prob - market_price

        # Filter by minimum absolute edge
        if abs(edge) < min_edge:
            continue

        signal = CryptoSignal(
            condition_id=market.condition_id,
            question=question,
            token_id=token_id_yes,
            token_id_no=token_id_no,
            asset=info.asset,
            symbol=info.symbol,
            strike=info.strike,
            direction=info.direction,
            market_type=market_type,
            expiry=info.expiry,
            neg_risk=False,  # Crypto markets are NOT NegRisk
            fee_enabled=fee_enabled,
            volume_24h=volume,
            liquidity=liquidity,
            model_prob=model_prob,
            edge=edge,
            current_exchange_price=exchange_price,
            market_price=market_price,
            sigma_hourly=sigma_hourly,
            hours_to_expiry=hours_to_expiry,
        )
        signals.append(signal)

    # Sort by absolute edge descending
    signals.sort(key=lambda s: abs(s.edge), reverse=True)

    if signals:
        logger.info(
            "crypto_scan_complete",
            total_signals=len(signals),
            top_edge=f"{signals[0].edge:.3f}" if signals else "0",
            assets=list(set(s.asset for s in signals)),
        )

    return signals
