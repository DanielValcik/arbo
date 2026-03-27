"""Quality Gate for Strategy B2 — Crypto Price Edge.

Filters crypto price trading signals through autoresearch-optimized
thresholds. Structurally identical to weather_quality_gate_c2.py.

Thresholds below are DEFAULTS — will be updated after autoresearch
sweep produces optimized parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from arbo.utils.logger import get_logger

logger = get_logger("crypto_quality_gate")

# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLDS — from autoresearch v4 (2026-03-27)
# Score=145.4, 409 trades, WR=85.3%, PnL=$11,107, Sharpe=7.75, DD=5.8%
# Dataset: 1,821 BTC markets, 87 days (2025-12-28 → 2026-03-26)
# Fee model: maker entry 0%, taker exit fee = price*(1-price)*0.25
# ═══════════════════════════════════════════════════════════════════════════════

# Entry thresholds
MIN_EDGE = 0.08
MAX_EDGE = 0.50
MIN_PRICE = 0.03
MAX_PRICE = 0.70
MIN_VOLUME_24H = 0.0    # PMD doesn't have volume; live uses liquidity check
MIN_LIQUIDITY = 5000.0   # $5K minimum for live trading
MIN_TIME_TO_EXPIRY_H = 4
MAX_TIME_TO_EXPIRY_H = 168  # 7 days max

# Volatility model
VOLATILITY_WINDOW = 72       # 72 hours of price history for sigma
VOLATILITY_METHOD = "realized"
SIGMA_SCALE = 0.3

# Exit thresholds
EXIT_ENABLED = True
MIN_HOLD_EDGE = 0.02         # Exit when edge drops below 2%
EXIT_SLIPPAGE_PCT = 0.01     # 1% slippage (tight spreads on crypto)
PROB_EXIT_FLOOR = 0.05       # Exit if model prob drops below 5%
PROFIT_TAKE_ALSO = True
PROFIT_TARGET_ABS = 0.20     # Take profit at +$0.20

# Sizing
KELLY_RAW_CAP = 0.20
PROB_SHARPENING = 1.0
SHRINKAGE = 0.0
MAX_AGGREGATE_PCT = 0.80     # Deploy up to 80% of capital
MAX_POSITION_PCT = 0.10      # 10% per single position
REENTRY_COOLDOWN_H = 1       # 1 hour cooldown per token

# Exchange price freshness
MAX_EXCHANGE_PRICE_AGE_S = 30.0  # Max 30 seconds stale

# Excluded assets (insufficient liquidity)
EXCLUDED_ASSETS: set[str] = {"SOL", "XRP", "DOGE", "ADA", "BNB"}

# Per-asset overrides (from autoresearch Gen 2)
ASSET_OVERRIDES: dict[str, dict[str, float]] = {
    "BTC": {
        "min_edge": 0.08,
        "max_price": 0.70,
        "min_price": 0.03,
        "kelly_raw_cap": 0.25,
        "prob_sharpening": 1.05,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY DECISION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class QualityDecision:
    """Result of quality gate check."""

    passed: bool
    reason: str = ""
    details: dict | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL CHECKING
# ═══════════════════════════════════════════════════════════════════════════════


def _get_threshold(param: str, asset: str, default: float) -> float:
    """Get threshold with per-asset override support."""
    override = ASSET_OVERRIDES.get(asset, {})
    return override.get(param, default)


def check_signal_quality(
    signal: Any,
    exchange_price_age_s: float = 0.0,
) -> QualityDecision:
    """Check if a crypto signal passes the quality gate.

    Args:
        signal: CryptoSignal from crypto_price_scanner.
        exchange_price_age_s: Age of the exchange price in seconds.

    Returns:
        QualityDecision with pass/fail and reason.
    """
    asset = signal.asset

    # 1. Asset exclusion
    if asset in EXCLUDED_ASSETS:
        return QualityDecision(False, f"excluded_asset:{asset}")

    # 2. Edge range
    min_edge = _get_threshold("min_edge", asset, MIN_EDGE)
    edge = abs(signal.edge)
    if edge < min_edge:
        return QualityDecision(False, f"edge_too_low:{edge:.3f}<{min_edge}")
    if edge > MAX_EDGE:
        return QualityDecision(False, f"edge_too_high:{edge:.3f}>{MAX_EDGE}")

    # 3. Price range
    min_price = _get_threshold("min_price", asset, MIN_PRICE)
    max_price = _get_threshold("max_price", asset, MAX_PRICE)
    price = signal.market_price
    if price < min_price:
        return QualityDecision(False, f"price_too_low:{price:.3f}<{min_price}")
    if price > max_price:
        return QualityDecision(False, f"price_too_high:{price:.3f}>{max_price}")

    # 4. Volume
    if signal.volume_24h < MIN_VOLUME_24H:
        return QualityDecision(
            False, f"volume_low:{signal.volume_24h:.0f}<{MIN_VOLUME_24H}"
        )

    # 5. Liquidity
    if signal.liquidity < MIN_LIQUIDITY:
        return QualityDecision(
            False, f"liquidity_low:{signal.liquidity:.0f}<{MIN_LIQUIDITY}"
        )

    # 6. Time to expiry
    if signal.hours_to_expiry < MIN_TIME_TO_EXPIRY_H:
        return QualityDecision(
            False, f"too_close:{signal.hours_to_expiry:.1f}h<{MIN_TIME_TO_EXPIRY_H}h"
        )
    if signal.hours_to_expiry > MAX_TIME_TO_EXPIRY_H:
        return QualityDecision(
            False, f"too_far:{signal.hours_to_expiry:.1f}h>{MAX_TIME_TO_EXPIRY_H}h"
        )

    # 7. Exchange price freshness
    if exchange_price_age_s > MAX_EXCHANGE_PRICE_AGE_S:
        return QualityDecision(
            False, f"price_stale:{exchange_price_age_s:.0f}s>{MAX_EXCHANGE_PRICE_AGE_S}s"
        )

    # 8. Model probability sanity
    if signal.model_prob < 0.01 or signal.model_prob > 0.99:
        return QualityDecision(
            False, f"prob_extreme:{signal.model_prob:.3f}"
        )

    return QualityDecision(True)


def filter_signals(signals: list[Any]) -> list[Any]:
    """Apply quality gate to all signals, return qualified ones.

    Args:
        signals: List of CryptoSignal objects.

    Returns:
        Filtered list of signals that pass all quality checks.
    """
    qualified = []
    rejected_reasons: dict[str, int] = {}

    for sig in signals:
        age = 0.0  # In live trading, this comes from BinanceWSFeed.get_price_age()
        decision = check_signal_quality(sig, exchange_price_age_s=age)

        if decision.passed:
            qualified.append(sig)
        else:
            reason = decision.reason.split(":")[0]
            rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1

    if rejected_reasons:
        logger.info(
            "crypto_quality_gate_filtered",
            passed=len(qualified),
            rejected=len(signals) - len(qualified),
            reasons=rejected_reasons,
        )

    return qualified
