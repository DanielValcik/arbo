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
# THRESHOLDS — from autoresearch v5 (2026-03-27, BTC+ETH)
# Score=155.9, 902 trades, WR=84.7%, PnL=$27,034, Sharpe=7.95, DD=4.4%
# Dataset: 3,745 labeled markets (1,825 BTC + 1,920 ETH), 87 days
# Fee model: maker entry 0%, taker exit fee = price*(1-price)*0.25
# OOS: $1,229 avg PnL across 3 walk-forward folds
# ═══════════════════════════════════════════════════════════════════════════════

# Entry thresholds
MIN_EDGE = 0.08
MAX_EDGE = 0.90
MIN_PRICE = 0.05
MAX_PRICE = 0.60
MIN_VOLUME_24H = 0.0    # PMD doesn't have volume; live uses liquidity check
MIN_LIQUIDITY = 0.0      # Gamma API doesn't always provide liquidity; crypto markets have deep books
MIN_TIME_TO_EXPIRY_H = 8
MAX_TIME_TO_EXPIRY_H = 168  # 7 days max

# Trading hours (UTC) — skip night sessions with poor WR
# Live data: 01:00 UTC = 33% WR, 22:00 = 50% WR vs 14:00-19:00 = 100% WR
TRADING_HOURS_UTC_START = 7   # 7:00 UTC (3 AM ET)
TRADING_HOURS_UTC_END = 21    # 21:00 UTC (5 PM ET)

# Volatility model
VOLATILITY_WINDOW = 168      # 168 hours (7 days) of price history for sigma
VOLATILITY_METHOD = "ewma"
SIGMA_SCALE = 0.8

# Exit thresholds
EXIT_ENABLED = True
MIN_HOLD_EDGE = 0.03         # Exit when edge drops below 3%
EXIT_SLIPPAGE_PCT = 0.01     # 1% slippage (tight spreads on crypto)
PROB_EXIT_FLOOR = 0.0        # Disabled
PROFIT_TAKE_ALSO = False     # No profit take — hold to resolution
PROFIT_TARGET_ABS = 0.30     # Only if PROFIT_TAKE_ALSO re-enabled

# Sizing
KELLY_RAW_CAP = 0.30         # Conservative — autoresearch showed 0.40-0.50 same WR but more risk
PROB_SHARPENING = 1.50       # Best single improvement from autoresearch v6 (+2.1 score)
SHRINKAGE = 0.02
MAX_AGGREGATE_PCT = 0.70     # Conservative — keep baseline risk level
MAX_POSITION_PCT = 0.08      # 8% per single position
REENTRY_COOLDOWN_H = 0       # No cooldown

# Exchange price freshness
MAX_EXCHANGE_PRICE_AGE_S = 30.0  # Max 30 seconds stale

# Excluded assets (insufficient liquidity)
EXCLUDED_ASSETS: set[str] = {"SOL", "XRP", "DOGE", "ADA", "BNB"}

# Per-asset overrides — none needed, both BTC and ETH profitable
ASSET_OVERRIDES: dict[str, dict[str, float]] = {}


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
    *,
    params: dict | None = None,
) -> QualityDecision:
    """Check if a crypto signal passes the quality gate.

    Args:
        signal: CryptoSignal from crypto_price_scanner.
        exchange_price_age_s: Age of the exchange price in seconds.
        params: Optional override dict; falls back to module constants per
            key. Used by Project PARALLEL variant evaluation.

    Returns:
        QualityDecision with pass/fail and reason.
    """
    p = params or {}
    min_edge_eff = p.get("MIN_EDGE", MIN_EDGE)
    max_edge_eff = p.get("MAX_EDGE", MAX_EDGE)
    min_price_eff = p.get("MIN_PRICE", MIN_PRICE)
    max_price_eff = p.get("MAX_PRICE", MAX_PRICE)
    min_t_eff = p.get("MIN_TIME_TO_EXPIRY_H", MIN_TIME_TO_EXPIRY_H)
    max_t_eff = p.get("MAX_TIME_TO_EXPIRY_H", MAX_TIME_TO_EXPIRY_H)

    asset = signal.asset

    # 1. Asset exclusion
    if asset in EXCLUDED_ASSETS:
        return QualityDecision(False, f"excluded_asset:{asset}")

    # 1b. Day/Night regime — different parameters for different sessions
    # Live data: day (07-21 UTC) = 93% WR, night (21-07 UTC) = 44% WR
    from datetime import datetime, timezone
    now_utc = datetime.now(timezone.utc).hour
    is_night = now_utc < TRADING_HOURS_UTC_START or now_utc >= TRADING_HOURS_UTC_END

    # Night: require higher edge + smaller position (compensate for lower WR)
    if is_night:
        min_edge = max(_get_threshold("min_edge", asset, min_edge_eff), 0.15)  # 15% min at night (vs 8% day)
    else:
        min_edge = _get_threshold("min_edge", asset, min_edge_eff)

    # 2. Edge range (min_edge already set above with day/night adjustment)
    edge = abs(signal.edge)
    if edge < min_edge:
        return QualityDecision(False, f"edge_too_low:{edge:.3f}<{min_edge}")
    if edge > max_edge_eff:
        return QualityDecision(False, f"edge_too_high:{edge:.3f}>{max_edge_eff}")

    # 3. Price range
    min_price = _get_threshold("min_price", asset, min_price_eff)
    max_price = _get_threshold("max_price", asset, max_price_eff)
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
    if signal.hours_to_expiry < min_t_eff:
        return QualityDecision(
            False, f"too_close:{signal.hours_to_expiry:.1f}h<{min_t_eff}h"
        )
    if signal.hours_to_expiry > max_t_eff:
        return QualityDecision(
            False, f"too_far:{signal.hours_to_expiry:.1f}h>{max_t_eff}h"
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


def filter_signals(
    signals: list[Any], *, params: dict | None = None,
) -> list[Any]:
    """Apply quality gate to all signals, return qualified ones.

    Args:
        signals: List of CryptoSignal objects.
        params: Optional override dict for Project PARALLEL variant evaluation.

    Returns:
        Filtered list of signals that pass all quality checks.
    """
    qualified = []
    rejected_reasons: dict[str, int] = {}

    for sig in signals:
        age = 0.0  # In live trading, this comes from BinanceWSFeed.get_price_age()
        decision = check_signal_quality(sig, exchange_price_age_s=age, params=params)

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
