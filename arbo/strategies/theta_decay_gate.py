"""Quality gate for Strategy A (Theta Decay).

Validates theta decay signals before execution. Each candidate must pass
all checks: z-score threshold, market criteria, risk allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from arbo.connectors.polygon_flow import PeakOptimismResult
from arbo.core.risk_manager import RiskManager
from arbo.utils.logger import get_logger

logger = get_logger("theta_decay_gate")

STRATEGY_ID = "A"

# Defaults (overridable via config)
ZSCORE_THRESHOLD = 3.0
LONGSHOT_PRICE_MAX = Decimal("0.15")
MIN_VOLUME_24H = Decimal("10000")
MIN_AGE_HOURS = 24
RESOLUTION_DAYS_MIN = 3
RESOLUTION_DAYS_MAX = 30
EXCLUDED_CATEGORIES = frozenset({"crypto"})


@dataclass
class ThetaGateDecision:
    """Result of theta decay quality gate check."""

    passed: bool
    reason: str
    condition_id: str = ""


def check_market_quality(
    market: Any,
    peak: PeakOptimismResult | None = None,
    zscore_threshold: float = ZSCORE_THRESHOLD,
    longshot_max: Decimal = LONGSHOT_PRICE_MAX,
    min_volume: Decimal = MIN_VOLUME_24H,
    resolution_min_days: int = RESOLUTION_DAYS_MIN,
    resolution_max_days: int = RESOLUTION_DAYS_MAX,
    excluded_categories: frozenset[str] = EXCLUDED_CATEGORIES,
) -> ThetaGateDecision:
    """Validate a market candidate for theta decay entry.

    Args:
        market: GammaMarket-like object with price_yes, volume_24h, etc.
        peak: PeakOptimismResult (None = no peak detected yet).
        zscore_threshold: Minimum z-score for entry.
        longshot_max: Maximum YES price for longshot.
        min_volume: Minimum 24h volume.
        resolution_min_days: Minimum days to resolution.
        resolution_max_days: Maximum days to resolution.
        excluded_categories: Categories to reject.

    Returns:
        ThetaGateDecision with pass/fail and reason.
    """
    cond_id = getattr(market, "condition_id", "")

    # 1. Longshot price check
    price_yes = getattr(market, "price_yes", None)
    if price_yes is None or price_yes >= longshot_max:
        return ThetaGateDecision(
            passed=False,
            reason=f"YES price {price_yes} >= {longshot_max} (not a longshot)",
            condition_id=cond_id,
        )

    if price_yes < Decimal("0.01"):
        return ThetaGateDecision(
            passed=False,
            reason=f"YES price {price_yes} is dust (< $0.01)",
            condition_id=cond_id,
        )

    # 2. Volume check
    volume = getattr(market, "volume_24h", Decimal("0"))
    if volume < min_volume:
        return ThetaGateDecision(
            passed=False,
            reason=f"Volume ${volume:,.0f} below ${min_volume:,.0f} minimum",
            condition_id=cond_id,
        )

    # 3. Category exclusion
    category = getattr(market, "category", "")
    if category in excluded_categories:
        return ThetaGateDecision(
            passed=False,
            reason=f"Category '{category}' is excluded",
            condition_id=cond_id,
        )

    # 4. Fee check
    if getattr(market, "fee_enabled", False):
        return ThetaGateDecision(
            passed=False,
            reason="Market has fees enabled",
            condition_id=cond_id,
        )

    # 5. Active check
    if not getattr(market, "active", True) or getattr(market, "closed", False):
        return ThetaGateDecision(
            passed=False,
            reason="Market is inactive or closed",
            condition_id=cond_id,
        )

    # 6. Resolution window
    end_date_str = getattr(market, "end_date", None)
    if not end_date_str:
        return ThetaGateDecision(
            passed=False,
            reason="No end_date — cannot verify resolution window",
            condition_id=cond_id,
        )

    try:
        if isinstance(end_date_str, str):
            end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        else:
            end_dt = end_date_str
        days = (end_dt - datetime.now(UTC)).days
        if days < resolution_min_days:
            return ThetaGateDecision(
                passed=False,
                reason=f"Resolution in {days}d (min {resolution_min_days}d)",
                condition_id=cond_id,
            )
        if days > resolution_max_days:
            return ThetaGateDecision(
                passed=False,
                reason=f"Resolution in {days}d (max {resolution_max_days}d)",
                condition_id=cond_id,
            )
    except (ValueError, TypeError):
        return ThetaGateDecision(
            passed=False,
            reason="Invalid end_date format",
            condition_id=cond_id,
        )

    # 7. Token IDs required
    if not getattr(market, "token_id_yes", None) or not getattr(
        market, "token_id_no", None
    ):
        return ThetaGateDecision(
            passed=False,
            reason="Missing token_id_yes or token_id_no",
            condition_id=cond_id,
        )

    # 8. Z-score check (if peak result provided)
    if peak is not None:
        if peak.zscore < zscore_threshold:
            return ThetaGateDecision(
                passed=False,
                reason=f"Z-score {peak.zscore:.2f} below {zscore_threshold}σ threshold",
                condition_id=cond_id,
            )

    logger.info(
        "theta_gate_passed",
        condition_id=cond_id[:20],
        price_yes=str(price_yes),
        zscore=round(peak.zscore, 2) if peak else None,
    )

    return ThetaGateDecision(
        passed=True,
        reason="All quality checks passed",
        condition_id=cond_id,
    )


def check_allocation(
    risk_manager: RiskManager,
    strategy_id: str = STRATEGY_ID,
) -> ThetaGateDecision:
    """Check if Strategy A allocation allows new trades.

    Args:
        risk_manager: Risk manager singleton.
        strategy_id: Strategy ID (default "A").

    Returns:
        ThetaGateDecision — passed if capital available, failed if halted/exhausted.
    """
    state = risk_manager.get_strategy_state(strategy_id)
    if state is None:
        return ThetaGateDecision(
            passed=False,
            reason=f"Strategy {strategy_id} has no allocation",
        )

    if state.is_halted:
        return ThetaGateDecision(
            passed=False,
            reason=f"Strategy {strategy_id} is halted (weekly drawdown exceeded)",
        )

    if state.available <= Decimal("0"):
        return ThetaGateDecision(
            passed=False,
            reason=f"Strategy {strategy_id} capital exhausted (deployed: {state.deployed})",
        )

    return ThetaGateDecision(
        passed=True,
        reason=f"Strategy {strategy_id} has ${state.available} available",
    )
