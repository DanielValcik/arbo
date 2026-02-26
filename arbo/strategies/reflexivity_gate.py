"""Quality gate for Strategy B (Reflexivity Surfer).

Validates reflexivity signals before execution. Checks:
- Divergence threshold met for the target phase
- Market is not in Phase START (no trading)
- Volume and liquidity minimums
- Strategy allocation available
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from arbo.core.risk_manager import RiskManager
from arbo.strategies.reflexivity_surfer import Phase
from arbo.utils.logger import get_logger

logger = get_logger("reflexivity_gate")

STRATEGY_ID = "B"

# Defaults (overridable)
BOOM_DIVERGENCE = -0.10
PEAK_DIVERGENCE = 0.20
MIN_VOLUME = Decimal("5000")
MIN_LIQUIDITY = Decimal("2000")


@dataclass
class ReflexivityGateDecision:
    """Result of reflexivity quality gate check."""

    passed: bool
    reason: str
    condition_id: str = ""


def check_reflexivity_signal(
    market: Any,
    phase: Phase,
    divergence: float,
    boom_threshold: float = BOOM_DIVERGENCE,
    peak_threshold: float = PEAK_DIVERGENCE,
    min_volume: Decimal = MIN_VOLUME,
    min_liquidity: Decimal = MIN_LIQUIDITY,
) -> ReflexivityGateDecision:
    """Validate a market candidate for reflexivity entry.

    Args:
        market: GammaMarket-like object.
        phase: Current phase of the market.
        divergence: Current divergence value.
        boom_threshold: Divergence threshold for Phase 2 entry.
        peak_threshold: Divergence threshold for Phase 3 entry.
        min_volume: Minimum 24h volume.
        min_liquidity: Minimum liquidity.

    Returns:
        ReflexivityGateDecision with pass/fail and reason.
    """
    cond_id = getattr(market, "condition_id", "")

    # 1. Phase must not be START — no trading in monitoring phase
    if phase == Phase.START:
        return ReflexivityGateDecision(
            passed=False,
            reason="Market in Phase START (no divergence detected)",
            condition_id=cond_id,
        )

    # 2. Divergence threshold check
    if phase == Phase.BOOM:
        if divergence > boom_threshold:
            return ReflexivityGateDecision(
                passed=False,
                reason=f"Divergence {divergence:.3f} above boom threshold {boom_threshold}",
                condition_id=cond_id,
            )
    elif phase == Phase.PEAK:
        if divergence < peak_threshold:
            return ReflexivityGateDecision(
                passed=False,
                reason=f"Divergence {divergence:.3f} below peak threshold {peak_threshold}",
                condition_id=cond_id,
            )

    # 3. Volume check
    volume = getattr(market, "volume_24h", Decimal("0"))
    if volume < min_volume:
        return ReflexivityGateDecision(
            passed=False,
            reason=f"Volume ${volume:,.0f} below ${min_volume:,.0f} minimum",
            condition_id=cond_id,
        )

    # 4. Liquidity check
    liquidity = getattr(market, "liquidity", Decimal("0"))
    if liquidity < min_liquidity:
        return ReflexivityGateDecision(
            passed=False,
            reason=f"Liquidity ${liquidity:,.0f} below ${min_liquidity:,.0f} minimum",
            condition_id=cond_id,
        )

    # 5. Active market
    if not getattr(market, "active", True) or getattr(market, "closed", False):
        return ReflexivityGateDecision(
            passed=False,
            reason="Market is inactive or closed",
            condition_id=cond_id,
        )

    # 6. Price sanity
    price_yes = getattr(market, "price_yes", None)
    if price_yes is None:
        return ReflexivityGateDecision(
            passed=False,
            reason="No price data",
            condition_id=cond_id,
        )
    if price_yes <= Decimal("0.02") or price_yes >= Decimal("0.98"):
        return ReflexivityGateDecision(
            passed=False,
            reason=f"Price {price_yes} too extreme for reflexivity trading",
            condition_id=cond_id,
        )

    # 7. Token IDs required
    if not getattr(market, "token_id_yes", None) or not getattr(
        market, "token_id_no", None
    ):
        return ReflexivityGateDecision(
            passed=False,
            reason="Missing token IDs",
            condition_id=cond_id,
        )

    logger.info(
        "reflexivity_gate_passed",
        condition_id=cond_id[:20],
        phase=phase.name,
        divergence=round(divergence, 4),
    )

    return ReflexivityGateDecision(
        passed=True,
        reason=f"Phase {phase.name} signal approved (divergence: {divergence:.3f})",
        condition_id=cond_id,
    )


def check_reflexivity_allocation(
    risk_manager: RiskManager,
    strategy_id: str = STRATEGY_ID,
) -> ReflexivityGateDecision:
    """Check if Strategy B allocation allows new trades.

    Args:
        risk_manager: Risk manager singleton.
        strategy_id: Strategy ID (default "B").

    Returns:
        ReflexivityGateDecision — passed if capital available.
    """
    state = risk_manager.get_strategy_state(strategy_id)
    if state is None:
        return ReflexivityGateDecision(
            passed=False,
            reason=f"Strategy {strategy_id} has no allocation",
        )

    if state.is_halted:
        return ReflexivityGateDecision(
            passed=False,
            reason=f"Strategy {strategy_id} is halted",
        )

    if state.available <= Decimal("0"):
        return ReflexivityGateDecision(
            passed=False,
            reason=f"Strategy {strategy_id} capital exhausted",
        )

    return ReflexivityGateDecision(
        passed=True,
        reason=f"Strategy {strategy_id} has ${state.available} available",
    )
