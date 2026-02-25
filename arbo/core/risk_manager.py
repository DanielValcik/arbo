"""Singleton risk manager — EVERY order MUST pass through this.

Hardcoded limits are NOT overridable by config. Changes require CEO approval.
See brief Section 5 for full specification.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

from arbo.utils.logger import get_logger

logger = get_logger("risk_manager")

# ================================================================
# HARDCODED LIMITS — DO NOT CHANGE WITHOUT CEO APPROVAL
# ================================================================
MAX_POSITION_PCT = Decimal("0.05")  # 5% capital per trade
DAILY_LOSS_PCT = Decimal("0.10")  # 10% daily loss → auto shutdown
WEEKLY_LOSS_PCT = Decimal("0.20")  # 20% weekly loss → shutdown + CEO escalation
WHALE_COPY_MAX_PCT = Decimal("0.025")  # 2.5% per copied whale position
MAX_MARKET_TYPE_PCT = Decimal("0.30")  # 30% max in one market type
MAX_CONFLUENCE_DOUBLE_PCT = Decimal("0.05")  # 5% hard cap at any confluence score
MAX_TOTAL_EXPOSURE_PCT = Decimal("0.80")  # 80% max capital deployed
MAX_POSITIONS_PER_MARKET = 1  # Only 1 position per market at a time
MIN_PAPER_WEEKS = 4  # 4 weeks paper trading before ANY live execution
KELLY_FRACTION = Decimal("0.5")  # Half-Kelly sizing


@dataclass
class TradeRequest:
    """Incoming trade request to be validated by risk manager."""

    market_id: str
    token_id: str
    side: str  # BUY or SELL
    price: Decimal
    size: Decimal  # USDC amount
    layer: int
    market_category: str
    confluence_score: int = 0
    is_whale_copy: bool = False


@dataclass
class RiskDecision:
    """Result of pre-trade risk check."""

    approved: bool
    reason: str
    adjusted_size: Decimal | None = None


@dataclass
class ExposureState:
    """Current portfolio exposure tracking."""

    capital: Decimal
    daily_pnl: Decimal = Decimal("0")
    weekly_pnl: Decimal = Decimal("0")
    open_positions_value: Decimal = Decimal("0")
    category_exposure: dict[str, Decimal] = field(default_factory=dict)
    market_positions: dict[str, int] = field(default_factory=dict)
    daily_reset_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    weekly_reset_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class RiskManager:
    """Singleton risk manager. All strategies share one instance.

    Usage:
        risk = RiskManager(capital=Decimal("2000"))
        decision = risk.pre_trade_check(trade_request)
        if decision.approved:
            # execute trade
            risk.post_trade_update(fill_result)
    """

    _instance: RiskManager | None = None
    _lock = asyncio.Lock()

    def __init__(self, capital: Decimal) -> None:
        self._state = ExposureState(capital=capital)
        self._shutdown = False
        self._shutdown_callbacks: list[object] = []

    @classmethod
    async def get_instance(cls, capital: Decimal | None = None) -> RiskManager:
        """Get or create the singleton risk manager."""
        async with cls._lock:
            if cls._instance is None:
                if capital is None:
                    raise ValueError("Capital must be provided for first initialization")
                cls._instance = cls(capital)
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton for testing."""
        cls._instance = None

    @property
    def is_shutdown(self) -> bool:
        """Whether emergency shutdown is active."""
        return self._shutdown

    @property
    def state(self) -> ExposureState:
        """Current exposure state (read-only access)."""
        return self._state

    def update_capital(self, current_capital: Decimal) -> None:
        """Update the capital base used for all risk limit calculations.

        D5 Bug 2 fix: Risk manager was using initial capital for sizing.
        Must be called after every trade, settlement, and hourly as safety net.

        Args:
            current_capital: Current portfolio total value (balance + positions).
        """
        old = self._state.capital
        self._state.capital = current_capital
        if old != current_capital:
            logger.info(
                "risk_capital_updated",
                old=str(old),
                new=str(current_capital),
            )

    def pre_trade_check(self, request: TradeRequest) -> RiskDecision:
        """Validate trade request against all risk limits.

        MUST be called before EVERY order. Returns (approved, reason).
        """
        if self._shutdown:
            return RiskDecision(approved=False, reason="Emergency shutdown active")

        self._maybe_reset_periods()

        # 1. Position size check
        max_size = self._state.capital * MAX_POSITION_PCT
        if request.size > max_size:
            logger.warning(
                "risk_rejected",
                check="position_size",
                requested=str(request.size),
                max_allowed=str(max_size),
                market_id=request.market_id,
            )
            return RiskDecision(
                approved=False,
                reason=f"Position size {request.size} exceeds {MAX_POSITION_PCT*100}% "
                f"limit ({max_size})",
            )

        # 2. Whale copy size check
        if request.is_whale_copy:
            whale_max = self._state.capital * WHALE_COPY_MAX_PCT
            if request.size > whale_max:
                return RiskDecision(
                    approved=False,
                    reason=f"Whale copy size {request.size} exceeds {WHALE_COPY_MAX_PCT*100}% "
                    f"limit ({whale_max})",
                    adjusted_size=whale_max,
                )

        # 3. Confluence double-size hard cap
        if request.confluence_score >= 3:
            confluence_max = self._state.capital * MAX_CONFLUENCE_DOUBLE_PCT
            if request.size > confluence_max:
                return RiskDecision(
                    approved=False,
                    reason=f"Confluence double-size {request.size} exceeds hard cap ({confluence_max})",
                    adjusted_size=confluence_max,
                )

        # 4. Daily loss check
        if abs(self._state.daily_pnl) >= self._state.capital * DAILY_LOSS_PCT:
            logger.critical(
                "daily_loss_limit_hit",
                daily_pnl=str(self._state.daily_pnl),
                limit=str(self._state.capital * DAILY_LOSS_PCT),
            )
            self._trigger_shutdown("Daily loss limit exceeded")
            return RiskDecision(approved=False, reason="Daily loss limit exceeded — shutdown")

        # 5. Weekly loss check
        if abs(self._state.weekly_pnl) >= self._state.capital * WEEKLY_LOSS_PCT:
            logger.critical(
                "weekly_loss_limit_hit",
                weekly_pnl=str(self._state.weekly_pnl),
                limit=str(self._state.capital * WEEKLY_LOSS_PCT),
            )
            self._trigger_shutdown("Weekly loss limit exceeded — CEO escalation required")
            return RiskDecision(approved=False, reason="Weekly loss limit exceeded — shutdown")

        # 6. Market type concentration check
        category = request.market_category
        current_exposure = self._state.category_exposure.get(category, Decimal("0"))
        category_max = self._state.capital * MAX_MARKET_TYPE_PCT
        if current_exposure + request.size > category_max:
            return RiskDecision(
                approved=False,
                reason=f"Category '{category}' exposure would exceed {MAX_MARKET_TYPE_PCT*100}% "
                f"limit ({category_max}). Current: {current_exposure}",
            )

        # 7. Total exposure cap
        exposure_max = self._state.capital * MAX_TOTAL_EXPOSURE_PCT
        if self._state.open_positions_value + request.size > exposure_max:
            logger.warning(
                "risk_rejected",
                check="total_exposure",
                current=str(self._state.open_positions_value),
                requested=str(request.size),
                max_allowed=str(exposure_max),
            )
            return RiskDecision(
                approved=False,
                reason=f"Total exposure {self._state.open_positions_value + request.size} "
                f"would exceed {MAX_TOTAL_EXPOSURE_PCT*100}% limit ({exposure_max})",
            )

        # 8. Per-market position limit
        market_pos_count = self._state.market_positions.get(request.market_id, 0)
        if market_pos_count >= MAX_POSITIONS_PER_MARKET:
            logger.warning(
                "risk_rejected",
                check="per_market_limit",
                market_id=request.market_id,
                current_positions=market_pos_count,
            )
            return RiskDecision(
                approved=False,
                reason=f"Market '{request.market_id}' already has {market_pos_count} "
                f"position(s) (max {MAX_POSITIONS_PER_MARKET})",
            )

        logger.info(
            "risk_approved",
            market_id=request.market_id,
            size=str(request.size),
            layer=request.layer,
            confluence=request.confluence_score,
        )
        return RiskDecision(approved=True, reason="All checks passed")

    def post_trade_update(
        self,
        market_id: str,
        market_category: str,
        size: Decimal,
        pnl: Decimal | None = None,
    ) -> None:
        """Update exposure after a trade fill or resolution.

        Args:
            market_id: The market condition ID.
            market_category: Category for concentration tracking.
            size: Trade size in USDC.
            pnl: Realized P&L if trade resolved (None for new positions).
        """
        # Update category exposure
        current = self._state.category_exposure.get(market_category, Decimal("0"))
        self._state.category_exposure[market_category] = current + size

        # Update open positions value and per-market tracking
        if pnl is None:
            # Opening a new position
            self._state.open_positions_value += size
            cur_count = self._state.market_positions.get(market_id, 0)
            self._state.market_positions[market_id] = cur_count + 1
        else:
            # Closing / resolving a position
            self._state.daily_pnl += pnl
            self._state.weekly_pnl += pnl
            self._state.open_positions_value = max(
                Decimal("0"), self._state.open_positions_value - size
            )
            cur_count = self._state.market_positions.get(market_id, 0)
            self._state.market_positions[market_id] = max(0, cur_count - 1)
            # Remove from category exposure
            self._state.category_exposure[market_category] = max(
                Decimal("0"), current + size - abs(size)
            )

        logger.info(
            "exposure_updated",
            market_id=market_id,
            daily_pnl=str(self._state.daily_pnl),
            weekly_pnl=str(self._state.weekly_pnl),
        )

        # Check if P&L crossed limits after update
        if (
            pnl is not None
            and pnl < 0
            and abs(self._state.daily_pnl) >= self._state.capital * DAILY_LOSS_PCT
        ):
            self._trigger_shutdown("Daily loss limit exceeded after trade resolution")

    def _trigger_shutdown(self, reason: str) -> None:
        """Activate emergency shutdown."""
        self._shutdown = True
        logger.critical("emergency_shutdown", reason=reason)

    async def emergency_shutdown(self, reason: str) -> None:
        """Cancel ALL orders, log reason, notify CEO via Slack.

        This is the public async version that can interact with external systems.
        """
        self._trigger_shutdown(reason)
        logger.critical("emergency_shutdown_initiated", reason=reason)
        # Actual order cancellation and Slack notification handled by caller

    def _maybe_reset_periods(self) -> None:
        """Reset daily/weekly counters if period has elapsed."""
        now = datetime.now(UTC)

        # Reset daily at midnight UTC
        if now.date() > self._state.daily_reset_at.date():
            logger.info("daily_pnl_reset", previous=str(self._state.daily_pnl))
            self._state.daily_pnl = Decimal("0")
            self._state.daily_reset_at = now

        # Reset weekly on Monday
        if now.isocalendar()[1] != self._state.weekly_reset_at.isocalendar()[1]:
            logger.info("weekly_pnl_reset", previous=str(self._state.weekly_pnl))
            self._state.weekly_pnl = Decimal("0")
            self._state.weekly_reset_at = now
