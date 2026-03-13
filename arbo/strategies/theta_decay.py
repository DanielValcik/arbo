"""Strategy A: Theta Decay — Sell Optimism Premium on Longshots.

Monitors longshot YES contracts (price < $0.092) for 3σ YES taker flow spikes
(peak retail optimism). When detected, buys the NO side and holds to
resolution. Captures the theta-like decay as hype fades.

Key autoresearch discovery: price-dependent discount factor — cheaper longshots
have more optimism bias, so the model discounts them more aggressively.
See research_a/AUTORESEARCH_A_REPORT.md for optimization details.

Entry: YES < $0.092, 3σ YES taker spike → buy NO (price-dependent discount model)
Exit: Hold to resolution, partial exit at NO +50%, stop loss at NO -20%
Sizing: 2-5% of capital, Kelly fraction 0.032
Max concurrent: 25 positions
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from arbo.config.settings import get_config
from arbo.connectors.polygon_flow import MarketFlowTracker, PeakOptimismResult
from arbo.core.risk_manager import (
    MAX_POSITION_PCT,
    RiskManager,
    TradeRequest,
)
from arbo.utils.logger import get_logger

logger = get_logger("theta_decay")

STRATEGY_ID = "A"


class ThetaDecay:
    """Strategy A: Theta Decay — sell optimism premium on longshot markets.

    Lifecycle:
    1. Filter markets for longshot YES < $0.092 with sufficient volume
    2. Register candidates with MarketFlowTracker for taker flow monitoring
    3. Wait for 3σ YES taker flow spike (peak retail optimism)
    4. Compute edge via price-dependent discount model
    5. Buy NO side when spike detected and edge passes quality gate
    6. Hold to resolution (partial exit / stop loss via exit monitor)
    """

    def __init__(
        self,
        risk_manager: RiskManager,
        flow_tracker: MarketFlowTracker,
        paper_engine: Any = None,
    ) -> None:
        cfg = get_config().theta_decay
        self._risk = risk_manager
        self._flow_tracker = flow_tracker
        self._paper_engine = paper_engine

        # Market filtering
        self._zscore_threshold = cfg.zscore_threshold
        self._longshot_max = Decimal(str(cfg.longshot_price_max))
        self._min_volume = Decimal(str(cfg.min_volume_24h))
        self._min_age_hours = cfg.min_age_hours
        self._resolution_min_days = cfg.resolution_window_days_min
        self._resolution_max_days = cfg.resolution_window_days_max
        self._excluded_categories = set(cfg.excluded_categories)

        # Entry model — price-dependent discount (autoresearch)
        self._discount_factor = cfg.discount_factor
        self._price_scale_intercept = cfg.discount_price_scale_intercept
        self._price_scale_slope = cfg.discount_price_scale_slope
        self._min_edge = Decimal(str(cfg.min_edge))
        self._max_edge = Decimal(str(cfg.max_edge))

        # Quality gate — resolution-adjusted zscore
        self._zscore_days_pivot = cfg.zscore_days_pivot
        self._zscore_penalty_power = cfg.zscore_penalty_power
        self._zscore_penalty_coeff = cfg.zscore_penalty_coeff

        # Exit rules
        self._partial_exit_pct = cfg.partial_exit_pct
        self._stop_loss_pct = cfg.stop_loss_pct

        # Sizing (%-based)
        self._kelly_fraction = cfg.kelly_fraction
        self._pos_pct_min = cfg.position_pct_min
        self._pos_pct_max = cfg.position_pct_max
        self._max_concurrent = cfg.max_concurrent_positions

        # State
        self._signals_generated: int = 0
        self._trades_placed: int = 0
        self._last_scan: datetime | None = None
        self._registered_markets: set[str] = set()
        self._active_positions: dict[str, _ThetaPosition] = {}

    async def init(self) -> None:
        """Initialize strategy (nothing async needed for now)."""
        logger.info(
            "theta_decay_init",
            zscore_threshold=self._zscore_threshold,
            longshot_max=str(self._longshot_max),
            max_concurrent=self._max_concurrent,
        )

    async def close(self) -> None:
        """Clean up resources."""
        logger.info(
            "theta_decay_close",
            signals=self._signals_generated,
            trades=self._trades_placed,
        )

    async def poll_cycle(self, markets: list[Any]) -> list[dict[str, Any]]:
        """Run one poll cycle: filter markets, check for 3σ spikes, trade.

        Args:
            markets: List of GammaMarket objects from market discovery.

        Returns:
            List of trade results for markets where trades were placed.
        """
        self._last_scan = datetime.now(UTC)
        logger.info("theta_decay_poll_start", markets_available=len(markets))

        # Check strategy allocation state
        strategy_state = self._risk.get_strategy_state(STRATEGY_ID)
        if strategy_state is None:
            logger.warning("strategy_a_no_allocation")
            return []
        if strategy_state.is_halted:
            logger.warning("strategy_a_halted", reason="weekly drawdown exceeded")
            return []

        available_capital = strategy_state.available
        if available_capital <= Decimal("0"):
            logger.info("strategy_a_no_capital", deployed=str(strategy_state.deployed))
            return []

        # Filter candidate markets
        candidates = self._filter_candidates(markets)
        if not candidates:
            return []

        # Register candidates for taker flow tracking
        for mkt in candidates:
            if mkt.condition_id not in self._registered_markets:
                yes_id = mkt.token_id_yes
                no_id = mkt.token_id_no
                if yes_id and no_id:
                    self._flow_tracker.register_market(mkt.condition_id, yes_id, no_id)
                    self._registered_markets.add(mkt.condition_id)

        # Check for 3σ peak optimism spikes
        peaks = self._flow_tracker.get_peak_optimism_markets(
            threshold=self._zscore_threshold
        )
        if not peaks:
            return []

        # Build condition_id → market lookup
        market_lookup = {m.condition_id: m for m in candidates}

        # Trade on peak optimism
        traded = []
        for peak in peaks:
            if peak.condition_id not in market_lookup:
                continue

            # Already have a position in this market
            if peak.condition_id in self._active_positions:
                continue

            # Check concurrent position limit
            if len(self._active_positions) >= self._max_concurrent:
                logger.info("strategy_a_max_positions", count=len(self._active_positions))
                break

            mkt = market_lookup[peak.condition_id]
            self._signals_generated += 1

            result = self._execute_entry(mkt, peak, available_capital)
            if result is not None:
                traded.append(result)
                available_capital -= Decimal(str(result["size"]))

        return traded

    def _filter_candidates(self, markets: list[Any]) -> list[Any]:
        """Filter markets matching theta decay criteria."""
        now = datetime.now(UTC)
        candidates = []

        for mkt in markets:
            # Price filter: longshot YES
            price_yes = mkt.price_yes
            if price_yes is None or price_yes >= self._longshot_max:
                continue

            # Skip near-zero (dust) prices
            if price_yes < Decimal("0.01"):
                continue

            # Volume filter
            if mkt.volume_24h < self._min_volume:
                continue

            # Category exclusion
            if mkt.category in self._excluded_categories:
                continue

            # Fee filter: prefer fee-free
            if mkt.fee_enabled:
                continue

            # Active market
            if not mkt.active or mkt.closed:
                continue

            # Resolution window
            if not self._check_resolution_window(mkt, now):
                continue

            # Token IDs required
            if not mkt.token_id_yes or not mkt.token_id_no:
                continue

            candidates.append(mkt)

        logger.info(
            "theta_decay_candidates",
            total_markets=len(markets),
            candidates=len(candidates),
        )
        return candidates

    def _check_resolution_window(self, mkt: Any, now: datetime) -> bool:
        """Check if market resolves within configured day window."""
        end_date_str = getattr(mkt, "end_date", None)
        if not end_date_str:
            return False

        try:
            # Parse ISO datetime
            if isinstance(end_date_str, str):
                end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            else:
                end_dt = end_date_str

            days_to_resolution = (end_dt - now).days
            return self._resolution_min_days <= days_to_resolution <= self._resolution_max_days
        except (ValueError, TypeError):
            return False

    def _compute_edge(
        self, price_yes: Decimal, zscore: float, days_to_resolution: int
    ) -> tuple[Decimal, Decimal] | None:
        """Compute edge using price-dependent discount model.

        Autoresearch discovery: cheaper longshots have more optimism bias.
        price_scale = intercept + slope * (price_yes / longshot_max)
        effective_discount = discount_factor * price_scale

        Also applies resolution-adjusted z-score threshold (quadratic penalty
        for longer-dated markets).

        Returns:
            (edge, model_no_prob) or None if quality gate fails.
        """
        f_price = float(price_yes)
        f_max = float(self._longshot_max)

        # Resolution-adjusted z-score threshold
        days_excess = max(0, days_to_resolution - self._zscore_days_pivot)
        adj_threshold = self._zscore_threshold + (
            days_excess ** self._zscore_penalty_power * self._zscore_penalty_coeff
        )
        if zscore < adj_threshold:
            return None

        # Price-dependent discount
        price_scale = self._price_scale_intercept + self._price_scale_slope * (f_price / f_max)
        effective_discount = self._discount_factor * price_scale

        model_yes_prob = f_price * effective_discount
        model_no_prob = 1.0 - model_yes_prob
        no_price = 1.0 - f_price
        edge = model_no_prob - no_price

        edge_dec = Decimal(str(round(edge, 6)))
        model_no_dec = Decimal(str(round(model_no_prob, 6)))

        # Edge bounds check
        if edge_dec < self._min_edge or edge_dec > self._max_edge:
            return None

        return edge_dec, model_no_dec

    def _compute_size(
        self, price_no: Decimal, edge: Decimal, available: Decimal, total: Decimal
    ) -> Decimal:
        """Compute position size with ultra-conservative Kelly, %-based bounds.

        Args:
            price_no: NO token price (our entry price).
            edge: Estimated edge (model_prob - market_price).
            available: Available capital for Strategy A.
            total: Total Strategy A allocation.

        Returns:
            Position size in USDC.
        """
        f_no = float(price_no)
        if f_no <= 0 or f_no >= 1:
            return Decimal("0")

        # Kelly: f* = edge / (1/no_price - 1) * kelly_fraction
        odds_minus_1 = (1.0 / f_no) - 1.0
        if odds_minus_1 <= 0:
            return Decimal("0")

        kelly_raw = float(edge) / odds_minus_1
        kelly_adjusted = kelly_raw * self._kelly_fraction
        raw_size = kelly_adjusted * float(available)

        # Clamp to %-based position bounds
        f_total = float(total)
        min_size = f_total * self._pos_pct_min
        max_size = f_total * self._pos_pct_max
        clamped = max(min_size, min(max_size, raw_size))

        # Don't exceed available capital
        clamped = min(clamped, float(available) * 0.95)

        if clamped < min_size:
            return Decimal("0")

        # Cap at MAX_POSITION_PCT of total capital (risk manager constraint)
        max_pos = float(self._risk._state.capital * MAX_POSITION_PCT)
        clamped = min(clamped, max_pos)

        return Decimal(str(round(clamped, 2)))

    def _execute_entry(
        self,
        mkt: Any,
        peak: PeakOptimismResult,
        available_capital: Decimal,
    ) -> dict[str, Any] | None:
        """Execute a theta decay entry: buy NO on peak optimism.

        Args:
            mkt: GammaMarket object.
            peak: PeakOptimismResult with z-score and ratio.
            available_capital: Remaining Strategy A capital.

        Returns:
            Trade result dict or None if rejected.
        """
        price_yes = mkt.price_yes
        price_no = mkt.price_no
        if price_no is None or price_yes is None:
            return None

        # Compute days to resolution for quality gate
        now = datetime.now(UTC)
        end_date_str = getattr(mkt, "end_date", None)
        days_to_res = 15  # Default if can't compute
        if end_date_str:
            try:
                if isinstance(end_date_str, str):
                    end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                else:
                    end_dt = end_date_str
                days_to_res = (end_dt - now).days
            except (ValueError, TypeError):
                pass

        # Compute edge with price-dependent discount + quality gate
        result = self._compute_edge(price_yes, peak.zscore, days_to_res)
        if result is None:
            return None
        edge, model_no_prob = result

        # Get total capital for %-based sizing
        strategy_state = self._risk.get_strategy_state(STRATEGY_ID)
        total_capital = strategy_state.allocated if strategy_state else available_capital

        # Compute size
        size = self._compute_size(price_no, edge, available_capital, total_capital)
        if size <= Decimal("0"):
            return None

        # Build trade request for risk check
        trade_req = TradeRequest(
            market_id=mkt.condition_id,
            token_id=mkt.token_id_no,
            side="BUY",
            price=price_no,
            size=size,
            layer=0,
            market_category=mkt.category,
            strategy=STRATEGY_ID,
        )

        # Risk check
        decision = self._risk.pre_trade_check(trade_req)
        if not decision.approved:
            logger.info(
                "theta_decay_trade_rejected",
                condition_id=mkt.condition_id[:20],
                reason=decision.reason,
            )
            return None

        # Adjust size if risk manager says so
        final_size = decision.adjusted_size if decision.adjusted_size else size

        # Execute via paper engine
        if self._paper_engine is None:
            return None

        trade = self._paper_engine.place_trade(
            market_condition_id=mkt.condition_id,
            token_id=mkt.token_id_no,
            side="BUY",
            market_price=price_no,
            model_prob=model_no_prob,
            layer=0,
            market_category=mkt.category,
            strategy=STRATEGY_ID,
            pre_computed_size=final_size,
        )

        if trade is None:
            return None

        # Strategy-level accounting (post_trade_update already called by paper_engine)
        self._risk.strategy_post_trade(STRATEGY_ID, final_size)

        # Track position internally
        self._active_positions[mkt.condition_id] = _ThetaPosition(
            condition_id=mkt.condition_id,
            token_id=mkt.token_id_no,
            entry_price=price_no,
            size=final_size,
            entry_zscore=peak.zscore,
            entered_at=datetime.now(UTC),
        )
        self._trades_placed += 1

        logger.info(
            "theta_decay_entry",
            condition_id=mkt.condition_id[:20],
            price_yes=str(price_yes),
            price_no=str(price_no),
            size=str(final_size),
            edge=str(edge),
            zscore=round(peak.zscore, 2),
        )

        return {
            "condition_id": mkt.condition_id,
            "side": "BUY_NO",
            "price": float(price_no),
            "size": float(final_size),
            "edge": float(edge),
            "zscore": peak.zscore,
            "strategy": STRATEGY_ID,
        }

    def check_exits(self, current_prices: dict[str, Decimal]) -> list[dict[str, Any]]:
        """Check active positions for partial exit or stop loss.

        Args:
            current_prices: {condition_id: current_no_price}

        Returns:
            List of exit actions taken.
        """
        exits = []
        to_remove = []

        for cond_id, pos in self._active_positions.items():
            current = current_prices.get(cond_id)
            if current is None:
                continue

            pnl_pct = (current - pos.entry_price) / pos.entry_price

            # Stop loss: NO dropped > stop_loss_pct below entry
            if pnl_pct <= -Decimal(str(self._stop_loss_pct)):
                exits.append({
                    "condition_id": cond_id,
                    "action": "stop_loss",
                    "entry_price": float(pos.entry_price),
                    "current_price": float(current),
                    "pnl_pct": float(pnl_pct),
                    "size": float(pos.size),
                })
                to_remove.append(cond_id)
                logger.info(
                    "theta_decay_stop_loss",
                    condition_id=cond_id[:20],
                    pnl_pct=float(pnl_pct),
                )

            # Partial exit: NO gained > partial_exit_pct above entry
            elif pnl_pct >= Decimal(str(self._partial_exit_pct)) and not pos.partial_exited:
                exit_size = pos.size * Decimal("0.5")
                pos.size -= exit_size
                pos.partial_exited = True
                exits.append({
                    "condition_id": cond_id,
                    "action": "partial_exit",
                    "entry_price": float(pos.entry_price),
                    "current_price": float(current),
                    "pnl_pct": float(pnl_pct),
                    "exit_size": float(exit_size),
                    "remaining_size": float(pos.size),
                })
                logger.info(
                    "theta_decay_partial_exit",
                    condition_id=cond_id[:20],
                    pnl_pct=float(pnl_pct),
                    exit_size=str(exit_size),
                )

        for cond_id in to_remove:
            del self._active_positions[cond_id]

        return exits

    def handle_resolution(
        self, condition_id: str, pnl: Decimal
    ) -> None:
        """Handle market resolution for a theta position.

        Args:
            condition_id: Market that resolved.
            pnl: Realized P&L from resolution.
        """
        if condition_id in self._active_positions:
            del self._active_positions[condition_id]

        # Update risk manager with realized P&L
        self._risk.strategy_post_trade(STRATEGY_ID, abs(pnl), pnl=pnl)

        logger.info(
            "theta_decay_resolution",
            condition_id=condition_id[:20],
            pnl=str(pnl),
            remaining_positions=len(self._active_positions),
        )

    @property
    def stats(self) -> dict[str, Any]:
        """Current strategy statistics."""
        strategy_state = self._risk.get_strategy_state(STRATEGY_ID)
        return {
            "strategy": STRATEGY_ID,
            "signals_generated": self._signals_generated,
            "trades_placed": self._trades_placed,
            "active_positions": len(self._active_positions),
            "registered_markets": len(self._registered_markets),
            "last_scan": self._last_scan.isoformat() if self._last_scan else None,
            "deployed": str(strategy_state.deployed) if strategy_state else "0",
            "available": str(strategy_state.available) if strategy_state else "0",
            "is_halted": strategy_state.is_halted if strategy_state else False,
        }


class _ThetaPosition:
    """Internal tracking of an active theta decay position."""

    __slots__ = (
        "condition_id",
        "token_id",
        "entry_price",
        "size",
        "entry_zscore",
        "entered_at",
        "partial_exited",
    )

    def __init__(
        self,
        condition_id: str,
        token_id: str,
        entry_price: Decimal,
        size: Decimal,
        entry_zscore: float,
        entered_at: datetime,
    ) -> None:
        self.condition_id = condition_id
        self.token_id = token_id
        self.entry_price = entry_price
        self.size = size
        self.entry_zscore = entry_zscore
        self.entered_at = entered_at
        self.partial_exited = False
