"""Paper trading engine for simulating trades without real capital.

Accepts trade signals, simulates fills at current midprice with configurable
slippage, tracks P&L per-trade, per-strategy, per-day, per-week.

See brief PM-004 for full specification.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum

from arbo.core.fee_model import calculate_taker_fee
from arbo.core.risk_manager import RiskManager, TradeRequest
from arbo.utils.logger import get_logger
from arbo.utils.odds import half_kelly

logger = get_logger("paper_engine")

# Bounded in-memory caches. DB (paper_trades table) is the authoritative store —
# these structures exist only for fast lookup during a single session. Without
# bounds, steady-state operation leaks ~100 MB/h (observed empirically over 17h
# uptime, 1.8 GB RSS reached). Each bound is conservative enough that the
# working set fits in memory for weeks while avoiding unbounded growth.
MAX_TRADES_IN_MEMORY = 10000           # ~10 MB at typical PaperTrade size
MAX_SNAPSHOTS_IN_MEMORY = 2000         # Hourly snapshots × ~83 days
MAX_TRADE_DETAILS_CACHE = 5000         # Open-position fallback lookup only

# Polygon gas cost per transaction (~$0.007)
POLYGON_GAS_COST_USD = Decimal("0.007")

# Taker slippage on top of CLOB price (0.5 cent = half a tick on 0.01 markets)
# Conservative estimate for small orders ($3-5) on weather markets with MIN_LIQUIDITY=200
CLOB_TAKER_SLIPPAGE = Decimal("0.005")


class TradeStatus(Enum):
    OPEN = "open"
    WON = "won"
    LOST = "lost"
    SOLD = "sold"  # Early exit (before market resolution)
    CANCELLED = "cancelled"


@dataclass
class PaperTrade:
    """A simulated trade in the paper trading engine."""

    id: int
    market_condition_id: str
    token_id: str
    layer: int
    side: str  # BUY or SELL
    price: Decimal
    fill_price: Decimal  # price after slippage
    size: Decimal  # USDC amount
    shares: Decimal  # number of shares acquired
    edge: Decimal
    confluence_score: int
    kelly_fraction: Decimal
    fee: Decimal
    strategy: str = ""  # RDH strategy ID: "A", "B", "C" (empty = legacy layer)
    status: TradeStatus = TradeStatus.OPEN
    actual_pnl: Decimal | None = None
    exit_price: Decimal | None = None  # Fill price on early exit
    exit_reason: str | None = None  # Why exited: edge_lost, profit_take, prob_floor
    placed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    resolved_at: datetime | None = None
    notes: str = ""
    trade_details: dict | None = None  # Comprehensive data for backtesting


@dataclass
class PaperPosition:
    """An open position in the paper portfolio."""

    market_condition_id: str
    token_id: str
    side: str
    avg_price: Decimal
    size: Decimal  # total USDC invested
    shares: Decimal
    layer: int
    strategy: str = ""
    current_price: Decimal | None = None
    opened_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate unrealized P&L based on current price."""
        if self.current_price is None:
            return Decimal("0")
        if self.side == "BUY":
            return self.shares * (self.current_price - self.avg_price)
        return self.shares * (self.avg_price - self.current_price)


@dataclass
class PortfolioSnapshot:
    """Hourly snapshot of portfolio state."""

    balance: Decimal
    unrealized_pnl: Decimal
    total_value: Decimal
    num_open_positions: int
    per_layer_pnl: dict[int, Decimal]
    snapshot_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class PaperTradingEngine:
    """Simulated trading engine for validating strategies without real capital.

    Features:
    - Accepts OrderArgs, simulates fill at current midprice
    - Configurable slippage (default 0.5%)
    - P&L tracking: per-trade, per-strategy (layer), per-day, per-week
    - Position management: open positions, unrealized P&L
    - Automatic resolution: call resolve_market() when market settles
    - Hourly snapshots of portfolio state
    """

    def __init__(
        self,
        initial_capital: Decimal,
        slippage_pct: Decimal = Decimal("0.005"),
        risk_manager: RiskManager | None = None,
    ) -> None:
        self._initial_capital = initial_capital
        self._balance = initial_capital  # available USDC
        self._slippage_pct = slippage_pct
        self._risk_manager = risk_manager
        # Bounded: deque drops oldest entries when full. DB is authoritative
        # for full trade history; this list only buffers recent trades for
        # in-process operations (match by token_id during resolution, etc.).
        self._trades: deque[PaperTrade] = deque(maxlen=MAX_TRADES_IN_MEMORY)
        self._positions: dict[str, PaperPosition] = {}  # key: token_id
        self._snapshots: deque[PortfolioSnapshot] = deque(maxlen=MAX_SNAPSHOTS_IN_MEMORY)
        self._next_trade_id = 1
        self._per_layer_realized_pnl: dict[int, Decimal] = {}
        self._per_strategy_realized_pnl: dict[str, Decimal] = {}
        # LRU: OrderedDict used as bounded cache (move_to_end on access,
        # popitem(last=False) to evict oldest). trade_details is read-mostly,
        # so occasional eviction is fine — falls back to DB query if missing.
        self._trade_details_cache: OrderedDict[str, dict] = OrderedDict()

    @property
    def balance(self) -> Decimal:
        """Available USDC balance."""
        return self._balance

    @property
    def total_value(self) -> Decimal:
        """Total portfolio value (balance + unrealized P&L)."""
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        invested = sum(p.size for p in self._positions.values())
        return self._balance + invested + unrealized

    @property
    def open_positions(self) -> list[PaperPosition]:
        """All open positions."""
        return list(self._positions.values())

    @property
    def trade_history(self) -> list[PaperTrade]:
        """All trades (open + resolved)."""
        return list(self._trades)

    def place_trade(
        self,
        market_condition_id: str,
        token_id: str,
        side: str,
        market_price: Decimal,
        model_prob: Decimal,
        layer: int,
        market_category: str,
        fee_enabled: bool = False,
        confluence_score: int = 0,
        is_whale_copy: bool = False,
        strategy: str = "",
        pre_computed_size: Decimal | None = None,
        volume_24h: Decimal | None = None,
        liquidity: Decimal | None = None,
        clob_fill_price: Decimal | None = None,
        trade_details: dict | None = None,
        bypass_risk_check: bool = False,
        is_live_capital: bool = False,
    ) -> PaperTrade | None:
        """Place a simulated paper trade.

        Calculates position size via half-Kelly, applies slippage and fees,
        checks risk manager, and records the trade.

        Args:
            market_condition_id: Market condition ID.
            token_id: CLOB token ID for the outcome to trade.
            side: "BUY" or "SELL".
            market_price: Current market price (0-1).
            model_prob: Our estimated probability.
            layer: Strategy layer (1-9).
            market_category: Category for risk concentration.
            fee_enabled: Whether this market has fees.
            confluence_score: Confluence score (0-5).
            is_whale_copy: Whether this is a whale copy trade.
            strategy: RDH strategy ID ("A", "B", "C"). Empty = legacy layer.
            pre_computed_size: Pre-calculated size from strategy ladder/sizing.
                When provided, skips internal Kelly calculation.

        Returns:
            PaperTrade if executed, None if rejected by risk manager.
        """
        # Calculate edge
        fee = calculate_taker_fee(market_price, fee_enabled)
        edge = abs(model_prob - market_price) - fee

        # Edge check: skip only if strategy didn't pre-compute sizing.
        # When pre_computed_size is set, the strategy has already validated
        # its own edge logic and signals (e.g. B3 uses CDF model). Trust it.
        if edge <= Decimal("0") and pre_computed_size is None:
            logger.debug(
                "paper_trade_no_edge",
                token_id=token_id,
                edge=str(edge),
                layer=layer,
            )
            return None
        if edge <= Decimal("0"):
            # Strategy provided pre_computed_size — trust strategy edge logic
            logger.debug(
                "paper_trade_zero_edge_pre_computed",
                token_id=token_id,
                strategy=strategy,
            )

        if pre_computed_size is not None:
            # RDH strategies compute their own sizing (Quarter-Kelly via ladder)
            size = pre_computed_size
            kelly = Decimal("0")  # Not used — strategy did its own sizing
        else:
            # Legacy layers: calculate position size via half-Kelly
            kelly = half_kelly(model_prob, market_price)
            max_pct = Decimal("0.05")
            if confluence_score >= 3:
                max_pct = Decimal("0.05")  # hard cap even at high confluence
            elif confluence_score >= 2:
                max_pct = Decimal("0.025")  # standard size

            position_pct = min(kelly, max_pct)
            size = (self._balance * position_pct).quantize(Decimal("0.01"))

        if size <= Decimal("0"):
            return None

        # Risk manager check (skipped for shadow variants — they're pure
        # counterfactuals, don't consume capital, don't occupy real positions.
        # Project PARALLEL per-variant paper trades use bypass_risk_check=True
        # because champion already occupies the per-market slot; challengers
        # need their own independent entries on same market for PnL comparison.)
        if self._risk_manager and not bypass_risk_check:
            request = TradeRequest(
                market_id=market_condition_id,
                token_id=token_id,
                side=side,
                price=market_price,
                size=size,
                layer=layer,
                market_category=market_category,
                confluence_score=confluence_score,
                is_whale_copy=is_whale_copy,
                strategy=strategy,
                is_live_capital=is_live_capital,
            )
            decision = self._risk_manager.pre_trade_check(request)
            if not decision.approved:
                logger.info(
                    "paper_trade_rejected",
                    reason=decision.reason,
                    token_id=token_id,
                    layer=layer,
                )
                return None
            # Use adjusted size if provided
            if decision.adjusted_size is not None:
                size = decision.adjusted_size

        # D5 Bug 4: Balance floor — never go negative.
        # Shadow variants skip this (they don't consume capital).
        if not bypass_risk_check and size > self._balance:
            logger.info(
                "paper_trade_insufficient_balance",
                size=str(size),
                balance=str(self._balance),
                token_id=token_id,
            )
            return None

        # Apply slippage — use real CLOB fill price when available
        # Even with CLOB prices, taker orders fill slightly worse than displayed
        if clob_fill_price is not None:
            if side == "BUY":
                fill_price = clob_fill_price + CLOB_TAKER_SLIPPAGE
            else:
                fill_price = clob_fill_price - CLOB_TAKER_SLIPPAGE
            fill_price = min(max(fill_price, Decimal("0.001")), Decimal("0.999"))
        elif side == "BUY":
            fill_price = market_price * (Decimal("1") + self._slippage_pct)
        else:
            fill_price = market_price * (Decimal("1") - self._slippage_pct)

        fill_price = min(max(fill_price, Decimal("0.001")), Decimal("0.999"))

        # Calculate shares
        shares = (size / fill_price).quantize(Decimal("0.01"))

        # Deduct from balance (trade size + Polygon gas).
        # Shadow variants skip this — they're counterfactuals, no real capital.
        if not bypass_risk_check:
            self._balance -= size + POLYGON_GAS_COST_USD

        # Create trade
        trade = PaperTrade(
            id=self._next_trade_id,
            market_condition_id=market_condition_id,
            token_id=token_id,
            layer=layer,
            side=side,
            price=market_price,
            fill_price=fill_price,
            size=size,
            shares=shares,
            edge=edge,
            confluence_score=confluence_score,
            kelly_fraction=kelly,
            fee=fee * size + POLYGON_GAS_COST_USD,
            strategy=strategy,
            trade_details=trade_details,
        )
        self._next_trade_id += 1
        self._trades.append(trade)

        # Update or create position — skipped for shadow variants to avoid
        # polluting champion's aggregate exposure metrics. Shadow trades exist
        # only as DB rows for per-variant PnL tracking; resolution uses
        # update_trade_by_id (by primary key), not position-based exit flow.
        if not bypass_risk_check:
            pos_key = token_id
            if pos_key in self._positions:
                pos = self._positions[pos_key]
                total_size = pos.size + size
                pos.avg_price = (pos.avg_price * pos.shares + fill_price * shares) / (
                    pos.shares + shares
                )
                pos.shares += shares
                pos.size = total_size
            else:
                self._positions[pos_key] = PaperPosition(
                    market_condition_id=market_condition_id,
                    token_id=token_id,
                    side=side,
                    avg_price=fill_price,
                    size=size,
                    shares=shares,
                    layer=layer,
                    strategy=strategy,
                )

        # Update risk manager (global exposure + per-strategy state)
        if self._risk_manager:
            self._risk_manager.post_trade_update(
                market_id=market_condition_id,
                market_category=market_category,
                size=size,
            )
            # CRITICAL: also increment per-strategy deployed/position_count
            # so MAX_POSITIONS_PER_STRATEGY and allocation cap are enforced.
            # Without this, strategies bypass their own risk limits.
            # is_live_capital routes to the live slot pool (real capital)
            # vs the paper pool (simulated).
            if strategy:
                try:
                    self._risk_manager.strategy_post_trade(
                        strategy, size, is_live_capital=is_live_capital,
                    )
                except Exception as e:
                    logger.warning("strategy_post_trade_error", strategy=strategy, error=str(e))

        diagnostic_mode = confluence_score == 1
        logger.info(
            "paper_trade_placed",
            trade_id=trade.id,
            token_id=token_id,
            side=side,
            price=str(market_price),
            fill_price=str(fill_price),
            size=str(size),
            shares=str(shares),
            edge=str(edge),
            layer=layer,
            confluence=confluence_score,
            diagnostic_mode=diagnostic_mode,
        )

        return trade

    def resolve_market(
        self,
        token_id: str,
        winning_outcome: bool,
    ) -> Decimal:
        """Resolve a market and calculate P&L.

        Args:
            token_id: The token ID of the position.
            winning_outcome: True if this token won (resolved YES).

        Returns:
            Realized P&L for this position.
        """
        if token_id not in self._positions:
            return Decimal("0")

        pos = self._positions[token_id]

        if pos.side == "BUY":
            if winning_outcome:
                # Won: each share pays $1
                payout = pos.shares * Decimal("1")
                pnl = payout - pos.size
            else:
                # Lost: shares worth $0
                pnl = -pos.size
        else:
            if winning_outcome:
                pnl = -pos.shares * (Decimal("1") - pos.avg_price)
            else:
                pnl = pos.shares * pos.avg_price

        # Return invested capital + P&L to balance (minus gas for claim tx)
        self._balance += pos.size + pnl - POLYGON_GAS_COST_USD

        # Track per-layer P&L
        layer = pos.layer
        self._per_layer_realized_pnl[layer] = (
            self._per_layer_realized_pnl.get(layer, Decimal("0")) + pnl
        )

        # Track per-strategy P&L
        if pos.strategy:
            self._per_strategy_realized_pnl[pos.strategy] = (
                self._per_strategy_realized_pnl.get(pos.strategy, Decimal("0")) + pnl
            )

        # Update trades — status reflects outcome, not P&L sign
        # (a winning trade with high entry price can have negative P&L after gas)
        status = TradeStatus.WON if winning_outcome else TradeStatus.LOST
        now = datetime.now(UTC)
        for trade in self._trades:
            if trade.token_id == token_id and trade.status == TradeStatus.OPEN:
                trade.status = status
                trade.actual_pnl = pnl
                trade.resolved_at = now

        # Remove position
        del self._positions[token_id]

        logger.info(
            "paper_trade_resolved",
            token_id=token_id,
            outcome="WIN" if winning_outcome else "LOSE",
            pnl=str(pnl),
            balance=str(self._balance),
            layer=layer,
        )

        return pnl

    def sell_position(
        self,
        token_id: str,
        sell_price: Decimal,
        exit_reason: str,
    ) -> Decimal:
        """Sell a position early (before market resolution).

        Calculates P&L based on sell_price vs entry fill_price, accounts for
        exit slippage and gas. Returns capital + P&L to balance.

        Args:
            token_id: The token ID of the position to sell.
            sell_price: The exit bid price from CLOB (slippage already in price).
            exit_reason: Reason for exit (edge_lost, profit_take, prob_floor).

        Returns:
            Realized P&L for this position.
        """
        if token_id not in self._positions:
            logger.warning("sell_position_not_found", token_id=token_id[:30])
            return Decimal("0")

        pos = self._positions[token_id]

        # P&L: shares × (sell_price - entry_price) for BUY side
        if pos.side == "BUY":
            pnl = pos.shares * (sell_price - pos.avg_price)
        else:
            pnl = pos.shares * (pos.avg_price - sell_price)

        # Gas for sell transaction
        pnl -= POLYGON_GAS_COST_USD

        # Return invested capital + P&L to balance
        self._balance += pos.size + pnl

        # Track per-layer P&L
        layer = pos.layer
        self._per_layer_realized_pnl[layer] = (
            self._per_layer_realized_pnl.get(layer, Decimal("0")) + pnl
        )

        # Track per-strategy P&L
        if pos.strategy:
            self._per_strategy_realized_pnl[pos.strategy] = (
                self._per_strategy_realized_pnl.get(pos.strategy, Decimal("0")) + pnl
            )

        # Update trades — mark as SOLD with exit details
        now = datetime.now(UTC)
        for trade in self._trades:
            if trade.token_id == token_id and trade.status == TradeStatus.OPEN:
                trade.status = TradeStatus.SOLD
                trade.actual_pnl = pnl
                trade.exit_price = sell_price
                trade.exit_reason = exit_reason
                trade.resolved_at = now

        # Remove position
        del self._positions[token_id]

        logger.info(
            "paper_position_sold",
            token_id=token_id[:30],
            side=pos.side,
            entry_price=str(pos.avg_price),
            exit_price=str(sell_price),
            pnl=str(pnl),
            reason=exit_reason,
            balance=str(self._balance),
            strategy=pos.strategy,
        )

        return pnl

    def get_trade_details(self, token_id: str) -> dict | None:
        """Get trade_details for a token_id.

        Checks in-memory trades first, then falls back to DB cache
        (populated during load_state_from_db after restart).
        """
        for trade in reversed(self._trades):
            if trade.token_id == token_id and trade.status == TradeStatus.OPEN:
                return trade.trade_details
        # Fallback: DB cache (loaded at startup). LRU touch on hit.
        td = self._trade_details_cache.get(token_id)
        if td is not None:
            self._trade_details_cache.move_to_end(token_id)
        return td

    def update_position_price(self, token_id: str, current_price: Decimal) -> None:
        """Update current price for an open position (for unrealized P&L)."""
        if token_id in self._positions:
            self._positions[token_id].current_price = current_price

    def take_snapshot(self) -> PortfolioSnapshot:
        """Take a snapshot of current portfolio state."""
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        snapshot = PortfolioSnapshot(
            balance=self._balance,
            unrealized_pnl=unrealized,
            total_value=self.total_value,
            num_open_positions=len(self._positions),
            per_layer_pnl=dict(self._per_layer_realized_pnl),
        )
        self._snapshots.append(snapshot)

        logger.info(
            "portfolio_snapshot",
            balance=str(snapshot.balance),
            unrealized=str(snapshot.unrealized_pnl),
            total_value=str(snapshot.total_value),
            positions=snapshot.num_open_positions,
        )

        return snapshot

    async def save_trade_to_db(self, trade: PaperTrade) -> int | None:
        """Persist a paper trade to the database (best-effort).

        Returns the DB row id (autoincrement PK) on success, None on error.
        Callers needing targeted updates (e.g. per-variant PnL) should use
        this id with update_trade_by_id.
        """
        try:
            from sqlalchemy.exc import SQLAlchemyError

            from arbo.utils.db import PaperTrade as PaperTradeDB
            from arbo.utils.db import get_session_factory

            factory = get_session_factory()
            async with factory() as session:
                db_trade = PaperTradeDB(
                    market_condition_id=trade.market_condition_id,
                    token_id=trade.token_id,
                    layer=trade.layer,
                    side=trade.side,
                    price=trade.price,
                    size=trade.size,
                    slippage=trade.fill_price - trade.price,
                    edge_at_exec=trade.edge,
                    confluence_score=trade.confluence_score,
                    kelly_fraction=trade.kelly_fraction,
                    status=trade.status.value,
                    fee_paid=trade.fee,
                    placed_at=trade.placed_at,
                    notes=trade.notes or None,
                    strategy=trade.strategy or None,
                    trade_details=trade.trade_details,
                )
                session.add(db_trade)
                await session.commit()
                await session.refresh(db_trade)
                return int(db_trade.id)
        except (SQLAlchemyError, ValueError, TypeError) as e:
            logger.warning("save_trade_to_db_failed", error=str(e))
            return None

    async def save_snapshot_to_db(self, snapshot: PortfolioSnapshot) -> None:
        """Persist a portfolio snapshot to the database (best-effort)."""
        try:
            from sqlalchemy.exc import SQLAlchemyError

            from arbo.utils.db import PaperSnapshot as PaperSnapshotDB
            from arbo.utils.db import get_session_factory

            factory = get_session_factory()
            async with factory() as session:
                db_snap = PaperSnapshotDB(
                    balance=snapshot.balance,
                    unrealized_pnl=snapshot.unrealized_pnl,
                    total_value=snapshot.total_value,
                    num_open_positions=snapshot.num_open_positions,
                    per_layer_pnl={str(k): str(v) for k, v in snapshot.per_layer_pnl.items()},
                    snapshot_at=snapshot.snapshot_at,
                )
                session.add(db_snap)
                await session.commit()
        except (SQLAlchemyError, ValueError, TypeError) as e:
            logger.warning("save_snapshot_to_db_failed", error=str(e))

    async def sync_positions_to_db(self) -> None:
        """Upsert current open positions to the database (best-effort)."""
        try:
            from sqlalchemy import delete
            from sqlalchemy.exc import SQLAlchemyError

            from arbo.utils.db import PaperPosition as PaperPositionDB
            from arbo.utils.db import get_session_factory

            factory = get_session_factory()
            async with factory() as session:
                await session.execute(delete(PaperPositionDB))
                for pos in self._positions.values():
                    db_pos = PaperPositionDB(
                        market_condition_id=pos.market_condition_id,
                        token_id=pos.token_id,
                        side=pos.side,
                        avg_price=pos.avg_price,
                        size=pos.size,
                        current_price=pos.current_price,
                        unrealized_pnl=pos.unrealized_pnl,
                        layer=pos.layer,
                        strategy=pos.strategy or None,
                        opened_at=pos.opened_at,
                    )
                    session.add(db_pos)
                await session.commit()
        except (SQLAlchemyError, ValueError, TypeError) as e:
            logger.warning("sync_positions_to_db_failed", error=str(e))

    async def update_resolved_trades_in_db(
        self,
        token_id: str,
        winning: bool | None = None,
        pnl: Decimal | None = None,
        exit_price: Decimal | None = None,
        exit_reason: str | None = None,
    ) -> None:
        """Update DB rows for trades that were just resolved or sold (best-effort).

        Args:
            token_id: The token ID of the resolved/sold position.
            winning: True if this token won. If provided with pnl, updates DB
                directly without needing in-memory trade data (works after restart).
            pnl: Realized P&L. Required when winning is provided.
            exit_price: Fill price on early exit (SOLD status).
            exit_reason: Reason for early exit.
        """
        try:
            from sqlalchemy import update

            from arbo.utils.db import PaperTrade as PaperTradeDB
            from arbo.utils.db import get_session_factory

            now = datetime.now(UTC)

            if exit_reason is not None and pnl is not None:
                # Early exit path
                status_str = "sold"
                actual_pnl = pnl
                resolved_at = now
            elif winning is not None and pnl is not None:
                # Direct resolution path
                status_str = "won" if winning else "lost"
                actual_pnl = pnl
                resolved_at = now
                exit_price = None
                exit_reason = None
            else:
                # Legacy path — find in-memory trade data
                resolved_trades = [
                    t
                    for t in self._trades
                    if t.token_id == token_id
                    and t.status in (TradeStatus.WON, TradeStatus.LOST, TradeStatus.SOLD)
                    and t.resolved_at is not None
                ]
                if not resolved_trades:
                    logger.warning(
                        "update_resolved_no_inmemory_trade",
                        token_id=token_id[:30],
                    )
                    return
                t = resolved_trades[0]
                status_str = t.status.value
                actual_pnl = t.actual_pnl
                resolved_at = t.resolved_at
                exit_price = t.exit_price
                exit_reason = t.exit_reason

            factory = get_session_factory()
            async with factory() as session:
                values: dict = {
                    "status": status_str,
                    "actual_pnl": actual_pnl,
                    "resolved_at": resolved_at,
                }
                if exit_price is not None:
                    values["exit_price"] = exit_price
                if exit_reason is not None:
                    values["exit_reason"] = exit_reason

                # Exclude shadow variants from blanket token_id update.
                # Shadow variants have independent resolution (update_trade_by_id)
                # based on their own entry/params/exit. Blanket-updating them
                # when champion's mirror cancels or resolves would corrupt
                # per-variant PnL history.
                from sqlalchemy import or_, not_
                shadow_expr = PaperTradeDB.trade_details.op("->>")("is_shadow_variant")
                await session.execute(
                    update(PaperTradeDB)
                    .where(
                        PaperTradeDB.token_id == token_id,
                        PaperTradeDB.status == "open",
                        or_(shadow_expr.is_(None), shadow_expr != "true"),
                    )
                    .values(**values)
                )
                await session.commit()
        except Exception as e:
            logger.warning("update_resolved_trades_db_failed", error=str(e), token_id=token_id)

    async def update_trade_by_id(
        self,
        trade_id: int,
        status: str,
        actual_pnl: Decimal | None = None,
        exit_price: Decimal | None = None,
        exit_reason: str | None = None,
    ) -> None:
        """Update ONE specific paper_trades row by primary key.

        Per-variant PnL tracking (Project PARALLEL): when multiple variants
        trade the same token with same/different params, update_resolved_
        trades_in_db would blanket-update all rows sharing token_id. This
        helper targets exactly one row by id — preserves independent
        per-variant PnL histories.
        """
        try:
            from sqlalchemy import update
            from arbo.utils.db import PaperTrade as PaperTradeDB
            from arbo.utils.db import get_session_factory

            now = datetime.now(UTC)
            values: dict = {"status": status, "resolved_at": now}
            if actual_pnl is not None:
                values["actual_pnl"] = actual_pnl
            if exit_price is not None:
                values["exit_price"] = exit_price
            if exit_reason is not None:
                values["exit_reason"] = exit_reason

            factory = get_session_factory()
            async with factory() as session:
                await session.execute(
                    update(PaperTradeDB)
                    .where(PaperTradeDB.id == trade_id)
                    .values(**values)
                )
                await session.commit()
        except Exception as e:
            logger.warning(
                "update_trade_by_id_failed", trade_id=trade_id, error=str(e),
            )

    async def load_state_from_db(self) -> None:
        """Restore open positions and balance from the database on startup."""
        try:
            from sqlalchemy import select
            from sqlalchemy.exc import SQLAlchemyError

            from arbo.utils.db import PaperPosition as PaperPositionDB
            from arbo.utils.db import PaperSnapshot as PaperSnapshotDB
            from arbo.utils.db import get_session_factory

            factory = get_session_factory()
            async with factory() as session:
                # Load positions
                result = await session.execute(select(PaperPositionDB))
                db_positions = result.scalars().all()
                for db_pos in db_positions:
                    pos = PaperPosition(
                        market_condition_id=db_pos.market_condition_id,
                        token_id=db_pos.token_id,
                        side=db_pos.side,
                        avg_price=Decimal(str(db_pos.avg_price)),
                        size=Decimal(str(db_pos.size)),
                        shares=(
                            Decimal(str(db_pos.size)) / Decimal(str(db_pos.avg_price))
                            if db_pos.avg_price
                            else Decimal("0")
                        ),
                        layer=db_pos.layer,
                        strategy=getattr(db_pos, "strategy", "") or "",
                        current_price=(
                            Decimal(str(db_pos.current_price)) if db_pos.current_price else None
                        ),
                        opened_at=db_pos.opened_at or datetime.now(UTC),
                    )
                    self._positions[pos.token_id] = pos

                # Load trade_details for open positions (needed for METAR resolution)
                if self._positions:
                    from arbo.utils.db import PaperTrade as PaperTradeDB

                    token_ids = list(self._positions.keys())
                    td_result = await session.execute(
                        select(PaperTradeDB.token_id, PaperTradeDB.trade_details)
                        .where(PaperTradeDB.token_id.in_(token_ids))
                        .where(PaperTradeDB.status == "open")
                        .where(PaperTradeDB.trade_details.isnot(None))
                    )
                    for td_row in td_result:
                        if td_row[1] and isinstance(td_row[1], dict):
                            # LRU insert with eviction
                            self._trade_details_cache[td_row[0]] = td_row[1]
                            self._trade_details_cache.move_to_end(td_row[0])
                            if len(self._trade_details_cache) > MAX_TRADE_DETAILS_CACHE:
                                self._trade_details_cache.popitem(last=False)
                    logger.info(
                        "trade_details_cache_loaded",
                        cached=len(self._trade_details_cache),
                        positions=len(self._positions),
                    )

                # Load last snapshot for balance
                result = await session.execute(
                    select(PaperSnapshotDB).order_by(PaperSnapshotDB.snapshot_at.desc()).limit(1)
                )
                last_snap = result.scalars().first()
                if last_snap is not None:
                    snapshot_balance = Decimal(str(last_snap.balance))

                    # D5 Bug 6: Cross-check snapshot balance against computed balance
                    # computed = initial_capital - sum(open position sizes)
                    total_invested = sum(p.size for p in self._positions.values())
                    computed_balance = self._initial_capital - total_invested
                    self._balance = min(snapshot_balance, computed_balance)

                    drift = abs(snapshot_balance - computed_balance)
                    if drift > Decimal("50"):
                        logger.warning(
                            "balance_drift_detected",
                            snapshot_balance=str(snapshot_balance),
                            computed_balance=str(computed_balance),
                            drift=str(drift),
                            using=str(self._balance),
                        )

                    logger.info(
                        "state_restored_from_db",
                        positions=len(self._positions),
                        balance=str(self._balance),
                        snapshot_balance=str(snapshot_balance),
                        computed_balance=str(computed_balance),
                    )
                else:
                    logger.info("no_db_state_found", using="initial_capital")

        except (SQLAlchemyError, ImportError, ValueError, TypeError) as e:
            logger.warning("load_state_from_db_failed", error=str(e))

    def get_stats(self) -> dict[str, object]:
        """Get summary statistics."""
        resolved = [t for t in self._trades if t.status in (TradeStatus.WON, TradeStatus.LOST)]
        wins = [t for t in resolved if t.status == TradeStatus.WON]
        losses = [t for t in resolved if t.status == TradeStatus.LOST]

        total_pnl = sum((t.actual_pnl for t in resolved if t.actual_pnl is not None), Decimal("0"))

        return {
            "initial_capital": self._initial_capital,
            "current_balance": self._balance,
            "total_value": self.total_value,
            "total_trades": len(self._trades),
            "open_trades": len([t for t in self._trades if t.status == TradeStatus.OPEN]),
            "resolved_trades": len(resolved),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(resolved) if resolved else 0,
            "total_pnl": total_pnl,
            "roi_pct": (total_pnl / self._initial_capital * 100) if self._initial_capital else 0,
            "per_layer_pnl": dict(self._per_layer_realized_pnl),
            "per_strategy_pnl": dict(self._per_strategy_realized_pnl),
        }

    def get_strategy_stats(self, strategy_id: str) -> dict[str, object]:
        """Get statistics for a specific RDH strategy (A, B, or C)."""
        strategy_trades = [t for t in self._trades if t.strategy == strategy_id]
        resolved = [
            t for t in strategy_trades if t.status in (TradeStatus.WON, TradeStatus.LOST)
        ]
        wins = [t for t in resolved if t.status == TradeStatus.WON]
        total_pnl = sum(
            (t.actual_pnl for t in resolved if t.actual_pnl is not None), Decimal("0")
        )
        total_invested = sum(t.size for t in strategy_trades)

        return {
            "strategy": strategy_id,
            "total_trades": len(strategy_trades),
            "open_trades": len([t for t in strategy_trades if t.status == TradeStatus.OPEN]),
            "resolved_trades": len(resolved),
            "wins": len(wins),
            "losses": len(resolved) - len(wins),
            "win_rate": len(wins) / len(resolved) if resolved else 0,
            "total_pnl": total_pnl,
            "total_invested": total_invested,
            "roi_pct": (total_pnl / total_invested * 100) if total_invested else Decimal("0"),
        }
