"""Paper trading engine for simulating trades without real capital.

Accepts trade signals, simulates fills at current midprice with configurable
slippage, tracks P&L per-trade, per-strategy, per-day, per-week.

See brief PM-004 for full specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum

from arbo.core.fee_model import calculate_taker_fee
from arbo.core.risk_manager import RiskManager, TradeRequest
from arbo.utils.logger import get_logger
from arbo.utils.odds import half_kelly

logger = get_logger("paper_engine")


class TradeStatus(Enum):
    OPEN = "open"
    WON = "won"
    LOST = "lost"
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
    placed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    resolved_at: datetime | None = None
    notes: str = ""


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
        self._trades: list[PaperTrade] = []
        self._positions: dict[str, PaperPosition] = {}  # key: token_id
        self._snapshots: list[PortfolioSnapshot] = []
        self._next_trade_id = 1
        self._per_layer_realized_pnl: dict[int, Decimal] = {}
        self._per_strategy_realized_pnl: dict[str, Decimal] = {}

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

        if edge <= Decimal("0"):
            logger.debug(
                "paper_trade_no_edge",
                token_id=token_id,
                edge=str(edge),
                layer=layer,
            )
            return None

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

        # Risk manager check
        if self._risk_manager:
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

        # D5 Bug 4: Balance floor — never go negative
        if size > self._balance:
            logger.info(
                "paper_trade_insufficient_balance",
                size=str(size),
                balance=str(self._balance),
                token_id=token_id,
            )
            return None

        # Apply slippage
        if side == "BUY":
            fill_price = market_price * (Decimal("1") + self._slippage_pct)
        else:
            fill_price = market_price * (Decimal("1") - self._slippage_pct)

        fill_price = min(max(fill_price, Decimal("0.001")), Decimal("0.999"))

        # Calculate shares
        shares = (size / fill_price).quantize(Decimal("0.01"))

        # Deduct from balance
        self._balance -= size

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
            fee=fee * size,
            strategy=strategy,
        )
        self._next_trade_id += 1
        self._trades.append(trade)

        # Update or create position
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

        # Update risk manager
        if self._risk_manager:
            self._risk_manager.post_trade_update(
                market_id=market_condition_id,
                market_category=market_category,
                size=size,
            )

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

        # Update trades
        status = TradeStatus.WON if pnl > 0 else TradeStatus.LOST
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

    async def save_trade_to_db(self, trade: PaperTrade) -> None:
        """Persist a paper trade to the database (best-effort)."""
        try:
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
                )
                session.add(db_trade)
                await session.commit()
        except Exception as e:
            logger.warning("save_trade_to_db_failed", error=str(e))

    async def save_snapshot_to_db(self, snapshot: PortfolioSnapshot) -> None:
        """Persist a portfolio snapshot to the database (best-effort)."""
        try:
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
        except Exception as e:
            logger.warning("save_snapshot_to_db_failed", error=str(e))

    async def sync_positions_to_db(self) -> None:
        """Upsert current open positions to the database (best-effort)."""
        try:
            from sqlalchemy import delete

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
                    )
                    session.add(db_pos)
                await session.commit()
        except Exception as e:
            logger.warning("sync_positions_to_db_failed", error=str(e))

    async def update_resolved_trades_in_db(self, token_id: str) -> None:
        """Update DB rows for trades that were just resolved (best-effort)."""
        try:
            from sqlalchemy import update

            from arbo.utils.db import PaperTrade as PaperTradeDB
            from arbo.utils.db import get_session_factory

            # Find the resolved in-memory trade data
            resolved_trades = [
                t for t in self._trades
                if t.token_id == token_id and t.status in (TradeStatus.WON, TradeStatus.LOST)
                and t.resolved_at is not None
            ]
            if not resolved_trades:
                return

            factory = get_session_factory()
            async with factory() as session:
                await session.execute(
                    update(PaperTradeDB)
                    .where(
                        PaperTradeDB.token_id == token_id,
                        PaperTradeDB.status == "open",
                    )
                    .values(
                        status=resolved_trades[0].status.value,
                        actual_pnl=resolved_trades[0].actual_pnl,
                        resolved_at=resolved_trades[0].resolved_at,
                    )
                )
                await session.commit()
        except Exception as e:
            logger.warning("update_resolved_trades_db_failed", error=str(e), token_id=token_id)

    async def load_state_from_db(self) -> None:
        """Restore open positions and balance from the database on startup."""
        try:
            from sqlalchemy import select

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
                        current_price=(
                            Decimal(str(db_pos.current_price)) if db_pos.current_price else None
                        ),
                    )
                    self._positions[pos.token_id] = pos

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

        except Exception as e:
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
