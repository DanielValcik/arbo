"""Strategy D — Shared Core Engine.

Sport-agnostic green book engine. Used by sport-specific variants:
  - strategy_d_nba.py (NBA)     → STRATEGY_NAME="D"
  - strategy_d_ufc.py (UFC)     → STRATEGY_NAME="D_UFC"  [planned]
  - strategy_d_nfl.py (NFL)     → STRATEGY_NAME="D_NFL"  [planned]
  - strategy_d_epl.py (EPL)     → STRATEGY_NAME="D_EPL"  [planned]

Architecture: see docs/STRATEGY_D_ARCHITECTURE.md

Provides:
  - Generic MarketData / Signal / Position dataclasses
  - Kelly sizing
  - Green book walk, stop loss, time exit
  - Elo + Pinnacle ensemble probability model
  - Risk manager / paper / live dispatch
  - Status reporting

Sport variants override class attributes:
  SPORT_NAME, STRATEGY_NAME, MIN_EDGE, GREEN_BOOK_DELTA, STOP_LOSS_DELTA,
  MAX_HOLD_FRACTION, BOTH_SIDES, GAME_DURATION_HOURS, team_map
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from arbo.core.risk_manager import RiskManager, TradeRequest
from arbo.utils.logger import get_logger


# ── Generic dataclasses ───────────────────────────────────────────────

@dataclass
class MarketData:
    """Generic sports market data. Sport-specific discovery populates this."""
    sport: str            # "nba", "ufc", "nfl", "epl"
    condition_id: str
    token_id_yes: str
    token_id_no: str
    question: str
    team_a: str          # Parsed team abbreviation
    team_b: str
    game_date: str       # YYYY-MM-DD
    game_time: str | None
    yes_price: float
    no_price: float
    volume: float
    neg_risk: bool


@dataclass
class DSignal:
    """Strategy D trade signal (sport-agnostic)."""
    market: MarketData
    side: str            # "yes" or "no"
    token_id: str
    entry_price: float
    model_prob: float
    edge: float
    kelly_size: float    # Dollar amount


@dataclass
class DPosition:
    """Open Strategy D position (sport-agnostic)."""
    sport: str
    condition_id: str
    token_id: str
    side: str            # "yes" or "no"
    entry_price: float
    entry_time: float
    model_prob: float
    edge: float
    shares: int
    cost: float
    question: str
    team_a: str
    team_b: str
    game_date: str
    max_price: float = 0.0
    min_price: float = 0.0
    n_price_checks: int = 0
    exit_price: float = 0.0
    exit_time: float = 0.0
    exit_reason: str = ""
    pnl: float = 0.0
    # CLV (Closing Line Value) — measures edge vs market's final price.
    # Positive CLV = entry was cheaper than close → real edge.
    # Research: CLV > 2¢ over 500 trades = long-term profitable strategy.
    close_price: float = 0.0   # Last price seen before resolution/exit
    clv: float = 0.0           # close_price - entry_price (for YES side)
    neg_risk: bool = False     # Critical for orderbook queries
    live_shares: int = 0
    live_entry_price: float = 0.0
    live_fill_status: str = ""


# ── Core class ────────────────────────────────────────────────────────

class StrategyDCore:
    """Shared green book engine. Sport variants inherit and override class attrs."""

    # ── Subclass overrides ────────────────────────────────────────────
    SPORT_NAME: str = "generic"
    STRATEGY_NAME: str = "D"
    STRATEGY_LABEL: str = "Green Book"

    # Quality gate
    MIN_EDGE: float = 0.16
    MAX_EDGE: float = 0.25
    MIN_PRICE: float = 0.20
    MAX_PRICE: float = 0.65

    # Green book
    GREEN_BOOK_DELTA: float = 0.17
    STOP_LOSS_DELTA: float = 0.15
    MAX_HOLD_FRACTION: float = 0.50
    GAME_DURATION_HOURS: float = 2.5  # NBA default; sport variants override

    # Trading
    BOTH_SIDES: bool = True
    MAX_CONCURRENT: int = 8
    COOLDOWN_AFTER_TRADE_S: int = 60

    # Sizing
    KELLY_FRACTION: float = 0.15
    KELLY_RAW_CAP: float = 0.10
    MAX_POSITION_PCT: float = 0.03

    # Model weights
    ELO_WEIGHT: float = 0.40
    PINNACLE_WEIGHT: float = 0.60

    # Risk layer (for TradeRequest)
    RISK_LAYER: int = 9  # Sports

    # ── Init ──────────────────────────────────────────────────────────

    def __init__(
        self,
        risk_manager: RiskManager,
        paper_engine: Any | None = None,
        live_executor: Any | None = None,
        orderbook_provider: Any | None = None,
        elo_ratings: dict[str, tuple[float, float]] | None = None,
        pinnacle_odds: dict[str, tuple[float, float]] | None = None,
    ):
        self._risk = risk_manager
        self._paper = paper_engine
        self._live = live_executor
        self._ob = orderbook_provider

        self._elo: dict[str, tuple[float, float]] = elo_ratings or {}
        self._pinnacle: dict[str, tuple[float, float]] = pinnacle_odds or {}

        self._positions: dict[str, DPosition] = {}
        self._last_trade_time: float = 0.0
        self._trades_today: int = 0
        self._daily_pnl: float = 0.0

        self._logger = get_logger(f"strategy_d_{self.SPORT_NAME}")
        self._logger.info(
            f"{self.__class__.__name__}_initialized",
            sport=self.SPORT_NAME,
            strategy=self.STRATEGY_NAME,
            min_edge=self.MIN_EDGE,
            delta=self.GREEN_BOOK_DELTA,
            both_sides=self.BOTH_SIDES,
        )

    # ── Model ─────────────────────────────────────────────────────────

    def compute_model_prob(self, team_a: str, team_b: str) -> float | None:
        """Probability of team_a winning, using Elo + Pinnacle ensemble."""
        elo_a = self._elo.get(team_a)
        elo_b = self._elo.get(team_b)

        elo_prob = None
        if elo_a and elo_b:
            diff = elo_a[0] - elo_b[0]
            elo_prob = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

        pin_prob = None
        # Try both team orderings with sport prefix
        for key_teams in [(team_a, team_b), (team_b, team_a)]:
            pin_key = f"{self.SPORT_NAME}_{key_teams[0]}_{key_teams[1]}"
            if pin_key in self._pinnacle:
                hp, ap = self._pinnacle[pin_key]
                pin_prob = hp if key_teams[0] == team_a else ap
                break

        if elo_prob is not None and pin_prob is not None:
            return self.ELO_WEIGHT * elo_prob + self.PINNACLE_WEIGHT * pin_prob
        return elo_prob or pin_prob

    # ── Signal Generation ─────────────────────────────────────────────

    def generate_signals(self, markets: list[MarketData]) -> list[DSignal]:
        """Scan markets for entry signals."""
        signals: list[DSignal] = []

        for market in markets:
            if market.condition_id in self._positions:
                continue

            prob = self.compute_model_prob(market.team_a, market.team_b)
            if prob is None:
                continue

            # YES side
            yes_edge = prob - market.yes_price
            if (self.MIN_EDGE <= yes_edge <= self.MAX_EDGE
                    and self.MIN_PRICE <= market.yes_price <= self.MAX_PRICE):
                size = self.kelly_size(yes_edge, market.yes_price)
                if size > 0:
                    signals.append(DSignal(
                        market=market, side="yes", token_id=market.token_id_yes,
                        entry_price=market.yes_price, model_prob=prob,
                        edge=yes_edge, kelly_size=size,
                    ))

            # NO side
            if self.BOTH_SIDES:
                no_edge = (1 - prob) - market.no_price
                if (self.MIN_EDGE <= no_edge <= self.MAX_EDGE
                        and self.MIN_PRICE <= market.no_price <= self.MAX_PRICE):
                    size = self.kelly_size(no_edge, market.no_price)
                    if size > 0:
                        signals.append(DSignal(
                            market=market, side="no", token_id=market.token_id_no,
                            entry_price=market.no_price, model_prob=1 - prob,
                            edge=no_edge, kelly_size=size,
                        ))

        self._logger.info(
            "scan_complete", sport=self.SPORT_NAME,
            markets=len(markets), signals=len(signals), positions=len(self._positions),
        )
        return signals

    def kelly_size(self, edge: float, price: float) -> float:
        """Kelly position size in USDC."""
        if price <= 0 or price >= 1:
            return 0.0
        p = max(0.01, min(0.99, price + edge))
        q = 1 - p
        b = (1 / price) - 1
        if b <= 0:
            return 0.0
        kelly_raw = max(0, min((b * p - q) / b, self.KELLY_RAW_CAP))
        ss = self._risk.get_strategy_state(self.STRATEGY_NAME)
        capital = float(ss.allocated) if ss else 300.0
        size = capital * self.KELLY_FRACTION * kelly_raw
        return min(size, capital * self.MAX_POSITION_PCT)

    # ── Entry ─────────────────────────────────────────────────────────

    async def execute_entry(self, signal: DSignal) -> DPosition | None:
        """Execute trade entry (paper or live)."""
        now = time.time()

        if now - self._last_trade_time < self.COOLDOWN_AFTER_TRADE_S:
            return None
        if len(self._positions) >= self.MAX_CONCURRENT:
            return None

        request = TradeRequest(
            market_id=signal.market.condition_id,
            token_id=signal.token_id,
            side="BUY",
            price=Decimal(str(signal.entry_price)),
            size=Decimal(str(signal.kelly_size)),
            layer=self.RISK_LAYER,
            market_category="Sports",
            strategy=self.STRATEGY_NAME,
        )
        decision = self._risk.pre_trade_check(request)
        if not decision.approved:
            return None

        shares = int(signal.kelly_size / signal.entry_price)
        if shares < 1:
            return None

        position = DPosition(
            sport=self.SPORT_NAME,
            condition_id=signal.market.condition_id,
            token_id=signal.token_id,
            side=signal.side,
            entry_price=signal.entry_price,
            entry_time=now,
            model_prob=signal.model_prob,
            edge=signal.edge,
            shares=shares,
            cost=shares * signal.entry_price,
            question=signal.market.question,
            team_a=signal.market.team_a,
            team_b=signal.market.team_b,
            game_date=signal.market.game_date,
            max_price=signal.entry_price,
            min_price=signal.entry_price,
            neg_risk=bool(signal.market.neg_risk),
        )

        if self._paper:
            trade_details = {
                "sport": self.SPORT_NAME,
                "side": signal.side,
                "team_a": signal.market.team_a,
                "team_b": signal.market.team_b,
                "question": signal.market.question,
                "edge": round(signal.edge, 4),
                "game_date": signal.market.game_date,
            }
            trade = self._paper.place_trade(
                market_condition_id=signal.market.condition_id,
                token_id=signal.token_id,
                side="BUY",
                market_price=Decimal(str(signal.entry_price)),
                model_prob=Decimal(str(signal.model_prob)),
                layer=self.RISK_LAYER,
                market_category="Sports",
                strategy=self.STRATEGY_NAME,
                trade_details=trade_details,
            )
            if trade:
                position.live_fill_status = "paper"
                self._logger.info(
                    "entry_paper", sport=self.SPORT_NAME, side=signal.side,
                    team_a=signal.market.team_a, team_b=signal.market.team_b,
                    price=signal.entry_price, edge=f"{signal.edge:.3f}", shares=shares,
                )

        if self._live:
            try:
                # Use LiveExecutor.buy() — MAKER order at best buy price (0% fee + rebate)
                neg_risk = bool(signal.market.neg_risk)
                size_usdc = shares * signal.entry_price
                fill = await self._live.buy(
                    token_id=signal.token_id,
                    price=signal.entry_price,
                    size_usdc=size_usdc,
                    neg_risk=neg_risk,
                    tick_size="0.01",
                    max_price=signal.entry_price + 0.02,  # 2¢ slippage cap
                )
                if fill and getattr(fill, "filled_shares", 0) > 0:
                    position.live_shares = int(fill.filled_shares)
                    position.live_entry_price = float(fill.fill_price)
                    position.live_fill_status = fill.status
                    self._logger.info(
                        "entry_live", sport=self.SPORT_NAME, side=signal.side,
                        price=position.live_entry_price, shares=position.live_shares,
                        status=fill.status,
                    )
                else:
                    position.live_fill_status = "skipped"
                    self._logger.info(
                        "entry_live_skipped", sport=self.SPORT_NAME,
                        reason=getattr(fill, "status", "no_fill"),
                    )
            except Exception as e:
                self._logger.error("live_entry_error", error=str(e))
                position.live_fill_status = "failed"

        self._positions[signal.market.condition_id] = position
        self._last_trade_time = now
        self._trades_today += 1
        return position

    # ── Exit ──────────────────────────────────────────────────────────

    def check_exits(self, current_prices: dict[str, float]) -> list[DPosition]:
        """Check all open positions for exit conditions."""
        exits: list[DPosition] = []
        now = time.time()

        for cid, pos in list(self._positions.items()):
            price = current_prices.get(pos.token_id)
            if price is None:
                continue

            pos.n_price_checks += 1
            pos.max_price = max(pos.max_price, price)
            pos.min_price = min(pos.min_price, price) if pos.min_price > 0 else price

            if pos.side == "yes":
                gb_target = pos.entry_price + self.GREEN_BOOK_DELTA
                sl_trigger = pos.entry_price - self.STOP_LOSS_DELTA
                gb_hit = price >= gb_target
                sl_hit = price <= sl_trigger
            else:
                gb_target = pos.entry_price - self.GREEN_BOOK_DELTA
                sl_trigger = pos.entry_price + self.STOP_LOSS_DELTA
                gb_hit = price <= gb_target
                sl_hit = price >= sl_trigger

            hold_hours = (now - pos.entry_time) / 3600
            time_exit = hold_hours >= (self.GAME_DURATION_HOURS * self.MAX_HOLD_FRACTION)

            if gb_hit:
                pos.exit_reason = "green_book"
                pos.exit_price = price
            elif sl_hit:
                pos.exit_reason = "stop_loss"
                pos.exit_price = price
            elif time_exit:
                pos.exit_reason = "time_exit"
                pos.exit_price = price
            else:
                continue

            pos.exit_time = now
            pos.close_price = price  # Last observed price = proxy for closing line
            if pos.side == "yes":
                pos.pnl = pos.shares * (pos.exit_price - pos.entry_price)
                pos.clv = price - pos.entry_price  # Positive = entry was underpriced
            else:
                pos.pnl = pos.shares * (pos.entry_price - pos.exit_price)
                pos.clv = pos.entry_price - price

            exits.append(pos)
            del self._positions[cid]
            self._daily_pnl += pos.pnl

            # Persist exit details + CLV to PaperTrade row directly
            if self._paper is not None:
                self._persist_exit_clv(pos, now)

            self._logger.info(
                "exit", sport=self.SPORT_NAME, reason=pos.exit_reason, side=pos.side,
                entry=f"{pos.entry_price:.3f}", exit_p=f"{pos.exit_price:.3f}",
                pnl=f"${pos.pnl:.2f}", clv=f"{pos.clv*100:+.2f}c",
                team_a=pos.team_a, team_b=pos.team_b,
                hold_min=f"{(now - pos.entry_time)/60:.0f}",
            )

        return exits

    # ── Status ────────────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Status for dashboard."""
        return {
            "strategy": self.STRATEGY_NAME,
            "sport": self.SPORT_NAME,
            "label": self.STRATEGY_LABEL,
            "open_positions": len(self._positions),
            "trades_today": self._trades_today,
            "daily_pnl": round(self._daily_pnl, 2),
            "params": {
                "min_edge": self.MIN_EDGE,
                "delta": self.GREEN_BOOK_DELTA,
                "stop_loss": self.STOP_LOSS_DELTA,
                "max_hold": self.MAX_HOLD_FRACTION,
                "both_sides": self.BOTH_SIDES,
            },
            "positions": [
                {
                    "sport": p.sport,
                    "side": p.side,
                    "team_a": p.team_a,
                    "team_b": p.team_b,
                    "entry": p.entry_price,
                    "edge": round(p.edge, 3),
                    "shares": p.shares,
                    "hold_min": round((time.time() - p.entry_time) / 60),
                }
                for p in self._positions.values()
            ],
        }

    def reset_daily(self) -> None:
        """Reset daily counters (called at midnight)."""
        self._trades_today = 0
        self._daily_pnl = 0.0

    def _persist_exit_clv(self, pos: DPosition, now: float) -> None:
        """Update PaperTrade row with exit details + CLV (best-effort, sync).

        Finds the open trade for this token_id and updates it with exit_price,
        exit_reason, actual_pnl, and CLV in trade_details.
        """
        try:
            from decimal import Decimal
            import json as _json
            from sqlalchemy import select, update
            from arbo.utils.db import PaperTrade, get_session_factory

            factory = get_session_factory()

            async def _do_update():
                async with factory() as session:
                    result = await session.execute(
                        select(PaperTrade)
                        .where(PaperTrade.token_id == pos.token_id)
                        .where(PaperTrade.status == "open")
                        .order_by(PaperTrade.placed_at.desc())
                        .limit(1)
                    )
                    row = result.scalar_one_or_none()
                    if row is None:
                        return
                    details = dict(row.trade_details) if row.trade_details else {}
                    details.update({
                        "sport": pos.sport,
                        "close_price": pos.close_price,
                        "clv": pos.clv,
                        "hold_min": round((now - pos.entry_time) / 60),
                    })
                    await session.execute(
                        update(PaperTrade)
                        .where(PaperTrade.id == row.id)
                        .values(
                            status="sold" if pos.exit_reason != "stop_loss" else "sold",
                            exit_price=Decimal(str(pos.exit_price)),
                            exit_reason=pos.exit_reason,
                            actual_pnl=Decimal(str(pos.pnl)),
                            resolved_at=__import__("datetime").datetime.fromtimestamp(now, tz=__import__("datetime").timezone.utc),
                            trade_details=details,
                        )
                    )
                    await session.commit()

            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_do_update())
            else:
                loop.run_until_complete(_do_update())
        except Exception as e:
            self._logger.warning("persist_exit_clv_failed", error=str(e))
