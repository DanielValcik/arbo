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

import asyncio
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

        # Project PARALLEL: variant_params override class attrs when set.
        # Default None → use class-attr defaults (current production behavior).
        # Promotion swaps this to new champion's YAML params at runtime
        # without restart.
        self._active_variant_params: dict | None = None
        # Shadow variant cache + write dedup
        self._variants_cache: list = []
        self._variants_cache_ts: float = 0.0
        self._shadow_written_d: set[tuple[str, str, int]] = set()
        self._last_shadow_sweep_ts_d: float = 0.0

        self._elo: dict[str, tuple[float, float]] = elo_ratings or {}
        self._pinnacle: dict[str, tuple[float, float]] = pinnacle_odds or {}

        self._positions: dict[str, DPosition] = {}
        self._last_trade_time: float = 0.0
        self._trades_today: int = 0
        self._daily_pnl: float = 0.0
        # Cooldown per market: don't re-enter same condition_id for 4 hours
        # (one NBA game is 2.5h; by 4h any signal should be stale).
        # Persisted to JSON across restarts — fixes POR/PHX-style restart
        # loop where gamma cache serves stale prices for recently-resolved
        # markets.
        self._recent_exits: dict[str, float] = self._load_recent_exits()
        # Live balance cache (updated externally by orchestrator every 5 min)
        # None = use fallback allocation from risk_manager
        self._live_balance_usdc: float | None = None
        self._live_balance_ts: float = 0.0

        self._logger = get_logger(f"strategy_d_{self.SPORT_NAME}")
        self._logger.info(
            f"{self.__class__.__name__}_initialized",
            sport=self.SPORT_NAME,
            strategy=self.STRATEGY_NAME,
            min_edge=self.MIN_EDGE,
            delta=self.GREEN_BOOK_DELTA,
            both_sides=self.BOTH_SIDES,
        )

    # ── Variant param accessor (Project PARALLEL) ─────────────────────

    def _p(self, name: str) -> Any:
        """Param accessor — variant override > class attr fallback.

        Used for Tier 1 params that PromotionEngine may swap at runtime.
        Tier 2/3 params still read directly via `self.X`.
        """
        if self._active_variant_params is not None:
            v = self._active_variant_params.get(name)
            if v is not None:
                return v
        return getattr(self, name)

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

    # Cooldown: don't re-enter same market for this many seconds after exit
    REENTRY_COOLDOWN_S = 4 * 3600  # 4 hours (covers any sport game duration)

    # ── Recent-exits persistence (restart-safe cooldown) ──────────────

    def _recent_exits_path(self) -> str:
        """Per-sport cooldown state file on disk.

        Default path lives under /opt/arbo/logs (writable on Dublin VPS
        with ProtectSystem=strict systemd sandbox). Override with
        ARBO_STATE_DIR env var for dev/test.
        """
        import os
        default = "/opt/arbo/logs/state"
        base = os.environ.get("ARBO_STATE_DIR", default)
        try:
            os.makedirs(base, exist_ok=True)
        except (OSError, PermissionError):
            # Fallback to /tmp if /opt/arbo/logs not writable (local dev)
            base = "/tmp/arbo_state"
            os.makedirs(base, exist_ok=True)
        return os.path.join(base, f"d_recent_exits_{self.SPORT_NAME}.json")

    def _load_recent_exits(self) -> dict[str, float]:
        """Load persisted cooldown map; prune stale entries.

        Entries with timestamp far in the future (permanent blocklist,
        *_RESOLVED exits) are kept regardless of cutoff.
        """
        import json
        import os
        path = self._recent_exits_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path) as f:
                data = json.load(f)
            now = time.time()
            cutoff = now - self.REENTRY_COOLDOWN_S
            # Keep: (a) within cooldown window, OR (b) permanent (far future)
            return {
                k: float(t) for k, t in data.items()
                if float(t) > cutoff or float(t) > now + 86400
            }
        except Exception:
            return {}

    def _persist_recent_exits(self) -> None:
        """Write cooldown map to disk (best-effort)."""
        import json
        import os
        try:
            path = self._recent_exits_path()
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._recent_exits, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            pass  # Best-effort — don't break trading flow on disk error

    def _is_market_resolved_or_stale(self, market: MarketData) -> bool:
        """Detect markets that look resolved/broken.

        Polymarket resolved markets show yes+no = 1.01 with one side = 0.01
        (loser) and other = 0.99 (winner). Healthy live market: sum ≈ 1.00
        but both sides > 0.03.
        """
        yes_p = market.yes_price
        no_p = market.no_price
        # Either side at minimum tick = resolved.
        # Bumped from 0.02/0.98 → 0.03/0.97 (2026-04-13) — symmetry with
        # discovery_nba.py threshold tightening after POR/PHX late-game entry.
        if yes_p <= 0.03 or no_p <= 0.03:
            return True
        if yes_p >= 0.97 or no_p >= 0.97:
            return True
        # Sum far from 1.0 = broken / stale orderbook
        if not (0.95 <= yes_p + no_p <= 1.05):
            return True
        return False

    def generate_signals(self, markets: list[MarketData]) -> list[DSignal]:
        """Scan markets for entry signals."""
        signals: list[DSignal] = []
        now = time.time()
        stale_count = 0
        cooldown_count = 0

        for market in markets:
            if market.condition_id in self._positions:
                continue

            # Cooldown: skip if we recently exited this market
            last_exit = self._recent_exits.get(market.condition_id)
            if last_exit and now - last_exit < self.REENTRY_COOLDOWN_S:
                cooldown_count += 1
                continue

            # Skip resolved/broken markets
            if self._is_market_resolved_or_stale(market):
                stale_count += 1
                continue

            prob = self.compute_model_prob(market.team_a, market.team_b)
            if prob is None:
                continue

            # YES side — Tier 1 params via _p() (variant-override aware)
            min_edge = self._p("MIN_EDGE")
            max_edge = self._p("MAX_EDGE")
            min_price = self._p("MIN_PRICE")
            max_price = self._p("MAX_PRICE")

            yes_edge = prob - market.yes_price
            no_edge = (1 - prob) - market.no_price

            # Project PARALLEL: schedule shadow eval for all variants
            # (best-effort, fire-and-forget — never blocks signal flow)
            try:
                self._schedule_shadow_eval_d(market, prob, yes_edge, no_edge)
            except Exception as _e:
                pass  # Don't break live execution

            if (min_edge <= yes_edge <= max_edge
                    and min_price <= market.yes_price <= max_price):
                size = self.kelly_size(yes_edge, market.yes_price)
                if size > 0:
                    signals.append(DSignal(
                        market=market, side="yes", token_id=market.token_id_yes,
                        entry_price=market.yes_price, model_prob=prob,
                        edge=yes_edge, kelly_size=size,
                    ))

            # NO side
            if self.BOTH_SIDES:
                if (min_edge <= no_edge <= max_edge
                        and min_price <= market.no_price <= max_price):
                    size = self.kelly_size(no_edge, market.no_price)
                    if size > 0:
                        signals.append(DSignal(
                            market=market, side="no", token_id=market.token_id_no,
                            entry_price=market.no_price, model_prob=1 - prob,
                            edge=no_edge, kelly_size=size,
                        ))

        self._logger.info(
            "scan_complete", sport=self.SPORT_NAME,
            markets=len(markets), signals=len(signals),
            positions=len(self._positions),
            stale=stale_count, cooldown=cooldown_count,
        )
        return signals

    def set_live_balance(self, balance_usdc: float) -> None:
        """Called by orchestrator to inject fresh live balance.

        When balance > 0, strategy sizes positions based on LIVE balance
        (not static STRATEGY_ALLOCATIONS). Share of balance = strategy's
        configured allocation / total allocation across all strategies.
        """
        self._live_balance_usdc = balance_usdc
        self._live_balance_ts = time.time()

    def _effective_capital(self) -> float:
        """Return capital to use for Kelly sizing.

        Priority:
          1. If live balance fresh (< 10min), use strategy's share of it
          2. Else fall back to risk_manager STRATEGY_ALLOCATIONS
        """
        # Live balance path
        if self._live_balance_usdc is not None and self._live_balance_usdc > 0:
            age = time.time() - self._live_balance_ts
            if age < 600:  # Fresh within 10 minutes
                # Strategy's share: my allocation / total allocation
                try:
                    from arbo.core.risk_manager import STRATEGY_ALLOCATIONS
                    total = float(sum(STRATEGY_ALLOCATIONS.values()))
                    my_share = float(STRATEGY_ALLOCATIONS.get(self.STRATEGY_NAME, 0))
                    if total > 0 and my_share > 0:
                        # Live allocation = balance * (my_share / total)
                        return self._live_balance_usdc * (my_share / total)
                except Exception:
                    pass

        # Fallback: static allocation from risk_manager
        ss = self._risk.get_strategy_state(self.STRATEGY_NAME)
        return float(ss.allocated) if ss else 300.0

    def kelly_size(self, edge: float, price: float) -> float:
        """Kelly position size in USDC (uses live balance if available)."""
        if price <= 0 or price >= 1:
            return 0.0
        p = max(0.01, min(0.99, price + edge))
        q = 1 - p
        b = (1 / price) - 1
        if b <= 0:
            return 0.0
        kelly_raw = max(0, min((b * p - q) / b, self.KELLY_RAW_CAP))
        capital = self._effective_capital()
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
                "variant_id": "champion_v1",  # Project PARALLEL — current live variant
                "sport": self.SPORT_NAME,
                "side": signal.side,
                "team_a": signal.market.team_a,
                "team_b": signal.market.team_b,
                "question": signal.market.question,
                "edge": round(signal.edge, 4),
                "model_prob": round(signal.model_prob, 4),
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

    # ── Project PARALLEL — Shadow variant eval + resolution sweep ────

    def _get_variants_d(self) -> list:
        """Return active D variants, refreshed every 60s."""
        now = time.time()
        if now - self._variants_cache_ts > 60:
            try:
                from arbo.core.variant_pool import get_active_variants
                self._variants_cache = get_active_variants(self.STRATEGY_NAME)
                self._variants_cache_ts = now
            except Exception as e:
                self._logger.warning("d_variant_load_error", error=str(e))
        return self._variants_cache

    def _schedule_shadow_eval_d(
        self,
        market: MarketData,
        prob: float,
        yes_edge: float,
        no_edge: float,
    ) -> None:
        """Fire-and-forget shadow eval for D — runs as background task."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(
            self._evaluate_shadow_variants_d(market, prob, yes_edge, no_edge),
            name=f"d_shadow_{market.condition_id[:12]}",
        )

    async def _evaluate_shadow_variants_d(
        self,
        market: MarketData,
        prob: float,
        yes_edge: float,
        no_edge: float,
    ) -> None:
        """Per-market × per-variant evaluation → shadow_variant_signals."""
        variants = self._get_variants_d()
        if not variants:
            return

        from datetime import datetime, timezone
        from arbo.utils.db import get_session_factory
        import sqlalchemy as sa
        import json

        now_ts = time.time()
        # Dedup per (condition_id, side, minute_bucket)
        minute_bucket = int(now_ts // 60)
        signal_dt = datetime.fromtimestamp(now_ts, tz=timezone.utc)

        if len(self._shadow_written_d) > 5000:
            self._shadow_written_d.clear()

        # Game end: estimate as game start + GAME_DURATION_HOURS (champion default)
        # Use champion's GAME_DURATION_HOURS for end_ts (variants share same game)
        try:
            from datetime import datetime as _dt
            game_dt = _dt.fromisoformat(
                str(market.game_date).replace("Z", "+00:00")
            )
            event_end_ts = game_dt.timestamp() + (
                self._p("GAME_DURATION_HOURS") * 3600
            )
        except Exception:
            event_end_ts = now_ts + (self.GAME_DURATION_HOURS * 3600)

        rows: list[dict] = []
        for v in variants:
            params = v.params or {}
            min_e = params.get("MIN_EDGE", self.MIN_EDGE)
            max_e = params.get("MAX_EDGE", self.MAX_EDGE)
            min_p = params.get("MIN_PRICE", self.MIN_PRICE)
            max_p = params.get("MAX_PRICE", self.MAX_PRICE)

            for side, edge, price, token_id in (
                ("yes", yes_edge, market.yes_price, market.token_id_yes),
                ("no", no_edge, market.no_price, market.token_id_no),
            ):
                if not self.BOTH_SIDES and side == "no":
                    continue
                key = (market.condition_id, side, minute_bucket)
                if (key, v.variant_id) in self._shadow_written_d:
                    continue
                qualified = (min_e <= edge <= max_e) and (min_p <= price <= max_p)
                skip_reason = None
                if not qualified:
                    if edge < min_e: skip_reason = "edge_below_min"
                    elif edge > max_e: skip_reason = "edge_above_max"
                    elif price < min_p: skip_reason = "price_below_min"
                    elif price > max_p: skip_reason = "price_above_max"
                meta = {
                    "sport": self.SPORT_NAME,
                    "team_a": market.team_a,
                    "team_b": market.team_b,
                    "side": side,
                    "game_date": str(market.game_date),
                    "yes_price": float(market.yes_price),
                    "no_price": float(market.no_price),
                }
                rows.append({
                    "strategy": self.STRATEGY_NAME,
                    "variant_id": v.variant_id,
                    "condition_id": market.condition_id,
                    "token_id": token_id,
                    "signal_ts": signal_dt,
                    "qualified": qualified,
                    "skip_reason": skip_reason,
                    "direction": side,
                    "entry_price": float(price),
                    "edge": float(edge),
                    "sigma": None,
                    "btc_at_start": None,
                    "btc_now": None,
                    "btc_move": None,
                    "market_gap": None,
                    "velocity": None,
                    "dir_delta": None,
                    "would_fill_at": float(price),
                    "event_start_ts": None,
                    "event_end_ts": event_end_ts,
                    "model_prob": float(prob if side == "yes" else 1 - prob),
                    "meta_json": json.dumps(meta),
                })
                self._shadow_written_d.add((key, v.variant_id))

        if not rows:
            return

        try:
            factory = get_session_factory()
            async with factory() as session:
                for r in rows:
                    try:
                        await session.execute(
                            sa.text("""
                                INSERT INTO shadow_variant_signals
                                  (strategy, variant_id, condition_id, token_id, signal_ts,
                                   qualified, skip_reason, direction, entry_price, edge, sigma,
                                   btc_at_start, btc_now, btc_move, market_gap, velocity,
                                   dir_delta, would_fill_at, event_start_ts, event_end_ts,
                                   model_prob, meta_json)
                                VALUES
                                  (:strategy, :variant_id, :condition_id, :token_id, :signal_ts,
                                   :qualified, :skip_reason, :direction, :entry_price, :edge, :sigma,
                                   :btc_at_start, :btc_now, :btc_move, :market_gap, :velocity,
                                   :dir_delta, :would_fill_at, :event_start_ts, :event_end_ts,
                                   :model_prob, CAST(:meta_json AS jsonb))
                                ON CONFLICT (strategy, variant_id, condition_id, signal_ts)
                                DO NOTHING
                            """),
                            r,
                        )
                    except Exception as e:
                        self._logger.debug(
                            "d_shadow_insert_error",
                            variant_id=r.get("variant_id"), error=str(e),
                        )
                await session.commit()
        except Exception as e:
            self._logger.warning("d_shadow_persist_error", error=str(e))

    async def sweep_shadow_resolutions_d(self) -> None:
        """Resolve D shadow rows via Polymarket Gamma. Throttled to 5min.

        Called by orchestrator (DWatchdog or main_rdh task) on cadence.
        """
        now = time.time()
        if now - self._last_shadow_sweep_ts_d < 300:
            return
        self._last_shadow_sweep_ts_d = now

        from arbo.utils.db import get_session_factory
        import aiohttp
        import sqlalchemy as sa
        import json as _json

        try:
            factory = get_session_factory()
            async with factory() as session:
                result = await session.execute(
                    sa.text("""
                        SELECT DISTINCT condition_id
                        FROM shadow_variant_signals
                        WHERE strategy = :s
                          AND resolution_outcome IS NULL
                          AND event_end_ts IS NOT NULL
                          AND event_end_ts < :now - 300
                        LIMIT 30
                    """),
                    {"s": self.STRATEGY_NAME, "now": now},
                )
                pending = [r["condition_id"] for r in result.mappings()]
        except Exception as e:
            self._logger.debug("d_sweep_query_error", error=str(e))
            return
        if not pending:
            return

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
        ) as http:
            for cid in pending:
                yes_won = None
                try:
                    async with http.get(
                        "https://gamma-api.polymarket.com/markets",
                        params={"condition_ids": cid},
                    ) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json()
                    if not data:
                        continue
                    m = data[0] if isinstance(data, list) else data
                    outcomes = m.get("outcomes")
                    prices = m.get("outcomePrices")
                    if isinstance(outcomes, str):
                        outcomes = _json.loads(outcomes)
                    if isinstance(prices, str):
                        prices = _json.loads(prices)
                    if not outcomes or not prices:
                        continue
                    for o, p in zip(outcomes, prices):
                        if str(o).lower() == "yes":
                            if str(p) == "1":
                                yes_won = True
                            elif str(p) == "0":
                                yes_won = False
                            break
                except Exception:
                    continue
                if yes_won is None:
                    continue
                try:
                    factory = get_session_factory()
                    async with factory() as session:
                        await session.execute(
                            sa.text("""
                                UPDATE shadow_variant_signals
                                SET resolution_outcome = :yes_won, resolution_ts = NOW()
                                WHERE strategy = :s AND condition_id = :cid
                                  AND resolution_outcome IS NULL
                            """),
                            {"yes_won": bool(yes_won), "cid": cid, "s": self.STRATEGY_NAME},
                        )
                        # PnL for qualified rows: direction='yes' wins iff yes_won;
                        # direction='no' wins iff NOT yes_won
                        await session.execute(
                            sa.text("""
                                UPDATE shadow_variant_signals
                                SET would_pnl_per_share = CASE
                                    WHEN (direction = 'yes' AND :yes_won = true)
                                      OR (direction = 'no'  AND :yes_won = false)
                                    THEN 1.0 - would_fill_at
                                    ELSE -would_fill_at
                                END
                                WHERE strategy = :s AND condition_id = :cid
                                  AND qualified = true AND would_pnl_per_share IS NULL
                                  AND would_fill_at IS NOT NULL AND would_fill_at > 0
                            """),
                            {"yes_won": bool(yes_won), "cid": cid, "s": self.STRATEGY_NAME},
                        )
                        await session.commit()
                except Exception as e:
                    self._logger.debug("d_sweep_update_error", cid=cid, error=str(e))

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

            # Tier 1 exit params via _p() (variant-override aware)
            gb_delta = self._p("GREEN_BOOK_DELTA")
            sl_delta = self._p("STOP_LOSS_DELTA")
            max_hold = self._p("MAX_HOLD_FRACTION")
            game_dur = self._p("GAME_DURATION_HOURS")

            if pos.side == "yes":
                gb_target = pos.entry_price + gb_delta
                sl_trigger = pos.entry_price - sl_delta
                gb_hit = price >= gb_target
                sl_hit = price <= sl_trigger
            else:
                gb_target = pos.entry_price - gb_delta
                sl_trigger = pos.entry_price + sl_delta
                gb_hit = price <= gb_target
                sl_hit = price >= sl_trigger

            hold_hours = (now - pos.entry_time) / 3600
            time_exit = hold_hours >= (game_dur * max_hold)

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

            # Sanity: if exit at resolution tick, market was already closed
            # — the green_book/stop_loss triggered on bogus data. Flag it.
            if price <= 0.015 or price >= 0.985:
                pos.exit_reason = f"{pos.exit_reason}_RESOLVED"
                self._logger.warning(
                    "exit_at_resolution_tick", sport=self.SPORT_NAME,
                    entry=pos.entry_price, exit_p=price, token=pos.token_id[:20],
                    team_a=pos.team_a, team_b=pos.team_b,
                    note="Market was already resolved at entry — discovery bug",
                )
            if pos.side == "yes":
                pos.pnl = pos.shares * (pos.exit_price - pos.entry_price)
                pos.clv = price - pos.entry_price  # Positive = entry was underpriced
            else:
                pos.pnl = pos.shares * (pos.entry_price - pos.exit_price)
                pos.clv = pos.entry_price - price

            exits.append(pos)
            del self._positions[cid]
            # Cooldown: don't re-enter for REENTRY_COOLDOWN_S normally.
            # For *_RESOLVED exits (market was already settled in gamma),
            # store timestamp FAR in the future (permanent blocklist) —
            # gamma cache can stay stale >24h, 4h cooldown expires too soon.
            if pos.exit_reason and pos.exit_reason.endswith("_RESOLVED"):
                # +100 years ≈ permanent
                self._recent_exits[cid] = now + 3.15e9
            else:
                self._recent_exits[cid] = now
            self._persist_recent_exits()
            self._daily_pnl += pos.pnl

            # Clean old cooldown entries (> 24h) to avoid memory growth
            cutoff = now - 86400
            self._recent_exits = {k: t for k, t in self._recent_exits.items() if t > cutoff}

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
