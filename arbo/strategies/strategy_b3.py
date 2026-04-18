"""Strategy B3 — Binance Oracle Scalper.

High-frequency scalper for BTC 5-minute Up/Down markets on Polymarket.
Uses Binance real-time price as fair value oracle. Momentum-based:
when BTC moves, buy the corresponding direction before Polymarket
adjusts. PostOnly orders = 0% fee + rebate.

Key differences from B2 (Crypto Price Edge):
- 5-minute hold horizon (vs hours/days for B2)
- Momentum signal, not value betting
- Profit target / stop loss exits, not edge-based hold
- ~33 trades/day (vs 3-5 for B2)
- No NegRisk (standard binary markets)
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from arbo.models.volatility_model import VolatilityEstimator
from arbo.strategies.b3_quality_gate import (
    BTC_STOP_PCT,
    EDGE_EXIT,
    PAPER_MATCH_LIVE,
    EDGE_SCALING,
    LIVE_MAX_FILL_PRICE,
    MAX_BET_SIZE,
    MAX_ENTRY_MKT_FV,
    MAX_HOLD_MIN,
    MAX_SHARES,
    MIN_ENTRY_MKT_FV,
    MIN_ORDER_SIZE,
    MIRROR_CANCEL_DEBOUNCE,
    POSITION_PCT,
    PROFIT_TARGET,
    REENTRY_COOLDOWN,
    SIGMA_FLOOR,
    SIGMA_METHOD,
    SIGMA_SCALE,
    SIGMA_WINDOW,
    SPREAD,
    STOP_LOSS,
    STRATEGY_NAME,
    USE_BTC_STOP,
    WINDOW_MIN,
)
from arbo.strategies.b3_scanner import B3Scanner, B3Signal
from arbo.utils.logger import get_logger

logger = get_logger("strategy_b3")

_SQRT2 = math.sqrt(2.0)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


@dataclass
class B3Position:
    """Tracked open B3 position."""

    condition_id: str
    token_id: str             # Token we bought (Up or Down)
    direction: int            # +1 = long Up, -1 = long Down
    entry_mkt_fv: float       # Market fair value at entry
    entry_time: float         # Unix timestamp
    event_start_ts: float     # Event start timestamp
    event_end_ts: float       # Event end timestamp (resolution)
    btc_at_start: float       # BTC price at event start
    btc_at_entry: float       # BTC price at entry time (for BTC stop)
    sigma_per_min: float      # Volatility at entry
    shares: float
    question: str = ""
    # Live execution tracking (dual mode)
    live_shares: int = 0          # Shares bought via live executor
    live_entry_price: float = 0.0  # Actual fill price from CLOB
    live_fill_status: str = ""     # "filled", "partial", "failed", "skipped"
    live_latency_ms: int = 0       # Entry fill latency
    live_error: str = ""           # Error reason if fill failed


@dataclass
class B3VariantPosition:
    """Per-variant paper position (Project PARALLEL).

    Each challenger variant (ch_edge_tight, ch_velocity_tight, etc.) that
    qualifies a signal gets its own position tracked here — independent of
    champion's B3Position. On resolution, each computes own PnL and updates
    its specific paper_trades row (paper_trade_id).
    """
    variant_id: str
    paper_trade_id: int       # DB PK for targeted update via update_trade_by_id
    condition_id: str
    token_id: str
    direction: int            # +1 = long Up, -1 = long Down
    entry_price: float        # best_ask (same as champion in mirror mode)
    shares: int
    size_usd: float
    event_end_ts: float       # For resolution detection


class StrategyB3:
    """Binance Oracle Scalper — 5-min BTC Up/Down."""

    def __init__(
        self,
        risk_manager: Any,
        paper_engine: Any | None = None,
        binance_ws: Any | None = None,
        rtds_feed: Any | None = None,
        execution_mode: str = "paper",
        live_executor: Any | None = None,
        ta_provider: Any | None = None,
    ) -> None:
        self._risk_manager = risk_manager
        self._paper_engine = paper_engine
        self._binance_ws = binance_ws
        self._rtds_feed = rtds_feed  # Chainlink resolution price feed
        self._execution_mode = execution_mode  # "paper", "dual", "live"
        self._live_executor = live_executor
        self._ta_provider = ta_provider  # TAFeatureProvider (background TA cache)
        self._adaptive_config: Any = None  # AdaptiveConfig (set by orchestrator)

        # Live config — capital fetched from wallet, env var as fallback
        import os
        self._live_capital_fallback = float(os.getenv("B3_LIVE_CAPITAL", "300"))
        self._live_capital: float = self._live_capital_fallback
        self._live_capital_last_check: float = 0.0
        self._live_entry_timeout_s = int(os.getenv("B3_LIVE_ENTRY_TIMEOUT_S", "10"))
        self._live_exit_maker_timeout_s = int(os.getenv("B3_LIVE_EXIT_MAKER_TIMEOUT_S", "5"))
        self._live_daily_pnl: float = 0.0  # Track live PnL for kill switch
        self._live_daily_loss_limit = float(os.getenv("B3_LIVE_DAILY_LOSS_LIMIT", "50"))

        # B3 scanner (manages event lifecycle)
        self._scanner = B3Scanner(rtds_feed=rtds_feed)

        # Volatility estimator (per-minute realized vol from 720 klines)
        # IMPORTANT: Only feed prices once per minute to match backtest's 1-min klines.
        # Binance WS updates every ~1s, but sigma must be per-minute, not per-update.
        self._vol_estimator = VolatilityEstimator(
            window=SIGMA_WINDOW,
            method=SIGMA_METHOD,
        )
        self._last_vol_update_ts: float = 0.0  # Unix ts of last vol estimator feed

        # Open position tracking
        self._open_positions: dict[str, B3Position] = {}  # token_id → position
        self._live_holding: dict[str, B3Position] = {}  # live positions waiting for resolution
        self._last_exit_time: dict[str, float] = {}  # condition_id → timestamp
        # Debounce: after a mirror-cancel (live failed to fill), block re-entry
        # on the same token for MIRROR_CANCEL_DEBOUNCE seconds. Without this,
        # signal generator keeps triggering on the same token every poll cycle
        # (5s), creating cascade duplicates (observed: 4 trades in 16s).
        self._last_mirror_attempt: dict[str, float] = {}  # token_id → timestamp
        # Orphan sweeper throttle: run at most every 15 min. Sweeper catches
        # champion rows stuck status=open past event_end_ts (mirror-cancel
        # legacy, restart survivors). Fixes silent data corruption where
        # unresolved paper_trades accumulate without affecting real capital.
        self._last_orphan_sweep: float = 0.0

        # Phase 2B: variant pool cache (champion + challengers)
        self._variants_cache: list = []
        self._variants_cache_ts: float = 0.0
        # Dedupe shadow writes per (condition_id, entry_minute_int)
        self._shadow_written: set[tuple[str, int]] = set()
        # Per-variant paper positions (Project PARALLEL Option B):
        # each challenger's independent paper trade — resolved at event end
        # with its own pnl → written to paper_trades.actual_pnl via
        # update_trade_by_id. Clean isolation per variant_id.
        self._variant_positions: dict[tuple[str, str], B3VariantPosition] = {}

    async def init(self) -> None:
        """Initialize scanner and restore state."""
        await self._scanner.init()

        # FIX #3: Bootstrap volatility estimator with 1440 historical klines.
        # Without this, sigma is noisy for hours after startup (backtest always
        # has full 1440-observation window).
        try:
            import aiohttp
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
            ) as session:
                url = "https://api.binance.com/api/v3/klines"
                async with session.get(url, params={
                    "symbol": "BTCUSDT", "interval": "1m", "limit": "1440",
                }) as resp:
                    if resp.status == 200:
                        klines = await resp.json()
                        prices = [float(k[4]) for k in klines]  # close prices
                        self._vol_estimator.load_historical("BTCUSDT", prices)
                        sigma = self._vol_estimator.get_sigma("BTCUSDT")
                        logger.info(
                            "b3_sigma_bootstrapped",
                            observations=len(prices),
                            sigma=f"{sigma:.6f}",
                        )
        except Exception as e:
            logger.warning("b3_sigma_bootstrap_error", error=str(e))

        # Restore B3 positions from paper engine.
        # B3 positions are ultra-short-lived (1-3 min). On restart, any open
        # B3 positions are stale (the 5-min event has already resolved).
        # Just LOG them — actual resolution happens in async cleanup_stale_positions()
        # called by orchestrator after init. This avoids force-marking as 'lost'
        # which would toxify data — we want REAL Gamma API resolution.
        if self._paper_engine:
            stale_b3 = [
                pos for pos in self._paper_engine.open_positions
                if getattr(pos, "strategy", "") == STRATEGY_NAME
            ]
            if stale_b3:
                logger.warning(
                    "b3_stale_positions_on_restart",
                    count=len(stale_b3),
                    msg="Stale B3 positions detected — will be resolved via Gamma API",
                )
                # Store for async cleanup
                self._stale_on_restart = list(stale_b3)
            else:
                self._stale_on_restart = []

    async def close(self) -> None:
        """Clean up resources."""
        await self._scanner.close()

    async def cleanup_stale_on_restart(self) -> None:
        """Resolve stale B3 positions via Gamma API after restart.

        Called by orchestrator after init. Uses REAL Chainlink resolution
        from Gamma API. Properly updates:
        - paper_engine._positions (in-memory)
        - paper_trades DB row (status, actual_pnl, exit_price, exit_reason)
        - risk_manager.post_trade_update + strategy_post_trade
        - paper_positions DB table (via sync_positions_to_db)

        For positions that can't be resolved (Gamma API doesn't know),
        marks them as 'sold' with exit_reason='orphaned' and actual_pnl=NULL
        so they're excluded from analytics.
        """
        stale = getattr(self, "_stale_on_restart", [])
        if not stale or not self._paper_engine:
            return

        logger.info("b3_cleanup_stale_starting", count=len(stale))
        resolved_count = 0
        orphaned_count = 0
        deferred_count = 0

        for pos in stale:
            try:
                token_id = getattr(pos, "token_id", None)
                cond_id = getattr(pos, "market_condition_id", None) or getattr(pos, "condition_id", None)
                size = getattr(pos, "size", None)
                avg_price = getattr(pos, "avg_price", None)
                shares = getattr(pos, "shares", None)
                if not token_id or size is None or avg_price is None:
                    orphaned_count += 1
                    continue

                # Get event timing from trade_details cache
                td = self._paper_engine._trade_details_cache.get(token_id, {}) if hasattr(self._paper_engine, "_trade_details_cache") else {}
                event_start_ts = td.get("event_start_ts")
                event_end_ts = td.get("event_end_ts")
                direction_str = td.get("direction", "")

                if not event_start_ts:
                    # Can't determine event — mark orphaned
                    await self._mark_orphaned(token_id, cond_id, size)
                    orphaned_count += 1
                    continue

                # Check if event has ended
                now_ts = time.time()
                if event_end_ts and now_ts < event_end_ts:
                    # Event still running. We must INSERT this position into
                    # _live_holding so check_exits can resolve it later once
                    # event_end_ts passes. Without this insertion, the trade
                    # is orphaned post-restart and stays unresolved forever
                    # unless we won (auto_redeem path) — losers never get
                    # their exit_price set (bug discovered 2026-04-13).
                    try:
                        live_shares = int(td.get("live_entry_shares") or 0)
                        live_entry_price = float(td.get("live_entry_price") or 0)
                        direction_int = 1 if direction_str == "up" else -1
                        btc_at_start = float(td.get("btc_at_start") or 0)
                        btc_at_entry = float(td.get("btc_now") or btc_at_start)
                        sigma_per_min = float(td.get("sigma") or 0)
                        if live_shares > 0 and live_entry_price > 0:
                            self._live_holding[token_id] = B3Position(
                                condition_id=cond_id or "",
                                token_id=token_id,
                                direction=direction_int,
                                entry_mkt_fv=live_entry_price,
                                entry_time=now_ts,
                                event_start_ts=event_start_ts,
                                event_end_ts=event_end_ts,
                                btc_at_start=btc_at_start,
                                btc_at_entry=btc_at_entry,
                                sigma_per_min=sigma_per_min,
                                shares=float(size or 0),
                                question=td.get("question", ""),
                                live_shares=live_shares,
                                live_entry_price=live_entry_price,
                                live_fill_status=str(td.get("live_fill_status") or "filled"),
                            )
                            logger.info(
                                "b3_stale_holding_restored",
                                token=token_id[:20],
                                wait_s=int(event_end_ts - now_ts),
                                msg="Inserted into _live_holding; check_exits will resolve at event end",
                            )
                    except Exception as e:
                        logger.warning("b3_stale_holding_restore_error",
                                       token=token_id[:20], error=str(e))
                    deferred_count += 1
                    continue  # check_exits will handle later (now with _live_holding entry)

                # Fetch real resolution from Gamma API
                pm_up_won = None
                try:
                    pm_up_won = await asyncio.wait_for(
                        self._scanner.fetch_resolution(event_start_ts),
                        timeout=15,
                    )
                except asyncio.TimeoutError:
                    logger.warning("b3_stale_gamma_timeout", token=token_id[:20])
                except Exception as e:
                    logger.warning("b3_stale_gamma_error", token=token_id[:20], error=str(e))

                if pm_up_won is None:
                    # Gamma can't resolve — mark orphaned (NULL pnl)
                    await self._mark_orphaned(token_id, cond_id, size)
                    orphaned_count += 1
                    continue

                # We have real resolution
                direction_int = 1 if direction_str == "up" else -1
                won = (direction_int == 1 and pm_up_won) or (direction_int == -1 and not pm_up_won)
                exit_price = Decimal("1.0") if won else Decimal("0.0")

                # Resolve in paper engine (in-memory) — returns realized PnL
                pnl = self._paper_engine.resolve_market(token_id, winning_outcome=won)

                # Update DB row to won/lost (use winning path to preserve W/L info)
                # NOTE: this overwrites exit_reason — we accept that to keep W/L stats clean
                await self._paper_engine.update_resolved_trades_in_db(
                    token_id=token_id,
                    winning=won,
                    pnl=pnl,
                )

                # Update risk manager exposure (CRITICAL — this is what unblocks new trades)
                if self._risk_manager and cond_id:
                    try:
                        self._risk_manager.post_trade_update(cond_id, "crypto_5min", size, pnl=pnl)
                        self._risk_manager.strategy_post_trade(STRATEGY_NAME, size, pnl=pnl)
                    except Exception as e:
                        logger.warning("b3_stale_risk_update_error", error=str(e))

                resolved_count += 1
                logger.info("b3_stale_resolved", token=token_id[:20],
                           direction=direction_str, won=won, pnl=str(pnl))
            except Exception as e:
                logger.warning("b3_stale_cleanup_error", error=str(e))
                orphaned_count += 1

        # After cleanup, sync paper engine state to DB
        try:
            await self._paper_engine.sync_positions_to_db()
        except Exception as e:
            logger.warning("b3_stale_sync_error", error=str(e))

        logger.info("b3_cleanup_stale_done",
                   resolved=resolved_count, orphaned=orphaned_count,
                   deferred=deferred_count, total=len(stale))
        self._stale_on_restart = []

    async def _mark_orphaned(self, token_id: str, cond_id: str | None, size: Any) -> None:
        """Mark a stale position as orphaned (unknown outcome).

        Removes from paper engine in-memory state, updates DB row to
        status='sold' with exit_reason='orphaned' and actual_pnl=NULL,
        and decrements risk manager counters.
        """
        try:
            # Remove from paper engine in-memory
            if hasattr(self._paper_engine, "_positions") and token_id in self._paper_engine._positions:
                del self._paper_engine._positions[token_id]

            # Update DB row to orphaned (NULL pnl — unknown outcome)
            await self._paper_engine.update_resolved_trades_in_db(
                token_id=token_id,
                winning=None,
                pnl=None,
                exit_price=None,
                exit_reason="orphaned",
            )

            # Clear risk manager exposure (pnl=Decimal(0) — neutral)
            if self._risk_manager and cond_id:
                try:
                    self._risk_manager.post_trade_update(cond_id, "crypto_5min", size, pnl=Decimal("0"))
                    self._risk_manager.strategy_post_trade(STRATEGY_NAME, size, pnl=Decimal("0"))
                except Exception:
                    pass

            logger.info("b3_stale_orphaned", token=token_id[:20])
        except Exception as e:
            logger.warning("b3_mark_orphaned_error", error=str(e))

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 2B — Shadow variant evaluation (champion + challengers)
    # ═══════════════════════════════════════════════════════════════════════

    def _get_variants(self) -> list:
        """Return active B3 variants, refreshed every 60s."""
        import time as _t
        now = _t.time()
        if now - self._variants_cache_ts > 60:
            try:
                from arbo.core.variant_pool import get_active_variants
                self._variants_cache = get_active_variants("B3")
                self._variants_cache_ts = now
            except Exception as e:
                logger.warning("b3_variant_load_error", error=str(e))
        return self._variants_cache

    @staticmethod
    def _b3_gates(
        *,
        edge: float,
        btc_move: float,
        velocity: float,
        abs_dir_delta: float,
        fill_price: float | None,
        entry_minute: int,
        entry_mkt_fv: float,
        params: dict,
    ) -> tuple[bool, str | None]:
        """Apply per-variant B3 5-min live gates."""
        if edge < params.get("LIVE_MIN_EDGE", 0.30):
            return False, "edge_below_min"
        if btc_move < params.get("LIVE_MIN_BTC_MOVE", 35.0):
            return False, "btc_move_below_min"
        if velocity > params.get("LIVE_MAX_VELOCITY", 60.0):
            return False, "velocity_above_max"
        if abs_dir_delta > params.get("LIVE_MAX_DIR_DELTA", 15.0):
            return False, "dir_delta_above_max"
        if fill_price is not None:
            if fill_price > params.get("LIVE_MAX_FILL_PRICE", 0.75):
                return False, "fill_above_max"
            if fill_price <= 0:
                return False, "no_orderbook"
        if entry_minute < params.get("MIN_ENTRY_MIN", 1):
            return False, "minute_below_min"
        if entry_minute > params.get("MAX_ENTRY_MIN", 3):
            return False, "minute_above_max"
        if entry_mkt_fv < params.get("MIN_ENTRY_MKT_FV", 0.0):
            return False, "fv_below_min"
        if entry_mkt_fv > params.get("MAX_ENTRY_MKT_FV", 1.0):
            return False, "fv_above_max"
        return True, None

    async def _evaluate_shadow_variants(
        self,
        *,
        sig: B3Signal,
        btc_move: float,
        velocity: float,
        dir_delta: float,
        abs_dir_delta: float,
        entry_mkt_fv: float,
        fill_price: float | None,
        market_gap: float | None,
        token_id: str,
    ) -> None:
        """Write one row per active variant to shadow_variant_signals.

        Dedupes per (condition_id, entry_minute_int) so a candidate re-evaluated
        inside the same minute is not written twice.
        """
        variants = self._get_variants()
        if not variants:
            return

        entry_min = int(sig.minutes_elapsed or 0)
        dedupe_key = (sig.condition_id, entry_min)
        if dedupe_key in self._shadow_written:
            return
        self._shadow_written.add(dedupe_key)
        # Prune dedupe set if it grows
        if len(self._shadow_written) > 5000:
            self._shadow_written.clear()

        import time as _t
        from datetime import datetime, timezone
        from arbo.utils.db import get_session_factory
        import sqlalchemy as sa

        signal_dt = datetime.fromtimestamp(_t.time(), tz=timezone.utc)
        direction_s = "up" if sig.direction == 1 else "down"

        try:
            factory = get_session_factory()
            async with factory() as session:
                for v in variants:
                    qualified, skip_reason = self._b3_gates(
                        edge=sig.edge,
                        btc_move=btc_move,
                        velocity=velocity,
                        abs_dir_delta=abs_dir_delta,
                        fill_price=fill_price,
                        entry_minute=entry_min,
                        entry_mkt_fv=entry_mkt_fv,
                        params=v.params,
                    )
                    row = {
                        "strategy": "B3",
                        "variant_id": v.variant_id,
                        "condition_id": sig.condition_id,
                        "token_id": token_id,
                        "signal_ts": signal_dt,
                        "qualified": qualified,
                        "skip_reason": skip_reason,
                        "direction": direction_s,
                        "entry_price": fill_price,
                        "edge": sig.edge,
                        "sigma": sig.sigma_per_min,
                        "btc_at_start": sig.btc_at_start,
                        "btc_now": sig.btc_now,
                        "btc_move": btc_move,
                        "market_gap": market_gap,
                        "velocity": velocity,
                        "dir_delta": dir_delta,
                        "would_fill_at": fill_price,
                        "event_start_ts": sig.event_start_ts,
                        "event_end_ts": sig.event_end_ts,
                    }
                    try:
                        await session.execute(
                            sa.text("""
                                INSERT INTO shadow_variant_signals
                                  (strategy, variant_id, condition_id, token_id, signal_ts,
                                   qualified, skip_reason, direction, entry_price, edge, sigma,
                                   btc_at_start, btc_now, btc_move, market_gap, velocity,
                                   dir_delta, would_fill_at, event_start_ts, event_end_ts)
                                VALUES
                                  (:strategy, :variant_id, :condition_id, :token_id, :signal_ts,
                                   :qualified, :skip_reason, :direction, :entry_price, :edge, :sigma,
                                   :btc_at_start, :btc_now, :btc_move, :market_gap, :velocity,
                                   :dir_delta, :would_fill_at, :event_start_ts, :event_end_ts)
                                ON CONFLICT (strategy, variant_id, condition_id, signal_ts)
                                DO NOTHING
                            """),
                            row,
                        )
                    except Exception as e:
                        logger.debug(
                            "b3_shadow_insert_error",
                            variant_id=v.variant_id,
                            error=str(e),
                        )
                await session.commit()
        except Exception as e:
            logger.warning("b3_shadow_evaluate_error", error=str(e))

    async def _sweep_shadow_resolutions(self) -> None:
        """Resolve any B3 shadow_variant_signals rows past their event_end_ts.

        Runs once per poll_cycle. For each unresolved condition_id with
        event ended > 30s ago, fetches Gamma resolution and updates all
        rows (qualified = fill-based PnL, unqualified = outcome only).
        """
        import time as _t
        from arbo.utils.db import get_session_factory
        import sqlalchemy as sa

        now_ts = _t.time()
        try:
            factory = get_session_factory()
            async with factory() as session:
                result = await session.execute(
                    sa.text("""
                        SELECT DISTINCT condition_id, event_start_ts, event_end_ts
                        FROM shadow_variant_signals
                        WHERE strategy = 'B3'
                          AND resolution_outcome IS NULL
                          AND event_end_ts IS NOT NULL
                          AND event_end_ts < :now - 30
                        LIMIT 20
                    """),
                    {"now": now_ts},
                )
                pending = list(result.mappings())
        except Exception as e:
            logger.debug("b3_shadow_sweep_query_error", error=str(e))
            return

        if not pending:
            return

        for row in pending:
            cid = row["condition_id"]
            event_start = row["event_start_ts"]
            try:
                import asyncio
                up_won = await asyncio.wait_for(
                    self._scanner.fetch_resolution(event_start),
                    timeout=10.0,
                )
            except Exception:
                continue
            if up_won is None:
                continue
            try:
                factory = get_session_factory()
                async with factory() as session:
                    await session.execute(
                        sa.text("""
                            UPDATE shadow_variant_signals
                            SET resolution_outcome = :up_won,
                                resolution_ts = NOW()
                            WHERE strategy = 'B3'
                              AND condition_id = :cid
                              AND resolution_outcome IS NULL
                        """),
                        {"up_won": bool(up_won), "cid": cid},
                    )
                    await session.execute(
                        sa.text("""
                            UPDATE shadow_variant_signals
                            SET would_pnl_per_share = CASE
                                WHEN (direction = 'up'   AND :up_won = true)
                                  OR (direction = 'down' AND :up_won = false)
                                THEN 1.0 - would_fill_at
                                ELSE -would_fill_at
                            END
                            WHERE strategy = 'B3'
                              AND condition_id = :cid
                              AND qualified = true
                              AND would_pnl_per_share IS NULL
                              AND would_fill_at IS NOT NULL
                              AND would_fill_at > 0
                        """),
                        {"up_won": bool(up_won), "cid": cid},
                    )
                    await session.commit()
            except Exception as e:
                logger.debug("b3_shadow_sweep_update_error", cid=cid, error=str(e))

    # ═══════════════════════════════════════════════════════════════════════
    # ENTRY: Poll Cycle
    # ═══════════════════════════════════════════════════════════════════════

    async def poll_cycle(self) -> list[B3Signal]:
        """Main entry scan. Called every 10-15 seconds.

        1. Fetch events from Gamma API (cached, refetches every 2 min)
        2. Get BTC price from Binance
        3. Compute per-minute realized volatility
        4. Scan events for entry signals (minute 2 check)
        5. Size and execute trades

        Returns:
            List of executed B3Signal objects.
        """
        executed: list[B3Signal] = []

        # 0. Record Chainlink price to rolling buffer (for accurate btc_at_start)
        self._scanner.record_cl_price()

        # Phase 2B: sweep resolutions for any B3 shadow_variant_signals past
        # event_end_ts. Runs best-effort; never blocks main flow.
        try:
            await self._sweep_shadow_resolutions()
        except Exception as _e:
            logger.debug("b3_shadow_sweep_error", error=str(_e))

        # "stopped" mode: skip entry logic entirely. check_exits() still
        # runs from the orchestrator so existing positions resolve
        # naturally. Set by B3_EXECUTION_MODE=stopped on Apr 18 after
        # -$595 drawdown over 3 days (see LEARNINGS B3-1).
        if self._execution_mode == "stopped":
            return executed

        # 1. Fetch events
        await self._scanner.fetch_events()

        # 2. Get BTC price
        btc_price = None
        now = time.time()
        if self._binance_ws:
            btc_price = self._binance_ws.get_price("BTCUSDT")
            # Feed to vol estimator ONLY once per minute (matches backtest 1-min klines).
            # If we feed every 15s, sigma becomes per-15s instead of per-minute.
            if btc_price and (now - self._last_vol_update_ts >= 60.0):
                self._vol_estimator.update("BTCUSDT", btc_price, now)
                self._last_vol_update_ts = now

        if not btc_price:
            return []

        # 2b. Retry btc_at_start for events where Binance kline wasn't ready.
        # Must happen BEFORE scan() and on every poll (not just fetch_events).
        # Critical: events fetched for future windows get btc_at_start=None,
        # and the kline becomes available ~60s after event start. If retry
        # only runs in fetch_events (every 2 min), we miss the minute-2 window.
        for ev in self._scanner._events.values():
            if ev.btc_at_start is None and now >= ev.start_ts + 60:
                btc_start = await self._scanner._fetch_btc_at_start(ev.start_ts)
                if btc_start:
                    ev.btc_at_start = btc_start
                    logger.info(
                        "b3_btc_start_filled",
                        question=ev.question[:50],
                        btc=f"${btc_start:,.2f}",
                    )

        # 3. Compute per-minute sigma
        sigma_per_min = self._vol_estimator.get_sigma("BTCUSDT", now)
        sigma_per_min = max(sigma_per_min, SIGMA_FLOOR)

        # 4. Get Chainlink resolution price for comparison
        chainlink_price = None
        if self._rtds_feed:
            chainlink_price = self._rtds_feed.get_price("btc/usd")

        # 5. Scan for signals (using Binance for now — fastest signal)
        signals = self._scanner.scan(btc_price, sigma_per_min)

        if not signals:
            return []

        # Log comparison: Binance vs Chainlink on every signal
        delta_str = ""
        if chainlink_price and btc_price:
            delta = btc_price - chainlink_price
            delta_str = f"${delta:+.2f}"

        logger.info(
            "b3_scan",
            btc_binance=f"${btc_price:.2f}",
            btc_chainlink=f"${chainlink_price:.2f}" if chainlink_price else "N/A",
            delta=delta_str or "N/A",
            sigma=f"{sigma_per_min:.6f}",
            events=self._scanner.active_event_count,
            signals=len(signals),
        )

        # 5. Process signals
        for sig in signals:
            # FIX #2: Chainlink direction filter — skip if oracles disagree.
            # Binance is ~$31 above Chainlink systematically. When model says UP
            # based on Binance but Chainlink shows DOWN, the trade is phantom.
            if chainlink_price and sig.btc_at_start > 0:
                binance_up = sig.btc_now >= sig.btc_at_start
                chainlink_up = chainlink_price >= sig.btc_at_start
                if binance_up != chainlink_up:
                    logger.info(
                        "b3_oracle_disagree",
                        direction="UP" if sig.direction == 1 else "DOWN",
                        binance=f"{'UP' if binance_up else 'DOWN'}",
                        chainlink=f"{'UP' if chainlink_up else 'DOWN'}",
                    )
                    continue

            # Min model FV check (max is checked on LIVE FILL price, not model FV)
            entry_mkt_fv = sig.market_fv_up if sig.direction == 1 else (1.0 - sig.market_fv_up)

            # Skip if already have position in this event
            token_id = sig.token_id_up if sig.direction == 1 else sig.token_id_down
            if token_id in self._open_positions:
                continue

            # Check any position in this event (up OR down)
            event_tokens = {sig.token_id_up, sig.token_id_down}
            if event_tokens & set(self._open_positions.keys()):
                continue

            # Mirror-cancel debounce: after live failed to fill in mirror mode,
            # block re-entry on this token for MIRROR_CANCEL_DEBOUNCE seconds.
            # Without this, stale signal on same token keeps triggering every
            # poll cycle → cascade duplicates in DB.
            last_mirror_attempt = self._last_mirror_attempt.get(token_id, 0)
            if now - last_mirror_attempt < MIRROR_CANCEL_DEBOUNCE:
                continue

            # Re-entry cooldown
            last_exit = self._last_exit_time.get(sig.condition_id, 0)
            if now - last_exit < REENTRY_COOLDOWN * 60:
                continue

            # Sizing: proportional to signal edge, capped
            raw_pct = min(POSITION_PCT, sig.edge * EDGE_SCALING)
            strat_state = self._risk_manager.get_strategy_state(STRATEGY_NAME)
            if strat_state is None:
                continue
            if strat_state.is_halted:
                logger.info("b3_halted")
                break

            # DUAL MODE + PAPER_MATCH_LIVE: paper uses SAME capital as live
            # (wallet balance) so paper trade size/shares/prices are 1:1
            # comparable with live. Otherwise paper uses risk_manager allocation
            # (broader research dataset, but not directly comparable to live).
            mirror_live = (
                self._execution_mode == "dual" and PAPER_MATCH_LIVE
                and self._live_capital > 0
            )
            if mirror_live:
                available = self._live_capital
                min_size = max(2.0, min(10.0, self._live_capital * 0.015))
            else:
                available = float(strat_state.allocated - strat_state.deployed)
                min_size = MIN_ORDER_SIZE
            if available <= 0:
                continue

            bet_size = min(available * raw_pct, MAX_BET_SIZE)
            if bet_size < min_size:
                continue

            entry_price = sig.entry_price + SPREAD / 2  # Buy at ask (paper default)
            if entry_price <= 0.01:
                continue

            # Liquidity check: fetch orderbook and compute max size
            liq = await self._check_liquidity(token_id, entry_price, bet_size)
            if liq is not None:
                bet_size = liq["safe_size"]
                if bet_size < min_size:
                    logger.info(
                        "b3_low_liquidity",
                        token=token_id[:20],
                        available_usd=f"${liq['available_usd']:.0f}",
                        safe_size=f"${bet_size:.0f}",
                        min_size=f"${min_size:.2f}",
                    )
                    continue

                # MARKET DISAGREEMENT CHECK (before order!)
                # If market price is >50pp away from model FV, trust the market.
                # Evidence: 5/5 wins had gap <35pp, 1 loss had gap 77pp.
                expected_fill = liq.get("best_ask", entry_price)
                market_gap = abs(entry_mkt_fv - expected_fill)
                if market_gap > 0.50:
                    logger.info(
                        "b3_market_disagrees",
                        model_fv=f"{entry_mkt_fv:.3f}",
                        market_price=f"{expected_fill:.3f}",
                        gap=f"{market_gap:.3f}",
                        msg="Market disagrees >50pp — skipping",
                    )
                    continue

                # DUAL MODE + PAPER_MATCH_LIVE: paper uses same CLOB best_ask
                # as live (so paper = "what live would execute"). Otherwise
                # paper keeps the model-FV + synthetic spread entry (wider
                # historical coverage for research).
                if mirror_live:
                    entry_price = expected_fill  # Real orderbook price

                # Dynamic sizing
                if expected_fill <= 0.30:
                    size_mult = 0.5   # Trh prodává lacino = opatrně
                elif expected_fill <= 0.55:
                    size_mult = 1.0
                elif expected_fill <= 0.75:
                    size_mult = 1.0
                else:
                    size_mult = 0.5   # Příliš drahé
                bet_size = min(bet_size * size_mult, MAX_BET_SIZE)
                if bet_size < min_size:
                    continue

                # In mirror_live mode, also apply LIVE_MAX_FILL_PRICE cap to
                # paper — so paper doesn't "trade" markets live would reject.
                if mirror_live and entry_price > LIVE_MAX_FILL_PRICE:
                    logger.info(
                        "b3_mirror_skip_fill_cap",
                        fill=f"{entry_price:.3f}",
                        cap=f"{LIVE_MAX_FILL_PRICE:.3f}",
                    )
                    continue

            shares = bet_size / entry_price
            shares = min(shares, MAX_SHARES)
            if shares < 1:
                continue

            actual_size = shares * entry_price

            # Risk check
            from arbo.core.risk_manager import TradeRequest

            trade_req = TradeRequest(
                market_id=sig.condition_id,
                token_id=token_id,
                side="BUY",
                price=Decimal(str(round(entry_price, 4))),
                size=Decimal(str(round(actual_size, 2))),
                layer=0,
                market_category="crypto_5min",
                strategy=STRATEGY_NAME,
            )
            decision = self._risk_manager.pre_trade_check(trade_req)
            if not decision.approved:
                logger.info("b3_risk_rejected", reason=decision.reason)
                continue

            actual_size = float(decision.adjusted_size or actual_size)
            shares = actual_size / entry_price

            # Compute BTC move (used in trade_details AND live qualification)
            btc_move_cl = abs(chainlink_price - sig.btc_at_start) if chainlink_price and sig.btc_at_start else 0
            btc_move_bin = abs(sig.btc_now - sig.btc_at_start) if sig.btc_now and sig.btc_at_start else 0
            btc_move = btc_move_cl if btc_move_cl > 0 else btc_move_bin  # CL preferred

            # V6.0 features (computed here so paper trade_details can log them
            # even when live execution is skipped)
            is_up_dir = sig.direction == 1
            _cl_for_calc = chainlink_price  # may be None
            _bin_for_calc = sig.btc_now
            _em_for_calc = sig.minutes_elapsed or 1.0
            velocity = btc_move / _em_for_calc
            if is_up_dir:
                dir_delta = (_bin_for_calc - _cl_for_calc) if (_bin_for_calc and _cl_for_calc) else 0
            else:
                dir_delta = (_cl_for_calc - _bin_for_calc) if (_bin_for_calc and _cl_for_calc) else 0
            abs_dir_delta = abs(dir_delta)

            # Compute live_qualified flag early — needed both for trade_details
            # and for mirror_live paper gating.
            _ac_q = self._adaptive_config
            _live_min_edge = _ac_q.get("LIVE_MIN_EDGE", 0.30) if _ac_q else 0.30
            _live_min_move = _ac_q.get("LIVE_MIN_BTC_MOVE", 35.0) if _ac_q else 35.0
            _live_max_vel = _ac_q.get("LIVE_MAX_VELOCITY", 60.0) if _ac_q else 60.0
            _live_max_dd = _ac_q.get("LIVE_MAX_DIR_DELTA", 15.0) if _ac_q else 15.0
            _live_max_fill = _ac_q.get("LIVE_MAX_FILL_PRICE", LIVE_MAX_FILL_PRICE) if _ac_q else LIVE_MAX_FILL_PRICE
            _fill_for_q = liq.get("best_ask", entry_price) if liq else entry_price
            live_qualified_flag = bool(
                sig.edge >= _live_min_edge
                and btc_move >= _live_min_move
                and velocity <= _live_max_vel
                and abs_dir_delta <= _live_max_dd
                and _fill_for_q <= _live_max_fill
            )

            # mirror_live: skip CHAMPION paper/live when live wouldn't qualify.
            # But challengers must still be evaluated — each has own filter and
            # could qualify where champion doesn't. Evaluate challengers first,
            # then skip the champion path.
            if mirror_live and not live_qualified_flag:
                logger.info(
                    "b3_mirror_skip_not_qualified",
                    edge=f"{sig.edge:.3f}", move=f"{btc_move:.1f}",
                    vel=f"{velocity:.1f}", dd=f"{abs_dir_delta:.1f}",
                    fill=f"{_fill_for_q:.3f}",
                )
                # Still evaluate challengers — their own filters may qualify
                try:
                    await self._place_challenger_paper_trades(
                        sig=sig,
                        token_id=token_id,
                        entry_price=_fill_for_q,
                        btc_move=btc_move,
                        velocity=velocity,
                        abs_dir_delta=abs_dir_delta,
                    )
                except Exception as _e:
                    logger.debug("b3_challenger_skip_path_error", error=str(_e))
                continue

            # Phase 2B: shadow evaluation of all B3 variants on this candidate.
            # Writes one row per variant to shadow_variant_signals (decision
            # + all features). Champion's row is counterfactual-of-itself
            # for paired-sample comparison vs challengers.
            try:
                await self._evaluate_shadow_variants(
                    sig=sig,
                    btc_move=btc_move,
                    velocity=velocity,
                    dir_delta=dir_delta,
                    abs_dir_delta=abs_dir_delta,
                    entry_mkt_fv=entry_mkt_fv,
                    fill_price=float(expected_fill) if 'expected_fill' in locals() else None,
                    market_gap=market_gap if 'market_gap' in locals() else None,
                    token_id=token_id,
                )
            except Exception as _e:
                logger.debug("b3_shadow_variant_error", error=str(_e))

            # Execute
            paper_trade = None
            if self._paper_engine:
                paper_trade = self._paper_engine.place_trade(
                    market_condition_id=sig.condition_id,
                    token_id=token_id,
                    side="BUY",
                    market_price=Decimal(str(round(sig.entry_price, 4))),
                    model_prob=Decimal(str(round(
                        sig.signal_fv_up if sig.direction == 1 else 1 - sig.signal_fv_up,
                        4,
                    ))),
                    layer=0,
                    market_category="crypto_5min",
                    fee_enabled=False,  # Maker = 0%
                    strategy=STRATEGY_NAME,
                    pre_computed_size=Decimal(str(round(actual_size, 2))),
                    clob_fill_price=Decimal(str(round(entry_price, 4))),
                    trade_details={
                        # Variant identity (Rapid Mode §11). Current V6.0 live config
                        # is the single champion per config/variants/b3/champion_v1.yaml.
                        # When multi-variant orchestrator activates (Phase 2+), this
                        # field will be set per-variant by ShadowOrchestrator.
                        "variant_id": "champion_v1",
                        "asset": "BTC",
                        "direction": "up" if sig.direction == 1 else "down",
                        "btc_at_start": sig.btc_at_start,
                        "btc_now": sig.btc_now,
                        "btc_chainlink": chainlink_price,
                        "btc_binance_chainlink_delta": round(btc_price - chainlink_price, 2) if chainlink_price else None,
                        "sigma_per_min": sig.sigma_per_min,
                        "signal_fv_up": sig.signal_fv_up,
                        "market_fv_up": sig.market_fv_up,
                        "edge": sig.edge,
                        "market_type": "5min_updown",
                        "event_start_ts": sig.event_start_ts,
                        "event_end_ts": sig.event_end_ts,
                        "entry_minutes_elapsed": round(sig.minutes_elapsed, 2),
                        "liq_available_usd": liq["available_usd"] if liq else None,
                        "liq_slippage": liq["slippage"] if liq else None,
                        "orderbook_spread": liq.get("spread") if liq else None,
                        "orderbook_best_bid": liq.get("best_bid") if liq else None,
                        "orderbook_best_ask": liq.get("best_ask") if liq else None,
                        "market_gap": round(abs(entry_mkt_fv - liq.get("best_ask", entry_price)), 3) if liq else None,
                        "btc_abs_move_binance": round(abs(sig.btc_now - sig.btc_at_start), 2) if sig.btc_now and sig.btc_at_start else None,
                        "btc_abs_move_chainlink": round(abs(chainlink_price - sig.btc_at_start), 2) if chainlink_price and sig.btc_at_start else None,
                        "btc_abs_move": round(btc_move, 2),
                        "move_risk": round(btc_move * sig.sigma_per_min * 1e6, 0),
                        # live_qualified reflects runtime filter values (from adaptive_config).
                        # Computed once above (live_qualified_flag) and reused here.
                        "live_qualified": live_qualified_flag,
                        "mirror_live": mirror_live,
                        "bin_cl_delta_abs": round(abs(btc_price - chainlink_price), 2) if chainlink_price else None,
                        "btc_at_start_source": "cl_buffer" if sig.btc_at_start and chainlink_price and abs(sig.btc_at_start - chainlink_price) < 50 else "binance_fallback",
                        "velocity_paper": round(velocity, 1),
                        "dir_delta_paper": round(dir_delta, 2),
                        "abs_dir_delta_paper": round(abs_dir_delta, 2),
                        # TA features (from background cache, None if unavailable)
                        **self._get_ta_trade_details(),
                    },
                )

            if paper_trade is None and self._paper_engine:
                continue

            # Phase 3.1: schedule mid_at_30s/60s capture for composite reward
            try:
                from arbo.core.mid_sampler import schedule_mid_capture
                _tid = getattr(paper_trade, "id", None) if paper_trade else None
                schedule_mid_capture(token_id, _tid)
            except Exception:
                pass

            # Live execution — V6.0 Dual Filter (2026-04-06)
            # Two simple pre-entry rules, validated on BOTH datasets:
            #   LIVE:  15t, 13W/2L, 87% WR, +$16.1 (bootstrap P(profit)=92%)
            #   PAPER: 38t, 26W/12L, 68% WR, +$20.1
            # Breakeven WR: 72%. Current WR: 87%. Safety margin: 15pp.
            #
            # 1. velocity = btc_move / entry_minutes ≤ 60 $/min
            #    Slow, sustained moves = reliable momentum (78% paper WR)
            #    Fast spikes >$80/min = bid-ask bounce, mean reversion (6% paper WR)
            #
            # 2. dir_delta = (CL - Binance) in our direction ≤ $15
            #    Small delta = Chainlink nearly caught up = move confirmed by both oracles
            #    Large delta = only Binance moved, CL hasn't confirmed = risky
            #
            # NO reversal exit — data: 9 exits cost $14+ (sold winning positions)
            # NO scoring model — V5.0 scoring was overfit (5/95 paper, 40% WR). Removed.
            # Live params: read from adaptive_config (Watchdog can change at runtime)
            # Falls back to defaults if no override set.
            # Defaults updated 2026-04-12 based on 256-trade sensitivity analysis.
            # See adaptive_config._get_default() for full data.
            _ac = self._adaptive_config
            LIVE_MIN_EDGE = _ac.get("LIVE_MIN_EDGE", 0.30) if _ac else 0.30
            LIVE_MIN_BTC_MOVE = _ac.get("LIVE_MIN_BTC_MOVE", 35.0) if _ac else 35.0
            LIVE_MAX_VELOCITY = _ac.get("LIVE_MAX_VELOCITY", 60.0) if _ac else 60.0
            LIVE_MAX_DIR_DELTA = _ac.get("LIVE_MAX_DIR_DELTA", 15.0) if _ac else 15.0
            live_shares = 0
            live_entry_price = 0.0
            live_fill_status = "skipped"
            live_latency_ms = 0
            is_up = is_up_dir  # already computed above
            _cl = chainlink_price  # None if unavailable — will skip live
            _cl_available = chainlink_price is not None and chainlink_price > 0
            # velocity, dir_delta, abs_dir_delta already computed above (line ~485)
            # Re-use them here
            if (sig.edge >= LIVE_MIN_EDGE and btc_move >= LIVE_MIN_BTC_MOVE
                    and (velocity > LIVE_MAX_VELOCITY
                         or abs_dir_delta > LIVE_MAX_DIR_DELTA)):
                logger.info(
                    "b3_live_filter_skip",
                    direction="UP" if is_up else "DOWN",
                    velocity=f"{velocity:.0f}",
                    dir_delta=f"{dir_delta:.1f}",
                    abs_dir_delta=f"{abs_dir_delta:.1f}",
                    btc_move=f"${btc_move:.0f}",
                    entry_mkt_fv=f"{entry_mkt_fv:.3f}",
                    reason="velocity" if velocity > LIVE_MAX_VELOCITY else "dir_delta",
                )

            if (
                self._execution_mode in ("dual", "live")
                and self._live_executor is not None
                and self._live_daily_pnl > -self._live_daily_loss_limit
                and sig.edge >= LIVE_MIN_EDGE
                and btc_move >= LIVE_MIN_BTC_MOVE
                and velocity <= LIVE_MAX_VELOCITY
                and _cl_available  # CL must confirm (no fallback to Binance)
                and abs_dir_delta <= LIVE_MAX_DIR_DELTA
                # NO model FV cap here — fill price cap on executor handles ITM
                # (MAX_ENTRY_MKT_FV is paper scanner's constant, wrong metric for live)
            ):
                logger.info(
                    "b3_live_qualified",
                    direction="UP" if sig.direction == 1 else "DOWN",
                    edge=f"{sig.edge:.3f}",
                    btc_move=f"${btc_move:.0f}",
                    model_fv=f"{entry_mkt_fv:.3f}",
                    velocity=f"{velocity:.0f}",
                    abs_dir_delta=f"{abs_dir_delta:.1f}",
                )
                try:
                    # Refresh wallet balance every 60s
                    if now - self._live_capital_last_check > 60:
                        bal = await self._live_executor.get_balance()
                        if bal > 10:
                            self._live_capital = bal
                        else:
                            self._live_capital = self._live_capital_fallback
                        self._live_capital_last_check = now
                        logger.info(
                            "b3_live_balance",
                            balance=f"${self._live_capital:.2f}",
                            source="wallet" if bal > 10 else "fallback",
                        )

                    # Same % sizing as paper, but on live wallet balance
                    live_size = min(
                        self._live_capital * raw_pct, MAX_BET_SIZE,
                    )
                    # Dynamic min order size: larger of
                    #   (a) wallet-scaled floor: 1.5% of capital, clamped [$2, $10]
                    #   (b) Polymarket 5-share minimum: 5 × entry_price
                    # (b) is the hard platform constraint — without it, live
                    # executor produces <5 shares → "Size (N) lower than the
                    # minimum: 5" rejection. Observed 19× before this fix.
                    # +1 cent safety buffer protects against int() rounding.
                    wallet_floor = max(2.0, min(10.0, self._live_capital * 0.015))
                    shares_floor = 5.0 * entry_price + 0.01
                    dynamic_min = max(wallet_floor, shares_floor)
                    if live_size < dynamic_min:
                        live_fill_status = "too_small"
                        raise ValueError(
                            f"Live size ${live_size:.2f} < dynamic min "
                            f"${dynamic_min:.2f} (wallet_floor=${wallet_floor:.2f}, "
                            f"shares_floor=${shares_floor:.2f} for price {entry_price:.3f})"
                        )
                    fill = await self._live_executor.buy(
                        token_id=token_id,
                        price=entry_price,
                        size_usdc=live_size,
                        neg_risk=False,
                        tick_size="0.01",
                        maker_timeout_s=self._live_entry_timeout_s,
                        max_price=LIVE_MAX_FILL_PRICE,  # 278-trade data: >0.75 fills net lose $40
                    )
                    live_shares = fill.shares_filled
                    live_entry_price = float(fill.fill_price) if fill.fill_price else 0.0
                    live_fill_status = fill.status
                    live_latency_ms = fill.latency_ms

                    # Defense-in-depth: pre-order max_price check should prevent
                    # fills above the cap. If we still see one, something drifted
                    # between the check and the fill — surface as warning.
                    # Position is real (money spent), so we still track it.
                    if live_entry_price > LIVE_MAX_FILL_PRICE and live_shares > 0:
                        logger.warning(
                            "b3_live_fill_above_cap",
                            fill_price=live_entry_price,
                            max_price=LIVE_MAX_FILL_PRICE,
                            msg="UNEXPECTED: fill above cap despite pre-order guard",
                        )

                    logger.info(
                        "b3_live_entry",
                        status=live_fill_status,
                        shares=live_shares,
                        price=live_entry_price,
                        latency_ms=fill.latency_ms,
                        paper_price=round(entry_price, 4),
                        slippage=(
                            round(live_entry_price - entry_price, 4) if live_entry_price else 0
                        ),
                    )

                    # V6.0: no post-fill rejection — hold all filled positions
                    # (reversal exit removed, no scoring threshold)

                except Exception as e:
                    live_fill_status = "error"
                    logger.error("b3_live_entry_error", error=str(e))

                # Update paper trade's trade_details with live fill info
                has_td = (
                    paper_trade
                    and hasattr(paper_trade, "trade_details")
                    and paper_trade.trade_details
                )
                if has_td:
                    paper_trade.trade_details["live_entry_price"] = live_entry_price
                    paper_trade.trade_details["live_entry_shares"] = live_shares
                    paper_trade.trade_details["live_fill_status"] = live_fill_status
                    paper_trade.trade_details["live_entry_latency_ms"] = live_latency_ms
                    paper_trade.trade_details["live_size_usd"] = round(live_size, 2)
                    paper_trade.trade_details["live_capital"] = self._live_capital
                    # V6.0 data collection (for future autoresearch)
                    if live_entry_price > 0:
                        _ee = entry_mkt_fv - live_entry_price
                        _ftm = live_entry_price / entry_mkt_fv if entry_mkt_fv > 0 else 1.0
                        _cl_r = abs((_cl or 0) - (sig.btc_at_start or 0)) / btc_move if btc_move > 0 else 0
                        _z = btc_move / (sig.btc_at_start * sig.sigma_per_min * math.sqrt(5)) if sig.btc_at_start and sig.sigma_per_min else 0
                        paper_trade.trade_details["velocity"] = round(velocity, 1)
                        paper_trade.trade_details["dir_delta"] = round(dir_delta, 2)
                        paper_trade.trade_details["eff_edge"] = round(_ee, 4)
                        paper_trade.trade_details["z_score"] = round(_z, 4)
                        paper_trade.trade_details["sigma_norm"] = round(sig.sigma_per_min / 0.0003, 4) if sig.sigma_per_min else None
                        paper_trade.trade_details["cl_ratio"] = round(_cl_r, 4)
                        paper_trade.trade_details["fill_to_model"] = round(_ftm, 4)
                        paper_trade.trade_details["combined_risk"] = round(velocity / LIVE_MAX_VELOCITY + abs_dir_delta / LIVE_MAX_DIR_DELTA, 3)
                        paper_trade.trade_details["v6_filters"] = f"vel={velocity:.0f}≤{LIVE_MAX_VELOCITY} |dd|={abs_dir_delta:.1f}≤{LIVE_MAX_DIR_DELTA}"

            # Project PARALLEL: per-variant paper trades — run FIRST (before
            # mirror cancel), so challengers always get their chance regardless
            # of champion's live fill outcome.
            try:
                await self._place_challenger_paper_trades(
                    sig=sig,
                    token_id=token_id,
                    entry_price=entry_price,
                    btc_move=btc_move,
                    velocity=velocity,
                    abs_dir_delta=abs_dir_delta,
                )
            except Exception as _e:
                logger.warning(
                    "b3_challenger_entry_error",
                    error=str(_e),
                    error_type=type(_e).__name__,
                )

            # MIRROR FAILURE PROPAGATION: if we're in dual+mirror mode AND live
            # didn't actually fill shares, cancel the paper trade so paper PnL
            # reflects real-world outcomes (not idealized fills). This keeps
            # paper <-> live 1:1 comparable even in partial/error edge cases.
            live_really_filled = live_shares > 0 and live_fill_status in (
                "filled", "partial",
            )
            if mirror_live and not live_really_filled:
                await self._cancel_mirror_paper_trade(
                    token_id=token_id,
                    cond_id=sig.condition_id,
                    size=actual_size,
                    live_fill_status=live_fill_status,
                )
                # Don't track position — paper has been unwound
                executed.append(sig)
                continue

            # Track position
            self._open_positions[token_id] = B3Position(
                condition_id=sig.condition_id,
                token_id=token_id,
                direction=sig.direction,
                entry_mkt_fv=sig.entry_price,
                entry_time=now,
                event_start_ts=sig.event_start_ts,
                event_end_ts=sig.event_end_ts,
                btc_at_start=sig.btc_at_start,
                btc_at_entry=sig.btc_now,
                sigma_per_min=sig.sigma_per_min,
                shares=shares,
                question=sig.question,
                live_shares=live_shares,
                live_entry_price=live_entry_price,
                live_fill_status=live_fill_status,
                live_latency_ms=live_latency_ms,
            )
            self._scanner.mark_traded(sig.condition_id)

            logger.info(
                "b3_entry",
                direction="UP" if sig.direction == 1 else "DOWN",
                entry_fv=f"{sig.entry_price:.3f}",
                edge=f"{sig.edge:.3f}",
                btc=f"${sig.btc_now:.0f}",
                size=f"${actual_size:.2f}",
                shares=f"{shares:.0f}",
                live=live_fill_status if self._execution_mode != "paper" else None,
            )

            # (Challenger eval moved earlier — before mirror cancel — so it
            # runs regardless of champion's live outcome.)

            executed.append(sig)

        return executed

    async def _place_challenger_paper_trades(
        self,
        sig: B3Signal,
        token_id: str,
        entry_price: float,
        btc_move: float,
        velocity: float,
        abs_dir_delta: float,
    ) -> None:
        """For each challenger variant that would qualify this signal,
        create independent paper trade with challenger's own params.

        Design: each variant gets its own paper_trades row keyed by
        variant_id in trade_details. Size = live_capital × variant's
        POSITION_PCT (mirror of champion's sizing rule). Entry = same
        best_ask as champion (fair comparison — both would have tried
        to fill at same price). Exit = resolution in B3 5-min. PnL
        computed at resolution and written via update_trade_by_id.
        """
        if not self._paper_engine or self._live_capital <= 0:
            logger.warning(
                "b3_challenger_guard_hit",
                paper_engine=bool(self._paper_engine),
                live_capital=self._live_capital,
            )
            return
        variants = self._get_variants()
        if not variants:
            logger.warning("b3_challenger_no_variants")
            return
        n_challenger = sum(1 for v in variants if v.status == "challenger")
        logger.info(
            "b3_challenger_eval_start",
            token=token_id[:16],
            variants=len(variants),
            challengers=n_challenger,
            edge=round(sig.edge, 3),
            vel=round(velocity, 1),
            dd=round(abs_dir_delta, 1),
            fill=round(entry_price, 3),
        )
        # Skip if already have variant position for this token (avoid double entry)
        now = time.time()

        for v in variants:
            if v.status != "challenger":
                continue  # Champion already traded above
            pos_key = (v.variant_id, token_id)
            if pos_key in self._variant_positions:
                continue  # Already positioned this variant on this token
            # Apply variant's own gate filter
            qualified, _skip = self._b3_gates(
                edge=sig.edge,
                btc_move=btc_move,
                velocity=velocity,
                abs_dir_delta=abs_dir_delta,
                fill_price=entry_price,
                entry_minute=int(sig.minutes_elapsed or 0),
                entry_mkt_fv=sig.market_fv_up if sig.direction == 1 else (1 - sig.market_fv_up),
                params=v.params,
            )
            logger.info(
                "b3_challenger_gate_result",
                variant_id=v.variant_id, qualified=qualified, skip=_skip,
            )
            if not qualified:
                continue
            # Sizing: variant's own POSITION_PCT × live_capital (mirror rule)
            v_pct = min(
                float(v.params.get("POSITION_PCT", POSITION_PCT)),
                sig.edge * float(v.params.get("EDGE_SCALING", EDGE_SCALING)),
            )
            v_size = min(self._live_capital * v_pct, MAX_BET_SIZE)
            v_min = max(2.0, min(10.0, self._live_capital * 0.015))
            if v_size < v_min:
                logger.warning(
                    "b3_challenger_size_too_small",
                    variant_id=v.variant_id,
                    v_size=round(v_size, 2), v_min=round(v_min, 2),
                )
                continue
            v_shares = int(v_size / entry_price)
            if v_shares < 1:
                logger.warning(
                    "b3_challenger_shares_zero",
                    variant_id=v.variant_id,
                    v_size=v_size, entry_price=entry_price,
                )
                continue

            # Place paper trade — own DB row with variant_id tagged
            try:
                v_trade = self._paper_engine.place_trade(
                    market_condition_id=sig.condition_id,
                    token_id=token_id,
                    side="BUY",
                    market_price=Decimal(str(round(sig.entry_price, 4))),
                    model_prob=Decimal(str(round(
                        sig.signal_fv_up if sig.direction == 1 else 1 - sig.signal_fv_up, 4,
                    ))),
                    layer=0,
                    market_category="crypto_5min",
                    fee_enabled=False,
                    strategy=STRATEGY_NAME,
                    pre_computed_size=Decimal(str(round(v_shares * entry_price, 2))),
                    clob_fill_price=Decimal(str(round(entry_price, 4))),
                    trade_details={
                        "variant_id": v.variant_id,
                        "is_shadow_variant": True,  # Challenger, not champion live
                        "direction": "up" if sig.direction == 1 else "down",
                        "entry_price": entry_price,
                        "edge": sig.edge,
                        "btc_abs_move": round(btc_move, 2),
                        "velocity_paper": round(velocity, 1),
                        "abs_dir_delta_paper": round(abs_dir_delta, 2),
                        "event_start_ts": sig.event_start_ts,
                        "event_end_ts": sig.event_end_ts,
                        "parent_variant": v.parent_variant,
                    },
                    bypass_risk_check=True,  # Shadow variant — counterfactual only
                )
            except Exception as e:
                logger.warning(
                    "b3_challenger_place_trade_error",
                    variant_id=v.variant_id, error=str(e),
                )
                continue
            if v_trade is None:
                logger.warning(
                    "b3_challenger_place_trade_null",
                    variant_id=v.variant_id,
                    reason="paper_engine.place_trade returned None",
                )
                continue
            # Race guard: reserve pos_key before await (save yields)
            self._variant_positions[pos_key] = None  # type: ignore[assignment]
            # Save to DB immediately and use the returned DB id (not
            # paper_engine local trade.id — those diverge after ~4k trades
            # and update_trade_by_id would target wrong row otherwise).
            db_id: int | None = None
            try:
                db_id = await self._paper_engine.save_trade_to_db(v_trade)
            except Exception as e:
                logger.warning(
                    "b3_challenger_save_trade_error",
                    variant_id=v.variant_id, error=str(e),
                )
            if db_id is None:
                logger.warning(
                    "b3_challenger_save_returned_null",
                    variant_id=v.variant_id,
                )
                self._variant_positions.pop(pos_key, None)
                continue
            self._variant_positions[pos_key] = B3VariantPosition(
                variant_id=v.variant_id,
                paper_trade_id=db_id,
                condition_id=sig.condition_id,
                token_id=token_id,
                direction=sig.direction,
                entry_price=entry_price,
                shares=v_shares,
                size_usd=v_shares * entry_price,
                event_end_ts=sig.event_end_ts,
            )
            logger.info(
                "b3_challenger_entry",
                variant_id=v.variant_id,
                direction="UP" if sig.direction == 1 else "DOWN",
                entry=f"{entry_price:.3f}",
                shares=v_shares,
                size=f"${v_shares * entry_price:.2f}",
                trade_id=int(v_trade.id),
            )

    async def _cancel_mirror_paper_trade(
        self, token_id: str, cond_id: str, size: float, live_fill_status: str,
    ) -> None:
        """Unwind paper trade when live failed to fill (mirror mode).

        Ordering invariant (fixes orphan bug):
        1. Mark in-memory PaperTrade status=SOLD + exit_reason. This blocks
           main_rdh's save loop from inserting it as status=open later.
        2. Save to DB directly with status=sold. Creates the authoritative
           DB row in one shot — no UPDATE race with save.
        3. Debounce re-entry on this token (120s) to prevent signal cascade.

        Previous bug: update_resolved_trades_in_db fired before save_trade_to_db
        (main_rdh runs save after poll_cycle returns), so UPDATE matched 0 rows
        and the subsequent save inserted with status=open → orphan forever.
        """
        from datetime import datetime, timezone
        try:
            # 1. Mark in-memory trade SOLD (main_rdh skips non-open trades).
            matched_trade = None
            if self._paper_engine is not None:
                from arbo.core.paper_engine import POLYGON_GAS_COST_USD, TradeStatus
                for t in reversed(self._paper_engine._trades):
                    if t.token_id == token_id and t.status == TradeStatus.OPEN:
                        t.status = TradeStatus.SOLD
                        t.actual_pnl = Decimal("0")
                        t.exit_reason = f"live_mirror_failed_{live_fill_status}"
                        t.resolved_at = datetime.now(timezone.utc)
                        matched_trade = t
                        break
                # Refund balance: place_trade deducted (size + gas) for this
                # champion trade, and the position never materialized in live.
                # Without refund, each mirror-cancel leaks ~$3-5 from paper
                # balance → over hundreds of cancels, sizing underfunds future
                # trades. load_state_from_db recomputes on restart but runtime
                # drift matters for in-session accounting.
                if matched_trade is not None and token_id in self._paper_engine._positions:
                    self._paper_engine._balance += (
                        matched_trade.size + POLYGON_GAS_COST_USD
                    )
                # Remove in-memory position if any (mirror mode may have created one)
                self._paper_engine._positions.pop(token_id, None)

            # 2. Persist cancel directly — inserts with status=sold, avoids
            #    the UPDATE-before-INSERT race.
            if self._paper_engine and matched_trade is not None:
                await self._paper_engine.save_trade_to_db(matched_trade)

            # 3. Debounce re-entry on this token to prevent cascade.
            self._last_mirror_attempt[token_id] = time.time()

            # Risk manager bookkeeping unchanged
            if self._risk_manager:
                try:
                    self._risk_manager.post_trade_update(
                        cond_id, "crypto_5min",
                        Decimal(str(round(size, 2))),
                        pnl=Decimal("0"),
                    )
                    self._risk_manager.strategy_post_trade(
                        STRATEGY_NAME,
                        Decimal(str(round(size, 2))),
                        pnl=Decimal("0"),
                    )
                except Exception:
                    pass
            logger.info(
                "b3_mirror_paper_cancelled",
                token=token_id[:20],
                live_fill_status=live_fill_status,
                trade_id=getattr(matched_trade, "id", None),
                reason="paper unwound to match live no-fill outcome",
            )
        except Exception as e:
            logger.warning("b3_cancel_mirror_error", error=str(e))

    # ═══════════════════════════════════════════════════════════════════════
    # LIQUIDITY CHECK
    # ═══════════════════════════════════════════════════════════════════════

    # Max slippage as fraction of our edge before we reduce size
    _MAX_SLIPPAGE_FRAC = 0.25  # Slippage must be < 25% of edge

    async def _check_liquidity(
        self, token_id: str, target_price: float, desired_size: float,
    ) -> dict | None:
        """Fetch orderbook and compute optimal trade size based on liquidity.

        Walks the ask side of the orderbook to find how much we can buy
        without slippage exceeding _MAX_SLIPPAGE_FRAC of our edge.

        Args:
            token_id: CLOB token to check.
            target_price: Our target entry price (FV + spread/2).
            desired_size: How much USD we want to trade.

        Returns:
            Dict with safe_size, available_usd, avg_price, slippage.
            None if orderbook fetch fails (proceed with desired_size).
        """
        try:
            import aiohttp

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=3),
            ) as session, session.get(
                f"https://clob.polymarket.com/book?token_id={token_id}",
            ) as resp:
                if resp.status != 200:
                    return None
                book = await resp.json()

            asks = book.get("asks", [])
            bids = book.get("bids", [])
            if not asks:
                return {"safe_size": 0, "available_usd": 0, "avg_price": 0, "slippage": 0, "spread": None}

            # Sort asks ascending by price
            asks_sorted = sorted(asks, key=lambda a: float(a["price"]))

            # Only consider asks in tradeable range (near our target price)
            max_price = min(target_price + 0.15, 0.95)
            min_price = max(target_price - 0.15, 0.05)
            relevant = [
                a for a in asks_sorted
                if min_price <= float(a["price"]) <= max_price
            ]

            if not relevant:
                return {"safe_size": 0, "available_usd": 0, "avg_price": 0, "slippage": 0}

            best_ask = float(relevant[0]["price"])
            total_available_usd = sum(
                float(a["size"]) * float(a["price"]) for a in relevant
            )

            # Walk asks to find max size where slippage < threshold
            # Slippage = avg_fill_price - best_ask
            max_slip = SPREAD * 2  # Max acceptable slippage ($0.02)
            filled_shares = 0.0
            total_cost = 0.0
            safe_size = 0.0

            for a in relevant:
                p = float(a["price"])
                s = float(a["size"])

                # Simulate filling this level
                test_shares = filled_shares + s
                test_cost = total_cost + s * p
                test_avg = test_cost / test_shares if test_shares > 0 else 0
                test_slip = test_avg - best_ask

                if test_slip > max_slip:
                    # This level would push slippage too high
                    # Partial fill: how many shares at this price keeps slip OK?
                    if filled_shares > 0:
                        # slip = (total_cost + n*p) / (filled + n) - best_ask <= max_slip
                        # total_cost + n*p <= (filled + n) * (best_ask + max_slip)
                        # n * (p - best_ask - max_slip) <= filled * (best_ask + max_slip) - total_cost
                        denom = p - best_ask - max_slip
                        if denom > 0:
                            max_n = (filled_shares * (best_ask + max_slip) - total_cost) / denom
                            max_n = max(0, min(s, max_n))
                            filled_shares += max_n
                            total_cost += max_n * p
                    break

                filled_shares += s
                total_cost += s * p

            safe_size = min(total_cost, desired_size)
            avg_price = total_cost / filled_shares if filled_shares > 0 else 0
            slippage = avg_price - best_ask if best_ask > 0 else 0

            logger.info(
                "b3_liquidity_check",
                token=token_id[:20],
                target_price=f"{target_price:.3f}",
                desired=f"${desired_size:.0f}",
                available=f"${total_available_usd:.0f}",
                safe=f"${safe_size:.0f}",
                best_ask=f"{best_ask:.3f}",
                avg_fill=f"{avg_price:.3f}",
                slippage=f"{slippage:.4f}",
            )

            # Compute bid-ask spread
            best_bid = 0.0
            if bids:
                best_bid = max(float(b["price"]) for b in bids)
            spread = round(best_ask - best_bid, 4) if best_bid > 0 else None

            return {
                "safe_size": safe_size,
                "available_usd": total_available_usd,
                "avg_price": avg_price,
                "slippage": slippage,
                "spread": spread,
                "best_bid": best_bid,
                "best_ask": best_ask,
            }

        except Exception as e:
            logger.debug("b3_liquidity_check_error", error=str(e))
            return None  # Fallback: use desired_size

    # ═══════════════════════════════════════════════════════════════════════
    # TA Features (background cache, logging only — no filtering)
    # ═══════════════════════════════════════════════════════════════════════

    def _get_ta_trade_details(self) -> dict:
        """Read TA features from background cache for trade_details logging.

        Returns dict of TA fields (all None if TA unavailable).
        These are logged to trade_details JSONB for future Watchdog analysis.
        No filtering happens here — just data collection.
        """
        if self._ta_provider is None:
            return {}
        ta = self._ta_provider.get("BTCUSDT")
        if ta is None:
            return {}
        return {
            "ta_rsi_5m": round(ta.rsi_5m, 1) if ta.rsi_5m is not None else None,
            "ta_adx_5m": round(ta.adx_5m, 1) if ta.adx_5m is not None else None,
            "ta_macd_hist_5m": round(ta.macd_hist_5m, 4) if ta.macd_hist_5m is not None else None,
            "ta_bb_width_5m": round(ta.bb_width_5m, 4) if ta.bb_width_5m is not None else None,
            "ta_recommend_5m": ta.recommend_5m,
            "ta_rsi_1h": round(ta.rsi_1h, 1) if ta.rsi_1h is not None else None,
            "ta_adx_1h": round(ta.adx_1h, 1) if ta.adx_1h is not None else None,
            "ta_multi_tf_aligned": ta.multi_tf_aligned,
            "ta_adx_regime": ta.adx_regime,
            "ta_rsi_zone": ta.rsi_zone,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # EXIT: Check Exits
    # ═══════════════════════════════════════════════════════════════════════

    async def check_exits(self) -> list[tuple[str, str, float, int, int, float]]:
        """Check open positions for exit triggers.

        Called every 10 seconds. Recomputes fair value with latest BTC price.

        Returns:
            List of (token_id, reason, exit_price, live_shares, direction,
                     live_entry_price) for triggered exits.
        """
        # Always run variant resolution sweep first — orphaned shadow
        # trades (restart survivors) need resolution even when in-memory
        # positions are empty.
        try:
            await self._resolve_variant_positions()
        except Exception as _e:
            logger.debug("b3_variant_resolve_error", error=str(_e))

        # Champion orphan sweeper — runs regardless of _open_positions state
        # (orphans exist precisely when in-memory tracking is lost/empty).
        now_tick = time.time()
        if now_tick - self._last_orphan_sweep > 900:
            self._last_orphan_sweep = now_tick
            try:
                await self._sweep_champion_orphans()
            except Exception as _e:
                logger.warning("b3_orphan_sweep_error", error=str(_e))

        if not self._open_positions and not self._live_holding:
            return []

        btc_price = None
        if self._binance_ws:
            btc_price = self._binance_ws.get_price("BTCUSDT")

        if not btc_price:
            return []

        now = time.time()
        triggered: list[tuple[str, str, float, int, int, float]] = []

        for token_id, pos in list(self._open_positions.items()):
            elapsed_min = (now - pos.event_start_ts) / 60.0
            t_remaining = WINDOW_MIN - elapsed_min
            hold_min = (now - pos.entry_time) / 60.0

            # V6.0: NO reversal exit — data shows it costs $14+ (sells winning positions)
            # Reversal removed 2026-04-06. Hold all live positions to resolution.

            # Resolution check: event has ended
            if now >= pos.event_end_ts:
                # For live positions: MUST use Polymarket oracle (Chainlink).
                # Binance disagrees with Chainlink on 25% of trades — NEVER
                # fall back to Binance for live. Keep retrying until confirmed.
                if pos.live_shares > 0:
                    try:
                        pm_up_won = await asyncio.wait_for(
                            self._scanner.fetch_resolution(pos.event_start_ts),
                            timeout=15,
                        )
                    except asyncio.TimeoutError:
                        logger.warning("b3_resolution_timeout", token=token_id[:20])
                        pm_up_won = None
                    if pm_up_won is None:
                        wait_s = int(now - pos.event_end_ts)
                        if wait_s > 1800:  # 30 min — force Binance fallback
                            logger.warning("b3_oracle_timeout", token=token_id[:20],
                                           wait_s=wait_s, msg="Falling back to Binance after 30min")
                            resolved_up = btc_price >= pos.btc_at_start
                            won = (pos.direction == 1 and resolved_up) or (
                                pos.direction == -1 and not resolved_up)
                            exit_price = 1.0 if won else 0.0
                            triggered.append((token_id, "resolution", exit_price,
                                pos.live_shares, pos.direction, pos.live_entry_price))
                            continue
                        if wait_s > 600:
                            logger.warning("b3_oracle_slow", token=token_id[:20], wait_s=wait_s)
                        continue
                    won = (pos.direction == 1 and pm_up_won) or (
                        pos.direction == -1 and not pm_up_won
                    )
                elif pos.btc_at_start > 0:
                    # Paper-only: Binance is fine (no real money)
                    resolved_up = btc_price >= pos.btc_at_start
                    won = (pos.direction == 1 and resolved_up) or (
                        pos.direction == -1 and not resolved_up
                    )
                else:
                    won = False
                exit_price = 1.0 if won else 0.0
                triggered.append((
                    token_id, "resolution", exit_price,
                    pos.live_shares, pos.direction, pos.live_entry_price,
                ))
                continue

            if t_remaining <= 0 or pos.btc_at_start <= 0:
                continue

            # Recompute fair values
            log_ratio = math.log(btc_price / pos.btc_at_start)
            sqrt_t = math.sqrt(max(t_remaining, 0.01))

            # Market FV (sigma_scale=1.0)
            sigma_rem_true = pos.sigma_per_min * sqrt_t
            if sigma_rem_true > 1e-12:
                market_fv_up = _norm_cdf(log_ratio / sigma_rem_true)
            else:
                market_fv_up = 1.0 if log_ratio > 0 else (0.0 if log_ratio < 0 else 0.5)
            market_fv_up = max(0.02, min(0.98, market_fv_up))

            # Signal FV (sigma_scale from params)
            sigma_rem_model = pos.sigma_per_min * SIGMA_SCALE * sqrt_t
            if sigma_rem_model > 1e-12:
                signal_fv_up = _norm_cdf(log_ratio / sigma_rem_model)
            else:
                signal_fv_up = 1.0 if log_ratio > 0 else (0.0 if log_ratio < 0 else 0.5)

            # Position-relative values
            pos_mkt_fv = market_fv_up if pos.direction == 1 else (1.0 - market_fv_up)
            pos_signal_fv = signal_fv_up if pos.direction == 1 else (1.0 - signal_fv_up)
            unrealized = pos_mkt_fv - pos.entry_mkt_fv

            reason = ""

            # PAPER_MATCH_LIVE=True: paper holds to resolution like live,
            # no early exits. This makes paper PnL directly comparable to live.
            if not PAPER_MATCH_LIVE:
                if unrealized >= PROFIT_TARGET:
                    reason = "profit"
                elif USE_BTC_STOP:
                    # BTC-price-based stop: linear, no CDF overshoot
                    btc_change = (btc_price - pos.btc_at_entry) / pos.btc_at_entry
                    if (pos.direction == 1 and btc_change <= -BTC_STOP_PCT) or (
                        pos.direction == -1 and btc_change >= BTC_STOP_PCT
                    ):
                        reason = "stop"
                elif unrealized <= -STOP_LOSS:
                    reason = "stop"

                if not reason:
                    if hold_min >= MAX_HOLD_MIN:
                        reason = "time"
                    elif abs(pos_signal_fv - 0.50) < EDGE_EXIT and hold_min >= 0.5:
                        reason = "edge_gone"

            if reason:
                exit_price = pos_mkt_fv - SPREAD / 2  # Sell at bid
                exit_price = max(0.01, exit_price)
                triggered.append((
                    token_id, reason, exit_price,
                    pos.live_shares, pos.direction, pos.live_entry_price,
                ))

        # Process exits
        for token_id, reason, exit_price, _live_shares, _dir, _lep in triggered:
            pos = self._open_positions.pop(token_id, None)
            if pos:
                self._last_exit_time[pos.condition_id] = now
                btc_change_pct = ((btc_price - pos.btc_at_entry) / pos.btc_at_entry * 100
                                  if pos.btc_at_entry > 0 else 0.0)
                logger.info(
                    "b3_exit",
                    direction="UP" if pos.direction == 1 else "DOWN",
                    reason=reason,
                    entry_fv=f"{pos.entry_mkt_fv:.3f}",
                    exit_fv=f"{exit_price:.3f}",
                    pnl=f"{exit_price - pos.entry_mkt_fv - SPREAD / 2:.3f}",
                    btc_change=f"{btc_change_pct:+.3f}%",
                )
                # Never-sell: if live has shares and paper exited early,
                # keep tracking for resolution
                if pos.live_shares > 0 and reason != "resolution":
                    self._live_holding[token_id] = pos

        # Check _live_holding: resolution only (no reversal exit in V6.0)
        for token_id in list(self._live_holding.keys()):
            pos = self._live_holding[token_id]

            if now >= pos.event_end_ts and pos.btc_at_start > 0:
                # Use Polymarket oracle — timeout after 30 min (auto-redeem handles money)
                try:
                    pm_up_won = await asyncio.wait_for(
                        self._scanner.fetch_resolution(pos.event_start_ts),
                        timeout=15,
                    )
                except asyncio.TimeoutError:
                    logger.warning("b3_resolution_timeout_holding", token=token_id[:20])
                    pm_up_won = None
                if pm_up_won is None:
                    wait_s = int(now - pos.event_end_ts)
                    if wait_s > 1800:  # 30 min — give up, auto-redeem already handled it
                        logger.warning("b3_holding_expired", token=token_id[:20],
                                       wait_s=wait_s, msg="Dropping — auto-redeem handles money")
                        self._live_holding.pop(token_id)
                        continue
                    if wait_s > 600:
                        logger.warning("b3_oracle_slow_holding", token=token_id[:20],
                                       wait_s=wait_s)
                    continue
                won = (pos.direction == 1 and pm_up_won) or (
                    pos.direction == -1 and not pm_up_won
                )
                exit_price = 1.0 if won else 0.0
                triggered.append((
                    token_id, "resolution", exit_price,
                    pos.live_shares, pos.direction, pos.live_entry_price,
                ))
                self._live_holding.pop(token_id)
                logger.info(
                    "b3_live_resolved",
                    direction="UP" if pos.direction == 1 else "DOWN",
                    won=won,
                    entry=f"{pos.live_entry_price:.3f}",
                    shares=pos.live_shares,
                )

        # Project PARALLEL: resolve per-variant paper positions
        try:
            await self._resolve_variant_positions()
        except Exception as _e:
            logger.debug("b3_variant_resolve_error", error=str(_e))

        return triggered

    async def _sweep_champion_orphans(self) -> None:
        """Resolve champion paper_trades stuck status=open past event_end_ts.

        Two orphan sources:
          (1) Mirror-cancel legacy — pre-fix rows where UPDATE fired before
              INSERT and left trade status=open with live_fill_status=failed.
          (2) Restart survivors — service died with open champion positions;
              _open_positions is in-memory (not DB-backed), so after restart
              B3 has no knowledge of them. paper_engine.load_state_from_db
              restores _positions table but that's a separate concern.

        This sweeper queries DB directly, resolves via Binance (paper-only
        live_shares=0) or Chainlink (live_shares>0), updates each row via
        update_trade_by_id. No in-memory state reconstruction needed —
        orphans are terminal, we just finalize them.
        """
        if self._paper_engine is None:
            return
        from arbo.utils.db import get_session_factory
        import sqlalchemy as sa
        now_ts = time.time()
        factory = get_session_factory()
        try:
            async with factory() as session:
                result = await session.execute(
                    sa.text("""
                        SELECT id, market_condition_id, token_id,
                               trade_details->>'direction' AS direction,
                               (trade_details->>'btc_at_start')::float AS btc_at_start,
                               (trade_details->>'event_end_ts')::float AS event_end_ts,
                               (trade_details->>'event_start_ts')::float AS event_start_ts,
                               COALESCE((trade_details->>'live_entry_shares')::int, 0) AS live_shares,
                               size AS size_usd,
                               price AS entry_price
                        FROM paper_trades
                        WHERE strategy = 'B3'
                          AND status = 'open'
                          AND (trade_details->>'is_shadow_variant' IS NULL
                               OR trade_details->>'is_shadow_variant' != 'true')
                          AND (trade_details->>'event_end_ts')::float IS NOT NULL
                          AND (trade_details->>'event_end_ts')::float < :now
                        LIMIT 50
                    """),
                    {"now": now_ts},
                )
                orphans = list(result.mappings())
        except Exception as e:
            logger.warning("b3_orphan_query_error", error=str(e))
            return

        if not orphans:
            return

        logger.info("b3_orphan_sweep_start", count=len(orphans))
        resolved_count = 0
        # Cache per-event resolution to avoid duplicate oracle calls
        resolution_cache: dict[float, bool | None] = {}

        for row in orphans:
            event_start = row["event_start_ts"]
            if event_start is None or event_start == 0:
                event_start = float(row["event_end_ts"]) - WINDOW_MIN * 60

            direction = 1 if row["direction"] == "up" else -1

            # Resolution: always use Polymarket oracle (Chainlink via scanner).
            # Why not Binance fallback? For stale orphans the "current" BTC
            # price is not the resolution price — we'd need historical klines
            # at event_end_ts. Polymarket's Chainlink oracle is already the
            # authoritative resolution snapshot; if unavailable, skip and
            # retry next sweep (throttle is 15 min, so eventually resolves).
            # live_shares distinction is irrelevant for resolution correctness.
            if event_start in resolution_cache:
                up_won = resolution_cache[event_start]
            else:
                try:
                    up_won = await asyncio.wait_for(
                        self._scanner.fetch_resolution(event_start),
                        timeout=10,
                    )
                except Exception:
                    up_won = None
                resolution_cache[event_start] = up_won

            if up_won is None:
                # Skip — retry next sweep
                continue

            won = (direction == 1 and up_won) or (direction == -1 and not up_won)
            exit_price = 1.0 if won else 0.0

            # PnL computation: prefer recorded live shares (accurate), else
            # derive from paper size/price (size_usd / entry_p).
            try:
                live_shares = int(row["live_shares"] or 0)
                entry_p = float(row["entry_price"] or 0)
                if live_shares > 0:
                    shares = live_shares
                else:
                    size_usd = float(row["size_usd"] or 0)
                    shares = int(size_usd / entry_p) if entry_p > 0 else 0
                pnl = shares * (exit_price - entry_p)

                await self._paper_engine.update_trade_by_id(
                    trade_id=int(row["id"]),
                    status="won" if won else "lost",
                    actual_pnl=Decimal(str(round(pnl, 4))),
                    exit_price=Decimal(str(exit_price)),
                    exit_reason="orphan_sweep",
                )
                resolved_count += 1
            except Exception as e:
                logger.warning(
                    "b3_orphan_resolve_error",
                    trade_id=row["id"], error=str(e),
                )

        if resolved_count > 0:
            logger.info("b3_orphan_sweep_done", resolved=resolved_count,
                        total_found=len(orphans))

    async def _resolve_variant_positions(self) -> None:
        """Resolve per-variant paper positions at event end.

        DB-backed: handles both in-memory positions and restart orphans.
        In-memory dict clears on service restart, but DB rows persist.
        """
        now = time.time()
        to_resolve: list[tuple[str, Any]] = []
        for key, pos in list(self._variant_positions.items()):
            if pos is None:
                continue  # Race-guard sentinel
            if now >= pos.event_end_ts:
                to_resolve.append((key, pos))

        # DB orphans: shadow trades still 'open' past event_end_ts that
        # aren't in _variant_positions (e.g. after service restart)
        try:
            from arbo.utils.db import get_session_factory
            import sqlalchemy as sa
            factory = get_session_factory()
            async with factory() as session:
                result = await session.execute(
                    sa.text("""
                        SELECT id, market_condition_id, token_id,
                               trade_details->>'variant_id' AS variant_id,
                               trade_details->>'direction' AS direction,
                               (trade_details->>'entry_price')::float AS entry_price,
                               (trade_details->>'event_end_ts')::float AS event_end_ts,
                               size AS size_usd
                        FROM paper_trades
                        WHERE strategy = 'B3'
                          AND trade_details->>'is_shadow_variant' = 'true'
                          AND status = 'open'
                          AND (trade_details->>'event_end_ts')::float < :now
                    """),
                    {"now": now},
                )
                for row in result.mappings():
                    key = (row["variant_id"], row["token_id"])
                    if key in self._variant_positions:
                        continue
                    entry_p = float(row["entry_price"] or 0)
                    size_usd = float(row["size_usd"] or 0)
                    shares = int(size_usd / entry_p) if entry_p > 0 else 0
                    pos = B3VariantPosition(
                        variant_id=row["variant_id"],
                        paper_trade_id=int(row["id"]),
                        condition_id=row["market_condition_id"],
                        token_id=row["token_id"],
                        direction=1 if row["direction"] == "up" else -1,
                        entry_price=entry_p,
                        shares=shares,
                        size_usd=size_usd,
                        event_end_ts=float(row["event_end_ts"] or 0),
                    )
                    to_resolve.append((key, pos))
        except Exception as e:
            logger.warning("b3_variant_db_query_error", error=str(e))

        if not to_resolve:
            return

        # Group by condition_id — one resolution fetch per event covers all variants
        by_cond: dict[str, list] = {}
        for key, pos in to_resolve:
            by_cond.setdefault(pos.condition_id, []).append((key, pos))

        for cond_id, positions in by_cond.items():
            # Use event_start_ts from any position (all share same condition)
            event_start = positions[0][1].event_end_ts - WINDOW_MIN * 60
            try:
                pm_up_won = await asyncio.wait_for(
                    self._scanner.fetch_resolution(event_start),
                    timeout=15,
                )
            except (asyncio.TimeoutError, Exception):
                continue  # Retry next cycle
            if pm_up_won is None:
                continue

            for key, pos in positions:
                # PnL per variant position
                if pos.direction == 1:
                    won = pm_up_won
                else:
                    won = not pm_up_won
                # Exit price: binary market resolution — 1.0 if our token won, 0.0 else
                exit_price = 1.0 if won else 0.0
                pnl = pos.shares * (exit_price - pos.entry_price)
                # Sign convention: pos.direction irrelevant here (shares were for
                # our token, exit is our token's value)
                try:
                    if self._paper_engine:
                        await self._paper_engine.update_trade_by_id(
                            trade_id=pos.paper_trade_id,
                            status="won" if won else "lost",
                            actual_pnl=Decimal(str(round(pnl, 4))),
                            exit_price=Decimal(str(exit_price)),
                        )
                    logger.info(
                        "b3_variant_resolved",
                        variant_id=pos.variant_id,
                        won=won,
                        entry=f"{pos.entry_price:.3f}",
                        shares=pos.shares,
                        pnl=f"${pnl:+.2f}",
                        trade_id=pos.paper_trade_id,
                    )
                except Exception as e:
                    logger.warning(
                        "b3_variant_resolve_update_error",
                        variant_id=pos.variant_id, error=str(e),
                    )
                self._variant_positions.pop(key, None)

    async def sell_live_position(
        self, token_id: str, exit_reason: str, paper_exit_price: float,
    ) -> dict | None:
        """Sell live position via taker (immediate). Returns comparison dict or None."""
        if self._live_executor is None:
            return None

        try:
            fill = await self._live_executor.sell(
                token_id=token_id,
                price=paper_exit_price,
                neg_risk=False,
                tick_size="0.01",
                maker_timeout_s=self._live_exit_maker_timeout_s,
                skip_sync=True,  # B3 holds 1-3 min, Data API too slow
            )
            live_exit_price = float(fill.fill_price) if fill.fill_price else 0.0
            slippage = round(paper_exit_price - live_exit_price, 4) if live_exit_price else 0
            logger.info(
                "b3_live_exit",
                reason=exit_reason,
                status=fill.status,
                shares=fill.shares_filled,
                live_price=live_exit_price,
                paper_price=round(paper_exit_price, 4),
                slippage=slippage,
                latency_ms=fill.latency_ms,
            )
            # Track live PnL for daily kill switch
            if fill.shares_filled > 0 and live_exit_price > 0:
                self._live_daily_pnl += fill.usdc_spent
            # Warn if shares stuck (will auto-resolve at market end)
            stuck = fill.shares_requested - fill.shares_filled
            if stuck > 0:
                logger.warning(
                    "b3_live_stranded_shares",
                    token=token_id[:20],
                    stuck=stuck,
                    msg="Will auto-resolve at market end ($1 or $0)",
                )
            return {
                "live_exit_status": fill.status,
                "live_exit_price": live_exit_price,
                "live_exit_shares": fill.shares_filled,
                "live_exit_latency_ms": fill.latency_ms,
                "live_exit_slippage": slippage,
            }
        except Exception as e:
            logger.error("b3_live_exit_error", token=token_id[:20], error=str(e))
            return {"live_exit_status": "error", "live_exit_error": str(e)}

    def handle_resolution(self, condition_id: str, pnl: float) -> None:
        """Clean up position after market resolves."""
        to_remove = [
            tid for tid, pos in self._open_positions.items()
            if pos.condition_id == condition_id
        ]
        for tid in to_remove:
            pos = self._open_positions.pop(tid, None)
            if pos:
                logger.info(
                    "b3_resolved",
                    direction="UP" if pos.direction == 1 else "DOWN",
                    pnl=f"${pnl:.2f}",
                )
