"""Strategy B3_15M — Binance Oracle Scalper (15-min variant).

Fork of strategy_b3.py for 15-minute BTC Up/Down markets. Shared
architecture (Chainlink oracle, never-sell, dual execution), divergent
filters calibrated from shadow autoresearch (144 real signals,
2026-04-12, 5-fold CV rank #1).

Key differences from B3 (5-min):
- 15-min hold horizon (vs 5-min)
- Entry minutes 4-11 (vs 2-3)
- LIVE_MAX_BTC_MOVE=80 cap (5-min has none) — reversal risk above $80
- LIVE_MAX_FILL_PRICE=1.01 (5-min caps at 0.75) — 15m high fill = best bucket
- LIVE_MAX_MARKET_GAP=0.30 filter (15m specific)
- LIVE_MIN_BTC_MOVE=0 (edge>=0.30 replaces lower cap)
- ~5-8 trades/day (vs ~30 for 5-min), 3-4x higher PnL/trade
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from arbo.models.volatility_model import VolatilityEstimator
from arbo.strategies.b3_15m_quality_gate import (
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
from arbo.strategies.b3_15m_scanner import B315MScanner, B315MSignal
from arbo.utils.logger import get_logger

logger = get_logger("strategy_b3_15m")

_SQRT2 = math.sqrt(2.0)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


@dataclass
class B315MPosition:
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


class StrategyB315M:
    """Binance Oracle Scalper — 15-min BTC Up/Down variant."""

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
        self._live_capital_fallback = float(os.getenv("B3_15M_LIVE_CAPITAL", "75"))
        self._live_capital: float = self._live_capital_fallback
        self._live_capital_last_check: float = 0.0
        self._live_entry_timeout_s = int(os.getenv("B3_15M_LIVE_ENTRY_TIMEOUT_S", "15"))
        self._live_exit_maker_timeout_s = int(os.getenv("B3_15M_LIVE_EXIT_MAKER_TIMEOUT_S", "8"))
        self._live_daily_pnl: float = 0.0  # Track live PnL for kill switch
        self._live_daily_loss_limit = float(os.getenv("B3_15M_LIVE_DAILY_LOSS_LIMIT", "20"))

        # B3_15M scanner (manages event lifecycle)
        self._scanner = B315MScanner(rtds_feed=rtds_feed)

        # Volatility estimator (per-minute realized vol from 720 klines)
        # IMPORTANT: Only feed prices once per minute to match backtest's 1-min klines.
        # Binance WS updates every ~1s, but sigma must be per-minute, not per-update.
        self._vol_estimator = VolatilityEstimator(
            window=SIGMA_WINDOW,
            method=SIGMA_METHOD,
        )
        self._last_vol_update_ts: float = 0.0  # Unix ts of last vol estimator feed

        # Open position tracking
        self._open_positions: dict[str, B315MPosition] = {}  # token_id → position
        self._live_holding: dict[str, B315MPosition] = {}  # live positions waiting for resolution
        self._last_exit_time: dict[str, float] = {}  # condition_id → timestamp

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
                    # Event still running. Insert into _live_holding so
                    # check_exits can resolve after event_end. See 5-min
                    # strategy_b3.py for full bug context (2026-04-13).
                    try:
                        live_shares = int(td.get("live_entry_shares") or 0)
                        live_entry_price = float(td.get("live_entry_price") or 0)
                        direction_int = 1 if direction_str == "up" else -1
                        btc_at_start = float(td.get("btc_at_start") or 0)
                        btc_at_entry = float(td.get("btc_now") or btc_at_start)
                        sigma_per_min = float(td.get("sigma") or 0)
                        if live_shares > 0 and live_entry_price > 0:
                            self._live_holding[token_id] = B315MPosition(
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
                                "b3_15m_stale_holding_restored",
                                token=token_id[:20],
                                wait_s=int(event_end_ts - now_ts),
                            )
                    except Exception as e:
                        logger.warning("b3_15m_stale_holding_restore_error",
                                       token=token_id[:20], error=str(e))
                    deferred_count += 1
                    continue

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
                        self._risk_manager.post_trade_update(cond_id, "crypto_15min", size, pnl=pnl)
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
                    self._risk_manager.post_trade_update(cond_id, "crypto_15min", size, pnl=Decimal("0"))
                    self._risk_manager.strategy_post_trade(STRATEGY_NAME, size, pnl=Decimal("0"))
                except Exception:
                    pass

            logger.info("b3_stale_orphaned", token=token_id[:20])
        except Exception as e:
            logger.warning("b3_mark_orphaned_error", error=str(e))

    # ═══════════════════════════════════════════════════════════════════════
    # ENTRY: Poll Cycle
    # ═══════════════════════════════════════════════════════════════════════

    async def poll_cycle(self) -> list[B315MSignal]:
        """Main entry scan. Called every 10-15 seconds.

        1. Fetch events from Gamma API (cached, refetches every 2 min)
        2. Get BTC price from Binance
        3. Compute per-minute realized volatility
        4. Scan events for entry signals (minute 2 check)
        5. Size and execute trades

        Returns:
            List of executed B315MSignal objects.
        """
        executed: list[B315MSignal] = []

        # 0. Record Chainlink price to rolling buffer (for accurate btc_at_start)
        self._scanner.record_cl_price()

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

            available = float(strat_state.allocated - strat_state.deployed)
            if available <= 0:
                continue

            bet_size = min(available * raw_pct, MAX_BET_SIZE)
            if bet_size < MIN_ORDER_SIZE:
                continue

            entry_price = sig.entry_price + SPREAD / 2  # Buy at ask
            if entry_price <= 0.01:
                continue

            # Liquidity check: fetch orderbook and compute max size
            liq = await self._check_liquidity(token_id, entry_price, bet_size)
            if liq is not None:
                bet_size = liq["safe_size"]
                if bet_size < MIN_ORDER_SIZE:
                    logger.info(
                        "b3_low_liquidity",
                        token=token_id[:20],
                        available_usd=f"${liq['available_usd']:.0f}",
                        safe_size=f"${bet_size:.0f}",
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
                if bet_size < MIN_ORDER_SIZE:
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
                market_category="crypto_15min",
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
                    market_category="crypto_15min",
                    fee_enabled=False,  # Maker = 0%
                    strategy=STRATEGY_NAME,
                    pre_computed_size=Decimal(str(round(actual_size, 2))),
                    clob_fill_price=Decimal(str(round(entry_price, 4))),
                    trade_details={
                        # Variant identity (Rapid Mode §11). Current live config
                        # is the single champion per config/variants/b3_15m/champion_v1.yaml.
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
                        # live_qualified reflects runtime filter values (from adaptive_config)
                        # B3_15M defaults from shadow autoresearch rank #1 (2026-04-12)
                        "live_qualified": bool(
                            sig.edge >= (self._adaptive_config.get("LIVE_MIN_EDGE", 0.30) if self._adaptive_config else 0.30)
                            and btc_move >= (self._adaptive_config.get("LIVE_MIN_BTC_MOVE", 0.0) if self._adaptive_config else 0.0)
                            and btc_move <= (self._adaptive_config.get("LIVE_MAX_BTC_MOVE", 80.0) if self._adaptive_config else 80.0)
                            and velocity <= (self._adaptive_config.get("LIVE_MAX_VELOCITY", 100.0) if self._adaptive_config else 100.0)
                            and abs_dir_delta <= (self._adaptive_config.get("LIVE_MAX_DIR_DELTA", 25.0) if self._adaptive_config else 25.0)
                            and abs(entry_mkt_fv - (liq.get("best_ask", entry_price) if liq else entry_price)) <= (self._adaptive_config.get("LIVE_MAX_MARKET_GAP", 0.30) if self._adaptive_config else 0.30)
                        ),
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
            # B3_15M live filters — from shadow autoresearch rank #1 (2026-04-12)
            # 5-fold CV, all folds positive, WR 94.4%, avg $0.222/share.
            _ac = self._adaptive_config
            LIVE_MIN_EDGE = _ac.get("LIVE_MIN_EDGE", 0.30) if _ac else 0.30
            LIVE_MIN_BTC_MOVE = _ac.get("LIVE_MIN_BTC_MOVE", 0.0) if _ac else 0.0
            LIVE_MAX_BTC_MOVE = _ac.get("LIVE_MAX_BTC_MOVE", 80.0) if _ac else 80.0
            LIVE_MAX_VELOCITY = _ac.get("LIVE_MAX_VELOCITY", 100.0) if _ac else 100.0
            LIVE_MAX_DIR_DELTA = _ac.get("LIVE_MAX_DIR_DELTA", 25.0) if _ac else 25.0
            LIVE_MAX_MARKET_GAP = _ac.get("LIVE_MAX_MARKET_GAP", 0.30) if _ac else 0.30
            live_shares = 0
            live_entry_price = 0.0
            live_fill_status = "skipped"
            live_latency_ms = 0
            is_up = is_up_dir
            _cl = chainlink_price
            _cl_available = chainlink_price is not None and chainlink_price > 0
            # Market gap: distance between model FV and actual market best_ask
            _best_ask = liq.get("best_ask", entry_price) if liq else entry_price
            market_gap = abs(entry_mkt_fv - _best_ask)
            if (sig.edge >= LIVE_MIN_EDGE
                    and btc_move <= LIVE_MAX_BTC_MOVE
                    and (velocity > LIVE_MAX_VELOCITY
                         or abs_dir_delta > LIVE_MAX_DIR_DELTA
                         or market_gap > LIVE_MAX_MARKET_GAP)):
                logger.info(
                    "b3_15m_live_filter_skip",
                    direction="UP" if is_up else "DOWN",
                    velocity=f"{velocity:.0f}",
                    dir_delta=f"{dir_delta:.1f}",
                    abs_dir_delta=f"{abs_dir_delta:.1f}",
                    btc_move=f"${btc_move:.0f}",
                    entry_mkt_fv=f"{entry_mkt_fv:.3f}",
                    market_gap=f"{market_gap:.3f}",
                    reason=(
                        "velocity" if velocity > LIVE_MAX_VELOCITY
                        else "dir_delta" if abs_dir_delta > LIVE_MAX_DIR_DELTA
                        else "market_gap"
                    ),
                )

            if (
                self._execution_mode in ("dual", "live")
                and self._live_executor is not None
                and self._live_daily_pnl > -self._live_daily_loss_limit
                and sig.edge >= LIVE_MIN_EDGE
                and btc_move >= LIVE_MIN_BTC_MOVE
                and btc_move <= LIVE_MAX_BTC_MOVE  # 15-min reversal cap (NEW vs 5-min)
                and velocity <= LIVE_MAX_VELOCITY
                and _cl_available
                and abs_dir_delta <= LIVE_MAX_DIR_DELTA
                and market_gap <= LIVE_MAX_MARKET_GAP  # 15-min gap filter (NEW)
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
                    if live_size < MIN_ORDER_SIZE:
                        live_fill_status = "too_small"
                        raise ValueError(
                            f"Live size ${live_size:.1f} < min ${MIN_ORDER_SIZE}"
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

            # Track position
            self._open_positions[token_id] = B315MPosition(
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

            executed.append(sig)

        return executed

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

        return triggered

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
