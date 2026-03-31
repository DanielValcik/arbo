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

import math
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from arbo.models.volatility_model import VolatilityEstimator
from arbo.strategies.b3_quality_gate import (
    BTC_STOP_PCT,
    EDGE_EXIT,
    EDGE_SCALING,
    MAX_BET_SIZE,
    MAX_HOLD_MIN,
    MAX_SHARES,
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
    ) -> None:
        self._risk_manager = risk_manager
        self._paper_engine = paper_engine
        self._binance_ws = binance_ws
        self._rtds_feed = rtds_feed  # Chainlink resolution price feed
        self._execution_mode = execution_mode  # "paper", "dual", "live"
        self._live_executor = live_executor

        # Live config — B3_LIVE_CAPITAL is the source of truth for sizing
        import os
        self._live_capital = float(os.getenv("B3_LIVE_CAPITAL", "300"))
        self._live_capital_last_check: float = 0.0
        self._live_entry_timeout_s = int(os.getenv("B3_LIVE_ENTRY_TIMEOUT_S", "10"))
        self._live_exit_maker_timeout_s = int(os.getenv("B3_LIVE_EXIT_MAKER_TIMEOUT_S", "5"))
        self._live_daily_pnl: float = 0.0  # Track live PnL for kill switch
        self._live_daily_loss_limit = float(os.getenv("B3_LIVE_DAILY_LOSS_LIMIT", "50"))

        # B3 scanner (manages event lifecycle)
        self._scanner = B3Scanner()

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
        self._last_exit_time: dict[str, float] = {}  # condition_id → timestamp

    async def init(self) -> None:
        """Initialize scanner and restore state."""
        await self._scanner.init()

        # Restore B3 positions from paper engine.
        # B3 positions are ultra-short-lived (1-3 min). On restart, any open
        # B3 positions are stale (the 5-min event has already resolved).
        # Force-close them instead of restoring — avoids btc_at_start=0 crash.
        if self._paper_engine:
            stale_b3 = [
                pos for pos in self._paper_engine.open_positions
                if getattr(pos, "strategy", "") == STRATEGY_NAME
            ]
            if stale_b3:
                logger.warning(
                    "b3_stale_positions_on_restart",
                    count=len(stale_b3),
                    msg="B3 positions are 1-3 min hold — stale after restart, will be resolved by market",
                )

    async def close(self) -> None:
        """Clean up resources."""
        await self._scanner.close()

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
            # where slippage < MAX_SLIPPAGE_PCT of edge
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
                        "event_end_ts": sig.event_end_ts,
                        "liq_available_usd": liq["available_usd"] if liq else None,
                        "liq_slippage": liq["slippage"] if liq else None,
                    },
                )

            if paper_trade is None and self._paper_engine:
                continue

            # Live execution (dual mode)
            live_shares = 0
            live_entry_price = 0.0
            live_fill_status = "skipped"
            live_latency_ms = 0

            if (
                self._execution_mode in ("dual", "live")
                and self._live_executor is not None
                and self._live_daily_pnl > -self._live_daily_loss_limit
            ):
                try:
                    # Same % sizing as paper, but on live capital (env var)
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
                    )
                    live_shares = fill.shares_filled
                    live_entry_price = float(fill.fill_price) if fill.fill_price else 0.0
                    live_fill_status = fill.status
                    live_latency_ms = fill.latency_ms
                    logger.info(
                        "b3_live_entry",
                        status=fill.status,
                        shares=fill.shares_filled,
                        price=live_entry_price,
                        latency_ms=fill.latency_ms,
                        paper_price=round(entry_price, 4),
                        slippage=(
                            round(live_entry_price - entry_price, 4) if live_entry_price else 0
                        ),
                    )
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
            if not asks:
                return {"safe_size": 0, "available_usd": 0, "avg_price": 0, "slippage": 0}

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

            return {
                "safe_size": safe_size,
                "available_usd": total_available_usd,
                "avg_price": avg_price,
                "slippage": slippage,
            }

        except Exception as e:
            logger.debug("b3_liquidity_check_error", error=str(e))
            return None  # Fallback: use desired_size

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
        if not self._open_positions:
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

            # Resolution check: event has ended
            if now >= pos.event_end_ts:
                # Event resolved — compute final FV based on whether BTC
                # ended above or below S_start (= resolution outcome).
                if pos.btc_at_start > 0:
                    resolved_up = btc_price >= pos.btc_at_start
                    won = (pos.direction == 1 and resolved_up) or (
                        pos.direction == -1 and not resolved_up
                    )
                    # Resolution: token redeems at $1 (won) or $0 (lost)
                    exit_price = 1.0 if won else 0.0
                else:
                    exit_price = pos.entry_mkt_fv  # Fallback
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
