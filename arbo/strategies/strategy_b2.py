"""Strategy B2 — Crypto Price Edge.

Trades Polymarket crypto price prediction markets using real-time
exchange prices (Binance) and a volatility-based probability model.

Key differences from C2 (weather):
- NOT NegRisk: standard binary markets with real orderbooks
- Exchange price source: Binance WebSocket (not weather forecast)
- Model: Log-normal volatility CDF (not EMOS)
- Higher liquidity: $1-5M per bucket (vs $100-500)
- Faster cycle: 60s poll, 30s exit check (vs 300s/60s)
- Fee model: daily markets have crypto fees, monthly = 0%
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from arbo.connectors.binance_ws import BinanceWSFeed
from arbo.core.fee_model import calculate_taker_fee
from arbo.models.volatility_model import VolatilityEstimator, estimate_crypto_prob
from arbo.strategies.crypto_price_scanner import (
    CryptoSignal,
    compute_hours_to_expiry,
    scan_crypto_markets,
)
from arbo.strategies.crypto_quality_gate import (
    KELLY_RAW_CAP,
    MIN_HOLD_EDGE,
    MIRROR_CANCEL_DEBOUNCE,
    PAPER_MATCH_LIVE,
    PROFIT_TARGET_ABS,
    VOLATILITY_METHOD,
    VOLATILITY_WINDOW,
    SIGMA_SCALE,
    filter_signals,
)
from arbo.utils.logger import get_logger

logger = get_logger("strategy_b2")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

STRATEGY_NAME = "B2"
KELLY_FRACTION = Decimal("0.25")  # Quarter-Kelly
MAX_TRADES_PER_SCAN = 3
MIN_HOURS_BETWEEN_TRADES = 0.5  # 30 min cooldown per token
LATENCY_RECHECK_DELAY_S = 2
VOLATILITY_GUARD_MAX_MOVE = 0.05  # Skip if price moved >5c during recheck
VOLUME_CAP_PCT = 0.05  # Max 5% of 24h volume per trade


@dataclass
class OpenPosition:
    """Tracked open B2 position for exit monitoring."""

    token_id: str
    condition_id: str
    asset: str
    symbol: str
    strike: float
    direction: str
    market_type: str
    entry_price: float  # Polymarket price at entry
    entry_exchange_price: float  # Binance price at entry
    sigma_at_entry: float
    hours_to_expiry_at_entry: float
    entry_time: float
    shares: int = 0
    # Live execution tracking (dual mode) — shares_owned on CLOB, used by
    # exit path to route through live_executor.sell vs paper sell_position.
    live_shares: int = 0
    live_entry_price: float = 0.0
    live_fill_status: str = ""


class StrategyB2:
    """Crypto Price Edge strategy orchestrator."""

    def __init__(
        self,
        risk_manager: Any,
        paper_engine: Any | None = None,
        binance_ws: BinanceWSFeed | None = None,
        orderbook_provider: Any | None = None,
        execution_mode: str = "paper",
        live_executor: Any | None = None,
        live_capital: float = 100.0,
        live_capital_fallback: float = 100.0,
        live_entry_timeout_s: int = 30,
        live_daily_loss_limit: float = 20.0,
    ) -> None:
        self._risk_manager = risk_manager
        self._paper_engine = paper_engine
        self._binance_ws = binance_ws
        self._orderbook_provider = orderbook_provider
        self._execution_mode = execution_mode
        self._live_executor = live_executor
        self._exit_manager_ref: Any | None = None

        # Live trading state (dual/live mode only)
        self._live_capital = live_capital
        self._live_capital_fallback = live_capital_fallback
        self._live_capital_last_check = 0.0
        self._live_entry_timeout_s = live_entry_timeout_s
        self._live_daily_loss_limit = live_daily_loss_limit
        self._live_daily_pnl = 0.0

        # Volatility estimator (maintains rolling price buffer)
        self._vol_estimator = VolatilityEstimator(
            window=VOLATILITY_WINDOW,
            method=VOLATILITY_METHOD,
        )

        # Open position tracking
        self._open_positions: dict[str, OpenPosition] = {}
        self._last_trade_time: dict[str, float] = {}
        # Debounce re-entry after mirror-cancel (see strategy_b3 for rationale).
        self._last_mirror_attempt: dict[str, float] = {}

        # Phase PARALLEL: variant pool cache + shadow dedup
        self._variants_cache: list = []
        self._variants_cache_ts: float = 0.0
        # Dedupe shadow writes per (token_id, minute) — avoids re-eval each scan
        self._shadow_written: set[tuple[str, int]] = set()
        self._last_shadow_sweep_ts: float = 0.0

    async def init(self) -> None:
        """Initialize strategy — restore B2 positions from DB with metadata.

        Paper_engine.open_positions only carries generic fields (avg_price,
        shares, size) — asset/strike/direction live in paper_trades.
        trade_details JSON. If we only restore from paper_engine we get
        asset="?" / strike=0 placeholders; any exit that fires before the
        next scan refreshes them sends broken Slack notifications like
        "B2 LIVE SELL — ? above $0". Query paper_trades directly to hydrate
        the full OpenPosition on init.
        """
        from arbo.utils.db import PaperTrade, get_session_factory
        import sqlalchemy as sa
        factory = get_session_factory()
        try:
            async with factory() as session:
                result = await session.execute(
                    sa.select(PaperTrade).where(
                        PaperTrade.strategy == STRATEGY_NAME,
                        PaperTrade.status == "open",
                        # Skip archived pre-reset rows so the strategy's
                        # in-memory state matches the clean-slate counters
                        # the rest of the system uses.
                        sa.or_(
                            PaperTrade.notes.is_(None),
                            ~PaperTrade.notes.ilike("%pre_reset%"),
                        ),
                    )
                )
                rows = list(result.scalars().all())
        except Exception as e:
            logger.warning("b2_restore_db_failed", error=str(e))
            rows = []

        for row in rows:
            td = row.trade_details or {}
            asset = str(td.get("asset") or "?")
            try:
                strike = float(td.get("strike") or 0)
            except (TypeError, ValueError):
                strike = 0.0
            entry_price = float(row.price or 0)
            size = float(row.size or 0)
            shares = int(size / entry_price) if entry_price > 0 else 0
            try:
                live_shares = int(td.get("live_entry_shares") or 0)
            except (TypeError, ValueError):
                live_shares = 0
            try:
                live_entry_price = float(td.get("live_entry_price") or 0)
            except (TypeError, ValueError):
                live_entry_price = 0.0
            self._open_positions[row.token_id] = OpenPosition(
                token_id=row.token_id,
                condition_id=row.market_condition_id or "",
                asset=asset,
                symbol=f"{asset}USDT" if asset and asset != "?" else "BTCUSDT",
                strike=strike,
                direction=str(td.get("direction") or "above"),
                market_type=str(td.get("market_type") or "daily_above"),
                entry_price=entry_price,
                entry_exchange_price=float(td.get("exchange_price") or 0),
                sigma_at_entry=float(td.get("sigma") or 0.02),
                hours_to_expiry_at_entry=float(td.get("hours_to_expiry") or 24),
                entry_time=(row.placed_at.timestamp() if row.placed_at else time.time()),
                shares=shares,
                live_shares=live_shares,
                live_entry_price=live_entry_price,
                live_fill_status=str(td.get("live_fill_status") or ""),
            )

        n_restored = len(self._open_positions)
        if n_restored:
            logger.info("b2_positions_restored", count=n_restored)

    async def close(self) -> None:
        """Clean up resources."""
        pass  # Binance WS lifecycle managed by main_rdh

    # ═══════════════════════════════════════════════════════════════════════
    # ENTRY: Poll Cycle
    # ═══════════════════════════════════════════════════════════════════════

    async def poll_cycle(self, markets: list[Any]) -> list[CryptoSignal]:
        """Main entry scan. Called every 60 seconds.

        1. Get exchange prices from Binance WS
        2. Compute volatility
        3. Scan markets → signals
        4. Quality gate filter
        5. Fetch CLOB orderbook (non-NegRisk)
        6. Kelly sizing + risk check
        7. Execute (paper or live)

        Returns:
            List of executed signals.
        """
        executed: list[CryptoSignal] = []

        # 1. Get exchange prices
        exchange_prices = {}
        if self._binance_ws:
            exchange_prices = self._binance_ws.get_prices()
            # Update vol estimator with latest prices
            for symbol, price in exchange_prices.items():
                self._vol_estimator.update(symbol, price, time.time())

        if not exchange_prices:
            logger.info("b2_no_exchange_prices")
            return []

        # 2. Scan markets
        signals = scan_crypto_markets(
            markets,
            exchange_prices,
            self._vol_estimator,
            current_time=time.time(),
        )

        if not signals:
            logger.info("b2_no_signals", crypto_markets=len(markets))
            # Still run shadow resolution sweep — events resolve regardless
            await self._sweep_shadow_resolutions()
            return []

        # 2b. Project PARALLEL: shadow-evaluate ALL variants on full
        # candidate set BEFORE champion's filter. Each variant's gates
        # decide pass/fail for the same candidate.
        try:
            await self._evaluate_shadow_variants(signals)
        except Exception as e:
            logger.debug("b2_shadow_eval_error", error=str(e))

        # 2c. Resolution sweep (best-effort, every 5 min)
        try:
            await self._sweep_shadow_resolutions()
        except Exception as e:
            logger.debug("b2_shadow_sweep_error", error=str(e))

        # 3. Quality gate (champion's filter — drives live execution)
        qualified = filter_signals(signals)

        logger.info(
            "b2_poll_summary",
            exchange_btc=exchange_prices.get("BTCUSDT", 0),
            crypto_markets=len(markets),
            signals=len(signals),
            qualified=len(qualified),
        )

        if not qualified:
            return []

        # 4. Process top signals
        now = time.time()
        attempted = 0
        skip_reasons: dict[str, int] = {}

        try:
          for sig in qualified[:50]:  # Try up to 50 to find viable entries past deep ITM/OTM
            if attempted >= MAX_TRADES_PER_SCAN:
                break

            token_id = sig.token_id

            # Skip if already have position
            if token_id in self._open_positions:
                skip_reasons["has_position"] = skip_reasons.get("has_position", 0) + 1
                continue

            # Skip if in pending exit
            if self._exit_manager_ref and token_id in self._exit_manager_ref._pending:
                skip_reasons["pending_exit"] = skip_reasons.get("pending_exit", 0) + 1
                continue

            # Cooldown check
            last_trade = self._last_trade_time.get(token_id, 0)
            if now - last_trade < MIN_HOURS_BETWEEN_TRADES * 3600:
                skip_reasons["cooldown"] = skip_reasons.get("cooldown", 0) + 1
                continue

            # 5. Fetch CLOB prices via /price endpoint (NOT /book)
            # Crypto markets use RFQM pricing — /book is empty but /price gives
            # executable prices with tight spreads (same mechanism as NegRisk weather)
            if not self._orderbook_provider:
                skip_reasons["no_ob_provider"] = skip_reasons.get("no_ob_provider", 0) + 1
                continue
            try:
                # Use neg_risk=True to force /price endpoint (works for ALL markets)
                ob_snap = await self._orderbook_provider.get_snapshot(
                    token_id, neg_risk=True
                )
            except Exception as e:
                skip_reasons["ob_error"] = skip_reasons.get("ob_error", 0) + 1
                continue

            if not ob_snap:
                skip_reasons["no_price"] = skip_reasons.get("no_price", 0) + 1
                continue

            # /price endpoint: best_bid=SELL price, best_ask=BUY price
            # For non-NegRisk: BUY might be < SELL (normal) or BUY > SELL (inverted)
            raw_bid = float(ob_snap.best_bid) if ob_snap.best_bid else 0
            raw_ask = float(ob_snap.best_ask) if ob_snap.best_ask else 0
            mid = float(ob_snap.midpoint) if ob_snap.midpoint else 0

            if mid <= 0.001 or mid >= 0.999:
                skip_reasons["no_price_data"] = skip_reasons.get("no_price_data", 0) + 1
                continue

            # Use midpoint — the /price endpoint gives tight spreads (~1c)
            spread = abs(raw_ask - raw_bid)
            spread_pct = spread / mid if mid > 0 else 1.0

            if spread_pct > 0.20:
                skip_reasons["wide_spread"] = skip_reasons.get("wide_spread", 0) + 1
                logger.debug("b2_wide_spread", mid=mid, bid=raw_bid, ask=raw_ask, spread_pct=f"{spread_pct:.2f}")
                continue

            # Entry price source — gated by PAPER_MATCH_LIVE.
            #
            # IMPORTANT: orderbook_provider stores Polymarket /price endpoint
            # semantically INVERTED vs a standard exchange orderbook:
            #   best_bid = get_price("SELL") = what a seller receives (LOW)
            #   best_ask = get_price("BUY")  = what a buyer pays     (HIGH?)
            # But empirically on non-NegRisk crypto markets (B2), the observed
            # values are the OPPOSITE: raw_bid is higher than raw_ask. Our
            # first 11 live fills confirmed: live taker BUY paid
            #   max(raw_bid, raw_ask) = the HIGH side,
            # while paper using raw_ask was filling at the LOW side — earning
            # ~\$0.02/share spread vs live (≈9.5% drag on deployed capital).
            #
            # Fix: don't rely on bid/ask labels. For a BUY leg the taker pays
            # whichever side is higher; for a SELL leg receives whichever is
            # lower. max/min is the invariant regardless of inversion.
            hi = max(raw_bid, raw_ask)
            lo = min(raw_bid, raw_ask)
            if PAPER_MATCH_LIVE:
                clob_price = hi if hi > 0.001 else mid
            else:
                clob_price = lo if lo > 0.001 else mid

            # 6. Revalidate with CLOB price
            from arbo.strategies.crypto_quality_gate import MIN_PRICE, MAX_PRICE, MIN_EDGE
            if clob_price < MIN_PRICE or clob_price > MAX_PRICE:
                skip_reasons["clob_price_range"] = skip_reasons.get("clob_price_range", 0) + 1
                continue

            # Gamma vs CLOB divergence check — large gap means stale/synthetic Gamma price
            gamma_clob_gap = abs(sig.market_price - clob_price)
            if gamma_clob_gap > 0.15:
                skip_reasons["gamma_clob_diverge"] = skip_reasons.get("gamma_clob_diverge", 0) + 1
                continue

            clob_edge = sig.model_prob - clob_price
            if clob_edge < MIN_EDGE:
                skip_reasons["clob_edge_low"] = skip_reasons.get("clob_edge_low", 0) + 1
                continue

            skip_reasons["reached_sizing"] = skip_reasons.get("reached_sizing", 0) + 1

            # 7. Kelly sizing
            prob = sig.model_prob
            price = clob_price
            if prob <= 0 or prob >= 1 or price <= 0 or price >= 1:
                continue

            odds = (1.0 / price) - 1.0
            if odds <= 0:
                continue

            kelly_raw = (prob * odds - (1 - prob)) / odds
            if kelly_raw <= 0:
                skip_reasons["kelly_neg"] = skip_reasons.get("kelly_neg", 0) + 1
                logger.info("b2_kelly_neg", asset=sig.asset, strike=float(sig.strike),
                            model_prob=f"{sig.model_prob:.4f}", clob_price=f"{clob_price:.4f}",
                            clob_edge=f"{clob_edge:.4f}", kelly=f"{kelly_raw:.4f}")
                continue
            kelly_raw = min(kelly_raw, KELLY_RAW_CAP)
            kelly_adj = kelly_raw * float(KELLY_FRACTION)

            # Get available capital from risk manager
            strat_state = self._risk_manager.get_strategy_state(STRATEGY_NAME)
            if strat_state is None:
                skip_reasons["no_strat_state"] = skip_reasons.get("no_strat_state", 0) + 1
                continue
            if strat_state.is_halted:
                skip_reasons["halted"] = skip_reasons.get("halted", 0) + 1
                continue
            available = float(strat_state.allocated - strat_state.deployed)
            if available <= 0:
                skip_reasons["no_capital"] = skip_reasons.get("no_capital", 0) + 1
                continue

            trade_size = round(float(available) * kelly_adj, 2)
            trade_size = min(trade_size, float(strat_state.allocated) * 0.10)  # Max 10% of allocation

            # Volume cap
            if sig.volume_24h > 0:
                trade_size = min(trade_size, sig.volume_24h * VOLUME_CAP_PCT)

            if trade_size < 1.0:
                skip_reasons["size_too_small"] = skip_reasons.get("size_too_small", 0) + 1
                continue

            # 8. Risk check
            from arbo.core.risk_manager import TradeRequest

            # Dual/live mode trades count against the LIVE slot budget.
            # Paper-only (execution_mode="paper") uses the paper budget.
            # When live ends up not filling in dual mode, mirror-cancel
            # currently skips the paper record — so this flag stays True
            # only for requests we genuinely expect to consume real capital.
            will_use_live_capital = self._execution_mode in ("dual", "live")

            trade_req = TradeRequest(
                market_id=sig.condition_id,
                token_id=token_id,
                side="BUY",
                price=Decimal(str(round(clob_price, 4))),
                size=Decimal(str(round(trade_size, 2))),  # Decimal for risk_manager
                layer=0,
                market_category="crypto",
                strategy=STRATEGY_NAME,
                is_live_capital=will_use_live_capital,
            )
            decision = self._risk_manager.pre_trade_check(trade_req)
            if not decision.approved:
                skip_reasons["risk_rejected"] = skip_reasons.get("risk_rejected", 0) + 1
                logger.info("b2_risk_rejected", reason=decision.reason, size=trade_size)
                continue

            actual_size = float(decision.adjusted_size or trade_size)

            # Mirror-cancel debounce — skip tokens we recently failed to fill.
            # Prevents cascade of paper trades when live keeps failing.
            now = time.time()
            if now - self._last_mirror_attempt.get(token_id, 0) < MIRROR_CANCEL_DEBOUNCE:
                skip_reasons["mirror_debounce"] = skip_reasons.get("mirror_debounce", 0) + 1
                continue

            # Execution mode branches:
            #   "paper"  — record paper only, no live call
            #   "live"   — fire live, abort if no fill (no paper record)
            #   "dual"   — fire live AND paper; if live fails, cancel paper
            mirror_live = (
                self._execution_mode == "dual"
                and PAPER_MATCH_LIVE
                and self._live_capital > 0
            )
            live_shares = 0
            live_fill_status = "skipped"
            live_entry_price = 0.0
            live_latency_ms = 0
            live_size = 0.0

            if self._execution_mode == "live" and self._live_executor:
                # Pure-live: single-sided execution, no paper record if live fails.
                # Taker fallback: B2 daily crypto markets often have thin maker
                # liquidity — maker-only resulted in 0% fill rate over 13 min.
                # Paying the ask (1-3c spread) is the cost of getting filled at
                # all on these markets.
                fill = await self._live_executor.buy(
                    token_id=token_id, price=clob_price, size_usdc=actual_size,
                    neg_risk=False, tick_size="0.01",
                    maker_timeout_s=self._live_entry_timeout_s,
                    fallback_to_taker=True,
                )
                if fill.status in ("filled", "partial") and fill.shares_filled > 0:
                    shares = fill.shares_filled
                    fill_price = float(fill.fill_price) if fill.fill_price else clob_price
                else:
                    continue
            elif mirror_live and self._live_executor:
                # Dual: size for live (from wallet), fire live first.
                # Kill-switch: stop live if cumulative daily PnL below limit.
                if self._live_daily_pnl <= -self._live_daily_loss_limit:
                    logger.info(
                        "b2_live_kill_switch",
                        daily_pnl=round(self._live_daily_pnl, 2),
                        limit=self._live_daily_loss_limit,
                    )
                    continue

                # Refresh wallet balance every 60s (shared wallet with B3)
                if now - self._live_capital_last_check > 60:
                    try:
                        bal = await self._live_executor.get_balance()
                        self._live_capital = bal if bal > 10 else self._live_capital_fallback
                        self._live_capital_last_check = now
                        logger.info(
                            "b2_live_balance",
                            balance=f"${self._live_capital:.2f}",
                            source="wallet" if bal > 10 else "fallback",
                        )
                    except Exception as e:
                        logger.warning("b2_live_balance_failed", error=str(e))

                # Live sizing: same % as paper, but on live wallet
                position_pct = actual_size / float(strat_state.allocated) if strat_state.allocated > 0 else 0.03
                live_size = min(self._live_capital * position_pct, 50.0)  # $50 hard cap
                # Polymarket 5-share minimum: shares = int(size / price); enforce
                wallet_floor = max(2.0, min(10.0, self._live_capital * 0.015))
                shares_floor = 5.0 * clob_price + 0.01
                live_min = max(wallet_floor, shares_floor)
                if live_size < live_min:
                    # Kelly sizing produced under the Polymarket 5-share
                    # minimum. If the shortfall is small enough to stay
                    # within a 5% per-trade wallet cap, boost up to the
                    # floor — missing the signal entirely over a few cents
                    # is worse than a mild sizing overshoot. Otherwise the
                    # signal is genuinely too small for this price level.
                    max_boost = self._live_capital * 0.05  # 5% hard cap per trade
                    if live_min <= max_boost:
                        logger.info(
                            "b2_live_size_boosted",
                            original=round(live_size, 2),
                            boosted_to=round(live_min, 2),
                            price=round(clob_price, 3),
                            cap_pct=5.0,
                        )
                        live_size = live_min
                    else:
                        logger.info(
                            "b2_live_size_too_small",
                            live_size=round(live_size, 2),
                            min_required=round(live_min, 2),
                            max_boost=round(max_boost, 2),
                            price=round(clob_price, 3),
                        )
                        self._last_mirror_attempt[token_id] = now
                        continue

                fill = await self._live_executor.buy(
                    token_id=token_id, price=clob_price, size_usdc=live_size,
                    neg_risk=False, tick_size="0.01",
                    maker_timeout_s=self._live_entry_timeout_s,
                    # See pure-live branch above for rationale on taker fallback.
                    fallback_to_taker=True,
                )
                live_fill_status = fill.status
                live_latency_ms = fill.latency_ms
                if fill.status in ("filled", "partial") and fill.shares_filled > 0:
                    live_shares = fill.shares_filled
                    live_entry_price = float(fill.fill_price) if fill.fill_price else clob_price
                    # Paper mirrors the ACTUAL fill price and shares
                    fill_price = live_entry_price
                    shares = live_shares
                    # actual_size = what live actually deployed
                    actual_size = live_shares * live_entry_price
                else:
                    # Live failed — don't create paper trade at all (avoids
                    # phantom record). Debounce re-entry on this token.
                    self._last_mirror_attempt[token_id] = now
                    logger.info(
                        "b2_mirror_live_skip",
                        token=token_id[:20],
                        status=fill.status,
                        reason="live no-fill → skip paper too",
                    )
                    continue
            else:
                # Pure paper
                shares = int(actual_size / clob_price)
                fill_price = clob_price

            # 10. Record trade in paper engine
            paper_trade = None
            if self._paper_engine:
                paper_trade = self._paper_engine.place_trade(
                    market_condition_id=sig.condition_id,
                    token_id=token_id,
                    side="BUY",
                    market_price=Decimal(str(round(clob_price, 4))),
                    model_prob=Decimal(str(round(sig.model_prob, 4))),
                    layer=0,
                    market_category="crypto",
                    fee_enabled=False,  # Maker entry = 0% fee; fee tracked separately
                    strategy=STRATEGY_NAME,
                    pre_computed_size=Decimal(str(round(actual_size, 2))),
                    clob_fill_price=Decimal(str(round(fill_price, 4))),
                    trade_details={
                        "variant_id": "champion_v1",  # Project PARALLEL — current live variant
                        "asset": sig.asset,
                        "strike": float(sig.strike),
                        "direction": sig.direction,
                        "exchange_price": sig.current_exchange_price,
                        "sigma": sig.sigma_hourly,
                        "hours_to_expiry": sig.hours_to_expiry,
                        "market_type": sig.market_type,
                        "clob_bid": raw_bid,
                        "clob_ask": raw_ask,
                        "clob_spread_pct": round(spread_pct * 100, 1),
                        # Flag so post-analysis can separate mirror-on vs
                        # mirror-off paper trades (pre- vs post-deploy).
                        "paper_match_live": PAPER_MATCH_LIVE,
                        # Live fill details (only populated in dual mode).
                        # live_shares > 0 distinguishes real-capital trades
                        # from paper-only ones in dashboards/analytics.
                        "mirror_live": mirror_live,
                        "live_fill_status": live_fill_status,
                        "live_entry_price": live_entry_price,
                        "live_entry_shares": live_shares,
                        "live_entry_latency_ms": live_latency_ms,
                        "live_size_usd": round(live_size, 2),
                        "live_capital": self._live_capital if mirror_live else 0,
                    },
                    # Route the paper-engine strategy_post_trade call to the
                    # correct pool. In dual mode the paper trade is a mirror
                    # of a real CLOB fill — it should occupy a LIVE slot, not
                    # steal from the paper budget.
                    is_live_capital=(live_shares > 0),
                )

            if paper_trade is None and self._paper_engine:
                skip_reasons["paper_rejected"] = skip_reasons.get("paper_rejected", 0) + 1
                continue

            # Track position
            self._open_positions[token_id] = OpenPosition(
                token_id=token_id,
                condition_id=sig.condition_id,
                asset=sig.asset,
                symbol=sig.symbol,
                strike=float(sig.strike),
                direction=sig.direction,
                market_type=sig.market_type,
                entry_price=fill_price,
                entry_exchange_price=sig.current_exchange_price,
                sigma_at_entry=sig.sigma_hourly,
                hours_to_expiry_at_entry=sig.hours_to_expiry,
                entry_time=time.time(),
                shares=shares,
                live_shares=live_shares,
                live_entry_price=live_entry_price,
                live_fill_status=live_fill_status,
            )
            self._last_trade_time[token_id] = time.time()

            logger.info(
                "b2_entry",
                asset=sig.asset,
                strike=float(sig.strike),
                direction=sig.direction,
                edge=f"{sig.edge:.3f}",
                price=f"{fill_price:.3f}",
                size=f"${actual_size:.2f}",
                exchange_price=f"${sig.current_exchange_price:.0f}",
                mode=self._execution_mode,
                live=live_fill_status if mirror_live else None,
                live_shares=live_shares if mirror_live else None,
            )

            executed.append(sig)
            attempted += 1

        except Exception as e:
            import traceback
            logger.error("b2_entry_loop_error", error=str(e), attempted=attempted,
                         tb=traceback.format_exc()[-300:])

        if skip_reasons or executed or qualified:
            logger.info(
                "b2_entry_summary",
                executed=len(executed),
                skip_reasons=skip_reasons,
                qualified=len(qualified),
            )

        return executed

    # ═══════════════════════════════════════════════════════════════════════
    # EXIT: Check Exits
    # ═══════════════════════════════════════════════════════════════════════

    async def check_exits(
        self, current_polymarket_prices: dict[str, float],
    ) -> list[tuple[str, str]]:
        """Check open positions for exit triggers.

        Called every 30 seconds. Recomputes probability with latest
        Binance exchange price.

        Args:
            current_polymarket_prices: token_id → current Polymarket YES price.

        Returns:
            List of (token_id, exit_reason) for triggered exits.
        """
        triggered: list[tuple[str, str]] = []

        # Get latest exchange prices
        exchange_prices = {}
        if self._binance_ws:
            exchange_prices = self._binance_ws.get_prices()

        for token_id, pos in list(self._open_positions.items()):
            current_poly_price = current_polymarket_prices.get(token_id)
            if current_poly_price is None:
                continue

            # 1. Profit take
            if current_poly_price >= pos.entry_price + PROFIT_TARGET_ABS:
                triggered.append((token_id, "profit_take"))
                continue

            # 2. Edge-based exit using latest exchange price
            exchange_price = exchange_prices.get(pos.symbol)
            if exchange_price is not None:
                # Recompute probability with latest exchange price
                # Time has passed since entry → hours_to_expiry decreased
                elapsed_hours = (time.time() - pos.entry_time) / 3600
                updated_hours = max(
                    pos.hours_to_expiry_at_entry - elapsed_hours, 0.01
                )

                sigma = self._vol_estimator.get_sigma(
                    pos.symbol, time.time()
                )

                updated_prob = estimate_crypto_prob(
                    current_price=exchange_price,
                    strike=pos.strike,
                    hours_to_expiry=updated_hours,
                    sigma_per_hour=sigma,
                    market_type=pos.market_type,
                    sigma_scale=SIGMA_SCALE,
                    direction=pos.direction,
                )

                updated_edge = updated_prob - current_poly_price

                if updated_edge < MIN_HOLD_EDGE:
                    triggered.append((token_id, "edge_lost"))
                    continue

        # Clean up triggered positions
        for tid, reason in triggered:
            self._open_positions.pop(tid, None)
            logger.info(
                "b2_exit_triggered",
                token=tid[:20],
                reason=reason,
            )

        return triggered

    def handle_resolution(self, condition_id: str, pnl: float) -> None:
        """Clean up position after market resolves."""
        to_remove = [
            tid for tid, pos in self._open_positions.items()
            if pos.condition_id == condition_id
        ]
        for tid in to_remove:
            self._open_positions.pop(tid, None)

    # ═══════════════════════════════════════════════════════════════════════
    # Project PARALLEL — Shadow variant evaluation + resolution sweep
    # ═══════════════════════════════════════════════════════════════════════

    def _get_variants(self) -> list:
        """Return active B2 variants, refreshed every 60s."""
        now = time.time()
        if now - self._variants_cache_ts > 60:
            try:
                from arbo.core.variant_pool import get_active_variants
                self._variants_cache = get_active_variants("B2")
                self._variants_cache_ts = now
            except Exception as e:
                logger.warning("b2_variant_load_error", error=str(e))
        return self._variants_cache

    async def _evaluate_shadow_variants(self, signals: list[CryptoSignal]) -> None:
        """Per-signal × per-variant evaluation → shadow_variant_signals.

        For each candidate signal, applies each variant's quality gate via
        `check_signal_quality(params=variant.params)`. Persists one row
        per (variant, candidate). Dedupes by (token_id, minute_bucket).
        """
        variants = self._get_variants()
        if not variants or not signals:
            return

        from datetime import datetime, timezone
        from arbo.utils.db import get_session_factory
        from arbo.strategies.crypto_quality_gate import check_signal_quality
        import sqlalchemy as sa
        import json

        now_ts = time.time()
        signal_dt = datetime.fromtimestamp(now_ts, tz=timezone.utc)
        minute_bucket = int(now_ts // 60)

        # Prune dedup set occasionally
        if len(self._shadow_written) > 5000:
            self._shadow_written.clear()

        rows: list[dict] = []
        for sig in signals:
            key = (sig.token_id, minute_bucket)
            if key in self._shadow_written:
                continue
            self._shadow_written.add(key)

            event_end_ts = (
                sig.expiry.timestamp() if sig.expiry else None
            )

            for v in variants:
                decision = check_signal_quality(
                    sig, exchange_price_age_s=0.0, params=v.params,
                )
                meta = {
                    "asset": sig.asset,
                    "strike": float(sig.strike),
                    "direction": sig.direction,
                    "market_type": sig.market_type,
                    "hours_to_expiry": sig.hours_to_expiry,
                }
                rows.append({
                    "strategy": "B2",
                    "variant_id": v.variant_id,
                    "condition_id": sig.condition_id,
                    "token_id": sig.token_id,
                    "signal_ts": signal_dt,
                    "qualified": decision.passed,
                    "skip_reason": (
                        decision.reason.split(":")[0]
                        if not decision.passed and decision.reason else None
                    ),
                    "direction": sig.direction,
                    "entry_price": float(sig.market_price),
                    "edge": float(sig.edge),
                    "sigma": float(sig.sigma_hourly),
                    "btc_at_start": float(sig.current_exchange_price),
                    "btc_now": float(sig.current_exchange_price),
                    "btc_move": 0.0,
                    "market_gap": None,
                    "velocity": None,
                    "dir_delta": None,
                    "would_fill_at": float(sig.market_price),
                    "event_start_ts": None,
                    "event_end_ts": event_end_ts,
                    "model_prob": float(sig.model_prob),
                    "meta_json": json.dumps(meta),
                })

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
                        logger.debug(
                            "b2_shadow_insert_error",
                            variant_id=r.get("variant_id"),
                            error=str(e),
                        )
                await session.commit()
        except Exception as e:
            logger.warning("b2_shadow_persist_error", error=str(e))

    async def _sweep_shadow_resolutions(self) -> None:
        """Resolve B2 shadow rows past event_end_ts via Polymarket Gamma.

        Throttled to every 5 minutes (B2 markets are daily/weekly — no
        need to poll faster).
        """
        now = time.time()
        if now - self._last_shadow_sweep_ts < 300:
            return
        self._last_shadow_sweep_ts = now

        from arbo.utils.db import get_session_factory
        import aiohttp
        import sqlalchemy as sa

        # Fetch unresolved (condition_id, event_end_ts) pairs
        try:
            factory = get_session_factory()
            async with factory() as session:
                result = await session.execute(
                    sa.text("""
                        SELECT DISTINCT condition_id, event_end_ts
                        FROM shadow_variant_signals
                        WHERE strategy = 'B2'
                          AND resolution_outcome IS NULL
                          AND event_end_ts IS NOT NULL
                          AND event_end_ts < :now - 300
                        LIMIT 30
                    """),
                    {"now": now},
                )
                pending = list(result.mappings())
        except Exception as e:
            logger.debug("b2_sweep_query_error", error=str(e))
            return

        if not pending:
            return

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
        ) as http:
            for row in pending:
                cid = row["condition_id"]
                up_won = await self._fetch_b2_resolution(http, cid)
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
                                WHERE strategy = 'B2'
                                  AND condition_id = :cid
                                  AND resolution_outcome IS NULL
                            """),
                            {"up_won": bool(up_won), "cid": cid},
                        )
                        # PnL only for qualified rows
                        # B2 direction='above' == YES side wins iff up_won
                        # B2 direction='below' == YES side wins iff NOT up_won
                        # Since we always BUY YES on the chosen direction,
                        # use direction string as proxy for which token bought
                        await session.execute(
                            sa.text("""
                                UPDATE shadow_variant_signals
                                SET would_pnl_per_share = CASE
                                    WHEN (direction = 'above' AND :up_won = true)
                                      OR (direction = 'below' AND :up_won = false)
                                    THEN 1.0 - would_fill_at
                                    ELSE -would_fill_at
                                END
                                WHERE strategy = 'B2'
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
                    logger.debug("b2_sweep_update_error", cid=cid, error=str(e))

    async def _fetch_b2_resolution(
        self, http: Any, condition_id: str
    ) -> bool | None:
        """Fetch resolution from Polymarket Gamma. Returns True if YES won."""
        try:
            async with http.get(
                "https://gamma-api.polymarket.com/markets",
                params={"condition_ids": condition_id, "closed": "true"},
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
            if not data:
                return None
            m = data[0] if isinstance(data, list) else data
            import json as _json
            outcomes = m.get("outcomes")
            prices = m.get("outcomePrices")
            if isinstance(outcomes, str):
                outcomes = _json.loads(outcomes)
            if isinstance(prices, str):
                prices = _json.loads(prices)
            if not outcomes or not prices:
                return None
            # YES outcome wins iff its price == "1"
            for o, p in zip(outcomes, prices):
                if str(o).lower() == "yes":
                    if str(p) == "1":
                        return True
                    elif str(p) == "0":
                        return False
                    return None
            return None
        except Exception:
            return None
