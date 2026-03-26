"""Strategy C2: EMOS + Edge Exit Fusion — Weather Markets.

Variant of Strategy C that combines:
  1. EMOS adaptive probability model (rolling_mae σ, ewma bias, window=21)
  2. Edge-based early exit (sell when EMOS says edge < 5%)
  3. Autoresearch-optimized quality gate (looser entry, smarter exit)

Key differences from C (C1f-ensemble):
  - EMOS probability: adaptive sigma+bias replaces fixed per-city values
  - Edge exit: sells position when updated edge drops below min_hold_edge
  - Looser entry: min_edge=0.03 (vs 0.10), catches more opportunities
  - Profit take: also exits at +$0.15 absolute price gain
  - 4 cities excluded: São Paulo, Tel Aviv, Tokyo, Lucknow
  - Per-city overrides: Dallas (loose), Miami (tight)

Autoresearch results (2026-03-25, 1401 trials):
  - In-sample: score=138.1, 1878 trades, WR=54.1%, $15,512, Sharpe 9.44
  - OOS (walk-forward 3 folds): score=121.0, $3,411 avg
  - vs baseline (edge-only, no EMOS): +16.0 score, +$10,229 PnL

Capital: $1,000 (separate from Strategy C)
Poll cycle: shared with C (same forecasts, same markets, own quality gate)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from arbo.connectors.orderbook_provider import (
    OrderbookProvider,
    available_depth,
    estimate_fill_price,
)
from arbo.connectors.weather_models import City, WeatherForecast
from arbo.connectors.weather_noaa import NOAAWeatherClient
from arbo.connectors.weather_openmeteo import OpenMeteoWeatherClient
from arbo.core.risk_manager import MAX_POSITION_PCT, RiskManager, TradeRequest
from arbo.strategies.weather_quality_gate_c2 import (
    CITY_OVERRIDES,
    EMOS_BIAS_METHOD,
    EMOS_EWMA_ALPHA,
    EMOS_SIGMA_FLOOR,
    EMOS_SIGMA_METHOD,
    EMOS_SIGMA_SCALE,
    EMOS_TRAINING_WINDOW,
    EXIT_SLIPPAGE_PCT,
    KELLY_RAW_CAP,
    MIN_HOLD_EDGE,
    PROB_EXIT_FLOOR,
    PROB_SHARPENING,
    PROFIT_TARGET_ABS,
    SHRINKAGE,
    check_signal_quality,
    filter_signals,
)
from arbo.strategies.weather_scanner import WeatherSignal, scan_weather_markets
from arbo.utils.logger import get_logger

logger = get_logger("strategy_c2")

STRATEGY_ID = "C2"
MAX_TRADES_PER_SCAN = 3
MIN_HOURS_TO_RESOLUTION = 6
LATENCY_RECHECK_DELAY_S = 2.0
VOLATILITY_GUARD_CENTS = Decimal("0.05")
MAX_VOLUME_POSITION_PCT = Decimal("0.05")


class StrategyC2:
    """EMOS + Edge Exit Fusion strategy for weather markets.

    Fully independent — fetches own forecasts, applies own quality gate,
    probability model, and exit logic. Does NOT depend on Strategy C.
    """

    def __init__(
        self,
        risk_manager: RiskManager,
        paper_engine: Any = None,
        metoffice_api_key: str = "",
        orderbook_provider: OrderbookProvider | None = None,
        execution_mode: str = "paper",
        live_executor: Any = None,
    ) -> None:
        self._risk = risk_manager
        self._paper_engine = paper_engine
        self._metoffice_key = metoffice_api_key
        self._orderbook = orderbook_provider
        self._execution_mode = execution_mode  # "paper" or "live"
        self._live_executor = live_executor

        # Own weather clients
        self._noaa: NOAAWeatherClient | None = None
        self._openmeteo: OpenMeteoWeatherClient | None = None

        # State
        self._forecasts: dict[City, WeatherForecast] = {}
        self._signals_generated: int = 0
        self._trades_placed: int = 0
        self._exits_triggered: int = 0
        self._drift_log: list[dict[str, Any]] = []

        # Open position tracking for exit checks
        # token_id → {entry_price, entry_edge, entry_prob, city, entry_ts}
        self._open_positions: dict[str, dict[str, Any]] = {}

    async def init(self) -> None:
        """Initialize C2 with own weather clients + restore open positions."""
        self._noaa = NOAAWeatherClient()
        self._openmeteo = OpenMeteoWeatherClient()

        # Restore open position tracking from paper engine (survives restarts)
        if self._paper_engine:
            for pos in self._paper_engine.open_positions:
                if getattr(pos, "strategy", "") != "C2":
                    continue
                td = self._paper_engine.get_trade_details(pos.token_id)
                if td:
                    self._open_positions[pos.token_id] = {
                        "entry_price": float(pos.avg_price),
                        "entry_edge": float(td.get("edge_at_scan", 0)),
                        "entry_prob": td.get("forecast_prob", 0),
                        "city": td.get("city", ""),
                        "entry_ts": str(pos.opened_at),
                        "signal": None,  # Can't restore full signal, but exit checks work without it
                    }

        logger.info(
            "strategy_c2_initialized",
            execution_mode=self._execution_mode,
            min_hold_edge=MIN_HOLD_EDGE,
            profit_target=PROFIT_TARGET_ABS,
            emos_window=EMOS_TRAINING_WINDOW,
            emos_sigma_method=EMOS_SIGMA_METHOD,
            independent=True,
            restored_positions=len(self._open_positions),
        )

    async def close(self) -> None:
        """Close weather clients."""
        if self._noaa:
            await self._noaa.close()
        if self._openmeteo:
            await self._openmeteo.close()

    async def fetch_forecasts(self) -> dict[City, WeatherForecast]:
        """Fetch forecasts from NOAA + Open-Meteo (independent from C).

        Adds 5s delay before Open-Meteo to avoid 429 rate limit collision
        with Strategy C's concurrent fetch.
        """
        import asyncio

        forecasts: dict[City, WeatherForecast] = {}

        if self._noaa:
            try:
                for f in await self._noaa.get_all_forecasts():
                    forecasts[f.city] = f
            except Exception as e:
                logger.error("c2_noaa_error", error=str(e))

        # Stagger Open-Meteo to avoid 429 collision with Strategy C
        await asyncio.sleep(5)

        if self._openmeteo:
            try:
                for f in await self._openmeteo.get_all_forecasts():
                    forecasts[f.city] = f
            except Exception as e:
                logger.error("c2_openmeteo_error", error=str(e))

        self._forecasts = forecasts
        logger.info("c2_forecasts_fetched", cities=len(forecasts))
        return forecasts

    @property
    def stats(self) -> dict[str, Any]:
        """Runtime stats for dashboard/status."""
        return {
            "signals_generated": self._signals_generated,
            "trades_placed": self._trades_placed,
            "exits_triggered": self._exits_triggered,
            "open_positions": len(self._open_positions),
        }

    async def poll_cycle(self, markets: list[Any]) -> list[WeatherSignal]:
        """Run one C2 poll cycle.

        Fetches own forecasts, scans markets, applies C2 quality gate.
        Fully independent from Strategy C.
        """
        import asyncio

        # 1. Fetch own forecasts
        forecasts = await self.fetch_forecasts()
        if not forecasts:
            logger.warning("c2_no_forecasts")
            return []

        # 2. Scan weather markets (same scanner, no ensemble_stds override)
        signals = scan_weather_markets(markets, forecasts)
        self._signals_generated += len(signals)
        if not signals:
            return []

        # 3. C2 quality gate
        fetch_times = [f.fetched_at for f in forecasts.values()]
        oldest_fetch = min(fetch_times) if fetch_times else datetime.now(timezone.utc)
        qualified = filter_signals(signals, oldest_fetch)

        if not qualified:
            logger.info("c2_no_signals_passed_gate", total_scanned=len(signals))
            return []

        # 4. Strategy state check
        strategy_state = self._risk.get_strategy_state(STRATEGY_ID)
        if strategy_state is None or strategy_state.is_halted:
            logger.warning("c2_halted_or_missing")
            return []

        available_capital = strategy_state.available

        # 5. Time filter
        now = datetime.now(timezone.utc)
        time_filtered = []
        for sig in qualified:
            td = sig.market.target_date
            resolution_approx = datetime(td.year, td.month, td.day, tzinfo=timezone.utc)
            hours_to_resolution = (resolution_approx - now).total_seconds() / 3600
            if hours_to_resolution < MIN_HOURS_TO_RESOLUTION:
                continue
            time_filtered.append(sig)

        if not time_filtered:
            return []

        # 6. Sort by edge descending, take top signals
        time_filtered.sort(key=lambda s: -s.edge)

        # 7. Fetch CLOB prices
        token_ids = []
        for sig in time_filtered:
            tid = (
                sig.market.token_id_yes
                if sig.direction == "BUY_YES"
                else sig.market.token_id_no
            )
            token_ids.append(tid)

        ob_snapshots: dict[str, Any] = {}
        if self._orderbook and token_ids:
            ob_snapshots = await self._orderbook.get_snapshots_batch(
                token_ids, neg_risk=True
            )

        # 8. Execute
        traded_signals = []
        trades_this_scan = 0
        max_pos = strategy_state.allocated * MAX_POSITION_PCT

        for sig in time_filtered:
            if trades_this_scan >= MAX_TRADES_PER_SCAN:
                break

            token_id = (
                sig.market.token_id_yes
                if sig.direction == "BUY_YES"
                else sig.market.token_id_no
            )

            # Already have open position for this token? Check paper engine (source of truth)
            if self._paper_engine and token_id in {
                p.token_id for p in self._paper_engine.open_positions
                if getattr(p, "strategy", "") == "C2"
            }:
                continue

            # Don't buy token that's being exited (ExitManager pending)
            if hasattr(self, "_exit_manager_ref") and self._exit_manager_ref:
                if token_id in self._exit_manager_ref.pending_exits:
                    continue

            gamma_price = Decimal(str(sig.market.market_price))
            execution_price = gamma_price
            clob_fill: Decimal | None = None

            city = sig.market.city.value if sig.market.city else ""
            ob_snap = ob_snapshots.get(token_id)
            if ob_snap is not None:
                depth = available_depth(ob_snap, "BUY")
                if depth < Decimal("1"):
                    logger.debug("c2_skip_no_depth", city=city, depth=str(depth))
                    continue

                p1 = estimate_fill_price(ob_snap, "BUY", max_pos)
                if p1 is not None:
                    expected_price = (
                        gamma_price
                        if sig.direction == "BUY_YES"
                        else Decimal("1") - gamma_price
                    )
                    if abs(float(p1 - expected_price)) > 0.15:
                        logger.debug("c2_skip_fill_anomaly", city=city, p1=str(p1), expected=str(expected_price))
                        continue

                    # Latency recheck
                    await asyncio.sleep(LATENCY_RECHECK_DELAY_S)
                    self._orderbook.invalidate(token_id)
                    ob_snap_2 = await self._orderbook.get_snapshot(
                        token_id, neg_risk=True
                    )
                    p2 = (
                        estimate_fill_price(ob_snap_2, "BUY", max_pos)
                        if ob_snap_2 is not None
                        else None
                    )

                    clob_fill = p2 if p2 is not None else p1
                    execution_price = clob_fill

                    # Volatility guard
                    if p2 is not None and abs(p2 - p1) > VOLATILITY_GUARD_CENTS:
                        logger.debug("c2_skip_volatility", city=city, p1=str(p1), p2=str(p2))
                        continue

                    # Revalidate edge with CLOB price
                    clob_edge = float(
                        Decimal(str(sig.forecast_probability)) - clob_fill
                    )
                    if clob_edge <= 0:
                        logger.debug("c2_skip_no_clob_edge", city=city, clob_fill=str(clob_fill), prob=round(sig.forecast_probability, 4))
                        continue

            # Kelly sizing (quarter-Kelly with C2 params)
            city = sig.market.city.value if sig.market.city else ""
            city_kelly_cap = CITY_OVERRIDES.get(city, {}).get("kelly_raw_cap", KELLY_RAW_CAP)
            edge = float(sig.edge)
            price = float(execution_price)

            if price <= 0 or price >= 1 or edge <= 0:
                continue
            prob = price + edge
            odds = (1.0 / price) - 1.0
            kelly_raw = (prob * odds - (1 - prob)) / odds if odds > 0 else 0
            kelly_raw = min(max(kelly_raw, 0), city_kelly_cap)
            kelly_adj = Decimal(str(round(kelly_raw * 0.25, 6)))
            trade_size = min(
                available_capital * kelly_adj,
                strategy_state.allocated * MAX_POSITION_PCT,
            )

            # Volume cap
            volume_cap = Decimal(str(sig.market.volume_24h)) * MAX_VOLUME_POSITION_PCT
            if volume_cap > 0:
                trade_size = min(trade_size, volume_cap)

            if trade_size < Decimal("1"):
                continue

            # Risk check
            trade_req = TradeRequest(
                market_id=sig.market.condition_id,
                token_id=token_id,
                side="BUY",
                price=execution_price,
                size=trade_size,
                layer=0,
                market_category="weather",
                strategy=STRATEGY_ID,
            )
            decision = self._risk.pre_trade_check(trade_req)
            if not decision.approved:
                continue

            # Execute trade (paper or live)
            trade_details = {
                "mode": self._execution_mode,
                "city": city,
                "target_date": str(sig.market.target_date),
                "bucket_text": sig.market.question[:120],
                "forecast_temp_c": sig.forecast_temp_c,
                "forecast_prob": sig.forecast_probability,
                "market_price_gamma": float(gamma_price),
                "clob_fill_p1": float(clob_fill) if clob_fill else None,
                "direction": sig.direction,
                "edge_at_scan": round(sig.edge, 4),
                "volume_24h": sig.market.volume_24h,
                "liquidity": sig.market.liquidity,
                "model": "emos_exit_fusion",
                "min_hold_edge": MIN_HOLD_EDGE,
                "profit_target": PROFIT_TARGET_ABS,
                "neg_risk": sig.market.neg_risk,
            }

            executed = False
            live_fill = None

            if self._execution_mode == "live" and self._live_executor:
                # LIVE: submit real CLOB order
                fill = await self._live_executor.buy(
                    token_id=token_id,
                    price=float(execution_price),
                    size_usdc=float(trade_size),
                    neg_risk=sig.market.neg_risk,
                )
                # Merge full monitoring data into trade_details
                trade_details.update(fill.to_monitoring_dict())
                trade_details["live_size_usdc"] = float(trade_size)
                trade_details["live_submitted_shares"] = round(float(trade_size) / float(execution_price), 2) if float(execution_price) > 0 else 0
                # Orderbook state at entry (for spread analysis)
                if ob_snap is not None:
                    trade_details["entry_ob_bid_depth"] = float(available_depth(ob_snap, "SELL"))
                    trade_details["entry_ob_ask_depth"] = float(available_depth(ob_snap, "BUY"))
                    if ob_snap.best_bid and ob_snap.best_ask:
                        trade_details["entry_spread"] = float(ob_snap.best_ask - ob_snap.best_bid)
                        trade_details["entry_spread_pct"] = round(float(ob_snap.best_ask - ob_snap.best_bid) / float(ob_snap.best_ask) * 100, 2) if float(ob_snap.best_ask) > 0 else 0

                if fill.status in ("filled", "partial") and fill.shares_filled > 0:
                    executed = True
                    # Record actual filled amount (not requested)
                    actual_size = Decimal(str(fill.usdc_spent)) if fill.usdc_spent > 0 else trade_size
                    if self._paper_engine:
                        self._paper_engine.place_trade(
                            market_condition_id=sig.market.condition_id,
                            token_id=token_id,
                            side="BUY",
                            market_price=gamma_price,
                            model_prob=Decimal(str(sig.forecast_probability)),
                            layer=0,
                            market_category="weather",
                            strategy=STRATEGY_ID,
                            pre_computed_size=actual_size,
                            volume_24h=Decimal(str(sig.market.volume_24h)),
                            liquidity=Decimal(str(sig.market.liquidity)),
                            clob_fill_price=fill.fill_price or clob_fill,
                            trade_details=trade_details,
                        )
                    live_fill = fill
                    if fill.status == "partial":
                        logger.info(
                            "c2_live_buy_partial",
                            city=city,
                            filled=fill.shares_filled,
                            requested=fill.shares_requested,
                        )
                else:
                    logger.warning(
                        "c2_live_buy_failed",
                        city=city,
                        status=fill.status,
                        error=fill.error,
                    )

            elif self._paper_engine:
                # PAPER: simulate trade
                trade_result = self._paper_engine.place_trade(
                    market_condition_id=sig.market.condition_id,
                    token_id=token_id,
                    side="BUY",
                    market_price=gamma_price,
                    model_prob=Decimal(str(sig.forecast_probability)),
                    layer=0,
                    market_category="weather",
                    strategy=STRATEGY_ID,
                    pre_computed_size=trade_size,
                    volume_24h=Decimal(str(sig.market.volume_24h)),
                    liquidity=Decimal(str(sig.market.liquidity)),
                    clob_fill_price=clob_fill,
                    trade_details=trade_details,
                )
                executed = bool(trade_result)

            if executed:
                self._risk.post_trade_update(
                    sig.market.condition_id, "weather", trade_size
                )
                self._risk.strategy_post_trade(STRATEGY_ID, trade_size)
                traded_signals.append(sig)
                self._trades_placed += 1
                trades_this_scan += 1

                # Track for exit checks
                fill_price = float(live_fill.fill_price) if live_fill and live_fill.fill_price else float(execution_price)
                self._open_positions[token_id] = {
                    "entry_price": fill_price,
                    "entry_edge": edge,
                    "entry_prob": sig.forecast_probability,
                    "city": city,
                    "entry_ts": datetime.now(timezone.utc).isoformat(),
                    "signal": sig,
                    "shares": float(trade_size) / fill_price if fill_price > 0 else 0,
                    "neg_risk": sig.market.neg_risk,
                }

                logger.info(
                    "c2_trade_placed",
                    mode=self._execution_mode,
                    city=city,
                    direction=sig.direction,
                    size=str(trade_size),
                    edge=round(sig.edge, 4),
                    fill=str(live_fill.fill_price) if live_fill else str(clob_fill) if clob_fill else "synthetic",
                )

        logger.info(
            "c2_poll_complete",
            signals=len(signals),
            qualified=len(qualified),
            trades=len(traded_signals),
        )
        return traded_signals

    async def check_exits(
        self,
        current_prices: dict[str, Decimal],
        forecasts: dict[City, WeatherForecast],
    ) -> list[tuple[str, str]]:
        """Check open C2 positions for edge-based exit triggers.

        Called each poll cycle. Computes updated probability using latest
        forecast and checks if edge has dropped below MIN_HOLD_EDGE.

        Returns list of (token_id, exit_reason) tuples.
        """
        from arbo.strategies.weather_scanner import estimate_bucket_probability

        triggered: list[tuple[str, str]] = []

        # Sync: remove from _open_positions any token that paper engine no longer has
        if self._paper_engine:
            live_tokens = {
                p.token_id for p in self._paper_engine.open_positions
                if getattr(p, "strategy", "") == "C2"
            }
            stale = [tid for tid in self._open_positions if tid not in live_tokens]
            for tid in stale:
                logger.info("c2_position_sync_removed", token_id=tid[:20])
                del self._open_positions[tid]

        if not self._open_positions:
            return triggered

        logger.info(
            "c2_exit_check_start",
            n_positions=len(self._open_positions),
            n_prices=len(current_prices),
        )

        for token_id, pos_data in list(self._open_positions.items()):
            current_price = current_prices.get(token_id)
            if current_price is None:
                continue

            entry_price = pos_data["entry_price"]
            city = pos_data["city"]
            signal: WeatherSignal | None = pos_data.get("signal")
            price_f = float(current_price)

            # Profit take: exit if price rose enough (no signal needed)
            if price_f >= entry_price + PROFIT_TARGET_ABS:
                logger.info(
                    "c2_exit_profit_take",
                    city=city,
                    token_id=token_id[:20],
                    entry=round(entry_price, 4),
                    current=round(price_f, 4),
                    gain=round(price_f - entry_price, 4),
                )
                triggered.append((token_id, "profit_take"))
                self._exits_triggered += 1
                continue

            # Edge-based exit: recompute probability with latest forecast
            # (requires signal with market info; skip if restored from DB without signal)
            city_enum = signal.market.city if signal else None
            if city_enum and city_enum in forecasts:
                forecast = forecasts[city_enum]
                daily = forecast.get_forecast_for_date(signal.market.target_date) if signal else None
                bucket = getattr(signal.market, "bucket", None) if signal else None
                if daily is not None and bucket is not None:
                    updated_prob = estimate_bucket_probability(
                        daily,
                        bucket,
                        is_high=signal.market.is_high_temp,
                        city=city,
                        days_out=0,
                    )
                    updated_edge = updated_prob - price_f

                    if updated_edge < MIN_HOLD_EDGE:
                        logger.info(
                            "c2_exit_edge_lost",
                            city=city,
                            token_id=token_id[:20],
                            entry_edge=round(pos_data["entry_edge"], 4),
                            updated_edge=round(updated_edge, 4),
                            min_hold_edge=MIN_HOLD_EDGE,
                        )
                        triggered.append((token_id, "edge_lost"))
                        self._exits_triggered += 1
                        continue

            # Probability floor
            if PROB_EXIT_FLOOR > 0:
                if price_f < PROB_EXIT_FLOOR:
                    logger.info(
                        "c2_exit_prob_floor",
                        city=city,
                        token_id=token_id[:20],
                        price=round(price_f, 4),
                    )
                    triggered.append((token_id, "prob_floor"))
                    self._exits_triggered += 1

        # Clean up tracked positions for triggered exits
        for tid, _ in triggered:
            self._open_positions.pop(tid, None)

        return triggered

    def handle_resolution(self, condition_id: str, pnl: Decimal) -> None:
        """Handle market resolution — clean up tracked position.

        Sync is also done in check_exits() via paper engine source of truth,
        but explicit cleanup here catches resolution immediately.
        """
        to_remove = []
        for tid, data in self._open_positions.items():
            sig = data.get("signal")
            if sig and sig.market.condition_id == condition_id:
                to_remove.append(tid)
            # Also check paper engine — if token is no longer open, remove
            elif self._paper_engine:
                still_open = any(
                    p.token_id == tid
                    for p in self._paper_engine.open_positions
                    if getattr(p, "strategy", "") == "C2"
                )
                if not still_open:
                    to_remove.append(tid)
        for tid in to_remove:
            self._open_positions.pop(tid, None)
