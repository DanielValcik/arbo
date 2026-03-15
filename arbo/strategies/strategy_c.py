"""Strategy C: Compound Weather Resolution Chaining.

Main strategy module that orchestrates:
1. Weather forecast fetching (NOAA, Met Office, Open-Meteo)
2. Weather market scanning (Gamma API)
3. Quality gate filtering
4. Temperature laddering
5. Resolution chain management
6. Paper trading integration
7. Data collection (scan logs, volume snapshots)

Poll cycle: fetch forecasts → scan markets → quality gate → ladder → execute
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

from arbo.connectors.orderbook_provider import (
    OrderbookProvider,
    available_depth,
    estimate_fill_price,
)
from arbo.connectors.weather_metoffice import MetOfficeWeatherClient
from arbo.connectors.weather_models import City, WeatherForecast
from arbo.connectors.weather_noaa import NOAAWeatherClient
from arbo.connectors.weather_openmeteo import OpenMeteoWeatherClient
from arbo.core.risk_manager import MAX_POSITION_PCT, RiskManager, TradeRequest
from arbo.strategies.resolution_chain import ResolutionChainEngine
from arbo.strategies.weather_ladder import build_ladders_by_city
from arbo.strategies.weather_quality_gate import (
    check_signal_quality,
    filter_signals,
)
from arbo.strategies.weather_scanner import WeatherSignal, scan_weather_markets
from arbo.utils.logger import get_logger

logger = get_logger("strategy_c")

STRATEGY_ID = "C"
DEFAULT_SCAN_INTERVAL_S = 1800  # 30 minutes
MAX_VOLUME_POSITION_PCT = Decimal("0.05")  # Max 5% of 24h volume per position
LATENCY_RECHECK_DELAY_S = 2.0  # Simulate order signing + submission latency
VOLATILITY_GUARD_CENTS = Decimal("0.05")  # Skip if price moves >5 cents in 2s
MAX_DRIFT_LOG_SIZE = 1000  # Rolling window for drift analytics


class StrategyC:
    """Compound Weather Resolution Chaining strategy.

    Lifecycle:
        1. init() → create weather clients
        2. poll_cycle() → fetch, scan, filter, ladder, trade (called periodically)
        3. handle_resolution() → advance chain on market settlement
    """

    def __init__(
        self,
        risk_manager: RiskManager,
        paper_engine: Any = None,
        metoffice_api_key: str = "",
        orderbook_provider: OrderbookProvider | None = None,
    ) -> None:
        self._risk = risk_manager
        self._paper_engine = paper_engine
        self._metoffice_key = metoffice_api_key
        self._orderbook = orderbook_provider

        # Weather clients
        self._noaa: NOAAWeatherClient | None = None
        self._metoffice: MetOfficeWeatherClient | None = None
        self._openmeteo: OpenMeteoWeatherClient | None = None

        # State
        self._chain_engine = ResolutionChainEngine()
        self._forecasts: dict[City, WeatherForecast] = {}
        self._last_scan: datetime | None = None
        self._signals_generated: int = 0
        self._trades_placed: int = 0

        # Price drift tracking (latency re-check analytics)
        self._drift_log: list[dict[str, Any]] = []
        self._trades_skipped_volatility: int = 0

    async def init(self) -> None:
        """Initialize weather clients."""
        self._noaa = NOAAWeatherClient()
        if self._metoffice_key:
            self._metoffice = MetOfficeWeatherClient(api_key=self._metoffice_key)
        self._openmeteo = OpenMeteoWeatherClient()
        logger.info(
            "strategy_c_initialized",
            has_metoffice=bool(self._metoffice_key),
        )

    async def close(self) -> None:
        """Close weather clients."""
        if self._noaa:
            await self._noaa.close()
        if self._metoffice:
            await self._metoffice.close()
        if self._openmeteo:
            await self._openmeteo.close()

    async def fetch_forecasts(self) -> dict[City, WeatherForecast]:
        """Fetch forecasts from all configured weather sources."""
        forecasts: dict[City, WeatherForecast] = {}

        # NOAA: NYC, Chicago
        if self._noaa:
            try:
                noaa_forecasts = await self._noaa.get_all_forecasts()
                for f in noaa_forecasts:
                    forecasts[f.city] = f
            except Exception as e:
                logger.error("noaa_fetch_error", error=str(e))

        # Met Office: London
        if self._metoffice:
            try:
                london = await self._metoffice.get_forecast()
                forecasts[london.city] = london
            except Exception as e:
                logger.error("metoffice_fetch_error", error=str(e))

        # Open-Meteo: Seoul, Buenos Aires
        if self._openmeteo:
            try:
                om_forecasts = await self._openmeteo.get_all_forecasts()
                for f in om_forecasts:
                    forecasts[f.city] = f
            except Exception as e:
                logger.error("openmeteo_fetch_error", error=str(e))

        self._forecasts = forecasts
        logger.info("forecasts_fetched", cities=len(forecasts))
        return forecasts

    async def poll_cycle(self, markets: list[Any]) -> list[WeatherSignal]:
        """Run one poll cycle of Strategy C.

        Args:
            markets: List of GammaMarket objects from market discovery.

        Returns:
            List of signals that were traded (or would be traded in paper mode).
        """
        # 1. Fetch forecasts
        forecasts = await self.fetch_forecasts()
        if not forecasts:
            logger.warning("no_forecasts_available")
            return []

        # 2. Scan weather markets
        signals = scan_weather_markets(markets, forecasts)
        self._signals_generated += len(signals)
        if not signals:
            logger.info("no_weather_signals")
            return []

        # 3. Quality gate
        # Use the earliest fetch time as freshness reference
        fetch_times = [f.fetched_at for f in forecasts.values()]
        oldest_fetch = min(fetch_times) if fetch_times else datetime.now(timezone.utc)
        qualified = filter_signals(signals, oldest_fetch)

        if not qualified:
            logger.info("no_signals_passed_quality_gate")
            return []

        # 4. Check strategy state
        strategy_state = self._risk.get_strategy_state(STRATEGY_ID)
        if strategy_state is None or strategy_state.is_halted:
            logger.warning("strategy_c_halted_or_missing")
            return []

        available_capital = strategy_state.available

        # 5. Build temperature ladders
        max_pos = self._risk._state.capital * MAX_POSITION_PCT
        ladders = build_ladders_by_city(qualified, available_capital, max_position_size=max_pos)

        # 6. Fetch live CLOB prices for all tokens we might trade
        token_ids = []
        for ladder in ladders:
            for position in ladder.positions:
                sig = position.signal
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

        # 7. Execute via paper engine (or live in the future)
        traded_signals = []
        for ladder in ladders:
            for position in ladder.positions:
                signal = position.signal

                # Volume-based position cap: max 5% of 24h volume
                volume_cap = Decimal(str(signal.market.volume_24h)) * MAX_VOLUME_POSITION_PCT
                trade_size = min(position.size_usdc, volume_cap)
                if trade_size < Decimal("1"):
                    logger.info(
                        "trade_skipped_volume_cap",
                        city=signal.market.city.value,
                        volume_24h=signal.market.volume_24h,
                        volume_cap=str(volume_cap),
                        kelly_size=str(position.size_usdc),
                    )
                    continue

                token_id = (
                    signal.market.token_id_yes
                    if signal.direction == "BUY_YES"
                    else signal.market.token_id_no
                )

                # CLOB live price: use real NegRisk price if available
                gamma_price = Decimal(str(signal.market.market_price))
                execution_price = gamma_price
                clob_fill: Decimal | None = None

                ob_snap = ob_snapshots.get(token_id)
                if ob_snap is not None:
                    # Check available depth
                    depth = available_depth(ob_snap, "BUY")
                    if depth < Decimal("1"):
                        logger.info(
                            "trade_skipped_no_depth",
                            city=signal.market.city.value,
                            token_id=token_id[:20],
                            depth=str(depth),
                        )
                        continue
                    if trade_size > depth:
                        logger.info(
                            "trade_size_capped_by_depth",
                            city=signal.market.city.value,
                            original=str(trade_size),
                            depth=str(depth),
                        )
                        trade_size = depth

                    # Get initial fill price from CLOB (P1)
                    p1 = estimate_fill_price(ob_snap, "BUY", trade_size)
                    if p1 is not None:
                        # For BUY_NO, expected NO price = 1 - gamma_yes
                        expected_price = (
                            gamma_price
                            if signal.direction == "BUY_YES"
                            else Decimal("1") - gamma_price
                        )
                        gamma_delta = float(p1 - expected_price)

                        # Sanity check: CLOB fill too far from Gamma expected
                        if abs(gamma_delta) > 0.15:
                            logger.warning(
                                "clob_fill_anomaly",
                                city=signal.market.city.value,
                                direction=signal.direction,
                                expected=str(expected_price),
                                clob_fill=str(p1),
                                delta=round(gamma_delta, 4),
                            )
                            continue

                        # --- Latency re-check: simulate order signing delay ---
                        await asyncio.sleep(LATENCY_RECHECK_DELAY_S)
                        self._orderbook.invalidate(token_id)
                        ob_snap_2 = await self._orderbook.get_snapshot(
                            token_id, neg_risk=True
                        )
                        p2 = (
                            estimate_fill_price(ob_snap_2, "BUY", trade_size)
                            if ob_snap_2 is not None
                            else None
                        )

                        # Track drift for analytics
                        drift = float(p2 - p1) if p2 is not None else 0.0
                        drift_entry = {
                            "p1": float(p1),
                            "p2": float(p2) if p2 else None,
                            "delta": round(drift, 6),
                            "city": signal.market.city.value,
                            "direction": signal.direction,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        self._drift_log.append(drift_entry)
                        if len(self._drift_log) > MAX_DRIFT_LOG_SIZE:
                            self._drift_log = self._drift_log[-MAX_DRIFT_LOG_SIZE:]

                        # Use P2 as fill price (more realistic)
                        clob_fill = p2 if p2 is not None else p1
                        execution_price = clob_fill

                        # Volatility guard: skip if price moved >5 cents
                        if p2 is not None and abs(p2 - p1) > VOLATILITY_GUARD_CENTS:
                            self._trades_skipped_volatility += 1
                            logger.warning(
                                "trade_skipped_volatility",
                                city=signal.market.city.value,
                                p1=str(p1),
                                p2=str(p2),
                                drift=round(drift, 4),
                            )
                            continue

                        logger.info(
                            "clob_latency_recheck",
                            city=signal.market.city.value,
                            direction=signal.direction,
                            gamma_yes=str(gamma_price),
                            expected=str(expected_price),
                            p1=str(p1),
                            p2=str(clob_fill),
                            drift=round(drift, 4),
                            gamma_delta=round(gamma_delta, 4),
                        )

                        # Revalidate edge with real CLOB price (P2)
                        clob_edge = float(
                            Decimal(str(signal.forecast_probability)) - clob_fill
                        )
                        if clob_edge <= 0:
                            logger.info(
                                "trade_skipped_no_clob_edge",
                                city=signal.market.city.value,
                                gamma_edge=round(signal.edge, 4),
                                clob_fill=str(clob_fill),
                                clob_edge=round(clob_edge, 4),
                            )
                            continue

                # Build trade request for risk manager
                trade_req = TradeRequest(
                    market_id=signal.market.condition_id,
                    token_id=token_id,
                    side="BUY",
                    price=execution_price,
                    size=trade_size,
                    layer=0,  # Strategy C uses strategy field, not layer
                    market_category="weather",
                    strategy=STRATEGY_ID,
                )

                # Risk check
                decision = self._risk.pre_trade_check(trade_req)
                if not decision.approved:
                    logger.info(
                        "trade_rejected_by_risk",
                        reason=decision.reason,
                        market_id=signal.market.condition_id,
                    )
                    continue

                # Execute paper trade
                if self._paper_engine:
                    # Build trade_details for comprehensive data capture
                    trade_details = {
                        "city": signal.market.city.value,
                        "target_date": str(signal.market.target_date),
                        "bucket_text": signal.market.question[:120],
                        "forecast_temp_c": signal.forecast_temp_c,
                        "forecast_prob": signal.forecast_probability,
                        "market_price_gamma": float(gamma_price),
                        "clob_fill_p1": float(clob_fill) if clob_fill else None,
                        "direction": signal.direction,
                        "edge_at_scan": round(signal.edge, 4),
                        "volume_24h": signal.market.volume_24h,
                        "liquidity": signal.market.liquidity,
                        "confidence": signal.confidence,
                    }

                    trade_result = self._paper_engine.place_trade(
                        market_condition_id=signal.market.condition_id,
                        token_id=token_id,
                        side="BUY",
                        market_price=gamma_price,
                        model_prob=Decimal(str(signal.forecast_probability)),
                        layer=0,
                        market_category="weather",
                        strategy=STRATEGY_ID,
                        pre_computed_size=trade_size,
                        volume_24h=Decimal(str(signal.market.volume_24h)),
                        liquidity=Decimal(str(signal.market.liquidity)),
                        clob_fill_price=clob_fill,
                        trade_details=trade_details,
                    )
                    if trade_result:
                        self._risk.post_trade_update(
                            signal.market.condition_id, "weather", trade_size
                        )
                        self._risk.strategy_post_trade(STRATEGY_ID, trade_size)
                        traded_signals.append(signal)
                        self._trades_placed += 1

                        logger.info(
                            "paper_trade_placed",
                            city=signal.market.city.value,
                            direction=signal.direction,
                            size=str(trade_size),
                            kelly_size=str(position.size_usdc),
                            volume_cap=str(volume_cap),
                            edge=round(signal.edge, 4),
                            clob_fill=str(clob_fill) if clob_fill else "synthetic",
                        )

        self._last_scan = datetime.now(timezone.utc)
        logger.info(
            "poll_cycle_complete",
            signals_total=len(signals),
            qualified=len(qualified),
            ladders=len(ladders),
            trades=len(traded_signals),
        )

        # Data collection (best-effort, non-blocking)
        traded_ids = {s.market.condition_id for s in traded_signals}
        traded_sizes = {
            s.market.condition_id: float(
                min(
                    Decimal(str(s.market.volume_24h)) * MAX_VOLUME_POSITION_PCT,
                    self._risk._state.capital * MAX_POSITION_PCT,
                )
            )
            for s in traded_signals
        }
        try:
            await self.save_scan_log(
                signals, qualified, traded_ids, traded_sizes, oldest_fetch
            )
        except Exception as e:
            logger.debug("scan_log_async_error", error=str(e))
        try:
            await self.save_volume_snapshot(markets)
        except Exception as e:
            logger.debug("volume_snapshot_async_error", error=str(e))

        return traded_signals

    def handle_resolution(
        self,
        market_id: str,
        pnl: Decimal,
    ) -> City | None:
        """Handle a market resolution within a chain.

        Looks up the chain by active market_id internally.

        Returns the next city to deploy to, or None if chain is done / no chain.
        """
        # Update risk manager
        self._risk.post_trade_update(
            market_id, "weather", abs(pnl), pnl=pnl
        )
        self._risk.strategy_post_trade(STRATEGY_ID, abs(pnl), pnl=pnl)

        # Find and advance chain
        chain_id = self._chain_engine.find_chain_by_market_id(market_id)
        if chain_id is None:
            logger.info("resolution_no_chain", market_id=market_id)
            return None

        return self._chain_engine.resolve(chain_id, market_id, pnl)

    def get_drift_log(self) -> list[dict[str, Any]]:
        """Return drift log for dashboard analytics."""
        return list(self._drift_log)

    @property
    def stats(self) -> dict[str, Any]:
        """Current strategy statistics."""
        strategy_state = self._risk.get_strategy_state(STRATEGY_ID)

        # Drift summary
        drift_samples = len(self._drift_log)
        avg_drift = 0.0
        avg_abs_drift = 0.0
        max_abs_drift = 0.0
        if drift_samples > 0:
            deltas = [d["delta"] for d in self._drift_log]
            abs_deltas = [abs(d) for d in deltas]
            avg_drift = round(sum(deltas) / drift_samples, 6)
            avg_abs_drift = round(sum(abs_deltas) / drift_samples, 6)
            max_abs_drift = round(max(abs_deltas), 6)

        return {
            "strategy": STRATEGY_ID,
            "signals_generated": self._signals_generated,
            "trades_placed": self._trades_placed,
            "cities_covered": len(self._forecasts),
            "active_chains": len(self._chain_engine.get_active_chains()),
            "total_chains": len(self._chain_engine.get_all_chains()),
            "last_scan": self._last_scan.isoformat() if self._last_scan else None,
            "deployed": str(strategy_state.deployed) if strategy_state else "N/A",
            "available": str(strategy_state.available) if strategy_state else "N/A",
            "is_halted": strategy_state.is_halted if strategy_state else False,
            "drift_samples": drift_samples,
            "avg_drift": avg_drift,
            "avg_abs_drift": avg_abs_drift,
            "max_abs_drift": max_abs_drift,
            "trades_skipped_volatility": self._trades_skipped_volatility,
        }

    # ------------------------------------------------------------------
    # Data collection: scan logs, volume snapshots, forecasts
    # ------------------------------------------------------------------

    async def save_scan_log(
        self,
        all_signals: list[WeatherSignal],
        qualified: list[WeatherSignal],
        traded_condition_ids: set[str],
        traded_sizes: dict[str, float],
        forecast_fetched_at: datetime,
    ) -> None:
        """Persist every scanned signal (qualified + rejected) to weather_scan_log.

        This captures the full picture: what was available, what passed quality gate,
        and what was actually traded. Critical for future backtesting analysis.
        """
        try:
            from arbo.utils.db import WeatherScanLog, get_session_factory

            now = datetime.now(timezone.utc)
            qualified_ids = {s.market.condition_id for s in qualified}
            factory = get_session_factory()

            async with factory() as session:
                for signal in all_signals:
                    is_qualified = signal.market.condition_id in qualified_ids
                    is_traded = signal.market.condition_id in traded_condition_ids

                    # Get quality gate reason for rejected signals
                    gate_reason = None
                    if not is_qualified:
                        decision = check_signal_quality(signal, forecast_fetched_at)
                        gate_reason = decision.reason

                    row = WeatherScanLog(
                        scan_at=now,
                        city=signal.market.city.value,
                        target_date=signal.market.target_date,
                        condition_id=signal.market.condition_id,
                        question=signal.market.question,
                        forecast_temp_c=signal.forecast_temp_c,
                        forecast_prob=signal.forecast_probability,
                        market_price=signal.market.market_price,
                        edge=signal.edge,
                        direction=signal.direction,
                        volume_24h=signal.market.volume_24h,
                        liquidity=signal.market.liquidity,
                        quality_gate_passed=is_qualified,
                        quality_gate_reason=gate_reason if not is_qualified else "passed",
                        traded=is_traded,
                        trade_size=traded_sizes.get(signal.market.condition_id),
                    )
                    session.add(row)
                await session.commit()

            logger.info(
                "scan_log_saved",
                total=len(all_signals),
                qualified=len(qualified),
                traded=len(traded_condition_ids),
            )
        except Exception as e:
            logger.debug("scan_log_save_error", error=str(e))

    async def save_volume_snapshot(self, markets: list[Any]) -> None:
        """Save daily volume per city from weather markets.

        Aggregates volume_24h and liquidity per city, stores one row per city per day.
        Uses upsert to avoid duplicates within the same day.
        """
        try:
            from collections import defaultdict

            from sqlalchemy.dialects.postgresql import insert as pg_insert

            from arbo.utils.db import CityVolumeDaily, get_session_factory

            today = date.today()
            city_data: dict[str, dict[str, Any]] = defaultdict(
                lambda: {"volume": 0.0, "liquidity": 0.0, "count": 0, "prices": []}
            )

            for m in markets:
                # Parse city from market question
                city_name = getattr(m, "city", None)
                if city_name is None:
                    from arbo.strategies.weather_scanner import parse_city

                    q = getattr(m, "question", "") or getattr(m, "title", "")
                    city_obj = parse_city(q)
                    if city_obj is None:
                        continue
                    city_name = city_obj.value
                elif hasattr(city_name, "value"):
                    city_name = city_name.value

                vol = float(getattr(m, "volume_24h", 0) or 0)
                liq = float(getattr(m, "liquidity", 0) or 0)
                price = float(getattr(m, "price_yes", 0) or getattr(m, "market_price", 0) or 0)

                city_data[city_name]["volume"] += vol
                city_data[city_name]["liquidity"] += liq
                city_data[city_name]["count"] += 1
                if price > 0:
                    city_data[city_name]["prices"].append(price)

            if not city_data:
                return

            factory = get_session_factory()
            async with factory() as session:
                for city, data in city_data.items():
                    avg_price = (
                        sum(data["prices"]) / len(data["prices"]) if data["prices"] else None
                    )
                    stmt = pg_insert(CityVolumeDaily).values(
                        city=city,
                        date=today,
                        volume_24h=Decimal(str(round(data["volume"], 2))),
                        liquidity=Decimal(str(round(data["liquidity"], 2))),
                        num_markets=data["count"],
                        avg_price=avg_price,
                    )
                    stmt = stmt.on_conflict_do_update(
                        constraint="uq_city_volume_daily",
                        set_={
                            "volume_24h": stmt.excluded.volume_24h,
                            "liquidity": stmt.excluded.liquidity,
                            "num_markets": stmt.excluded.num_markets,
                            "avg_price": stmt.excluded.avg_price,
                            "captured_at": datetime.now(timezone.utc),
                        },
                    )
                    await session.execute(stmt)
                await session.commit()

            logger.info("volume_snapshot_saved", cities=len(city_data))
        except Exception as e:
            logger.debug("volume_snapshot_save_error", error=str(e))
