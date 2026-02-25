"""Strategy C: Compound Weather Resolution Chaining.

Main strategy module that orchestrates:
1. Weather forecast fetching (NOAA, Met Office, Open-Meteo)
2. Weather market scanning (Gamma API)
3. Quality gate filtering
4. Temperature laddering
5. Resolution chain management
6. Paper trading integration

Poll cycle: fetch forecasts → scan markets → quality gate → ladder → execute
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from arbo.connectors.weather_models import City, WeatherForecast
from arbo.connectors.weather_noaa import NOAAWeatherClient
from arbo.connectors.weather_metoffice import MetOfficeWeatherClient
from arbo.connectors.weather_openmeteo import OpenMeteoWeatherClient
from arbo.core.risk_manager import MAX_POSITION_PCT, RiskManager, TradeRequest
from arbo.strategies.resolution_chain import ResolutionChainEngine
from arbo.strategies.weather_ladder import build_ladders_by_city
from arbo.strategies.weather_quality_gate import filter_signals
from arbo.strategies.weather_scanner import WeatherSignal, scan_weather_markets
from arbo.utils.logger import get_logger

logger = get_logger("strategy_c")

STRATEGY_ID = "C"
DEFAULT_SCAN_INTERVAL_S = 1800  # 30 minutes


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
    ) -> None:
        self._risk = risk_manager
        self._paper_engine = paper_engine
        self._metoffice_key = metoffice_api_key

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

        # 6. Execute via paper engine (or live in the future)
        traded_signals = []
        for ladder in ladders:
            for position in ladder.positions:
                signal = position.signal

                # Build trade request for risk manager
                trade_req = TradeRequest(
                    market_id=signal.market.condition_id,
                    token_id=(
                        signal.market.token_id_yes
                        if signal.direction == "BUY_YES"
                        else signal.market.token_id_no
                    ),
                    side="BUY",
                    price=Decimal(str(signal.market.market_price)),
                    size=position.size_usdc,
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
                    trade_result = self._paper_engine.place_trade(
                        market_condition_id=signal.market.condition_id,
                        token_id=trade_req.token_id,
                        side="BUY",
                        market_price=Decimal(str(signal.market.market_price)),
                        model_prob=Decimal(str(signal.forecast_probability)),
                        layer=0,
                        market_category="weather",
                        strategy=STRATEGY_ID,
                        pre_computed_size=position.size_usdc,
                    )
                    if trade_result:
                        self._risk.post_trade_update(
                            signal.market.condition_id, "weather", position.size_usdc
                        )
                        self._risk.strategy_post_trade(STRATEGY_ID, position.size_usdc)
                        traded_signals.append(signal)
                        self._trades_placed += 1

                        logger.info(
                            "paper_trade_placed",
                            city=signal.market.city.value,
                            direction=signal.direction,
                            size=str(position.size_usdc),
                            edge=round(signal.edge, 4),
                        )

        self._last_scan = datetime.now(timezone.utc)
        logger.info(
            "poll_cycle_complete",
            signals_total=len(signals),
            qualified=len(qualified),
            ladders=len(ladders),
            trades=len(traded_signals),
        )
        return traded_signals

    def handle_resolution(
        self,
        chain_id: str,
        market_id: str,
        pnl: Decimal,
    ) -> City | None:
        """Handle a market resolution within a chain.

        Returns the next city to deploy to, or None if chain is done.
        """
        # Update risk manager
        self._risk.post_trade_update(
            market_id, "weather", abs(pnl), pnl=pnl
        )
        self._risk.strategy_post_trade(STRATEGY_ID, abs(pnl), pnl=pnl)

        # Advance chain
        return self._chain_engine.resolve(chain_id, market_id, pnl)

    @property
    def stats(self) -> dict[str, Any]:
        """Current strategy statistics."""
        strategy_state = self._risk.get_strategy_state(STRATEGY_ID)
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
        }
