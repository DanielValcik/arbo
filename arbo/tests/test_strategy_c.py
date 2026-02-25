"""Tests for Strategy C: Compound Weather Resolution Chaining.

Integration tests covering the full poll cycle from forecast to trade.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest
from aioresponses import aioresponses

from arbo.connectors.weather_models import City, WeatherSource
from arbo.core.paper_engine import PaperTradingEngine, TradeStatus
from arbo.core.risk_manager import RiskManager
from arbo.strategies.strategy_c import StrategyC


# Mock market matching GammaMarket interface
@dataclass
class MockGammaMarket:
    condition_id: str = "0xweather123"
    question: str = "Will the high temperature in NYC be above 75°F on March 15?"
    category: str = "weather"
    price_yes: Decimal | None = Decimal("0.30")
    price_no: Decimal | None = Decimal("0.70")
    token_id_yes: str = "tok_yes_1"
    token_id_no: str = "tok_no_1"
    neg_risk: bool = False
    fee_enabled: bool = False
    volume_24h: Decimal = Decimal("50000")
    liquidity: Decimal = Decimal("25000")
    slug: str = "weather-nyc"


# Mock paper engine
class MockPaperEngine:
    def __init__(self) -> None:
        self.trades: list[dict[str, Any]] = []

    def place_trade(self, **kwargs: Any) -> dict[str, Any] | None:
        trade = {"id": len(self.trades) + 1, **kwargs}
        self.trades.append(trade)
        return trade


# Sample NOAA response
NOAA_RESPONSE = {
    "properties": {
        "periods": [
            {
                "number": 1,
                "name": "Saturday",
                "startTime": "2026-03-15T06:00:00-04:00",
                "endTime": "2026-03-15T18:00:00-04:00",
                "isDaytime": True,
                "temperature": 82,
                "temperatureUnit": "F",
                "shortForecast": "Sunny",
                "probabilityOfPrecipitation": {"unitCode": "wmoUnit:percent", "value": 5},
            },
            {
                "number": 2,
                "name": "Saturday Night",
                "startTime": "2026-03-15T18:00:00-04:00",
                "endTime": "2026-03-16T06:00:00-04:00",
                "isDaytime": False,
                "temperature": 60,
                "temperatureUnit": "F",
                "shortForecast": "Clear",
                "probabilityOfPrecipitation": {"unitCode": "wmoUnit:percent", "value": 0},
            },
        ],
    },
}

NYC_URL = re.compile(r"https://api\.weather\.gov/gridpoints/OKX/33,37/forecast.*")
CHICAGO_URL = re.compile(r"https://api\.weather\.gov/gridpoints/LOT/76,73/forecast.*")
OPENMETEO_URL = re.compile(r"https://api\.open-meteo\.com/v1/forecast.*")

OPENMETEO_RESPONSE = {
    "daily": {
        "time": ["2026-03-15"],
        "temperature_2m_max": [25.0],
        "temperature_2m_min": [15.0],
        "precipitation_probability_max": [10],
        "weather_code": [1],
    },
}


@pytest.fixture
def risk() -> RiskManager:
    RiskManager.reset()
    return RiskManager(capital=Decimal("2000"))


@pytest.fixture
def paper_engine() -> MockPaperEngine:
    return MockPaperEngine()


@pytest.fixture
def strategy(risk: RiskManager, paper_engine: MockPaperEngine) -> StrategyC:
    return StrategyC(
        risk_manager=risk,
        paper_engine=paper_engine,
        metoffice_api_key="",  # Skip Met Office in tests
    )


class TestStrategyInit:
    """Test strategy initialization."""

    async def test_init_creates_clients(self, strategy: StrategyC) -> None:
        await strategy.init()
        assert strategy._noaa is not None
        assert strategy._openmeteo is not None
        assert strategy._metoffice is None  # No API key
        await strategy.close()


class TestFetchForecasts:
    """Test forecast fetching."""

    async def test_fetch_forecasts(self, strategy: StrategyC) -> None:
        await strategy.init()
        with aioresponses() as m:
            m.get(NYC_URL, payload=NOAA_RESPONSE)
            m.get(CHICAGO_URL, payload=NOAA_RESPONSE)
            m.get(OPENMETEO_URL, payload=OPENMETEO_RESPONSE)
            m.get(OPENMETEO_URL, payload=OPENMETEO_RESPONSE)

            forecasts = await strategy.fetch_forecasts()

        assert len(forecasts) >= 2  # At least NOAA cities
        await strategy.close()


class TestPollCycle:
    """Test full poll cycle."""

    async def test_poll_with_mispriced_market(
        self,
        strategy: StrategyC,
        paper_engine: MockPaperEngine,
    ) -> None:
        """Market at 0.30 but forecast says ~90% → should trade."""
        await strategy.init()

        markets = [
            MockGammaMarket(
                question="Will the high temperature in NYC be above 75°F on March 15?",
                price_yes=Decimal("0.30"),
                volume_24h=Decimal("50000"),
                liquidity=Decimal("25000"),
            ),
        ]

        with aioresponses() as m:
            m.get(NYC_URL, payload=NOAA_RESPONSE)
            m.get(CHICAGO_URL, payload=NOAA_RESPONSE)
            m.get(OPENMETEO_URL, payload=OPENMETEO_RESPONSE)
            m.get(OPENMETEO_URL, payload=OPENMETEO_RESPONSE)

            traded = await strategy.poll_cycle(markets)

        # Should have found the edge and placed a trade
        assert len(traded) >= 1
        assert len(paper_engine.trades) >= 1
        assert strategy._trades_placed >= 1
        await strategy.close()

    async def test_poll_with_no_edge(
        self,
        strategy: StrategyC,
        paper_engine: MockPaperEngine,
    ) -> None:
        """Fairly priced market → no trade."""
        await strategy.init()

        # NOAA says 82°F high ≈ 27.8°C, threshold at 75°F ≈ 23.9°C
        # So ~94% probability above 75°F
        # If market is priced at 0.90, edge is ~4% → below 5% threshold
        markets = [
            MockGammaMarket(
                question="Will the high temperature in NYC be above 75°F on March 15?",
                price_yes=Decimal("0.92"),
                volume_24h=Decimal("50000"),
                liquidity=Decimal("25000"),
            ),
        ]

        with aioresponses() as m:
            m.get(NYC_URL, payload=NOAA_RESPONSE)
            m.get(CHICAGO_URL, payload=NOAA_RESPONSE)
            m.get(OPENMETEO_URL, payload=OPENMETEO_RESPONSE)
            m.get(OPENMETEO_URL, payload=OPENMETEO_RESPONSE)

            traded = await strategy.poll_cycle(markets)

        assert len(traded) == 0
        assert len(paper_engine.trades) == 0
        await strategy.close()

    async def test_poll_with_empty_markets(
        self,
        strategy: StrategyC,
    ) -> None:
        """No markets → no signals."""
        await strategy.init()
        with aioresponses() as m:
            m.get(NYC_URL, payload=NOAA_RESPONSE)
            m.get(CHICAGO_URL, payload=NOAA_RESPONSE)
            m.get(OPENMETEO_URL, payload=OPENMETEO_RESPONSE)
            m.get(OPENMETEO_URL, payload=OPENMETEO_RESPONSE)

            traded = await strategy.poll_cycle([])
        assert traded == []
        await strategy.close()


class TestStrategyStats:
    """Test strategy statistics."""

    async def test_stats_initial(self, strategy: StrategyC) -> None:
        stats = strategy.stats
        assert stats["strategy"] == "C"
        assert stats["signals_generated"] == 0
        assert stats["trades_placed"] == 0

    async def test_stats_after_poll(
        self,
        strategy: StrategyC,
    ) -> None:
        await strategy.init()

        markets = [
            MockGammaMarket(
                question="Will the high temperature in NYC be above 75°F on March 15?",
                price_yes=Decimal("0.30"),
            ),
        ]

        with aioresponses() as m:
            m.get(NYC_URL, payload=NOAA_RESPONSE)
            m.get(CHICAGO_URL, payload=NOAA_RESPONSE)
            m.get(OPENMETEO_URL, payload=OPENMETEO_RESPONSE)
            m.get(OPENMETEO_URL, payload=OPENMETEO_RESPONSE)

            await strategy.poll_cycle(markets)

        stats = strategy.stats
        assert stats["signals_generated"] >= 1
        assert stats["last_scan"] is not None
        await strategy.close()


class TestResolutionHandling:
    """Test market resolution handling."""

    def test_handle_resolution(
        self,
        strategy: StrategyC,
        risk: RiskManager,
    ) -> None:
        # Start a chain
        chain = strategy._chain_engine.start_chain(
            Decimal("200"),
            city_sequence=[City.NYC, City.CHICAGO],
        )
        strategy._chain_engine.set_active_market(chain.chain_id, "nyc_mkt")

        # Simulate a trade being open (update risk state)
        risk.post_trade_update("nyc_mkt", "weather", Decimal("50"))
        risk.strategy_post_trade("C", Decimal("50"))

        # Resolve with profit
        next_city = strategy.handle_resolution(
            chain.chain_id, "nyc_mkt", pnl=Decimal("10")
        )

        assert next_city == City.CHICAGO
        assert chain.cumulative_pnl == Decimal("10")


class TestE2EWithRealPaperEngine:
    """E2E integration tests using real PaperTradingEngine (not mock)."""

    @pytest.fixture
    def real_risk(self) -> RiskManager:
        RiskManager.reset()
        return RiskManager(capital=Decimal("2000"))

    @pytest.fixture
    def real_paper(self, real_risk: RiskManager) -> PaperTradingEngine:
        return PaperTradingEngine(
            initial_capital=Decimal("2000"),
            risk_manager=real_risk,
        )

    @pytest.fixture
    def real_strategy(
        self, real_risk: RiskManager, real_paper: PaperTradingEngine
    ) -> StrategyC:
        return StrategyC(
            risk_manager=real_risk,
            paper_engine=real_paper,
            metoffice_api_key="",
        )

    async def test_e2e_poll_trade_resolve(
        self,
        real_strategy: StrategyC,
        real_paper: PaperTradingEngine,
        real_risk: RiskManager,
    ) -> None:
        """Full E2E: forecast → scan → quality gate → ladder → risk → paper trade → resolve."""
        await real_strategy.init()

        markets = [
            MockGammaMarket(
                question="Will the high temperature in NYC be above 75°F on March 15?",
                price_yes=Decimal("0.30"),
                volume_24h=Decimal("50000"),
                liquidity=Decimal("25000"),
            ),
        ]

        with aioresponses() as m:
            m.get(NYC_URL, payload=NOAA_RESPONSE)
            m.get(CHICAGO_URL, payload=NOAA_RESPONSE)
            m.get(OPENMETEO_URL, payload=OPENMETEO_RESPONSE)
            m.get(OPENMETEO_URL, payload=OPENMETEO_RESPONSE)

            traded = await real_strategy.poll_cycle(markets)

        # Trade should be placed with real paper engine
        assert len(traded) >= 1
        assert len(real_paper.trade_history) >= 1

        # Verify trade has strategy="C" field
        trade = real_paper.trade_history[0]
        assert trade.strategy == "C"
        assert trade.status == TradeStatus.OPEN
        assert trade.size > Decimal("0")

        # Balance should have decreased
        assert real_paper.balance < Decimal("2000")

        # Resolve the market (weather above 75°F → YES wins)
        token_id = trade.token_id
        pnl = real_paper.resolve_market(token_id, winning_outcome=True)
        assert pnl > Decimal("0")  # Won the trade

        # Balance restored + profit
        assert real_paper.balance > Decimal("2000") - trade.size

        # Per-strategy P&L tracked
        strategy_stats = real_paper.get_strategy_stats("C")
        assert strategy_stats["total_trades"] == 1
        assert strategy_stats["wins"] == 1
        assert strategy_stats["total_pnl"] > Decimal("0")

        await real_strategy.close()

    async def test_e2e_strategy_field_on_trade(
        self,
        real_strategy: StrategyC,
        real_paper: PaperTradingEngine,
    ) -> None:
        """Verify strategy='C' is set on all trades from Strategy C."""
        await real_strategy.init()

        markets = [
            MockGammaMarket(
                question="Will the high temperature in NYC be above 75°F on March 15?",
                price_yes=Decimal("0.30"),
                volume_24h=Decimal("50000"),
                liquidity=Decimal("25000"),
            ),
        ]

        with aioresponses() as m:
            m.get(NYC_URL, payload=NOAA_RESPONSE)
            m.get(CHICAGO_URL, payload=NOAA_RESPONSE)
            m.get(OPENMETEO_URL, payload=OPENMETEO_RESPONSE)
            m.get(OPENMETEO_URL, payload=OPENMETEO_RESPONSE)

            await real_strategy.poll_cycle(markets)

        for trade in real_paper.trade_history:
            assert trade.strategy == "C"

        await real_strategy.close()

    async def test_e2e_resolution_chain_lifecycle(
        self,
        real_strategy: StrategyC,
        real_risk: RiskManager,
    ) -> None:
        """Full chain lifecycle: start → deploy → resolve → advance → complete."""
        # Start a 2-city chain
        chain = real_strategy._chain_engine.start_chain(
            Decimal("200"),
            city_sequence=[City.NYC, City.CHICAGO],
        )

        # Deploy to NYC
        real_strategy._chain_engine.set_active_market(chain.chain_id, "nyc_mkt")
        real_risk.post_trade_update("nyc_mkt", "weather", Decimal("50"))
        real_risk.strategy_post_trade("C", Decimal("50"))

        # Resolve NYC with profit → should advance to Chicago
        next_city = real_strategy.handle_resolution(
            chain.chain_id, "nyc_mkt", pnl=Decimal("15")
        )
        assert next_city == City.CHICAGO
        assert chain.current_capital == Decimal("215")

        # Deploy to Chicago
        real_strategy._chain_engine.set_active_market(chain.chain_id, "chi_mkt")
        real_risk.post_trade_update("chi_mkt", "weather", Decimal("50"))
        real_risk.strategy_post_trade("C", Decimal("50"))

        # Resolve Chicago with loss → chain complete (only 2 cities)
        next_city = real_strategy.handle_resolution(
            chain.chain_id, "chi_mkt", pnl=Decimal("-5")
        )
        assert next_city is None  # Chain complete
        assert chain.cumulative_pnl == Decimal("10")  # 15 - 5
        assert chain.is_complete

    async def test_e2e_per_strategy_stats(
        self,
        real_paper: PaperTradingEngine,
    ) -> None:
        """get_strategy_stats returns correct per-strategy breakdown."""
        # No trades yet
        stats = real_paper.get_strategy_stats("C")
        assert stats["total_trades"] == 0
        assert stats["total_pnl"] == Decimal("0")

        # Place a trade with strategy="C"
        real_paper.place_trade(
            market_condition_id="weather_test",
            token_id="tok_c_1",
            side="BUY",
            market_price=Decimal("0.40"),
            model_prob=Decimal("0.55"),
            layer=0,
            market_category="weather",
            strategy="C",
        )

        # Place a legacy trade (no strategy)
        real_paper.place_trade(
            market_condition_id="sports_test",
            token_id="tok_legacy",
            side="BUY",
            market_price=Decimal("0.50"),
            model_prob=Decimal("0.60"),
            layer=2,
            market_category="sports",
        )

        stats_c = real_paper.get_strategy_stats("C")
        assert stats_c["total_trades"] == 1
        assert stats_c["open_trades"] == 1

        stats_legacy = real_paper.get_strategy_stats("")
        assert stats_legacy["total_trades"] == 1

        # Overall stats should show both
        overall = real_paper.get_stats()
        assert overall["total_trades"] == 2
        assert "per_strategy_pnl" in overall
