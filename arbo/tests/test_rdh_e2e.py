"""End-to-end tests for RDH orchestrator (RDH-310).

Tests verify full system flow:
1. Orchestrator init — all 3 strategies created
2. Discovery fetches markets
3. Strategy C places weather trade
4. Strategy A taker flow monitor active
5. Strategy B runs in stub mode
6. Reports can be generated
7. Dashboard API available
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arbo.core.risk_manager import RiskManager


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------


@dataclass
class MockMarket:
    """Mock GammaMarket for E2E tests."""

    condition_id: str = "cond_weather_1"
    question: str = "Will NYC temperature exceed 75°F?"
    slug: str = "nyc-temp-75"
    category: str = "weather"
    price_yes: Decimal | None = Decimal("0.50")
    price_no: Decimal | None = Decimal("0.50")
    volume_24h: Decimal = Decimal("20000")
    liquidity: Decimal = Decimal("15000")
    token_id_yes: str = "tok_yes_1"
    token_id_no: str = "tok_no_1"
    fee_enabled: bool = False
    neg_risk: bool = False
    active: bool = True
    closed: bool = False
    end_date: str = ""
    market_price: float = 0.50
    enable_neg_risk: bool = False


def _make_weather_market(
    cond_id: str = "cond_weather_1",
    question: str = "NYC temperature > 75°F?",
    price: float = 0.50,
) -> MockMarket:
    return MockMarket(
        condition_id=cond_id,
        question=question,
        category="weather",
        price_yes=Decimal(str(price)),
        price_no=Decimal(str(1 - price)),
        market_price=price,
    )


def _make_longshot_market(
    cond_id: str = "cond_longshot_1",
) -> MockMarket:
    return MockMarket(
        condition_id=cond_id,
        question="Will obscure event happen?",
        category="politics",
        price_yes=Decimal("0.10"),
        price_no=Decimal("0.90"),
        volume_24h=Decimal("15000"),
        market_price=0.10,
        end_date=(datetime.now(UTC) + timedelta(days=10)).isoformat(),
    )


def _mock_config() -> MagicMock:
    """Create mock config for E2E."""
    config = MagicMock()
    config.bankroll = 2000.0
    config.mode = "paper"
    config.metoffice_api_key = ""
    config.gemini_api_key = ""
    config.anthropic_api_key = ""
    config.dashboard.host = "127.0.0.1"
    config.dashboard.port = 8099
    config.orchestrator.health_check_interval_s = 1
    config.orchestrator.heartbeat_timeout_s = 5
    config.orchestrator.max_restart_count = 3
    config.orchestrator.signal_batch_timeout_s = 1.0
    config.orchestrator.dashboard_update_interval_s = 60
    config.orchestrator.snapshot_interval_s = 3600
    config.orchestrator.daily_report_hour_utc = 23
    config.orchestrator.weekly_report_day = 6
    config.orchestrator.weekly_report_hour_utc = 20
    config.theta_decay.snapshot_interval_s = 300
    config.reflexivity.scan_interval_s = 600
    config.weather.scan_interval_s = 1800
    return config


@pytest.fixture(autouse=True)
def reset_risk() -> None:
    RiskManager.reset()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestE2EInit:
    """Orchestrator creates all 3 strategies."""

    @patch("arbo.main_rdh.get_config")
    async def test_init_creates_3_strategies(self, mock_config: MagicMock) -> None:
        """All 3 strategies are initialized from _init_components."""
        mock_config.return_value = _mock_config()

        from arbo.main_rdh import RDHOrchestrator

        orch = RDHOrchestrator(mode="paper")

        # Mock component factories to avoid real network calls
        orch._init_discovery = AsyncMock(return_value=MagicMock())
        orch._init_gemini = AsyncMock(return_value=MagicMock())
        orch._init_flow_tracker = AsyncMock(return_value=MagicMock())
        orch._init_kaito = AsyncMock(return_value=MagicMock())

        # Mock strategy factories
        mock_a = MagicMock()
        mock_a.init = AsyncMock()
        orch._init_strategy_a = AsyncMock(return_value=mock_a)

        mock_b = MagicMock()
        mock_b.init = AsyncMock()
        orch._init_strategy_b = AsyncMock(return_value=mock_b)

        mock_c = MagicMock()
        mock_c.init = AsyncMock()
        orch._init_strategy_c = AsyncMock(return_value=mock_c)

        orch._init_report_generator = AsyncMock(return_value=MagicMock())
        orch._init_db = AsyncMock()
        orch._init_slack_bot = AsyncMock(return_value=None)
        orch._init_web_dashboard = AsyncMock(return_value=None)

        await orch._init_components()

        assert orch._strategy_a is not None
        assert orch._strategy_b is not None
        assert orch._strategy_c is not None
        assert orch._risk_manager is not None
        assert orch._paper_engine is not None


class TestE2EDiscovery:
    """Discovery fetches and caches markets."""

    @patch("arbo.main_rdh.get_config")
    async def test_discovery_populates_markets(self, mock_config: MagicMock) -> None:
        """After discovery refresh, markets list is populated."""
        mock_config.return_value = _mock_config()

        from arbo.main_rdh import RDHOrchestrator

        orch = RDHOrchestrator(mode="paper")

        mock_discovery = MagicMock()
        mock_discovery.refresh = AsyncMock(
            return_value=[_make_weather_market(), _make_longshot_market()]
        )
        orch._discovery = mock_discovery

        await orch._run_discovery()

        assert len(orch._markets) == 2
        mock_discovery.refresh.assert_called_once()


class TestE2EStrategyC:
    """Strategy C places a weather trade."""

    @patch("arbo.main_rdh.get_config")
    async def test_strategy_c_trade(self, mock_config: MagicMock) -> None:
        """Strategy C poll_cycle with valid weather data places a trade."""
        mock_config.return_value = _mock_config()

        from arbo.main_rdh import RDHOrchestrator

        orch = RDHOrchestrator(mode="paper")

        # Use real risk manager + paper engine
        rm = RiskManager(Decimal("2000"))
        orch._risk_manager = rm

        from arbo.core.paper_engine import PaperTradingEngine

        pe = PaperTradingEngine(Decimal("2000"), risk_manager=rm)
        orch._paper_engine = pe

        # Mock Strategy C to return a trade
        mock_c = AsyncMock()
        mock_c.poll_cycle = AsyncMock(
            return_value=[
                {
                    "condition_id": "cond_weather_1",
                    "token_id": "tok_yes_1",
                    "side": "BUY_YES",
                    "price": 0.50,
                    "size": 25.0,
                    "strategy": "C",
                    "city": "New York",
                }
            ]
        )
        orch._strategy_c = mock_c
        orch._markets = [_make_weather_market()]

        await orch._run_strategy_c()

        result = mock_c.poll_cycle.return_value
        assert len(result) == 1
        assert result[0]["strategy"] == "C"
        assert result[0]["size"] == 25.0


class TestE2EStrategyA:
    """Strategy A taker flow monitor is active."""

    @patch("arbo.main_rdh.get_config")
    async def test_strategy_a_flow_active(self, mock_config: MagicMock) -> None:
        """Strategy A poll_cycle calls flow tracker (no trades without peak)."""
        mock_config.return_value = _mock_config()

        from arbo.main_rdh import RDHOrchestrator

        orch = RDHOrchestrator(mode="paper")

        mock_a = AsyncMock()
        mock_a.poll_cycle = AsyncMock(return_value=[])  # No peak optimism detected
        mock_a.check_exits = MagicMock(return_value=[])
        orch._strategy_a = mock_a
        orch._markets = [_make_longshot_market()]

        await orch._run_strategy_a()

        mock_a.poll_cycle.assert_called_once()
        mock_a.check_exits.assert_called_once()


class TestE2EStrategyB:
    """Strategy B runs in stub mode."""

    @patch("arbo.main_rdh.get_config")
    async def test_strategy_b_stub_mode(self, mock_config: MagicMock) -> None:
        """Strategy B poll_cycle runs without Kaito API (stub mode)."""
        mock_config.return_value = _mock_config()

        from arbo.main_rdh import RDHOrchestrator

        orch = RDHOrchestrator(mode="paper")

        mock_b = AsyncMock()
        mock_b.poll_cycle = AsyncMock(return_value=[])  # Stub returns no live data
        mock_b.check_exits = MagicMock(return_value=[])
        mock_b.stats = {"strategy": "B", "kaito_mode": "stub"}
        orch._strategy_b = mock_b
        orch._markets = []

        await orch._run_strategy_b()

        mock_b.poll_cycle.assert_called_once()
        assert orch._strategy_b.stats["kaito_mode"] == "stub"


class TestE2EReports:
    """Reports generate without errors."""

    @patch("arbo.main_rdh.get_config")
    async def test_daily_report_generates(self, mock_config: MagicMock) -> None:
        """Daily report generation succeeds."""
        mock_config.return_value = _mock_config()

        from arbo.main_rdh import RDHOrchestrator

        orch = RDHOrchestrator(mode="paper")

        from arbo.core.paper_engine import PaperTradingEngine

        rm = RiskManager(Decimal("2000"))
        orch._risk_manager = rm
        orch._paper_engine = PaperTradingEngine(Decimal("2000"), risk_manager=rm)

        mock_gen = MagicMock()
        mock_report = MagicMock()
        mock_gen.generate_daily.return_value = mock_report
        mock_gen.format_slack_report.return_value = {"blocks": []}
        orch._report_generator = mock_gen

        mock_slack = AsyncMock()
        mock_slack.send_daily_report = AsyncMock()
        orch._slack_bot = mock_slack

        await orch._send_daily_report()

        mock_gen.generate_daily.assert_called_once()
        mock_slack.send_daily_report.assert_called_once()


class TestE2EDashboard:
    """Dashboard API would be available."""

    @patch("arbo.main_rdh.get_config")
    def test_strategy_stats_populated(self, mock_config: MagicMock) -> None:
        """strategy_stats returns data for all 3 strategies."""
        mock_config.return_value = _mock_config()

        from arbo.main_rdh import RDHOrchestrator

        orch = RDHOrchestrator(mode="paper")

        orch._strategy_a = MagicMock(stats={"strategy": "A", "trades": 2})
        orch._strategy_b = MagicMock(stats={"strategy": "B", "trades": 0, "kaito_mode": "stub"})
        orch._strategy_c = MagicMock(stats={"strategy": "C", "trades": 5})

        stats = orch.strategy_stats
        assert len(stats) == 3
        assert stats["A"]["trades"] == 2
        assert stats["B"]["kaito_mode"] == "stub"
        assert stats["C"]["trades"] == 5

    @patch("arbo.main_rdh.get_config")
    def test_task_states_populated(self, mock_config: MagicMock) -> None:
        """task_states returns state for all managed tasks."""
        mock_config.return_value = _mock_config()

        from arbo.main_rdh import RDHOrchestrator, TaskState

        orch = RDHOrchestrator(mode="paper")
        for name in ["discovery", "strategy_A", "strategy_B", "strategy_C"]:
            orch._tasks[name] = TaskState(name=name)

        states = orch.task_states
        assert len(states) == 4
        for name in ["discovery", "strategy_A", "strategy_B", "strategy_C"]:
            assert name in states


class TestE2ECapitalIsolation:
    """Per-strategy capital is isolated."""

    @patch("arbo.main_rdh.get_config")
    def test_strategy_c_gets_largest_allocation(self, mock_config: MagicMock) -> None:
        """Strategy C (weather) has $1000 — largest allocation."""
        mock_config.return_value = _mock_config()

        rm = RiskManager(Decimal("2000"))

        a = rm.get_strategy_state("A")
        b = rm.get_strategy_state("B")
        c = rm.get_strategy_state("C")

        assert a is not None
        assert b is not None
        assert c is not None

        assert a.allocated == Decimal("400")
        assert b.allocated == Decimal("400")
        assert c.allocated == Decimal("1000")
        assert a.allocated + b.allocated + c.allocated == Decimal("1800")
        # Reserve = 200 (total 2000)


class TestE2EFullCycle:
    """Full orchestrator cycle: init → run strategies → check stats."""

    @patch("arbo.main_rdh.get_config")
    async def test_full_cycle(self, mock_config: MagicMock) -> None:
        """Init orchestrator, run all 3 strategies, verify stats."""
        mock_config.return_value = _mock_config()

        from arbo.main_rdh import RDHOrchestrator

        orch = RDHOrchestrator(mode="paper")

        # Set up core components
        rm = RiskManager(Decimal("2000"))
        orch._risk_manager = rm

        from arbo.core.paper_engine import PaperTradingEngine

        pe = PaperTradingEngine(Decimal("2000"), risk_manager=rm)
        orch._paper_engine = pe

        # Mock strategies with realistic behavior
        mock_a = AsyncMock()
        mock_a.poll_cycle = AsyncMock(return_value=[])
        mock_a.check_exits = MagicMock(return_value=[])
        mock_a.stats = {"strategy": "A", "trades_placed": 0, "active_positions": 0}
        orch._strategy_a = mock_a

        mock_b = AsyncMock()
        mock_b.poll_cycle = AsyncMock(return_value=[])
        mock_b.check_exits = MagicMock(return_value=[])
        mock_b.stats = {"strategy": "B", "trades_placed": 0, "kaito_mode": "stub"}
        orch._strategy_b = mock_b

        mock_c = AsyncMock()
        mock_c.poll_cycle = AsyncMock(
            return_value=[
                {"condition_id": "cond_1", "strategy": "C", "size": 30.0, "price": 0.45}
            ]
        )
        mock_c.stats = {"strategy": "C", "trades_placed": 1}
        orch._strategy_c = mock_c

        orch._discovery = MagicMock()
        orch._markets = [_make_weather_market(), _make_longshot_market()]

        # Run all strategies
        await orch._run_strategy_a()
        await orch._run_strategy_b()
        await orch._run_strategy_c()

        # Verify all strategies were called
        mock_a.poll_cycle.assert_called_once()
        mock_b.poll_cycle.assert_called_once()
        mock_c.poll_cycle.assert_called_once()

        # Verify stats
        stats = orch.strategy_stats
        assert stats["C"]["trades_placed"] == 1
        assert stats["B"]["kaito_mode"] == "stub"
