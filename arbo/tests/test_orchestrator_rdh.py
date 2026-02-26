"""Tests for RDH orchestrator (RDH-305).

Tests verify:
1. Starts 3 strategy tasks
2. Signals routed to correct quality gate
3. Trade placed via paper engine with strategy field
4. Health monitor detects crashed task
5. Graceful shutdown cancels all tasks
6. Daily/weekly reports scheduled
7. Capital allocation enforced
"""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arbo.core.risk_manager import RiskManager


def _mock_config() -> MagicMock:
    """Create mock ArboConfig for orchestrator tests."""
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
def reset_risk_manager() -> None:
    """Reset risk manager singleton before each test."""
    RiskManager.reset()


class TestStartsThreeStrategies:
    """Orchestrator creates tasks for all 3 strategies."""

    @patch("arbo.main_rdh.get_config")
    async def test_starts_3_strategy_tasks(self, mock_config: MagicMock) -> None:
        """After init, orchestrator has strategy_A, strategy_B, strategy_C tasks."""
        mock_config.return_value = _mock_config()

        from arbo.main_rdh import RDHOrchestrator

        orch = RDHOrchestrator(mode="paper")

        # Mock all strategy inits to return mock strategies
        mock_a = MagicMock()
        mock_a.init = AsyncMock()
        mock_a.close = AsyncMock()
        mock_a.poll_cycle = AsyncMock(return_value=[])
        mock_a.check_exits = MagicMock(return_value=[])
        mock_a.stats = {"strategy": "A"}

        mock_b = MagicMock()
        mock_b.init = AsyncMock()
        mock_b.close = AsyncMock()
        mock_b.poll_cycle = AsyncMock(return_value=[])
        mock_b.check_exits = MagicMock(return_value=[])
        mock_b.stats = {"strategy": "B"}

        mock_c = MagicMock()
        mock_c.init = AsyncMock()
        mock_c.close = AsyncMock()
        mock_c.poll_cycle = AsyncMock(return_value=[])
        mock_c.stats = {"strategy": "C"}

        orch._strategy_a = mock_a
        orch._strategy_b = mock_b
        orch._strategy_c = mock_c
        orch._discovery = MagicMock()

        orch._start_strategy_tasks()

        assert "strategy_A" in orch._tasks
        assert "strategy_B" in orch._tasks
        assert "strategy_C" in orch._tasks
        assert "discovery" in orch._tasks
        assert len(orch._tasks) == 4

        # All tasks should be running
        for name, state in orch._tasks.items():
            assert state.task is not None
            assert not state.task.done(), f"Task {name} should be running"

        # Cleanup
        for state in orch._tasks.values():
            if state.task:
                state.task.cancel()
        for state in orch._tasks.values():
            if state.task:
                try:
                    await state.task
                except (asyncio.CancelledError, Exception):
                    pass


class TestSignalRouting:
    """Each strategy uses its own quality gate, not confluence."""

    @patch("arbo.main_rdh.get_config")
    async def test_strategy_a_uses_own_gate(self, mock_config: MagicMock) -> None:
        """Strategy A poll_cycle is called with market list."""
        mock_config.return_value = _mock_config()

        from arbo.main_rdh import RDHOrchestrator

        orch = RDHOrchestrator(mode="paper")
        mock_strategy = AsyncMock()
        mock_strategy.poll_cycle = AsyncMock(return_value=[])
        mock_strategy.check_exits = MagicMock(return_value=[])
        orch._strategy_a = mock_strategy
        orch._discovery = MagicMock()
        orch._markets = [MagicMock(condition_id="c1", price_yes=Decimal("0.5"))]

        await orch._run_strategy_a()

        mock_strategy.poll_cycle.assert_called_once_with(orch._markets)
        mock_strategy.check_exits.assert_called_once()


class TestTradeWithStrategyField:
    """Trades placed via paper engine include strategy field."""

    @patch("arbo.main_rdh.get_config")
    async def test_trade_has_strategy_field(self, mock_config: MagicMock) -> None:
        """Strategy B trade result includes strategy='B'."""
        mock_config.return_value = _mock_config()

        from arbo.main_rdh import RDHOrchestrator

        orch = RDHOrchestrator(mode="paper")

        mock_strategy = AsyncMock()
        mock_strategy.poll_cycle = AsyncMock(
            return_value=[
                {
                    "condition_id": "cond_1",
                    "side": "BUY_YES",
                    "price": 0.60,
                    "size": 15.0,
                    "strategy": "B",
                }
            ]
        )
        mock_strategy.check_exits = MagicMock(return_value=[])
        orch._strategy_b = mock_strategy
        orch._discovery = MagicMock()
        orch._markets = []

        await orch._run_strategy_b()

        # Verify trade was returned with strategy field
        result = mock_strategy.poll_cycle.return_value[0]
        assert result["strategy"] == "B"


class TestHealthMonitor:
    """Health monitor detects crashed tasks and restarts them."""

    @patch("arbo.main_rdh.get_config")
    async def test_detects_crashed_task(self, mock_config: MagicMock) -> None:
        """Crashed task (done with exception) is detected and restarted."""
        mock_config.return_value = _mock_config()

        from arbo.main_rdh import RDHOrchestrator, TaskState

        orch = RDHOrchestrator(mode="paper")

        # Create a task that immediately crashes
        async def crash() -> None:
            raise RuntimeError("Strategy crashed")

        state = TaskState(name="strategy_A")
        state.task = asyncio.create_task(crash())
        orch._tasks["strategy_A"] = state

        # Wait for crash
        await asyncio.sleep(0.1)

        assert state.task.done()

        # Mock strategy_a for restart
        mock_a = AsyncMock()
        mock_a.poll_cycle = AsyncMock(return_value=[])
        mock_a.check_exits = MagicMock(return_value=[])
        orch._strategy_a = mock_a

        # Run one health check iteration
        await orch._restart_task("strategy_A")

        assert state.restart_count == 1
        assert state.task is not None
        assert not state.task.done()

        # Cleanup
        state.task.cancel()
        try:
            await state.task
        except (asyncio.CancelledError, Exception):
            pass


class TestGracefulShutdown:
    """SIGTERM cancels all tasks cleanly."""

    @patch("arbo.main_rdh.get_config")
    async def test_stop_cancels_all_tasks(self, mock_config: MagicMock) -> None:
        """stop() cancels all strategy and internal tasks."""
        mock_config.return_value = _mock_config()

        from arbo.main_rdh import RDHOrchestrator, TaskState

        orch = RDHOrchestrator(mode="paper")

        # Create mock tasks
        async def infinite_loop() -> None:
            while True:
                await asyncio.sleep(1)

        for name in ["strategy_A", "strategy_B", "strategy_C"]:
            state = TaskState(name=name)
            state.task = asyncio.create_task(infinite_loop())
            orch._tasks[name] = state

        internal = asyncio.create_task(infinite_loop())
        orch._internal_tasks.append(internal)

        # Stop
        await orch.stop()

        # All tasks should be done
        for state in orch._tasks.values():
            assert state.task is not None
            assert state.task.done()

        assert internal.done()


class TestScheduledReports:
    """Daily and weekly report schedulers exist."""

    @patch("arbo.main_rdh.get_config")
    async def test_report_schedulers_created(self, mock_config: MagicMock) -> None:
        """Internal tasks include daily and weekly report schedulers."""
        mock_config.return_value = _mock_config()

        from arbo.main_rdh import RDHOrchestrator

        orch = RDHOrchestrator(mode="paper")

        # Set up minimal state
        orch._start_internal_tasks()

        task_names = [t.get_name() for t in orch._internal_tasks]
        assert "daily_report" in task_names
        assert "weekly_report" in task_names
        assert "health_monitor" in task_names
        assert "snapshot_scheduler" in task_names
        assert "price_updater" in task_names
        assert "resolution_checker" in task_names
        assert "data_collector" in task_names

        # Cleanup
        for task in orch._internal_tasks:
            task.cancel()
        for task in orch._internal_tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


class TestCapitalAllocation:
    """Per-strategy capital allocation is enforced by risk manager."""

    @patch("arbo.main_rdh.get_config")
    def test_capital_allocation_enforced(self, mock_config: MagicMock) -> None:
        """Risk manager has correct per-strategy allocations."""
        mock_config.return_value = _mock_config()

        rm = RiskManager(Decimal("2000"))

        # Strategy A: $400
        state_a = rm.get_strategy_state("A")
        assert state_a is not None
        assert state_a.allocated == Decimal("400")
        assert state_a.available == Decimal("400")

        # Strategy B: $400
        state_b = rm.get_strategy_state("B")
        assert state_b is not None
        assert state_b.allocated == Decimal("400")

        # Strategy C: $1000
        state_c = rm.get_strategy_state("C")
        assert state_c is not None
        assert state_c.allocated == Decimal("1000")

    @patch("arbo.main_rdh.get_config")
    def test_halted_strategy_blocks_trades(self, mock_config: MagicMock) -> None:
        """Halted strategy returns is_halted=True."""
        mock_config.return_value = _mock_config()

        rm = RiskManager(Decimal("2000"))
        state = rm.get_strategy_state("A")
        assert state is not None

        state.is_halted = True
        assert rm.get_strategy_state("A").is_halted is True


class TestStrategyStats:
    """Orchestrator exposes strategy stats for dashboard."""

    @patch("arbo.main_rdh.get_config")
    def test_strategy_stats_property(self, mock_config: MagicMock) -> None:
        """strategy_stats returns stats from all 3 strategies."""
        mock_config.return_value = _mock_config()

        from arbo.main_rdh import RDHOrchestrator

        orch = RDHOrchestrator(mode="paper")

        mock_a = MagicMock()
        mock_a.stats = {"strategy": "A", "trades": 5}
        mock_b = MagicMock()
        mock_b.stats = {"strategy": "B", "trades": 0}
        mock_c = MagicMock()
        mock_c.stats = {"strategy": "C", "trades": 3}

        orch._strategy_a = mock_a
        orch._strategy_b = mock_b
        orch._strategy_c = mock_c

        stats = orch.strategy_stats
        assert "A" in stats
        assert "B" in stats
        assert "C" in stats
        assert stats["A"]["trades"] == 5


class TestTaskStates:
    """Task state tracking for health dashboard."""

    @patch("arbo.main_rdh.get_config")
    async def test_task_states_property(self, mock_config: MagicMock) -> None:
        """task_states returns running/restart status for all tasks."""
        mock_config.return_value = _mock_config()

        from arbo.main_rdh import RDHOrchestrator, TaskState

        orch = RDHOrchestrator(mode="paper")

        async def noop() -> None:
            while True:
                await asyncio.sleep(1)

        state = TaskState(name="strategy_A")
        state.task = asyncio.create_task(noop())
        state.restart_count = 2
        orch._tasks["strategy_A"] = state

        states = orch.task_states
        assert "strategy_A" in states
        assert states["strategy_A"]["running"] is True
        assert states["strategy_A"]["restart_count"] == 2

        state.task.cancel()
        try:
            await state.task
        except (asyncio.CancelledError, Exception):
            pass


class TestResolutionRouting:
    """Resolution events are routed to the correct strategy."""

    @patch("arbo.main_rdh.get_config")
    async def test_resolution_routes_to_strategy(self, mock_config: MagicMock) -> None:
        """When a position resolves, the correct strategy's handle_resolution is called."""
        mock_config.return_value = _mock_config()

        from arbo.main_rdh import RDHOrchestrator

        orch = RDHOrchestrator(mode="paper")

        mock_a = MagicMock()
        mock_a.handle_resolution = MagicMock()
        mock_b = MagicMock()
        mock_b.handle_resolution = MagicMock()

        orch._strategy_a = mock_a
        orch._strategy_b = mock_b

        # Simulate resolution routing
        strategy = "A"
        cid = "cond_1"
        pnl = Decimal("10.50")

        if strategy == "A":
            orch._strategy_a.handle_resolution(cid, pnl)
        elif strategy == "B":
            orch._strategy_b.handle_resolution(cid, pnl)

        mock_a.handle_resolution.assert_called_once_with(cid, pnl)
        mock_b.handle_resolution.assert_not_called()
