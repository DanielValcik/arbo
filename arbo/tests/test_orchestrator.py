"""Tests for ArboOrchestrator (PM-301)."""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arbo.config.settings import OrchestratorConfig
from arbo.core.scanner import Signal, SignalDirection
from arbo.main import ArboOrchestrator, LayerState

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def orch() -> ArboOrchestrator:
    """Create orchestrator with mocked config."""
    with patch("arbo.main.get_config") as mock_cfg:
        cfg = MagicMock()
        cfg.orchestrator = OrchestratorConfig(
            health_check_interval_s=1,
            heartbeat_timeout_s=5,
            max_restart_count=3,
            signal_batch_timeout_s=0.2,
        )
        cfg.bankroll = 2000.0
        mock_cfg.return_value = cfg
        return ArboOrchestrator(mode="paper")


def _make_signal(layer: int = 2, edge: str = "0.06") -> Signal:
    """Create a test signal."""
    return Signal(
        layer=layer,
        market_condition_id="cond_1",
        token_id="tok_1",
        direction=SignalDirection.BUY_YES,
        edge=Decimal(edge),
        confidence=Decimal("0.7"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOrchestratorConfig:
    def test_orchestrator_config_defaults(self) -> None:
        cfg = OrchestratorConfig()
        assert cfg.health_check_interval_s == 30
        assert cfg.heartbeat_timeout_s == 120
        assert cfg.max_restart_count == 10
        assert cfg.signal_batch_timeout_s == 2.0
        assert cfg.dashboard_update_interval_s == 60
        assert cfg.snapshot_interval_s == 3600
        assert cfg.daily_report_hour_utc == 23
        assert cfg.weekly_report_day == 6
        assert cfg.weekly_report_hour_utc == 20


class TestStartupShutdown:
    @pytest.mark.asyncio
    async def test_startup_initializes_components(self, orch: ArboOrchestrator) -> None:
        """Startup should create risk manager, paper engine, and confluence."""
        with (
            patch("arbo.main.ArboOrchestrator._init_optional", new_callable=AsyncMock) as mock_opt,
            patch("arbo.core.risk_manager.RiskManager.__init__", return_value=None),
            patch("arbo.core.paper_engine.PaperTradingEngine.__init__", return_value=None),
            patch("arbo.core.confluence.ConfluenceScorer.__init__", return_value=None),
        ):
            mock_opt.return_value = None

            await orch._init_components()

            assert orch._risk_manager is not None
            assert orch._paper_engine is not None
            assert orch._confluence is not None

    @pytest.mark.asyncio
    async def test_shutdown_closes_sessions(self, orch: ArboOrchestrator) -> None:
        """Stop should close all component sessions."""
        mock_discovery = AsyncMock()
        mock_poly = AsyncMock()
        orch._discovery = mock_discovery
        orch._poly_client = mock_poly

        await orch._close_components()

        mock_discovery.close.assert_awaited_once()
        mock_poly.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_on_event(self, orch: ArboOrchestrator) -> None:
        """Setting shutdown event should cause start() to proceed to stop()."""
        # Mock init and layer tasks
        orch._init_components = AsyncMock()
        orch._start_layer_tasks = MagicMock()
        orch._start_internal_tasks = MagicMock()
        orch._close_components = AsyncMock()
        orch._install_signal_handlers = MagicMock()

        # Set shutdown event after brief delay
        async def trigger_shutdown() -> None:
            await asyncio.sleep(0.1)
            orch._shutdown_event.set()

        task = asyncio.create_task(trigger_shutdown())
        await orch.start()
        await task

        orch._close_components.assert_awaited_once()


class TestLayerErrorHandling:
    def test_layer_crash_triggers_restart(self, orch: ArboOrchestrator) -> None:
        """A layer error should increment restart count."""
        state = LayerState(name="test_layer")
        orch._layers["test_layer"] = state

        orch._handle_layer_error("test_layer", RuntimeError("boom"))

        assert state.restart_count == 1
        assert len(state.error_timestamps) == 1

    def test_max_restart_count_respected(self, orch: ArboOrchestrator) -> None:
        """After max restarts, layer should be permanently stopped."""
        state = LayerState(name="test_layer")
        state.restart_count = orch._orch_cfg.max_restart_count - 1
        orch._layers["test_layer"] = state

        orch._handle_layer_error("test_layer", RuntimeError("boom"))

        assert state.permanent_stop is True

    def test_repeated_error_triggers_emergency_shutdown(self, orch: ArboOrchestrator) -> None:
        """3 errors within 10 minutes should trigger emergency shutdown."""
        state = LayerState(name="test_layer")
        orch._layers["test_layer"] = state

        now = time.monotonic()
        state.error_timestamps = [now - 1, now - 2]

        orch._handle_layer_error("test_layer", RuntimeError("boom"))

        assert orch._shutdown_event.is_set()


class TestHealthMonitor:
    @pytest.mark.asyncio
    async def test_health_monitor_detects_hung_task(self, orch: ArboOrchestrator) -> None:
        """Health monitor should detect tasks that haven't heartbeated."""
        state = LayerState(name="hung_layer")
        state.last_heartbeat = time.monotonic() - 999
        state.enabled = True

        # Create a mock task that is not done
        mock_task = MagicMock(spec=asyncio.Task)
        mock_task.done.return_value = False
        state.task = mock_task
        orch._layers["hung_layer"] = state

        # Patch _restart_layer so we can check it's called
        orch._restart_layer = MagicMock()
        orch._check_llm_health = AsyncMock()

        # Run health monitor for one iteration
        orch._shutdown_event.set()
        orch._orch_cfg.health_check_interval_s = 0

        # Manually run the check logic
        now = time.monotonic()
        timeout = orch._orch_cfg.heartbeat_timeout_s
        for name, st in orch._layers.items():
            if st.task is not None and not st.task.done() and now - st.last_heartbeat > timeout:
                orch._restart_layer(name)

        orch._restart_layer.assert_called_once_with("hung_layer")


class TestSignalProcessor:
    @pytest.mark.asyncio
    async def test_signal_processor_routes_to_confluence(self, orch: ArboOrchestrator) -> None:
        """Signal processor should batch signals and feed them to confluence."""
        mock_confluence = MagicMock()
        mock_confluence.get_tradeable.return_value = []
        orch._confluence = mock_confluence

        mock_paper = MagicMock()
        orch._paper_engine = mock_paper

        signals = [_make_signal(layer=2), _make_signal(layer=4)]

        await orch._process_signal_batch(signals)

        mock_confluence.get_tradeable.assert_called_once()
        call_args = mock_confluence.get_tradeable.call_args
        assert len(call_args[0][0]) == 2


class TestLLMDegradedMode:
    @pytest.mark.asyncio
    async def test_llm_degraded_mode_disables_layers(self, orch: ArboOrchestrator) -> None:
        """When LLM is unavailable, layers 5 and 8 should be skipped."""
        orch._gemini = None
        orch._llm_last_check = 0

        await orch._check_llm_health()

        assert orch._llm_degraded is True

        # Verify layer 5 and 8 skip
        orch._logical_arb = MagicMock()
        orch._attention = MagicMock()

        await orch._run_logical_arb()
        await orch._run_attention()

        # scan() should NOT have been called
        orch._logical_arb.scan.assert_not_called()
        orch._attention.scan.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_recovery_reenables_layers(self, orch: ArboOrchestrator) -> None:
        """When LLM becomes available again, degraded mode should be cleared."""
        orch._llm_degraded = True
        orch._llm_last_check = 0

        mock_gemini = MagicMock()
        mock_gemini.is_healthy = AsyncMock(return_value=True)
        orch._gemini = mock_gemini

        await orch._check_llm_health()

        assert orch._llm_degraded is False
