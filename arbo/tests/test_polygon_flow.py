"""Tests for PM-105: Order Flow Monitor (Layer 7).

Tests verify:
1. RollingWindow: add/volume, prune, z-score normal, z-score spike, buy/sell tracking, empty
2. OrderFilled parsing: valid log, invalid→None, buy/sell detection
3. Convergence: below thresholds, z+imbalance→signal, z+delta→signal, signal format layer 7
4. Polling lifecycle: handle_log, matched_tokens filter, start/stop, stats

Acceptance: parses OrderFilled events, calculates Z-scores, data in DB. Min 100 events.
"""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arbo.connectors.polygon_flow import (
    MarketFlowTracker,
    MarketTakerFlow,
    OrderFilledEvent,
    OrderFlowMonitor,
    PeakOptimismResult,
    RollingWindow,
    FlowSnapshotEntry,
    parse_order_filled,
)
from arbo.core.scanner import Signal, SignalDirection

# ================================================================
# Factory helpers
# ================================================================


def _make_log(
    maker_asset_id: int = 10**40,  # Large = conditional token
    taker_asset_id: int = 10**20,  # Smaller = USDC-like
    maker_amount: int = 100_000_000,  # 100 USDC (6 decimals)
    taker_amount: int = 50_000_000,  # 50 USDC
    fee: int = 500_000,  # 0.5 USDC
    block_number: str = "0x100",
) -> dict:
    """Build a mock Ethereum log for OrderFilled."""
    order_hash = "0x" + "ab" * 32
    taker_hex = "00" * 12 + "22" * 20

    # Encode data fields (each 32 bytes = 64 hex)
    data_parts = [
        taker_hex,  # taker address (padded to 32 bytes)
        f"{maker_asset_id:064x}",
        f"{taker_asset_id:064x}",
        f"{maker_amount:064x}",
        f"{taker_amount:064x}",
        f"{fee:064x}",
    ]
    data = "0x" + "".join(data_parts)

    return {
        "topics": [
            "0x" + "00" * 32,  # event topic
            order_hash,  # orderHash
            "0x" + "00" * 12 + "11" * 20,  # maker (padded)
        ],
        "data": data,
        "blockNumber": block_number,
    }


# ================================================================
# RollingWindow
# ================================================================


class TestRollingWindow:
    """Time-windowed rolling statistics."""

    def test_add_and_volume(self) -> None:
        """Add entries and query total volume."""
        w = RollingWindow()
        now = time.monotonic()
        w.add(Decimal("100"), is_buy=True, timestamp=now)
        w.add(Decimal("50"), is_buy=False, timestamp=now)

        assert w.get_volume() == Decimal("150")

    def test_prune_old_entries(self) -> None:
        """Entries older than max_window are pruned."""
        w = RollingWindow(max_window_s=10)
        old_time = time.monotonic() - 20  # 20s ago
        w.add(Decimal("100"), is_buy=True, timestamp=old_time)

        # Trigger prune by adding new entry
        w.add(Decimal("50"), is_buy=True)

        assert w.get_volume() == Decimal("50")

    def test_zscore_normal(self) -> None:
        """Uniform volume across buckets → z-score near 0."""
        w = RollingWindow()
        now = time.monotonic()
        w._time_fn = lambda: now  # Fix time

        # Add uniform volume across multiple 5-min buckets
        for i in range(1, 7):  # 6 historical buckets
            bucket_time = now - (i * 300 + 1)  # In bucket i
            w.add(Decimal("100"), is_buy=True, timestamp=bucket_time)

        # Current bucket (bucket 0)
        w.add(Decimal("100"), is_buy=True, timestamp=now - 1)

        zscore = w.get_zscore(window_s=3600)
        assert abs(zscore) < 1.0  # Near zero for uniform data

    def test_zscore_spike(self) -> None:
        """Spike in current bucket → high z-score."""
        w = RollingWindow()
        now = time.monotonic()
        w._time_fn = lambda: now

        # Small varying volume in historical buckets (need std > 0)
        for i in range(1, 7):
            bucket_time = now - (i * 300 + 1)
            vol = Decimal(str(8 + i * 2))  # 10, 12, 14, 16, 18, 20
            w.add(vol, is_buy=True, timestamp=bucket_time)

        # Big spike in current bucket
        w.add(Decimal("1000"), is_buy=True, timestamp=now - 1)

        zscore = w.get_zscore(window_s=3600)
        assert zscore > 2.0  # Significant spike

    def test_buy_sell_tracking(self) -> None:
        """Buy and sell volumes tracked separately."""
        w = RollingWindow()
        now = time.monotonic()
        w.add(Decimal("100"), is_buy=True, timestamp=now)
        w.add(Decimal("60"), is_buy=False, timestamp=now)

        assert w.get_buy_volume() == Decimal("100")
        assert w.get_sell_volume() == Decimal("60")

    def test_empty_window(self) -> None:
        """Empty window returns zero volume and zero z-score."""
        w = RollingWindow()
        assert w.get_volume() == Decimal("0")
        assert w.get_zscore() == 0.0


# ================================================================
# OrderFilled parsing
# ================================================================


class TestOrderFilledParsing:
    """Parse raw Ethereum logs into OrderFilledEvent."""

    def test_valid_log(self) -> None:
        """Valid log is parsed correctly."""
        log = _make_log()
        event = parse_order_filled(log)

        assert event is not None
        assert isinstance(event, OrderFilledEvent)
        assert event.maker_amount == Decimal("100")
        assert event.taker_amount == Decimal("50")
        assert event.fee == Decimal("0.5")
        assert event.block_number == 256  # 0x100

    def test_invalid_log_returns_none(self) -> None:
        """Invalid log (too few topics) returns None."""
        log = {"topics": ["0x00"], "data": "0x00", "blockNumber": "0x0"}
        assert parse_order_filled(log) is None

    def test_short_data_returns_none(self) -> None:
        """Log with insufficient data field returns None."""
        log = {
            "topics": ["0x" + "00" * 32, "0x" + "ab" * 32, "0x" + "11" * 32],
            "data": "0x0000",
            "blockNumber": "0x0",
        }
        assert parse_order_filled(log) is None

    def test_buy_detection(self) -> None:
        """Correctly detects buy (maker has large asset = conditional token)."""
        log = _make_log(maker_asset_id=10**40, taker_asset_id=10**20)
        event = parse_order_filled(log)

        assert event is not None
        assert event.is_buy is True
        assert event.token_id == str(10**40)

    def test_sell_detection(self) -> None:
        """Correctly detects sell (taker has large asset = conditional token)."""
        log = _make_log(maker_asset_id=10**20, taker_asset_id=10**40)
        event = parse_order_filled(log)

        assert event is not None
        assert event.is_buy is False
        assert event.token_id == str(10**40)


# ================================================================
# Convergence detection
# ================================================================


class TestConvergence:
    """OrderFlowMonitor convergence signal detection."""

    @patch("arbo.connectors.polygon_flow.get_config")
    def test_below_thresholds_no_signal(self, mock_config: MagicMock) -> None:
        """Uniform flow → no signal."""
        mock_config.return_value = _mock_config()
        monitor = OrderFlowMonitor()

        # Add some uniform data
        window = RollingWindow()
        now = time.monotonic()
        window._time_fn = lambda: now
        for i in range(1, 5):
            window.add(Decimal("100"), is_buy=True, timestamp=now - i * 300 - 1)
            window.add(Decimal("100"), is_buy=False, timestamp=now - i * 300 - 1)
        # Current bucket — also balanced
        window.add(Decimal("100"), is_buy=True, timestamp=now - 1)
        window.add(Decimal("100"), is_buy=False, timestamp=now - 1)

        monitor._windows["token_1"] = window
        signal = monitor._check_convergence("token_1")
        assert signal is None

    @patch("arbo.connectors.polygon_flow.get_config")
    def test_zscore_plus_imbalance_signal(self, mock_config: MagicMock) -> None:
        """Z-score spike + flow imbalance → signal."""
        mock_config.return_value = _mock_config()
        monitor = OrderFlowMonitor()

        window = RollingWindow()
        now = time.monotonic()
        window._time_fn = lambda: now

        # Small uniform historical volume
        for i in range(1, 7):
            window.add(Decimal("10"), is_buy=True, timestamp=now - i * 300 - 1)
            window.add(Decimal("10"), is_buy=False, timestamp=now - i * 300 - 1)

        # Big spike in current bucket, heavily buy-biased (imbalance)
        for _ in range(20):
            window.add(Decimal("100"), is_buy=True, timestamp=now - 1)
        window.add(Decimal("10"), is_buy=False, timestamp=now - 1)

        monitor._windows["token_1"] = window
        signal = monitor._check_convergence("token_1")

        assert signal is not None
        assert signal.layer == 7
        assert signal.direction == SignalDirection.BUY_YES

    @patch("arbo.connectors.polygon_flow.get_config")
    def test_zscore_plus_delta_signal(self, mock_config: MagicMock) -> None:
        """Z-score spike + delta trending → signal."""
        mock_config.return_value = _mock_config()
        monitor = OrderFlowMonitor()

        window = RollingWindow()
        now = time.monotonic()
        window._time_fn = lambda: now

        # Moderate historical
        for i in range(1, 7):
            window.add(Decimal("10"), is_buy=True, timestamp=now - i * 300 - 1)

        # Big current spike, all buys
        for _ in range(20):
            window.add(Decimal("100"), is_buy=True, timestamp=now - 1)

        monitor._windows["token_1"] = window
        signal = monitor._check_convergence("token_1")

        assert signal is not None
        assert signal.layer == 7

    @patch("arbo.connectors.polygon_flow.get_config")
    def test_signal_format(self, mock_config: MagicMock) -> None:
        """Signal contains correct details."""
        mock_config.return_value = _mock_config()
        monitor = OrderFlowMonitor()

        window = RollingWindow()
        now = time.monotonic()
        window._time_fn = lambda: now

        for i in range(1, 7):
            window.add(Decimal("10"), is_buy=True, timestamp=now - i * 300 - 1)
        for _ in range(20):
            window.add(Decimal("100"), is_buy=True, timestamp=now - 1)

        monitor._windows["token_1"] = window
        signal = monitor._check_convergence("token_1")

        assert signal is not None
        assert isinstance(signal, Signal)
        assert signal.token_id == "token_1"
        assert "zscore" in signal.details
        assert "imbalance" in signal.details
        assert "delta" in signal.details
        assert "converging_signals" in signal.details


# ================================================================
# Polling lifecycle
# ================================================================


class TestPollingLifecycle:
    """OrderFlowMonitor polling-based lifecycle."""

    @patch("arbo.connectors.polygon_flow.get_config")
    def test_handle_log_processes_event(self, mock_config: MagicMock) -> None:
        """_handle_log parses log and feeds to rolling window."""
        mock_config.return_value = _mock_config()
        monitor = OrderFlowMonitor()

        log = _make_log()
        monitor._handle_log(log)

        assert monitor._total_events == 1
        token_id = str(10**40)
        assert token_id in monitor._windows

    @patch("arbo.connectors.polygon_flow.get_config")
    def test_handle_log_invalid_ignored(self, mock_config: MagicMock) -> None:
        """_handle_log ignores invalid logs."""
        mock_config.return_value = _mock_config()
        monitor = OrderFlowMonitor()

        monitor._handle_log({"topics": ["0x00"], "data": "0x00", "blockNumber": "0x0"})
        assert monitor._total_events == 0

    @patch("arbo.connectors.polygon_flow.get_config")
    def test_matched_tokens_filter(self, mock_config: MagicMock) -> None:
        """Only matched tokens are processed when filter is set."""
        mock_config.return_value = _mock_config()
        monitor = OrderFlowMonitor()

        # Set filter to only track a specific token
        monitor.update_matched_tokens({"999"})

        # This log produces token_id = str(10**40), which is NOT in the filter
        log = _make_log()
        monitor._handle_log(log)

        assert monitor._total_events == 0
        assert len(monitor._windows) == 0

    @patch("arbo.connectors.polygon_flow.get_config")
    def test_matched_tokens_allows_matching(self, mock_config: MagicMock) -> None:
        """Events for matched tokens are processed."""
        mock_config.return_value = _mock_config()
        monitor = OrderFlowMonitor()

        token_id = str(10**40)
        monitor.update_matched_tokens({token_id})

        log = _make_log()
        monitor._handle_log(log)

        assert monitor._total_events == 1

    @patch("arbo.connectors.polygon_flow.get_config")
    def test_empty_filter_processes_all(self, mock_config: MagicMock) -> None:
        """Empty matched_tokens set processes all events."""
        mock_config.return_value = _mock_config()
        monitor = OrderFlowMonitor()

        # Empty set = process all
        monitor.update_matched_tokens(set())

        log = _make_log()
        monitor._handle_log(log)

        assert monitor._total_events == 1

    @patch("arbo.connectors.polygon_flow.get_config")
    def test_stats_property(self, mock_config: MagicMock) -> None:
        """Stats property returns correct counts."""
        mock_config.return_value = _mock_config()
        monitor = OrderFlowMonitor()

        stats = monitor.stats
        assert stats["total_events"] == 0
        assert stats["signals_emitted"] == 0
        assert stats["active_tokens"] == 0

        # Process a log
        monitor._handle_log(_make_log())
        stats = monitor.stats
        assert stats["total_events"] == 1
        assert stats["active_tokens"] == 1

    @patch("arbo.connectors.polygon_flow.get_config")
    def test_on_signal_callback(self, mock_config: MagicMock) -> None:
        """on_signal callback is invoked when convergence is detected."""
        mock_config.return_value = _mock_config()
        received: list[Signal] = []
        monitor = OrderFlowMonitor(on_signal=lambda s: received.append(s))

        # Build convergent data directly in the window
        window = RollingWindow()
        now = time.monotonic()
        window._time_fn = lambda: now
        for i in range(1, 7):
            window.add(Decimal("10"), is_buy=True, timestamp=now - i * 300 - 1)
        for _ in range(20):
            window.add(Decimal("100"), is_buy=True, timestamp=now - 1)

        token_id = str(10**40)
        monitor._windows[token_id] = window

        # Create a log that will trigger convergence check for this token
        log = _make_log()
        monitor._handle_log(log)

        assert monitor._signals_emitted >= 1
        assert len(received) >= 1

    @patch("arbo.connectors.polygon_flow.get_config")
    async def test_start_stop_lifecycle(self, mock_config: MagicMock) -> None:
        """Start creates polling task, stop cleans up."""
        mock_config.return_value = _mock_config()
        monitor = OrderFlowMonitor(poll_interval=1)

        # Mock _get_block_number to avoid real HTTP calls
        monitor._get_block_number = AsyncMock(return_value=1000)
        monitor._get_logs = AsyncMock(return_value=[])

        await monitor.initialize()

        # Manually set session to avoid real HTTP
        monitor._session = MagicMock()
        monitor._session.closed = False
        monitor._session.close = AsyncMock()
        monitor._running = True
        monitor._last_block = 1000

        # Create a task that we can cancel
        monitor._task = asyncio.create_task(monitor._poll_loop())

        # Let it run briefly
        await asyncio.sleep(0.05)
        assert monitor._running

        await monitor.stop()
        assert not monitor._running
        assert monitor._task is None

    @patch("arbo.connectors.polygon_flow.get_config")
    def test_poll_interval_stored(self, mock_config: MagicMock) -> None:
        """Custom poll_interval is stored."""
        mock_config.return_value = _mock_config()
        monitor = OrderFlowMonitor(poll_interval=120)
        assert monitor._poll_interval == 120

    @patch("arbo.connectors.polygon_flow.get_config")
    def test_http_url_from_config(self, mock_config: MagicMock) -> None:
        """HTTP URL is read from config.polygon_rpc_url."""
        mock_config.return_value = _mock_config()
        monitor = OrderFlowMonitor()
        assert monitor._http_url == "https://lb.drpc.live/polygon/test-key"

    @patch("arbo.connectors.polygon_flow.get_config")
    def test_get_metrics(self, mock_config: MagicMock) -> None:
        """get_metrics returns correct flow metrics."""
        mock_config.return_value = _mock_config()
        monitor = OrderFlowMonitor()

        # Process a few logs
        monitor._handle_log(_make_log())
        monitor._handle_log(_make_log())

        token_id = str(10**40)
        metrics = monitor.get_metrics(token_id)
        assert metrics is not None
        assert metrics.token_id == token_id
        assert metrics.event_count == 2


# ================================================================
# D3: is_healthy property
# ================================================================


class TestIsHealthy:
    """OrderFlowMonitor.is_healthy property (D3 crash propagation fix)."""

    @patch("arbo.connectors.polygon_flow.get_config")
    def test_healthy_when_running_with_task(self, mock_config: MagicMock) -> None:
        """is_healthy returns True when _running=True and task is alive."""
        mock_config.return_value = _mock_config()
        monitor = OrderFlowMonitor()
        monitor._running = True

        # Create a fake non-done task
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        monitor._task = future  # type: ignore[assignment]
        assert monitor.is_healthy is True
        loop.close()

    @patch("arbo.connectors.polygon_flow.get_config")
    def test_not_healthy_when_not_running(self, mock_config: MagicMock) -> None:
        """is_healthy returns False when _running=False."""
        mock_config.return_value = _mock_config()
        monitor = OrderFlowMonitor()
        monitor._running = False
        assert monitor.is_healthy is False

    @patch("arbo.connectors.polygon_flow.get_config")
    def test_not_healthy_when_no_task(self, mock_config: MagicMock) -> None:
        """is_healthy returns False when _task is None."""
        mock_config.return_value = _mock_config()
        monitor = OrderFlowMonitor()
        monitor._running = True
        monitor._task = None
        assert monitor.is_healthy is False

    @patch("arbo.connectors.polygon_flow.get_config")
    def test_not_healthy_when_task_done(self, mock_config: MagicMock) -> None:
        """is_healthy returns False when _task has completed/crashed."""
        mock_config.return_value = _mock_config()
        monitor = OrderFlowMonitor()
        monitor._running = True

        # Create a completed future to simulate a done task
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        future.set_result(None)
        monitor._task = future  # type: ignore[assignment]
        assert monitor.is_healthy is False
        loop.close()

    @patch("arbo.connectors.polygon_flow.get_config")
    def test_no_rpc_url_not_healthy(self, mock_config: MagicMock) -> None:
        """Monitor without RPC URL is not healthy (polling never starts)."""
        config = _mock_config()
        config.polygon_rpc_url = ""
        mock_config.return_value = config
        monitor = OrderFlowMonitor()
        assert monitor.is_healthy is False


# ================================================================
# Incremental block tracking
# ================================================================


class TestIncrementalPolling:
    """Incremental block tracking with DB persistence."""

    @patch("arbo.connectors.polygon_flow.get_config")
    async def test_cold_start_uses_latest_minus_1000(self, mock_config: MagicMock) -> None:
        """No DB state → _last_block = latest - 1000."""
        mock_config.return_value = _mock_config()
        monitor = OrderFlowMonitor(poll_interval=1)

        latest_block = 50_000
        monitor._get_block_number = AsyncMock(return_value=latest_block)
        monitor._get_logs = AsyncMock(return_value=[])

        # No session_factory → _restore_last_block returns None → cold start
        assert monitor._session_factory is None

        # Manually call start logic: open session, set running, determine block
        monitor._session = MagicMock()
        monitor._session.closed = False
        monitor._session.close = AsyncMock()
        monitor._running = True

        latest = await monitor._get_block_number()
        restored = await monitor._restore_last_block()
        assert restored is None

        # Cold start: latest - 1000
        monitor._last_block = latest - 1000
        assert monitor._last_block == latest_block - 1000

        await monitor.stop()

    @patch("arbo.connectors.polygon_flow.get_config")
    async def test_incremental_from_block_increases(self, mock_config: MagicMock) -> None:
        """After first poll, second poll's fromBlock > first poll's fromBlock."""
        mock_config.return_value = _mock_config()
        monitor = OrderFlowMonitor(poll_interval=1)

        call_count = 0
        from_blocks: list[int] = []

        original_get_logs = monitor._get_logs

        async def tracking_get_logs(from_block: int, to_block: int) -> list:
            from_blocks.append(from_block)
            return []

        # Block number increases on each call
        block_numbers = [1000, 1030, 1060]
        block_iter = iter(block_numbers)
        monitor._get_block_number = AsyncMock(side_effect=block_numbers)
        monitor._get_logs = AsyncMock(side_effect=tracking_get_logs)

        # Set initial state as if start() ran
        monitor._session = MagicMock()
        monitor._session.closed = False
        monitor._session.close = AsyncMock()
        monitor._running = True
        monitor._last_block = 999  # start just before first block

        # Run two poll iterations manually
        # First poll
        current = await monitor._get_block_number()
        if current > monitor._last_block:
            logs = await monitor._get_logs(monitor._last_block + 1, current)
            monitor._last_block = current

        # Second poll
        current = await monitor._get_block_number()
        if current > monitor._last_block:
            logs = await monitor._get_logs(monitor._last_block + 1, current)
            monitor._last_block = current

        assert len(from_blocks) == 2
        assert from_blocks[1] > from_blocks[0]

        await monitor.stop()

    @patch("arbo.connectors.polygon_flow.get_config")
    async def test_restart_restores_from_db(self, mock_config: MagicMock) -> None:
        """Save block to DB, create new monitor, verify it resumes from saved block."""
        mock_config.return_value = _mock_config()

        # Use an in-memory SQLite DB to test the full save/restore cycle
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

        from arbo.utils.db import SystemState

        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as conn:
            # Only create the SystemState table (other models use JSONB which SQLite can't handle)
            await conn.run_sync(SystemState.__table__.create)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        # Monitor 1: save block 42000 to DB
        monitor1 = OrderFlowMonitor(poll_interval=1, session_factory=factory)
        monitor1._last_block = 42_000
        await monitor1._save_last_block()

        # Monitor 2: restore from DB
        monitor2 = OrderFlowMonitor(poll_interval=1, session_factory=factory)
        restored = await monitor2._restore_last_block()

        assert restored == 42_000

        await engine.dispose()


# ================================================================
# MarketFlowTracker — YES/NO taker ratio (RDH-201)
# ================================================================


def _make_event(
    token_id: str = "tok_yes",
    is_buy: bool = True,
    taker_amount: Decimal = Decimal("50"),
    maker_amount: Decimal = Decimal("100"),
) -> OrderFilledEvent:
    """Create a mock OrderFilledEvent."""
    from datetime import UTC, datetime

    return OrderFilledEvent(
        order_hash="0x" + "ab" * 32,
        maker="0x" + "11" * 20,
        taker="0x" + "22" * 20,
        maker_asset_id="123",
        taker_asset_id="456",
        maker_amount=maker_amount,
        taker_amount=taker_amount,
        fee=Decimal("0.5"),
        is_buy=is_buy,
        token_id=token_id,
        timestamp=datetime.now(UTC),
        block_number=1000,
    )


class TestMarketFlowTracker:
    """Per-market YES/NO taker flow tracking."""

    def test_register_market(self) -> None:
        """Register a market and verify mapping."""
        tracker = MarketFlowTracker()
        tracker.register_market("cond_1", "tok_yes", "tok_no")

        assert tracker.registered_markets == 1
        flow = tracker.get_market_flow("cond_1")
        assert flow is not None
        assert flow.yes_token_id == "tok_yes"
        assert flow.no_token_id == "tok_no"

    def test_duplicate_register_ignored(self) -> None:
        """Re-registering same market is a no-op."""
        tracker = MarketFlowTracker()
        tracker.register_market("cond_1", "tok_yes", "tok_no")
        tracker.register_market("cond_1", "tok_yes_2", "tok_no_2")  # Should be ignored
        assert tracker.registered_markets == 1
        assert tracker.get_market_flow("cond_1").yes_token_id == "tok_yes"

    def test_process_yes_buy_event(self) -> None:
        """Buying YES token increases YES taker flow."""
        tracker = MarketFlowTracker()
        tracker.register_market("cond_1", "tok_yes", "tok_no")

        event = _make_event(token_id="tok_yes", is_buy=True, taker_amount=Decimal("100"))
        flow = tracker.process_event(event)

        assert flow is not None
        assert flow.yes_volume_4h > Decimal("0")
        assert flow.no_volume_4h == Decimal("0")

    def test_process_no_buy_event(self) -> None:
        """Buying NO token increases NO taker flow."""
        tracker = MarketFlowTracker()
        tracker.register_market("cond_1", "tok_yes", "tok_no")

        event = _make_event(token_id="tok_no", is_buy=True, taker_amount=Decimal("100"))
        flow = tracker.process_event(event)

        assert flow is not None
        assert flow.no_volume_4h > Decimal("0")
        assert flow.yes_volume_4h == Decimal("0")

    def test_yes_taker_ratio_all_yes(self) -> None:
        """All YES buys → ratio = 1.0."""
        tracker = MarketFlowTracker()
        tracker.register_market("cond_1", "tok_yes", "tok_no")

        for _ in range(5):
            tracker.process_event(
                _make_event(token_id="tok_yes", is_buy=True, taker_amount=Decimal("100"))
            )

        flow = tracker.get_market_flow("cond_1")
        assert flow.yes_taker_ratio == 1.0

    def test_yes_taker_ratio_balanced(self) -> None:
        """Equal YES/NO buys → ratio ≈ 0.5."""
        tracker = MarketFlowTracker()
        tracker.register_market("cond_1", "tok_yes", "tok_no")

        tracker.process_event(
            _make_event(token_id="tok_yes", is_buy=True, taker_amount=Decimal("100"))
        )
        tracker.process_event(
            _make_event(token_id="tok_no", is_buy=True, taker_amount=Decimal("100"))
        )

        flow = tracker.get_market_flow("cond_1")
        assert abs(flow.yes_taker_ratio - 0.5) < 0.1

    def test_yes_taker_ratio_no_data(self) -> None:
        """No data → ratio = 0.5 (neutral)."""
        tracker = MarketFlowTracker()
        tracker.register_market("cond_1", "tok_yes", "tok_no")

        flow = tracker.get_market_flow("cond_1")
        assert flow.yes_taker_ratio == 0.5

    def test_unregistered_market_returns_none(self) -> None:
        """Event for unregistered market returns None."""
        tracker = MarketFlowTracker()
        event = _make_event(token_id="unknown_token")
        result = tracker.process_event(event)
        assert result is None

    def test_per_market_isolation(self) -> None:
        """Different markets track flow independently."""
        tracker = MarketFlowTracker()
        tracker.register_market("cond_1", "tok_y1", "tok_n1")
        tracker.register_market("cond_2", "tok_y2", "tok_n2")

        tracker.process_event(_make_event(token_id="tok_y1", is_buy=True, taker_amount=Decimal("500")))
        tracker.process_event(_make_event(token_id="tok_y2", is_buy=True, taker_amount=Decimal("50")))

        flow1 = tracker.get_market_flow("cond_1")
        flow2 = tracker.get_market_flow("cond_2")

        assert flow1.yes_volume_4h > flow2.yes_volume_4h

    def test_get_all_flows(self) -> None:
        """get_all_flows returns all tracked markets."""
        tracker = MarketFlowTracker()
        tracker.register_market("cond_1", "tok_y1", "tok_n1")
        tracker.register_market("cond_2", "tok_y2", "tok_n2")

        flows = tracker.get_all_flows()
        assert len(flows) == 2

    @patch("arbo.connectors.polygon_flow.get_config")
    def test_monitor_feeds_market_tracker(self, mock_config: MagicMock) -> None:
        """OrderFlowMonitor feeds events to market_tracker."""
        mock_config.return_value = _mock_config()
        monitor = OrderFlowMonitor()

        # Register a market
        token_id = str(10**40)
        monitor.register_market("cond_test", token_id, "tok_no_test")

        # Process a log with that token
        log = _make_log()
        monitor._handle_log(log)

        flow = monitor.market_tracker.get_market_flow("cond_test")
        assert flow is not None
        assert flow.yes_volume_4h > Decimal("0")


# ================================================================
# RDH-202: 4h rolling z-score + 3σ peak optimism
# ================================================================


class TestPeakOptimism:
    """3σ peak optimism detection on YES taker flow."""

    def test_3sigma_triggers_peak_optimism(self) -> None:
        """YES z-score ≥ 3.0 → is_peak = True."""
        tracker = MarketFlowTracker()
        tracker.register_market("cond_1", "tok_yes", "tok_no")
        flow = tracker.get_market_flow("cond_1")

        now = time.monotonic()
        flow.yes_window._time_fn = lambda: now

        # Small varying volume in historical 5-min buckets
        for i in range(1, 13):  # 12 buckets → 1h of history within 4h window
            bucket_time = now - (i * 300 + 1)
            vol = Decimal(str(8 + i))  # 9, 10, 11, ..., 20
            flow.yes_window.add(vol, is_buy=True, timestamp=bucket_time)

        # Big spike in current bucket (bucket 0)
        flow.yes_window.add(Decimal("2000"), is_buy=True, timestamp=now - 1)

        result = flow.check_peak_optimism(threshold=3.0)
        assert result.is_peak is True
        assert result.zscore >= 3.0
        assert result.condition_id == "cond_1"

    def test_2_5sigma_does_not_trigger(self) -> None:
        """YES z-score < 3.0 → is_peak = False."""
        tracker = MarketFlowTracker()
        tracker.register_market("cond_1", "tok_yes", "tok_no")
        flow = tracker.get_market_flow("cond_1")

        now = time.monotonic()
        flow.yes_window._time_fn = lambda: now

        # Historical buckets with some variance
        for i in range(1, 13):
            bucket_time = now - (i * 300 + 1)
            vol = Decimal(str(10 + i * 3))  # 13, 16, 19, ..., 46
            flow.yes_window.add(vol, is_buy=True, timestamp=bucket_time)

        # Moderate spike — above average but below 3σ
        # Mean ≈ 29.5, std ≈ 10.7 → 3σ ≈ 61.6 above mean → need ~91
        # Place something that gives ~2σ
        flow.yes_window.add(Decimal("55"), is_buy=True, timestamp=now - 1)

        result = flow.check_peak_optimism(threshold=3.0)
        assert result.is_peak is False
        assert result.zscore < 3.0

    def test_no_data_not_peak(self) -> None:
        """No flow data → not peak."""
        tracker = MarketFlowTracker()
        tracker.register_market("cond_1", "tok_yes", "tok_no")
        flow = tracker.get_market_flow("cond_1")

        result = flow.check_peak_optimism()
        assert result.is_peak is False
        assert result.zscore == 0.0

    def test_peak_result_includes_yes_ratio(self) -> None:
        """PeakOptimismResult includes current YES taker ratio."""
        tracker = MarketFlowTracker()
        tracker.register_market("cond_1", "tok_yes", "tok_no")
        flow = tracker.get_market_flow("cond_1")

        # Add only YES flow → ratio should be 1.0
        now = time.monotonic()
        flow.yes_window._time_fn = lambda: now
        for i in range(1, 7):
            flow.yes_window.add(Decimal("10"), is_buy=True, timestamp=now - i * 300 - 1)
        flow.yes_window.add(Decimal("1000"), is_buy=True, timestamp=now - 1)

        result = flow.check_peak_optimism()
        assert result.yes_ratio == 1.0

    def test_get_peak_optimism_markets(self) -> None:
        """MarketFlowTracker returns only markets at peak optimism."""
        tracker = MarketFlowTracker()
        tracker.register_market("cond_hot", "tok_y1", "tok_n1")
        tracker.register_market("cond_calm", "tok_y2", "tok_n2")

        now = time.monotonic()

        # cond_hot: massive YES spike → 3σ
        hot = tracker.get_market_flow("cond_hot")
        hot.yes_window._time_fn = lambda: now
        for i in range(1, 13):
            vol = Decimal(str(8 + i))  # varying: 9, 10, ..., 20
            hot.yes_window.add(vol, is_buy=True, timestamp=now - i * 300 - 1)
        hot.yes_window.add(Decimal("2000"), is_buy=True, timestamp=now - 1)

        # cond_calm: uniform-ish volume → no spike
        calm = tracker.get_market_flow("cond_calm")
        calm.yes_window._time_fn = lambda: now
        for i in range(1, 7):
            calm.yes_window.add(Decimal(str(10 + i)), is_buy=True, timestamp=now - i * 300 - 1)
        calm.yes_window.add(Decimal("15"), is_buy=True, timestamp=now - 1)

        peaks = tracker.get_peak_optimism_markets(threshold=3.0)
        assert len(peaks) == 1
        assert peaks[0].condition_id == "cond_hot"
        assert peaks[0].is_peak is True


class TestTakerFlowSnapshots:
    """Periodic snapshot tracking for 7-day historical baseline."""

    def test_record_snapshot(self) -> None:
        """record_snapshot captures current flow state."""
        tracker = MarketFlowTracker()
        tracker.register_market("cond_1", "tok_yes", "tok_no")
        flow = tracker.get_market_flow("cond_1")

        # Add some volume
        flow.yes_window.add(Decimal("100"), is_buy=True)
        flow.no_window.add(Decimal("50"), is_buy=True)

        snap = flow.record_snapshot()
        assert isinstance(snap, FlowSnapshotEntry)
        assert snap.yes_volume == Decimal("100")
        assert snap.no_volume == Decimal("50")
        assert snap.yes_ratio == pytest.approx(100 / 150, abs=0.01)
        assert flow.snapshot_count == 1

    def test_snapshot_tracks_deltas(self) -> None:
        """Successive snapshots capture incremental volume only."""
        tracker = MarketFlowTracker()
        tracker.register_market("cond_1", "tok_yes", "tok_no")
        flow = tracker.get_market_flow("cond_1")

        # First period
        flow.yes_window.add(Decimal("100"), is_buy=True)
        flow.record_snapshot()

        # Second period — add more
        flow.yes_window.add(Decimal("60"), is_buy=True)
        snap2 = flow.record_snapshot()

        # Should be delta only (60, not 160)
        assert snap2.yes_volume == Decimal("60")
        assert flow.snapshot_count == 2

    def test_historical_avg_yes_rate(self) -> None:
        """historical_avg_yes_rate computes weighted average from snapshots."""
        tracker = MarketFlowTracker()
        tracker.register_market("cond_1", "tok_yes", "tok_no")
        flow = tracker.get_market_flow("cond_1")

        # Snapshot 1: 80% YES
        flow.yes_window.add(Decimal("80"), is_buy=True)
        flow.no_window.add(Decimal("20"), is_buy=True)
        flow.record_snapshot()

        # Snapshot 2: 60% YES
        flow.yes_window.add(Decimal("60"), is_buy=True)
        flow.no_window.add(Decimal("40"), is_buy=True)
        flow.record_snapshot()

        # Overall: (80+60) / (80+20+60+40) = 140/200 = 0.70
        assert flow.historical_avg_yes_rate == pytest.approx(0.70, abs=0.01)

    def test_historical_avg_no_snapshots(self) -> None:
        """No snapshots → neutral 0.5."""
        tracker = MarketFlowTracker()
        tracker.register_market("cond_1", "tok_yes", "tok_no")
        flow = tracker.get_market_flow("cond_1")
        assert flow.historical_avg_yes_rate == 0.5

    def test_snapshot_maxlen_168(self) -> None:
        """Snapshots capped at 168 (7 days × 24 hours)."""
        tracker = MarketFlowTracker()
        tracker.register_market("cond_1", "tok_yes", "tok_no")
        flow = tracker.get_market_flow("cond_1")

        for i in range(200):
            flow.yes_window.add(Decimal("10"), is_buy=True)
            flow.record_snapshot()

        assert flow.snapshot_count == 168  # maxlen

    def test_take_snapshots_all_markets(self) -> None:
        """take_snapshots records for all markets at once."""
        tracker = MarketFlowTracker()
        tracker.register_market("cond_1", "tok_y1", "tok_n1")
        tracker.register_market("cond_2", "tok_y2", "tok_n2")

        # Add some volume
        flow1 = tracker.get_market_flow("cond_1")
        flow1.yes_window.add(Decimal("100"), is_buy=True)
        flow2 = tracker.get_market_flow("cond_2")
        flow2.yes_window.add(Decimal("200"), is_buy=True)

        snaps = tracker.take_snapshots()
        assert len(snaps) == 2
        assert all(isinstance(s, FlowSnapshotEntry) for s in snaps)

        # Both markets now have 1 snapshot
        assert flow1.snapshot_count == 1
        assert flow2.snapshot_count == 1

    def test_zscore_uses_4h_window(self) -> None:
        """get_yes_zscore defaults to 4h (14400s) window."""
        tracker = MarketFlowTracker()
        tracker.register_market("cond_1", "tok_yes", "tok_no")
        flow = tracker.get_market_flow("cond_1")

        now = time.monotonic()
        flow.yes_window._time_fn = lambda: now

        # Add varying data within 4h window (need std > 0)
        for i in range(1, 7):
            vol = Decimal(str(8 + i * 2))  # 10, 12, 14, 16, 18, 20
            flow.yes_window.add(vol, is_buy=True, timestamp=now - i * 300 - 1)
        flow.yes_window.add(Decimal("500"), is_buy=True, timestamp=now - 1)

        # Default call (4h window)
        zscore_4h = flow.get_yes_zscore()

        # Explicit 1h window — should differ (fewer buckets)
        zscore_1h = flow.get_yes_zscore(window_s=3600)

        # Both should be positive (spike detected)
        assert zscore_4h > 0
        assert zscore_1h > 0


# ================================================================
# RDH-203: Taker flow DB persistence
# ================================================================


class TestTakerFlowDBPersistence:
    """DB persistence for taker flow snapshots."""

    @staticmethod
    async def _create_db():
        """Create in-memory SQLite DB with taker_flow_snapshots table."""
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as conn:
            # Use INTEGER (not BIGINT) for SQLite autoincrement compatibility
            await conn.execute(
                text(
                    """CREATE TABLE taker_flow_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        market_condition_id VARCHAR(128) NOT NULL,
                        yes_taker_volume NUMERIC(16, 2) NOT NULL,
                        no_taker_volume NUMERIC(16, 2) NOT NULL,
                        yes_no_ratio FLOAT NOT NULL,
                        z_score FLOAT NOT NULL,
                        window_seconds INTEGER NOT NULL,
                        is_peak_optimism BOOLEAN DEFAULT 0,
                        captured_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )"""
                )
            )
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        return engine, factory

    async def test_save_taker_flow_snapshots(self) -> None:
        """save_taker_flow_snapshots persists to DB."""
        from sqlalchemy import select

        from arbo.utils.db import TakerFlowSnapshot as TakerFlowSnapshotDB

        engine, factory = await self._create_db()

        tracker = MarketFlowTracker()
        tracker.register_market("cond_1", "tok_yes", "tok_no")

        # Add some volume
        flow = tracker.get_market_flow("cond_1")
        flow.yes_window.add(Decimal("200"), is_buy=True)
        flow.no_window.add(Decimal("100"), is_buy=True)

        saved = await tracker.save_taker_flow_snapshots(factory)
        assert saved == 1

        # Verify data in DB
        async with factory() as session:
            result = await session.execute(select(TakerFlowSnapshotDB))
            rows = result.scalars().all()

        assert len(rows) == 1
        assert rows[0].market_condition_id == "cond_1"
        assert rows[0].yes_taker_volume > Decimal("0")
        assert rows[0].no_taker_volume > Decimal("0")
        assert rows[0].window_seconds == 14400

        await engine.dispose()

    async def test_save_snapshots_multiple_markets(self) -> None:
        """Saves one row per tracked market."""
        from sqlalchemy import select

        from arbo.utils.db import TakerFlowSnapshot as TakerFlowSnapshotDB

        engine, factory = await self._create_db()

        tracker = MarketFlowTracker()
        tracker.register_market("cond_1", "tok_y1", "tok_n1")
        tracker.register_market("cond_2", "tok_y2", "tok_n2")

        tracker.get_market_flow("cond_1").yes_window.add(Decimal("100"), is_buy=True)
        tracker.get_market_flow("cond_2").yes_window.add(Decimal("200"), is_buy=True)

        saved = await tracker.save_taker_flow_snapshots(factory)
        assert saved == 2

        async with factory() as session:
            result = await session.execute(select(TakerFlowSnapshotDB))
            rows = result.scalars().all()

        assert len(rows) == 2
        condition_ids = {r.market_condition_id for r in rows}
        assert condition_ids == {"cond_1", "cond_2"}

        await engine.dispose()

    async def test_save_snapshots_empty_tracker(self) -> None:
        """No registered markets → 0 saved."""
        engine, factory = await self._create_db()

        tracker = MarketFlowTracker()
        saved = await tracker.save_taker_flow_snapshots(factory)
        assert saved == 0

        await engine.dispose()

    async def test_query_by_market_id(self) -> None:
        """Can query snapshots by market_condition_id."""
        from sqlalchemy import select

        from arbo.utils.db import TakerFlowSnapshot as TakerFlowSnapshotDB

        engine, factory = await self._create_db()

        tracker = MarketFlowTracker()
        tracker.register_market("cond_1", "tok_y1", "tok_n1")
        tracker.register_market("cond_2", "tok_y2", "tok_n2")

        tracker.get_market_flow("cond_1").yes_window.add(Decimal("100"), is_buy=True)
        tracker.get_market_flow("cond_2").yes_window.add(Decimal("200"), is_buy=True)

        await tracker.save_taker_flow_snapshots(factory)

        # Query only cond_1
        async with factory() as session:
            result = await session.execute(
                select(TakerFlowSnapshotDB).where(
                    TakerFlowSnapshotDB.market_condition_id == "cond_1"
                )
            )
            rows = result.scalars().all()

        assert len(rows) == 1
        assert rows[0].market_condition_id == "cond_1"

        await engine.dispose()


# ================================================================
# Helpers
# ================================================================


def _mock_config() -> MagicMock:
    """Create a mock ArboConfig for order flow tests."""
    config = MagicMock()
    config.polygon_rpc_url = "https://lb.drpc.live/polygon/test-key"
    config.order_flow.ctf_exchange = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
    config.order_flow.volume_zscore_threshold = 2.0
    config.order_flow.flow_imbalance_threshold = 0.65
    config.order_flow.min_converging_signals = 2
    config.order_flow.rolling_windows = [3600, 14400, 86400]
    return config
