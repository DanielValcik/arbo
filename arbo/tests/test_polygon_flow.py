"""Tests for PM-105: Order Flow Monitor (Layer 7).

Tests verify:
1. RollingWindow: add/volume, prune, z-score normal, z-score spike, buy/sell tracking, empty
2. OrderFilled parsing: valid log, invalid→None, buy/sell detection
3. Convergence: below thresholds, z+imbalance→signal, z+delta→signal, signal format layer 7

Acceptance: parses OrderFilled events, calculates Z-scores, data in DB. Min 100 events.
"""

from __future__ import annotations

import time
from decimal import Decimal
from unittest.mock import MagicMock, patch

from arbo.connectors.polygon_flow import (
    OrderFilledEvent,
    OrderFlowMonitor,
    RollingWindow,
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
# Helpers
# ================================================================


def _mock_config() -> MagicMock:
    """Create a mock ArboConfig for order flow tests."""
    config = MagicMock()
    config.alchemy_key = "test-key"
    config.order_flow.ctf_exchange = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
    config.order_flow.volume_zscore_threshold = 2.0
    config.order_flow.flow_imbalance_threshold = 0.65
    config.order_flow.min_converging_signals = 2
    config.order_flow.rolling_windows = [3600, 14400, 86400]
    return config
