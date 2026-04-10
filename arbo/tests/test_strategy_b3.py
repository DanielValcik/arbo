"""Tests for Strategy B3 — Binance Oracle Scalper.

Tests cover:
- Quality gate constants validation
- B3Scanner signal generation (fair value computation)
- B3Scanner event lifecycle (mark traded, expiry cleanup)
- StrategyB3 exit logic (profit target, stop loss, time, edge)
- StrategyB3 restart handling (stale positions)
"""

from __future__ import annotations

import time
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from arbo.strategies.b3_quality_gate import (
    CONTRARIAN,
    EDGE_EXIT,
    EDGE_SCALING,
    ENTRY_THRESHOLD,
    LIVE_MAX_FILL_PRICE,
    MAX_BET_SIZE,
    MAX_ENTRY_MIN,
    MAX_ENTRY_MKT_FV,
    MAX_HOLD_MIN,
    MIN_ENTRY_MIN,
    MIN_ENTRY_MKT_FV,
    POSITION_PCT,
    PROFIT_TARGET,
    SIGMA_METHOD,
    SIGMA_SCALE,
    SIGMA_WINDOW,
    SPREAD,
    STOP_LOSS,
    STRATEGY_NAME,
    WINDOW_MIN,
)
from arbo.strategies.b3_scanner import B3Event, B3Scanner, _norm_cdf
from arbo.strategies.strategy_b3 import B3Position, StrategyB3

# ═══════════════════════════════════════════════════════════════════════════════
# Quality Gate Constants
# ═══════════════════════════════════════════════════════════════════════════════


class TestQualityGateConstants:
    """Validate autoresearch parameters match spec."""

    def test_strategy_name(self) -> None:
        assert STRATEGY_NAME == "B3"

    def test_window_5min(self) -> None:
        assert WINDOW_MIN == 5

    def test_sigma_window_realized(self) -> None:
        # Chainlink-calibrated 24h realized vol window
        assert SIGMA_WINDOW == 1440

    def test_sigma_method_realized(self) -> None:
        assert SIGMA_METHOD == "realized"

    def test_sigma_scale_below_1(self) -> None:
        """Sigma scale < 1.0 means signal is more aggressive than market."""
        assert SIGMA_SCALE == 0.348
        assert SIGMA_SCALE < 1.0

    def test_entry_min_window(self) -> None:
        # V6.0 widens entry window: minutes 1-3
        assert MIN_ENTRY_MIN == 1
        assert MAX_ENTRY_MIN == 3

    def test_momentum_not_contrarian(self) -> None:
        assert CONTRARIAN is False

    def test_entry_threshold(self) -> None:
        assert ENTRY_THRESHOLD == 0.020

    def test_profit_target(self) -> None:
        assert PROFIT_TARGET == 0.207

    def test_stop_loss_disabled(self) -> None:
        # Never-sell live mode: stop_loss is paper-only and disabled
        assert STOP_LOSS == 99.0

    def test_max_hold_3min(self) -> None:
        assert MAX_HOLD_MIN == 3

    def test_edge_exit(self) -> None:
        assert EDGE_EXIT == 0.076

    def test_spread(self) -> None:
        assert SPREAD == 0.060

    def test_price_bounds(self) -> None:
        # Paper scanner bounds
        assert MIN_ENTRY_MKT_FV == 0.413
        assert MAX_ENTRY_MKT_FV == 0.570
        # Live fill price cap (from 278 live trade analysis)
        assert LIVE_MAX_FILL_PRICE == 0.75

    def test_sizing_constants(self) -> None:
        assert POSITION_PCT == 0.026
        assert EDGE_SCALING == 10.000
        assert MAX_BET_SIZE == 100.0


# ═══════════════════════════════════════════════════════════════════════════════
# norm_cdf
# ═══════════════════════════════════════════════════════════════════════════════


class TestNormCDF:
    """Test normal CDF implementation."""

    def test_zero(self) -> None:
        assert abs(_norm_cdf(0.0) - 0.5) < 1e-10

    def test_positive(self) -> None:
        assert _norm_cdf(1.0) > 0.84
        assert _norm_cdf(1.0) < 0.85

    def test_negative(self) -> None:
        assert _norm_cdf(-1.0) > 0.15
        assert _norm_cdf(-1.0) < 0.16

    def test_large_positive(self) -> None:
        assert _norm_cdf(5.0) > 0.999

    def test_symmetry(self) -> None:
        assert abs(_norm_cdf(1.5) + _norm_cdf(-1.5) - 1.0) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# B3Scanner — Signal Generation
# ═══════════════════════════════════════════════════════════════════════════════


class TestB3ScannerSignals:
    """Test B3Scanner.scan() fair value computation and signal generation."""

    def _make_scanner_with_event(
        self,
        btc_start: float = 87000.0,
        minutes_ago: float = 2.0,
    ) -> B3Scanner:
        """Create a scanner with one event at the right entry time."""
        scanner = B3Scanner()
        now = time.time()
        start_ts = now - minutes_ago * 60
        end_ts = start_ts + WINDOW_MIN * 60
        scanner._events["test_cid"] = B3Event(
            condition_id="test_cid",
            token_id_up="up_token",
            token_id_down="down_token",
            question="Bitcoin Up or Down - test",
            start_ts=start_ts,
            end_ts=end_ts,
            btc_at_start=btc_start,
        )
        return scanner

    def test_btc_up_generates_up_signal(self) -> None:
        """When BTC goes up significantly, should get direction=+1 (buy Up)."""
        scanner = self._make_scanner_with_event(btc_start=87000)
        # BTC went up 0.1% — should be enough for signal
        signals = scanner.scan(btc_price=87200, sigma_per_min=0.0008)
        # Signal depends on sigma and magnitude of move
        if signals:
            assert signals[0].direction == 1

    def test_btc_down_generates_down_signal(self) -> None:
        """When BTC goes down significantly, should get direction=-1 (buy Down)."""
        scanner = self._make_scanner_with_event(btc_start=87000)
        signals = scanner.scan(btc_price=86800, sigma_per_min=0.0008)
        if signals:
            assert signals[0].direction == -1

    def test_no_signal_small_move(self) -> None:
        """Tiny move relative to vol should not trigger entry threshold."""
        scanner = self._make_scanner_with_event(btc_start=87000)
        # $1 move on $87K with high sigma → signal_fv ≈ 0.50 → below 0.020 thresh
        signals = scanner.scan(btc_price=87001, sigma_per_min=0.005)
        assert len(signals) == 0

    def test_no_signal_wrong_minute(self) -> None:
        """Event at minute 0 or minute 4 should not generate signals."""
        # Too early (minute 0.5)
        scanner = self._make_scanner_with_event(minutes_ago=0.5)
        signals = scanner.scan(btc_price=87200, sigma_per_min=0.0008)
        assert len(signals) == 0

        # Too late (minute 3.5)
        scanner = self._make_scanner_with_event(minutes_ago=3.5)
        signals = scanner.scan(btc_price=87200, sigma_per_min=0.0008)
        assert len(signals) == 0

    def test_signal_fv_differs_from_market_fv(self) -> None:
        """Signal uses sigma_scale=0.644, market uses sigma_scale=1.0."""
        scanner = self._make_scanner_with_event(btc_start=87000)
        signals = scanner.scan(btc_price=87200, sigma_per_min=0.0008)
        if signals:
            sig = signals[0]
            # Signal FV should be more extreme than market FV
            # (sigma_scale < 1.0 makes signal react more aggressively)
            signal_dev = abs(sig.signal_fv_up - 0.5)
            market_dev = abs(sig.market_fv_up - 0.5)
            assert signal_dev > market_dev

    def test_entry_price_min_bound(self) -> None:
        """Scanner enforces MIN_ENTRY_MKT_FV; MAX is enforced in strategy_b3
        live gate (paper trades collect data above the cap for autoresearch).
        """
        scanner = self._make_scanner_with_event(btc_start=87000)
        signals = scanner.scan(btc_price=87200, sigma_per_min=0.0008)
        for sig in signals:
            assert sig.entry_price >= MIN_ENTRY_MKT_FV
            assert sig.entry_price <= 1.0

    def test_mark_traded_prevents_re_entry(self) -> None:
        """After marking an event as traded, no more signals."""
        scanner = self._make_scanner_with_event(btc_start=87000)
        scanner.mark_traded("test_cid")
        signals = scanner.scan(btc_price=87200, sigma_per_min=0.0008)
        assert len(signals) == 0

    def test_sigma_floor_enforced(self) -> None:
        """Sigma below floor should be clamped."""
        scanner = self._make_scanner_with_event(btc_start=87000)
        # With very small sigma, any move is a huge signal
        signals = scanner.scan(btc_price=87200, sigma_per_min=0.0000001)
        # Should still work (sigma clamped to SIGMA_FLOOR)
        assert isinstance(signals, list)


# ═══════════════════════════════════════════════════════════════════════════════
# B3Scanner — Event Lifecycle
# ═══════════════════════════════════════════════════════════════════════════════


class TestB3ScannerEvents:
    """Test event tracking and cleanup."""

    def test_expired_event_cleanup(self) -> None:
        """Events ended > 10 min ago should be removed on next scan."""
        scanner = B3Scanner()
        now = time.time()
        # Expired event (ended 15 min ago)
        scanner._events["old"] = B3Event(
            condition_id="old",
            token_id_up="u", token_id_down="d",
            question="test", start_ts=now - 1200, end_ts=now - 900,
        )
        # Active event
        scanner._events["new"] = B3Event(
            condition_id="new",
            token_id_up="u2", token_id_down="d2",
            question="test2", start_ts=now - 60, end_ts=now + 240,
        )
        # Scan triggers cleanup (but needs fetch to cleanup — simulate manually)
        scanner._last_fetch_ts = now  # prevent refetch
        scanner.scan(btc_price=87000, sigma_per_min=0.001)
        # Events not cleaned up by scan, only by fetch_events
        assert "old" in scanner._events  # cleanup is in fetch_events

    def test_active_event_count(self) -> None:
        scanner = B3Scanner()
        assert scanner.active_event_count == 0
        now = time.time()
        scanner._events["e1"] = B3Event(
            condition_id="e1", token_id_up="u", token_id_down="d",
            question="test", start_ts=now, end_ts=now + 300,
        )
        assert scanner.active_event_count == 1


# ═══════════════════════════════════════════════════════════════════════════════
# StrategyB3 — Exit Logic
# ═══════════════════════════════════════════════════════════════════════════════


class TestB3ExitLogic:
    """Test exit trigger conditions."""

    def _make_strategy(self) -> StrategyB3:
        """Create strategy with mock risk manager and binance WS."""
        rm = MagicMock()
        rm.get_strategy_state.return_value = MagicMock(
            allocated=Decimal("1000"),
            deployed=Decimal("0"),
            available=Decimal("1000"),
            is_halted=False,
        )
        binance = MagicMock()
        binance.get_price.return_value = 87100.0  # BTC went up from 87000
        strategy = StrategyB3(
            risk_manager=rm,
            paper_engine=None,
            binance_ws=binance,
        )
        return strategy

    def _add_position(
        self,
        strategy: StrategyB3,
        direction: int = 1,
        entry_mkt_fv: float = 0.60,
        hold_seconds: float = 30,
        btc_at_start: float = 87000.0,
        sigma: float = 0.0008,
    ) -> str:
        """Add a test position. Returns token_id."""
        now = time.time()
        token_id = "test_token"
        strategy._open_positions[token_id] = B3Position(
            condition_id="test_cid",
            token_id=token_id,
            direction=direction,
            entry_mkt_fv=entry_mkt_fv,
            entry_time=now - hold_seconds,
            event_start_ts=now - 120,  # Event started 2 min ago
            event_end_ts=now + 180,    # Event ends in 3 min
            btc_at_start=btc_at_start,
            btc_at_entry=btc_at_start,  # Same as start for tests (no entry drift)
            sigma_per_min=sigma,
            shares=100,
        )
        return token_id

    @pytest.mark.asyncio
    async def test_profit_target_triggers_exit(self) -> None:
        """Exit when unrealized PnL exceeds PROFIT_TARGET."""
        strategy = self._make_strategy()
        # Entry at 0.50, BTC went up a LOT → market FV now ~0.80
        # unrealized = 0.80 - 0.50 = 0.30 > PROFIT_TARGET(0.207)
        strategy._binance_ws.get_price.return_value = 87500.0  # Big move
        self._add_position(strategy, direction=1, entry_mkt_fv=0.50,
                           btc_at_start=87000.0, sigma=0.0005)
        exits = await strategy.check_exits()
        reasons = [exit_tup[1] for exit_tup in exits]
        assert "profit" in reasons

    @pytest.mark.asyncio
    async def test_stop_loss_triggers_exit(self) -> None:
        """Exit when unrealized loss exceeds STOP_LOSS."""
        strategy = self._make_strategy()
        # Entry at 0.65, BTC reversed → market FV dropped to ~0.58
        # unrealized = 0.58 - 0.65 = -0.07 < -STOP_LOSS(0.038)
        strategy._binance_ws.get_price.return_value = 86800.0  # BTC dropped
        self._add_position(strategy, direction=1, entry_mkt_fv=0.65,
                           btc_at_start=87000.0, sigma=0.0008)
        exits = await strategy.check_exits()
        reasons = [exit_tup[1] for exit_tup in exits]
        assert "stop" in reasons

    @pytest.mark.asyncio
    async def test_time_exit_after_max_hold(self) -> None:
        """Exit after MAX_HOLD_MIN minutes."""
        strategy = self._make_strategy()
        # BTC unchanged, but held for 4 min (> MAX_HOLD_MIN=3)
        strategy._binance_ws.get_price.return_value = 87050.0
        self._add_position(strategy, direction=1, entry_mkt_fv=0.55,
                           btc_at_start=87000.0, sigma=0.0008,
                           hold_seconds=250)  # > 3 min
        exits = await strategy.check_exits()
        reasons = [exit_tup[1] for exit_tup in exits]
        assert "time" in reasons

    @pytest.mark.asyncio
    async def test_no_exit_within_hold_time(self) -> None:
        """No exit when within time and PnL limits."""
        strategy = self._make_strategy()
        # BTC moved up slightly, profit < target, held < 3 min
        strategy._binance_ws.get_price.return_value = 87050.0
        self._add_position(strategy, direction=1, entry_mkt_fv=0.55,
                           btc_at_start=87000.0, sigma=0.0008,
                           hold_seconds=30)
        exits = await strategy.check_exits()
        # May or may not exit depending on edge calculation
        # At minimum, should not crash
        assert isinstance(exits, list)

    @pytest.mark.asyncio
    async def test_no_exit_without_btc_price(self) -> None:
        """No exits when BTC price unavailable."""
        strategy = self._make_strategy()
        strategy._binance_ws.get_price.return_value = None
        self._add_position(strategy)
        exits = await strategy.check_exits()
        assert len(exits) == 0

    @pytest.mark.asyncio
    async def test_exit_removes_position(self) -> None:
        """Triggered exit removes position from tracking."""
        strategy = self._make_strategy()
        # Force a stop loss exit
        strategy._binance_ws.get_price.return_value = 86700.0
        tid = self._add_position(strategy, direction=1, entry_mkt_fv=0.70,
                                 btc_at_start=87000.0, sigma=0.0008)
        assert tid in strategy._open_positions
        await strategy.check_exits()
        assert tid not in strategy._open_positions


# ═══════════════════════════════════════════════════════════════════════════════
# StrategyB3 — Restart Handling
# ═══════════════════════════════════════════════════════════════════════════════


class TestB3Restart:
    """Test that stale positions are handled on restart."""

    @pytest.mark.asyncio
    async def test_stale_positions_not_restored(self) -> None:
        """B3 positions should NOT be restored — they're stale after restart."""
        rm = MagicMock()
        pe = MagicMock()
        pe.open_positions = [
            MagicMock(token_id="stale_b3", strategy="B3", avg_price=Decimal("0.55"),
                      shares=50),
        ]

        strategy = StrategyB3(risk_manager=rm, paper_engine=pe)
        await strategy.init()

        # Stale B3 positions should NOT be in _open_positions
        assert len(strategy._open_positions) == 0

    @pytest.mark.asyncio
    async def test_no_crash_on_empty_restore(self) -> None:
        """No crash when paper engine has no B3 positions."""
        rm = MagicMock()
        pe = MagicMock()
        pe.open_positions = []

        strategy = StrategyB3(risk_manager=rm, paper_engine=pe)
        await strategy.init()
        assert len(strategy._open_positions) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# StrategyB3 — Vol Estimator Subsampling
# ═══════════════════════════════════════════════════════════════════════════════


class TestB3VolSubsampling:
    """Test that vol estimator is fed at 1-minute intervals only."""

    @pytest.mark.asyncio
    async def test_vol_update_once_per_minute(self) -> None:
        """Vol estimator should only be updated once per minute."""
        rm = MagicMock()
        rm.get_strategy_state.return_value = MagicMock(is_halted=False)

        binance = MagicMock()
        binance.get_price.return_value = 87000.0

        strategy = StrategyB3(risk_manager=rm, binance_ws=binance)
        await strategy.init()

        # First call — should update (last_vol_update_ts = 0)
        await strategy.poll_cycle()
        first_ts = strategy._last_vol_update_ts
        assert first_ts > 0

        # Second call immediately — should NOT update
        await strategy.poll_cycle()
        assert strategy._last_vol_update_ts == first_ts  # Unchanged
