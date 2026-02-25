"""Tests for Strategy A: Theta Decay — Sell Optimism Premium on Longshots.

Tests verify:
1. Market filtering: longshot YES < $0.15, volume, category, resolution window
2. Peak optimism entry: 3σ spike → buy NO
3. Position sizing: quarter-Kelly, $20-50 clamp
4. Exit management: partial exit at +50%, stop loss at -30%
5. Resolution handling
6. Max concurrent positions limit
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest

from arbo.connectors.polygon_flow import MarketFlowTracker
from arbo.core.risk_manager import RiskManager
from arbo.strategies.theta_decay import STRATEGY_ID, ThetaDecay


# ================================================================
# Mock market matching GammaMarket interface
# ================================================================


@dataclass
class MockMarket:
    condition_id: str = "0xlongshot1"
    question: str = "Will X happen?"
    category: str = "politics"
    price_yes: Decimal | None = Decimal("0.10")
    price_no: Decimal | None = Decimal("0.90")
    token_id_yes: str = "tok_yes_1"
    token_id_no: str = "tok_no_1"
    fee_enabled: bool = False
    neg_risk: bool = False
    volume_24h: Decimal = Decimal("50000")
    liquidity: Decimal = Decimal("25000")
    active: bool = True
    closed: bool = False
    end_date: str = ""  # Set per test
    slug: str = "longshot-test"

    def __post_init__(self) -> None:
        if not self.end_date:
            # Default: 10 days from now (within 3-30 day window)
            future = datetime.now(UTC) + timedelta(days=10)
            self.end_date = future.isoformat()


# Mock paper engine
class MockPaperEngine:
    def __init__(self) -> None:
        self.trades: list[dict[str, Any]] = []

    def place_trade(self, **kwargs: Any) -> dict[str, Any] | None:
        trade = {"id": len(self.trades) + 1, **kwargs}
        self.trades.append(trade)
        return trade


# ================================================================
# Fixtures
# ================================================================


@pytest.fixture
def risk() -> RiskManager:
    RiskManager.reset()
    return RiskManager(capital=Decimal("2000"))


@pytest.fixture
def flow_tracker() -> MarketFlowTracker:
    return MarketFlowTracker()


@pytest.fixture
def paper_engine() -> MockPaperEngine:
    return MockPaperEngine()


@pytest.fixture
def strategy(
    risk: RiskManager,
    flow_tracker: MarketFlowTracker,
    paper_engine: MockPaperEngine,
) -> ThetaDecay:
    return ThetaDecay(
        risk_manager=risk,
        flow_tracker=flow_tracker,
        paper_engine=paper_engine,
    )


def _inject_3sigma_spike(
    flow_tracker: MarketFlowTracker,
    condition_id: str,
) -> None:
    """Inject a 3σ YES taker flow spike for a registered market."""
    flow = flow_tracker.get_market_flow(condition_id)
    if flow is None:
        return

    now = time.monotonic()
    flow.yes_window._time_fn = lambda: now

    # Small varying historical volume in 5-min buckets
    for i in range(1, 13):
        vol = Decimal(str(8 + i))
        flow.yes_window.add(vol, is_buy=True, timestamp=now - i * 300 - 1)

    # Massive spike in current bucket → 3σ+
    flow.yes_window.add(Decimal("2000"), is_buy=True, timestamp=now - 1)


# ================================================================
# Market filtering
# ================================================================


class TestMarketFiltering:
    """Filter markets for theta decay candidates."""

    async def test_accepts_longshot_yes(self, strategy: ThetaDecay) -> None:
        """YES < $0.15 is accepted."""
        await strategy.init()
        candidates = strategy._filter_candidates([MockMarket(price_yes=Decimal("0.10"))])
        assert len(candidates) == 1
        await strategy.close()

    async def test_rejects_non_longshot(self, strategy: ThetaDecay) -> None:
        """YES >= $0.15 is rejected."""
        await strategy.init()
        candidates = strategy._filter_candidates([MockMarket(price_yes=Decimal("0.20"))])
        assert len(candidates) == 0
        await strategy.close()

    async def test_rejects_dust_price(self, strategy: ThetaDecay) -> None:
        """YES < $0.01 is rejected (dust)."""
        await strategy.init()
        candidates = strategy._filter_candidates([MockMarket(price_yes=Decimal("0.005"))])
        assert len(candidates) == 0
        await strategy.close()

    async def test_rejects_low_volume(self, strategy: ThetaDecay) -> None:
        """Volume < $10K is rejected."""
        await strategy.init()
        candidates = strategy._filter_candidates(
            [MockMarket(volume_24h=Decimal("5000"))]
        )
        assert len(candidates) == 0
        await strategy.close()

    async def test_rejects_crypto(self, strategy: ThetaDecay) -> None:
        """Crypto category is excluded."""
        await strategy.init()
        candidates = strategy._filter_candidates([MockMarket(category="crypto")])
        assert len(candidates) == 0
        await strategy.close()

    async def test_rejects_fee_markets(self, strategy: ThetaDecay) -> None:
        """Fee-enabled markets are rejected."""
        await strategy.init()
        candidates = strategy._filter_candidates([MockMarket(fee_enabled=True)])
        assert len(candidates) == 0
        await strategy.close()

    async def test_rejects_too_soon_resolution(self, strategy: ThetaDecay) -> None:
        """Market resolving in < 3 days is rejected."""
        await strategy.init()
        soon = (datetime.now(UTC) + timedelta(days=1)).isoformat()
        candidates = strategy._filter_candidates([MockMarket(end_date=soon)])
        assert len(candidates) == 0
        await strategy.close()

    async def test_rejects_too_late_resolution(self, strategy: ThetaDecay) -> None:
        """Market resolving in > 30 days is rejected."""
        await strategy.init()
        far = (datetime.now(UTC) + timedelta(days=60)).isoformat()
        candidates = strategy._filter_candidates([MockMarket(end_date=far)])
        assert len(candidates) == 0
        await strategy.close()

    async def test_accepts_valid_resolution_window(self, strategy: ThetaDecay) -> None:
        """Market resolving in 3-30 days is accepted."""
        await strategy.init()
        good = (datetime.now(UTC) + timedelta(days=15)).isoformat()
        candidates = strategy._filter_candidates([MockMarket(end_date=good)])
        assert len(candidates) == 1
        await strategy.close()

    async def test_rejects_no_end_date(self, strategy: ThetaDecay) -> None:
        """Market with no end_date is rejected."""
        await strategy.init()
        mkt = MockMarket()
        mkt.end_date = None  # Override __post_init__ default
        candidates = strategy._filter_candidates([mkt])
        assert len(candidates) == 0
        await strategy.close()


# ================================================================
# Entry via 3σ peak optimism
# ================================================================


class TestPeakOptimismEntry:
    """Buy NO when 3σ YES taker flow spike detected."""

    async def test_entry_on_3sigma(
        self,
        strategy: ThetaDecay,
        flow_tracker: MarketFlowTracker,
        paper_engine: MockPaperEngine,
    ) -> None:
        """3σ spike on longshot → buy NO trade placed."""
        await strategy.init()

        mkt = MockMarket(
            condition_id="cond_longshot",
            price_yes=Decimal("0.10"),
            price_no=Decimal("0.90"),
        )

        # First poll: register market with flow tracker
        await strategy.poll_cycle([mkt])
        assert "cond_longshot" in strategy._registered_markets

        # Inject 3σ spike
        _inject_3sigma_spike(flow_tracker, "cond_longshot")

        # Second poll: should detect and trade
        traded = await strategy.poll_cycle([mkt])

        assert len(traded) >= 1
        assert traded[0]["side"] == "BUY_NO"
        assert traded[0]["strategy"] == "A"
        assert len(paper_engine.trades) >= 1
        assert strategy._trades_placed >= 1
        await strategy.close()

    async def test_no_entry_without_spike(
        self,
        strategy: ThetaDecay,
        flow_tracker: MarketFlowTracker,
        paper_engine: MockPaperEngine,
    ) -> None:
        """No 3σ spike → no trade."""
        await strategy.init()
        mkt = MockMarket(condition_id="cond_calm")
        traded = await strategy.poll_cycle([mkt])
        assert len(traded) == 0
        assert len(paper_engine.trades) == 0
        await strategy.close()

    async def test_no_duplicate_position(
        self,
        strategy: ThetaDecay,
        flow_tracker: MarketFlowTracker,
        paper_engine: MockPaperEngine,
    ) -> None:
        """Already have position → skip market."""
        await strategy.init()
        mkt = MockMarket(condition_id="cond_dup")

        # First trade
        await strategy.poll_cycle([mkt])
        _inject_3sigma_spike(flow_tracker, "cond_dup")
        await strategy.poll_cycle([mkt])
        first_count = len(paper_engine.trades)

        # Re-inject spike — should not trade again
        _inject_3sigma_spike(flow_tracker, "cond_dup")
        await strategy.poll_cycle([mkt])

        assert len(paper_engine.trades) == first_count
        await strategy.close()


# ================================================================
# Position sizing
# ================================================================


class TestPositionSizing:
    """Quarter-Kelly sizing clamped to $20-50."""

    async def test_size_clamped_minimum(self, strategy: ThetaDecay) -> None:
        """Size below $20 is clamped up to $20."""
        # Small edge → tiny Kelly → clamp to $20 min
        size = strategy._compute_size(
            price_no=Decimal("0.90"),
            edge=Decimal("0.04"),
            available=Decimal("400"),
        )
        assert size >= Decimal("20")

    async def test_size_clamped_maximum(self, strategy: ThetaDecay) -> None:
        """Size above $50 is clamped down to $50."""
        # Large edge → big Kelly → clamp to $50 max
        size = strategy._compute_size(
            price_no=Decimal("0.90"),
            edge=Decimal("0.50"),
            available=Decimal("400"),
        )
        assert size <= Decimal("50")

    async def test_size_within_range(self, strategy: ThetaDecay) -> None:
        """Normal edge produces size in $20-50 range."""
        size = strategy._compute_size(
            price_no=Decimal("0.90"),
            edge=Decimal("0.10"),
            available=Decimal("400"),
        )
        assert Decimal("20") <= size <= Decimal("50")

    async def test_size_respects_max_position_pct(
        self, risk: RiskManager, flow_tracker: MarketFlowTracker
    ) -> None:
        """Size never exceeds MAX_POSITION_PCT of total capital."""
        # With $2000 capital, MAX_POSITION_PCT=5% → max $100
        strategy = ThetaDecay(risk_manager=risk, flow_tracker=flow_tracker)
        size = strategy._compute_size(
            price_no=Decimal("0.90"),
            edge=Decimal("0.30"),
            available=Decimal("400"),
        )
        assert size <= Decimal("100")  # 5% of $2000


# ================================================================
# Exit management
# ================================================================


class TestExitManagement:
    """Partial exit and stop loss."""

    async def test_partial_exit_at_50pct(self, strategy: ThetaDecay) -> None:
        """NO price up 50% → sell half."""
        await strategy.init()
        strategy._active_positions["cond_1"] = _make_position(
            "cond_1", entry_price=Decimal("0.80"), size=Decimal("40")
        )

        exits = strategy.check_exits({"cond_1": Decimal("1.20")})  # +50%

        assert len(exits) == 1
        assert exits[0]["action"] == "partial_exit"
        assert exits[0]["exit_size"] == 20.0
        # Position still active (half remaining)
        assert "cond_1" in strategy._active_positions
        await strategy.close()

    async def test_no_double_partial_exit(self, strategy: ThetaDecay) -> None:
        """Partial exit only happens once."""
        await strategy.init()
        strategy._active_positions["cond_1"] = _make_position(
            "cond_1", entry_price=Decimal("0.80"), size=Decimal("40")
        )

        # First partial exit
        strategy.check_exits({"cond_1": Decimal("1.20")})

        # Second check at same price → no exit
        exits = strategy.check_exits({"cond_1": Decimal("1.25")})
        assert len(exits) == 0
        await strategy.close()

    async def test_stop_loss_at_30pct(self, strategy: ThetaDecay) -> None:
        """NO price down 30% → close all."""
        await strategy.init()
        strategy._active_positions["cond_1"] = _make_position(
            "cond_1", entry_price=Decimal("0.80"), size=Decimal("40")
        )

        exits = strategy.check_exits({"cond_1": Decimal("0.56")})  # -30%

        assert len(exits) == 1
        assert exits[0]["action"] == "stop_loss"
        # Position removed
        assert "cond_1" not in strategy._active_positions
        await strategy.close()

    async def test_no_exit_in_range(self, strategy: ThetaDecay) -> None:
        """Price within range → no exit."""
        await strategy.init()
        strategy._active_positions["cond_1"] = _make_position(
            "cond_1", entry_price=Decimal("0.80"), size=Decimal("40")
        )

        exits = strategy.check_exits({"cond_1": Decimal("0.85")})  # +6.25%
        assert len(exits) == 0
        assert "cond_1" in strategy._active_positions
        await strategy.close()


# ================================================================
# Resolution handling
# ================================================================


class TestResolution:
    """Market resolution handling."""

    async def test_handle_resolution_removes_position(
        self, strategy: ThetaDecay
    ) -> None:
        """Resolution removes position and updates P&L."""
        await strategy.init()
        strategy._active_positions["cond_1"] = _make_position("cond_1")

        strategy.handle_resolution("cond_1", pnl=Decimal("15"))

        assert "cond_1" not in strategy._active_positions
        await strategy.close()

    async def test_handle_resolution_unknown_market(
        self, strategy: ThetaDecay
    ) -> None:
        """Resolution for unknown market doesn't crash."""
        await strategy.init()
        strategy.handle_resolution("unknown", pnl=Decimal("0"))
        await strategy.close()


# ================================================================
# Concurrent position limit
# ================================================================


class TestConcurrentLimit:
    """Max 10 concurrent positions."""

    async def test_max_concurrent_positions(
        self,
        strategy: ThetaDecay,
        flow_tracker: MarketFlowTracker,
        paper_engine: MockPaperEngine,
    ) -> None:
        """Cannot exceed max concurrent positions."""
        await strategy.init()

        # Fill up with 10 positions
        for i in range(10):
            cond_id = f"cond_{i}"
            strategy._active_positions[cond_id] = _make_position(cond_id)

        # Try to trade on an 11th market
        mkt = MockMarket(condition_id="cond_11")
        await strategy.poll_cycle([mkt])
        _inject_3sigma_spike(flow_tracker, "cond_11")
        traded = await strategy.poll_cycle([mkt])

        assert len(traded) == 0  # Hit limit
        await strategy.close()


# ================================================================
# Stats
# ================================================================


class TestStats:
    """Strategy statistics."""

    async def test_initial_stats(self, strategy: ThetaDecay) -> None:
        stats = strategy.stats
        assert stats["strategy"] == "A"
        assert stats["signals_generated"] == 0
        assert stats["trades_placed"] == 0
        assert stats["active_positions"] == 0

    async def test_stats_after_trade(
        self,
        strategy: ThetaDecay,
        flow_tracker: MarketFlowTracker,
    ) -> None:
        await strategy.init()
        mkt = MockMarket(condition_id="cond_stat")

        # Register + spike + trade
        await strategy.poll_cycle([mkt])
        _inject_3sigma_spike(flow_tracker, "cond_stat")
        await strategy.poll_cycle([mkt])

        stats = strategy.stats
        assert stats["signals_generated"] >= 1
        assert stats["trades_placed"] >= 1
        assert stats["active_positions"] >= 1
        assert stats["last_scan"] is not None
        await strategy.close()


# ================================================================
# Helpers
# ================================================================


def _make_position(
    condition_id: str = "cond_test",
    entry_price: Decimal = Decimal("0.85"),
    size: Decimal = Decimal("30"),
) -> Any:
    """Create a _ThetaPosition for testing."""
    from arbo.strategies.theta_decay import _ThetaPosition

    return _ThetaPosition(
        condition_id=condition_id,
        token_id="tok_no_test",
        entry_price=entry_price,
        size=size,
        entry_zscore=3.5,
        entered_at=datetime.now(UTC),
    )
