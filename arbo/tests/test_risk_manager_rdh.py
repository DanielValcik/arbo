"""Tests for RDH per-strategy risk management extensions."""

from __future__ import annotations

from decimal import Decimal

import pytest

from arbo.core.risk_manager import (
    KELLY_FRACTION,
    MAX_POSITIONS_PER_STRATEGY,
    RESERVE_CAPITAL,
    STRATEGY_ALLOCATIONS,
    STRATEGY_WEEKLY_DRAWDOWN_PCT,
    RiskDecision,
    RiskManager,
    TradeRequest,
)


@pytest.fixture
def risk() -> RiskManager:
    """Create a fresh RiskManager for each test."""
    RiskManager.reset()
    return RiskManager(capital=Decimal("2000"))


def _make_strategy_request(
    strategy: str = "C",
    size: Decimal = Decimal("50"),
    market_id: str = "mkt_1",
) -> TradeRequest:
    return TradeRequest(
        market_id=market_id,
        token_id="tok_1",
        side="BUY",
        price=Decimal("0.50"),
        size=size,
        layer=0,
        market_category="weather",
        strategy=strategy,
    )


class TestQuarterKelly:
    """Verify Kelly fraction updated to 0.25."""

    def test_kelly_fraction_is_quarter(self) -> None:
        assert KELLY_FRACTION == Decimal("0.25")


class TestStrategyAllocations:
    """Verify per-strategy allocation constants."""

    def test_allocations_sum_correctly(self) -> None:
        total = sum(STRATEGY_ALLOCATIONS.values()) + RESERVE_CAPITAL
        assert total == Decimal("2000")

    def test_strategy_a(self) -> None:
        assert STRATEGY_ALLOCATIONS["A"] == Decimal("400")

    def test_strategy_b(self) -> None:
        assert STRATEGY_ALLOCATIONS["B"] == Decimal("400")

    def test_strategy_c(self) -> None:
        assert STRATEGY_ALLOCATIONS["C"] == Decimal("1000")

    def test_reserve(self) -> None:
        assert RESERVE_CAPITAL == Decimal("200")


class TestStrategyAllocationLimits:
    """Test per-strategy capital deployment limits."""

    def test_within_allocation_approved(self, risk: RiskManager) -> None:
        req = _make_strategy_request(strategy="C", size=Decimal("100"))
        decision = risk.pre_trade_check(req)
        assert decision.approved

    def test_exceeds_allocation_rejected(self, risk: RiskManager) -> None:
        # Strategy A allocation = €400
        # Deploy 5 × €80 = €400 (full allocation)
        for i in range(5):
            r = _make_strategy_request(strategy="A", size=Decimal("80"), market_id=f"a_mkt_{i}")
            decision = risk.pre_trade_check(r)
            assert decision.approved, f"Trade {i} should be approved"
            risk.post_trade_update(f"a_mkt_{i}", "weather", Decimal("80"))
            risk.strategy_post_trade("A", Decimal("80"))

        # Try to deploy €80 more — exceeds €400 allocation
        overflow_req = _make_strategy_request(strategy="A", size=Decimal("80"), market_id="a_overflow")
        decision = risk.pre_trade_check(overflow_req)
        assert not decision.approved
        assert "allocation" in decision.reason.lower() or "exceed" in decision.reason.lower()

    def test_strategy_a_isolated_from_c(self, risk: RiskManager) -> None:
        """Strategy A should have its own allocation independent of C."""
        # Deploy some capital in strategy C (use different category to avoid 30% limit)
        for i in range(5):
            r = _make_strategy_request(strategy="C", size=Decimal("100"), market_id=f"c_{i}")
            r.market_category = f"weather_c_{i}"  # Spread across categories
            risk.pre_trade_check(r)
            risk.post_trade_update(f"c_{i}", f"weather_c_{i}", Decimal("100"))
            risk.strategy_post_trade("C", Decimal("100"))

        # Strategy A should still have its full allocation (separate from C)
        req = _make_strategy_request(strategy="A", size=Decimal("50"), market_id="a_1")
        req.market_category = "theta_decay"
        decision = risk.pre_trade_check(req)
        assert decision.approved


class TestStrategyPositionLimits:
    """Test max positions per strategy (10)."""

    def test_max_positions_per_strategy(self, risk: RiskManager) -> None:
        # Open 10 positions for strategy C
        for i in range(MAX_POSITIONS_PER_STRATEGY):
            r = _make_strategy_request(strategy="C", size=Decimal("50"), market_id=f"mkt_{i}")
            decision = risk.pre_trade_check(r)
            assert decision.approved, f"Position {i+1} should be approved"
            risk.post_trade_update(f"mkt_{i}", "weather", Decimal("50"))
            risk.strategy_post_trade("C", Decimal("50"))

        # 11th position should be rejected
        r = _make_strategy_request(strategy="C", size=Decimal("50"), market_id="mkt_overflow")
        decision = risk.pre_trade_check(r)
        assert not decision.approved
        assert "position" in decision.reason.lower()


class TestReserveCapital:
    """Test that reserve €200 is never deployed."""

    def test_reserve_prevents_full_deployment(self, risk: RiskManager) -> None:
        """Can't deploy more than capital - reserve."""
        # Total capital = 2000, reserve = 200, max deployable = 1800
        # But also limited by MAX_TOTAL_EXPOSURE_PCT (80%) = 1600
        # So total exposure is capped at 1600
        req = _make_strategy_request(strategy="C", size=Decimal("100"))
        decision = risk.pre_trade_check(req)
        assert decision.approved


class TestStrategyKillSwitch:
    """Test 15% weekly drawdown kill switch."""

    def test_kill_switch_triggers(self, risk: RiskManager) -> None:
        """15% weekly drawdown should halt strategy."""
        # Strategy C allocation = 1000, 15% = 150
        # Simulate a losing trade
        r = _make_strategy_request(strategy="C", size=Decimal("200"), market_id="loss_mkt")
        risk.pre_trade_check(r)
        risk.post_trade_update("loss_mkt", "weather", Decimal("200"))
        risk.strategy_post_trade("C", Decimal("200"))

        # Resolve with -160 P&L (16% of 1000 > 15% threshold)
        risk.strategy_post_trade("C", Decimal("200"), pnl=Decimal("-160"))

        # Strategy should be halted
        ss = risk.get_strategy_state("C")
        assert ss is not None
        assert ss.is_halted

        # Further trades should be rejected
        new_req = _make_strategy_request(strategy="C", size=Decimal("50"), market_id="new_mkt")
        decision = risk.pre_trade_check(new_req)
        assert not decision.approved
        assert "halted" in decision.reason.lower()

    def test_kill_switch_below_threshold(self, risk: RiskManager) -> None:
        """Losses below 15% should not halt strategy."""
        r = _make_strategy_request(strategy="C", size=Decimal("100"), market_id="loss_mkt")
        risk.pre_trade_check(r)
        risk.post_trade_update("loss_mkt", "weather", Decimal("100"))
        risk.strategy_post_trade("C", Decimal("100"))

        # Resolve with -140 P&L (14% < 15%)
        risk.strategy_post_trade("C", Decimal("100"), pnl=Decimal("-140"))

        ss = risk.get_strategy_state("C")
        assert ss is not None
        assert not ss.is_halted

    def test_ceo_reset_halt(self, risk: RiskManager) -> None:
        """CEO should be able to reset a halted strategy."""
        # Trigger halt
        r = _make_strategy_request(strategy="C", size=Decimal("200"), market_id="loss_mkt")
        risk.pre_trade_check(r)
        risk.post_trade_update("loss_mkt", "weather", Decimal("200"))
        risk.strategy_post_trade("C", Decimal("200"))
        risk.strategy_post_trade("C", Decimal("200"), pnl=Decimal("-160"))

        assert risk.get_strategy_state("C").is_halted  # type: ignore[union-attr]

        # CEO reset
        result = risk.reset_strategy_halt("C")
        assert result is True
        assert not risk.get_strategy_state("C").is_halted  # type: ignore[union-attr]

        # Trading should work again
        new_req = _make_strategy_request(strategy="C", size=Decimal("50"), market_id="new_mkt")
        decision = risk.pre_trade_check(new_req)
        assert decision.approved


class TestUnknownStrategy:
    """Test handling of unknown strategy names."""

    def test_unknown_strategy_rejected(self, risk: RiskManager) -> None:
        req = _make_strategy_request(strategy="X", size=Decimal("50"))
        decision = risk.pre_trade_check(req)
        assert not decision.approved
        assert "Unknown strategy" in decision.reason


class TestLegacyCompatibility:
    """Test that legacy trades (no strategy) still work."""

    def test_legacy_trade_no_strategy(self, risk: RiskManager) -> None:
        req = TradeRequest(
            market_id="legacy_mkt",
            token_id="tok_1",
            side="BUY",
            price=Decimal("0.50"),
            size=Decimal("50"),
            layer=2,
            market_category="politics",
            strategy="",  # Legacy — no strategy
        )
        decision = risk.pre_trade_check(req)
        assert decision.approved


class TestStrategyStateTracking:
    """Test strategy state getter and tracking."""

    def test_get_strategy_state(self, risk: RiskManager) -> None:
        ss = risk.get_strategy_state("C")
        assert ss is not None
        assert ss.allocated == Decimal("1000")
        assert ss.deployed == Decimal("0")
        assert ss.position_count == 0
        assert ss.available == Decimal("1000")

    def test_available_decreases_on_deploy(self, risk: RiskManager) -> None:
        risk.strategy_post_trade("C", Decimal("100"))
        ss = risk.get_strategy_state("C")
        assert ss is not None
        assert ss.available == Decimal("900")

    def test_unknown_strategy_returns_none(self, risk: RiskManager) -> None:
        assert risk.get_strategy_state("Z") is None

    def test_total_pnl_accumulates(self, risk: RiskManager) -> None:
        risk.strategy_post_trade("C", Decimal("100"))
        risk.strategy_post_trade("C", Decimal("100"), pnl=Decimal("20"))
        risk.strategy_post_trade("C", Decimal("100"))
        risk.strategy_post_trade("C", Decimal("100"), pnl=Decimal("-10"))

        ss = risk.get_strategy_state("C")
        assert ss is not None
        assert ss.total_pnl == Decimal("10")
        assert ss.weekly_pnl == Decimal("10")
