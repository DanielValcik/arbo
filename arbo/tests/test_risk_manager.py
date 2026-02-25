"""Tests for risk manager — hardcoded limits enforcement.

Tests verify:
1. Order >5% capital → REJECTED
2. After 10% daily loss → auto shutdown
3. Approved order → exposure updated
4. Emergency shutdown → all blocked
5. Whale copy limits
6. Category concentration limits
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from arbo.core.risk_manager import (
    MAX_MARKET_TYPE_PCT,
    MAX_POSITION_PCT,
    WHALE_COPY_MAX_PCT,
    RiskManager,
    TradeRequest,
)


@pytest.fixture(autouse=True)
def reset_singleton() -> None:
    """Reset risk manager singleton between tests."""
    RiskManager.reset()


@pytest.fixture
def risk_manager() -> RiskManager:
    """Create a fresh risk manager with €2000 capital."""
    return RiskManager(capital=Decimal("2000"))


def _make_request(
    size: Decimal = Decimal("50"),
    layer: int = 2,
    category: str = "soccer",
    confluence: int = 2,
    is_whale_copy: bool = False,
) -> TradeRequest:
    """Helper to create a trade request."""
    return TradeRequest(
        market_id="test_market",
        token_id="test_token",
        side="BUY",
        price=Decimal("0.50"),
        size=size,
        layer=layer,
        market_category=category,
        confluence_score=confluence,
        is_whale_copy=is_whale_copy,
    )


class TestPositionSizeLimit:
    """PM-007 acceptance test 1: Order on 6% capital → REJECTED."""

    def test_order_within_limit_approved(self, risk_manager: RiskManager) -> None:
        """Order for 5% (€100) should be approved."""
        request = _make_request(size=Decimal("100"))  # 5% of €2000
        decision = risk_manager.pre_trade_check(request)
        assert decision.approved is True

    def test_order_exceeds_limit_rejected(self, risk_manager: RiskManager) -> None:
        """Order for 6% (€120) should be rejected."""
        request = _make_request(size=Decimal("120"))  # 6% of €2000
        decision = risk_manager.pre_trade_check(request)
        assert decision.approved is False
        assert "5.0%" in decision.reason or "position" in decision.reason.lower()

    def test_order_at_exact_limit(self, risk_manager: RiskManager) -> None:
        """Order for exactly 5% should be approved."""
        max_size = Decimal("2000") * MAX_POSITION_PCT
        request = _make_request(size=max_size)
        decision = risk_manager.pre_trade_check(request)
        assert decision.approved is True


class TestDailyLossLimit:
    """PM-007 acceptance test 2: After 10% daily loss → auto shutdown."""

    def test_daily_loss_triggers_shutdown(self, risk_manager: RiskManager) -> None:
        """Losing 10% of capital should trigger shutdown."""
        # Simulate losses via post_trade_update
        risk_manager.post_trade_update(
            market_id="m1",
            market_category="soccer",
            size=Decimal("100"),
            pnl=Decimal("-200"),  # €200 loss = 10% of €2000
        )

        # Next trade should be rejected
        request = _make_request(size=Decimal("50"))
        decision = risk_manager.pre_trade_check(request)
        assert decision.approved is False
        assert risk_manager.is_shutdown is True

    def test_below_daily_loss_continues(self, risk_manager: RiskManager) -> None:
        """Losing <10% should not trigger shutdown."""
        risk_manager.post_trade_update(
            market_id="m1",
            market_category="soccer",
            size=Decimal("100"),
            pnl=Decimal("-150"),  # 7.5% loss
        )
        request = _make_request(size=Decimal("50"))
        decision = risk_manager.pre_trade_check(request)
        assert decision.approved is True
        assert risk_manager.is_shutdown is False


class TestPostTradeUpdate:
    """PM-007 acceptance test 3: Approved order → exposure updated."""

    def test_exposure_tracks_category(self, risk_manager: RiskManager) -> None:
        """post_trade_update should track category exposure."""
        risk_manager.post_trade_update(
            market_id="m1", market_category="crypto", size=Decimal("100")
        )
        assert risk_manager.state.category_exposure["crypto"] == Decimal("100")

    def test_pnl_updates_daily(self, risk_manager: RiskManager) -> None:
        """P&L should accumulate in daily tracker."""
        risk_manager.post_trade_update(
            market_id="m1", market_category="soccer", size=Decimal("50"), pnl=Decimal("25")
        )
        risk_manager.post_trade_update(
            market_id="m2", market_category="soccer", size=Decimal("50"), pnl=Decimal("-10")
        )
        assert risk_manager.state.daily_pnl == Decimal("15")
        assert risk_manager.state.weekly_pnl == Decimal("15")


class TestEmergencyShutdown:
    """PM-007 acceptance test 4: Emergency shutdown blocks all orders."""

    def test_shutdown_blocks_all_orders(self, risk_manager: RiskManager) -> None:
        """After emergency shutdown, all orders are rejected."""
        risk_manager._trigger_shutdown("test reason")
        request = _make_request(size=Decimal("10"))
        decision = risk_manager.pre_trade_check(request)
        assert decision.approved is False
        assert "shutdown" in decision.reason.lower()


class TestWhaleCopyLimit:
    """Whale copy trades capped at 2.5% of capital."""

    def test_whale_copy_within_limit(self, risk_manager: RiskManager) -> None:
        whale_max = Decimal("2000") * WHALE_COPY_MAX_PCT
        request = _make_request(size=whale_max, is_whale_copy=True)
        decision = risk_manager.pre_trade_check(request)
        assert decision.approved is True

    def test_whale_copy_exceeds_limit(self, risk_manager: RiskManager) -> None:
        request = _make_request(size=Decimal("60"), is_whale_copy=True)  # 3% > 2.5%
        decision = risk_manager.pre_trade_check(request)
        assert decision.approved is False
        assert decision.adjusted_size is not None


class TestCategoryConcentration:
    """Max 30% in one market category."""

    def test_category_limit_blocks_new_orders(self, risk_manager: RiskManager) -> None:
        """After 30% in one category, new orders blocked."""
        category_max = Decimal("2000") * MAX_MARKET_TYPE_PCT
        # Fill up the category
        risk_manager.post_trade_update(market_id="m1", market_category="crypto", size=category_max)
        # New order in same category should be blocked
        request = _make_request(size=Decimal("10"), category="crypto")
        decision = risk_manager.pre_trade_check(request)
        assert decision.approved is False
        assert "crypto" in decision.reason.lower()

    def test_different_category_allowed(self, risk_manager: RiskManager) -> None:
        """Orders in different categories should work independently."""
        risk_manager.post_trade_update(
            market_id="m1",
            market_category="crypto",
            size=Decimal("2000") * MAX_MARKET_TYPE_PCT,
        )
        request = _make_request(size=Decimal("50"), category="soccer")
        decision = risk_manager.pre_trade_check(request)
        assert decision.approved is True


class TestUpdateCapital:
    """D5 Bug 2: Risk manager uses initial capital, not current balance."""

    def test_update_capital_changes_state(self, risk_manager: RiskManager) -> None:
        """update_capital() updates the capital used for limit calculations."""
        assert risk_manager.state.capital == Decimal("2000")
        risk_manager.update_capital(Decimal("1500"))
        assert risk_manager.state.capital == Decimal("1500")

    def test_limits_recalculated_after_capital_update(self, risk_manager: RiskManager) -> None:
        """After capital update, position limits are based on new capital."""
        risk_manager.update_capital(Decimal("1000"))  # Halved
        # 5% of 1000 = 50, so 60 should be rejected
        request = _make_request(size=Decimal("60"))
        decision = risk_manager.pre_trade_check(request)
        assert decision.approved is False

    def test_limits_use_new_capital_approves_smaller(self, risk_manager: RiskManager) -> None:
        """After capital update, appropriately sized orders pass."""
        risk_manager.update_capital(Decimal("1000"))
        # 5% of 1000 = 50
        request = _make_request(size=Decimal("50"))
        decision = risk_manager.pre_trade_check(request)
        assert decision.approved is True
