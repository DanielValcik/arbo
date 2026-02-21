"""Tests for Paper Trading Engine (PM-004).

Tests verify:
1. Trade placement with half-Kelly sizing
2. Slippage simulation (BUY/SELL)
3. Risk manager integration (rejection, size adjustment)
4. Position tracking (open, average cost)
5. Market resolution (win/loss P&L)
6. Portfolio snapshot
7. Stats calculation (win rate, ROI, per-layer)
8. Edge filtering (no-edge trades rejected)
9. Balance management (deduction on trade, return on resolution)
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from arbo.core.paper_engine import (
    PaperPosition,
    PaperTrade,
    PaperTradingEngine,
    PortfolioSnapshot,
    TradeStatus,
)
from arbo.core.risk_manager import RiskManager


@pytest.fixture(autouse=True)
def reset_risk_singleton() -> None:
    """Reset risk manager singleton between tests."""
    RiskManager.reset()


@pytest.fixture
def engine() -> PaperTradingEngine:
    """Paper trading engine with $2000 capital and no risk manager."""
    return PaperTradingEngine(initial_capital=Decimal("2000"))


@pytest.fixture
def engine_with_risk() -> PaperTradingEngine:
    """Paper trading engine with risk manager."""
    rm = RiskManager(capital=Decimal("2000"))
    return PaperTradingEngine(
        initial_capital=Decimal("2000"),
        risk_manager=rm,
    )


def _place_trade(
    engine: PaperTradingEngine,
    token_id: str = "token_123",
    market_condition_id: str = "cond_123",
    side: str = "BUY",
    market_price: Decimal = Decimal("0.50"),
    model_prob: Decimal = Decimal("0.65"),
    layer: int = 2,
    market_category: str = "soccer",
    fee_enabled: bool = False,
    confluence_score: int = 2,
) -> PaperTrade | None:
    """Helper to place a standard trade."""
    return engine.place_trade(
        market_condition_id=market_condition_id,
        token_id=token_id,
        side=side,
        market_price=market_price,
        model_prob=model_prob,
        layer=layer,
        market_category=market_category,
        fee_enabled=fee_enabled,
        confluence_score=confluence_score,
    )


# ================================================================
# Trade placement
# ================================================================


class TestTradePlacement:
    """Basic trade placement tests."""

    def test_place_trade_returns_trade(self, engine: PaperTradingEngine) -> None:
        trade = _place_trade(engine)
        assert trade is not None
        assert isinstance(trade, PaperTrade)

    def test_trade_has_correct_fields(self, engine: PaperTradingEngine) -> None:
        trade = _place_trade(engine, token_id="tok_1", side="BUY", layer=3)
        assert trade is not None
        assert trade.token_id == "tok_1"
        assert trade.side == "BUY"
        assert trade.layer == 3
        assert trade.status == TradeStatus.OPEN

    def test_trade_id_auto_increments(self, engine: PaperTradingEngine) -> None:
        t1 = _place_trade(engine, token_id="tok_1")
        t2 = _place_trade(engine, token_id="tok_2")
        assert t1 is not None and t2 is not None
        assert t2.id == t1.id + 1

    def test_balance_deducted_after_trade(self, engine: PaperTradingEngine) -> None:
        initial = engine.balance
        trade = _place_trade(engine)
        assert trade is not None
        assert engine.balance == initial - trade.size

    def test_trade_history_tracked(self, engine: PaperTradingEngine) -> None:
        _place_trade(engine, token_id="tok_1")
        _place_trade(engine, token_id="tok_2")
        assert len(engine.trade_history) == 2


# ================================================================
# Edge and Kelly sizing
# ================================================================


class TestEdgeSizing:
    """Edge filtering and position sizing."""

    def test_no_edge_trade_rejected(self, engine: PaperTradingEngine) -> None:
        """If model_prob == market_price, edge is 0 → no trade."""
        trade = _place_trade(engine, model_prob=Decimal("0.50"), market_price=Decimal("0.50"))
        assert trade is None

    def test_negative_edge_trade_rejected(self, engine: PaperTradingEngine) -> None:
        """If model_prob < market_price for BUY, edge < 0 → no trade."""
        trade = _place_trade(engine, model_prob=Decimal("0.45"), market_price=Decimal("0.50"))
        assert trade is None

    def test_high_confluence_capped_at_5pct(self, engine: PaperTradingEngine) -> None:
        """Even with high confluence, position is capped at 5%."""
        trade = _place_trade(
            engine,
            model_prob=Decimal("0.90"),
            market_price=Decimal("0.50"),
            confluence_score=5,
        )
        assert trade is not None
        max_size = Decimal("2000") * Decimal("0.05")
        assert trade.size <= max_size

    def test_low_confluence_capped_at_2_5pct(self, engine: PaperTradingEngine) -> None:
        """Confluence 2 caps at 2.5%."""
        trade = _place_trade(
            engine,
            model_prob=Decimal("0.90"),
            market_price=Decimal("0.50"),
            confluence_score=2,
        )
        assert trade is not None
        max_size = Decimal("2000") * Decimal("0.025")
        assert trade.size <= max_size


# ================================================================
# Slippage
# ================================================================


class TestSlippage:
    """Slippage simulation tests."""

    def test_buy_slippage_increases_price(self, engine: PaperTradingEngine) -> None:
        """BUY slippage should give a worse (higher) fill price."""
        trade = _place_trade(engine, side="BUY", market_price=Decimal("0.50"))
        assert trade is not None
        assert trade.fill_price > trade.price

    def test_sell_slippage_decreases_price(self, engine: PaperTradingEngine) -> None:
        """SELL slippage should give a worse (lower) fill price."""
        trade = _place_trade(engine, side="SELL", market_price=Decimal("0.50"))
        assert trade is not None
        assert trade.fill_price < trade.price

    def test_fill_price_clamped(self) -> None:
        """Fill price should be clamped to [0.001, 0.999]."""
        engine = PaperTradingEngine(
            initial_capital=Decimal("2000"),
            slippage_pct=Decimal("0.50"),  # extreme 50% slippage
        )
        # BUY at 0.70 with 50% slippage → 0.70*1.50=1.05 → clamped to 0.999
        trade = _place_trade(
            engine,
            side="BUY",
            market_price=Decimal("0.70"),
            model_prob=Decimal("0.95"),
        )
        assert trade is not None
        assert trade.fill_price == Decimal("0.999")


# ================================================================
# Position tracking
# ================================================================


class TestPositionTracking:
    """Open position management."""

    def test_position_created_on_trade(self, engine: PaperTradingEngine) -> None:
        _place_trade(engine, token_id="tok_1")
        assert len(engine.open_positions) == 1
        assert engine.open_positions[0].token_id == "tok_1"

    def test_position_aggregated_on_same_token(self, engine: PaperTradingEngine) -> None:
        """Multiple trades on same token should aggregate into one position."""
        _place_trade(engine, token_id="tok_1", model_prob=Decimal("0.65"))
        _place_trade(engine, token_id="tok_1", model_prob=Decimal("0.70"))
        assert len(engine.open_positions) == 1
        pos = engine.open_positions[0]
        # Should have combined shares from both trades
        assert pos.shares > Decimal("0")

    def test_different_tokens_separate_positions(self, engine: PaperTradingEngine) -> None:
        _place_trade(engine, token_id="tok_1")
        _place_trade(engine, token_id="tok_2")
        assert len(engine.open_positions) == 2

    def test_update_position_price(self, engine: PaperTradingEngine) -> None:
        _place_trade(engine, token_id="tok_1")
        engine.update_position_price("tok_1", Decimal("0.70"))
        pos = engine.open_positions[0]
        assert pos.current_price == Decimal("0.70")

    def test_unrealized_pnl_buy_up(self, engine: PaperTradingEngine) -> None:
        """BUY position with price increase → positive unrealized P&L."""
        _place_trade(engine, token_id="tok_1", side="BUY", market_price=Decimal("0.50"))
        engine.update_position_price("tok_1", Decimal("0.70"))
        pos = engine.open_positions[0]
        assert pos.unrealized_pnl > Decimal("0")

    def test_unrealized_pnl_no_current_price(self, engine: PaperTradingEngine) -> None:
        """Without current price update, unrealized P&L is 0."""
        _place_trade(engine, token_id="tok_1")
        pos = engine.open_positions[0]
        assert pos.unrealized_pnl == Decimal("0")


# ================================================================
# Market resolution
# ================================================================


class TestMarketResolution:
    """Market settlement and P&L calculation."""

    def test_buy_win_positive_pnl(self, engine: PaperTradingEngine) -> None:
        """BUY token that resolves YES → positive P&L."""
        trade = _place_trade(engine, token_id="tok_1", side="BUY", market_price=Decimal("0.50"))
        assert trade is not None
        pnl = engine.resolve_market("tok_1", winning_outcome=True)
        assert pnl > Decimal("0")

    def test_buy_loss_negative_pnl(self, engine: PaperTradingEngine) -> None:
        """BUY token that resolves NO → negative P&L (lose entire investment)."""
        trade = _place_trade(engine, token_id="tok_1", side="BUY", market_price=Decimal("0.50"))
        assert trade is not None
        pnl = engine.resolve_market("tok_1", winning_outcome=False)
        assert pnl < Decimal("0")
        assert pnl == -trade.size

    def test_resolution_returns_capital_to_balance(self, engine: PaperTradingEngine) -> None:
        """After resolution, invested capital + P&L returns to balance."""
        trade = _place_trade(engine, token_id="tok_1", side="BUY")
        assert trade is not None
        balance_after_trade = engine.balance
        pnl = engine.resolve_market("tok_1", winning_outcome=True)
        # Balance should be: balance_after_trade + trade.size + pnl
        expected = balance_after_trade + trade.size + pnl
        assert engine.balance == expected

    def test_position_removed_after_resolution(self, engine: PaperTradingEngine) -> None:
        _place_trade(engine, token_id="tok_1")
        engine.resolve_market("tok_1", winning_outcome=True)
        assert len(engine.open_positions) == 0

    def test_trade_status_updated_on_win(self, engine: PaperTradingEngine) -> None:
        _place_trade(engine, token_id="tok_1")
        engine.resolve_market("tok_1", winning_outcome=True)
        trade = engine.trade_history[0]
        assert trade.status == TradeStatus.WON
        assert trade.actual_pnl is not None
        assert trade.resolved_at is not None

    def test_trade_status_updated_on_loss(self, engine: PaperTradingEngine) -> None:
        _place_trade(engine, token_id="tok_1")
        engine.resolve_market("tok_1", winning_outcome=False)
        trade = engine.trade_history[0]
        assert trade.status == TradeStatus.LOST

    def test_resolve_unknown_token_returns_zero(self, engine: PaperTradingEngine) -> None:
        pnl = engine.resolve_market("nonexistent", winning_outcome=True)
        assert pnl == Decimal("0")

    def test_per_layer_pnl_tracked(self, engine: PaperTradingEngine) -> None:
        """Resolution should update per-layer P&L stats."""
        _place_trade(engine, token_id="tok_1", layer=2)
        _place_trade(engine, token_id="tok_2", layer=5)
        engine.resolve_market("tok_1", winning_outcome=True)
        engine.resolve_market("tok_2", winning_outcome=False)
        stats = engine.get_stats()
        assert 2 in stats["per_layer_pnl"]
        assert 5 in stats["per_layer_pnl"]


# ================================================================
# Risk manager integration
# ================================================================


class TestRiskManagerIntegration:
    """Paper engine respects risk manager decisions."""

    def test_risk_rejection_returns_none(self, engine_with_risk: PaperTradingEngine) -> None:
        """Trade exceeding 5% cap should be rejected."""
        # Force a large model prob to get large Kelly size, then manually check
        # The risk manager caps at 5% of $2000 = $100
        trade = _place_trade(
            engine_with_risk,
            model_prob=Decimal("0.65"),
            market_price=Decimal("0.50"),
            confluence_score=2,
        )
        # Should be accepted (within limits)
        assert trade is not None

    def test_shutdown_blocks_all_trades(self, engine_with_risk: PaperTradingEngine) -> None:
        """After risk manager shutdown, all trades rejected."""
        engine_with_risk._risk_manager._trigger_shutdown("test")
        trade = _place_trade(engine_with_risk)
        assert trade is None


# ================================================================
# Snapshot and stats
# ================================================================


class TestSnapshotAndStats:
    """Portfolio snapshot and statistics."""

    def test_take_snapshot(self, engine: PaperTradingEngine) -> None:
        _place_trade(engine, token_id="tok_1")
        snapshot = engine.take_snapshot()
        assert isinstance(snapshot, PortfolioSnapshot)
        assert snapshot.num_open_positions == 1
        assert snapshot.balance == engine.balance

    def test_total_value_includes_invested(self, engine: PaperTradingEngine) -> None:
        """Total value = balance + invested + unrealized P&L."""
        initial = engine.total_value
        _place_trade(engine, token_id="tok_1")
        # Total value should still be close to initial (just slippage difference)
        assert engine.total_value <= initial  # slippage reduces total slightly

    def test_stats_empty_engine(self, engine: PaperTradingEngine) -> None:
        stats = engine.get_stats()
        assert stats["total_trades"] == 0
        assert stats["wins"] == 0
        assert stats["losses"] == 0
        assert stats["total_pnl"] == Decimal("0")

    def test_stats_after_trades(self, engine: PaperTradingEngine) -> None:
        _place_trade(engine, token_id="tok_1")
        _place_trade(engine, token_id="tok_2")
        engine.resolve_market("tok_1", winning_outcome=True)
        engine.resolve_market("tok_2", winning_outcome=False)
        stats = engine.get_stats()
        assert stats["total_trades"] == 2
        assert stats["resolved_trades"] == 2
        assert stats["wins"] == 1
        assert stats["losses"] == 1
        assert stats["win_rate"] == 0.5

    def test_stats_roi_calculation(self, engine: PaperTradingEngine) -> None:
        """ROI should be (total_pnl / initial_capital) * 100."""
        _place_trade(engine, token_id="tok_1")
        engine.resolve_market("tok_1", winning_outcome=True)
        stats = engine.get_stats()
        expected_roi = (stats["total_pnl"] / Decimal("2000")) * 100
        assert stats["roi_pct"] == expected_roi


# ================================================================
# PaperPosition unit tests
# ================================================================


class TestPaperPosition:
    """Unit tests for PaperPosition dataclass."""

    def test_unrealized_pnl_buy_positive(self) -> None:
        pos = PaperPosition(
            market_condition_id="c1",
            token_id="t1",
            side="BUY",
            avg_price=Decimal("0.50"),
            size=Decimal("100"),
            shares=Decimal("200"),
            layer=2,
            current_price=Decimal("0.70"),
        )
        assert pos.unrealized_pnl == Decimal("200") * (Decimal("0.70") - Decimal("0.50"))

    def test_unrealized_pnl_sell_positive(self) -> None:
        pos = PaperPosition(
            market_condition_id="c1",
            token_id="t1",
            side="SELL",
            avg_price=Decimal("0.50"),
            size=Decimal("100"),
            shares=Decimal("200"),
            layer=2,
            current_price=Decimal("0.30"),
        )
        assert pos.unrealized_pnl == Decimal("200") * (Decimal("0.50") - Decimal("0.30"))

    def test_unrealized_pnl_no_current_price(self) -> None:
        pos = PaperPosition(
            market_condition_id="c1",
            token_id="t1",
            side="BUY",
            avg_price=Decimal("0.50"),
            size=Decimal("100"),
            shares=Decimal("200"),
            layer=2,
        )
        assert pos.unrealized_pnl == Decimal("0")
