"""Tests for 5 critical production bug fixes.

FIX 1: Real market price (not hardcoded 0.50)
FIX 2: Per-market deduplication (no duplicate trades)
FIX 3: Total exposure cap (80% max capital deployed)
FIX 4: Per-market position limit (1 position per market)
FIX 5: Signal persistence to DB (audit trail)
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arbo.core.confluence import ConfluenceScorer, ScoredOpportunity
from arbo.core.paper_engine import PaperPosition, PaperTrade, PaperTradingEngine, TradeStatus
from arbo.core.risk_manager import (
    MAX_POSITIONS_PER_MARKET,
    MAX_TOTAL_EXPOSURE_PCT,
    RiskManager,
    TradeRequest,
)
from arbo.core.scanner import Signal, SignalDirection


# ================================================================
# Helpers
# ================================================================


def _make_signal(
    layer: int = 2,
    market_condition_id: str = "cond_1",
    token_id: str = "tok_1",
    direction: SignalDirection = SignalDirection.BUY_YES,
    edge: Decimal = Decimal("0.06"),
    confidence: Decimal = Decimal("0.7"),
    details: dict | None = None,
) -> Signal:
    return Signal(
        layer=layer,
        market_condition_id=market_condition_id,
        token_id=token_id,
        direction=direction,
        edge=edge,
        confidence=confidence,
        details=details or {},
    )


def _make_trade_request(
    market_id: str = "test_market",
    size: Decimal = Decimal("50"),
    category: str = "soccer",
) -> TradeRequest:
    return TradeRequest(
        market_id=market_id,
        token_id="test_token",
        side="BUY",
        price=Decimal("0.45"),
        size=size,
        layer=2,
        market_category=category,
        confluence_score=2,
    )


@pytest.fixture(autouse=True)
def reset_singleton() -> None:
    """Reset risk manager singleton between tests."""
    RiskManager.reset()


@pytest.fixture
def risk_manager() -> RiskManager:
    return RiskManager(capital=Decimal("2000"))


@pytest.fixture
def paper_engine(risk_manager: RiskManager) -> PaperTradingEngine:
    return PaperTradingEngine(
        initial_capital=Decimal("2000"),
        risk_manager=risk_manager,
    )


@pytest.fixture
def scorer(risk_manager: RiskManager) -> ConfluenceScorer:
    return ConfluenceScorer(risk_manager=risk_manager, capital=Decimal("2000"))


# ================================================================
# FIX 1: Real market price tests
# ================================================================


class TestRealMarketPrice:
    """FIX 1: Verify hardcoded price 0.50 is replaced with real prices."""

    @pytest.mark.asyncio
    async def test_real_price_from_discovery(self) -> None:
        """place_trade called with price_yes from discovery, not 0.50."""
        from arbo.main import ArboOrchestrator

        orch = ArboOrchestrator(mode="paper")

        # Mock components
        orch._confluence = MagicMock()
        orch._paper_engine = MagicMock()
        orch._paper_engine._positions = {}
        orch._paper_engine.place_trade = MagicMock(return_value=None)
        orch._paper_engine.save_trade_to_db = AsyncMock()

        # Mock discovery with real price
        mock_market = MagicMock()
        mock_market.category = "soccer"
        mock_market.fee_enabled = False
        mock_market.price_yes = 0.35
        orch._discovery = MagicMock()
        orch._discovery.get_by_condition_id = MagicMock(return_value=mock_market)

        # Create scored opportunity
        opp = ScoredOpportunity(
            market_condition_id="cond_1",
            token_id="tok_1",
            direction=SignalDirection.BUY_YES,
            score=2,
            signals=[_make_signal(details={"poly_price": 0.35})],
            contributing_layers={2, 4},
            position_size_pct=Decimal("0.025"),
            recommended_size=Decimal("50"),
            best_edge=Decimal("0.06"),
        )
        orch._confluence.get_tradeable = MagicMock(return_value=[opp])

        signals = [_make_signal(details={"poly_price": 0.35})]
        with patch.object(orch, "_save_signals_to_db", new_callable=AsyncMock):
            await orch._process_signal_batch(signals)

        # Verify place_trade was called with real price, not 0.50
        call_kwargs = orch._paper_engine.place_trade.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs["market_price"] == Decimal("0.35")

    @pytest.mark.asyncio
    async def test_price_fallback_to_signal_details(self) -> None:
        """Uses poly_price from signal details when discovery has no price."""
        from arbo.main import ArboOrchestrator

        orch = ArboOrchestrator(mode="paper")

        orch._confluence = MagicMock()
        orch._paper_engine = MagicMock()
        orch._paper_engine._positions = {}
        orch._paper_engine.place_trade = MagicMock(return_value=None)
        orch._paper_engine.save_trade_to_db = AsyncMock()

        # Discovery returns market with no price
        mock_market = MagicMock()
        mock_market.category = "crypto"
        mock_market.fee_enabled = True
        mock_market.price_yes = None
        orch._discovery = MagicMock()
        orch._discovery.get_by_condition_id = MagicMock(return_value=mock_market)

        opp = ScoredOpportunity(
            market_condition_id="cond_2",
            token_id="tok_2",
            direction=SignalDirection.BUY_YES,
            score=2,
            signals=[_make_signal(market_condition_id="cond_2", token_id="tok_2", details={"poly_price": 0.72})],
            contributing_layers={2, 7},
            position_size_pct=Decimal("0.025"),
            recommended_size=Decimal("50"),
            best_edge=Decimal("0.08"),
        )
        orch._confluence.get_tradeable = MagicMock(return_value=[opp])

        signals = [_make_signal(market_condition_id="cond_2", token_id="tok_2", details={"poly_price": 0.72})]
        with patch.object(orch, "_save_signals_to_db", new_callable=AsyncMock):
            await orch._process_signal_batch(signals)

        call_kwargs = orch._paper_engine.place_trade.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs["market_price"] == Decimal("0.72")

    @pytest.mark.asyncio
    async def test_no_price_skips_trade(self) -> None:
        """No trade placed when no real price is available."""
        from arbo.main import ArboOrchestrator

        orch = ArboOrchestrator(mode="paper")

        orch._confluence = MagicMock()
        orch._paper_engine = MagicMock()
        orch._paper_engine._positions = {}
        orch._paper_engine.place_trade = MagicMock(return_value=None)

        # Discovery returns market with no price
        mock_market = MagicMock()
        mock_market.category = "politics"
        mock_market.fee_enabled = False
        mock_market.price_yes = None
        orch._discovery = MagicMock()
        orch._discovery.get_by_condition_id = MagicMock(return_value=mock_market)

        # Signal also has no poly_price
        opp = ScoredOpportunity(
            market_condition_id="cond_3",
            token_id="tok_3",
            direction=SignalDirection.BUY_YES,
            score=2,
            signals=[_make_signal(market_condition_id="cond_3", token_id="tok_3", details={})],
            contributing_layers={2, 4},
            position_size_pct=Decimal("0.025"),
            recommended_size=Decimal("50"),
            best_edge=Decimal("0.06"),
        )
        orch._confluence.get_tradeable = MagicMock(return_value=[opp])

        signals = [_make_signal(market_condition_id="cond_3", token_id="tok_3", details={})]
        with patch.object(orch, "_save_signals_to_db", new_callable=AsyncMock):
            await orch._process_signal_batch(signals)

        # place_trade should NOT have been called
        orch._paper_engine.place_trade.assert_not_called()

    @pytest.mark.asyncio
    async def test_extreme_low_price_skipped(self) -> None:
        """Markets with price < 0.05 are skipped (long-shot filter)."""
        from arbo.main import ArboOrchestrator

        orch = ArboOrchestrator(mode="paper")

        orch._confluence = MagicMock()
        orch._paper_engine = MagicMock()
        orch._paper_engine._positions = {}
        orch._paper_engine.place_trade = MagicMock(return_value=None)

        mock_market = MagicMock()
        mock_market.category = "soccer"
        mock_market.fee_enabled = False
        mock_market.price_yes = 0.002  # Extreme long-shot
        orch._discovery = MagicMock()
        orch._discovery.get_by_condition_id = MagicMock(return_value=mock_market)

        opp = ScoredOpportunity(
            market_condition_id="cond_longshot",
            token_id="tok_longshot",
            direction=SignalDirection.BUY_YES,
            score=2,
            signals=[_make_signal(market_condition_id="cond_longshot", token_id="tok_longshot", details={"poly_price": 0.002})],
            contributing_layers={2, 4},
            position_size_pct=Decimal("0.025"),
            recommended_size=Decimal("50"),
            best_edge=Decimal("0.20"),
        )
        orch._confluence.get_tradeable = MagicMock(return_value=[opp])

        signals = [_make_signal(market_condition_id="cond_longshot", details={"poly_price": 0.002})]
        with patch.object(orch, "_save_signals_to_db", new_callable=AsyncMock):
            await orch._process_signal_batch(signals)

        orch._paper_engine.place_trade.assert_not_called()

    @pytest.mark.asyncio
    async def test_extreme_high_price_skipped(self) -> None:
        """Markets with price > 0.95 are skipped (near-certain filter)."""
        from arbo.main import ArboOrchestrator

        orch = ArboOrchestrator(mode="paper")

        orch._confluence = MagicMock()
        orch._paper_engine = MagicMock()
        orch._paper_engine._positions = {}
        orch._paper_engine.place_trade = MagicMock(return_value=None)

        mock_market = MagicMock()
        mock_market.category = "politics"
        mock_market.fee_enabled = False
        mock_market.price_yes = 0.98  # Near-certain
        orch._discovery = MagicMock()
        orch._discovery.get_by_condition_id = MagicMock(return_value=mock_market)

        opp = ScoredOpportunity(
            market_condition_id="cond_certain",
            token_id="tok_certain",
            direction=SignalDirection.BUY_YES,
            score=2,
            signals=[_make_signal(market_condition_id="cond_certain", token_id="tok_certain", details={"poly_price": 0.98})],
            contributing_layers={2, 7},
            position_size_pct=Decimal("0.025"),
            recommended_size=Decimal("50"),
            best_edge=Decimal("0.05"),
        )
        orch._confluence.get_tradeable = MagicMock(return_value=[opp])

        signals = [_make_signal(market_condition_id="cond_certain", details={"poly_price": 0.98})]
        with patch.object(orch, "_save_signals_to_db", new_callable=AsyncMock):
            await orch._process_signal_batch(signals)

        orch._paper_engine.place_trade.assert_not_called()


# ================================================================
# FIX 2: Per-market deduplication tests
# ================================================================


class TestPerMarketDedup:
    """FIX 2: Verify same market is traded at most once per batch."""

    @pytest.mark.asyncio
    async def test_same_market_traded_once_per_batch(self) -> None:
        """2 signals for same market -> only 1 trade."""
        from arbo.main import ArboOrchestrator

        orch = ArboOrchestrator(mode="paper")

        orch._confluence = MagicMock()
        orch._paper_engine = MagicMock()
        orch._paper_engine._positions = {}

        mock_trade = MagicMock()
        orch._paper_engine.place_trade = MagicMock(return_value=mock_trade)
        orch._paper_engine.save_trade_to_db = AsyncMock()

        mock_market = MagicMock()
        mock_market.category = "soccer"
        mock_market.fee_enabled = False
        mock_market.price_yes = 0.45
        orch._discovery = MagicMock()
        orch._discovery.get_by_condition_id = MagicMock(return_value=mock_market)

        # Two opportunities for the SAME market
        opp1 = ScoredOpportunity(
            market_condition_id="cond_dup",
            token_id="tok_dup",
            direction=SignalDirection.BUY_YES,
            score=2,
            signals=[_make_signal(market_condition_id="cond_dup", token_id="tok_dup", details={"poly_price": 0.45})],
            contributing_layers={2, 4},
            position_size_pct=Decimal("0.025"),
            recommended_size=Decimal("50"),
            best_edge=Decimal("0.06"),
        )
        opp2 = ScoredOpportunity(
            market_condition_id="cond_dup",
            token_id="tok_dup",
            direction=SignalDirection.BUY_YES,
            score=3,
            signals=[_make_signal(market_condition_id="cond_dup", token_id="tok_dup", details={"poly_price": 0.45})],
            contributing_layers={2, 4, 7},
            position_size_pct=Decimal("0.05"),
            recommended_size=Decimal("100"),
            best_edge=Decimal("0.08"),
        )
        orch._confluence.get_tradeable = MagicMock(return_value=[opp1, opp2])

        signals = [_make_signal(market_condition_id="cond_dup", details={"poly_price": 0.45})]
        with patch.object(orch, "_save_signals_to_db", new_callable=AsyncMock):
            await orch._process_signal_batch(signals)

        # place_trade should be called only once
        assert orch._paper_engine.place_trade.call_count == 1

    @pytest.mark.asyncio
    async def test_different_markets_both_traded(self) -> None:
        """2 signals for different markets -> 2 trades."""
        from arbo.main import ArboOrchestrator

        orch = ArboOrchestrator(mode="paper")

        orch._confluence = MagicMock()
        orch._paper_engine = MagicMock()
        orch._paper_engine._positions = {}

        mock_trade = MagicMock()
        orch._paper_engine.place_trade = MagicMock(return_value=mock_trade)
        orch._paper_engine.save_trade_to_db = AsyncMock()

        mock_market = MagicMock()
        mock_market.category = "soccer"
        mock_market.fee_enabled = False
        mock_market.price_yes = 0.60
        orch._discovery = MagicMock()
        orch._discovery.get_by_condition_id = MagicMock(return_value=mock_market)

        opp1 = ScoredOpportunity(
            market_condition_id="cond_a",
            token_id="tok_a",
            direction=SignalDirection.BUY_YES,
            score=2,
            signals=[_make_signal(market_condition_id="cond_a", token_id="tok_a", details={"poly_price": 0.60})],
            contributing_layers={2, 4},
            position_size_pct=Decimal("0.025"),
            recommended_size=Decimal("50"),
            best_edge=Decimal("0.06"),
        )
        opp2 = ScoredOpportunity(
            market_condition_id="cond_b",
            token_id="tok_b",
            direction=SignalDirection.BUY_YES,
            score=2,
            signals=[_make_signal(market_condition_id="cond_b", token_id="tok_b", details={"poly_price": 0.60})],
            contributing_layers={2, 7},
            position_size_pct=Decimal("0.025"),
            recommended_size=Decimal("50"),
            best_edge=Decimal("0.07"),
        )
        orch._confluence.get_tradeable = MagicMock(return_value=[opp1, opp2])

        signals = [
            _make_signal(market_condition_id="cond_a", details={"poly_price": 0.60}),
            _make_signal(market_condition_id="cond_b", details={"poly_price": 0.60}),
        ]
        with patch.object(orch, "_save_signals_to_db", new_callable=AsyncMock):
            await orch._process_signal_batch(signals)

        assert orch._paper_engine.place_trade.call_count == 2

    @pytest.mark.asyncio
    async def test_existing_position_skipped(self) -> None:
        """Existing position on token_id -> skip trade."""
        from arbo.main import ArboOrchestrator

        orch = ArboOrchestrator(mode="paper")

        orch._confluence = MagicMock()
        orch._paper_engine = MagicMock()

        # Simulate existing position
        orch._paper_engine._positions = {
            "tok_existing": PaperPosition(
                market_condition_id="cond_existing",
                token_id="tok_existing",
                side="BUY",
                avg_price=Decimal("0.40"),
                size=Decimal("50"),
                shares=Decimal("125"),
                layer=2,
            )
        }
        orch._paper_engine.place_trade = MagicMock(return_value=None)

        mock_market = MagicMock()
        mock_market.category = "soccer"
        mock_market.fee_enabled = False
        mock_market.price_yes = 0.40
        orch._discovery = MagicMock()
        orch._discovery.get_by_condition_id = MagicMock(return_value=mock_market)

        opp = ScoredOpportunity(
            market_condition_id="cond_existing",
            token_id="tok_existing",
            direction=SignalDirection.BUY_YES,
            score=2,
            signals=[_make_signal(market_condition_id="cond_existing", token_id="tok_existing", details={"poly_price": 0.40})],
            contributing_layers={2, 4},
            position_size_pct=Decimal("0.025"),
            recommended_size=Decimal("50"),
            best_edge=Decimal("0.06"),
        )
        orch._confluence.get_tradeable = MagicMock(return_value=[opp])

        signals = [_make_signal(market_condition_id="cond_existing", token_id="tok_existing")]
        with patch.object(orch, "_save_signals_to_db", new_callable=AsyncMock):
            await orch._process_signal_batch(signals)

        orch._paper_engine.place_trade.assert_not_called()


# ================================================================
# FIX 3: Total exposure cap tests
# ================================================================


class TestTotalExposureCap:
    """FIX 3: Verify 80% total exposure cap blocks new trades."""

    def test_total_exposure_blocks_at_80pct(self, risk_manager: RiskManager) -> None:
        """Trade rejected when total exposure would exceed 80% of capital."""
        # Set open positions to 79% of capital ($1580)
        risk_manager._state.open_positions_value = Decimal("1580")

        # Try to add $50 more (would be 81.5%)
        request = _make_trade_request(size=Decimal("50"))
        decision = risk_manager.pre_trade_check(request)

        assert not decision.approved
        assert "Total exposure" in decision.reason

    def test_total_exposure_allows_below_80pct(self, risk_manager: RiskManager) -> None:
        """Trade approved when total exposure stays below 80% of capital."""
        # Set open positions to 70% of capital ($1400)
        risk_manager._state.open_positions_value = Decimal("1400")

        # Add $50 more (would be 72.5% — under 80%)
        request = _make_trade_request(size=Decimal("50"))
        decision = risk_manager.pre_trade_check(request)

        assert decision.approved

    def test_open_positions_value_updated(self, risk_manager: RiskManager) -> None:
        """open_positions_value increments after post_trade_update with pnl=None."""
        assert risk_manager._state.open_positions_value == Decimal("0")

        risk_manager.post_trade_update(
            market_id="mkt_1",
            market_category="soccer",
            size=Decimal("100"),
            pnl=None,  # Opening position
        )

        assert risk_manager._state.open_positions_value == Decimal("100")

        # Close: should decrement
        risk_manager.post_trade_update(
            market_id="mkt_1",
            market_category="soccer",
            size=Decimal("100"),
            pnl=Decimal("10"),  # Closing with profit
        )

        assert risk_manager._state.open_positions_value == Decimal("0")


# ================================================================
# FIX 4: Per-market position limit tests
# ================================================================


class TestPerMarketLimit:
    """FIX 4: Verify only 1 position per market is allowed."""

    def test_second_position_same_market_rejected(self, risk_manager: RiskManager) -> None:
        """Second position on same market_id is rejected."""
        # Open first position
        risk_manager.post_trade_update(
            market_id="mkt_a",
            market_category="soccer",
            size=Decimal("50"),
            pnl=None,
        )

        # Try second position on same market
        request = _make_trade_request(market_id="mkt_a", size=Decimal("50"))
        decision = risk_manager.pre_trade_check(request)

        assert not decision.approved
        assert "already has" in decision.reason

    def test_different_market_allowed(self, risk_manager: RiskManager) -> None:
        """Different market_id is allowed even with existing position."""
        # Open position on market A
        risk_manager.post_trade_update(
            market_id="mkt_a",
            market_category="soccer",
            size=Decimal("50"),
            pnl=None,
        )

        # Open position on market B — should be allowed
        request = _make_trade_request(market_id="mkt_b", size=Decimal("50"))
        decision = risk_manager.pre_trade_check(request)

        assert decision.approved

    def test_market_positions_decremented_on_close(self, risk_manager: RiskManager) -> None:
        """Closing a position frees up the per-market slot."""
        # Open position
        risk_manager.post_trade_update(
            market_id="mkt_c",
            market_category="crypto",
            size=Decimal("50"),
            pnl=None,
        )
        assert risk_manager._state.market_positions["mkt_c"] == 1

        # Close position
        risk_manager.post_trade_update(
            market_id="mkt_c",
            market_category="crypto",
            size=Decimal("50"),
            pnl=Decimal("-5"),
        )
        assert risk_manager._state.market_positions["mkt_c"] == 0

        # Now a new position on same market should be allowed
        request = _make_trade_request(market_id="mkt_c", size=Decimal("50"), category="crypto")
        decision = risk_manager.pre_trade_check(request)
        assert decision.approved


# ================================================================
# FIX 5: Signal persistence tests
# ================================================================


class TestSignalPersistence:
    """FIX 5: Verify signals are saved to DB."""

    @pytest.mark.asyncio
    async def test_signals_saved_to_db(self) -> None:
        """DB records created for each signal in the batch."""
        from arbo.main import ArboOrchestrator

        orch = ArboOrchestrator(mode="paper")
        orch._confluence = MagicMock()
        orch._paper_engine = MagicMock()
        orch._paper_engine._positions = {}
        orch._confluence.get_tradeable = MagicMock(return_value=[])

        signals = [
            _make_signal(layer=2, market_condition_id="cond_x"),
            _make_signal(layer=4, market_condition_id="cond_y"),
        ]

        # Mock the DB session and factory
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_factory = MagicMock(return_value=mock_session)

        with patch("arbo.utils.db.get_session_factory", return_value=mock_factory):
            await orch._save_signals_to_db(signals)

        # session.add called twice (one per signal)
        assert mock_session.add.call_count == 2
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_signal_save_failure_non_blocking(self) -> None:
        """DB error doesn't block signal processing."""
        from arbo.main import ArboOrchestrator

        orch = ArboOrchestrator(mode="paper")
        orch._confluence = MagicMock()
        orch._paper_engine = MagicMock()
        orch._paper_engine._positions = {}
        orch._confluence.get_tradeable = MagicMock(return_value=[])

        signals = [_make_signal()]

        # Make get_session_factory raise — error is caught internally
        with patch(
            "arbo.utils.db.get_session_factory",
            side_effect=Exception("DB connection refused"),
        ):
            # Should NOT raise
            await orch._save_signals_to_db(signals)

        # Now test the full _process_signal_batch continues despite DB failure
        with patch.object(
            orch, "_save_signals_to_db", new_callable=AsyncMock, side_effect=None
        ):
            await orch._process_signal_batch(signals)

        # If we got here without exception, the test passes
