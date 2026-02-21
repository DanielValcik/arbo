"""Tests for PaperTradingEngine DB persistence methods."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arbo.core.paper_engine import (
    PaperTrade,
    PaperTradingEngine,
    PortfolioSnapshot,
)
from arbo.core.risk_manager import RiskManager


@pytest.fixture(autouse=True)
def reset_risk_singleton() -> None:
    RiskManager.reset()


@pytest.fixture
def engine() -> PaperTradingEngine:
    return PaperTradingEngine(initial_capital=Decimal("2000"))


def _make_trade() -> PaperTrade:
    return PaperTrade(
        id=1,
        market_condition_id="cond_1",
        token_id="tok_1",
        layer=2,
        side="BUY",
        price=Decimal("0.50"),
        fill_price=Decimal("0.5025"),
        size=Decimal("50.00"),
        shares=Decimal("99.50"),
        edge=Decimal("0.06"),
        confluence_score=2,
        kelly_fraction=Decimal("0.025"),
        fee=Decimal("0.25"),
        placed_at=datetime.now(UTC),
    )


def _make_snapshot() -> PortfolioSnapshot:
    return PortfolioSnapshot(
        balance=Decimal("1950.00"),
        unrealized_pnl=Decimal("10.00"),
        total_value=Decimal("2010.00"),
        num_open_positions=2,
        per_layer_pnl={2: Decimal("10.00")},
    )


class TestSaveTradeToDb:
    @pytest.mark.asyncio
    async def test_save_trade_creates_db_model(self, engine: PaperTradingEngine) -> None:
        """save_trade_to_db should add a PaperTrade row to the session."""
        trade = _make_trade()

        mock_session = AsyncMock()
        mock_session.add = MagicMock()  # add() is sync
        mock_factory = MagicMock(return_value=mock_session)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("arbo.utils.db.get_session_factory", return_value=mock_factory):
            await engine.save_trade_to_db(trade)

        mock_session.add.assert_called_once()
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_save_trade_db_failure_does_not_crash(self, engine: PaperTradingEngine) -> None:
        """DB failure in save_trade_to_db should log warning, not raise."""
        trade = _make_trade()

        with patch(
            "arbo.utils.db.get_session_factory",
            side_effect=Exception("DB down"),
        ):
            # Should not raise
            await engine.save_trade_to_db(trade)


class TestSaveSnapshotToDb:
    @pytest.mark.asyncio
    async def test_save_snapshot_maps_fields(self, engine: PaperTradingEngine) -> None:
        """save_snapshot_to_db should persist snapshot fields correctly."""
        snapshot = _make_snapshot()

        mock_session = AsyncMock()
        mock_session.add = MagicMock()  # add() is sync
        mock_factory = MagicMock(return_value=mock_session)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("arbo.utils.db.get_session_factory", return_value=mock_factory):
            await engine.save_snapshot_to_db(snapshot)

        mock_session.add.assert_called_once()
        db_obj = mock_session.add.call_args[0][0]
        assert db_obj.balance == Decimal("1950.00")
        assert db_obj.num_open_positions == 2


class TestLoadStateFromDb:
    @pytest.mark.asyncio
    async def test_load_state_populates_positions(self, engine: PaperTradingEngine) -> None:
        """load_state_from_db should restore positions and balance."""
        # Mock DB position
        mock_pos = MagicMock()
        mock_pos.market_condition_id = "cond_1"
        mock_pos.token_id = "tok_1"
        mock_pos.side = "BUY"
        mock_pos.avg_price = Decimal("0.50")
        mock_pos.size = Decimal("100.00")
        mock_pos.layer = 2
        mock_pos.current_price = Decimal("0.55")

        # Mock DB snapshot
        mock_snap = MagicMock()
        mock_snap.balance = Decimal("1900.00")

        mock_session = AsyncMock()
        mock_factory = MagicMock(return_value=mock_session)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        # Mock scalars for positions and snapshot queries
        pos_result = MagicMock()
        pos_result.scalars.return_value.all.return_value = [mock_pos]
        snap_result = MagicMock()
        snap_result.scalars.return_value.first.return_value = mock_snap

        mock_session.execute = AsyncMock(side_effect=[pos_result, snap_result])

        with patch("arbo.utils.db.get_session_factory", return_value=mock_factory):
            await engine.load_state_from_db()

        assert "tok_1" in engine._positions
        assert engine._balance == Decimal("1900.00")

    @pytest.mark.asyncio
    async def test_load_state_db_failure_does_not_crash(self, engine: PaperTradingEngine) -> None:
        """DB failure should not crash the engine."""
        with patch(
            "arbo.utils.db.get_session_factory",
            side_effect=Exception("DB down"),
        ):
            await engine.load_state_from_db()

        # Engine still works with initial capital
        assert engine.balance == Decimal("2000")
