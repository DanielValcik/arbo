"""Tests for PaperTradingEngine DB persistence methods."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import OperationalError

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
            side_effect=OperationalError("SELECT 1", {}, Exception("DB down")),
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
        mock_pos.strategy = "A"
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

        # Backfill query: paper_trades WHERE status='open' — returns no extra rows.
        backfill_result = MagicMock()
        backfill_result.scalars.return_value.all.return_value = []

        # trade_details query returns empty result set
        td_result = MagicMock()
        td_result.__iter__ = MagicMock(return_value=iter([]))
        mock_session.execute = AsyncMock(
            side_effect=[pos_result, backfill_result, td_result, snap_result]
        )

        with patch("arbo.utils.db.get_session_factory", return_value=mock_factory):
            await engine.load_state_from_db()

        assert "tok_1" in engine._positions
        assert engine._balance == Decimal("1900.00")

    @pytest.mark.asyncio
    async def test_load_state_db_failure_does_not_crash(self, engine: PaperTradingEngine) -> None:
        """DB failure should not crash the engine."""
        with patch(
            "arbo.utils.db.get_session_factory",
            side_effect=OperationalError("SELECT 1", {}, Exception("DB down")),
        ):
            await engine.load_state_from_db()

        # Engine still works with initial capital
        assert engine.balance == Decimal("2000")

    @pytest.mark.asyncio
    async def test_load_state_backfills_from_paper_trades_when_mirror_empty(
        self, engine: PaperTradingEngine
    ) -> None:
        """If paper_positions mirror is empty but paper_trades has open rows,
        backfill _positions from paper_trades (authoritative).

        Regression guard for 2026-04-22 B2-13 incident: paper_positions wiped
        by sync_positions_to_db mid-restart → resolve_market silently skipped
        real-capital positions → stuck status='open' forever.
        """
        # paper_positions mirror is empty (the bug condition).
        pos_result = MagicMock()
        pos_result.scalars.return_value.all.return_value = []

        # paper_trades has two open rows (one B2 live, one pure paper) plus
        # a pre_reset row that MUST be skipped.
        live_trade = MagicMock()
        live_trade.token_id = "tok_live"
        live_trade.market_condition_id = "cond_live"
        live_trade.side = "BUY"
        live_trade.price = Decimal("0.10")
        live_trade.size = Decimal("2.50")
        live_trade.layer = 0
        live_trade.strategy = "B2"
        live_trade.placed_at = datetime.now(UTC)
        live_trade.notes = None
        live_trade.trade_details = {"live_fill_status": "filled"}

        paper_trade = MagicMock()
        paper_trade.token_id = "tok_paper"
        paper_trade.market_condition_id = "cond_paper"
        paper_trade.side = "BUY"
        paper_trade.price = Decimal("0.20")
        paper_trade.size = Decimal("5.00")
        paper_trade.layer = 0
        paper_trade.strategy = "C"
        paper_trade.placed_at = datetime.now(UTC)
        paper_trade.notes = None
        paper_trade.trade_details = None

        archived_trade = MagicMock()
        archived_trade.token_id = "tok_archived"
        archived_trade.market_condition_id = "cond_archived"
        archived_trade.side = "BUY"
        archived_trade.price = Decimal("0.30")
        archived_trade.size = Decimal("1.00")
        archived_trade.layer = 0
        archived_trade.strategy = "B2"
        archived_trade.placed_at = datetime.now(UTC)
        archived_trade.notes = "[pre_reset_2026-04-16_16:35]"
        archived_trade.trade_details = {"live_fill_status": "filled"}

        backfill_result = MagicMock()
        backfill_result.scalars.return_value.all.return_value = [
            live_trade,
            paper_trade,
            archived_trade,
        ]

        td_result = MagicMock()
        td_result.__iter__ = MagicMock(return_value=iter([]))

        snap_result = MagicMock()
        snap_result.scalars.return_value.first.return_value = None

        mock_session = AsyncMock()
        mock_factory = MagicMock(return_value=mock_session)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.execute = AsyncMock(
            side_effect=[pos_result, backfill_result, td_result, snap_result]
        )

        with patch("arbo.utils.db.get_session_factory", return_value=mock_factory):
            await engine.load_state_from_db()

        assert "tok_live" in engine._positions
        assert "tok_paper" in engine._positions
        assert "tok_archived" not in engine._positions  # pre_reset filter
        assert engine._positions["tok_live"].strategy == "B2"
        assert engine._positions["tok_live"].size == Decimal("2.50")
        # shares = size / price = 2.50 / 0.10 = 25
        assert engine._positions["tok_live"].shares == Decimal("25")

    @pytest.mark.asyncio
    async def test_load_state_backfill_skips_tokens_already_in_mirror(
        self, engine: PaperTradingEngine
    ) -> None:
        """Backfill must not override rows already loaded from paper_positions."""
        mirror_pos = MagicMock()
        mirror_pos.market_condition_id = "cond_1"
        mirror_pos.token_id = "tok_shared"
        mirror_pos.side = "BUY"
        mirror_pos.avg_price = Decimal("0.50")
        mirror_pos.size = Decimal("100.00")
        mirror_pos.layer = 0
        mirror_pos.strategy = "B3"
        mirror_pos.current_price = None
        mirror_pos.opened_at = datetime.now(UTC)

        pos_result = MagicMock()
        pos_result.scalars.return_value.all.return_value = [mirror_pos]

        # paper_trades row for same token — should be skipped.
        dup_trade = MagicMock()
        dup_trade.token_id = "tok_shared"
        dup_trade.market_condition_id = "cond_1"
        dup_trade.side = "BUY"
        dup_trade.price = Decimal("0.99")  # different → would override if used
        dup_trade.size = Decimal("999.00")
        dup_trade.layer = 0
        dup_trade.strategy = "B3"
        dup_trade.placed_at = datetime.now(UTC)
        dup_trade.notes = None
        dup_trade.trade_details = {"live_fill_status": "filled"}

        backfill_result = MagicMock()
        backfill_result.scalars.return_value.all.return_value = [dup_trade]

        td_result = MagicMock()
        td_result.__iter__ = MagicMock(return_value=iter([]))

        snap_result = MagicMock()
        snap_result.scalars.return_value.first.return_value = None

        mock_session = AsyncMock()
        mock_factory = MagicMock(return_value=mock_session)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.execute = AsyncMock(
            side_effect=[pos_result, backfill_result, td_result, snap_result]
        )

        with patch("arbo.utils.db.get_session_factory", return_value=mock_factory):
            await engine.load_state_from_db()

        # The mirror row wins — backfill is a fallback only.
        assert engine._positions["tok_shared"].avg_price == Decimal("0.50")
        assert engine._positions["tok_shared"].size == Decimal("100.00")
