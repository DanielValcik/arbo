"""Tests for paper trading position tracker."""

from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.agents.arb_scanner import ArbOpportunity
from src.execution.position_tracker import PaperTracker
from src.utils.db import Base, Bet, DailyPnl


@pytest.fixture
async def db_session():
    """Create an in-memory SQLite async session for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        # Only create tables that don't use JSONB (SQLite doesn't support it)
        await conn.run_sync(
            Base.metadata.create_all,
            tables=[Bet.__table__, DailyPnl.__table__],
        )

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()


def _arb_opp(
    back_odds: Decimal = Decimal("2.30"),
    lay_odds: Decimal = Decimal("2.14"),
    edge: Decimal = Decimal("0.05"),
) -> ArbOpportunity:
    return ArbOpportunity(
        event_name="Liverpool vs Arsenal",
        market_type="h2h",
        selection="Liverpool",
        back_source="bet365",
        back_odds=back_odds,
        lay_source="matchbook",
        lay_odds=lay_odds,
        edge=edge,
        commission=Decimal("0.04"),
    )


class TestPlacePaperBet:
    async def test_place_bet_returns_id(self, db_session: AsyncSession) -> None:
        tracker = PaperTracker(session=db_session, bankroll=Decimal("2000"))
        bet_id = await tracker.place_paper_bet(_arb_opp(), event_id=1)

        assert isinstance(bet_id, int)
        assert bet_id > 0

    async def test_bet_stored_in_db(self, db_session: AsyncSession) -> None:
        tracker = PaperTracker(session=db_session, bankroll=Decimal("2000"))
        bet_id = await tracker.place_paper_bet(_arb_opp(), event_id=1)

        result = await db_session.execute(select(Bet).where(Bet.id == bet_id))
        bet = result.scalar_one()

        assert bet.is_paper is True
        assert bet.strategy == "arb"
        assert bet.selection == "Liverpool"
        assert bet.platform == "bet365"
        assert bet.side == "back"
        assert bet.status == "pending"
        assert bet.odds == Decimal("2.30")
        assert bet.stake > 0

    async def test_stake_capped_at_max_bet_pct(self, db_session: AsyncSession) -> None:
        """Stake should not exceed 5% of bankroll."""
        tracker = PaperTracker(session=db_session, bankroll=Decimal("2000"))
        # Very high edge would produce large Kelly stake
        bet_id = await tracker.place_paper_bet(_arb_opp(edge=Decimal("0.50")), event_id=1)

        result = await db_session.execute(select(Bet).where(Bet.id == bet_id))
        bet = result.scalar_one()

        max_stake = Decimal("2000") * Decimal("0.05")
        assert bet.stake <= max_stake

    async def test_stake_minimum_5_eur(self, db_session: AsyncSession) -> None:
        """Stake should not go below 5 EUR."""
        tracker = PaperTracker(session=db_session, bankroll=Decimal("100"))
        bet_id = await tracker.place_paper_bet(_arb_opp(edge=Decimal("0.001")), event_id=1)

        result = await db_session.execute(select(Bet).where(Bet.id == bet_id))
        bet = result.scalar_one()

        assert bet.stake >= Decimal("5.00")


class TestSettlePaperBet:
    async def test_settle_win_bookmaker_no_commission(self, db_session: AsyncSession) -> None:
        """Bookmaker back bets have no commission on winnings."""
        tracker = PaperTracker(session=db_session, bankroll=Decimal("2000"))
        bet_id = await tracker.place_paper_bet(_arb_opp(), event_id=1)
        await db_session.commit()

        await tracker.settle_paper_bet(bet_id, won=True)
        await db_session.commit()

        result = await db_session.execute(select(Bet).where(Bet.id == bet_id))
        bet = result.scalar_one()

        assert bet.status == "settled"
        assert bet.actual_pnl is not None
        assert bet.actual_pnl > 0
        assert bet.settled_at is not None
        assert bet.commission_paid == Decimal(0)  # No commission for bookmaker bets

    async def test_settle_win_matchbook_with_commission(self, db_session: AsyncSession) -> None:
        """Matchbook exchange bets have 4% commission on winnings."""
        opp = _arb_opp()
        opp = opp.model_copy(update={"back_source": "matchbook"})
        tracker = PaperTracker(session=db_session, bankroll=Decimal("2000"))
        bet_id = await tracker.place_paper_bet(opp, event_id=1)
        await db_session.commit()

        await tracker.settle_paper_bet(bet_id, won=True)
        await db_session.commit()

        result = await db_session.execute(select(Bet).where(Bet.id == bet_id))
        bet = result.scalar_one()

        assert bet.status == "settled"
        assert bet.actual_pnl > 0
        assert bet.commission_paid > 0

    async def test_settle_loss(self, db_session: AsyncSession) -> None:
        tracker = PaperTracker(session=db_session, bankroll=Decimal("2000"))
        bet_id = await tracker.place_paper_bet(_arb_opp(), event_id=1)
        await db_session.commit()

        await tracker.settle_paper_bet(bet_id, won=False)
        await db_session.commit()

        result = await db_session.execute(select(Bet).where(Bet.id == bet_id))
        bet = result.scalar_one()

        assert bet.status == "settled"
        assert bet.actual_pnl < 0
        assert bet.commission_paid == Decimal(0)

    async def test_settle_nonexistent_bet(self, db_session: AsyncSession) -> None:
        """Settling a non-existent bet should not crash."""
        tracker = PaperTracker(session=db_session, bankroll=Decimal("2000"))
        await tracker.settle_paper_bet(99999, won=True)
        # Should log warning but not raise


class TestGetOpenPaperBets:
    async def test_returns_open_bets(self, db_session: AsyncSession) -> None:
        tracker = PaperTracker(session=db_session, bankroll=Decimal("2000"))
        await tracker.place_paper_bet(_arb_opp(), event_id=1)
        await db_session.commit()

        open_bets = await tracker.get_open_paper_bets()
        assert len(open_bets) == 1

    async def test_excludes_settled_bets(self, db_session: AsyncSession) -> None:
        tracker = PaperTracker(session=db_session, bankroll=Decimal("2000"))
        bet_id = await tracker.place_paper_bet(_arb_opp(), event_id=1)
        await db_session.commit()

        await tracker.settle_paper_bet(bet_id, won=True)
        await db_session.commit()

        open_bets = await tracker.get_open_paper_bets()
        assert len(open_bets) == 0
