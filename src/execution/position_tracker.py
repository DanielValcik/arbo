"""Paper trading position tracker.

Logs paper bets, settles them, tracks open positions,
and aggregates daily P&L using existing Bet and DailyPnl models.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: TC002

from src.agents.arb_scanner import ArbOpportunity  # noqa: TC001
from src.utils.db import Bet, DailyPnl
from src.utils.logger import get_logger
from src.utils.odds import half_kelly

log = get_logger("paper_tracker")

# Hardcoded per devbrief Section 7
MAX_BET_PCT = Decimal("0.05")


class PaperTracker:
    """Tracks paper bets and aggregates P&L."""

    def __init__(self, session: AsyncSession, bankroll: Decimal) -> None:
        self._session = session
        self._bankroll = bankroll

    async def place_paper_bet(self, opp: ArbOpportunity, event_id: int) -> int:
        """Place a paper bet for an arb opportunity. Returns bet DB id."""
        # Calculate stake using half-Kelly, capped at MAX_BET_PCT
        kelly_fraction = half_kelly(opp.edge, opp.back_odds)
        stake = min(kelly_fraction * self._bankroll, MAX_BET_PCT * self._bankroll)
        stake = max(stake, Decimal("5.00"))  # MIN_BET_EUR

        potential_pnl = stake * (opp.back_odds - Decimal(1))

        bet = Bet(
            event_id=event_id,
            strategy="arb",
            platform=opp.back_source,
            side="back",
            selection=opp.selection,
            odds=opp.back_odds,
            stake=stake,
            potential_pnl=potential_pnl,
            edge_at_exec=opp.edge,
            status="pending",
            is_paper=True,
        )
        self._session.add(bet)
        await self._session.flush()

        log.info(
            "paper_bet_placed",
            bet_id=bet.id,
            selection=opp.selection,
            odds=str(opp.back_odds),
            stake=str(stake),
            edge=str(opp.edge),
        )
        return bet.id  # type: ignore[return-value]

    async def settle_paper_bet(self, bet_id: int, won: bool) -> None:
        """Settle a paper bet with win/loss outcome."""
        stmt = select(Bet).where(Bet.id == bet_id)
        result = await self._session.execute(stmt)
        bet = result.scalar_one_or_none()

        if bet is None:
            log.warning("settle_bet_not_found", bet_id=bet_id)
            return

        if won:
            actual_pnl = bet.stake * (bet.odds - Decimal(1))
            commission_paid = actual_pnl * Decimal("0.04")  # Matchbook commission
            actual_pnl -= commission_paid
        else:
            actual_pnl = -bet.stake
            commission_paid = Decimal(0)

        await self._session.execute(
            update(Bet)
            .where(Bet.id == bet_id)
            .values(
                actual_pnl=actual_pnl,
                commission_paid=commission_paid,
                status="settled",
                settled_at=datetime.now(UTC),
            )
        )

        log.info(
            "paper_bet_settled",
            bet_id=bet_id,
            won=won,
            actual_pnl=str(actual_pnl),
        )

    async def get_open_paper_bets(self) -> list[Bet]:
        """Get all open (non-settled) paper bets."""
        stmt = select(Bet).where(
            Bet.is_paper.is_(True),
            Bet.status.in_(["pending", "matched"]),
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def aggregate_daily_pnl(self, target_date: date, bankroll_start: Decimal) -> None:
        """Aggregate all settled bets for a given date into daily_pnl."""
        # Sum up settled bets for target date
        stmt = select(
            func.count(Bet.id).label("num_bets"),
            func.count(Bet.id).filter(Bet.actual_pnl > 0).label("num_wins"),
            func.coalesce(func.sum(Bet.stake), 0).label("total_staked"),
            func.coalesce(func.sum(Bet.actual_pnl), 0).label("gross_pnl"),
            func.coalesce(func.sum(Bet.commission_paid), 0).label("total_commission"),
        ).where(
            Bet.is_paper.is_(True),
            Bet.status == "settled",
            func.date(Bet.settled_at) == target_date,
        )
        result = await self._session.execute(stmt)
        row = result.one()

        net_pnl = row.gross_pnl - row.total_commission
        bankroll_end = bankroll_start + net_pnl
        roi_pct = (net_pnl / bankroll_start * 100) if bankroll_start > 0 else Decimal(0)

        # Upsert daily P&L
        existing = await self._session.execute(select(DailyPnl).where(DailyPnl.date == target_date))
        pnl_row = existing.scalar_one_or_none()

        if pnl_row:
            await self._session.execute(
                update(DailyPnl)
                .where(DailyPnl.date == target_date)
                .values(
                    num_bets=row.num_bets,
                    num_wins=row.num_wins,
                    total_staked=row.total_staked,
                    gross_pnl=row.gross_pnl,
                    total_commission=row.total_commission,
                    net_pnl=net_pnl,
                    bankroll_start=bankroll_start,
                    bankroll_end=bankroll_end,
                    roi_pct=roi_pct,
                )
            )
        else:
            pnl = DailyPnl(
                date=target_date,
                num_bets=row.num_bets,
                num_wins=row.num_wins,
                total_staked=row.total_staked,
                gross_pnl=row.gross_pnl,
                total_commission=row.total_commission,
                net_pnl=net_pnl,
                bankroll_start=bankroll_start,
                bankroll_end=bankroll_end,
                roi_pct=roi_pct,
            )
            self._session.add(pnl)

        log.info(
            "daily_pnl_aggregated",
            date=str(target_date),
            num_bets=row.num_bets,
            net_pnl=str(net_pnl),
        )
