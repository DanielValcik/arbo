"""Reset Strategy C2 — delete all trades and positions for clean start."""
import asyncio
from arbo.utils.db import PaperTrade, get_session_factory
from sqlalchemy import select, delete


async def main():
    factory = get_session_factory()
    async with factory() as session:
        # Count
        result = await session.execute(
            select(PaperTrade).where(PaperTrade.strategy == "C2")
        )
        trades = result.scalars().all()
        print(f"C2 trades to delete: {len(trades)}")
        for t in trades:
            td = t.trade_details or {}
            city = td.get("city", "?")
            direction = td.get("direction", "?")
            pnl = str(t.actual_pnl) if t.actual_pnl else "-"
            print(f"  {city:14s} {direction:8s} status={t.status:6s} pnl={pnl}")

        # Delete C2 trades
        await session.execute(
            delete(PaperTrade).where(PaperTrade.strategy == "C2")
        )
        await session.commit()
        print(f"Deleted {len(trades)} C2 paper_trades.")

    # Delete C2 positions
    async with factory() as session:
        try:
            from arbo.utils.db import PaperPosition
            r = await session.execute(
                delete(PaperPosition).where(PaperPosition.strategy == "C2")
            )
            await session.commit()
            print("Deleted C2 paper_positions.")
        except Exception as e:
            print(f"paper_positions delete: {e}")

    print("\nC2 reset complete. Restart arbo to begin fresh.")


asyncio.run(main())
