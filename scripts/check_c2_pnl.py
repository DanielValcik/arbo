"""Verify C2 P&L calculations are correct."""
import asyncio
from decimal import Decimal
from arbo.utils.db import PaperTrade, get_session_factory
from sqlalchemy import select


async def main():
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(PaperTrade)
            .where(PaperTrade.strategy == "C2")
            .order_by(PaperTrade.placed_at)
        )
        trades = result.scalars().all()

        total_pnl = 0.0
        wins = 0
        losses = 0
        suspicious = []

        for t in trades:
            td = t.trade_details or {}
            pnl = float(t.actual_pnl) if t.actual_pnl else 0
            total_pnl += pnl
            city = td.get("city", "?")
            direction = td.get("direction", "?")
            clob_fill = td.get("clob_fill_p1")
            entry_price = float(clob_fill) if clob_fill else float(t.price)
            exit_price = float(t.exit_price) if t.exit_price else None
            size = float(t.size)

            if pnl >= 0:
                wins += 1
            else:
                losses += 1

            # Verify P&L calculation manually
            if exit_price and entry_price > 0:
                shares = size / entry_price
                expected_pnl = shares * (exit_price - entry_price) - 0.007  # gas
                diff = abs(pnl - expected_pnl)
                ok = "OK" if diff < 0.1 else "MISMATCH"
                if diff >= 0.1:
                    suspicious.append(city)
            else:
                expected_pnl = None
                ok = "N/A"

            ep_str = f"{exit_price:.4f}" if exit_price else "-"
            exp_str = f"${expected_pnl:+.2f}" if expected_pnl is not None else "-"
            move = exit_price - entry_price if exit_price else 0
            pct = (move / entry_price * 100) if entry_price > 0 and exit_price else 0

            print(
                f"{city:12s} {direction:8s} "
                f"entry={entry_price:.4f} exit={ep_str} "
                f"move={move:+.4f} ({pct:+.1f}%) "
                f"size=${size:.2f} "
                f"pnl=${pnl:+.2f} expected={exp_str} [{ok}] "
                f"reason={t.exit_reason or t.status}"
            )

        n = wins + losses
        print()
        print(f"Total: {n} trades, {wins}W/{losses}L")
        print(f"Win rate: {wins/n*100:.0f}%" if n > 0 else "No trades")
        print(f"Total PnL: ${total_pnl:+.2f}")
        print(f"Avg PnL/trade: ${total_pnl/n:.2f}" if n > 0 else "")

        if suspicious:
            print(f"\nSUSPICIOUS: {suspicious}")
        else:
            print(f"\nAll P&L calculations verified OK")

        # Check: are all exits profitabe? That's suspicious for real trading
        if n > 0 and wins == n:
            print(f"\nWARNING: 100% win rate ({n}/{n}) — verify exit prices are real CLOB bids")


asyncio.run(main())
