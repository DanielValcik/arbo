"""Verify C2 P&L — use fill_price (with slippage), not raw clob_fill."""
import asyncio
from decimal import Decimal
from arbo.utils.db import PaperTrade, get_session_factory
from sqlalchemy import select

GAS = 0.007
ENTRY_SLIPPAGE = 0.005  # CLOB_TAKER_SLIPPAGE


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
        issues = []

        for t in trades:
            td = t.trade_details or {}
            pnl = float(t.actual_pnl) if t.actual_pnl else 0
            total_pnl += pnl
            city = td.get("city", "?")

            clob_fill = float(td.get("clob_fill_p1") or 0)
            fill_price = clob_fill + ENTRY_SLIPPAGE  # What paper engine actually used
            exit_price = float(t.exit_price) if t.exit_price else None
            size = float(t.size)

            if exit_price and fill_price > 0:
                shares = size / fill_price
                expected_pnl = shares * (exit_price - fill_price) - GAS
                diff = abs(pnl - expected_pnl)
                ok = "OK" if diff < 0.1 else f"DIFF={diff:.2f}"
                if diff >= 0.1:
                    issues.append(city)
            else:
                expected_pnl = None
                ok = "OPEN"

            move = (exit_price - fill_price) if exit_price else 0
            ep_str = f"{exit_price:.4f}" if exit_price else "-"
            exp_str = f"${expected_pnl:+.2f}" if expected_pnl is not None else "-"

            print(
                f"{city:12s} clob={clob_fill:.4f} fill={fill_price:.4f} "
                f"exit={ep_str} move={move:+.4f} "
                f"shares={size/fill_price:.1f} "
                f"pnl=${pnl:+.2f} expected={exp_str} [{ok}] "
                f"{t.exit_reason or t.status}"
            )

        n = len([t for t in trades if t.status != "open"])
        print(f"\nTotal PnL: ${total_pnl:+.2f}")
        if issues:
            print(f"ISSUES in: {issues}")
        else:
            print("All P&L calculations VERIFIED OK")


asyncio.run(main())
