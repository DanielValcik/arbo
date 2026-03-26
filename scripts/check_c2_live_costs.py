"""Analyze C2 trades — would they be profitable in live trading?

Compares paper P&L against realistic live costs:
- Gas: ~$0.004-0.007 per transaction (Polygon ~30-50 gwei, POL ~$0.25)
- Spread: paper already uses CLOB fill (ask+slippage) for entry and best_bid for exit
- The question: is best_bid realistic? And is gas accounted correctly?
"""
import asyncio
from arbo.utils.db import PaperTrade, get_session_factory
from sqlalchemy import select


GAS_ENTRY = 0.007   # Current paper model (POLYGON_GAS_COST_USD)
GAS_EXIT = 0.007    # Current paper model
REAL_GAS_ENTRY = 0.005  # Realistic: ~200K gas × 50 gwei × $0.25 POL
REAL_GAS_EXIT = 0.005   # Same


async def main():
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(PaperTrade)
            .where(PaperTrade.strategy == "C2")
            .where(PaperTrade.status == "sold")
            .order_by(PaperTrade.placed_at)
        )
        trades = result.scalars().all()

    print("=== C2 LIVE VIABILITY ANALYSIS ===\n")
    print(f"Paper gas model: ${GAS_ENTRY} entry + ${GAS_EXIT} exit = ${GAS_ENTRY+GAS_EXIT} round-trip")
    print(f"Real gas estimate: ${REAL_GAS_ENTRY} entry + ${REAL_GAS_EXIT} exit = ${REAL_GAS_ENTRY+REAL_GAS_EXIT} round-trip")
    print()

    total_paper_pnl = 0
    total_live_pnl = 0
    live_wins = 0
    live_losses = 0

    # Group by city to see patterns
    city_stats = {}

    for t in trades:
        td = t.trade_details or {}
        city = td.get("city", "?")
        clob_fill = float(td.get("clob_fill_p1") or 0)
        entry_fill = clob_fill + 0.005  # CLOB_TAKER_SLIPPAGE
        exit_price = float(t.exit_price) if t.exit_price else 0
        size = float(t.size)
        paper_pnl = float(t.actual_pnl or 0)
        shares = size / entry_fill if entry_fill > 0 else 0

        # Paper P&L: shares × (exit - entry) - $0.007 gas
        # Live P&L would be same BUT:
        # 1. Entry fill might be worse (maker/taker depends on order type)
        # 2. Exit fill might be worse (need to hit actual bid with depth)
        # 3. Gas is slightly different

        # Key metric: price move per share
        move = exit_price - entry_fill
        spread_earned = move * shares  # Gross profit from price move

        # In live: is there enough depth at bid to sell our shares?
        # With $12-50 size at $0.02-0.08 price = 150-2500 shares
        # Weather bid depth was ~$18 median for cheap tokens
        # If our sell > bid depth, we get partial fill or worse price

        gas_diff = (REAL_GAS_ENTRY + REAL_GAS_EXIT) - (GAS_ENTRY + GAS_EXIT)
        live_pnl = paper_pnl + gas_diff  # Adjust for gas difference

        total_paper_pnl += paper_pnl
        total_live_pnl += live_pnl

        if live_pnl > 0:
            live_wins += 1
        else:
            live_losses += 1

        if city not in city_stats:
            city_stats[city] = {"trades": 0, "paper_pnl": 0, "moves": [], "sizes": []}
        city_stats[city]["trades"] += 1
        city_stats[city]["paper_pnl"] += paper_pnl
        city_stats[city]["moves"].append(move)
        city_stats[city]["sizes"].append(size)

    print(f"{'City':12s} {'Trades':>6} {'Paper PnL':>10} {'Avg Move':>10} {'Avg Size':>9} {'Avg Shares':>11}")
    print("-" * 65)
    for city, s in sorted(city_stats.items(), key=lambda x: -x[1]["paper_pnl"]):
        avg_move = sum(s["moves"]) / len(s["moves"])
        avg_size = sum(s["sizes"]) / len(s["sizes"])
        avg_entry = avg_size / (avg_size / avg_move) if avg_move > 0 else 0.05
        avg_shares = avg_size / 0.03  # rough
        print(f"{city:12s} {s['trades']:>6} ${s['paper_pnl']:>9.2f} ${avg_move:>9.4f} ${avg_size:>8.2f} {avg_shares:>10.0f}")

    print(f"\n{'='*65}")
    print(f"Total trades: {len(trades)}")
    print(f"Paper PnL: ${total_paper_pnl:+.2f}")
    print(f"Gas adjustment: ${(REAL_GAS_ENTRY+REAL_GAS_EXIT - GAS_ENTRY-GAS_EXIT) * len(trades):+.4f}")
    print(f"Estimated live PnL: ${total_live_pnl:+.2f}")
    print(f"Live viability: {live_wins}W/{live_losses}L")

    print(f"\n=== KEY RISK: BID DEPTH ===")
    print(f"Median weather bid depth: ~$18 (from spread analysis)")
    print(f"C2 avg trade size: ${sum(s['sizes']) / len(trades) if trades else 0:.2f}")
    print(f"If trade size > bid depth → partial fill or slippage on exit")

    print(f"\n=== RECOMMENDATION ===")
    for city, s in sorted(city_stats.items(), key=lambda x: -x[1]["paper_pnl"]):
        avg_move = sum(s["moves"]) / len(s["moves"])
        if avg_move <= 0.002:
            print(f"  {city}: avg move ${avg_move:.4f} — RISKY in live (spread eats profit)")
        elif avg_move <= 0.005:
            print(f"  {city}: avg move ${avg_move:.4f} — MARGINAL in live")
        else:
            print(f"  {city}: avg move ${avg_move:.4f} — VIABLE in live")


asyncio.run(main())
