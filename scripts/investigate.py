"""Emergency investigation: missing capital, phantom positions."""
import asyncio
import sqlalchemy as sa
from sqlalchemy import text
from arbo.utils.db import get_session_factory


async def investigate():
    factory = get_session_factory()
    async with factory() as session:
        # 1. All open positions (raw SQL to avoid ORM column issues)
        result = await session.execute(text("""
            SELECT market_condition_id, side, size, avg_price, layer, opened_at, token_id, current_price, unrealized_pnl
            FROM paper_positions
            ORDER BY opened_at DESC
        """))
        positions = result.fetchall()
        print("=== OPEN POSITIONS ===")
        total_deployed = 0
        total_unrealized = 0
        for p in positions:
            print(f"  cond={p[0][:24]}... side={p[1]} size=${p[2]} price={p[3]} layer={p[4]} opened={p[5]} unrealized={p[8]}")
            total_deployed += float(p[2])
            total_unrealized += float(p[8] or 0)
        print(f"  TOTAL: {len(positions)} positions, deployed=${total_deployed:.2f}, unrealized_pnl=${total_unrealized:.2f}")

        # 2. Check paper_trades schema
        result = await session.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'paper_trades'
            ORDER BY ordinal_position
        """))
        cols = [r[0] for r in result.fetchall()]
        print(f"\n=== PAPER_TRADES SCHEMA ===")
        print(f"  Columns: {cols}")
        print(f"  Has 'strategy': {'strategy' in cols}")

        # 3. All trades (raw SQL)
        result = await session.execute(text("""
            SELECT id, market_condition_id, side, size, price, layer, status, actual_pnl, notes, placed_at, resolved_at
            FROM paper_trades
            ORDER BY placed_at DESC
            LIMIT 50
        """))
        trades = result.fetchall()
        print(f"\n=== TRADES (last 50) ===")
        for t in trades:
            print(f"  id={t[0]} cond={t[1][:20]}... side={t[2]} size=${t[3]} price={t[4]} layer={t[5]} status={t[6]} pnl={t[7]} notes={t[8]} placed={t[9]}")
        print(f"  Shown: {len(trades)}")

        # 4. Count all trades
        result = await session.execute(text("SELECT COUNT(*), SUM(actual_pnl), SUM(size) FROM paper_trades"))
        row = result.fetchone()
        print(f"\n=== TRADE TOTALS ===")
        print(f"  Total trades: {row[0]}")
        print(f"  Total realized P&L: ${row[1]}")
        print(f"  Total invested: ${row[2]}")

        # 5. Trades by status
        result = await session.execute(text("""
            SELECT status, COUNT(*), SUM(size), SUM(actual_pnl)
            FROM paper_trades
            GROUP BY status
            ORDER BY COUNT(*) DESC
        """))
        print(f"\n=== TRADES BY STATUS ===")
        for r in result.fetchall():
            print(f"  {r[0]}: count={r[1]}, total_size=${r[2]}, total_pnl=${r[3]}")

        # 6. Orphan check: positions without matching open trades
        print(f"\n=== ORPHAN POSITION CHECK ===")
        for p in positions:
            result = await session.execute(text("""
                SELECT COUNT(*) FROM paper_trades
                WHERE market_condition_id = :cond AND status = 'open'
            """), {"cond": p[0]})
            trade_count = result.scalar()
            result2 = await session.execute(text("""
                SELECT COUNT(*) FROM paper_trades
                WHERE market_condition_id = :cond
            """), {"cond": p[0]})
            any_trade = result2.scalar()
            status = "ORPHAN (no trades at all!)" if any_trade == 0 else f"trades={any_trade}, open={trade_count}"
            print(f"  {p[0][:24]}... size=${p[2]}  {status}")

        # 7. Recent signals
        result = await session.execute(text("""
            SELECT id, layer, direction, edge, detected_at, details->>'strategy' as strategy,
                   details->>'category' as category
            FROM signals
            ORDER BY detected_at DESC
            LIMIT 20
        """))
        print(f"\n=== RECENT SIGNALS ===")
        for r in result.fetchall():
            print(f"  id={r[0]} layer={r[1]} dir={r[2]} edge={r[3]} strategy={r[5]} cat={r[6]} at={r[4]}")

        # 8. Snapshots
        result = await session.execute(text("""
            SELECT balance, total_value, unrealized_pnl, snapshot_at
            FROM paper_snapshots
            ORDER BY snapshot_at DESC
            LIMIT 10
        """))
        print(f"\n=== RECENT SNAPSHOTS ===")
        for r in result.fetchall():
            print(f"  balance=${r[0]} total_value=${r[1]} unrealized=${r[2]} at={r[3]}")

        # 9. Paper engine state
        result = await session.execute(text("""
            SELECT balance, total_value, unrealized_pnl, snapshot_at
            FROM paper_snapshots
            ORDER BY snapshot_at DESC
            LIMIT 1
        """))
        latest = result.fetchone()
        if latest:
            missing = 2000 - float(latest[0]) - total_deployed
            print(f"\n=== CAPITAL ACCOUNTING ===")
            print(f"  Latest balance: ${latest[0]}")
            print(f"  Deployed in positions: ${total_deployed:.2f}")
            print(f"  Sum: ${float(latest[0]) + total_deployed:.2f}")
            print(f"  Expected: $2,000.00")
            print(f"  Missing: ${missing:.2f}")

        # 10. Check Alembic version
        result = await session.execute(text("SELECT version_num FROM alembic_version"))
        version = result.scalar()
        print(f"\n=== ALEMBIC VERSION ===")
        print(f"  Current: {version}")

        # 11. Check if load_state_from_db was called (look for position creation pattern)
        result = await session.execute(text("""
            SELECT opened_at, COUNT(*) as cnt, SUM(size)
            FROM paper_positions
            GROUP BY opened_at
            ORDER BY opened_at
        """))
        print(f"\n=== POSITION CREATION BATCHES ===")
        for r in result.fetchall():
            print(f"  {r[0]}: {r[1]} positions, ${r[2]} total")


asyncio.run(investigate())
