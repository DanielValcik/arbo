"""Repair B3 trades with missing live resolution data.

Root cause: check_exits() returned early when _open_positions was empty,
preventing _live_holding positions from being resolved in the DB. This left
trades with live_exit_price=0 and live_exit_status="resolution" even though
the live position actually auto-resolved at $1 (win) or $0 (loss) on Polymarket.

This script:
1. Finds B3 trades with fake resolution data (live_exit_price=0,
   live_exit_status="resolution", but exit_reason != "resolution")
2. Fetches BTC close price at each event_end_ts from Binance 1min klines
3. Determines actual resolution (BTC >= btc_at_start → UP wins, else DOWN wins)
4. Updates live_exit_price in DB

Usage:
    python -m scripts.fix_b3_live_exits           # dry-run (default)
    python -m scripts.fix_b3_live_exits --commit   # actually modify DB
"""

import asyncio
import sys

import aiohttp
import sqlalchemy as sa

from arbo.utils.db import PaperTrade as PT
from arbo.utils.db import get_session_factory
from arbo.utils.logger import get_logger

logger = get_logger("fix_b3_live_exits")

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


async def fetch_btc_at_timestamps(
    timestamps: list[float],
) -> dict[float, float]:
    """Fetch BTC close price at each Unix timestamp from Binance 1m klines.

    Returns mapping {timestamp → btc_close_price}.
    """
    result: dict[float, float] = {}
    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        for ts in sorted(set(timestamps)):
            # Fetch 1-min kline covering the resolution timestamp.
            # startTime = 1 min before, endTime = 1 min after → 2-3 klines.
            start_ms = int(ts * 1000) - 60_000
            end_ms = int(ts * 1000) + 60_000
            params = {
                "symbol": "BTCUSDT",
                "interval": "1m",
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 3,
            }
            try:
                async with session.get(BINANCE_KLINES_URL, params=params) as resp:
                    if resp.status != 200:
                        logger.warning("binance_kline_error", ts=ts, status=resp.status)
                        continue
                    klines = await resp.json()
                    if not klines:
                        continue
                    # Pick the kline whose open_time is closest to (but ≤) ts.
                    # Kline format: [open_time_ms, open, high, low, close, ...]
                    best_close: float | None = None
                    best_diff = float("inf")
                    for k in klines:
                        open_ts = k[0] / 1000.0
                        close_ts = open_ts + 60.0  # end of this 1-min candle
                        # We want the candle that contains event_end_ts
                        if open_ts <= ts < close_ts:
                            best_close = float(k[4])
                            break
                        diff = abs(ts - open_ts)
                        if diff < best_diff:
                            best_diff = diff
                            best_close = float(k[4])
                    if best_close is not None:
                        result[ts] = best_close
            except Exception as e:
                logger.warning("binance_fetch_error", ts=ts, error=str(e))
            # Binance rate limit: 1200 req/min → 0.1s between requests is safe
            await asyncio.sleep(0.1)

    return result


async def fix_b3_live_exits(dry_run: bool = True) -> None:
    """Find and repair B3 trades with fake live_exit_price=0."""
    factory = get_session_factory()

    async with factory() as session:
        # ── Step 1: Find stale trades ──
        # Criteria: B3 trade, paper exited early (not resolution),
        # has live entry, has fake live_exit_status="resolution" with price=0
        result = await session.execute(
            sa.select(PT)
            .where(PT.strategy == "B3")
            .where(PT.status.in_(["won", "lost", "sold"]))
            .order_by(PT.placed_at.asc())
        )
        all_trades = result.scalars().all()

        stale: list[PT] = []
        for t in all_trades:
            d = t.trade_details or {}
            if (
                d.get("live_fill_status") in ("filled", "partial")
                and d.get("live_entry_price", 0) > 0
                and d.get("live_entry_shares", 0) > 0
                and d.get("live_exit_status") == "resolution"
                and d.get("live_exit_price", -1) == 0
                and t.exit_reason != "resolution"  # Paper exited early → bug
                and d.get("event_end_ts")
                and d.get("btc_at_start")
                and d.get("direction")
            ):
                stale.append(t)

        if not stale:
            print("✓ No stale B3 trades found. Nothing to repair.")
            return

        print(f"Found {len(stale)} stale B3 trades to repair:\n")

        # ── Step 2: Fetch BTC prices at resolution times ──
        event_end_timestamps = [t.trade_details["event_end_ts"] for t in stale]
        unique_ts = sorted(set(event_end_timestamps))
        print(f"Fetching BTC prices for {len(unique_ts)} unique resolution times...")
        btc_prices = await fetch_btc_at_timestamps(unique_ts)
        print(f"Got BTC prices for {len(btc_prices)}/{len(unique_ts)} timestamps\n")

        # ── Step 3: Determine resolution and update ──
        fixed = 0
        skipped = 0
        total_old_pnl = 0.0
        total_new_pnl = 0.0

        for t in stale:
            d = t.trade_details
            event_end = d["event_end_ts"]
            btc_start = d["btc_at_start"]
            direction = d["direction"]  # "up" or "down"
            live_entry = d["live_entry_price"]
            live_shares = d["live_entry_shares"]

            btc_at_end = btc_prices.get(event_end)
            if btc_at_end is None:
                placed = t.placed_at.strftime("%m/%d %H:%M") if t.placed_at else "?"
                print(f"  SKIP: No BTC price for event_end_ts={event_end} (trade {t.id}, {placed})")
                skipped += 1
                continue

            # Resolution logic (matches strategy_b3.py check_exits)
            resolved_up = btc_at_end >= btc_start
            won = (direction == "up" and resolved_up) or (direction == "down" and not resolved_up)
            actual_exit_price = 1.0 if won else 0.0

            old_pnl = (0.0 - live_entry) * live_shares  # What chart showed (bug)
            new_pnl = (actual_exit_price - live_entry) * live_shares

            total_old_pnl += old_pnl
            total_new_pnl += new_pnl

            placed = t.placed_at.strftime("%m/%d %H:%M") if t.placed_at else "?"
            outcome = "WON $1" if won else "LOST $0"
            print(
                f"  #{t.id:>4} ({placed}) {direction.upper():>4} "
                f"BTC ${btc_start:>10,.2f}→${btc_at_end:>10,.2f}  "
                f"{outcome:>7}  "
                f"live P&L: ${old_pnl:>+8.2f} → ${new_pnl:>+8.2f}  "
                f"(exit_reason={t.exit_reason})"
            )

            if not dry_run:
                t.trade_details = {
                    **d,
                    "live_exit_price": actual_exit_price,
                    "live_exit_shares": live_shares,
                    "live_exit_status": "resolution",
                    "live_exit_latency_ms": 0,
                    "repair_note": f"Fixed by fix_b3_live_exits.py. BTC at resolution: ${btc_at_end:,.2f}",
                }
            fixed += 1

        # ── Step 4: Commit ──
        print(f"\n{'='*60}")
        print(f"Repair summary:")
        print(f"  Trades to fix:  {fixed}")
        print(f"  Skipped:        {skipped}")
        print(f"  Old chart P&L:  ${total_old_pnl:>+.2f}  (bug: all counted as $0 exit)")
        print(f"  New chart P&L:  ${total_new_pnl:>+.2f}  (correct resolution)")
        print(f"  P&L correction: ${total_new_pnl - total_old_pnl:>+.2f}")

        if dry_run:
            print(f"\n  DRY RUN — no changes made. Run with --commit to apply.")
        elif fixed > 0:
            await session.commit()
            print(f"\n  ✓ {fixed} trades updated in DB.")
        else:
            print(f"\n  Nothing to commit.")


if __name__ == "__main__":
    commit = "--commit" in sys.argv
    if commit:
        print("MODE: COMMIT (will modify database)\n")
    else:
        print("MODE: DRY RUN (read-only, pass --commit to apply)\n")
    asyncio.run(fix_b3_live_exits(dry_run=not commit))
