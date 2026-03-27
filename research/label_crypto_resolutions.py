"""Label crypto market resolutions using Binance klines.

For each closed market, determines if YES or NO won based on:
- Daily "above": Binance 1m candle CLOSE at resolution time >= strike → YES won
- Monthly "hit": ANY Binance 1m candle HIGH reached strike during market period → YES won

Adds 'won' column to markets table.

Usage:
    python3 research/label_crypto_resolutions.py
"""

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "crypto_price_pmd.sqlite"


def main():
    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Add won column if not exists
    try:
        conn.execute("ALTER TABLE markets ADD COLUMN won INTEGER")
        conn.commit()
        print("Added 'won' column to markets table")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Get all closed markets with strike prices
    markets = conn.execute("""
        SELECT market_id, asset, question, strike_price, direction, market_type,
               end_date, status
        FROM markets
        WHERE status = 'closed' AND strike_price IS NOT NULL
    """).fetchall()

    print(f"Processing {len(markets)} closed markets with strikes...")

    labeled = 0
    unlabeled = 0
    yes_wins = 0
    no_wins = 0

    for m in markets:
        market_id = m["market_id"]
        asset = m["asset"]
        strike = m["strike_price"]
        direction = m["direction"] or "above"
        market_type = m["market_type"] or "daily_above"
        end_date = m["end_date"]

        if not end_date or not asset:
            unlabeled += 1
            continue

        symbol = f"{asset}USDT"

        # Parse resolution time
        try:
            res_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            res_ts = int(res_dt.timestamp())
        except (ValueError, TypeError):
            unlabeled += 1
            continue

        won = None

        if market_type == "monthly_hit":
            # Monthly hit: check if ANY candle high reached strike during market period
            start_date = m.get("start_date", "")
            if start_date:
                try:
                    start_ts = int(datetime.fromisoformat(
                        start_date.replace("Z", "+00:00")).timestamp())
                except (ValueError, TypeError):
                    start_ts = res_ts - 30 * 86400  # Default 30 days back
            else:
                start_ts = res_ts - 30 * 86400

            if direction == "above":
                row = conn.execute("""
                    SELECT MAX(high) FROM binance_klines
                    WHERE symbol = ? AND ts >= ? AND ts <= ?
                """, (symbol, start_ts, res_ts)).fetchone()
                if row and row[0] is not None:
                    won = 1 if row[0] >= strike else 0
            else:
                row = conn.execute("""
                    SELECT MIN(low) FROM binance_klines
                    WHERE symbol = ? AND ts >= ? AND ts <= ?
                """, (symbol, start_ts, res_ts)).fetchone()
                if row and row[0] is not None:
                    won = 1 if row[0] <= strike else 0
        else:
            # Daily above: check Binance close at resolution time
            # Get closest 1m candle close at or just before resolution
            row = conn.execute("""
                SELECT close FROM binance_klines
                WHERE symbol = ? AND ts <= ? AND ts >= ?
                ORDER BY ts DESC LIMIT 1
            """, (symbol, res_ts, res_ts - 300)).fetchone()  # Within 5 min

            if row and row["close"] is not None:
                binance_close = row["close"]
                if direction == "above":
                    won = 1 if binance_close >= strike else 0
                else:
                    won = 1 if binance_close <= strike else 0

        if won is not None:
            conn.execute("UPDATE markets SET won = ? WHERE market_id = ?",
                         (won, market_id))
            labeled += 1
            if won:
                yes_wins += 1
            else:
                no_wins += 1
        else:
            unlabeled += 1

    conn.commit()

    # Stats
    total = conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
    total_labeled = conn.execute("SELECT COUNT(*) FROM markets WHERE won IS NOT NULL").fetchone()[0]
    total_yes = conn.execute("SELECT COUNT(*) FROM markets WHERE won = 1").fetchone()[0]
    total_no = conn.execute("SELECT COUNT(*) FROM markets WHERE won = 0").fetchone()[0]

    print(f"\nResults:")
    print(f"  Total markets:  {total}")
    print(f"  Labeled:        {total_labeled} ({total_labeled/max(total,1)*100:.0f}%)")
    print(f"  YES won:        {total_yes}")
    print(f"  NO won:         {total_no}")
    print(f"  Unlabeled:      {total - total_labeled}")
    print(f"  This run: labeled={labeled}, yes={yes_wins}, no={no_wins}, skip={unlabeled}")

    # Per-asset breakdown
    for row in conn.execute("""
        SELECT asset, market_type,
               COUNT(*) as total,
               SUM(won IS NOT NULL) as labeled,
               SUM(won = 1) as yes_wins,
               SUM(won = 0) as no_wins
        FROM markets
        GROUP BY asset, market_type
    """).fetchall():
        print(f"  {row[0]} {row[1]}: {row[2]} total, {row[3]} labeled, "
              f"{row[4] or 0} YES / {row[5] or 0} NO")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
