"""
Fetch Chainlink resolution truth for all BTC 5-min Up/Down markets.

Downloads actual Polymarket market outcomes (Chainlink oracle) for every
5-min BTC window that overlaps with our Binance kline dataset.

Stores in SQLite: chainlink_resolutions (ts INTEGER PK, up_won INTEGER)
where ts = event start timestamp, up_won = 1 (Up won) or 0 (Down won).

Usage:
    python3 research/fetch_chainlink_resolutions.py
"""

import json
import sqlite3
import ssl
import time
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "crypto_price_pmd.sqlite"

GAMMA_URL = "https://gamma-api.polymarket.com"

# SSL context (macOS cert issue workaround)
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE


def get_kline_range() -> tuple[int, int]:
    """Get min/max timestamps from Binance klines."""
    conn = sqlite3.connect(str(DB_PATH))
    row = conn.execute(
        "SELECT MIN(ts), MAX(ts) FROM binance_klines WHERE symbol='BTCUSDT'"
    ).fetchone()
    conn.close()
    return row[0], row[1]


def fetch_resolution(start_ts: int) -> int | None:
    """Fetch market resolution from Gamma API.

    Returns 1 if Up won, 0 if Down won, None if no data/not resolved.
    """
    slug = f"btc-updown-5m-{start_ts}"
    url = f"{GAMMA_URL}/events?slug={slug}"
    req = urllib.request.Request(url, headers={"User-Agent": "Arbo/1.0"})
    try:
        data = json.loads(urllib.request.urlopen(req, timeout=10, context=_ssl_ctx).read())
        if not data:
            return None
        event = data[0] if isinstance(data, list) else data
        markets = event.get("markets", [])
        if not markets:
            return None
        m = markets[0]
        if not m.get("closed"):
            return None
        outcomes = json.loads(m.get("outcomes", "[]"))
        prices = json.loads(m.get("outcomePrices", "[]"))
        for i, o in enumerate(outcomes):
            if o.lower() == "up" and i < len(prices):
                if prices[i] == "1":
                    return 1  # Up won
                for j, o2 in enumerate(outcomes):
                    if o2.lower() == "down" and j < len(prices) and prices[j] == "1":
                        return 0  # Down won
                return None  # Not resolved
    except Exception:
        return None
    return None


def main() -> None:
    min_ts, max_ts = get_kline_range()
    print(f"Binance klines range: {min_ts} → {max_ts}")

    # Align to 5-min boundaries
    first_window = (min_ts // 300) * 300
    last_window = (max_ts // 300) * 300

    total_windows = (last_window - first_window) // 300
    print(f"Total 5-min windows to check: {total_windows:,}")

    # Create table
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chainlink_resolutions (
            ts INTEGER PRIMARY KEY,
            up_won INTEGER NOT NULL
        )
    """)
    conn.commit()

    # Check existing
    existing = set(
        r[0] for r in conn.execute("SELECT ts FROM chainlink_resolutions").fetchall()
    )
    print(f"Already have {len(existing):,} resolutions")

    fetched = 0
    errors = 0
    skipped = 0
    batch = []

    for ts in range(first_window, last_window + 1, 300):
        if ts in existing:
            skipped += 1
            continue

        result = fetch_resolution(ts)
        if result is not None:
            batch.append((ts, result))
            fetched += 1
        else:
            errors += 1

        # Progress every 500
        total_done = fetched + errors + skipped
        if total_done % 500 == 0 and total_done > 0:
            pct = total_done / total_windows * 100
            print(
                f"  [{pct:5.1f}%] done={total_done:,} fetched={fetched:,} "
                f"errors={errors:,} skipped={skipped:,}"
            )

        # Batch insert every 100
        if len(batch) >= 100:
            conn.executemany(
                "INSERT OR REPLACE INTO chainlink_resolutions (ts, up_won) VALUES (?, ?)",
                batch,
            )
            conn.commit()
            batch = []

        # Rate limit: ~5 req/s
        time.sleep(0.2)

    # Final batch
    if batch:
        conn.executemany(
            "INSERT OR REPLACE INTO chainlink_resolutions (ts, up_won) VALUES (?, ?)",
            batch,
        )
        conn.commit()

    total_in_db = conn.execute("SELECT COUNT(*) FROM chainlink_resolutions").fetchone()[0]
    conn.close()

    print(f"\nDone! Fetched {fetched:,}, errors {errors:,}, skipped {skipped:,}")
    print(f"Total resolutions in DB: {total_in_db:,}")


if __name__ == "__main__":
    main()
