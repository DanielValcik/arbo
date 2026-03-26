"""Download Historical Trades from Goldsky Subgraph.

Queries the Polymarket Goldsky subgraph directly for trades on
weather temperature markets only. This gives us real on-chain
trade data going back to when weather markets first appeared
(~November 2025).

Unlike the poly_data full dump (6.2 GB), this downloads ONLY
weather market trades (~100 MB) by filtering on specific token IDs.

Usage:
    python3 research/download_goldsky_trades.py

Output:
    research/data/price_history.sqlite  — enriched with trade-based prices
"""

import json
import math
import sqlite3
import ssl
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()
    SSL_CTX.check_hostname = False
    SSL_CTX.verify_mode = ssl.CERT_NONE

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "price_history.sqlite"
EVENTS_CACHE = DATA_DIR / "polymarket_weather_events.json"
LOG_PATH = Path(__file__).parent / "download_goldsky_log.txt"

GOLDSKY_URL = (
    "https://api.goldsky.com/api/public/"
    "project_cl6mb8i9h0003e201j6li0diw/"
    "subgraphs/orderbook-subgraph/0.0.1/gn"
)

BATCH_SIZE = 1000
REQUEST_DELAY = 0.3  # Conservative rate limiting


def log(msg: str):
    print(msg, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


def graphql_query(query_str: str, retries: int = 3) -> dict | None:
    """Execute a GraphQL query against Goldsky subgraph."""
    payload = json.dumps({"query": query_str}).encode()
    for attempt in range(retries):
        req = urllib.request.Request(
            GOLDSKY_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "ArboResearch/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
                result = json.loads(resp.read().decode())
                if "errors" in result:
                    log(f"  GraphQL error: {result['errors']}")
                    return None
                return result.get("data", {})
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                log(f"  Query failed after {retries} attempts: {e}")
                return None
    return None


def fetch_trades_for_token(
    token_id: str, side: str = "maker"
) -> list[dict]:
    """Fetch ALL trades for a specific token ID from Goldsky.

    Paginates through all results using timestamp cursor.
    side: "maker" or "taker" — query makerAssetId or takerAssetId.
    """
    all_trades = []
    last_timestamp = "0"
    field = f"{side}AssetId"

    while True:
        query = f'''query {{
            orderFilledEvents(
                first: {BATCH_SIZE},
                orderBy: timestamp,
                orderDirection: asc,
                where: {{ {field}: "{token_id}", timestamp_gt: "{last_timestamp}" }}
            ) {{
                timestamp
                makerAssetId
                makerAmountFilled
                takerAssetId
                takerAmountFilled
                transactionHash
            }}
        }}'''

        data = graphql_query(query)
        if not data:
            break

        events = data.get("orderFilledEvents", [])
        if not events:
            break

        all_trades.extend(events)
        last_timestamp = events[-1]["timestamp"]

        if len(events) < BATCH_SIZE:
            break

        time.sleep(REQUEST_DELAY)

    return all_trades


def trades_to_prices(
    trades: list[dict], token_id: str
) -> list[dict]:
    """Convert raw trades into price points.

    Each trade has makerAssetId/takerAssetId and amounts.
    For NegRisk weather markets:
    - If makerAssetId == token_id: maker is selling the token
      price = takerAmountFilled / makerAmountFilled (USDC per token)
    - If takerAssetId == token_id: taker is buying the token
      price = makerAmountFilled / takerAmountFilled

    Amounts are in raw units (6 decimals for USDC, variable for tokens).
    """
    prices = []
    for trade in trades:
        ts = int(trade["timestamp"])
        maker_amount = int(trade["makerAmountFilled"])
        taker_amount = int(trade["takerAmountFilled"])

        if maker_amount == 0 or taker_amount == 0:
            continue

        if trade["makerAssetId"] == token_id:
            # Maker has the token, taker pays USDC
            # price = USDC / tokens
            price = taker_amount / maker_amount
        elif trade["takerAssetId"] == token_id:
            # Taker has the token, maker pays USDC
            price = maker_amount / taker_amount
        else:
            continue

        # Sanity check: price should be 0-1 for prediction markets
        if 0.0001 <= price <= 1.0:
            prices.append({"t": ts, "p": round(price, 6)})

    return prices


def aggregate_to_hourly(prices: list[dict]) -> list[dict]:
    """Aggregate tick-level prices to hourly VWAP-like averages."""
    if not prices:
        return []

    by_hour = defaultdict(list)
    for p in prices:
        hour_ts = (p["t"] // 3600) * 3600
        by_hour[hour_ts].append(p["p"])

    result = []
    for hour_ts in sorted(by_hour):
        vals = by_hour[hour_ts]
        avg_price = sum(vals) / len(vals)
        result.append({"t": hour_ts, "p": round(avg_price, 6)})

    return result


def get_existing_tokens_with_goldsky(conn: sqlite3.Connection) -> set[str]:
    """Get tokens that already have Goldsky-sourced prices."""
    # We'll use a separate table or tag to track Goldsky data
    try:
        cursor = conn.execute(
            "SELECT DISTINCT token_id FROM goldsky_prices"
        )
        return {row[0] for row in cursor}
    except sqlite3.OperationalError:
        return set()


def setup_goldsky_table(conn: sqlite3.Connection):
    """Create table for Goldsky trade-derived prices."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS goldsky_trades (
            token_id       TEXT NOT NULL,
            ts             INTEGER NOT NULL,
            price          REAL NOT NULL,
            PRIMARY KEY (token_id, ts)
        );
        CREATE INDEX IF NOT EXISTS idx_goldsky_token
            ON goldsky_trades(token_id);
    """)
    conn.commit()


def main():
    t_start = time.time()

    log(f"\n{'=' * 70}")
    log(f"GOLDSKY TRADE DOWNLOAD — Weather Markets")
    log(f"Start: {datetime.now().isoformat()}")
    log(f"{'=' * 70}\n")

    # Load events
    if not EVENTS_CACHE.exists():
        log("ERROR: No events cache. Run download_price_history.py first.")
        return

    with open(EVENTS_CACHE) as f:
        raw_events = json.load(f)
    log(f"Loaded {len(raw_events)} events from cache")

    # Extract all YES token IDs
    token_to_event = {}  # token_id -> (event_title, event_id)
    for ev in raw_events:
        event_id = str(ev.get("id", ""))
        title = ev.get("title", "")
        for mkt in ev.get("markets", []):
            clob_raw = mkt.get("clobTokenIds", "")
            if isinstance(clob_raw, str):
                try:
                    ids = json.loads(clob_raw)
                except (json.JSONDecodeError, TypeError):
                    continue
            elif isinstance(clob_raw, list):
                ids = clob_raw
            else:
                continue
            if ids:
                token_to_event[str(ids[0])] = (title, event_id)

    log(f"Weather market YES tokens: {len(token_to_event)}")

    # Setup database
    conn = sqlite3.connect(str(DB_PATH))
    setup_goldsky_table(conn)
    already_fetched = get_existing_tokens_with_goldsky(conn)
    tokens_to_fetch = [t for t in token_to_event if t not in already_fetched]

    log(f"Already fetched: {len(already_fetched)}")
    log(f"Tokens to fetch: {len(tokens_to_fetch)}")

    if not tokens_to_fetch:
        log("Nothing to fetch!")
        conn.close()
        return

    # Estimate time
    est_minutes = len(tokens_to_fetch) * REQUEST_DELAY * 2 / 60  # ×2 for maker+taker
    log(f"Estimated time: ~{est_minutes:.0f} minutes\n")

    fetched = 0
    empty = 0
    total_trades = 0
    total_prices = 0
    errors = 0

    for i, token_id in enumerate(tokens_to_fetch):
        try:
            # Fetch trades where this token is on either side
            trades_maker = fetch_trades_for_token(token_id, "maker")
            time.sleep(REQUEST_DELAY)
            trades_taker = fetch_trades_for_token(token_id, "taker")

            all_trades = trades_maker + trades_taker

            if all_trades:
                prices = trades_to_prices(all_trades, token_id)
                hourly = aggregate_to_hourly(prices)

                if hourly:
                    conn.executemany(
                        "INSERT OR IGNORE INTO goldsky_trades "
                        "(token_id, ts, price) VALUES (?, ?, ?)",
                        [(token_id, h["t"], h["p"]) for h in hourly],
                    )
                    total_trades += len(all_trades)
                    total_prices += len(hourly)
                    fetched += 1
                else:
                    empty += 1
            else:
                empty += 1

        except Exception as e:
            errors += 1
            if errors <= 10:
                title = token_to_event.get(token_id, ("?", "?"))[0][:40]
                log(f"  Error: {title}: {e}")

        # Progress
        if (i + 1) % 50 == 0:
            conn.commit()
            pct = (i + 1) / len(tokens_to_fetch) * 100
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(tokens_to_fetch) - i - 1) / rate if rate > 0 else 0
            log(f"  [{pct:5.1f}%] {i+1}/{len(tokens_to_fetch)} | "
                f"{fetched} data, {empty} empty, {errors} err | "
                f"{total_trades} trades → {total_prices} prices | "
                f"ETA: {eta/60:.0f}min")

        time.sleep(REQUEST_DELAY)

    conn.commit()

    # Stats
    elapsed = time.time() - t_start
    log(f"\n{'=' * 70}")
    log(f"DOWNLOAD COMPLETE — {elapsed/60:.1f} minutes")
    log(f"{'=' * 70}")
    log(f"  Tokens with trades: {fetched}")
    log(f"  Tokens empty: {empty}")
    log(f"  Errors: {errors}")
    log(f"  Total raw trades: {total_trades:,}")
    log(f"  Total hourly prices: {total_prices:,}")

    # Date range
    row = conn.execute(
        "SELECT MIN(ts), MAX(ts) FROM goldsky_trades"
    ).fetchone()
    if row and row[0]:
        dt_min = datetime.utcfromtimestamp(row[0])
        dt_max = datetime.utcfromtimestamp(row[1])
        days = (dt_max - dt_min).days
        log(f"  Date range: {dt_min.strftime('%Y-%m-%d')} → "
            f"{dt_max.strftime('%Y-%m-%d')} ({days} days)")

    # Per-city summary using events table
    try:
        rows = conn.execute("""
            SELECT e.city,
                   COUNT(DISTINCT b.token_id) as tokens,
                   COUNT(g.ts) as prices
            FROM events e
            JOIN buckets b ON b.event_id = e.event_id
            JOIN goldsky_trades g ON g.token_id = b.token_id
            WHERE e.city IS NOT NULL
            GROUP BY e.city
            ORDER BY prices DESC
        """).fetchall()
        if rows:
            log(f"\n  {'City':<16} {'Tokens':>7} {'Prices':>8}")
            for city, tokens, prices in rows:
                log(f"  {city:<16} {tokens:>7} {prices:>8}")
    except Exception:
        pass

    conn.close()
    log(f"\nDone. Database: {DB_PATH}")


if __name__ == "__main__":
    main()
