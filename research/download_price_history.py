"""Download Historical Price Data from Polymarket CLOB API.

Fetches hourly price history for all resolved weather temperature markets
using the official /prices-history endpoint. Stores data in SQLite for
efficient querying in backtests.

Data availability: Polymarket retains ~30 days of price history for
resolved markets. Run this script regularly to capture data before
it expires.

Usage:
    python3 research/download_price_history.py [--refresh-events] [--fidelity 60]

Output:
    research/data/price_history.sqlite   — SQLite database
    research/data/price_history_meta.json — Download metadata
"""

import argparse
import json
import math
import os
import re
import sqlite3
import ssl
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

try:
    import certifi

    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()
    SSL_CONTEXT.check_hostname = False
    SSL_CONTEXT.verify_mode = ssl.CERT_NONE

DATA_DIR = Path(__file__).parent / "data"
EVENTS_CACHE = DATA_DIR / "polymarket_weather_events.json"
DB_PATH = DATA_DIR / "price_history.sqlite"
META_PATH = DATA_DIR / "price_history_meta.json"

CLOB_BASE = "https://clob.polymarket.com"
GAMMA_BASE = "https://gamma-api.polymarket.com"

# Rate limiting: conservative 5 req/s
REQUEST_DELAY = 0.2

# ── City mapping ──────────────────────────────────────────────────────

CITY_MAP = {
    "nyc": "nyc",
    "new york": "nyc",
    "chicago": "chicago",
    "london": "london",
    "seoul": "seoul",
    "buenos aires": "buenos_aires",
    "atlanta": "atlanta",
    "toronto": "toronto",
    "ankara": "ankara",
    "são paulo": "sao_paulo",
    "sao paulo": "sao_paulo",
    "miami": "miami",
    "paris": "paris",
    "dallas": "dallas",
    "seattle": "seattle",
    "wellington": "wellington",
    "tokyo": "tokyo",
    "munich": "munich",
    "los angeles": "los_angeles",
    "washington dc": "dc",
    " dc ": "dc",
    "tel aviv": "tel_aviv",
    "lucknow": "lucknow",
}


# ── Parsing helpers ───────────────────────────────────────────────────


def parse_city(title: str) -> str | None:
    """Extract city ID from event title."""
    t = title.lower()
    for pattern, city_id in CITY_MAP.items():
        if pattern in t:
            return city_id
    return None


def parse_target_date(title: str) -> date | None:
    """Extract target date from event title (e.g. 'on March 10')."""
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
        "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    patterns = [
        re.compile(
            r"on\s+(January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+(\d{1,2})",
            re.I,
        ),
        re.compile(
            r"on\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"\s+(\d{1,2})",
            re.I,
        ),
    ]
    for pat in patterns:
        m = pat.search(title)
        if m:
            month = months.get(m.group(1).lower())
            day = int(m.group(2))
            if month:
                for year in [2026, 2025, 2024]:
                    try:
                        d = date(year, month, day)
                        if d <= date.today():
                            return d
                    except ValueError:
                        continue
    return None


def parse_bucket(question: str) -> dict | None:
    """Parse temperature bucket from market question.

    Returns dict with keys: low_val, high_val, unit, bucket_type, low_c, high_c.
    """
    q = question.strip()

    # "between 47-48°F" or "between 8-9°C"
    m = re.search(r"between\s+(-?\d+)\s*-\s*(-?\d+)\s*°\s*([FC])", q, re.I)
    if m:
        low, high, unit = float(m.group(1)), float(m.group(2)), m.group(3).upper()
        return _to_celsius(low, high, unit, "range")

    # "38°F or below" or "8°C or below"
    m = re.search(r"(-?\d+)\s*°\s*([FC])\s+or\s+below", q, re.I)
    if m:
        high, unit = float(m.group(1)), m.group(2).upper()
        return _to_celsius(None, high, unit, "below")

    # "49°F or higher"
    m = re.search(r"(-?\d+)\s*°\s*([FC])\s+or\s+(?:higher|above|more)", q, re.I)
    if m:
        low, unit = float(m.group(1)), m.group(2).upper()
        return _to_celsius(low, None, unit, "above")

    # "be 6°C on" (exact single degree)
    m = re.search(r"be\s+(-?\d+)\s*°\s*([FC])\s+on", q, re.I)
    if m:
        val, unit = float(m.group(1)), m.group(2).upper()
        return _to_celsius(val, val, unit, "exact")

    return None


def _f_to_c(f: float) -> float:
    return (f - 32) * 5 / 9


def _to_celsius(low, high, unit, btype):
    if unit == "F":
        low_c = _f_to_c(low) if low is not None else None
        high_c = _f_to_c(high + 1) if high is not None else None
        if btype == "exact":
            low_c = _f_to_c(low)
            high_c = _f_to_c(low + 1)
        elif btype == "below":
            high_c = _f_to_c(high + 1)
    else:
        low_c = low
        high_c = (high + 1) if high is not None else None
        if btype == "exact":
            high_c = low + 1
        elif btype == "below":
            high_c = high + 1

    return {
        "low_val": low, "high_val": high, "unit": unit,
        "bucket_type": btype, "low_c": low_c, "high_c": high_c,
    }


# ── Database ──────────────────────────────────────────────────────────


def create_database(db_path: Path) -> sqlite3.Connection:
    """Create SQLite database with schema."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS events (
            event_id       TEXT PRIMARY KEY,
            title          TEXT NOT NULL,
            city           TEXT,
            target_date    TEXT,
            start_date     TEXT,
            end_date       TEXT,
            closed_time    TEXT,
            volume         REAL,
            neg_risk       INTEGER,
            neg_risk_id    TEXT,
            n_buckets      INTEGER,
            resolution_src TEXT
        );

        CREATE TABLE IF NOT EXISTS buckets (
            token_id       TEXT PRIMARY KEY,
            token_id_no    TEXT,
            event_id       TEXT NOT NULL,
            condition_id   TEXT,
            question       TEXT,
            low_c          REAL,
            high_c         REAL,
            bucket_type    TEXT,
            unit           TEXT,
            won            INTEGER,
            volume         REAL,
            last_trade_price REAL,
            FOREIGN KEY (event_id) REFERENCES events(event_id)
        );

        CREATE TABLE IF NOT EXISTS prices (
            token_id       TEXT NOT NULL,
            ts             INTEGER NOT NULL,
            price          REAL NOT NULL,
            PRIMARY KEY (token_id, ts)
        );

        CREATE INDEX IF NOT EXISTS idx_prices_token ON prices(token_id);
        CREATE INDEX IF NOT EXISTS idx_buckets_event ON buckets(event_id);
        CREATE INDEX IF NOT EXISTS idx_events_city ON events(city);
        CREATE INDEX IF NOT EXISTS idx_events_date ON events(target_date);
    """)
    conn.commit()
    return conn


# ── API calls ─────────────────────────────────────────────────────────


def _http_get(url: str, retries: int = 3) -> dict | list | None:
    """GET JSON with retry and rate limiting."""
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={
            "User-Agent": "ArboResearch/1.0",
            "Accept": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=30, context=SSL_CONTEXT) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 2 ** (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif e.code >= 500:
                time.sleep(1)
            else:
                return None
        except Exception:
            time.sleep(0.5)
    return None


def fetch_price_history(token_id: str, fidelity: int = 60) -> list[dict]:
    """Fetch price history for a single token from CLOB API.

    Returns list of {t: unix_timestamp, p: price}.
    """
    url = (
        f"{CLOB_BASE}/prices-history"
        f"?market={token_id}&interval=max&fidelity={fidelity}"
    )
    data = _http_get(url)
    if data and isinstance(data, dict):
        return data.get("history", [])
    return []


def fetch_weather_events_fresh(max_pages: int = 100) -> list[dict]:
    """Fetch all resolved weather events from Gamma API (fresh)."""
    print("Fetching resolved weather events from Polymarket Gamma API...")
    all_events = []
    offset = 0
    batch = 50

    for page in range(max_pages):
        url = (
            f"{GAMMA_BASE}/events"
            f"?tag_slug=weather&closed=true&limit={batch}&offset={offset}"
        )
        events = _http_get(url)
        if not events:
            break

        for ev in events:
            title = ev.get("title", "")
            if "highest temperature" in title.lower():
                all_events.append(ev)

        print(f"  Page {page}: {len(events)} events, "
              f"{len(all_events)} temperature events total")
        offset += batch

        if len(events) < batch:
            break
        time.sleep(0.3)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVENTS_CACHE, "w") as f:
        json.dump(all_events, f)
    print(f"Saved {len(all_events)} events to {EVENTS_CACHE}")
    return all_events


def load_events() -> list[dict]:
    """Load events from cache."""
    if not EVENTS_CACHE.exists():
        return fetch_weather_events_fresh()
    with open(EVENTS_CACHE) as f:
        return json.load(f)


# ── Main pipeline ─────────────────────────────────────────────────────


def insert_event(conn: sqlite3.Connection, ev: dict, city: str | None,
                 target_dt: date | None) -> str:
    """Insert event into database, return event_id."""
    event_id = str(ev.get("id", ""))
    conn.execute(
        """INSERT OR REPLACE INTO events
           (event_id, title, city, target_date, start_date, end_date,
            closed_time, volume, neg_risk, neg_risk_id, n_buckets,
            resolution_src)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            event_id,
            ev.get("title", ""),
            city,
            target_dt.isoformat() if target_dt else None,
            ev.get("startDate"),
            ev.get("endDate"),
            ev.get("closedTime"),
            float(ev.get("volume", 0) or 0),
            1 if ev.get("negRisk") or ev.get("enableNegRisk") else 0,
            ev.get("negRiskMarketID"),
            len(ev.get("markets", [])),
            ev.get("resolutionSource"),
        ),
    )
    return event_id


def insert_bucket(conn: sqlite3.Connection, event_id: str, mkt: dict,
                  bucket_info: dict | None) -> str | None:
    """Insert bucket into database, return YES token_id."""
    clob_raw = mkt.get("clobTokenIds", "")
    if isinstance(clob_raw, str):
        try:
            token_ids = json.loads(clob_raw)
        except (json.JSONDecodeError, TypeError):
            return None
    elif isinstance(clob_raw, list):
        token_ids = clob_raw
    else:
        return None

    if not token_ids:
        return None

    yes_token = str(token_ids[0])
    no_token = str(token_ids[1]) if len(token_ids) > 1 else None

    # Determine if this bucket won
    outcome_prices = mkt.get("outcomePrices", "")
    if isinstance(outcome_prices, str):
        try:
            outcome_prices = json.loads(outcome_prices)
        except (json.JSONDecodeError, TypeError):
            outcome_prices = []
    won = 0
    if outcome_prices:
        try:
            won = 1 if float(outcome_prices[0]) > 0.5 else 0
        except (ValueError, TypeError, IndexError):
            pass

    conn.execute(
        """INSERT OR REPLACE INTO buckets
           (token_id, token_id_no, event_id, condition_id, question,
            low_c, high_c, bucket_type, unit, won, volume,
            last_trade_price)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            yes_token,
            no_token,
            event_id,
            mkt.get("conditionId"),
            mkt.get("question", ""),
            bucket_info.get("low_c") if bucket_info else None,
            bucket_info.get("high_c") if bucket_info else None,
            bucket_info.get("bucket_type") if bucket_info else None,
            bucket_info.get("unit") if bucket_info else None,
            won,
            float(mkt.get("volume", 0) or 0),
            float(mkt.get("lastTradePrice", 0) or 0),
        ),
    )
    return yes_token


def insert_prices(conn: sqlite3.Connection, token_id: str,
                  history: list[dict]) -> int:
    """Insert price history, return count of new rows."""
    if not history:
        return 0
    conn.executemany(
        "INSERT OR IGNORE INTO prices (token_id, ts, price) VALUES (?, ?, ?)",
        [(token_id, h["t"], h["p"]) for h in history],
    )
    return len(history)


def get_fetched_tokens(conn: sqlite3.Connection) -> set[str]:
    """Get set of token_ids that already have price data."""
    cursor = conn.execute("SELECT DISTINCT token_id FROM prices")
    return {row[0] for row in cursor}


def main():
    parser = argparse.ArgumentParser(description="Download Polymarket price history")
    parser.add_argument("--refresh-events", action="store_true",
                        help="Re-fetch events from Gamma API")
    parser.add_argument("--fidelity", type=int, default=60,
                        help="Price granularity in minutes (default: 60)")
    parser.add_argument("--max-events", type=int, default=0,
                        help="Limit number of events to process (0=all)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ── 1. Load events ──
    if args.refresh_events or not EVENTS_CACHE.exists():
        raw_events = fetch_weather_events_fresh()
    else:
        raw_events = load_events()
    print(f"\nLoaded {len(raw_events)} temperature events")

    # ── 2. Create database ──
    conn = create_database(DB_PATH)
    already_fetched = get_fetched_tokens(conn)
    print(f"Database: {DB_PATH}")
    print(f"Already fetched: {len(already_fetched)} tokens")

    # ── 3. Parse and insert events + buckets, collect tokens to fetch ──
    # Sort events newest-first: Polymarket only retains ~30 days of price
    # history, so we want to fetch recent events first.
    sorted_events = sorted(
        raw_events,
        key=lambda ev: ev.get("closedTime", "") or "",
        reverse=True,
    )

    tokens_to_fetch = []  # (token_id, event_title, closed_time)
    total_events = 0
    total_buckets = 0
    skipped_no_city = 0

    for ev in sorted_events:
        title = ev.get("title", "")
        city = parse_city(title)
        target_dt = parse_target_date(title)
        closed_time = ev.get("closedTime", "")

        if not city:
            skipped_no_city += 1

        event_id = insert_event(conn, ev, city, target_dt)
        total_events += 1

        for mkt in ev.get("markets", []):
            bucket_info = parse_bucket(mkt.get("question", ""))
            yes_token = insert_bucket(conn, event_id, mkt, bucket_info)
            if yes_token:
                total_buckets += 1
                if yes_token not in already_fetched:
                    tokens_to_fetch.append(
                        (yes_token, title, closed_time)
                    )

    conn.commit()

    if args.max_events > 0:
        tokens_to_fetch = tokens_to_fetch[:args.max_events * 8]

    print(f"\nParsed: {total_events} events, {total_buckets} buckets")
    print(f"Skipped (no city match): {skipped_no_city}")
    print(f"Tokens to fetch: {len(tokens_to_fetch)} "
          f"(skipping {len(already_fetched)} already fetched)")

    if not tokens_to_fetch:
        print("\nNothing new to fetch!")
        _print_stats(conn, t_start)
        conn.close()
        return

    # ── 4. Fetch price history ──
    print(f"\nFetching price history (fidelity={args.fidelity}min)...")
    print(f"Estimated time: ~{len(tokens_to_fetch) * REQUEST_DELAY / 60:.0f} minutes")
    print()

    fetched = 0
    empty = 0
    errors = 0
    total_points = 0
    consecutive_empty = 0
    CONSECUTIVE_EMPTY_STOP = 200  # Stop after 200 consecutive empties

    for i, (token_id, event_title, closed_time) in enumerate(tokens_to_fetch):
        try:
            history = fetch_price_history(token_id, args.fidelity)
            if history:
                n = insert_prices(conn, token_id, history)
                total_points += n
                fetched += 1
                consecutive_empty = 0
            else:
                empty += 1
                consecutive_empty += 1
        except Exception as e:
            errors += 1
            consecutive_empty += 1
            if errors <= 5:
                print(f"  Error fetching {token_id[:20]}...: {e}")

        # Stop early if we've hit the historical data boundary
        if consecutive_empty >= CONSECUTIVE_EMPTY_STOP:
            print(f"\n  Stopping: {CONSECUTIVE_EMPTY_STOP} consecutive "
                  f"empty responses (older events have no price data)")
            break

        # Progress update every 100 tokens
        if (i + 1) % 100 == 0:
            conn.commit()
            pct = (i + 1) / len(tokens_to_fetch) * 100
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (len(tokens_to_fetch) - i - 1)
            print(f"  [{pct:5.1f}%] {i + 1}/{len(tokens_to_fetch)} tokens | "
                  f"{fetched} with data, {empty} empty | "
                  f"{total_points} price points | "
                  f"ETA: {eta / 60:.0f}min")

        time.sleep(REQUEST_DELAY)

    conn.commit()

    print(f"\n{'=' * 60}")
    print(f"Download complete!")
    print(f"  Tokens fetched: {fetched} with data, {empty} empty, {errors} errors")
    print(f"  Total price points: {total_points:,}")

    # ── 5. Save metadata ──
    _print_stats(conn, t_start)
    _save_meta(conn, t_start, fetched, empty, errors, total_points, args.fidelity)

    conn.close()


def _print_stats(conn: sqlite3.Connection, t_start: float):
    """Print database statistics."""
    print(f"\n{'─' * 60}")
    print("DATABASE STATISTICS")
    print(f"{'─' * 60}")

    n_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    n_buckets = conn.execute("SELECT COUNT(*) FROM buckets").fetchone()[0]
    n_prices = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    n_tokens_with_data = conn.execute(
        "SELECT COUNT(DISTINCT token_id) FROM prices"
    ).fetchone()[0]

    print(f"  Events:               {n_events:,}")
    print(f"  Buckets (tokens):     {n_buckets:,}")
    print(f"  Tokens with prices:   {n_tokens_with_data:,}")
    print(f"  Total price points:   {n_prices:,}")

    # Date range
    row = conn.execute(
        "SELECT MIN(ts), MAX(ts) FROM prices"
    ).fetchone()
    if row and row[0]:
        dt_min = datetime.utcfromtimestamp(row[0])
        dt_max = datetime.utcfromtimestamp(row[1])
        days = (dt_max - dt_min).days
        print(f"  Price data range:     {dt_min.strftime('%Y-%m-%d')} → "
              f"{dt_max.strftime('%Y-%m-%d')} ({days} days)")

    # Per-city stats
    rows = conn.execute("""
        SELECT e.city, COUNT(DISTINCT e.event_id) as events,
               COUNT(DISTINCT b.token_id) as buckets,
               COUNT(p.ts) as prices,
               SUM(b.won) as wins
        FROM events e
        JOIN buckets b ON b.event_id = e.event_id
        LEFT JOIN prices p ON p.token_id = b.token_id
        WHERE e.city IS NOT NULL
        GROUP BY e.city
        ORDER BY events DESC
    """).fetchall()

    if rows:
        print(f"\n  {'City':<16} {'Events':>7} {'Buckets':>8} "
              f"{'Prices':>8} {'Won':>5}")
        for city, events, buckets, prices, wins in rows:
            print(f"  {city:<16} {events:>7} {buckets:>8} "
                  f"{prices:>8} {wins or 0:>5}")

    # Events with price data
    events_with_prices = conn.execute("""
        SELECT COUNT(DISTINCT e.event_id)
        FROM events e
        JOIN buckets b ON b.event_id = e.event_id
        JOIN prices p ON p.token_id = b.token_id
    """).fetchone()[0]
    print(f"\n  Events with price data: {events_with_prices}/{n_events}")

    elapsed = time.time() - t_start
    db_size = DB_PATH.stat().st_size / (1024 * 1024) if DB_PATH.exists() else 0
    print(f"  Database size:        {db_size:.1f} MB")
    print(f"  Elapsed time:         {elapsed / 60:.1f} min")


def _save_meta(conn, t_start, fetched, empty, errors, total_points, fidelity):
    """Save download metadata to JSON."""
    n_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    n_prices = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]

    row = conn.execute("SELECT MIN(ts), MAX(ts) FROM prices").fetchone()
    price_range = None
    if row and row[0]:
        price_range = {
            "start": datetime.utcfromtimestamp(row[0]).isoformat(),
            "end": datetime.utcfromtimestamp(row[1]).isoformat(),
        }

    meta = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "fidelity_minutes": fidelity,
        "events_total": n_events,
        "tokens_fetched": fetched,
        "tokens_empty": empty,
        "tokens_errors": errors,
        "total_price_points": n_prices,
        "price_data_range": price_range,
        "elapsed_seconds": round(time.time() - t_start, 1),
        "source": "Polymarket CLOB /prices-history",
        "notes": (
            "Price history is only available for ~30 days after market close. "
            "Run this script regularly to capture data before it expires. "
            "Each price point has {t: unix_timestamp, p: YES_price}."
        ),
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {META_PATH}")


if __name__ == "__main__":
    main()
