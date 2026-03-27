"""Download Historical Crypto Price Data from Polymarket + Binance.

Fetches resolved crypto price prediction markets from Polymarket Gamma API,
downloads price trajectories from CLOB /prices-history, and corresponding
Binance 1-minute klines. Stores everything in SQLite for backtesting.

Market types:
- Daily "Above": "Will BTC be above $X on date, 12PM ET?" (Binance close)
- Monthly "Hit": "What price will BTC hit in month?" (any candle touch)

Usage:
    python3 research/download_crypto_price_history.py [--refresh-events] [--fidelity 10]

Output:
    research/data/crypto_price_pmd.sqlite — SQLite database
"""

import argparse
import json
import os
import re
import sqlite3
import ssl
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

try:
    import certifi

    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()
    SSL_CONTEXT.check_hostname = False
    SSL_CONTEXT.verify_mode = ssl.CERT_NONE

DATA_DIR = Path(__file__).parent / "data"
EVENTS_CACHE = DATA_DIR / "polymarket_crypto_events.json"
DB_PATH = DATA_DIR / "crypto_price_pmd.sqlite"

CLOB_BASE = "https://clob.polymarket.com"
GAMMA_BASE = "https://gamma-api.polymarket.com"
BINANCE_BASE = "https://api.binance.com"

# Rate limiting
REQUEST_DELAY = 0.2  # 5 req/s for Polymarket
BINANCE_DELAY = 0.05  # 20 req/s for Binance (conservative, limit is 1200 weight/min)

# ── Asset mapping ──────────────────────────────────────────────────

CRYPTO_KEYWORDS: dict[str, tuple[str, str]] = {
    # keyword -> (asset, binance_symbol)
    "bitcoin": ("BTC", "BTCUSDT"),
    "btc": ("BTC", "BTCUSDT"),
    "ethereum": ("ETH", "ETHUSDT"),
    "eth": ("ETH", "ETHUSDT"),
    "solana": ("SOL", "SOLUSDT"),
    "sol": ("SOL", "SOLUSDT"),
    "xrp": ("XRP", "XRPUSDT"),
    "ripple": ("XRP", "XRPUSDT"),
    "dogecoin": ("DOGE", "DOGEUSDT"),
    "doge": ("DOGE", "DOGEUSDT"),
    "bnb": ("BNB", "BNBUSDT"),
    "cardano": ("ADA", "ADAUSDT"),
    "ada": ("ADA", "ADAUSDT"),
}

# Strike price patterns
_STRIKE_PATTERNS = [
    re.compile(r"\$([0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?)"),  # $95,000
    re.compile(r"\$([0-9]+(?:\.[0-9]+)?)\s*k", re.IGNORECASE),  # $95k
    re.compile(r"\$([0-9]+(?:\.[0-9]+)?)"),  # $95000
]

# Month name -> number
_MONTH_MAP: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


# ── Parsing helpers ───────────────────────────────────────────────


def parse_asset(title: str) -> tuple[str, str] | None:
    """Extract (asset, binance_symbol) from event/market title."""
    t = title.lower()
    for keyword, (asset, symbol) in CRYPTO_KEYWORDS.items():
        if keyword in t:
            return asset, symbol
    return None


def parse_strike(question: str) -> Decimal | None:
    """Extract strike price from question text."""
    for pattern in _STRIKE_PATTERNS:
        match = pattern.search(question)
        if match:
            value_str = match.group(1).replace(",", "")
            try:
                value = Decimal(value_str)
                # Handle k notation
                full_match = question[match.start() : match.end() + 2]
                if re.search(r"k\b", full_match, re.IGNORECASE):
                    value *= 1000
                return value
            except Exception:
                continue
    return None


def parse_expiry_date(text: str) -> date | None:
    """Extract expiry date from event/market title."""
    patterns = [
        re.compile(
            r"(?:by|on|before)?\s*"
            r"(January|February|March|April|May|June|July|August|September|"
            r"October|November|December)\s+(\d{1,2})(?:,?\s*(\d{4}))?",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:by|on|before)?\s*"
            r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"\s+(\d{1,2})(?:,?\s*(\d{4}))?",
            re.IGNORECASE,
        ),
    ]
    for pat in patterns:
        m = pat.search(text)
        if m:
            month_str = m.group(1).lower()
            day = int(m.group(2))
            year = int(m.group(3)) if m.group(3) else None
            month = _MONTH_MAP.get(month_str)
            if month:
                # Try recent years
                years = [year] if year else [2026, 2025, 2024]
                for y in years:
                    try:
                        return date(y, month, day)
                    except ValueError:
                        continue
    return None


def classify_market_type(title: str, question: str) -> str:
    """Classify market as daily_above, monthly_hit, 5min, 15min, or unknown."""
    t = (title + " " + question).lower()

    if "5 min" in t or "5min" in t:
        return "5min"
    if "15 min" in t or "15min" in t:
        return "15min"
    if "up or down" in t:
        return "5min"
    if "what price" in t and "hit" in t:
        return "monthly_hit"
    if "above" in t or "below" in t:
        return "daily_above"
    return "unknown"


def parse_direction(question: str) -> str:
    """Parse direction from market question."""
    q = question.lower()
    if "below" in q or "down" in q:
        return "below"
    return "above"


# ── Database ──────────────────────────────────────────────────────


def create_database(db_path: Path) -> sqlite3.Connection:
    """Create SQLite database with crypto price schema."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS events (
            event_id       TEXT PRIMARY KEY,
            title          TEXT NOT NULL,
            asset          TEXT,
            market_type    TEXT,
            resolution_date TEXT,
            start_date     TEXT,
            end_date       TEXT,
            closed_time    TEXT,
            volume         REAL,
            neg_risk       INTEGER,
            n_buckets      INTEGER
        );

        CREATE TABLE IF NOT EXISTS buckets (
            token_id       TEXT PRIMARY KEY,
            token_id_no    TEXT,
            event_id       TEXT NOT NULL,
            condition_id   TEXT,
            question       TEXT,
            strike_price   REAL,
            direction      TEXT,
            won            INTEGER,
            volume         REAL,
            last_trade_price REAL,
            fee_enabled    INTEGER DEFAULT 0,
            FOREIGN KEY (event_id) REFERENCES events(event_id)
        );

        CREATE TABLE IF NOT EXISTS prices (
            token_id       TEXT NOT NULL,
            ts             INTEGER NOT NULL,
            price          REAL NOT NULL,
            PRIMARY KEY (token_id, ts)
        );

        CREATE TABLE IF NOT EXISTS binance_klines (
            symbol         TEXT NOT NULL,
            ts             INTEGER NOT NULL,
            open           REAL NOT NULL,
            high           REAL NOT NULL,
            low            REAL NOT NULL,
            close          REAL NOT NULL,
            volume         REAL NOT NULL,
            PRIMARY KEY (symbol, ts)
        );

        CREATE INDEX IF NOT EXISTS idx_prices_token ON prices(token_id);
        CREATE INDEX IF NOT EXISTS idx_buckets_event ON buckets(event_id);
        CREATE INDEX IF NOT EXISTS idx_events_asset ON events(asset);
        CREATE INDEX IF NOT EXISTS idx_events_type ON events(market_type);
        CREATE INDEX IF NOT EXISTS idx_klines_symbol ON binance_klines(symbol);
    """)
    conn.commit()
    return conn


# ── API calls ─────────────────────────────────────────────────────


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


def fetch_price_history(token_id: str, fidelity: int = 10) -> list[dict]:
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


def _is_crypto_price_event(ev: dict) -> bool:
    """Check if an event is a crypto price prediction market."""
    title = ev.get("title", "").lower()
    has_crypto = any(kw in title for kw in [
        "bitcoin", "btc", "ethereum", "eth", "solana", "xrp", "doge", "bnb",
    ])
    has_price = any(kw in title for kw in [
        "above", "below", "price", "hit",
    ])
    has_updown = "up or down" in title
    return has_crypto and (has_price or has_updown)


def fetch_crypto_events_fresh(max_pages: int = 200) -> list[dict]:
    """Fetch crypto price events from Gamma API.

    Strategy: fetch RECENTLY closed events (last 30 days) + active events.
    Polymarket only retains ~30 days of price history, so older events
    are useless for backtesting.
    """
    all_events: list[dict] = []
    seen_ids: set[str] = set()

    def _add_events(events: list[dict]) -> int:
        added = 0
        for ev in events:
            eid = ev.get("id", "")
            if eid in seen_ids:
                continue
            if _is_crypto_price_event(ev):
                all_events.append(ev)
                seen_ids.add(eid)
                added += 1
        return added

    # ── 1. Recently closed events (have price history) ──
    print("Fetching RECENTLY CLOSED crypto events (last 30 days)...")
    cutoff = datetime.now(timezone.utc) - timedelta(days=35)
    cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")

    offset = 0
    batch = 50
    for page in range(max_pages):
        url = (
            f"{GAMMA_BASE}/events"
            f"?closed=true&limit={batch}&offset={offset}"
            f"&order=closedTime&ascending=false"
        )
        events = _http_get(url)
        if not events:
            break

        _add_events(events)

        # Check if we've gone past 35 days — stop fetching
        oldest_closed = None
        for ev in events:
            ct = ev.get("closedTime", "")
            if ct and (oldest_closed is None or ct < oldest_closed):
                oldest_closed = ct
        if oldest_closed and oldest_closed < cutoff_str:
            print(f"  Page {page}: reached events older than 35 days, stopping")
            break

        print(f"  Page {page}: {len(events)} events, "
              f"{len(all_events)} crypto price events total")
        offset += batch

        if len(events) < batch:
            break
        time.sleep(0.3)

    print(f"  Recently closed: {len(all_events)} crypto price events")

    # ── 2. Active events (for paper/live trading) ──
    print("\nFetching ACTIVE crypto events...")
    offset = 0
    for page in range(20):
        url = (
            f"{GAMMA_BASE}/events"
            f"?active=true&closed=false&limit={batch}&offset={offset}"
        )
        events = _http_get(url)
        if not events:
            break

        added = _add_events(events)
        print(f"  Page {page}: {len(events)} events, +{added} crypto, "
              f"{len(all_events)} total")
        offset += batch

        if len(events) < batch:
            break
        time.sleep(0.3)

    # ── 3. Tag-based: recently closed crypto ──
    print("\nFetching via tag_slug=crypto (recent only)...")
    offset = 0
    for page in range(max_pages):
        url = (
            f"{GAMMA_BASE}/events"
            f"?tag_slug=crypto&closed=true&limit={batch}&offset={offset}"
            f"&order=closedTime&ascending=false"
        )
        events = _http_get(url)
        if not events:
            break

        added = _add_events(events)

        # Stop at 35 day boundary
        oldest_closed = None
        for ev in events:
            ct = ev.get("closedTime", "")
            if ct and (oldest_closed is None or ct < oldest_closed):
                oldest_closed = ct
        if oldest_closed and oldest_closed < cutoff_str:
            print(f"  Tag page {page}: reached 35-day boundary, stopping")
            break

        print(f"  Tag page {page}: +{added} new, {len(all_events)} total")
        offset += batch

        if len(events) < batch:
            break
        time.sleep(0.3)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVENTS_CACHE, "w") as f:
        json.dump(all_events, f)
    print(f"\nSaved {len(all_events)} crypto events to {EVENTS_CACHE}")
    return all_events


def load_events() -> list[dict]:
    """Load events from cache."""
    if not EVENTS_CACHE.exists():
        return fetch_crypto_events_fresh()
    with open(EVENTS_CACHE) as f:
        return json.load(f)


# ── Binance klines ────────────────────────────────────────────────


def fetch_binance_klines(
    symbol: str, start_ms: int, end_ms: int, interval: str = "1m",
) -> list[list]:
    """Fetch Binance klines (OHLCV) with pagination.

    Returns raw kline arrays: [open_time, open, high, low, close, volume, ...]
    """
    all_klines: list[list] = []
    current_start = start_ms

    while current_start < end_ms:
        url = (
            f"{BINANCE_BASE}/api/v3/klines"
            f"?symbol={symbol}&interval={interval}"
            f"&startTime={current_start}&endTime={end_ms}&limit=1000"
        )
        data = _http_get(url)
        if not data or not isinstance(data, list) or len(data) == 0:
            break

        all_klines.extend(data)
        # Next batch starts after last candle
        last_close_time = int(data[-1][6])  # close_time field
        current_start = last_close_time + 1

        if len(data) < 1000:
            break
        time.sleep(BINANCE_DELAY)

    return all_klines


def insert_klines(conn: sqlite3.Connection, symbol: str,
                  klines: list[list]) -> int:
    """Insert Binance klines into database. Returns count."""
    if not klines:
        return 0
    rows = []
    for k in klines:
        ts = int(k[0]) // 1000  # Convert ms -> seconds
        rows.append((
            symbol, ts,
            float(k[1]),  # open
            float(k[2]),  # high
            float(k[3]),  # low
            float(k[4]),  # close
            float(k[5]),  # volume
        ))
    conn.executemany(
        "INSERT OR IGNORE INTO binance_klines "
        "(symbol, ts, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    return len(rows)


# ── Main pipeline ─────────────────────────────────────────────────


def insert_event(conn: sqlite3.Connection, ev: dict) -> str | None:
    """Parse and insert crypto event. Returns event_id or None."""
    title = ev.get("title", "")
    event_id = str(ev.get("id", ""))

    asset_info = parse_asset(title)
    if not asset_info:
        return None
    asset, binance_symbol = asset_info

    # Determine market type from first market question or title
    markets = ev.get("markets", [])
    first_q = markets[0].get("question", "") if markets else ""
    market_type = classify_market_type(title, first_q)

    resolution_date = parse_expiry_date(title)
    if not resolution_date and markets:
        resolution_date = parse_expiry_date(first_q)

    conn.execute(
        """INSERT OR REPLACE INTO events
           (event_id, title, asset, market_type, resolution_date,
            start_date, end_date, closed_time, volume, neg_risk, n_buckets)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            event_id,
            title,
            asset,
            market_type,
            resolution_date.isoformat() if resolution_date else None,
            ev.get("startDate"),
            ev.get("endDate"),
            ev.get("closedTime"),
            float(ev.get("volume", 0) or 0),
            1 if ev.get("negRisk") or ev.get("enableNegRisk") else 0,
            len(markets),
        ),
    )
    return event_id


def insert_bucket(conn: sqlite3.Connection, event_id: str,
                  mkt: dict) -> str | None:
    """Parse and insert bucket from market data. Returns YES token_id."""
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

    question = mkt.get("question", "")
    strike = parse_strike(question)
    direction = parse_direction(question)

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

    fee_enabled = 1 if mkt.get("feesEnabled", False) or mkt.get("feeType") is not None else 0

    conn.execute(
        """INSERT OR REPLACE INTO buckets
           (token_id, token_id_no, event_id, condition_id, question,
            strike_price, direction, won, volume, last_trade_price, fee_enabled)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            yes_token,
            no_token,
            event_id,
            mkt.get("conditionId"),
            question,
            float(strike) if strike else None,
            direction,
            won,
            float(mkt.get("volume", 0) or 0),
            float(mkt.get("lastTradePrice", 0) or 0),
            fee_enabled,
        ),
    )
    return yes_token


def insert_prices(conn: sqlite3.Connection, token_id: str,
                  history: list[dict]) -> int:
    """Insert price history, return count."""
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


def get_fetched_kline_ranges(conn: sqlite3.Connection) -> dict[str, tuple[int, int]]:
    """Get (min_ts, max_ts) per symbol from existing klines."""
    cursor = conn.execute(
        "SELECT symbol, MIN(ts), MAX(ts) FROM binance_klines GROUP BY symbol"
    )
    return {row[0]: (row[1], row[2]) for row in cursor}


# ── Binance download for events ──────────────────────────────────


def download_binance_for_events(conn: sqlite3.Connection):
    """Download Binance klines covering all crypto events in the database."""
    print("\n" + "=" * 60)
    print("DOWNLOADING BINANCE KLINES")
    print("=" * 60)

    # Find date range per asset
    rows = conn.execute("""
        SELECT asset, MIN(resolution_date), MAX(resolution_date)
        FROM events
        WHERE asset IS NOT NULL AND resolution_date IS NOT NULL
        GROUP BY asset
    """).fetchall()

    if not rows:
        print("  No events with dates found — skipping Binance download")
        return

    existing_ranges = get_fetched_kline_ranges(conn)

    for asset, min_date_str, max_date_str in rows:
        symbol = f"{asset}USDT"
        min_date = date.fromisoformat(min_date_str)
        max_date = date.fromisoformat(max_date_str)

        # Only download klines for relevant period — cap at 60 days back
        # (matches Polymarket price history retention window)
        earliest_allowed = date.today() - timedelta(days=60)
        min_date = max(min_date, earliest_allowed)

        # Extend range: 7 days before earliest event, 1 day after latest
        start_date = min_date - timedelta(days=7)
        end_date = min(max_date + timedelta(days=1), date.today() + timedelta(days=1))

        start_ms = int(datetime.combine(start_date, datetime.min.time(),
                                        tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(datetime.combine(end_date, datetime.min.time(),
                                      tzinfo=timezone.utc).timestamp() * 1000)

        # Check if we already have klines for this range
        if symbol in existing_ranges:
            existing_min, existing_max = existing_ranges[symbol]
            existing_min_ms = existing_min * 1000
            existing_max_ms = existing_max * 1000
            if existing_min_ms <= start_ms and existing_max_ms >= end_ms // 1000 * 1000:
                print(f"  {symbol}: already have klines for {start_date} → {end_date}")
                continue

        days = (end_date - start_date).days
        print(f"\n  {symbol}: {start_date} → {end_date} ({days} days)")
        print(f"    Estimated: ~{days * 1440} candles (1m interval)")

        # Download in daily chunks to avoid timeout
        total_klines = 0
        current_start_ms = start_ms

        while current_start_ms < end_ms:
            chunk_end_ms = min(current_start_ms + 24 * 3600 * 1000, end_ms)
            klines = fetch_binance_klines(symbol, current_start_ms, chunk_end_ms)
            if klines:
                n = insert_klines(conn, symbol, klines)
                total_klines += n
            current_start_ms = chunk_end_ms

            # Progress every 7 days of data
            if total_klines > 0 and total_klines % (7 * 1440) < 1440:
                conn.commit()
                elapsed_days = (current_start_ms - start_ms) / (24 * 3600 * 1000)
                print(f"    Progress: {elapsed_days:.0f}/{days} days, "
                      f"{total_klines:,} klines")

        conn.commit()
        print(f"    Done: {total_klines:,} klines for {symbol}")


# ── Statistics ────────────────────────────────────────────────────


def print_stats(conn: sqlite3.Connection, t_start: float):
    """Print database statistics."""
    print(f"\n{'─' * 60}")
    print("DATABASE STATISTICS")
    print(f"{'─' * 60}")

    n_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    n_buckets = conn.execute("SELECT COUNT(*) FROM buckets").fetchone()[0]
    n_prices = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    n_tokens_data = conn.execute(
        "SELECT COUNT(DISTINCT token_id) FROM prices"
    ).fetchone()[0]
    n_klines = conn.execute("SELECT COUNT(*) FROM binance_klines").fetchone()[0]

    print(f"  Events:               {n_events:,}")
    print(f"  Buckets (tokens):     {n_buckets:,}")
    print(f"  Tokens with prices:   {n_tokens_data:,}")
    print(f"  Total price points:   {n_prices:,}")
    print(f"  Binance klines:       {n_klines:,}")

    # Date range
    row = conn.execute("SELECT MIN(ts), MAX(ts) FROM prices").fetchone()
    if row and row[0]:
        dt_min = datetime.utcfromtimestamp(row[0])
        dt_max = datetime.utcfromtimestamp(row[1])
        days = (dt_max - dt_min).days
        print(f"  Price data range:     {dt_min:%Y-%m-%d} → {dt_max:%Y-%m-%d} ({days} days)")

    # Per-asset stats
    rows = conn.execute("""
        SELECT e.asset, e.market_type,
               COUNT(DISTINCT e.event_id) as events,
               COUNT(DISTINCT b.token_id) as buckets,
               COUNT(p.ts) as prices,
               SUM(b.won) as wins
        FROM events e
        JOIN buckets b ON b.event_id = e.event_id
        LEFT JOIN prices p ON p.token_id = b.token_id
        WHERE e.asset IS NOT NULL
        GROUP BY e.asset, e.market_type
        ORDER BY e.asset, e.market_type
    """).fetchall()

    if rows:
        print(f"\n  {'Asset':<6} {'Type':<14} {'Events':>7} {'Buckets':>8} "
              f"{'Prices':>8} {'Won':>5}")
        for asset, mtype, events, buckets, prices, wins in rows:
            print(f"  {asset or '?':<6} {mtype or '?':<14} {events:>7} {buckets:>8} "
                  f"{prices:>8} {wins or 0:>5}")

    # Binance klines per symbol
    kline_rows = conn.execute("""
        SELECT symbol, COUNT(*) as cnt, MIN(ts), MAX(ts)
        FROM binance_klines
        GROUP BY symbol
    """).fetchall()

    if kline_rows:
        print(f"\n  Binance klines:")
        for symbol, cnt, min_ts, max_ts in kline_rows:
            dt_min = datetime.utcfromtimestamp(min_ts)
            dt_max = datetime.utcfromtimestamp(max_ts)
            print(f"    {symbol}: {cnt:,} candles "
                  f"({dt_min:%Y-%m-%d} → {dt_max:%Y-%m-%d})")

    elapsed = time.time() - t_start
    db_size = DB_PATH.stat().st_size / (1024 * 1024) if DB_PATH.exists() else 0
    print(f"\n  Database size:        {db_size:.1f} MB")
    print(f"  Elapsed time:         {elapsed / 60:.1f} min")


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Download Polymarket crypto price history + Binance klines"
    )
    parser.add_argument("--refresh-events", action="store_true",
                        help="Re-fetch events from Gamma API")
    parser.add_argument("--fidelity", type=int, default=10,
                        help="Price granularity in minutes (default: 10)")
    parser.add_argument("--max-events", type=int, default=0,
                        help="Limit number of events to process (0=all)")
    parser.add_argument("--skip-binance", action="store_true",
                        help="Skip Binance kline download")
    parser.add_argument("--skip-prices", action="store_true",
                        help="Skip Polymarket price history download")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ── 1. Load events ──
    if args.refresh_events or not EVENTS_CACHE.exists():
        raw_events = fetch_crypto_events_fresh()
    else:
        raw_events = load_events()
    print(f"\nLoaded {len(raw_events)} crypto price events")

    # ── 2. Create database ──
    conn = create_database(DB_PATH)
    already_fetched = get_fetched_tokens(conn)
    print(f"Database: {DB_PATH}")
    print(f"Already fetched: {len(already_fetched)} tokens")

    # ── 3. Parse and insert events + buckets ──
    sorted_events = sorted(
        raw_events,
        key=lambda ev: ev.get("closedTime", "") or "",
        reverse=True,
    )

    tokens_to_fetch: list[tuple[str, str, str]] = []
    total_events = 0
    total_buckets = 0
    skipped_no_asset = 0

    for ev in sorted_events:
        event_id = insert_event(conn, ev)
        if not event_id:
            skipped_no_asset += 1
            continue
        total_events += 1

        for mkt in ev.get("markets", []):
            yes_token = insert_bucket(conn, event_id, mkt)
            if yes_token:
                total_buckets += 1
                if yes_token not in already_fetched:
                    tokens_to_fetch.append(
                        (yes_token, ev.get("title", ""), ev.get("closedTime", ""))
                    )

    conn.commit()

    if args.max_events > 0:
        tokens_to_fetch = tokens_to_fetch[:args.max_events * 20]

    print(f"\nParsed: {total_events} events, {total_buckets} buckets")
    print(f"Skipped (no asset match): {skipped_no_asset}")
    print(f"Tokens to fetch: {len(tokens_to_fetch)} "
          f"(skipping {len(already_fetched)} already fetched)")

    # ── 4. Fetch Polymarket price history ──
    if not args.skip_prices and tokens_to_fetch:
        print(f"\nFetching price history (fidelity={args.fidelity}min)...")
        print(f"Estimated time: ~{len(tokens_to_fetch) * REQUEST_DELAY / 60:.0f} minutes\n")

        fetched = 0
        empty = 0
        errors = 0
        total_points = 0
        consecutive_empty = 0
        # Don't stop early — recent events are mixed with old ones in the list
        CONSECUTIVE_EMPTY_STOP = max(500, len(tokens_to_fetch))

        for i, (token_id, title, closed_time) in enumerate(tokens_to_fetch):
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

            if consecutive_empty >= CONSECUTIVE_EMPTY_STOP:
                print(f"\n  Stopping: {CONSECUTIVE_EMPTY_STOP} consecutive empty responses")
                break

            if (i + 1) % 50 == 0:
                conn.commit()
                pct = (i + 1) / len(tokens_to_fetch) * 100
                print(f"  [{pct:5.1f}%] {i + 1}/{len(tokens_to_fetch)} | "
                      f"{fetched} with data, {empty} empty | "
                      f"{total_points:,} price points")

            time.sleep(REQUEST_DELAY)

        conn.commit()
        print(f"\nPrice download: {fetched} with data, {empty} empty, "
              f"{errors} errors, {total_points:,} points")
    elif not tokens_to_fetch:
        print("\nNo new tokens to fetch!")

    # ── 5. Download Binance klines ──
    if not args.skip_binance:
        download_binance_for_events(conn)

    # ── 6. Stats ──
    print_stats(conn, t_start)
    conn.close()


if __name__ == "__main__":
    main()
