"""Download Crypto Price Market Data from PolymarketData.co.

Downloads 10-minute price history for BTC/ETH crypto price prediction
markets + Binance 1m klines. Uses PMD Ultra tier for unlimited history.

Tags: "bitcoin" (BTC daily above), "ethereum" (ETH daily above)
Monthly "hit" markets found via keyword search.

Expected: ~60K BTC + ~58K ETH markets, ~2-5 GB at 10-min resolution.

Usage:
    PYTHONPATH=. python3 research/download_crypto_pmd.py
    PYTHONPATH=. python3 research/download_crypto_pmd.py --asset btc --resolution 10m
    PYTHONPATH=. python3 research/download_crypto_pmd.py --asset all --resume

Output:
    research/data/crypto_price_pmd.sqlite
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()
    SSL_CTX.check_hostname = False
    SSL_CTX.verify_mode = ssl.CERT_NONE

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "crypto_price_pmd.sqlite"
CACHE_DIR = DATA_DIR / "crypto_pmd_cache"
PROGRESS_PATH = DATA_DIR / "crypto_pmd_progress.txt"

API_BASE = "https://api.polymarketdata.co/v1"
BINANCE_BASE = "https://api.binance.com"

# Asset → PMD tag mapping
ASSETS = {
    "btc": {"tag": "bitcoin", "name": "Bitcoin", "symbol": "BTCUSDT"},
    "eth": {"tag": "ethereum", "name": "Ethereum", "symbol": "ETHUSDT"},
}

# Strike price patterns
_STRIKE_PATTERNS = [
    re.compile(r"\$?([0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?)"),  # 95,000
    re.compile(r"\$([0-9]+(?:\.[0-9]+)?)\s*k", re.IGNORECASE),  # $95k
    re.compile(r"(\d{4,})"),  # 95000 (4+ digit number)
]


def log(msg: str) -> None:
    print(msg, flush=True)


# ── PMD API Client ───────────────────────────────────────────────


class PMDClient:
    """PolymarketData API client."""

    def __init__(self, api_key: str, min_interval: float = 0.05):
        self.api_key = api_key
        self.min_interval = min_interval
        self.last_request = 0.0
        self.request_count = 0

    def get(self, path: str, params: dict | None = None) -> dict | None:
        url = f"{API_BASE}{path}"
        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
            url = f"{url}?{query}"

        elapsed = time.time() - self.last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        for attempt in range(3):
            req = urllib.request.Request(url, headers={
                "X-API-Key": self.api_key,
                "User-Agent": "ArboCrypto/1.0",
                "Accept": "application/json",
            })
            try:
                with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
                    self.request_count += 1
                    self.last_request = time.time()
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(2 ** (attempt + 1))
                elif e.code in (403, 404, 422):
                    return None
                else:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        return None
            except Exception:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    return None
        return None


# ── SQLite Database ──────────────────────────────────────────────


def create_db(db_path: Path) -> sqlite3.Connection:
    """Create crypto price database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS markets (
            market_id    TEXT PRIMARY KEY,
            asset        TEXT NOT NULL,
            question     TEXT,
            status       TEXT,
            start_date   TEXT,
            end_date     TEXT,
            tokens_json  TEXT,
            strike_price REAL,
            direction    TEXT,
            market_type  TEXT
        );

        CREATE TABLE IF NOT EXISTS prices (
            token_id    TEXT NOT NULL,
            ts          INTEGER NOT NULL,
            price       REAL NOT NULL,
            PRIMARY KEY (token_id, ts)
        );

        CREATE TABLE IF NOT EXISTS binance_klines (
            symbol      TEXT NOT NULL,
            ts          INTEGER NOT NULL,
            open        REAL NOT NULL,
            high        REAL NOT NULL,
            low         REAL NOT NULL,
            close       REAL NOT NULL,
            volume      REAL NOT NULL,
            PRIMARY KEY (symbol, ts)
        );

        CREATE INDEX IF NOT EXISTS idx_prices_token ON prices(token_id);
        CREATE INDEX IF NOT EXISTS idx_markets_asset ON markets(asset);
        CREATE INDEX IF NOT EXISTS idx_markets_type ON markets(market_type);
        CREATE INDEX IF NOT EXISTS idx_klines_symbol ON binance_klines(symbol);
    """)
    conn.commit()
    return conn


# ── Progress Tracking ────────────────────────────────────────────

def load_done() -> set[str]:
    if PROGRESS_PATH.exists():
        return {line.strip() for line in PROGRESS_PATH.read_text().splitlines() if line.strip()}
    return set()


def mark_done(market_id: str) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "a") as f:
        f.write(market_id + "\n")


# ── Parsing ──────────────────────────────────────────────────────


def parse_strike(question: str) -> float | None:
    """Extract strike price from market question."""
    for pattern in _STRIKE_PATTERNS:
        match = pattern.search(question)
        if match:
            value_str = match.group(1).replace(",", "")
            try:
                val = float(value_str)
                if val >= 100:  # Must be a realistic price
                    return val
            except ValueError:
                continue
    return None


def classify_question(question: str) -> tuple[str, str]:
    """Classify question as (direction, market_type)."""
    q = question.lower()
    if "what price" in q and "hit" in q:
        direction = "above"  # hit markets are upward by default
        return direction, "monthly_hit"
    if "up or down" in q:
        return "above", "5min"
    direction = "below" if "below" in q else "above"
    return direction, "daily_above"


# ── Market Discovery ─────────────────────────────────────────────


def discover_markets(client: PMDClient, asset_key: str) -> list[dict]:
    """Discover crypto price markets for an asset via PMD."""
    asset = ASSETS[asset_key]
    tag = asset["tag"]

    cache_path = CACHE_DIR / f"{asset_key}.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        age_h = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_h < 12:
            markets = json.loads(cache_path.read_text())
            log(f"  Loaded {len(markets)} markets from cache ({age_h:.1f}h old)")
            return markets

    log(f"  Discovering markets via PMD tag={tag}...")
    all_markets: list[dict] = []
    cursor = None
    page = 0

    while True:
        resp = client.get("/markets", {
            "tags": tag,
            "limit": 1000,
            "cursor": cursor,
        })
        if not resp or not resp.get("data"):
            break

        for m in resp["data"]:
            q = (m.get("question") or "").lower()
            # Filter for price prediction markets only
            if "above" in q or "below" in q or "hit" in q or "up or down" in q:
                if any(kw in q for kw in ["$", "price", "above", "below"]):
                    all_markets.append(m)

        page += 1
        if page % 10 == 0:
            log(f"    Page {page}: {len(all_markets)} price markets so far...")

        cursor = resp.get("metadata", {}).get("next_cursor")
        if not cursor:
            break

    cache_path.write_text(json.dumps(all_markets, default=str))
    log(f"  Found {len(all_markets)} price markets, cached to {cache_path.name}")
    return all_markets


# ── Price Download ───────────────────────────────────────────────


def download_market_prices(
    client: PMDClient, market: dict, conn: sqlite3.Connection,
    resolution: str, max_history_days: int,
) -> int:
    """Download prices for one market. Returns count of price points."""
    market_id = str(market.get("id", ""))
    if not market_id:
        return 0

    # Time range
    m_start = market.get("start_date") or market.get("created_at")
    if m_start:
        try:
            start_ts = int(datetime.fromisoformat(
                str(m_start).replace("Z", "+00:00")).timestamp())
        except (ValueError, TypeError):
            start_ts = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp())
    else:
        start_ts = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp())

    min_allowed = int((datetime.now(timezone.utc) - timedelta(days=max_history_days)).timestamp())
    start_ts = max(start_ts, min_allowed)
    end_ts = int(datetime.now(timezone.utc).timestamp())

    # Paginated price download
    total = 0
    cursor = None
    tokens_map: dict[str, str] = {}

    while True:
        resp = client.get(f"/markets/{market_id}/prices", {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "resolution": resolution,
            "limit": 200,
            "cursor": cursor,
        })
        if not resp:
            break

        if not tokens_map:
            tokens_map = resp.get("tokens", {})

        data = resp.get("data", {})
        tokens_from_market = market.get("tokens", [])

        for label, points in data.items():
            if not points:
                continue

            token_id = tokens_map.get(label, "")
            if not token_id:
                for t in tokens_from_market:
                    if t.get("label") == label:
                        token_id = t.get("id", "")
                        break
            if not token_id:
                continue

            rows = []
            for pt in points:
                ts_val = pt.get("t")
                p = pt.get("p")
                if ts_val is None or p is None:
                    continue
                if isinstance(ts_val, str):
                    try:
                        ts_val = int(datetime.fromisoformat(
                            ts_val.replace("Z", "+00:00")).timestamp())
                    except (ValueError, TypeError):
                        continue
                rows.append((str(token_id), int(ts_val), float(p)))

            if rows:
                conn.executemany(
                    "INSERT OR IGNORE INTO prices (token_id, ts, price) VALUES (?, ?, ?)",
                    rows,
                )
                total += len(rows)

        cursor = resp.get("metadata", {}).get("next_cursor")
        if not cursor:
            break

    conn.commit()
    return total


# ── Binance Klines ───────────────────────────────────────────────


def download_binance_klines(conn: sqlite3.Connection, symbols: list[str], days: int = 90):
    """Download Binance 1m klines for volatility computation."""
    log(f"\n{'='*60}")
    log("DOWNLOADING BINANCE KLINES")
    log(f"{'='*60}")

    for symbol in symbols:
        # Check existing range
        row = conn.execute(
            "SELECT MIN(ts), MAX(ts) FROM binance_klines WHERE symbol=?",
            (symbol,)
        ).fetchone()

        end_date = date.today() + timedelta(days=1)
        start_date = end_date - timedelta(days=days)
        start_ms = int(datetime.combine(start_date, datetime.min.time(),
                                        tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(datetime.combine(end_date, datetime.min.time(),
                                      tzinfo=timezone.utc).timestamp() * 1000)

        if row and row[0] and row[1]:
            existing_min = row[0] * 1000
            existing_max = row[1] * 1000
            if existing_min <= start_ms + 86400000 and existing_max >= end_ms - 86400000:
                log(f"  {symbol}: already have klines for last {days} days")
                continue

        total_days = (end_date - start_date).days
        log(f"\n  {symbol}: {start_date} → {end_date} ({total_days} days)")

        total_klines = 0
        current_start = start_ms

        while current_start < end_ms:
            chunk_end = min(current_start + 24 * 3600 * 1000, end_ms)
            url = (
                f"{BINANCE_BASE}/api/v3/klines"
                f"?symbol={symbol}&interval=1m"
                f"&startTime={current_start}&endTime={chunk_end}&limit=1000"
            )
            req = urllib.request.Request(url, headers={
                "User-Agent": "ArboCrypto/1.0",
            })
            try:
                with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
                    klines = json.loads(resp.read())
                if klines:
                    rows = [(
                        symbol, int(k[0]) // 1000,
                        float(k[1]), float(k[2]), float(k[3]),
                        float(k[4]), float(k[5]),
                    ) for k in klines]
                    conn.executemany(
                        "INSERT OR IGNORE INTO binance_klines "
                        "(symbol,ts,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?)",
                        rows,
                    )
                    total_klines += len(rows)
            except Exception as e:
                log(f"    Binance error: {e}")
                time.sleep(1)

            current_start = chunk_end
            if total_klines > 0 and total_klines % (7 * 1440) < 1440:
                conn.commit()
                elapsed_days = (current_start - start_ms) / (24 * 3600 * 1000)
                log(f"    Progress: {elapsed_days:.0f}/{total_days} days, "
                    f"{total_klines:,} klines")
            time.sleep(0.02)  # Binance rate limit

        conn.commit()
        log(f"    Done: {total_klines:,} klines for {symbol}")


# ── Main ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Download crypto price data from PolymarketData.co")
    parser.add_argument("--asset", default="all", help="Asset key (btc/eth) or 'all'")
    parser.add_argument("--resolution", default="10m", choices=["1m", "10m", "1h"])
    parser.add_argument("--max-history-days", type=int, default=9999)
    parser.add_argument("--max-markets", type=int, default=0, help="Limit markets per asset (0=all)")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--skip-binance", action="store_true")
    parser.add_argument("--binance-days", type=int, default=90)
    parser.add_argument("--worker-id", type=int, default=0, help="Worker ID for parallel download (0-based)")
    parser.add_argument("--num-workers", type=int, default=1, help="Total number of parallel workers")
    parser.add_argument("--db", default=None, help="Custom DB path (for parallel workers)")
    args = parser.parse_args()

    # Load API key
    api_key = os.environ.get("POLYMARKETDATA_API_KEY", "")
    if not api_key:
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("POLYMARKETDATA_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not api_key:
        log("ERROR: POLYMARKETDATA_API_KEY not found in .env")
        sys.exit(1)

    client = PMDClient(api_key, min_interval=0.05)
    db_path = Path(args.db) if args.db else DB_PATH
    conn = create_db(db_path)
    done = load_done() if args.resume else set()
    t_start = time.time()

    assets_to_process = list(ASSETS.keys()) if args.asset == "all" else [args.asset]

    log(f"\n{'='*60}")
    log(f"  Crypto Price Data Download (PolymarketData.co)")
    log(f"  Assets: {assets_to_process}, Resolution: {args.resolution}")
    log(f"  {datetime.now(timezone.utc).isoformat()}")
    log(f"{'='*60}\n")

    total_markets = 0
    total_prices = 0

    for asset_key in assets_to_process:
        if asset_key not in ASSETS:
            log(f"  Unknown asset: {asset_key}")
            continue

        asset = ASSETS[asset_key]
        log(f"\n{'─'*40}")
        log(f"  {asset['name']} ({asset_key.upper()})")
        log(f"{'─'*40}")

        # Discover markets
        markets = discover_markets(client, asset_key)
        remaining = [m for m in markets if str(m.get("id", "")) not in done]

        if args.max_markets > 0:
            remaining = remaining[:args.max_markets]

        # Parallel worker sharding: each worker takes every Nth market
        if args.num_workers > 1:
            remaining = [m for i, m in enumerate(remaining) if i % args.num_workers == args.worker_id]

        log(f"  Markets: {len(markets)} total, {len(remaining)} for this worker"
            + (f" (worker {args.worker_id}/{args.num_workers})" if args.num_workers > 1 else ""))

        asset_prices = 0
        for idx, market in enumerate(remaining):
            market_id = str(market.get("id", ""))
            question = market.get("question", "")

            # Parse metadata
            strike = parse_strike(question)
            direction, market_type = classify_question(question)

            # Register market in DB
            conn.execute(
                """INSERT OR REPLACE INTO markets
                   (market_id, asset, question, status, start_date, end_date,
                    tokens_json, strike_price, direction, market_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (market_id, asset_key.upper(), question,
                 market.get("status"), market.get("start_date"),
                 market.get("end_date"), json.dumps(market.get("tokens", [])),
                 strike, direction, market_type),
            )

            # Download prices
            n = download_market_prices(client, market, conn, args.resolution,
                                       args.max_history_days)
            asset_prices += n
            total_prices += n
            total_markets += 1

            mark_done(market_id)

            if (idx + 1) % 100 == 0:
                elapsed = time.time() - t_start
                rate = client.request_count / max(elapsed, 1) * 60
                log(f"  [{idx+1}/{len(remaining)}] {asset_prices:,} prices  "
                    f"req={client.request_count}  {elapsed/60:.0f}min  {rate:.0f}req/min")

        log(f"  {asset['name']}: {len(remaining)} markets, {asset_prices:,} prices")

    # Binance klines
    if not args.skip_binance:
        symbols = [ASSETS[a]["symbol"] for a in assets_to_process if a in ASSETS]
        download_binance_klines(conn, symbols, args.binance_days)

    # Summary
    elapsed = time.time() - t_start
    total_db_prices = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    total_db_markets = conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
    total_db_tokens = conn.execute("SELECT COUNT(DISTINCT token_id) FROM prices").fetchone()[0]
    total_db_klines = conn.execute("SELECT COUNT(*) FROM binance_klines").fetchone()[0]
    db_size = os.path.getsize(str(DB_PATH)) / (1024**2)

    log(f"\n{'='*60}")
    log(f"  DOWNLOAD COMPLETE")
    log(f"{'='*60}")
    log(f"  Duration: {elapsed/60:.1f} min")
    log(f"  API requests: {client.request_count:,}")
    log(f"  Markets processed: {total_markets:,}")
    log(f"  New prices: {total_prices:,}")
    log(f"")
    log(f"  Database: {DB_PATH}")
    log(f"  Size: {db_size:.1f} MB")
    log(f"  Total markets: {total_db_markets:,}")
    log(f"  Total tokens: {total_db_tokens:,}")
    log(f"  Total prices: {total_db_prices:,}")
    log(f"  Binance klines: {total_db_klines:,}")

    # Per-asset
    for row in conn.execute("""
        SELECT asset, market_type, COUNT(*) as n, SUM(strike_price IS NOT NULL) as with_strike
        FROM markets GROUP BY asset, market_type ORDER BY asset, market_type
    """).fetchall():
        log(f"  {row[0]} {row[1]}: {row[2]} markets ({row[3]} with strike)")

    log(f"{'='*60}")
    conn.close()


if __name__ == "__main__":
    main()
