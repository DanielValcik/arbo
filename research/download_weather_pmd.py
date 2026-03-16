"""Download Weather Temperature Market Data from PolymarketData.co.

Downloads 1-minute price history for all temperature markets across
20 cities used in Strategy C. Stores in a separate SQLite database
for weather backtesting.

Cities: NYC, Chicago, London, Seoul, Buenos Aires, Atlanta, Toronto,
Ankara, São Paulo, Miami, Paris, Dallas, Seattle, Wellington, Tokyo,
Munich, Los Angeles, Washington DC, Tel Aviv, Lucknow.

Expected: ~18,700 markets, ~5 GB at 1-min resolution.

Usage:
    PYTHONPATH=. python3 research/download_weather_pmd.py
    PYTHONPATH=. python3 research/download_weather_pmd.py --city london --resolution 1m
    PYTHONPATH=. python3 research/download_weather_pmd.py --city all --resolution 10m --resume

Output:
    research/data/weather_pmd.sqlite — Dedicated weather price database
    research/data/weather_pmd_progress.txt — Append-only progress log
    research/data/weather_pmd_cache/ — Market discovery cache per city

Requires: POLYMARKETDATA_API_KEY in .env (Ultra tier for 1-min + unlimited history)
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()
    SSL_CTX.check_hostname = False
    SSL_CTX.verify_mode = ssl.CERT_NONE

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "weather_pmd.sqlite"
CACHE_DIR = DATA_DIR / "weather_pmd_cache"
PROGRESS_PATH = DATA_DIR / "weather_pmd_progress.txt"

API_BASE = "https://api.polymarketdata.co/v1"

# All 20 cities with their PolymarketData tags and display names
CITIES = {
    "nyc": {"tag": "nyc", "name": "New York City", "id": "nyc"},
    "chicago": {"tag": "chicago", "name": "Chicago", "id": "chicago"},
    "london": {"tag": "london", "name": "London", "id": "london"},
    "seoul": {"tag": "seoul", "name": "Seoul", "id": "seoul"},
    "buenos_aires": {"tag": "buenos-aires", "name": "Buenos Aires", "id": "buenos_aires"},
    "atlanta": {"tag": "atlanta", "name": "Atlanta", "id": "atlanta"},
    "toronto": {"tag": "toronto", "name": "Toronto", "id": "toronto"},
    "ankara": {"tag": "ankara", "name": "Ankara", "id": "ankara"},
    "sao_paulo": {"tag": "sao-paulo", "name": "São Paulo", "id": "sao_paulo"},
    "miami": {"tag": "miami", "name": "Miami", "id": "miami"},
    "paris": {"tag": "paris", "name": "Paris", "id": "paris"},
    "dallas": {"tag": "dallas", "name": "Dallas", "id": "dallas"},
    "seattle": {"tag": "seattle", "name": "Seattle", "id": "seattle"},
    "wellington": {"tag": "wellington", "name": "Wellington", "id": "wellington"},
    "tokyo": {"tag": "tokyo", "name": "Tokyo", "id": "tokyo"},
    "munich": {"tag": "munich", "name": "Munich", "id": "munich"},
    "los_angeles": {"tag": "los-angeles", "name": "Los Angeles", "id": "los_angeles"},
    "dc": {"tag": "washington-dc", "name": "Washington DC", "id": "dc"},
    "tel_aviv": {"tag": "tel-aviv", "name": "Tel Aviv", "id": "tel_aviv"},
    "lucknow": {"tag": "lucknow", "name": "Lucknow", "id": "lucknow"},
}


def log(msg: str) -> None:
    print(msg, flush=True)


# ── API Client ───────────────────────────────────────────────────────

class PMDClient:
    """Minimal PolymarketData API client."""

    def __init__(self, api_key: str, min_interval: float = 0.05):
        self.api_key = api_key
        self.min_interval = min_interval
        self.last_request = 0.0
        self.request_count = 0

    def _get(self, path: str, params: dict | None = None) -> dict | None:
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
                "User-Agent": "ArboWeather/1.0",
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
                elif e.code in (403, 404):
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


# ── SQLite Database ──────────────────────────────────────────────────

import sqlite3


def create_db(db_path: Path) -> sqlite3.Connection:
    """Create weather price database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-32000")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS markets (
            market_id   TEXT PRIMARY KEY,
            city        TEXT NOT NULL,
            question    TEXT,
            status      TEXT,
            start_date  TEXT,
            end_date    TEXT,
            tokens_json TEXT
        );

        CREATE TABLE IF NOT EXISTS prices (
            token_id    TEXT NOT NULL,
            ts          INTEGER NOT NULL,
            price       REAL NOT NULL,
            PRIMARY KEY (token_id, ts)
        );

        CREATE INDEX IF NOT EXISTS idx_prices_token ON prices(token_id);
        CREATE INDEX IF NOT EXISTS idx_markets_city ON markets(city);
    """)
    conn.commit()
    return conn


# ── Progress Tracking (append-only, safe) ────────────────────────────

def load_done() -> set[str]:
    if PROGRESS_PATH.exists():
        return {line.strip() for line in PROGRESS_PATH.read_text().splitlines() if line.strip()}
    return set()


def mark_done(market_id: str) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "a") as f:
        f.write(market_id + "\n")


# ── Discovery ────────────────────────────────────────────────────────

def discover_city_markets(client: PMDClient, city_key: str) -> list[dict]:
    """Discover temperature markets for a city."""
    city = CITIES[city_key]
    tag = city["tag"]

    cache_path = CACHE_DIR / f"{city_key}.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        age_h = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_h < 24:
            markets = json.loads(cache_path.read_text())
            return markets

    all_markets: list[dict] = []
    cursor = None

    while True:
        resp = client._get("/markets", {
            "tags": tag,
            "limit": 1000,
            "cursor": cursor,
        })
        if not resp or not resp.get("data"):
            break

        for m in resp["data"]:
            q = (m.get("question") or "").lower()
            if "temperature" in q or "°" in q:
                all_markets.append(m)

        cursor = resp.get("metadata", {}).get("next_cursor")
        if not cursor:
            break

    cache_path.write_text(json.dumps(all_markets, default=str))
    return all_markets


# ── Price Download ───────────────────────────────────────────────────

def download_market(client: PMDClient, market: dict, conn: sqlite3.Connection,
                    resolution: str, max_history_days: int) -> int:
    """Download prices for one market. Returns count of price points."""
    market_id = str(market.get("id", ""))
    tokens = market.get("tokens", [])
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

    # Paginated download
    total = 0
    cursor = None
    tokens_map: dict[str, str] = {}

    while True:
        resp = client._get(f"/markets/{market_id}/prices", {
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
        for label, points in data.items():
            if not points:
                continue

            token_id = tokens_map.get(label, "")
            if not token_id:
                for t in tokens:
                    if t.get("label") == label:
                        token_id = t.get("id", "")
                        break
            if not token_id:
                continue

            rows = []
            for pt in points:
                ts = pt.get("t")
                p = pt.get("p")
                if ts is None or p is None:
                    continue
                if isinstance(ts, str):
                    try:
                        ts = int(datetime.fromisoformat(
                            ts.replace("Z", "+00:00")).timestamp())
                    except (ValueError, TypeError):
                        continue
                rows.append((str(token_id), int(ts), float(p)))

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


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download weather price data from PolymarketData.co")
    parser.add_argument("--city", default="all", help="City key or 'all' (default: all)")
    parser.add_argument("--resolution", default="1m", choices=["1m", "10m", "1h"])
    parser.add_argument("--max-history-days", type=int, default=9999)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--db", default=None)
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

    cities_to_process = list(CITIES.keys()) if args.city == "all" else [args.city]

    log(f"\n{'='*60}")
    log(f"  Weather Temperature Data Download")
    log(f"  Cities: {len(cities_to_process)}, Resolution: {args.resolution}")
    log(f"  {datetime.now(timezone.utc).isoformat()}")
    log(f"{'='*60}\n")

    total_markets = 0
    total_prices = 0

    for city_key in cities_to_process:
        if city_key not in CITIES:
            log(f"  Unknown city: {city_key}")
            continue

        city = CITIES[city_key]
        log(f"\n{'─'*40}")
        log(f"  {city['name']} ({city_key})")
        log(f"{'─'*40}")

        # Discover markets
        markets = discover_city_markets(client, city_key)
        remaining = [m for m in markets if str(m.get("id", "")) not in done]
        log(f"  Markets: {len(markets)} total, {len(remaining)} remaining")

        city_prices = 0
        for idx, market in enumerate(remaining):
            market_id = str(market.get("id", ""))
            q = (market.get("question") or "")[:55]

            # Register market in DB
            conn.execute(
                """INSERT OR REPLACE INTO markets
                   (market_id, city, question, status, start_date, end_date, tokens_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (market_id, city_key, market.get("question"),
                 market.get("status"), market.get("start_date"),
                 market.get("end_date"), json.dumps(market.get("tokens", []))),
            )

            # Download prices
            n = download_market(client, market, conn, args.resolution, args.max_history_days)
            city_prices += n
            total_prices += n
            total_markets += 1

            mark_done(market_id)

            if (idx + 1) % 50 == 0:
                elapsed = time.time() - t_start
                log(f"  [{idx+1}/{len(remaining)}] {city_prices:,} prices  "
                    f"req={client.request_count}  {elapsed/60:.0f}min")

        log(f"  {city['name']}: {len(remaining)} markets, {city_prices:,} prices")

    # Summary
    elapsed = time.time() - t_start

    # DB stats
    total_db_prices = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    total_db_markets = conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
    total_db_tokens = conn.execute("SELECT COUNT(DISTINCT token_id) FROM prices").fetchone()[0]
    db_size = os.path.getsize(str(db_path)) / (1024**2)

    # Per-city breakdown
    city_stats = conn.execute("""
        SELECT m.city, COUNT(DISTINCT m.market_id) as n_markets,
               COUNT(p.ts) as n_prices
        FROM markets m
        LEFT JOIN prices p ON p.token_id IN (
            SELECT json_extract(value, '$.id')
            FROM markets m2, json_each(m2.tokens_json)
            WHERE m2.market_id = m.market_id
        )
        GROUP BY m.city
        ORDER BY n_prices DESC
    """).fetchall()

    log(f"\n{'='*60}")
    log(f"  DOWNLOAD COMPLETE")
    log(f"{'='*60}")
    log(f"  Duration: {elapsed/60:.1f} min")
    log(f"  API requests: {client.request_count:,}")
    log(f"  Markets processed: {total_markets:,}")
    log(f"  Prices downloaded: {total_prices:,}")
    log(f"")
    log(f"  Database: {db_path}")
    log(f"  Size: {db_size:.1f} MB")
    log(f"  Total markets: {total_db_markets:,}")
    log(f"  Total tokens: {total_db_tokens:,}")
    log(f"  Total prices: {total_db_prices:,}")
    log(f"{'='*60}")

    conn.close()


if __name__ == "__main__":
    main()
