"""Download Historical Sports Price Data from PolymarketData.co API.

Professional-grade minutely price data for Polymarket sports markets.
This is our PRIMARY data source for realistic backtesting.

API tiers:
    Free:   10-min resolution, 1 month history, 10 RPM
    Trader: 1-min resolution, 1 month history, 100 RPM
    Pro:    1-min resolution, 3 months history, 500 RPM
    Ultra:  1-min resolution, UNLIMITED history, 2000 RPM

Strategy: Start with Free tier to validate data quality, then upgrade
to Ultra ($360/mo) for 1 month to download complete historical dataset.

Usage:
    # Set API key in .env: POLYMARKETDATA_API_KEY=pk_live_xxx
    PYTHONPATH=. python3 research_d/download_polymarketdata.py --discover
    PYTHONPATH=. python3 research_d/download_polymarketdata.py --sport nba --resolution 1m
    PYTHONPATH=. python3 research_d/download_polymarketdata.py --sport all --resolution 10m

Output:
    research_d/data/sports_backtest.sqlite — prices table (minutely)
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_d.sports_db import SportsDB

DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = DATA_DIR / "pmd_cache"
PROGRESS_PATH = DATA_DIR / "pmd_progress.json"

API_BASE = "https://api.polymarketdata.co/v1"

# Tags for sports discovery
SPORT_TAGS = {
    "nba": ["nba"],
    "epl": ["epl", "premier-league"],
    "nfl": ["nfl"],
    "soccer": ["soccer", "football"],
    "ufc": ["ufc", "mma"],
    "ncaab": ["ncaab", "march-madness"],
    "mlb": ["mlb"],
    "nhl": ["nhl"],
}


def log(msg: str) -> None:
    print(msg, flush=True)


# ── API Client ───────────────────────────────────────────────────────

class PMDataClient:
    """PolymarketData.co API client with rate limiting and retry."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.request_count = 0
        self.last_request_time = 0.0
        self.min_interval = 0.1  # Will be adjusted based on plan

    def _request(self, path: str, params: dict[str, Any] | None = None) -> dict | None:
        """Make authenticated API request with retry."""
        url = f"{API_BASE}{path}"
        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
            url = f"{url}?{query}"

        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        for attempt in range(3):
            req = urllib.request.Request(
                url,
                headers={
                    "X-API-Key": self.api_key,
                    "User-Agent": "ArboResearch/2.0",
                    "Accept": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
                    self.request_count += 1
                    self.last_request_time = time.time()
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = 2 ** (attempt + 1)
                    log(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                elif e.code == 403:
                    body = e.read().decode() if e.fp else ""
                    log(f"  403 Forbidden: {body[:200]}")
                    return None
                elif e.code == 404:
                    return None
                else:
                    log(f"  HTTP {e.code}: {e.reason}")
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        return None
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    log(f"  Request failed: {e}")
                    return None
        return None

    def health(self) -> dict | None:
        """Check API health and validate key."""
        return self._request("/health")

    def usage(self) -> dict | None:
        """Get current usage and plan info."""
        return self._request("/usage")

    def list_tags(self) -> list[str]:
        """Get all available tags."""
        resp = self._request("/tags")
        if resp and "data" in resp:
            return resp["data"]
        return []

    def list_markets(
        self,
        tags: list[str] | None = None,
        search: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict | None:
        """List markets with optional tag/search filtering."""
        params: dict[str, Any] = {"limit": limit}
        if tags:
            params["tags"] = ",".join(tags)
        if search:
            params["search"] = search
        if cursor:
            params["cursor"] = cursor
        return self._request("/markets", params)

    def get_market(self, id_or_slug: str) -> dict | None:
        """Get market details by ID or slug."""
        return self._request(f"/markets/{id_or_slug}")

    def get_prices(
        self,
        market_id: str,
        start_ts: int,
        end_ts: int,
        resolution: str = "1m",
        limit: int = 200,
        cursor: str | None = None,
    ) -> dict | None:
        """Get price history for all tokens in a market.

        Args:
            market_id: Market ID or slug.
            start_ts: Start timestamp (Unix seconds).
            end_ts: End timestamp (Unix seconds, exclusive).
            resolution: "1m", "10m", "1h", "6h", "1d".
            limit: Max points per page (1-200).
            cursor: Pagination cursor.
        """
        params: dict[str, Any] = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "resolution": resolution,
            "limit": limit,
        }
        if cursor:
            params["cursor"] = cursor
        return self._request(f"/markets/{market_id}/prices", params)

    def get_token_prices(
        self,
        token_id: str,
        start_ts: int,
        end_ts: int,
        resolution: str = "1m",
        limit: int = 200,
        cursor: str | None = None,
    ) -> dict | None:
        """Get price history for a single token."""
        params: dict[str, Any] = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "resolution": resolution,
            "limit": limit,
        }
        if cursor:
            params["cursor"] = cursor
        return self._request(f"/tokens/{token_id}/prices", params)

    def get_books(
        self,
        market_id: str,
        start_ts: int,
        end_ts: int,
        resolution: str = "1m",
        limit: int = 200,
        cursor: str | None = None,
    ) -> dict | None:
        """Get orderbook history for a market."""
        params: dict[str, Any] = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "resolution": resolution,
            "limit": limit,
        }
        if cursor:
            params["cursor"] = cursor
        return self._request(f"/markets/{market_id}/books", params)


# ── Discovery ────────────────────────────────────────────────────────

def discover_sports_markets(
    client: PMDataClient,
    sport: str | None = None,
) -> list[dict]:
    """Discover all sports markets via PolymarketData API.

    Args:
        client: API client.
        sport: Optional sport filter ("nba", "epl", etc.).

    Returns:
        List of market dicts with id, slug, question, tokens, etc.
    """
    cache_path = CACHE_DIR / f"markets_{sport or 'all'}.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < 12:
            markets = json.loads(cache_path.read_text())
            log(f"Loaded {len(markets)} markets from cache ({age_hours:.1f}h old)")
            return markets

    tags = []
    if sport and sport in SPORT_TAGS:
        tags = SPORT_TAGS[sport]
    elif sport == "all" or sport is None:
        for tag_list in SPORT_TAGS.values():
            tags.extend(tag_list)
        tags = list(set(tags))

    all_markets: list[dict] = []
    seen_ids: set[str] = set()

    for tag in tags:
        cursor = None
        tag_count = 0

        while True:
            resp = client.list_markets(tags=[tag], limit=100, cursor=cursor)
            if not resp or "data" not in resp:
                break

            for m in resp["data"]:
                mid = str(m.get("id", ""))
                if mid and mid not in seen_ids:
                    seen_ids.add(mid)
                    all_markets.append(m)
                    tag_count += 1

            # Pagination
            meta = resp.get("metadata", {})
            cursor = meta.get("next_cursor")
            if not cursor:
                break

        log(f"  Tag '{tag}': {tag_count} new markets (total: {len(all_markets)})")

    # Cache
    cache_path.write_text(json.dumps(all_markets, default=str))
    log(f"Cached {len(all_markets)} markets")

    return all_markets


# ── Price Download ───────────────────────────────────────────────────

def download_market_prices(
    client: PMDataClient,
    market: dict,
    db: SportsDB,
    resolution: str = "10m",
    start_date: str | None = None,
    end_date: str | None = None,
) -> int:
    """Download complete price history for one market.

    Handles pagination (200 points per page) and stores in SQLite.

    Args:
        client: API client.
        market: Market dict from discovery.
        db: SportsDB instance.
        resolution: Price resolution ("1m", "10m", "1h", "6h", "1d").
        start_date: Start date YYYY-MM-DD (default: market start).
        end_date: End date YYYY-MM-DD (default: now).

    Returns:
        Number of price points stored.
    """
    market_id = str(market.get("id", ""))
    slug = market.get("slug", "")
    question = market.get("question", "")[:60]
    tokens = market.get("tokens", [])

    if not market_id and not slug:
        return 0

    identifier = market_id or slug

    # Determine time range
    if start_date:
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc).timestamp())
    else:
        # Use market start_date or default 18 months back
        m_start = market.get("start_date") or market.get("created_at")
        if m_start:
            try:
                start_ts = int(datetime.fromisoformat(
                    str(m_start).replace("Z", "+00:00")).timestamp())
            except (ValueError, TypeError):
                start_ts = int((datetime.now(timezone.utc) - timedelta(days=540)).timestamp())
        else:
            start_ts = int((datetime.now(timezone.utc) - timedelta(days=540)).timestamp())

    if end_date:
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc).timestamp())
    else:
        end_ts = int(datetime.now(timezone.utc).timestamp())

    # Download prices with pagination
    total_points = 0
    cursor = None

    # Download in 24-hour chunks for efficiency
    chunk_seconds = 86400  # 1 day
    current_start = start_ts

    while current_start < end_ts:
        chunk_end = min(current_start + chunk_seconds, end_ts)

        cursor = None
        while True:
            resp = client.get_prices(
                identifier,
                start_ts=current_start,
                end_ts=chunk_end,
                resolution=resolution,
                limit=200,
                cursor=cursor,
            )

            if not resp:
                break

            # Response has: data = {token_label: [{t, p}, ...]}
            data = resp.get("data", {})
            tokens_map = resp.get("tokens", {})

            for label, price_points in data.items():
                if not price_points:
                    continue

                # Get token_id for this label
                token_id = tokens_map.get(label, "")
                if not token_id:
                    # Try to find from market tokens
                    for t in tokens:
                        if t.get("label") == label or t.get("outcome") == label:
                            token_id = t.get("id", t.get("token_id", ""))
                            break

                if not token_id:
                    continue

                # Convert to (ts, price) pairs
                prices = []
                for pt in price_points:
                    ts = pt.get("t")
                    p = pt.get("p")
                    if ts is not None and p is not None:
                        # ts might be ISO string or Unix
                        if isinstance(ts, str):
                            try:
                                ts = int(datetime.fromisoformat(
                                    ts.replace("Z", "+00:00")).timestamp())
                            except (ValueError, TypeError):
                                continue
                        prices.append((int(ts), float(p)))

                if prices:
                    db.insert_prices_simple(str(token_id), prices)
                    total_points += len(prices)

            # Pagination
            meta = resp.get("metadata", {})
            cursor = meta.get("next_cursor")
            if not cursor:
                break

        current_start = chunk_end

    return total_points


# ── Game Matching + Market Registration ──────────────────────────────

def register_market_in_db(
    market: dict,
    db: SportsDB,
    sport: str,
) -> str | None:
    """Register a PolymarketData market in our SportsDB.

    Creates game + market records. Returns game_id or None.
    """
    market_id = str(market.get("id", ""))
    slug = market.get("slug", "")
    question = market.get("question", "")
    tokens = market.get("tokens", [])
    end_date_raw = market.get("end_date") or market.get("resolution_date", "")

    # Parse game date from end_date
    game_date = ""
    if end_date_raw:
        try:
            dt = datetime.fromisoformat(str(end_date_raw).replace("Z", "+00:00"))
            game_date = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            pass

    # Generate game_id
    game_id = f"{sport}_pmd_{market_id}"

    # Parse teams from question (basic)
    home_team = "HOME"
    away_team = "AWAY"

    from research_d.download_sports_prices import (
        parse_teams_from_question,
        team_to_abbreviation,
    )
    teams = parse_teams_from_question(question)
    if teams:
        away_name, home_name = teams
        away_team = team_to_abbreviation(away_name, sport)
        home_team = team_to_abbreviation(home_name, sport)
        game_id = f"{sport}_{game_date.replace('-', '')}_{away_team}_{home_team}" if game_date else game_id

    # Upsert game
    status = market.get("status", "resolved")
    db.upsert_game(
        game_id=game_id,
        sport=sport,
        league=sport.upper(),
        home_team=home_team,
        away_team=away_team,
        game_date=game_date or "2025-01-01",
        status="final" if status in ("resolved", "closed") else "scheduled",
        extra_json=json.dumps({
            "source": "polymarketdata",
            "pmd_id": market_id,
            "slug": slug,
        }),
    )

    # Upsert market tokens
    for token in tokens:
        token_id = str(token.get("id", token.get("token_id", "")))
        if not token_id:
            continue

        outcome = token.get("label", token.get("outcome", "YES"))
        won = None
        if token.get("winner") is True:
            won = 1
        elif token.get("winner") is False:
            won = 0

        db.upsert_market(
            token_id=token_id,
            game_id=game_id,
            event_id=market_id,
            condition_id=market.get("condition_id", ""),
            question=question,
            outcome=outcome.lower(),
            volume=float(market.get("volume", 0) or 0),
            won=won,
        )

    return game_id


# ── Progress Tracking ────────────────────────────────────────────────

def load_progress() -> set[str]:
    if PROGRESS_PATH.exists():
        return set(json.loads(PROGRESS_PATH.read_text()))
    return set()


def save_progress(done: set[str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(sorted(done)))


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download sports price data from PolymarketData.co API.",
    )
    parser.add_argument(
        "--sport", default="all",
        help="Sport filter: nba, epl, nfl, all (default: all).",
    )
    parser.add_argument(
        "--resolution", default="10m",
        choices=["1m", "10m", "1h", "6h", "1d"],
        help="Price resolution (default: 10m; 1m requires Trader+).",
    )
    parser.add_argument(
        "--start-date", default=None,
        help="Start date YYYY-MM-DD (default: market start).",
    )
    parser.add_argument(
        "--end-date", default=None,
        help="End date YYYY-MM-DD (default: now).",
    )
    parser.add_argument(
        "--discover", action="store_true",
        help="Only discover markets, don't download prices.",
    )
    parser.add_argument(
        "--max-markets", type=int, default=0,
        help="Max markets to process (0=unlimited).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-downloaded markets.",
    )
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
        log("ERROR: POLYMARKETDATA_API_KEY not found in env or .env")
        log("  Sign up at https://www.polymarketdata.co and add key to .env")
        sys.exit(1)

    client = PMDataClient(api_key)
    t_start = time.time()

    log(f"\n{'='*60}")
    log(f"  PolymarketData.co Sports Price Download")
    log(f"  {datetime.now(timezone.utc).isoformat()}")
    log(f"  Sport: {args.sport}  Resolution: {args.resolution}")
    log(f"{'='*60}\n")

    # Check API health + plan
    health = client.health()
    if not health:
        log("ERROR: API health check failed. Check your API key.")
        sys.exit(1)
    log(f"API health: {health.get('status', '?')}")

    usage = client.usage()
    if usage:
        plan_name = usage.get("plan", "?")
        limits = usage.get("limits", {})
        log(f"Plan: {plan_name}")
        log(f"  Resolution: {limits.get('granularity_allowed', '?')}")
        log(f"  History: {limits.get('max_history_days', '?')} days")
        rpm = limits.get("requests_per_minute", 10)
        log(f"  RPM: {rpm}")
        log(f"  Requests remaining: {limits.get('requests_remaining', '?')}")

        # Adjust rate limit based on plan
        if isinstance(rpm, (int, float)) and rpm > 0:
            client.min_interval = 60.0 / rpm

    # Discover available tags
    tags = client.list_tags()
    sports_tags = [t for t in tags if any(
        st in t.lower() for st_list in SPORT_TAGS.values() for st in st_list
    )]
    log(f"\nAvailable sports tags: {sports_tags[:20]}")

    # Discover sports markets
    sport = args.sport if args.sport != "all" else None
    markets = discover_sports_markets(client, sport)

    if not markets:
        log("No sports markets found.")
        return

    log(f"\nDiscovered {len(markets)} sports markets")

    if args.discover:
        # Just show discovery results
        for m in markets[:20]:
            q = m.get("question", "")[:55]
            mid = m.get("id", "?")
            vol = float(m.get("volume", 0) or 0)
            status = m.get("status", "?")
            n_tokens = len(m.get("tokens", []))
            log(f"  [{mid:>8s}] {status:>8s} vol=${vol:>10,.0f} tokens={n_tokens} {q}")
        if len(markets) > 20:
            log(f"  ... and {len(markets) - 20} more")
        return

    # Download prices
    db = SportsDB(args.db)
    done = load_progress() if args.resume else set()
    remaining = [m for m in markets if str(m.get("id", "")) not in done]

    if args.max_markets > 0:
        remaining = remaining[:args.max_markets]

    log(f"\nMarkets to process: {len(remaining)} (done: {len(done)})")

    total_prices = 0
    total_markets = 0

    for idx, market in enumerate(remaining):
        market_id = str(market.get("id", ""))
        question = market.get("question", "")[:55]
        vol = float(market.get("volume", 0) or 0)

        pct = (idx + 1) / len(remaining) * 100
        if idx % 10 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / max(idx, 1) * (len(remaining) - idx)
            log(f"\n[{idx+1}/{len(remaining)}] ({pct:.0f}%) "
                f"ETA: {eta/60:.0f}min  prices={total_prices:,}  "
                f"req={client.request_count}")

        # Determine sport for this market
        market_tags = market.get("tags", [])
        detected_sport = "unknown"
        for s, tag_list in SPORT_TAGS.items():
            if any(t in (market_tags if isinstance(market_tags, list) else []) for t in tag_list):
                detected_sport = s
                break
            if any(t in question.lower() for t in tag_list):
                detected_sport = s
                break

        # Register in DB
        game_id = register_market_in_db(market, db, detected_sport)

        # Download prices
        n = download_market_prices(
            client, market, db,
            resolution=args.resolution,
            start_date=args.start_date,
            end_date=args.end_date,
        )

        if n > 0:
            total_prices += n
            total_markets += 1
            log(f"  [{market_id:>8s}] {n:>5d} prices  {question}")
        else:
            log(f"  [{market_id:>8s}]     0 prices  {question}")

        done.add(market_id)
        if (idx + 1) % 50 == 0:
            save_progress(done)

    save_progress(done)

    # Summary
    elapsed = time.time() - t_start
    stats = db.stats()
    log(f"\n{'='*60}")
    log(f"  DOWNLOAD COMPLETE")
    log(f"  Duration: {elapsed/60:.1f} min")
    log(f"  Markets processed: {total_markets} with prices / {len(remaining)} total")
    log(f"  Total prices: {total_prices:,}")
    log(f"  API requests: {client.request_count}")
    log(f"\n  DB stats:")
    for k, v in stats.items():
        log(f"    {k:>20s}: {v:>8,d}")
    log(f"{'='*60}")

    db.close()


if __name__ == "__main__":
    main()
