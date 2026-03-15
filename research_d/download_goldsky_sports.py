"""Download Historical Sports Trades from Goldsky Subgraph.

Queries the Polymarket Goldsky orderbook subgraph for trades on
sports markets (NBA, EPL, NFL, UFC, etc.). This gives us real on-chain
trade data going back months — much further than the CLOB /prices-history
endpoint which retains only ~30 days.

Two-phase approach:
1. First discover sports markets via Gamma API (or use existing cache)
2. Then download all trades for each market's YES token from Goldsky
3. Convert trades to hourly prices via VWAP aggregation
4. Store in the sports_backtest.sqlite database

The Goldsky subgraph indexes all OrderFilledEvents on Polymarket's
CTF Exchange contracts (both standard and NegRisk).

Data characteristics:
- Historical coverage: ~12+ months (as far back as markets existed)
- Resolution: tick-level trades → aggregated to hourly VWAP
- No authentication required (public endpoint)
- Rate limiting: conservative 0.3s between requests

Usage:
    python3 research_d/download_goldsky_sports.py --sport all
    python3 research_d/download_goldsky_sports.py --sport nba --max-tokens 100
    python3 research_d/download_goldsky_sports.py --resume  # continue from last run

Output:
    research_d/data/sports_backtest.sqlite  — prices table enriched with Goldsky data
    research_d/data/goldsky_sports_log.txt  — detailed download log
    research_d/data/goldsky_sports_meta.json — download metadata
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()
    SSL_CTX.check_hostname = False
    SSL_CTX.verify_mode = ssl.CERT_NONE

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_d.sports_db import SportsDB

# ── Constants ────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
LOG_PATH = DATA_DIR / "goldsky_sports_log.txt"
META_PATH = DATA_DIR / "goldsky_sports_meta.json"
EVENTS_CACHE = DATA_DIR / "sports_events_cache.json"
PROGRESS_PATH = DATA_DIR / "goldsky_sports_progress.json"

GOLDSKY_URL = (
    "https://api.goldsky.com/api/public/"
    "project_cl6mb8i9h0003e201j6li0diw/"
    "subgraphs/orderbook-subgraph/0.0.1/gn"
)

GAMMA_BASE = "https://gamma-api.polymarket.com"

BATCH_SIZE = 1000
REQUEST_DELAY = 0.3    # Conservative: ~3 req/s
GAMMA_DELAY = 0.2

# Sport detection tags on Polymarket
SPORT_TAGS = {
    "nba": ["nba", "basketball"],
    "epl": ["epl", "premier-league", "premier league"],
    "nfl": ["nfl", "american-football", "americanfootball"],
    "soccer": ["soccer", "football", "ucl", "champions-league",
               "la-liga", "serie-a", "mls"],
    "ufc": ["ufc", "mma", "boxing"],
    "mlb": ["mlb", "baseball"],
    "nhl": ["nhl", "hockey"],
    "ncaab": ["ncaab", "march-madness", "college-basketball"],
    "sports": ["sports"],
}


# ── Logging ──────────────────────────────────────────────────────────

def log(msg: str) -> None:
    """Print and write to log file."""
    print(msg, flush=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


# ── Goldsky GraphQL ──────────────────────────────────────────────────

def graphql_query(query_str: str, retries: int = 3) -> dict | None:
    """Execute a GraphQL query against Goldsky subgraph.

    Args:
        query_str: GraphQL query string.
        retries: Number of retries on failure.

    Returns:
        Data dict from response, or None on failure.
    """
    payload = json.dumps({"query": query_str}).encode()
    for attempt in range(retries):
        req = urllib.request.Request(
            GOLDSKY_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "ArboResearch/2.0",
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
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                log(f"  Query failed after {retries} attempts: {e}")
                return None
    return None


def fetch_trades_for_token(
    token_id: str, side: str = "maker"
) -> list[dict]:
    """Fetch ALL trades for a specific token ID from Goldsky.

    Uses cursor-based pagination via timestamp. Fetches both sides
    of the orderbook (maker + taker) to get complete trade history.

    Args:
        token_id: Polymarket token ID (YES token).
        side: "maker" or "taker" — which side to query.

    Returns:
        List of trade events with timestamp, amounts, etc.
    """
    all_trades: list[dict] = []
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


def fetch_all_trades_for_token(token_id: str) -> list[dict]:
    """Fetch trades from both maker and taker sides."""
    maker_trades = fetch_trades_for_token(token_id, "maker")
    time.sleep(REQUEST_DELAY)
    taker_trades = fetch_trades_for_token(token_id, "taker")

    # Deduplicate by transaction hash
    seen_txs: set[str] = set()
    all_trades: list[dict] = []
    for trade in maker_trades + taker_trades:
        tx = trade.get("transactionHash", "")
        if tx not in seen_txs:
            seen_txs.add(tx)
            all_trades.append(trade)

    return sorted(all_trades, key=lambda t: int(t["timestamp"]))


# ── Price Computation ────────────────────────────────────────────────

def trades_to_prices(
    trades: list[dict], token_id: str
) -> list[dict]:
    """Convert raw Goldsky trades into price points.

    For Polymarket trades:
    - If makerAssetId == token_id: maker sells token, price = USDC/token
    - If takerAssetId == token_id: taker buys token, price = USDC/token

    Args:
        trades: Raw trade events from Goldsky.
        token_id: The YES token ID we're tracking.

    Returns:
        List of {"t": unix_ts, "p": price} dicts.
    """
    prices: list[dict] = []
    for trade in trades:
        ts = int(trade["timestamp"])
        maker_amount = int(trade["makerAmountFilled"])
        taker_amount = int(trade["takerAmountFilled"])

        if maker_amount == 0 or taker_amount == 0:
            continue

        if trade["makerAssetId"] == token_id:
            price = taker_amount / maker_amount
        elif trade["takerAssetId"] == token_id:
            price = maker_amount / taker_amount
        else:
            continue

        # Sanity: prediction market prices are 0-1
        if 0.0001 <= price <= 1.0:
            prices.append({"t": ts, "p": round(price, 6)})

    return prices


def aggregate_to_hourly(prices: list[dict]) -> list[tuple[int, float]]:
    """Aggregate tick-level prices to hourly VWAP.

    Args:
        prices: List of {"t": unix_ts, "p": price} dicts.

    Returns:
        List of (hour_ts, avg_price) tuples, sorted by time.
    """
    if not prices:
        return []

    by_hour: dict[int, list[float]] = defaultdict(list)
    for p in prices:
        hour_ts = (p["t"] // 3600) * 3600
        by_hour[hour_ts].append(p["p"])

    return sorted(
        (hour_ts, round(sum(vals) / len(vals), 6))
        for hour_ts, vals in by_hour.items()
    )


# ── Gamma API — Sports Market Discovery ──────────────────────────────

def _http_get(url: str) -> Any:
    """Simple HTTP GET with retry."""
    for attempt in range(3):
        req = urllib.request.Request(url, headers={"User-Agent": "ArboResearch/2.0"})
        try:
            with urllib.request.urlopen(req, timeout=20, context=SSL_CTX) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                log(f"  HTTP GET failed: {url} — {e}")
                return None


def discover_sports_events(sport_filter: str | None = None) -> list[dict]:
    """Discover sports events from Gamma API.

    Fetches all closed sports events (resolved markets) by querying
    multiple sport-related tags.

    Args:
        sport_filter: Optional sport to filter ("nba", "epl", etc.)

    Returns:
        List of Gamma API event dicts.
    """
    # Use cached events if available and recent
    if EVENTS_CACHE.exists():
        with open(EVENTS_CACHE) as f:
            cached = json.load(f)
        log(f"Loaded {len(cached)} events from cache")
        if sport_filter:
            cached = [e for e in cached if _event_matches_sport(e, sport_filter)]
            log(f"  Filtered to {len(cached)} {sport_filter} events")
        return cached

    log("Discovering sports events from Gamma API...")

    tags_to_query = ["sports"]
    if sport_filter and sport_filter in SPORT_TAGS:
        tags_to_query = SPORT_TAGS[sport_filter]
    else:
        for tag_list in SPORT_TAGS.values():
            tags_to_query.extend(tag_list)
        tags_to_query = list(set(tags_to_query))

    all_events: dict[str, dict] = {}  # event_id → event (deduplicate)

    for tag in tags_to_query:
        offset = 0
        while True:
            url = (
                f"{GAMMA_BASE}/events"
                f"?tag={tag}&closed=true&limit=100&offset={offset}"
            )
            data = _http_get(url)
            time.sleep(GAMMA_DELAY)

            if not data or not isinstance(data, list) or len(data) == 0:
                break

            for event in data:
                eid = event.get("id", "")
                if eid and eid not in all_events:
                    all_events[eid] = event

            log(f"  Tag '{tag}': offset={offset}, got {len(data)} events "
                f"(total unique: {len(all_events)})")

            if len(data) < 100:
                break
            offset += 100

    events_list = list(all_events.values())

    # Cache for future runs
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVENTS_CACHE, "w") as f:
        json.dump(events_list, f)
    log(f"Cached {len(events_list)} events to {EVENTS_CACHE}")

    if sport_filter:
        events_list = [e for e in events_list if _event_matches_sport(e, sport_filter)]
        log(f"Filtered to {len(events_list)} {sport_filter} events")

    return events_list


def _event_matches_sport(event: dict, sport: str) -> bool:
    """Check if an event matches a sport filter."""
    tags = []
    if isinstance(event.get("tags"), list):
        tags = [t.get("label", "").lower() if isinstance(t, dict) else str(t).lower()
                for t in event["tags"]]
    elif isinstance(event.get("tags"), str):
        tags = [event["tags"].lower()]

    title = event.get("title", "").lower()

    sport_keywords = SPORT_TAGS.get(sport, [sport])
    return any(kw in " ".join(tags + [title]) for kw in sport_keywords)


def extract_token_ids(events: list[dict]) -> list[dict]:
    """Extract YES token IDs from Gamma API events.

    Args:
        events: Gamma API event list.

    Returns:
        List of dicts: {"token_id": str, "event_id": str, "title": str, "question": str}
    """
    tokens: list[dict] = []

    for event in events:
        event_id = event.get("id", "")
        title = event.get("title", "")
        markets = event.get("markets", [])

        for market in markets:
            # clobTokenIds is a JSON-encoded string, not a list!
            raw_token_ids = market.get("clobTokenIds", "[]")
            if isinstance(raw_token_ids, str):
                try:
                    token_ids = json.loads(raw_token_ids)
                except json.JSONDecodeError:
                    continue
            else:
                token_ids = raw_token_ids

            if not token_ids:
                continue

            # First token is YES
            yes_token = token_ids[0]
            question = market.get("question", "")

            tokens.append({
                "token_id": yes_token,
                "event_id": event_id,
                "title": title,
                "question": question,
                "volume": float(market.get("volume", 0) or 0),
                "neg_risk": 1 if event.get("enableNegRisk") else 0,
            })

    return tokens


# ── Progress Tracking ────────────────────────────────────────────────

def load_progress() -> set[str]:
    """Load set of already-downloaded token IDs."""
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as f:
            return set(json.load(f))
    return set()


def save_progress(done: set[str]) -> None:
    """Save progress to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(sorted(done), f)


# ── Main Pipeline ────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download sports trades from Goldsky subgraph.",
    )
    parser.add_argument(
        "--sport",
        choices=["nba", "epl", "nfl", "soccer", "ufc", "all"],
        default="all",
        help="Sport filter (default: all).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=0,
        help="Max tokens to download (0=unlimited).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last run (skip already-downloaded tokens).",
    )
    parser.add_argument(
        "--refresh-events", action="store_true",
        help="Force re-fetch events from Gamma API (ignore cache).",
    )
    parser.add_argument(
        "--hourly", action="store_true", default=True,
        help="Aggregate to hourly prices (default: True).",
    )
    parser.add_argument(
        "--db", default=None,
        help="Path to SQLite database.",
    )
    args = parser.parse_args()

    t_start = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log(f"\n{'='*70}")
    log(f"GOLDSKY SPORTS TRADE DOWNLOAD")
    log(f"Start: {datetime.now(timezone.utc).isoformat()}")
    log(f"Sport: {args.sport}")
    log(f"Resume: {args.resume}")
    log(f"{'='*70}\n")

    # Phase 1: Discover sports events
    sport_filter = None if args.sport == "all" else args.sport

    if args.refresh_events and EVENTS_CACHE.exists():
        EVENTS_CACHE.unlink()
        log("Cleared events cache")

    events = discover_sports_events(sport_filter)
    if not events:
        log("No sports events found. Exiting.")
        return

    # Phase 2: Extract token IDs
    tokens = extract_token_ids(events)
    log(f"\nExtracted {len(tokens)} YES tokens from {len(events)} events")

    # Sort by volume (highest first — most important data)
    tokens.sort(key=lambda t: t["volume"], reverse=True)

    if args.max_tokens > 0:
        tokens = tokens[:args.max_tokens]
        log(f"Limited to top {args.max_tokens} tokens by volume")

    # Phase 3: Load progress for resume
    done_tokens = load_progress() if args.resume else set()
    remaining = [t for t in tokens if t["token_id"] not in done_tokens]
    log(f"Already downloaded: {len(done_tokens)}")
    log(f"Remaining: {len(remaining)}")

    if not remaining:
        log("All tokens already downloaded. Nothing to do.")
        return

    # Phase 4: Download trades and compute prices
    db = SportsDB(args.db)

    total_trades = 0
    total_prices = 0
    errors = 0

    for idx, token_info in enumerate(remaining):
        token_id = token_info["token_id"]
        title = token_info["title"][:60]
        question = token_info["question"][:60]

        # Progress report
        pct = (idx + 1) / len(remaining) * 100
        elapsed = time.time() - t_start
        eta = elapsed / max(idx, 1) * (len(remaining) - idx)

        if idx % 20 == 0 or idx == len(remaining) - 1:
            log(f"\n[{idx+1}/{len(remaining)}] ({pct:.1f}%) "
                f"ETA: {eta/60:.0f}min  "
                f"trades={total_trades:,}  prices={total_prices:,}")

        log(f"  Token {token_id[:16]}...  {title}")

        try:
            # Fetch trades from both sides
            trades = fetch_all_trades_for_token(token_id)

            if not trades:
                log(f"    No trades found")
                done_tokens.add(token_id)
                continue

            # Convert to prices
            prices = trades_to_prices(trades, token_id)

            if not prices:
                log(f"    {len(trades)} trades but no valid prices")
                done_tokens.add(token_id)
                continue

            # Aggregate to hourly
            if args.hourly:
                hourly = aggregate_to_hourly(prices)
            else:
                hourly = [(p["t"], p["p"]) for p in prices]

            # Store in database
            n_stored = db.insert_prices_simple(token_id, hourly)

            total_trades += len(trades)
            total_prices += len(hourly)

            log(f"    {len(trades)} trades → {len(hourly)} hourly prices "
                f"({_ts_range(hourly)})")

            # Track progress
            done_tokens.add(token_id)
            if (idx + 1) % 50 == 0:
                save_progress(done_tokens)

        except Exception as e:
            log(f"    ERROR: {e}")
            errors += 1
            if errors > 20:
                log("Too many errors. Stopping.")
                break

        time.sleep(REQUEST_DELAY)

    # Save final progress
    save_progress(done_tokens)

    # Phase 5: Summary
    elapsed = time.time() - t_start
    stats = db.stats()

    log(f"\n{'='*70}")
    log(f"GOLDSKY DOWNLOAD COMPLETE")
    log(f"{'='*70}")
    log(f"  Duration: {elapsed/60:.1f} minutes")
    log(f"  Tokens processed: {len(done_tokens)}")
    log(f"  Total trades: {total_trades:,}")
    log(f"  Total hourly prices: {total_prices:,}")
    log(f"  Errors: {errors}")
    log(f"\n  Database stats:")
    for key, value in stats.items():
        log(f"    {key:>20s}: {value:>8,d}")
    log(f"{'='*70}\n")

    # Save metadata
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sport": args.sport,
        "tokens_processed": len(done_tokens),
        "total_trades": total_trades,
        "total_prices": total_prices,
        "errors": errors,
        "duration_minutes": round(elapsed / 60, 1),
        "db_stats": stats,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    log(f"Metadata saved to {META_PATH}")

    db.close()


def _ts_range(prices: list[tuple[int, float]]) -> str:
    """Format time range for logging."""
    if not prices:
        return "empty"
    t_min = datetime.fromtimestamp(prices[0][0], tz=timezone.utc)
    t_max = datetime.fromtimestamp(prices[-1][0], tz=timezone.utc)
    return f"{t_min.strftime('%Y-%m-%d')} → {t_max.strftime('%Y-%m-%d')}"


if __name__ == "__main__":
    main()
