"""Download Sports Orderbook Data from pmxt Archive.

The pmxt archive (archive.pmxt.dev) provides hourly Parquet snapshots of
ALL Polymarket orderbook data since Feb 21, 2026. Each file contains
book_snapshot and price_change events with best_bid/best_ask.

This script:
1. Discovers sports market conditionIds from Gamma API
2. Downloads pmxt Parquet files (300-750 MB each)
3. Filters to sports conditionIds using PyArrow predicate pushdown
4. Extracts best_bid/best_ask as hourly prices
5. Stores in sports_backtest.sqlite

This gives us ~22 days of hourly price data for ALL active sports
markets — much more coverage than CLOB (30-day, per-token) or
Goldsky (tick-level but sparse for recent markets).

Requirements:
    pip install pyarrow

Usage:
    PYTHONPATH=. python3 research_d/download_pmxt_sports.py
    PYTHONPATH=. python3 research_d/download_pmxt_sports.py --start 2026-03-01 --end 2026-03-15
    PYTHONPATH=. python3 research_d/download_pmxt_sports.py --hours-per-day 4  # sample 4 hours/day

Output:
    research_d/data/sports_backtest.sqlite — prices table enriched
    research_d/data/pmxt_cache/            — cached filtered Parquet extracts
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import sys
import tempfile
import time
import urllib.request
from collections import defaultdict
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
PMXT_CACHE_DIR = DATA_DIR / "pmxt_cache"
CONDITION_IDS_PATH = DATA_DIR / "sports_condition_ids.json"

R2_BASE = "https://r2.pmxt.dev"
GAMMA_BASE = "https://gamma-api.polymarket.com"
FIRST_AVAILABLE = datetime(2026, 2, 21, 16, tzinfo=timezone.utc)

# Sports tags to query on Gamma API
SPORT_TAGS = ["nba", "epl", "nfl", "soccer", "ufc", "mma", "mlb", "nhl"]


def log(msg: str) -> None:
    print(msg, flush=True)


# ── Gamma API: Discover Sports ConditionIds ──────────────────────────

def _http_get_json(url: str) -> Any:
    """HTTP GET returning parsed JSON."""
    for attempt in range(3):
        req = urllib.request.Request(url, headers={"User-Agent": "ArboResearch/2.0"})
        try:
            with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                log(f"  HTTP error: {url} — {e}")
                return None


def discover_sports_condition_ids(
    force_refresh: bool = False,
) -> dict[str, dict]:
    """Discover sports market conditionIds from Gamma API.

    Returns:
        Dict mapping conditionId → {token_id, question, sport_tag, event_id, ...}
    """
    if CONDITION_IDS_PATH.exists() and not force_refresh:
        with open(CONDITION_IDS_PATH) as f:
            data = json.load(f)
        if isinstance(data, dict) and len(data) > 0:
            log(f"Loaded {len(data)} conditionIds from cache")
            return data

    log("Discovering sports conditionIds from Gamma API...")
    all_cids: dict[str, dict] = {}

    for tag in SPORT_TAGS:
        offset = 0
        tag_count = 0
        while offset < 5000:
            # Get ACTIVE (not closed) sports events for recent data
            url = f"{GAMMA_BASE}/events?tag={tag}&limit=100&offset={offset}"
            data = _http_get_json(url)
            time.sleep(0.2)

            if not data or not isinstance(data, list) or len(data) == 0:
                break

            for event in data:
                markets = event.get("markets", [])
                for market in markets:
                    cid = market.get("conditionId", "")
                    if not cid:
                        continue

                    # Parse clobTokenIds
                    raw_tokens = market.get("clobTokenIds", "[]")
                    if isinstance(raw_tokens, str):
                        try:
                            token_ids = json.loads(raw_tokens)
                        except json.JSONDecodeError:
                            continue
                    else:
                        token_ids = raw_tokens

                    if not token_ids:
                        continue

                    all_cids[cid] = {
                        "condition_id": cid,
                        "token_id_yes": token_ids[0],
                        "token_id_no": token_ids[1] if len(token_ids) > 1 else None,
                        "question": market.get("question", ""),
                        "sport_tag": tag,
                        "event_id": event.get("id", ""),
                        "event_title": event.get("title", ""),
                        "volume": float(market.get("volume", 0) or 0),
                        "closed": market.get("closed", False),
                        "end_date": market.get("endDate", ""),
                        "neg_risk": 1 if event.get("enableNegRisk") else 0,
                    }
                    tag_count += 1

            if len(data) < 100:
                break
            offset += 100

        log(f"  Tag '{tag}': {tag_count} markets (total: {len(all_cids)})")

    # Also get closed/resolved events for backtest data
    for tag in SPORT_TAGS:
        offset = 0
        while offset < 3000:
            url = f"{GAMMA_BASE}/events?tag={tag}&closed=true&limit=100&offset={offset}"
            data = _http_get_json(url)
            time.sleep(0.2)

            if not data or not isinstance(data, list) or len(data) == 0:
                break

            for event in data:
                for market in event.get("markets", []):
                    cid = market.get("conditionId", "")
                    if not cid or cid in all_cids:
                        continue

                    raw_tokens = market.get("clobTokenIds", "[]")
                    if isinstance(raw_tokens, str):
                        try:
                            token_ids = json.loads(raw_tokens)
                        except json.JSONDecodeError:
                            continue
                    else:
                        token_ids = raw_tokens

                    if not token_ids:
                        continue

                    all_cids[cid] = {
                        "condition_id": cid,
                        "token_id_yes": token_ids[0],
                        "token_id_no": token_ids[1] if len(token_ids) > 1 else None,
                        "question": market.get("question", ""),
                        "sport_tag": tag,
                        "event_id": event.get("id", ""),
                        "event_title": event.get("title", ""),
                        "volume": float(market.get("volume", 0) or 0),
                        "closed": market.get("closed", True),
                        "end_date": market.get("endDate", ""),
                        "neg_risk": 1 if event.get("enableNegRisk") else 0,
                    }

            if len(data) < 100:
                break
            offset += 100

    # Save cache
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONDITION_IDS_PATH, "w") as f:
        json.dump(all_cids, f)
    log(f"Cached {len(all_cids)} sports conditionIds")

    return all_cids


# ── pmxt Parquet Download + Processing ───────────────────────────────

def download_and_process_hour(
    hour_dt: datetime,
    condition_ids: set[str],
    cid_to_token: dict[str, str],
    db: SportsDB,
    keep_parquet: bool = False,
) -> int:
    """Download one pmxt hourly Parquet file and extract sports prices.

    Args:
        hour_dt: UTC datetime for the hour to download.
        condition_ids: Set of conditionIds to filter on.
        cid_to_token: Mapping conditionId → YES token_id.
        db: SportsDB instance.
        keep_parquet: Whether to keep the raw Parquet file.

    Returns:
        Number of price points extracted and stored.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        log("ERROR: pyarrow not installed. Run: pip3 install pyarrow")
        sys.exit(1)

    ts_str = hour_dt.strftime("%Y-%m-%dT%H")
    filename = f"polymarket_orderbook_{ts_str}.parquet"
    url = f"{R2_BASE}/{filename}"

    # Check if already processed
    cache_marker = PMXT_CACHE_DIR / f"{ts_str}.done"
    if cache_marker.exists():
        return 0

    PMXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Download to temp file (files are 300-750 MB)
    tmp_path = PMXT_CACHE_DIR / f"{ts_str}.tmp.parquet"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ArboResearch/2.0"})
        with urllib.request.urlopen(req, timeout=120, context=SSL_CTX) as resp:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

            if total_size > 0 and downloaded < total_size * 0.9:
                log(f"  {ts_str}: incomplete download ({downloaded}/{total_size})")
                tmp_path.unlink(missing_ok=True)
                return 0

    except Exception as e:
        log(f"  {ts_str}: download failed — {e}")
        tmp_path.unlink(missing_ok=True)
        return 0

    # Read and filter with PyArrow predicate pushdown
    try:
        # Read only rows matching our conditionIds
        table = pq.read_table(
            str(tmp_path),
            filters=[("market_id", "in", list(condition_ids))],
        )

        if table.num_rows == 0:
            log(f"  {ts_str}: 0 sports rows (file {downloaded // (1024*1024)}MB)")
            cache_marker.touch()
            tmp_path.unlink(missing_ok=True)
            return 0

        # Extract best_bid prices from book_snapshot events
        prices_to_insert: list[tuple[str, int, float]] = []

        for i in range(table.num_rows):
            update_type = table.column("update_type")[i].as_py()
            if update_type != "book_snapshot":
                continue

            market_id = table.column("market_id")[i].as_py()
            data_str = table.column("data")[i].as_py()

            if not data_str or market_id not in cid_to_token:
                continue

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            best_bid = data.get("best_bid")
            best_ask = data.get("best_ask")
            ts = data.get("timestamp")

            if best_bid is None or ts is None:
                continue

            token_id = cid_to_token[market_id]
            price = float(best_bid)  # best_bid = YES price

            if 0.001 <= price <= 0.999:
                # Round to minute for dedup
                minute_ts = (int(ts) // 60) * 60
                prices_to_insert.append((token_id, minute_ts, price))

        # Deduplicate by token+minute (keep last)
        deduped: dict[tuple[str, int], float] = {}
        for token_id, ts, price in prices_to_insert:
            deduped[(token_id, ts)] = price

        # Insert into DB
        if deduped:
            price_rows = [(tid, ts, p) for (tid, ts), p in deduped.items()]
            db.conn.executemany(
                "INSERT OR IGNORE INTO prices (token_id, ts, price) VALUES (?, ?, ?)",
                price_rows,
            )
            db.conn.commit()

        n_stored = len(deduped)
        n_tokens = len(set(tid for tid, _ in deduped))
        log(f"  {ts_str}: {n_stored} prices from {n_tokens} tokens "
            f"({table.num_rows} sports rows, file {downloaded // (1024*1024)}MB)")

        cache_marker.touch()

    except Exception as e:
        log(f"  {ts_str}: processing error — {e}")
        n_stored = 0

    finally:
        if not keep_parquet:
            tmp_path.unlink(missing_ok=True)

    return n_stored


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download sports orderbook data from pmxt archive.",
    )
    parser.add_argument(
        "--start", default="2026-02-22",
        help="Start date (YYYY-MM-DD), default: 2026-02-22.",
    )
    parser.add_argument(
        "--end", default=None,
        help="End date (YYYY-MM-DD), default: today.",
    )
    parser.add_argument(
        "--hours-per-day", type=int, default=6,
        help="Hours to sample per day (default: 6 = every 4h). Use 24 for all.",
    )
    parser.add_argument(
        "--refresh-cids", action="store_true",
        help="Force re-discover conditionIds from Gamma API.",
    )
    parser.add_argument(
        "--keep-parquet", action="store_true",
        help="Keep raw Parquet files (WARNING: 300-750 MB each!).",
    )
    parser.add_argument(
        "--db", default=None,
    )
    args = parser.parse_args()

    t_start = time.time()
    log(f"\n{'='*60}")
    log(f"  pmxt Sports Orderbook Download")
    log(f"  Start: {datetime.now(timezone.utc).isoformat()}")
    log(f"{'='*60}\n")

    # Phase 1: Discover sports conditionIds
    cid_map = discover_sports_condition_ids(args.refresh_cids)
    if not cid_map:
        log("No sports conditionIds found. Exiting.")
        return

    condition_ids = set(cid_map.keys())
    cid_to_token = {cid: info["token_id_yes"] for cid, info in cid_map.items()}
    log(f"\nFiltering pmxt data for {len(condition_ids)} sports conditionIds")

    # Also register markets in DB
    db = SportsDB(args.db)
    for cid, info in cid_map.items():
        # Create a game record if needed
        from research_d.download_sports_prices import (
            parse_teams_from_question,
            team_to_abbreviation,
            generate_game_id,
            parse_sport_from_tags,
        )
        sport = info["sport_tag"]
        question = info["question"]
        teams = parse_teams_from_question(question)

        if teams:
            away_name, home_name = teams
            away_abbr = team_to_abbreviation(away_name, sport)
            home_abbr = team_to_abbreviation(home_name, sport)
            end_date = info.get("end_date", "")
            game_date = end_date[:10] if end_date else ""
            game_id = generate_game_id(sport, game_date, home_abbr, away_abbr)
        else:
            game_id = f"{sport}_{info['event_id'][:12]}"
            game_date = info.get("end_date", "")[:10]
            home_abbr = "HOME"
            away_abbr = "AWAY"

        db.upsert_game(
            game_id=game_id,
            sport=sport,
            league=sport.upper(),
            home_team=home_abbr,
            away_team=away_abbr,
            game_date=game_date or "2026-01-01",
            status="final" if info.get("closed") else "scheduled",
        )
        db.upsert_market(
            token_id=info["token_id_yes"],
            game_id=game_id,
            event_id=info["event_id"],
            condition_id=cid,
            token_id_no=info.get("token_id_no"),
            question=question,
            outcome="moneyline",
            volume=info.get("volume"),
            neg_risk=info.get("neg_risk", 0),
        )

    db.conn.commit()
    log(f"Registered {len(cid_map)} markets in DB")

    # Phase 2: Generate hour list
    start_dt = max(
        datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc),
        FIRST_AVAILABLE,
    )
    end_dt = (
        datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.end
        else datetime.now(timezone.utc)
    )

    # Generate hours to download (sampled)
    hours_to_download: list[datetime] = []
    current = start_dt
    step = 24 // args.hours_per_day  # e.g., 6 hours/day → step=4 → sample at 0,4,8,12,16,20

    while current <= end_dt:
        hours_to_download.append(current)
        current += timedelta(hours=step)

    log(f"\nHours to download: {len(hours_to_download)} "
        f"({args.hours_per_day}/day, {start_dt.date()} → {end_dt.date()})")

    # Phase 3: Download and process
    total_prices = 0
    errors = 0

    for idx, hour_dt in enumerate(hours_to_download):
        pct = (idx + 1) / len(hours_to_download) * 100
        if idx % 10 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / max(idx, 1) * (len(hours_to_download) - idx)
            log(f"\n[{idx+1}/{len(hours_to_download)}] ({pct:.0f}%) "
                f"ETA: {eta/60:.0f}min  total_prices={total_prices:,}")

        try:
            n = download_and_process_hour(
                hour_dt, condition_ids, cid_to_token, db, args.keep_parquet,
            )
            total_prices += n
        except Exception as e:
            log(f"  ERROR at {hour_dt}: {e}")
            errors += 1
            if errors > 10:
                log("Too many errors, stopping.")
                break

        time.sleep(2)  # Politeness delay

    # Phase 4: Summary
    elapsed = time.time() - t_start
    stats = db.stats()

    log(f"\n{'='*60}")
    log(f"  pmxt DOWNLOAD COMPLETE")
    log(f"  Duration: {elapsed/60:.1f} minutes")
    log(f"  Hours processed: {len(hours_to_download)}")
    log(f"  Total prices extracted: {total_prices:,}")
    log(f"  Errors: {errors}")
    log(f"\n  Database stats:")
    for key, value in stats.items():
        log(f"    {key:>20s}: {value:>8,d}")
    log(f"{'='*60}\n")

    db.close()


if __name__ == "__main__":
    main()
