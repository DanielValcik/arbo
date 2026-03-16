"""Parallel PolymarketData.co downloader — 6 workers for 6× speed.

Splits 142K markets into N chunks and runs download_polymarketdata.py
in parallel subprocesses. Each worker handles a different slice of markets.

The original single-threaded downloader uses only ~16% of the 2000 RPM
Ultra quota. With 6 workers we use ~31% and finish in ~5 days vs 30.

Usage:
    PYTHONPATH=. python3 research_d/download_pmd_parallel.py --workers 6 --resolution 10m
    PYTHONPATH=. python3 research_d/download_pmd_parallel.py --workers 6 --resolution 1m --game-window-hours 48 --market-type moneyline

All workers share the same SQLite DB (WAL mode handles concurrent writes)
and the same progress file (with file locking).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
CACHE_PATH = DATA_DIR / "pmd_cache" / "markets_all.json"
PROGRESS_PATH = DATA_DIR / "pmd_progress.json"


def load_markets() -> list[dict]:
    """Load cached markets."""
    if not CACHE_PATH.exists():
        print(f"ERROR: No market cache at {CACHE_PATH}")
        print("Run discovery first: python3 research_d/download_polymarketdata.py --discover --sport all")
        sys.exit(1)
    return json.loads(CACHE_PATH.read_text())


def load_progress() -> set[str]:
    """Load already-completed market IDs."""
    if PROGRESS_PATH.exists():
        return set(json.loads(PROGRESS_PATH.read_text()))
    return set()


def filter_markets(markets: list[dict], market_type: str) -> list[dict]:
    """Filter markets by type."""
    if market_type == "all":
        return markets
    keywords = {
        "moneyline": ["winner", "win", "beat", "vs", "moneyline"],
        "spread": ["spread"],
        "over_under": ["o/u", "over/under", "total"],
        "props": ["assists", "rebounds", "points", "steals", "blocks",
                  "threes", "yards", "touchdowns", "goals scored"],
    }
    kws = keywords.get(market_type, [])
    return [m for m in markets if any(
        kw in (m.get("question") or "").lower() for kw in kws
    )]


def split_into_chunks(items: list, n_chunks: int) -> list[list]:
    """Split list into N roughly equal chunks."""
    chunk_size = math.ceil(len(items) / n_chunks)
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def main():
    parser = argparse.ArgumentParser(
        description="Parallel PolymarketData downloader.",
    )
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel workers.")
    parser.add_argument("--resolution", default="10m", choices=["1m", "10m", "1h"])
    parser.add_argument("--max-history-days", type=int, default=9999)
    parser.add_argument("--game-window-hours", type=int, default=0)
    parser.add_argument("--market-type", default="all",
                        choices=["all", "moneyline", "spread", "over_under", "props"])
    parser.add_argument("--db", default=None)
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  Parallel PolymarketData Download")
    print(f"  Workers: {args.workers}, Resolution: {args.resolution}")
    print(f"  {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}")

    # Load and filter markets
    markets = load_markets()
    print(f"Total markets in cache: {len(markets):,}")

    markets = filter_markets(markets, args.market_type)
    print(f"After type filter ({args.market_type}): {len(markets):,}")

    done = load_progress()
    remaining = [m for m in markets if str(m.get("id", "")) not in done]
    print(f"Already done: {len(done):,}, Remaining: {len(remaining):,}")

    if not remaining:
        print("All markets done!")
        return

    # Split into chunks for workers
    chunks = split_into_chunks(remaining, args.workers)
    print(f"Split into {len(chunks)} chunks: {[len(c) for c in chunks]}")

    # Write chunk files
    chunk_dir = DATA_DIR / "pmd_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_files = []
    for i, chunk in enumerate(chunks):
        chunk_path = chunk_dir / f"chunk_{i}.json"
        chunk_path.write_text(json.dumps([str(m.get("id", "")) for m in chunk]))
        chunk_files.append(chunk_path)

    # Write worker script
    worker_script = DATA_DIR / "pmd_worker.py"
    worker_script.write_text(f'''"""PMD worker — processes a chunk of market IDs."""
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pathlib import Path
from research_d.download_polymarketdata import (
    PMDataClient, download_market_prices, register_market_in_db,
    load_progress, mark_done, log,
)
from research_d.sports_db import SportsDB
from datetime import datetime, timedelta, timezone

chunk_file = sys.argv[1]
worker_id = sys.argv[2]
resolution = "{args.resolution}"
max_history_days = {args.max_history_days}
game_window_hours = {args.game_window_hours}

# Load API key
api_key = ""
for line in Path(".env").read_text().splitlines():
    if line.startswith("POLYMARKETDATA_API_KEY="):
        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
        break

client = PMDataClient(api_key)
# Set rate limit per worker (shared 2000 RPM / N workers)
client.min_interval = 60.0 / (2000 / {args.workers})

db = SportsDB({repr(args.db)})

# Load all markets for metadata lookup
all_markets = json.loads(
    Path("research_d/data/pmd_cache/markets_all.json").read_text()
)
market_by_id = {{str(m.get("id", "")): m for m in all_markets}}

# Load chunk
chunk_ids = json.loads(Path(chunk_file).read_text())
log(f"[Worker {{worker_id}}] Processing {{len(chunk_ids)}} markets")

total_prices = 0
for idx, mid in enumerate(chunk_ids):
    # Check if already done (shared progress)
    done = load_progress()
    if mid in done:
        continue

    market = market_by_id.get(mid)
    if not market:
        continue

    q = (market.get("question") or "")[:50]

    # Determine sport
    detected_sport = "unknown"
    from research_d.download_polymarketdata import SPORT_TAGS
    for s, tags in SPORT_TAGS.items():
        if any(t in q.lower() for t in tags):
            detected_sport = s
            break

    register_market_in_db(market, db, detected_sport)

    # Game window mode
    effective_start = None
    if game_window_hours > 0:
        m_end = market.get("end_date") or market.get("resolution_date", "")
        if m_end:
            try:
                end_dt = datetime.fromisoformat(str(m_end).replace("Z", "+00:00"))
                start_dt = end_dt - timedelta(hours=game_window_hours)
                effective_start = start_dt.strftime("%Y-%m-%d")
            except:
                pass

    n = download_market_prices(
        client, market, db,
        resolution=resolution,
        start_date=effective_start,
        max_history_days=max_history_days,
    )
    total_prices += n

    # Append to progress log (safe for parallel workers, no race condition)
    mark_done(mid)

    if (idx + 1) % 20 == 0:
        pct = (idx + 1) / len(chunk_ids) * 100
        log(f"[Worker {{worker_id}}] {{idx+1}}/{{len(chunk_ids)}} ({{pct:.0f}}%) prices={{total_prices:,}}")

log(f"[Worker {{worker_id}}] DONE: {{total_prices:,}} prices from {{len(chunk_ids)}} markets")
db.close()
''')

    # Launch workers
    print(f"\nLaunching {len(chunks)} workers...")
    processes = []
    for i, chunk_path in enumerate(chunk_files):
        log_path = DATA_DIR / f"pmd_worker_{i}.log"
        p = subprocess.Popen(
            [sys.executable, str(worker_script), str(chunk_path), str(i)],
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        processes.append((i, p, log_path))
        print(f"  Worker {i}: PID {p.pid}, chunk {len(chunks[i])} markets, log {log_path.name}")

    # Monitor
    print(f"\nAll {len(processes)} workers running. Monitoring...")
    start_time = time.time()

    while True:
        time.sleep(60)
        alive = sum(1 for _, p, _ in processes if p.poll() is None)
        done = load_progress()
        elapsed_h = (time.time() - start_time) / 3600
        rate = len(done) / max(elapsed_h, 0.01)

        remaining_count = len(remaining) - len(done)
        eta_h = remaining_count / max(rate, 1)

        print(f"  [{elapsed_h:.1f}h] Done: {len(done):,}/{len(remaining):,} "
              f"({len(done)/len(remaining)*100:.1f}%) "
              f"Workers alive: {alive} "
              f"Rate: {rate:.0f}/h ETA: {eta_h:.1f}h ({eta_h/24:.1f}d)")

        if alive == 0:
            print("\nAll workers finished!")
            break

    # Final stats
    elapsed = time.time() - start_time
    done = load_progress()
    print(f"\n{'='*60}")
    print(f"  PARALLEL DOWNLOAD COMPLETE")
    print(f"  Duration: {elapsed/3600:.1f} hours")
    print(f"  Markets done: {len(done):,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
