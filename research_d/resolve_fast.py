"""Fast parallel resolution backfill for Strategy D PMD database.

Only resolves SPORTS markets (skips politics/crypto/entertainment).
Uses 10 concurrent workers with PMD API rate limiting (2000 RPM).

~60K queries → ~30 minutes (vs ~6 hours sequential).

Usage on VPS:
    cd /opt/arbo
    export $(grep POLYMARKETDATA_API_KEY .env | xargs)
    PYTHONPATH=. python3 research_d/resolve_fast.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()
    SSL_CTX.check_hostname = False
    SSL_CTX.verify_mode = ssl.CERT_NONE

DB_PATH = Path(__file__).parent / "data" / "sports_backtest.sqlite"
PROGRESS_FILE = Path(__file__).parent / "data" / "resolve_fast_progress.txt"
PMD_API_BASE = "https://api.polymarketdata.co/v1"

# ── Rate limiter ──────────────────────────────────────────────────────

class TokenBucket:
    """Thread-safe token bucket rate limiter."""
    def __init__(self, rate: float, burst: int = 1):
        self.rate = rate         # tokens per second
        self.burst = burst
        self.tokens = float(burst)
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.burst, self.tokens + (now - self.last) * self.rate)
                self.last = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
            time.sleep(0.01)


# ── API call ──────────────────────────────────────────────────────────

def pmd_get_market(pmd_id: str, api_key: str, bucket: TokenBucket) -> dict | None:
    """Get market from PMD API with rate limiting and retry."""
    bucket.acquire()
    url = f"{PMD_API_BASE}/markets/{pmd_id}"
    for attempt in range(3):
        req = urllib.request.Request(url, headers={
            "X-API-Key": api_key,
            "Accept": "application/json",
            "User-Agent": "ArboResolveFast/1.0",
        })
        try:
            with urllib.request.urlopen(req, timeout=15, context=SSL_CTX) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2 ** (attempt + 1))
            elif e.code in (403, 404):
                return None
            else:
                if attempt < 2:
                    time.sleep(1)
                else:
                    return None
        except Exception:
            if attempt < 2:
                time.sleep(1)
            else:
                return None
    return None


# ── Worker ────────────────────────────────────────────────────────────

def resolve_market(pmd_id: str, api_key: str, bucket: TokenBucket) -> list[tuple[int, str]]:
    """Resolve a single PMD market. Returns list of (won, token_id) updates."""
    resp = pmd_get_market(pmd_id, api_key, bucket)
    if not resp:
        return []

    status = resp.get("status", "")
    resolved_token_id = str(resp.get("resolved_token_id") or "")
    tokens = resp.get("tokens", [])

    if status not in ("closed", "resolved") or not resolved_token_id:
        return []

    updates = []
    for token in tokens:
        token_id = str(token.get("id", ""))
        if not token_id:
            continue
        won = 1 if token_id == resolved_token_id else 0
        updates.append((won, token_id))
    return updates


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.environ.get("POLYMARKETDATA_API_KEY", "")
    if not api_key:
        print("ERROR: Set POLYMARKETDATA_API_KEY in environment")
        sys.exit(1)

    db_path = DB_PATH
    if len(sys.argv) > 1:
        db_path = Path(sys.argv[1])

    print(f"Database: {db_path}", flush=True)

    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")

    # Load already-done IDs
    done: set[str] = set()
    if PROGRESS_FILE.exists():
        done = set(PROGRESS_FILE.read_text().strip().split("\n"))
        done.discard("")
    print(f"Already resolved: {len(done)} event_ids from previous run", flush=True)

    # Get SPORTS-ONLY unresolved event_ids
    rows = conn.execute("""
        SELECT DISTINCT m.event_id
        FROM markets m
        JOIN games g ON m.game_id = g.game_id
        WHERE m.won IS NULL
        AND g.sport != 'unknown'
        AND m.event_id IS NOT NULL
        AND m.event_id != ''
    """).fetchall()
    pmd_ids = [r[0] for r in rows if r[0] not in done]
    total = len(pmd_ids)
    print(f"Sports markets to resolve: {total} event_ids", flush=True)

    if total == 0:
        print("Nothing to resolve!")
        conn.close()
        return

    # Rate limiter: 2000 RPM = 33/s, but leave headroom → 28/s
    bucket = TokenBucket(rate=28.0, burst=10)
    workers = 10

    print(f"Starting {workers} parallel workers (rate limit: 28 req/s)...", flush=True)
    t0 = time.time()
    resolved_count = 0
    queried_count = 0
    batch_updates: list[tuple[int, str]] = []
    progress_fd = open(PROGRESS_FILE, "a")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Submit all tasks
        future_to_id = {
            pool.submit(resolve_market, pid, api_key, bucket): pid
            for pid in pmd_ids
        }

        for future in as_completed(future_to_id):
            pid = future_to_id[future]
            queried_count += 1

            try:
                updates = future.result()
                if updates:
                    batch_updates.extend(updates)
                    resolved_count += len(updates)
            except Exception as e:
                pass  # skip errors silently

            # Track progress
            progress_fd.write(f"{pid}\n")

            # Flush to DB every 500 queries
            if queried_count % 500 == 0:
                if batch_updates:
                    conn.executemany(
                        "UPDATE markets SET won = ? WHERE token_id = ? AND won IS NULL",
                        batch_updates,
                    )
                    conn.commit()
                    batch_updates = []
                progress_fd.flush()

                elapsed = time.time() - t0
                rate = queried_count / elapsed
                eta = (total - queried_count) / rate if rate > 0 else 0
                print(
                    f"  [{queried_count}/{total}] "
                    f"resolved={resolved_count} "
                    f"rate={rate:.1f}/s "
                    f"ETA={eta/60:.0f}min",
                    flush=True,
                )

    # Final flush
    if batch_updates:
        conn.executemany(
            "UPDATE markets SET won = ? WHERE token_id = ? AND won IS NULL",
            batch_updates,
        )
        conn.commit()
    progress_fd.close()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} minutes", flush=True)
    print(f"Queried: {queried_count}, Resolved: {resolved_count} tokens", flush=True)

    # Stats
    won1 = conn.execute("SELECT COUNT(*) FROM markets WHERE won = 1").fetchone()[0]
    won0 = conn.execute("SELECT COUNT(*) FROM markets WHERE won = 0").fetchone()[0]
    null_count = conn.execute("SELECT COUNT(*) FROM markets WHERE won IS NULL").fetchone()[0]
    print(f"Final: won=1: {won1}, won=0: {won0}, unresolved: {null_count}", flush=True)

    conn.close()


if __name__ == "__main__":
    main()
