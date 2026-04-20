# Cesta A — Bid/Ask + Volume Download Pipeline Upgrade

> **Parallel task** to ML exit-timing (Cesta B). Extends current Pass 2
> download to also capture bid/ask + volume_1m, unblocking orderbook
> microstructure features for future model iterations.
> **Status:** PLAN — ready to execute when Cesta B lands.
> **Date:** 2026-04-20

---

## 1. Current State (LEARNINGS.md D3)

PolymarketData.co data pipeline on arbo-download VPS:

| Field | Schema | Current state | Pipeline |
|---|---|---|---|
| `price` | ✓ column | **populated** | `insert_prices_simple()` (ts, price) — used by Pass 2 |
| `bid` | ✓ column | NULL 100% | `insert_prices()` supports — not called |
| `ask` | ✓ column | NULL 100% | `insert_prices()` supports — not called |
| `volume_1m` | ✓ column | NULL 100% | `insert_prices()` supports — not called |

**Root cause:** The Pass 2 worker (`pass2_worker_sharded.py`) calls
`download_market_prices()` which queries `/markets/{id}/prices` endpoint
(returns `{ts, price}` tuples) and writes via `insert_prices_simple()`.

**The fix:** Use a second endpoint `/markets/{id}/books` that returns
orderbook snapshots with bid/ask + volume. Already defined in
`download_polymarketdata.py` line 229 (`PMDataClient.get_books()`) but
never called.

---

## 2. Proposed Changes

### 2.1 New function `download_market_books()`

In `research_d/download_polymarketdata.py`, mirror `download_market_prices()`:

```python
def download_market_books(
    client: PMDataClient,
    market: dict,
    db: SportsDB,
    resolution: str = "1m",
    start_date: str | None = None,
    max_history_days: int = 9999,
) -> int:
    """Download orderbook history for a market. Writes (ts, price, bid, ask, volume_1m).

    Returns total points written.
    """
    # ... pagination loop like download_market_prices ...
    # Call client.get_books(...) instead of .get_prices(...)
    # Parse response and extract bid/ask/volume per token_id per ts
    # Use db.insert_prices(token_id, [(ts, price, bid, ask, vol), ...])
```

**Key difference:** response parsing — `/books` returns per-timestamp
orderbook levels. Need to compute best bid/ask and capture `volume_1m`
from the `trades` aggregation (if endpoint provides it; else need
additional `/trades` endpoint).

### 2.2 New worker: `pass2b_books_worker.py`

Copy `pass2_worker_sharded.py`, swap:
```python
# OLD:
n = download_market_prices(client, market, db, resolution='1m', ...)
# NEW:
n = download_market_books(client, market, db, resolution='1m', ...)
```

Separate progress files: `pmd_pass2b_progress_books_{worker_id}.txt` so
books downloads don't collide with existing price progress tracking.

### 2.3 Upsert vs insert semantics

Existing `insert_prices_simple()` uses `INSERT OR IGNORE`. A books pass
after a prices pass will hit conflicts on `(token_id, ts)` PK and the
new bid/ask will NOT be written.

**Options:**
- **A) UPDATE path**: run a post-process step
  `UPDATE prices SET bid=?, ask=?, volume_1m=? WHERE token_id=? AND ts=?`
  matched by `(token_id, ts)`. Slow but safe.
- **B) UPSERT via INSERT … ON CONFLICT**: add
  `INSERT … ON CONFLICT(token_id, ts) DO UPDATE SET bid=excluded.bid, …`
  to a new `upsert_prices()` method in `sports_db.py`.

Option **B** is cleaner. Add `upsert_prices(token_id, prices_full)` that
uses the 6-column signature and `ON CONFLICT DO UPDATE`.

### 2.4 Validation step

After download, run a SQL audit:
```sql
-- Coverage of bid/ask on NBA moneyline tokens
SELECT
    COUNT(*) AS total,
    SUM(bid IS NOT NULL) AS with_bid,
    SUM(volume_1m IS NOT NULL) AS with_vol
FROM prices p
JOIN markets m ON p.token_id = m.token_id
JOIN games g ON m.game_id = g.game_id
WHERE g.sport = 'nba';
```

**Acceptance criterion:** ≥ 80% bid/ask coverage on NBA moneylines.

---

## 3. Scope & Cost

### 3.1 API quota

- PMD per-tier rate limits (Ultra tier, active): ~2000 RPM shared across
  workers. Earlier Pass 2 ran with 3 workers × 450 RPM = 1350 RPM.
- NBA moneyline markets: 10,927 per `pmd_pass2_progress_moneyline*.txt`
- Per-market API calls: ~5-20 pages (depends on trajectory length)
- Estimated total requests: ~50K-200K
- At 450 RPM: ~2-7 hours wall-clock

### 3.2 Storage

- Current DB: 291 GB (price-only)
- With bid/ask + volume_1m populated: ~390 GB estimated (+33%)
- Disk free on `/mnt/arbo-data`: 297 GB (per `download_status.json`)
- **Not enough headroom.** Need either:
  - Add block storage volume to arbo-download (AWS Lightsail)
  - Or archive older Pass 1 (10-min) data to compressed format

### 3.3 Implementation effort

| Task | Effort |
|---|---|
| Inspect `/books` response format (probe script) | 1h |
| Write `download_market_books` + unit tests | 2h |
| Add `upsert_prices` to `sports_db.py` | 1h |
| Write `pass2b_books_worker.py` | 1h |
| Run download (NBA moneylines) | 2-7h |
| Validate coverage | 30min |
| **Total** | **7-12h (mostly unattended)** |

---

## 4. Integration with Cesta B

Once bid/ask data lands, extend `exit_timing_features.py`:

```python
# New features (insertion near existing vol_* features):
"spread_bps",             # (ask - bid) / mid * 10000
"depth_imbalance",        # (ask_qty - bid_qty) / (ask_qty + bid_qty)  [if available]
"volume_1m_now",          # trade volume in last 1 min
"volume_1h_cumulative",   # trade volume in last hour
"price_impact_kyle",      # |Δprice| / sqrt(volume) — Kyle's lambda approximation
```

Re-train existing XGBoost model with expanded feature set. Expected
C-index lift: +0.03-0.07 (based on Ng et al. 2026 findings on
order-imbalance predictive power).

---

## 5. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| `/books` endpoint rate-limit different from `/prices` | Test first, adjust RPM budget |
| Response format differs from expected | Probe script before committing full run |
| Disk full mid-download | Monitor free space; pause at 80% full |
| PMD API returns partial data for older markets | Re-run failures; log gaps |
| UPSERT deadlocks in SQLite with WAL journal | Serialize writes; single-writer worker |
| Download interrupted → partial per-token data | Resume via progress files (same pattern as Pass 2) |

---

## 6. Go/No-Go Gate (before executing)

Run BEFORE committing to the full download:

1. **Storage check**: `df -h /mnt/arbo-data` → ≥ 100 GB free
2. **API probe**: call `get_books()` on 5 sample NBA tokens, log response
   schema. If schema isn't {bid, ask, volume}, abort.
3. **Cost estimate**: run for 10 markets, measure API calls + rows →
   extrapolate to 10,927 markets, decide if quota OK.
4. **Write test**: verify `upsert_prices()` correctly updates existing
   rows (not just inserts).

Only if all 4 pass → execute full download.

---

## 7. Alternative: Skip until ML v2 needs it

**Argument for delay:** Cesta B v1 already achieved C-index 0.67 with
price-only features. We don't strictly need bid/ask yet. If Cesta B v2
(fixed policy formulation) matches or beats baseline, we might ship
without Cesta A.

**Trigger to reconsider:** If Cesta B iterations plateau at C-index < 0.7
OR if policy PnL stays within 20% of baseline — then invest in Cesta A.

Otherwise defer.

---

## 8. Next Step

**Execute only after Cesta B v2 conclusion.** If v2 shows room for
improvement from microstructure, execute §6 go/no-go probes, then §2
implementation. If v2 is already production-ready, shelve Cesta A.
