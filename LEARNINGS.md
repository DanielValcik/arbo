# LEARNINGS.md

> **Authoritative operational memory for Arbo.** Every bug, fix, architectural
> decision, and performance observation lives here. Consult this BEFORE
> making any strategy change, optimization, or debugging decision. Update
> after every discovery — entries MUST be added the same session they're
> learned.

## How to use this file

### When to consult
- **Before** any strategy parameter change (even "small" ones)
- **Before** autoresearch validation of existing strategy
- **Before** deploying new code that touches live execution
- **During** investigation of production anomalies — check if the pattern
  has been seen before
- **After** discovery of any surprise (whether good or bad)

### When to update
- **Immediately** after finding a bug (even before fixing it)
- **Immediately** after a fix is verified in production
- **Immediately** after a surprising performance observation (WR, fill rate,
  slippage, latency)
- **Immediately** after making an architectural decision with trade-offs
- Claude Code MUST append on every discovery, autonomously, without being
  asked

### Entry format
Each entry should be a self-contained learning. Include:
- **What happened** (concrete observation, with timestamps/numbers)
- **Why** (root cause, not just symptom)
- **Fix / decision** (what changed, which commit)
- **Lesson** (what to remember for next time, generalized)

### Keep it clean
- New discoveries go to the end of the relevant section
- When a learning is superseded by a better one, move the old one to
  "Archived" with a reference
- No duplicates — if it's already in global lessons, don't repeat per
  strategy

---

## Global lessons (system-wide)

### G1. Paper ≠ Live without explicit parity
Paper engines that compute PnL from `min(bid, ask)` on entry and `max(bid, ask)` on exit **earn the spread** — live taker pays the opposite sides. Without `PAPER_MATCH_LIVE=True` and correct high/low mapping, paper systematically inflates PnL by the spread (typically 2-4¢/share on Polymarket crypto markets).

**Rule:** every new dual-mode strategy MUST implement:
- `PAPER_MATCH_LIVE` flag (default True) in its quality gate module
- Entry uses `max(raw_bid, raw_ask)` (ignore inverted orderbook labels)
- Exit uses `min(raw_bid, raw_ask)`

Source: B2 dual-mode deploy (2026-04-16), first fix at 13:42 UTC.

### G2. Polymarket `/price` endpoint has inverted orderbook semantics
`orderbook_provider.get_snapshot()` returns:
- `best_bid = get_price("SELL")` → the LOW side (what seller receives)
- `best_ask = get_price("BUY")` → the HIGH side (what buyer pays)

These labels are **inverted** from standard orderbook convention. Never rely
on `.best_bid`/`.best_ask` names — always use `max()`/`min()` of both to
decide which side to hit.

### G3. `live_executor._shares_owned` is a cache, not truth
The dict is only populated via:
1. `_ensure_clob()` on first call (initial API sync)
2. Direct updates after `buy()` / wipe+rebuild after `sell()`

It's subject to:
- Polymarket data-api lag (new on-chain buys don't appear for seconds)
- Service restarts (new process starts empty)
- `sell()` calling `_sync_positions()` which **wipes** the dict

**Rule:** For any exit routing decision ("do we hold real shares?"), force a
fresh sync via `_sync_positions()` at the start of each exit check (throttle
to 20s to avoid stampede). Wallet API is the source of truth, never the cache.

Source: B2 exit routing bug (2026-04-16, trade 5139/5140).

### G4. Polymarket platform minimum is 5 shares per order
Hard constraint, not configurable. Orders with <5 shares get rejected with
"Size (N) lower than the minimum: 5". Live_executor must enforce this
client-side in both `buy()` and `sell()`.

For sizing: `min_size = max(wallet_floor, 5 * entry_price + $0.01)` — the
5-share floor scales with entry price; at price 0.80 you need ≥ $4.01, at
0.20 ≥ $1.01.

### G5. SQLAlchemy `:param` parser collides with Postgres `::type` cast
`sa.text("... :px::numeric ...")` fails with `PostgresSyntaxError: syntax error at or near ":"`. SQLAlchemy reads `:px::numeric` as `:px:` + bind `:numeric`.

**Rule:** always use `CAST(:px AS numeric)` in raw-SQL bound queries. Never
`::` for typed bind parameters.

Source: B2 live-exit upsert (2026-04-16).

### G6. Bounded collections, not unbounded lists
In-process caches (`_trades`, `_snapshots`, `_trade_details_cache`,
`orderbook._cache`, `adaptive_config._change_log`) grow unbounded by default
and cause 100 MB/h memory leaks over multi-day runs. DB is authoritative
for history; memory is a buffer.

Use `deque(maxlen=N)` and `OrderedDict` (LRU) with explicit eviction.
Document the bound and its rationale.

Source: paper_engine + orderbook_provider cleanup (2026-04-16).

### G7. Strategy's `_open_positions` dict ≠ `paper_engine._positions`
Strategies maintain their own in-memory position state for check_exits.
This state is NOT automatically restored from DB on service restart —
only `paper_engine._positions` is (via `load_state_from_db`).

If a strategy's exit logic iterates `self._open_positions`, any trades
placed pre-restart will NOT be checked. Add a strategy-level DB restore
in `init()` that hydrates position fields (asset, strike, direction,
live_shares) from `paper_trades.trade_details`, not just the generic
PaperPosition fields.

Source: B2 init hydration fix (2026-04-16).

### G8. Force-sync before exit routing, snapshot before check_exits
Two coordination rules that apply to any dual-mode strategy:
- **Force-sync**: Before deciding paper-vs-live routing, refresh
  `live_executor._shares_owned` (via `_sync_positions`). Prevents stale
  cache taking paper path for live-held positions.
- **Snapshot**: `check_exits()` pops triggered positions from
  `_open_positions`. Main orchestrator must snapshot the dict **before**
  calling check_exits, else downstream `get(token_id)` returns None and
  Slack/reconciliation lose metadata.

Source: B2 refactor (2026-04-16).

### G9. Dashboard/Slack counter consistency
If a counter is shown in both dashboard and Slack, they MUST compute from
the same source semantics. Specifically for "live closed":
- Require BOTH `live_entry_shares > 0` AND
  `live_exit_status IN (resolution, filled, partial, maker, taker, maker+taker)`
- `live_exit_status` is populated by the ExitManager reconciliation path.
  Paper-fallback closures (live was NOT actually sold) don't set it.
- This filter makes "Live PnL" reflect real on-chain outcomes, not
  paper-computed phantom closures.

### G10. Pre-reset archive pattern for clean slates
When a strategy's history is polluted (bugs, mis-reconciled data, broken
counters), mark rows with `notes LIKE '%pre_reset_<timestamp>%'` and
filter out from all user-facing queries (dashboard, restore, strategy
init). Preserves audit trail but resets counters cleanly.

Source: B2 reset on 2026-04-16 16:35 UTC.

---

## Strategy B2 — Crypto Price Edge

Status: **LIVE dual-mode since 2026-04-16 11:20 UTC**, $100 wallet capital.
First production day is a chaotic bug hunt; see entries below.

### Performance baseline

- **Paper (pre-2026-04-16, pre-mirror fix):** 324 trades / 7 days, 80% WR,
  avg +$6.72/trade, avg size $55, avg hold 2.4h. **Heavily inflated** by
  spread (see B2-1).
- **Live day 1 (2026-04-16, post-fix):** too few samples after pre-reset,
  N=1 real close @ -$0.35. Re-measure after 3-7 days.

### Bugs & Fixes (chronological, 2026-04-16)

#### B2-1. Paper entry used LOW side → spread inflation
**What:** Paper `clob_price = min(raw_bid, raw_ask)` = low side = paper
earned the spread that live pays. First 11 live trades showed +$0.02-0.04
slippage per share vs paper (~9.5% drag on entry alone).

**Why:** `PAPER_MATCH_LIVE` flag initially used `raw_ask` but orderbook_provider's
naming is inverted for Polymarket `/price` endpoint (see G2) — `raw_ask`
was the LOW side, not HIGH.

**Fix (13:42 UTC):** `clob_price = max(raw_bid, raw_ask)` when PAPER_MATCH_LIVE.
Commit `1bf552c`.

**Lesson:** Always verify bid/ask semantics empirically (first few live fills
should have 0¢ slippage). Never trust field names on inverted venues.

#### B2-2. Mirror-cancel orphans (17 stuck-open rows)
**What:** `_cancel_mirror_paper_trade` called UPDATE before main_rdh's
save_trade_to_db INSERT — UPDATE matched 0 rows, INSERT landed with
status=open → orphan forever.

**Fix:** Mark in-memory trade SOLD first, save directly from cancel path;
main_rdh save loop skips SOLD. Commit `e2e8388`.

#### B2-3. Cascade duplicates (4 rows/token in 16s)
**What:** After mirror-cancel, token not in `_open_positions`; next poll
cycle re-entered same signal. Observed: 4 B3 trades on same token in 16s.

**Fix:** `MIRROR_CANCEL_DEBOUNCE=120s` dict; block re-entry per token.
Commit `e2e8388`.

#### B2-4. Live executor <5 shares rejected (19× observed)
**What:** `int(size_usdc / maker_price)` produced 4 shares; Polymarket
rejected with "Size (4) lower than the minimum: 5".

**Fix:** Client-side 5-share guard in buy/sell + strategy sizing uses
`max(wallet_floor, 5 * entry_price + $0.01)`. Commit `e2e8388`.

**Lesson:** See G4. Sizing must account for both wallet % and platform
minimum; miss either and fills break in opposite directions.

#### B2-5. Maker-only = 0% fill rate on B2 markets
**What:** First 4 live buys, 0 filled after 30s maker timeout. B2 daily
crypto markets lack active counter-offers at our bid level.

**Fix:** Taker fallback after 30s maker timeout — pay `sell_price` (ask)
as GTC+immediate. Paper parity preserved since paper already uses `max()`.
Commit `5399e85`.

**Lesson:** Maker-first is a B3 pattern (5-min scalping, active books).
For thin daily markets, maker + taker fallback is mandatory. Cost:
~1-3¢/share spread, acceptable at signal edge >10¢.

#### B2-6. Sizing blocked below 5-share floor
**What:** Kelly-equivalent sizing at $100 wallet often came to $2.56
while shares_floor was $2.91 ($0.58 × 5 + $0.01). Half of signals
skipped.

**Fix:** When `live_size < shares_floor AND shares_floor < 5% wallet`,
boost to shares_floor. Missing signal over $0.05 shortfall is worse than
mild sizing overshoot (Quarter-Kelly still respected). Commit `83b4645`.

#### B2-7. B2 exit silently posted "B2 LIVE SELL" for paper-path closes
**What:** Slack said "B2 LIVE SELL … Live: 0.56 → 0.54 … Trade: -$0.13"
but live executor never sold anything — real shares still on CLOB.

**Why:** `was_live=True` in paper fallback path triggered the
`_notify_b2_live_resolve` even when ExitManager had not run.

**Fix:** Paper-fallback path no longer posts "LIVE SELL". Emits
`b2_paper_closed_live_held` warning log for later reconciliation.
"LIVE SELL" is now reserved for actual ExitManager completions. Commit
`55406d8`.

**Lesson:** A Slack message must mean what it says. Never name a channel
event after the intent when the action didn't actually happen.

#### B2-8. ExitManager completions didn't propagate to paper_engine/DB
**What:** First real B2 LIVE SELL at 16:02 (BTC above $78000, -$0.42)
succeeded on CLOB but paper_trades row stayed status=open, paper_engine
still held position, dashboard showed $0.

**Fix:** After `process_exits()` returns completed_exits, for each:
1. `paper_engine.sell_position(token, live_fill_price)` — closes paper
2. `risk_manager.strategy_post_trade(..., is_live_capital=True)` — updates pool
3. `update_resolved_trades_in_db` + `_upsert_b2_live_exit_details`
   — writes `live_exit_status`, `live_exit_price`, `live_pnl` into
   trade_details JSONB

Without this, Slack/DB/paper_engine/risk_manager all diverge. Commit `56ae6d0`.

**Lesson:** ExitManager is strategy-agnostic. The propagation chain
(on-chain → paper_engine → DB → dashboard → Slack) is the caller's
responsibility. Any strategy integrating ExitManager must close the loop.

#### B2-9. Paper/live pool separation (architectural)
**What:** `MAX_POSITIONS_PER_STRATEGY=10` was shared between paper and
live. 10 legacy paper positions blocked every new live attempt.

**Fix:** Split `StrategyState` fields:
- `position_count` / `deployed` → paper pool
  (`MAX_PAPER_POSITIONS_PER_STRATEGY=30`)
- `live_position_count` / `live_deployed` → live pool
  (`MAX_LIVE_POSITIONS_PER_STRATEGY=10`)
- `TradeRequest.is_live_capital` routes to correct pool.

Commit `fe74922`.

**Lesson:** Paper is data collection (cheap, lots of rows). Live is real
capital (scarce slots, tight limits). They must never compete for the
same budget.

#### B2-10. `_shares_owned` stale after restart → paper-path for live positions
**What:** Service restart 4s after 2 live BUYs. New process's initial
`_sync_positions` didn't see the fresh buys (Polymarket data-api lag).
Exit check saw `_shares_owned.get(token) = 0` → paper-fallback closed
positions that still had real shares on CLOB.

**Fix:** Force-sync on every exit check (throttled 20s), not just when
empty. Wallet API is source of truth. Commit `3d07ac5`.

**Lesson:** See G3. In-memory caches in live trading systems should be
assumed stale unless refreshed immediately before use.

#### B2-11. SQL cast collision in `_upsert_b2_live_exit_details`
**What:** `sa.text("... :px::numeric ...")` raises
`PostgresSyntaxError`. Commit `42ac15c`.

**Lesson:** See G5. Never use `::` cast with SQLAlchemy bind params.

#### B2-12. PnL restore query missed `taker`/`maker` statuses
**What:** After the `CAST()` SQL fix + manual patch, trade 5138 had
`live_exit_status='taker'` but the restore query filtered
`IN ('resolution','filled','partial')` only. Slack counter hydrated at
$0 instead of -$0.35.

**Fix:** Expand filter to `IN (resolution, filled, partial, maker, taker, maker+taker)`.
Five occurrences updated. Commit `231f8f9`.

**Lesson:** Whenever live_executor introduces a new fill type, update ALL
filters that distinguish "real live exit" from "not". Search for the
existing status set and update in sync.

#### B2-14. Task watcher killed B2 mid-poll during serial live attempts
**What:** B2 task stopped logging after 19:04:26 (one successful live
entry), health watcher reported `health_task_hung` at 19:04:28, then hit
`task_max_restarts` at 19:09:08 → task permanently stopped. B2 was dead
for ~40 min until user-triggered monitoring noticed.

**Why:** `OrchestratorConfig.heartbeat_timeout_s=120` was below B2's
legitimate worst-case poll runtime. With 20+ qualified signals per
poll and each serializing through a 30s maker + 5s taker fallback, a
cycle with 3 live attempts easily reaches 2 min. Heartbeat is only
updated between poll iterations (line 1858-1860 main_rdh), not during
the `coro_factory()` execution.

**Fix:** Bump `heartbeat_timeout_s` from 120 → 300 in settings.
Commit (pending — ships with this LEARNINGS update).

**Lesson:** The health watcher timeout must be > (max legitimate poll
runtime + safety margin). Any strategy that serializes remote I/O per
signal (B2 taker fallback, large market scans, slow external APIs)
bumps the floor. Alternative fix for future: plumb mid-cycle heartbeat
pings from strategies so long legitimate operations don't look like
hangs.

#### B2-13. Wallet API pagination default = 100
**What:** Initial reconciliation of "which tokens are held on-chain" used
`data-api.polymarket.com/positions?user=X` which defaults to limit=100.
With 20+ B2 positions + dozens of stale B3 rows, the filter missed real
held tokens → false "auto-redeemed" marking → later had to un-mark 13
rows.

**Fix:** Always pass `&limit=500` explicitly when querying wallet.

**Lesson:** Default pagination limits are quiet lies. Any reconciliation
query must verify with an explicit, generous limit.

### Architectural decisions

#### B2-A1. `B2_EXECUTION_MODE=dual`, $100 live capital
Dual mode is mandatory for B2 validation — pure-paper is inflated (see
B2-1 + G1), pure-live can't collect research data. $100 is small enough
to cap Day-1 risk given the number of unknowns but large enough to pass
the 5-share minimum at typical B2 prices (0.10-0.60).

#### B2-A2. Taker fallback default True for B2
Unlike B3 (maker-only works on liquid 5-min markets), B2 daily markets
lack maker counter-offers. Taker fallback at `sell_price` pays ~1-3¢/share
spread but achieves 100% fill rate. Critical for any validation data.

#### B2-A3. Live capital hydration from `get_balance()` every 60s
Real wallet balance drives sizing. Fallback to `B2_LIVE_STARTING_CAPITAL`
env only when API returns <$10 (transient failure). Never use a hardcoded
cap — it diverges from reality.

#### B2-A4. Per-B2 LIVE slot pool, separate from paper
(See B2-9.) `MAX_LIVE_POSITIONS_PER_STRATEGY=10`, independent of the 30
paper slot budget. Protects real-capital concentration without starving
paper data collection.

### Performance observations (cumulative)

Track per-week. Update entries here when material data comes in.

**Week of 2026-04-13** (pre-dual-mode, paper only):
- Paper: 324 trades, 80% WR, +$6.72/trade avg
- Inflated by spread — actual edge unknown until B2-1 fix verified

**Week of 2026-04-16** (first live day):
- Live entered: 6 (3 post-pre_reset cutoff)
- Real live closed: 1 (-$0.35)
- Paper-path closes: 2 (bug, not real sells)
- Wallet evolution: $100 → $26 (locked in positions) → force-close
  recovered $35 → ~$52 liquid
- **Takeaway:** too few samples for any edge conclusion; all statistics
  await clean post-fix dataset.

### Gotchas

- **Don't restart during active trading.** Pre-restart buys can orphan
  from new-process state (see B2-10, but even with force-sync there's a
  few-second vulnerability window from API lag).
- **Pre_reset rows stay in DB forever.** They're filtered from counters
  via `notes NOT ILIKE '%pre_reset%'` but populate trade history queries
  unless explicitly excluded. Always add the filter on new queries.
- **Polymarket shares <5 on CLOB are stranded.** Can't sell individually,
  must wait for market resolution. Avoid accumulating sub-5 positions.
- **Wallet balance ≠ buying power.** Some USDC is locked in open
  positions; effective buying power = `balance - sum(open position costs)`.
  `live_executor.get_balance()` returns liquid USDC only.

### Open questions / to investigate

1. **Paper vs live WR divergence:** paper 80% vs live (early) 7-14%. Is
   real edge much smaller than paper suggested, or is it early-sample
   noise? Need N ≥ 30 real closes.
2. **5-share stranded positions:** 5 B2 positions stuck with 4 shares
   (below Polymarket minimum). Option: top-up wallet so new buys can
   aggregate onto these tokens and break through the floor.
3. **Restart hardening:** can `live_executor` persist `_shares_owned` to
   disk/DB so new process starts with correct cache? Or use Polymarket's
   user-ws for real-time position sync instead of polling data-api?

---

## Strategy B3 — Binance Oracle Scalper (5-min)

Status: LIVE since pre-2026-04-16. See project memory for long history.

_(Populate as new discoveries happen. Today 2026-04-16 focused on B2; no
new B3 entries this session.)_

### Known pinned (imported from memory)

- V6.0 Dual Filter (velocity ≤ 60, dir_delta ≤ 15, see `b3_model_history.md`)
- Watchdog auto-tuning T1 params (see memory `watchdog_autonomy.md`)
- Paper vs live pricing inversion — same G1 issue, addressed by
  `PAPER_MATCH_LIVE` + mirror mode

---

## Strategy B3_15M

Status: LIVE since 2026-04-12. See `b3_15m_deployed.md`.

_(Add entries as discovered.)_

---

## Strategy D — NBA / UFC / EPL

Status: NBA live paper, UFC/EPL paper.

_(Add entries as discovered.)_

---

## Archived

_(Move superseded entries here with reference to the newer learning.)_
