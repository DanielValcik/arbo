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

### G12. systemd service user == command user → no sudo available

When a systemd unit runs `User=arbo`, the process IS arbo. Any call to
`sudo -u arbo <cmd>` fails because `arbo` isn't in sudoers — it can
only call the command directly. Same for `sudo journalctl` — you need
to either be in `systemd-journal` / `adm` group, or skip sudo.

**Discovered 2026-04-18 00:10 UTC** when the first scheduled daily
retrospective auto-generated EMPTY metrics tables. The retrospective
script detected `/opt/arbo/.env` (VPS mode) and prepended `sudo -u arbo`
to every psql/journalctl call — which silently failed because arbo
can't sudo. Fix: detect current user via `$USER`/`whoami`, drop the
sudo wrapper when already running as arbo, and add arbo to the
`systemd-journal` group for clean journal access.

**Lesson:** any automation script that shells out to privileged
commands must key off the *current* user, not a "VPS mode" boolean.
`sudo -u $TARGET` is an identity-transition, not an access-grant — it
only works from a user who can transition. Prefer `os.environ["USER"]`
+ conditional sudo over assuming a fixed context. (Fix in
`scripts/retrospective.py` commit `c6bffb5`.)

---

### G11. Gamma `/markets?condition_ids=X` returns empty for CLOSED markets
The Gamma API endpoint by default only lists **active/open** markets.
Once a market closes and resolves, `GET /markets?condition_ids=<cid>`
returns `[]` — not the resolved outcome. To fetch resolution you MUST
pass `closed=true` alongside the filter:

```
# WRONG (returns []):
GET /markets?condition_ids=0xABC...
# RIGHT:
GET /markets?condition_ids=0xABC...&closed=true
```

**Impact discovered 2026-04-17 06:30 UTC:** All shadow variant resolution
sweeps (B2, D) had been silently failing since Project PARALLEL shipped.
576,248 B2 shadow signals qualified as expired → 0 resolved → bandit
allocator had no data to pick champion → weeks of evaluation wasted.

**Fix:** `strategy_b2.py:1014` and `strategy_d_core.py:831` — add
`"closed": "true"` to the params dict.

Lesson: whenever a resolver returns None silently (no errors, no logs)
but work queue keeps growing, the API is filtering you out — add
explicit metric (`skipped_fetch_fail` / `skipped_not_closed`) BEFORE the
silent return. The D version already had these counters; B2 did not, so
the bug hid longer on B2.

---

### G14. User decision fatigue — the framework is only as good as the operator's ability to act on it

**Observed 2026-04-19 after canary promotion shipped:** the operator
received two simultaneous Slack promotion candidates (ch_short_fuse +
ch_edge_tight) with a dead "Auto-approve in 24h" line, plus a drift
alert, plus an auto-challenger notice. Each message was individually
correct but the SET was confusing. First question was "what happens if
I approve both?" — exactly the question the system should have
prevented by design.

**Root cause:** The promotion engine emitted every Tier-1 candidate
above threshold. It didn't know that only one canary can run at a
time (that guard lives downstream in `pool_manager`). It also didn't
rank or pick — it just dumped candidates. And the Slack formatter
carried over an "Auto-approve" line from a feature that was planned
but never implemented. Each piece was individually sane; composed,
they pushed decisions back onto the user.

**Fix (commit pending):**

1. **Watchdog `_promotion_cycle`** now guards on "incubate already
   active" — if one canary is running, no new candidates emit at all.
2. **Top-1 only** — candidates sorted by `p_better` desc, only the
   best is posted per cycle.
3. **Auto-approve threshold** — candidates with P(better)≥0.85 AND
   N≥1000 AND DSR Δ≥0.20 are auto-promoted to incubate by the
   watchdog. Operator sees a notification, not a question.
4. **Slack text rewritten** — removed the dead "Auto-approve in 24h"
   line, replaced with what the Approve button will actually do.
5. **Daily digest (`scripts/parallel_digest.py`)** — Gemini writes ONE
   plain-Czech morning briefing per strategy at 07:00 UTC that
   aggregates champion status, incubate progress, drift, and pending
   decisions. Replaces the stream of ad-hoc alerts.

**Lesson:** every notification you send the operator is an implicit
"figure this out" request. Before a system ships to a non-engineer
user, walk through a week of their Slack and check: would they know
exactly what to do, without asking? If they have to interpret, the
framework is leaking complexity. **Prefer silence + autonomous action
over a prompt**; prefer a daily digest over ad-hoc alerts; prefer a
single best candidate over a list.

Related: `docs/KNOWLEDGE_BASE.md` — a living user-facing doc that
describes exactly what the system does autonomously vs what it asks
about. Kept current by CLAUDE.md rule — if a change isn't in the KB,
the change doesn't ship.

---

### G13. Shadow stats can't justify live promotion alone — need canary stage

**Problem:** Project PARALLEL's promotion engine flagged candidates
based on `shadow_variant_signals` — a DB of theoretical PnL as if each
variant's gate had admitted each candidate. The shadow paper stats
systematically overstate edge:

- Shadow fills at midpoint or inferred price, live pays real ask/bid
  (spread cost — see G1)
- Shadow has no fill latency, slippage, or partial fills
- Shadow price series can be stale by seconds; live orderbook moves

Consequence: a variant that beats champion on shadow stats can lose
money live. Promotion directly from shadow → champion risks flipping
the production strategy onto a broken thesis that only looked good in
simulation.

**Fix (2026-04-19):** canary promotion flow — Slack "Approve" now
routes the challenger into an `incubate` stage that receives a bounded
fraction (default 20%) of live signals for real fills. Watchdog
collects ≥15 paired live trades and runs block-bootstrap against the
champion's live PnL. Only escalates to champion when P(live better) >
0.70. Reverts to challenger-shadow when P < 0.30 after ≥20 trades.

Spec: `docs/CANARY_PROMOTION_SPEC.md`. Implementation:
`pool_manager.promote_to_incubate`, `strategy_b2._get_live_routing` +
`_route_signal`, `b2_watchdog._incubate_cycle`.

**Lesson:** any multi-variant optimization framework needs a live-data
validation stage between shadow and full production. Shadow is a
filter, not a decision. The stage can be brief (days) when trade
volume is high, or slow (weeks) when it's low — but it has to exist,
or the framework is ceremony without safety.

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

#### B2-15. `jsonb_build_object` variadic params broke asyncpg type inference
**What:** After the `::numeric` → `CAST()` fix landed (B2-11), the first
new live exits that tried to write `live_exit_*` fields into
trade_details still failed with
`IndeterminateDatatypeError: could not determine data type of parameter $2`.
Seven B2 LIVE SELL messages fired in Slack between 19:55 and 19:58
but trade_details never got updated → dashboard Live counter stayed
at 1 instead of 7.

**Why:** `jsonb_build_object(k1, v1, k2, v2, k3, v3)` is variadic.
When the call is `..., CAST(:px AS numeric), :status, CAST(:pnl AS numeric)`,
asyncpg can't resolve the middle arg's type because it's a raw string
literal sandwiched between two numerics — Postgres function-overload
resolution fails before execution.

**Fix:** Cast **every** parameter explicitly, including the obvious
`:status` (as `text`) and the WHERE-clause `:tid` (as `varchar`).
Commit `2f43d71`. Backfilled affected rows (5142-5159) with manual
UPDATE copying actual_pnl → live_pnl and exit_price → live_exit_price.

**Lesson:** asyncpg + jsonb_build_object + mixed-type variadic params →
cast absolutely everything. Don't rely on asyncpg's inference when the
function is variadic; it gives up rather than falling back to ambiguous
overloads. Apply this rule to any future jsonb helpers in raw SQL.

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

### B2-16. Entry vs exit `sigma_scale` mismatch → instant edge_lost cascade

**Observed 2026-04-17 06:30–06:40 UTC (Slack alert run):**
Every new BUY exited within 3-5 minutes with `edge_lost`. Pattern from
live feed:

- 06:33 BUY BTC above $78k @ 0.12 (edge +32¢) → 06:36 SELL @ 0.10 (−$0.55)
- 06:33 BUY ETH above $2500 @ 0.12 (edge +29¢) → 06:36 SELL @ 0.10 (−$0.60)
- 06:33 BUY ETH above $2400 @ 0.15 (edge +26¢) → 06:36 SELL @ 0.14 (−$0.37)

14 live trades → -$4.93 cumulative, all edge_lost, avg hold ~4 min.

**Why:** Entry and exit computed probability with **different `sigma_scale`
values** against the same strike/spot/hours data. `estimate_crypto_prob`
takes `sigma_scale` as kwarg (default `1.0`). Exit (`strategy_b2.check_exits`)
passed `sigma_scale=SIGMA_SCALE` (= `0.8` from `crypto_quality_gate.py:45`).
Entry (`crypto_price_scanner.scan_crypto_markets`) did not pass it at all,
so it used the default `1.0`.

Consequence: entry saw wider vol tails → prob inflated (e.g., OTM strike
looked ~44% likely). 30 seconds later, exit recomputed with `sigma_scale=0.8`
→ narrower tails → prob collapsed to ~12% → `updated_edge = 0.12 - 0.10 =
0.02 < MIN_HOLD_EDGE (0.03)` → immediate exit. Every buy was guaranteed
to fire edge_lost before the market had time to move.

**Fix (commit pending):** thread `sigma_scale` kwarg through
`scan_crypto_markets` and call it with `sigma_scale=SIGMA_SCALE` in
`strategy_b2.py:247`. Entry and exit now use identical calibration.

**Lesson:** **Entry and exit of the same position MUST share identical
model calibration.** Any kwarg (sigma_scale, prob_sharpening, shrinkage)
that biases one path vs the other produces artificial "drift" that looks
like market behavior but is purely arithmetic. Audit every other strategy
(C2, B3, B3_15M, D) for the same pattern: compare signal-generator call
vs exit-recompute call side by side. If they differ on any kwarg, fix
before shipping any more live trades.

This bug was HIDDEN BY paper mode because paper used whichever path
populated model_prob first and just stored it — no recomputation
divergence. Only live mirror mode surfaced it via the rapid BUY→SELL
loop.

---

### B2-17. Fresh sigma in `check_exits` whipsaws → false edge_lost on flat markets

**Observed 2026-04-17 07:00–07:04 UTC (AFTER B2-16 sigma_scale fix):**
Even with matching `sigma_scale=0.8` on both paths, edge_lost kept firing
within 3 minutes. Smoking gun: trade 5304 bought ETH $2400 at `0.17`
and exited 3 min later at **identical `0.17`** with `edge_lost`. Price
was flat — yet the edge model decided the position lost edge.

**Why:** `check_exits` called `vol_estimator.get_sigma(pos.symbol,
time.time())` which returns the CURRENT sigma from a rolling 24-obs
window. Each poll cycle adds one new Binance-WS price observation; the
window slides and any large return dropping out the back end produces a
step change in sigma.

With only 24 observations of 1-minute samples, a single $200 BTC move
entering/leaving the window can shift sigma by ±20%. For OTM strikes
(prob 0.20–0.30), a 20% sigma drop can erase 3–5 points of probability
in one poll. Combined with a `MIN_HOLD_EDGE` hysteresis of only 5 points
(entry 0.08, exit 0.03), **the strategy was guaranteed to exit on
sigma-driven noise alone**, independent of any real market movement.

**Fix (commit pending):** freeze `sigma_at_entry` in the `OpenPosition`
dataclass (already stored — just use it) and read it back in
`check_exits` instead of re-querying `vol_estimator`. Exchange price and
hours_to_expiry continue to update (those are real signals); only sigma
is frozen.

**Lesson (generalised):**
- A **rolling-window realised-volatility estimator is too noisy** to
  drive exit decisions on short timeframes. The window length
  is set for long-horizon calibration, not intra-position recomputation.
- When an entry gate is paired with an exit gate on the same quantity,
  the exit computation should reuse the entry-time snapshot of any
  high-frequency noisy inputs. Only inputs that change with real
  meaning (price, time) should update.
- Corollary: audit **any** model that recomputes probability at both
  signal time and exit time. If the model has hidden state that
  changes (sigma, bias, shrinkage estimate), ensure the position
  caches and reuses the entry snapshot.

---

### B2-19. Sigma fix validated: first real B2 post-fix wins

**Observed 2026-04-17 13:00–17:00 UTC (6-10h after B2-16 + B2-17 fixes):**
Two B2 trades that entered shortly after the sigma-freeze fix hit
resolution with real profit — the first live B2 wins since the
dual-mode launch.

- **Trade 5310** ETH above $2400. Entry 0.21 @ 07:11, exit 0.45 @ 13:10
  via `edge_lost` (edge flipped below MIN_HOLD_EDGE as the event got
  close to expiry) → **+$2.35 on $2.31 invested** (≈100% ROI).
- **Trade 5309** BTC above $78k. Entry 0.14 @ 07:11, exit 0.45 @ 14:38
  via `profit_take` (price climbed +$2.7k on Binance) →
  **+$6.41 on $3.22 invested** (≈200% ROI).

Cumulative post-fix stats after N=4 closes: **+$8.57 net**,
2W/2L, avg hold 202 min (vs ~3 min pre-fix). Both wins explicitly
required the fixes — with the pre-fix calibration, each trade would
have fired `edge_lost` within 3 minutes of entry for small losses on
stale/noisy sigma.

**Lesson:** the sigma-frozen exit gate is the *only* thing that gives
a slow-moving thesis time to play out. B2's edge model expects daily
markets, not minute-level whipsaw. Hold the calibration, let the
market come to the position.

Archive note: the N=14 pre-fix -$4.93 result in the session transcript
does NOT represent B2's true performance — it measures how the system
behaves under a broken exit gate. Future retrospectives should discount
anything before commit `af7bb2e` when comparing calibration choices.

---

### B2-18. `event_end_ts` parses START not END of market date range

**Observed 2026-04-17 10:50 UTC:**
Shadow resolver stuck at 58.5% resolved. Diagnostic: `SELECT` on stuck
condition_ids showed `event_end_ts = 2026-04-13 00:00:00 UTC` — 4 days
past. But Gamma returned empty. CLOB showed the market is still open
with `end_date_iso = 2026-04-20T00:00:00Z`.

**Why:** `market_discovery.categorize_crypto_market()` parses the FIRST
date matched by the regex (`April 13` in "Bitcoin reach $80k April
13-19") and creates `expiry = datetime(2026, 4, 13)` — midnight of the
START of the weekly window. Actual market end is a week later.

For daily markets ("Bitcoin above $78k on April 17"), the same bug is
smaller: expiry stored as `2026-04-17 00:00:00` but market actually
ends `2026-04-17 23:59:59` — off by 24h, which the `_sweep_shadow`
throttle > 300s tolerates but causes the resolver to query ~24h before
Gamma has posted the resolution.

**Fix deferred:** the parsing needs a second-date regex for ranges
(`April 13-19` → match group 2 for end day) and for daily markets
should add 23h59m or parse from Gamma's `end_date_iso` directly. Touched
by anyone hardening the shadow pipeline next.

**Impact:** non-breaking. Shadow N grows slower than it could (~58% of
expired signals resolved vs ~95% expected). Still enough data (N>2000
per variant) for statistical analysis. Markets eventually resolve when
their TRUE end passes.

**Lesson:** when a parser returns a single date from a range
expression, validate against the canonical source (`end_date_iso` from
Gamma `GET /markets/<cid>`) before using it as an end-timestamp. The
ambiguity is silent and the consequences (stale shadow data, missing
backtest resolutions) masquerade as "resolver slow."

---

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

### 🔴 DO NOT backtest live hypotheses on pre-mirror paper data

**Observed 2026-04-17 ~07:40 UTC:**

Live N=10 suggested a hypothesis: tight filter (spread ≤10%, price ≥$0.15)
would have eliminated 6/8 losers while keeping 2/2 winners. Net live
projection: -$4.30 → +$0.32.

Tried to validate on pre_reset data (N=838 archived paper trades):

| Bucket | N | WR | Avg PnL | Total |
|-|-|-|-|-|
| HIGH SPREAD (>10%) | 327 | 92% | +\$7.36 | +\$2,406 |
| TIGHT FILTER (hypothesis) | 409 | 77% | +\$2.34 | +\$956 |
| LOW PRICE (<0.15) | 102 | 64% | +\$0.34 | +\$35 |

Pre_reset data showed the **opposite** conclusion — high-spread trades
were paper's BEST profit category, not worst. Total paper PnL on
high-spread bucket: +\$2,406 vs +\$956 for tight-filter bucket.

**Why the contradiction:** paper was earning the spread (B2-1). Wider
spread = more "profit" to pocket at the paper level. The 92% WR and
+\$7.36/trade on high-spread trades is the ceiling of spread arbitrage,
not real edge. Pre_reset data is not just noisy for live hypotheses —
it's systematically **inverted**. Validating a "tight spread helps"
hypothesis on data that rewards wide spreads will always reject it.

**Lesson (generalizable):** Any retroactive validation of a live-edge
hypothesis against paper data requires the paper to already have
correct paper-live parity (PAPER_MATCH_LIVE=True with proper bid/ask
semantics). Pre-mirror paper data is strictly worse than useless for
this purpose — it can actively mislead. Only post-mirror live data
counts.

**Implication for B2:** Cannot fast-validate the tight-filter hypothesis.
Must collect N=30+ under current live semantics (~10 days at 3
trades/day). Alternative: activate B2 shadow variants (Project PARALLEL
framework) so parallel parameter configurations generate paper-valid
data. Currently 0 shadow rows for B2 — TBD whether it's disabled or
not wired.

### 🟡 Post-fix clob_edge_low dominates signal rejection

**Observed 2026-04-16 21:00-21:25 UTC:**

30 min of B2 poll cycles with 3-8 qualified signals each, **0 executed**
across the window. All skipped with reason `clob_edge_low`.

The quality gate flow:
1. Scanner computes edge from Gamma API price (stale, may lag real book)
2. Quality gate filter passes signals where `|model_prob - gamma_price| ≥ MIN_EDGE (0.08)`
3. Before sizing, we fetch CLOB orderbook and compute `clob_edge = model_prob - clob_price` (where `clob_price = max(bid, ask)` post-mirror fix)
4. If `clob_edge < MIN_EDGE`, skip with `clob_edge_low`

Post-PAPER_MATCH_LIVE fix (B2-1), `clob_price = max(bid, ask)` = HIGH
side (what live taker actually pays). This shrinks the measured edge
vs. the pre-fix `min(bid, ask)` path by the spread amount (~2-4¢ per
share). Signals that showed 10¢ edge on Gamma might show only 6-8¢
edge against the real CLOB ask, failing the 8¢ threshold.

**Positive reading:** Quality gate is now doing its job — blocking
structurally-negative-EV trades where spread eats the signal. Without
this filter, B2 would enter, get whipsawed by 3-5 min edge_lost exits,
and lose the spread on every round-trip (see B2-14 / edge pattern entry
above).

**Negative reading:** Confirms the "paper edge was spread arbitrage"
hypothesis. If real edge (post-spread) is consistently < 8¢, B2 rarely
trades at all in this regime. Low sample rate → slow statistical
validation.

**Decision pending:** Is it worth lowering MIN_EDGE to 0.05 to collect
more live samples faster, accepting that some will be spread-dominated
losses? Or hold 0.08 and just wait? Waiting preserves capital but
drags out the validation timeline. This is an optimization question
that should go through the `/optimize` framework, not a one-off tweak.

**Don't tweak this parameter without autoresearch validation against
post-fix data.** Paper-era optimization was solving the wrong problem.

### 🔴 Edge pattern: paper 80% WR → live 20% WR on longshot whipsaws

**Observed 2026-04-16 20:30 UTC, N=10 real live closes:**

After all spread/parity/reconciliation fixes shipped, the first clean
sample of live B2 data shows dramatic divergence from paper:

| | Paper (pre-mirror) | Live (post-fix) |
|-|-|-|
| WR | 80% | 20% (2W/8L) |
| Avg PnL/trade | +\$6.72 | -\$0.43 |
| Avg hold | 2.4h | 3-65 min |

**Root pattern identified:**
1. Model finds edge on low-price tokens (\$0.06-\$0.13, ~6-13% implied
   prob = "longshots").
2. Live enters — pays the ask, taker fee, slippage.
3. 3-5 min later the exchange price moves 0.5-2% against our direction.
4. Model recomputes with `SIGMA_SCALE=0.8` + current price → edge
   drops below `MIN_HOLD_EDGE=0.03`.
5. Exit triggers `edge_lost` → sells at the bid (low side). Loss = ~2-4¢
   per share crossing the spread plus the adverse move.
6. Every whipsaw locks in cost of execution with no time for mean
   reversion to recover.

**Why paper didn't show this:**
Pre-PAPER_MATCH_LIVE fix, paper `clob_price = min(bid, ask)` = low side
on entry AND high side implicitly on exit (different code path). Paper
entered at the favorable side and exited at the other favorable side —
it "earned the spread" on every trip. Multiplied across 324 trades,
this masqueraded as an 80% WR strategy. Real execution pays the spread
on both sides — ~4¢/share round-trip, enough to turn B2's ~10¢ signal
edge negative on ~30-share positions at \$0.10 entry prices.

**This is not a bug** — it's the hypothesis we were testing. We deployed
dual mode specifically to measure whether paper's edge was real or
spread-arbitrage. The answer, with N=10, is leaning strongly toward
"artifact." Before drawing a final conclusion:
- N=10 is small; need ≥ 30 real exits for statistical significance
- Recent 3 trades (-\$3.15 of -\$4.30 total) happened in a single
  volatility burst — could be regime, not strategy
- Need at least one resolution (market expiry → \$1) to see if
  "hold-to-resolution" changes the picture (paper baseline has few
  resolutions; most exits were edge_lost like live)

**Next steps if pattern persists:**
1. Tighten entry: `MIN_PRICE ≥ 0.15` (avoid longshots where spread
   dominates signal) — routine params change, goes through `/optimize`.
2. Loosen exit: `MIN_HOLD_EDGE ≥ 0.08` or add min-hold-time 30 min
   to avoid whipsaw.
3. Re-autoresearch on **post-fix live data** (not pre-fix paper) —
   paper optimization was solving the wrong problem (spread earning,
   not edge capture).

**Lesson (generalizable):** Any strategy with high paper WR and short
avg hold is suspect for spread artifact. Validate by computing
"spread-free" backtest PnL = backtest PnL − 2 × avg_spread × avg_shares
× N_trades. If this number is ≤ 0, the strategy has no real edge;
paper is counting the spread as profit.

### Low-frequency, high-quality entries after mirror fix

**Observed 2026-04-17 early hours:**

60+ min window with 0 executed trades, then a single entry at 21:34:
BTC above $74000 @ $0.59, edge +35¢. Exchange price $74,951 (already
above strike). This is qualitatively different from the pre-fix
losers:

| | Losers (B2-1 to 8) | This entry |
|-|-|-|
| Entry price | $0.06–$0.13 | $0.59 |
| Signal edge | 8–15¢ | 35¢ |
| Hours to expiry | 30–130h | 146h |
| Market state vs strike | Well below | Already above |

Pattern: post-quality-gate, B2 only fires when the signal has real
margin over the spread — often on high-conviction positions where
the market is already on the favorable side of the strike.

**Implication for validation:** expect fewer but higher-quality
trades than the inflated paper baseline predicted. Target sample
rate likely 1-5 trades per hour (vs paper's ~2/hour during similar
windows). Needs ~48h of data for N=30.

**Memory health:** bounded collections are working. Post-restart
memory grew ~155 MB over 3h = 50 MB/h, down from the pre-bounded
100 MB/h rate. Still trends up but within the 2 GB/day ceiling
we budgeted for.

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

Status: **STOPPED as of 2026-04-18 08:30 UTC** (entry-side).
`B3_EXECUTION_MODE=stopped` in `/opt/arbo/.env`; check_exits continues
so existing positions can resolve naturally.

### B3-1. Entry stopped after 3-day drawdown

**Observed 2026-04-15 → 2026-04-18 via daily retrospective:**

| Day | Trades | WR | Net PnL |
|-----|--------|-----|---------|
| Apr 12 | 202 | 74.8% | +\$432 |
| Apr 13 | 261 | 70.1% | +\$1 |
| Apr 14 | 135 | 68.1% | +\$348 |
| Apr 15 | 27  | 33.3% | -\$27 |
| Apr 16 | 152 | 63.2% | -\$300 |
| Apr 17 | 190 | 62.6% | -\$286 |
| Apr 18 (early) | 1 | 0% | -\$10 |

Win rate didn't collapse — held ~63% — but avg per-trade PnL flipped
from +\$2 to -\$1.50. **Asymmetric losses:** winners stayed small while
losers grew. Signature of a regime shift, not random noise.

**Action:** entry-side STOP until we can run a proper autoresearch
validation of current V6.0 parameters against post-Apr-14 data. Did
NOT alter parameters or shut down the strategy entirely — `check_exits`
still runs so the 100+ open B3 positions can resolve via Gamma API
instead of rotting.

**Mechanism:** `strategy_b3.poll_cycle` has a new early-return when
`_execution_mode == "stopped"`. The orchestrator still calls poll
every 10-15s but it returns `[]` immediately after the shadow sweep.

**Next step (user decision):** before re-enabling, either
(a) `/optimize B3` against last 2 weeks of price data to find new
    parameters that handle the Apr 15+ regime, or
(b) Compare Project PARALLEL B3 shadow variants and promote a better
    challenger.

Don't flip back to `paper` blindly — the -$595 over 3 days is the kind
of drawdown that can hide a real structural shift in BTC
microstructure (e.g., liquidity profile on 5-min Polymarket events
changing with broader market conditions).

---

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
