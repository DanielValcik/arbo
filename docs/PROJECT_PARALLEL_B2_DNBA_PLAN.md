# Project PARALLEL extension to B2 + D_NBA

> **Status:** PLAN — awaiting CEO go-ahead.
> **Drafted:** 2026-04-13
> **Drafter:** Claude (autonomous extension after Phases 1-4 ship)
> **Scope:** apply same multi-variant infrastructure already running on B3 / B3_15M to two more strategies.

---

## 1. Why

B3 + B3_15M now have shadow-evaluated challengers, Gemini auto-challenger, PromotionEngine, drift detection, TS bandit. B2 (crypto price) and D_NBA (NBA green book) still rely on hard-coded constants in `crypto_quality_gate.py` and class attrs on `StrategyDNba`. Goal: one consistent optimization framework across all live strategies.

**Out of scope:** D_EPL and D_UFC stay discovery-only for now. Strategy C / C2 (weather) untouched (already have separate optimization track via EMOS).

---

## 2. Pre-existing infra to reuse (no changes required)

| Component | File | Reuse for B2 + D_NBA |
|---|---|---|
| Variant pool | `arbo/core/variant_pool.py` | New strategy dirs `arbo/config/variants/{b2,d}/` |
| Shadow table | `shadow_variant_signals` (PG) | Same schema; add rows with `strategy='B2'` / `'D'` |
| PerformanceAnalyzer | `arbo/core/performance_analyzer.py` | Add per-strategy `_compute_failure_modes` buckets |
| HypothesisGenerator | `arbo/core/hypothesis_generator.py` | TIER_1_PARAMS extended with B2/D-specific gates |
| PoolManager | `arbo/core/pool_manager.py` | Path-based, already strategy-agnostic |
| PromotionEngine | `arbo/core/promotion_engine.py` | Reads shadow_variant_signals by `strategy` arg |
| Slack workflow | `arbo/dashboard/slack_promotion.py` | Add B2/D channels to `STRATEGY_CHANNELS` |
| BO sweep | `research/bo_sweep.py` | Add `B2` and `D` to `SEARCH_SPACES` |
| Bandit allocator | `arbo/core/bandit_allocator.py` | Strategy-agnostic |
| Drift monitor | `arbo/core/drift_monitor.py` | Strategy-agnostic |
| Mid-trade sampler | `arbo/core/mid_sampler.py` | B2 yes; D_NBA needs evaluation (NBA prices move slow) |
| Watchdog wiring | `arbo/core/b3_watchdog.py` | Need new `B2Watchdog` and `DNBAWatchdog` (or generalize) |

---

## 3. Plan — Strategy B2 (crypto price daily/weekly)

### 3.1 Inventory of params to make variant-aware

From `arbo/strategies/crypto_quality_gate.py` — these are the live filters:

| Param | Current | Tier | Rationale to vary? |
|---|---|---|---|
| `MIN_EDGE` | 0.08 | 1 | Yes — tighter edge may improve hit rate at cost of volume |
| `MAX_EDGE` | 0.90 | 1 | Probably stable; cap for "too good to be true" |
| `MIN_PRICE` | 0.05 | 1 | Yes — exclude deep OTM longshots |
| `MAX_PRICE` | 0.60 | 1 | Yes — exclude deep ITM (low payoff/risk) |
| `MIN_HOLD_EDGE` | 0.03 | 1 | Yes — exit-trigger sensitivity |
| `MIN_TIME_TO_EXPIRY_H` | 8 | 1 | Yes — short-fuse trades may have edge decay |
| `MAX_TIME_TO_EXPIRY_H` | 168 | 1 | Yes — far-expiry markets less liquid |
| `KELLY_RAW_CAP` | 0.30 | 2 | NO (sizing — Tier 2, CEO approval) |
| `SIGMA_SCALE` | 0.8 | 2 | NO (model param — Tier 2) |
| `PROB_SHARPENING` | 1.50 | 2 | NO (model param — Tier 2) |
| `MAX_POSITION_PCT` | 0.08 | 2 | NO (sizing — Tier 2) |

**Tier 1 changes auto-challenger may propose:** MIN_EDGE, MIN_PRICE, MAX_PRICE, MIN_HOLD_EDGE, MIN/MAX_TIME_TO_EXPIRY_H. Add to `TIER_1_PARAMS` in `hypothesis_generator.py` with bounds.

### 3.2 Refactor `scan_crypto_markets` + `filter_signals`

Both functions currently import constants directly. Refactor to accept optional `params: dict | None = None` (defaulting to module constants if None). Pattern is identical to what we did for `b3_15m_scanner.scan()`. Files touched:
- `arbo/strategies/crypto_price_scanner.py`
- `arbo/strategies/crypto_quality_gate.py` — `filter_signals(signals, params=None)`

### 3.3 Wire shadow eval into `StrategyB2.poll_cycle`

After computing `qualified` list (post `filter_signals`), evaluate same candidates against **all** active variants and persist to `shadow_variant_signals`. Mirror approach used in `b3_15m_shadow.py::_evaluate_variants`.

For each pre-`filter_signals` candidate (`signals` list), per variant:
- Apply variant params via `filter_signals(signals, params=variant.params)`
- For each candidate → `qualified` (pass) / not — write row
- Capture features: edge, price, hours_to_expiry, sigma, asset, direction
- Set `qualified=True` if passed, with `would_fill_at` = mid price from CLOB snap (already fetched)

Schema mapping (re-use existing `shadow_variant_signals` columns, treat opaquely):
- `condition_id`, `token_id`, `signal_ts` — direct
- `direction` — "above" / "below" → store as-is
- `entry_price` → CLOB midpoint
- `edge` → `sig.model_prob - clob_price`
- `sigma` → `sig.sigma_hourly`
- `btc_at_start` → exchange_price (overload — semantic OK)
- `would_fill_at` → CLOB midpoint
- `event_end_ts` → market expiry

### 3.4 Resolution sweep

B2 markets resolve at `event_end_ts` via Polymarket Gamma `outcomePrices`. Add `_sweep_b2_shadow_resolutions` to `StrategyB2` (mirrors `_sweep_shadow_resolutions` in `strategy_b3.py`):
- SELECT distinct condition_id from `shadow_variant_signals WHERE strategy='B2' AND resolution_outcome IS NULL AND event_end_ts < now()-300`
- For each: fetch resolution from gamma `/events?slug=...` — for crypto markets the slug pattern is `<asset>-{above|below}-<strike>-<date>`
- Update rows: outcome + would_pnl (qualified rows = (1-fill) if won else -fill)

Call from `poll_cycle` once per cycle (every 60s), best-effort.

### 3.5 Champion + 3 challengers

| File | Param change | Hypothesis |
|---|---|---|
| `champion_v1.yaml` | (baseline) | Mirror current `crypto_quality_gate.py` constants |
| `ch_edge_tight.yaml` | `MIN_EDGE: 0.08 → 0.12` | Tighter edge — fewer trades, higher precision |
| `ch_price_low.yaml` | `MAX_PRICE: 0.60 → 0.50` | Avoid deep-ITM (low payoff per dollar risk) |
| `ch_short_fuse.yaml` | `MIN_TIME_TO_EXPIRY_H: 8 → 24` | Avoid <24h markets where edge may be illusory (model time-decay) |

All challengers keep KELLY/SIGMA/sizing identical to champion (Tier 2 — out of scope).

### 3.6 PerformanceAnalyzer failure-mode buckets

Add `B2` block to `_compute_failure_modes`:

| Feature | Bucket |
|---|---|
| `entry_price` | `> 0.50` (deep ITM) |
| `entry_price` | `> 0.40` |
| `edge` | `< 0.10` |
| `hours_to_expiry` | `< 12` |
| `hours_to_expiry` | `> 96` |
| `sigma` | (high vol regime — TBD bucket) |
| `asset` | `= 'ETH'` (vs BTC default — for early ETH-vs-BTC analysis) |

### 3.7 Optuna BO search space

Add to `research/bo_sweep.py::SEARCH_SPACES`:
```python
"B2": {
    "MIN_EDGE":              ("float", 0.05, 0.20),
    "MIN_PRICE":             ("float", 0.03, 0.10),
    "MAX_PRICE":             ("float", 0.40, 0.75),
    "MIN_HOLD_EDGE":         ("float", 0.01, 0.08),
    "MIN_TIME_TO_EXPIRY_H":  ("float", 4.0, 36.0),
    "MAX_TIME_TO_EXPIRY_H":  ("float", 96.0, 240.0),
}
```
And matching `_evaluate_params(strategy='B2')` block in same file: replay shadow signals through these gates, score by Sharpe × volume penalty.

### 3.8 Watchdog daemon for B2

Two options:
- (A) **Generalize `B3Watchdog`** to take strategy + adaptive_config in init, drop the `_get_gemini_decision` part if not needed for B2 (B2 already has its own monitoring). The `_autochallenger_cycle`, `_promotion_cycle`, `_drift_cycle` are already strategy-agnostic.
- (B) **New `B2Watchdog`** class that runs only the new cycles (autochallenger + promotion + drift), no anomaly-based runtime config changes (those need B2-specific metrics module that doesn't yet exist).

**Recommended: (B)** — simpler, smaller diff, doesn't touch B3 code. New file `arbo/core/b2_watchdog.py` ~120 LOC, only the 3 new cycles, periodic Slack heartbeat.

---

## 4. Plan — Strategy D_NBA (NBA green book)

D_NBA is class-attribute-driven, so refactor pattern differs slightly.

### 4.1 Inventory of params

From `StrategyDNba` class attrs:

| Attr | Current | Tier | Vary? |
|---|---|---|---|
| `MIN_EDGE` | 0.16 | 1 | Yes |
| `MAX_EDGE` | 0.25 | 1 | Yes |
| `MIN_PRICE` | 0.20 | 1 | Yes |
| `MAX_PRICE` | 0.65 | 1 | Yes |
| `GREEN_BOOK_DELTA` | 0.17 | 1 | Yes — exit profit-take threshold |
| `STOP_LOSS_DELTA` | 0.15 | 1 | Yes — exit stop-loss threshold |
| `MAX_HOLD_FRACTION` | 0.50 | 1 | Yes — fraction of game duration before time exit |
| `BOTH_SIDES` | True | 2 | NO (architectural — Tier 2) |
| `KELLY_FRACTION` | 0.15 | 2 | NO (sizing) |
| `KELLY_RAW_CAP` | 0.10 | 2 | NO (sizing) |
| `MAX_POSITION_PCT` | 0.03 | 2 | NO (sizing) |
| `ELO_WEIGHT` / `PINNACLE_WEIGHT` | 0.40/0.60 | 2 | NO (model param) |
| `MAX_CONCURRENT` | 8 | 2 | NO (capital concentration) |

### 4.2 Refactor pattern: introduce `_p(name)` helper

`StrategyDCore` reads `self.MIN_EDGE` etc. ~20 places. Cleanest refactor: add helper:

```python
def _p(self, name: str) -> Any:
    """Param accessor — variant override > class attr fallback."""
    if self._active_variant_params is not None:
        v = self._active_variant_params.get(name)
        if v is not None:
            return v
    return getattr(self, name)
```

Then replace `self.MIN_EDGE` → `self._p("MIN_EDGE")` in `generate_signals`, `kelly_size`, `check_exits`. About 20 line changes.

For shadow eval, we don't change `self._active_variant_params` at runtime — it stays None for the live champion path. We just need to evaluate **other variants' decisions** given the same `MarketData` + `prob`.

### 4.3 Wire shadow eval into `StrategyDCore.generate_signals`

Currently `generate_signals(markets)` returns champion's decisions. Add parallel:
```python
async def _evaluate_shadow_variants(self, market, prob):
    """Per market candidate, evaluate all D_NBA variants → write to shadow table."""
```
Called inside the for-loop over markets, AFTER `compute_model_prob`. For each variant:
- Apply variant's MIN_EDGE/MAX_EDGE/MIN_PRICE/MAX_PRICE on yes_edge + no_edge
- Determine qualified yes/no per side
- Write row(s) to `shadow_variant_signals` with strategy='D'

Features captured in row:
- `condition_id` = market.condition_id
- `direction` = "yes" / "no"
- `entry_price` = market.yes_price / market.no_price
- `edge` = yes_edge / no_edge
- `would_fill_at` = market price (no orderbook fetch — D uses Gamma quotes directly)
- `event_end_ts` = game start + GAME_DURATION_HOURS (resolution time)
- Special D_NBA columns we need:
  - `velocity`, `dir_delta` are B3-specific — leave NULL
  - Use existing column for `model_prob`? **Decision: store in `sigma` column** (overload — semantically "model uncertainty proxy"; document this). Or extend table with `model_prob FLOAT` — small migration.
  - **Recommendation:** small migration `014_shadow_variant_extras` adds `model_prob FLOAT NULL` and `meta_json JSONB NULL` so D-specific data fits cleanly.

### 4.4 Resolution sweep for D_NBA

NBA games resolve via Polymarket Gamma (winner outcome posted). Need:
- New helper `arbo/strategies/d_resolution.py`:
  - `async def fetch_d_resolution(condition_id) -> bool | None` — returns True if YES side won, False if NO won, None if not yet resolved
  - Uses `https://gamma-api.polymarket.com/markets?condition_ids=...`
- Sweep method in `StrategyDCore._sweep_d_shadow_resolutions` mirroring B3's pattern.
- Called from `poll_cycle` once per cycle.

For PnL computation: `qualified=true` rows get PnL = (1-fill) if `direction='yes' AND yes_won` (or `direction='no' AND not yes_won`) else -fill.

### 4.5 Champion + 3 challengers

| File | Param change | Hypothesis |
|---|---|---|
| `champion_v1.yaml` | (baseline) | Mirror current `StrategyDNba` class attrs |
| `ch_edge_tight.yaml` | `MIN_EDGE: 0.16 → 0.20` | Tighter edge — same hypothesis as B3/B3_15M |
| `ch_gb_loose.yaml` | `GREEN_BOOK_DELTA: 0.17 → 0.12` | Lock profits earlier — fewer reversals |
| `ch_sl_tight.yaml` | `STOP_LOSS_DELTA: 0.15 → 0.10` | Tighter stop — limit max loss per trade |

### 4.6 PerformanceAnalyzer failure-mode buckets

Add `D` block (note strategy name is "D" not "D_NBA" — single string per architecture decision):

| Feature | Bucket |
|---|---|
| `entry_price` | `> 0.55` (favorite — green book unreliable) |
| `entry_price` | `< 0.30` (longshot) |
| `edge` | `< 0.18` (low-confidence) |
| `direction` | `= 'no'` (BOTH_SIDES no-side losing more often?) |

NB: D failure modes need access to `paper_trades.trade_details->>'side'`, `'team_a'`, etc. — already stored.

### 4.7 BO search space

```python
"D": {
    "MIN_EDGE":          ("float", 0.10, 0.25),
    "MAX_EDGE":          ("float", 0.20, 0.40),
    "MIN_PRICE":         ("float", 0.15, 0.35),
    "MAX_PRICE":         ("float", 0.55, 0.80),
    "GREEN_BOOK_DELTA":  ("float", 0.10, 0.25),
    "STOP_LOSS_DELTA":   ("float", 0.08, 0.20),
    "MAX_HOLD_FRACTION": ("float", 0.30, 0.80),
}
```
Plus matching `_evaluate_params(strategy='D')` block — replays shadow signals through these gates, scores by Sharpe.

### 4.8 D Watchdog

Same architecture as B2: small new file `arbo/core/d_watchdog.py` running only `_autochallenger_cycle` + `_promotion_cycle` + `_drift_cycle`. ~120 LOC.

### 4.9 main_rdh wiring

Need to instantiate `B2Watchdog` and `DWatchdog` in `_init_optional` block of main_rdh.py, register their `run()` as background tasks (mirror B3Watchdog instantiation pattern at lines ~1657).

### 4.10 Dashboard cards

Two new Variant Leaderboard cards in `dashboard.html` (B2 yellow border, D pink border) following the existing B3 5-min / B3_15M template. Same `fetchVariantLeaderboard()` JS — just two more strategy IDs.

---

## 5. Shared infrastructure changes

### 5.1 `TIER_1_PARAMS` extension

Add to `arbo/core/hypothesis_generator.py`:
```python
TIER_1_PARAMS = {
    # ... existing B3/B3_15M params ...
    # B2 additions
    "MIN_EDGE":              (0.05, 0.25),
    "MAX_EDGE":              (0.50, 0.99),
    "MIN_PRICE":             (0.03, 0.20),
    "MAX_PRICE":             (0.30, 0.80),
    "MIN_HOLD_EDGE":         (0.00, 0.15),
    "MIN_TIME_TO_EXPIRY_H":  (1.0, 48.0),
    "MAX_TIME_TO_EXPIRY_H":  (48.0, 336.0),
    # D additions
    "GREEN_BOOK_DELTA":      (0.05, 0.30),
    "STOP_LOSS_DELTA":       (0.05, 0.25),
    "MAX_HOLD_FRACTION":     (0.20, 1.00),
}
```
Note: `MIN_EDGE`/`MAX_EDGE`/`MIN_PRICE`/`MAX_PRICE` are now shared across B2 + D + B3 (same name, different bounds intentionally — bounds are set wide enough to satisfy all). This is OK because PoolManager validates per-strategy YAMLs, not against global defaults.

### 5.2 Migration 014 (optional for D)

If we want clean D-specific columns:
```sql
ALTER TABLE shadow_variant_signals
  ADD COLUMN model_prob FLOAT NULL,
  ADD COLUMN meta_json JSONB NULL;
```
Backward-compatible. Otherwise, overload `sigma` column with documentation. **Recommend the migration** for clarity.

### 5.3 `STRATEGY_CHANNELS` in slack_promotion.py

```python
STRATEGY_CHANNELS = {
    "B3":     "C0APX4K8Z2N",
    "B3_15M": "C0APX4K8Z2N",
    "B2":     "<TBD — check existing B2 channel or reuse default>",
    "D":      "<TBD — check existing D NBA channel or reuse default>",
}
```
Need to confirm channel IDs from existing B2 / D Slack notifications.

---

## 6. Risk register

| Risk | Mitigation |
|---|---|
| D refactor (`self.X` → `self._p("X")`) breaks live execution | Run unit tests after each batch of 5 replacements; deploy in shadow-only mode first |
| B2 shadow rows pollute existing autoresearch dataset | NO — `crypto_price_*` autoresearch reads its own SQLite tables, not `shadow_variant_signals` |
| D_NBA NBA games happen sparsely (0-15/day) — slow shadow accumulation | Accept; PromotionEngine MIN_PAIRED_N=100 means weeks of data needed; mark in dashboard |
| `model_prob` column added to shadow_variant_signals breaks existing queries | Migration is additive (NULL default). Existing queries unaffected |
| Multiple watchdog Slack messages clutter channel | Add per-strategy rate limit (1 msg/strategy/hour for non-critical) |
| Auto-challenger generates duplicate-of-existing-variant for D | HypothesisGenerator already dedupes; new `D`-specific bounds prevent overlap |

---

## 7. Acceptance criteria

- [ ] B2 has 4 active variants (champion + 3 challengers) loaded from YAML
- [ ] `shadow_variant_signals` has rows with `strategy='B2'` after 1 poll cycle
- [ ] B2 resolution sweep updates `would_pnl_per_share` after first market resolves
- [ ] `/api/variants?strategy=B2` returns variant rows with shadow stats
- [ ] D has 4 active variants loaded from YAML
- [ ] `shadow_variant_signals` has rows with `strategy='D'` after first NBA market candidate
- [ ] D resolution sweep populates outcomes after first NBA game ends
- [ ] `/api/variants?strategy=D` returns variant rows
- [ ] Dashboard shows two new Variant Leaderboard cards (B2 + D)
- [ ] B2Watchdog + DWatchdog visible in arbo logs as background tasks
- [ ] `python research/bo_sweep.py --strategy B2 --n-trials 50` runs to completion
- [ ] `python research/bo_sweep.py --strategy D --n-trials 50` runs to completion
- [ ] No regressions in B3 / B3_15M variant flow (sanity check after deploy)

---

## 8. Time estimate

| Phase | Est. |
|---|---|
| B2 refactor (params-aware scanner + filter_signals) | 30 min |
| B2 shadow eval + sweep + Slack channel + watchdog | 45 min |
| B2 YAMLs + dashboard card + BO search space | 20 min |
| Migration 014 (model_prob + meta_json) | 15 min |
| D_NBA `_p()` helper refactor | 30 min |
| D_NBA shadow eval + sweep + watchdog | 45 min |
| D YAMLs + dashboard card + BO search space | 20 min |
| Slack channels + TIER_1_PARAMS extension + commits | 20 min |
| Verify on Dublin + log walkthrough | 30 min |
| **Total** | **~4h** |

---

## 9. Commit plan

Sequential commits (each independently deployable, no half-states):
1. `feat(db): migration 014 — model_prob + meta_json on shadow_variant_signals`
2. `refactor(b2): params-aware scan_crypto_markets + filter_signals`
3. `feat(b2): 4 variant YAMLs + shadow eval + resolution sweep`
4. `feat(b2): B2Watchdog daemon (autochallenger + promotion + drift cycles)`
5. `refactor(d): introduce _p() variant param accessor on StrategyDCore`
6. `feat(d): 4 variant YAMLs + shadow eval + resolution sweep`
7. `feat(d): DWatchdog daemon`
8. `feat(framework): extend TIER_1_PARAMS + Optuna search spaces for B2 + D`
9. `feat(dashboard): B2 + D Variant Leaderboard cards`
10. `feat(slack): B2/D channel routing in promotion bot`

10 commits. Each tested locally + deployed sequentially. Stop+report at each major checkpoint.

---

## 10. Rollback strategy

Each component additive — no breaking changes to existing strategies:
- Migration 014 backward-compatible (NEW NULLABLE columns)
- Shadow eval is a side-effect (doesn't gate live trading)
- New watchdogs are background-only — disable by removing `asyncio.create_task` registration in main_rdh
- YAMLs are file-based — `git revert` restores previous state

If catastrophic: `git revert <last-good-sha>..HEAD && deploy` — full Project PARALLEL B2/D extension reverted, B3/B3_15M untouched.
