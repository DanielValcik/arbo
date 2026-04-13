# Arbo Strategy Optimization — Empirical Learnings Log

> Append-only log of every optimization decision, observation, and outcome across all Arbo strategies.
> Complements `STRATEGY_OPTIMIZATION_FRAMEWORK.md` (theory) with accumulated empirical evidence (practice).
> Rule: every strategy decision (change, deploy, revert, kill) MUST be logged here with date, rationale, and outcome. Review before any new decision on the same strategy.

## How to Use This Log

### Before making any strategy decision:
1. Search this log for the relevant strategy name and decision type
2. Check if similar decision was made before — what was the outcome?
3. If pattern found (same mistake repeated, same bug recurring) → STOP, address root cause instead

### After making any strategy decision:
1. Append a new entry in the relevant section below using the template
2. Update the entry when outcome is known (usually after N trades post-decision)
3. Tag as `KEPT` / `REVERTED` / `KILLED` / `ONGOING`

### Entry Template

```markdown
### [YYYY-MM-DD] [STRATEGY] — [short title]

**Observation**: what triggered the decision (data, alert, anomaly).
**Data at decision time**:
  - Live N: X trades
  - Shadow N: Y trades (if applicable)
  - Live Sharpe / WR / PnL: ...
  - Bayesian posterior: shadow weight X%, live weight Y%

**Hypothesis**: what we thought the problem was.
**Framework gates checked**: ✓/✗ each rule (§10.1 min N, §5.3 regime, §8.2 P(better), etc.)
**Decision**: what we did (change X from A to B, or held, or killed).
**Evidence basis**: P(better) bootstrap result, DSR, PBO, regime detection output.
**Revert triggers armed**: yes/no + specific thresholds.
**Outcome (updated after N trades)**:
  - Measured effect: ...
  - Verdict: `KEPT` / `REVERTED` / `INCONCLUSIVE` / `ONGOING`
  - Lesson: what we'd do differently next time.
```

---

## Section 1: Meta-Learnings (lessons that apply across all strategies)

### 2026-04-13 ALL — Bayesian dominance of small live N over large shadow N

**Observation**: B3_15M had live N=12 showing -$3.69, shadow N=144 showed +$9.89 rank #1. Instinct was to change params based on live data.

**Lesson**: With inverse-variance weighting, live N=12 vs shadow N=144 gives shadow 92.3% weight. Live alone cannot statistically override shadow unless extreme deviation (>3σ) or regime change detected. **Below N=100 live, do not override shadow-derived params.**

**Applicability**: ALL strategies. Standard error of WR at N<50 is too wide to distinguish skill from luck for Arbo's typical WRs (60-85%).

---

### 2026-04-13 ALL — Shadow data assumes maker fills always succeed

**Observation**: Shadow autoresearch for B3_15M predicted 92% WR for fill ≥0.75. Live reality gave 57% WR in same bucket.

**Lesson**: Shadow data records `would_fill_at = best_bid` which assumes 100% PostOnly fill success. Real PostOnly fill rate is 70-80% AND fills disproportionately happen when trend reverses (adverse selection, "winner's curse"). Shadow overestimates WR for high-fill tokens because shadow couldn't model which fills would actually occur.

**Applicability**: Any strategy using PostOnly maker orders. Apply a conservative haircut to shadow WR for high-fill buckets until live validates. **Framework §6** covers this.

---

### 2026-04-13 ALL — Never match redeem events by condition_id — match by token_id

**Observation**: 31 of 273 historical B3 trades had wrong `live_exit_price` because `_resolve_b3_from_redeem` matched Polymarket auto-redeem events by condition_id instead of token_id. Every market has 2 token_ids (UP + DOWN); when one side won, BOTH sides got marked as $1 win.

**Lesson**: Polymarket's `conditionId` is the market, `asset` (token_id) is the specific outcome. `redeemable=True` is also ambiguous — it means "market resolved, redeem tx possible", not "we won". The true discriminator for win is `currentValue > 0` OR `tokenId in redeemed_token_ids`.

**Applicability**: Any Polymarket integration. Always track at token_id granularity, never condition_id for win/loss determination.

**Fixed**: commit 56595ad. Historical 31 trades backfilled via Gamma API audit.

---

### 2026-04-13 ALL — cleanup_stale_on_restart loses positions when event in-flight

**Observation**: 5 live trades orphaned (live_exit_price=NULL forever) because service restart happened between entry and event_end_ts. `cleanup_stale_on_restart` marked them "deferred" but never re-inserted into `_live_holding` dict. check_exits never saw them. If our side WON, auto_redeem fast path would resolve; if our side LOST, it stayed unresolved (Polymarket returns currentValue=0 for losers, filtered out by auto_redeem).

**Lesson**: Any state that lives only in-memory must have a restore-from-DB path on startup. A "deferred" comment in code saying "will handle later" without actual mechanism to handle it = silent orphan bug.

**Applicability**: Any strategy holding positions across potential restart. Always validate startup restoration covers all in-flight states.

**Fixed**: commit 06d3362. Historical 5 trades backfilled with real Gamma outcomes (all losses).

---

## Section 2: Strategy B3 5-min

### Model baseline (V6.0 deployed 2026-04-06 18:27 UTC, commit 596d5fd)

- **Params**: LIVE_MIN_EDGE=0.30, LIVE_MIN_BTC_MOVE=35, LIVE_MAX_VELOCITY=60, LIVE_MAX_DIR_DELTA=15, LIVE_MAX_FILL_PRICE=0.75
- **Capital**: $100 (env: B3_LIVE_CAPITAL=100)
- **Dual Filter basis**: 278 live-trade analysis pre-V6.0
- **Strategy**: never-sell, Chainlink resolution

### Decision log

### [2026-04-13] B3 5-min — HOLD current params (framework v1.0 first application)

**Observation**: V6.0-era live PnL slightly negative (-$2.39 over 28 trades). WR 67.9% vs breakeven 72.3% (at avg entry 0.723). User asked to review with new framework methodology.

**Data at decision time**:
- Live N (V6.0 era): 28 trades
- Live WR: 67.9% (19W / 9L)
- Avg entry: $0.723 → breakeven WR 72.3%
- Margin below breakeven: -4.4pp
- Total PnL V6.0: -$2.39
- Std per-trade PnL: $3.79
- Annualized Sharpe (V6.0 era): -0.72
- All-time B3 live: +$64.59 / 281 trades (pre-V6.0 contributed +$67)
- BTC vol shadow vs live: 36.3% vs 39.0% (ratio 1.08x)

**Hypothesis**: Strategy may have negative expected value due to payout asymmetry (high fill → small win ceiling, large loss floor). OR small-sample noise (|Z|=-0.52 vs breakeven, within 1σ).

**Framework gates checked**:
- ✗ Min N for filter loosening: needs 100, have 28 V6.0-era → cannot loosen
- ✗ Min N for filter tightening: needs 50, have 28 → cannot tighten
- ✓ Min N for risk reduction: 0 → allowed any time
- ✓ Regime check: vol ratio 1.08x → no regime change
- Z-score live vs breakeven: -0.52 → WR 67.9% not statistically distinguishable from 72.3% at N=28
- Posterior under shadow prior: shadow data for 5-min lives in 278-trade pre-V6.0 analysis, not directly comparable

**Decision**: **HOLD** current V6.0 params. No filter changes. Continue monitoring.

**Evidence basis**: N=28 insufficient for any change per §10.1. Vol regime matches shadow. WR 67.9% not statistically different from breakeven 72.3% (|Z|<1).

**Revert triggers armed** (per §8.3):
- 50-trade rolling Sharpe drops > 50% from -0.72 → pause and re-validate
- 50-trade rolling WR drops > 5pp below 67.9% → pause
- Cumulative V6.0 PnL drops below -$20 (more than 2× current) → pause for review

**Outcome (to update at N=50 V6.0 era, estimated 4-5 weeks at current rate)**:
- Measured effect: TBD
- Verdict: ONGOING
- Lesson: TBD

---

## Section 3: Strategy B3_15M

### Model baseline (deployed 2026-04-12 08:33 UTC)

- **Source**: Shadow autoresearch, rank #1 config, 5-fold CV on 144 resolved signals
- **Params**: min_edge=0.30, max_btc_move=80, max_market_gap=0.30, entry min 4-11, fill uncapped, sigma_scale=0.526
- **Shadow performance**: 49 qualified trades, WR 94.4%, +$9.89, avg $0.222/share, σ=0.103
- **Live capital**: $75 (env: B3_15M_LIVE_CAPITAL=75)

### Decision log

### [2026-04-13] B3_15M — HOLD current params (framework v1.0 first application)

**Observation**: Live PnL -$0.83 over 13 trades. WR 69.2%. Shadow predicted +$0.222/share (~+$35 expected on 13 trades of avg 12 shares). Live diverged negatively.

**Data at decision time**:
- Live N: 13 trades (V6.0 deploy 4/12 08:33)
- Live WR: 69.2% (9W / 4L)
- Avg entry: $0.715 → breakeven WR 71.5%
- Margin below breakeven: -2.3pp
- Total PnL: -$0.83
- Annualized Sharpe: -0.22
- All 4 losses were post-restart orphans (resolved via backfill, exposed cleanup_stale bug)

**Framework gates checked**:
- ✗ All change thresholds require N ≥ 50+ → have 13 → refuse any change
- ✓ Risk reduction always allowed
- **Bayesian posterior** (shadow N=144, live N=13):
  - weight_shadow = 91.7%
  - weight_live = 8.3%
  - Posterior per-share PnL: **+$0.203** (still positive, dominated by shadow)
- **Z-score live vs shadow**: -2.05 (approaching statistical divergence but still within 2σ)
- **Regime check**: vol ratio 1.08x → no regime change detected
- **Autoresearch gates** (retrospective on shadow result):
  - ✗ DSR not computed with N_trials=235k
  - ✗ PBO not explicitly computed
  - ✓ CPCV-ish (5-fold time-ordered) used
  - ⚠️ Shadow simulated `would_fill_at=best_bid` which assumes 100% maker fill (adverse selection unaccounted — Meta-Learning 2 in Section 1)

**Hypothesis**: Live underperformance is (a) sample variance at small N, OR (b) shadow overestimated WR for high-fill bucket due to adverse selection. Cannot distinguish at N=13.

**Decision**: **HOLD** current shadow-derived params. No filter changes. No fill cap change despite temptation (recorded as anti-pattern on 4/12 and 4/13).

**Evidence basis**: N=13 is 7.7% of evidence vs 92.3% shadow. Z=-2.05 is at boundary of significance but not conclusive. Cannot reject shadow at this sample size. §10.1 explicitly forbids filter changes below N=50-100.

**Revert triggers armed**:
- Rolling 10-trade WR drops below 50% → pause
- Cumulative PnL drops below -$15 (more than 3× current) → pause
- Regime change detected (vol ratio > 1.5x) → re-validate on new regime

**Outcome (to update at N=50)**:
- Measured effect: TBD
- Verdict: ONGOING
- Lesson: TBD — especially: does Z-score to shadow decrease (confirming shadow) or increase (suggesting shadow overfit)?

---

## Section 3.5: Monitoring Plan (what to watch passively)

### B3 5-min
- Checkpoint at V6.0 N=50 (mid-decision): ~4-5 weeks at 5-6 trades/week
- Full review at V6.0 N=100: ~10-12 weeks
- Dashboard metrics to monitor: 50-trade rolling WR, rolling Sharpe, PSI on velocity/dir_delta

### B3_15M
- Checkpoint at N=50: ~3 weeks at 3-4 trades/day active
- Full review at N=100: ~6-8 weeks (allows Bayesian weight to reach 42/58)
- Dashboard metrics: WR by fill bucket (shadow said high fill best, live says opposite — key metric to track)
- **Regime flag**: re-check BTC vol ratio monthly

### Pre-scheduled decisions (do NOT trigger before dates)
- 2026-04-20: mid-checkpoint B3 5m (if N reaches 50)
- 2026-04-27: mid-checkpoint B3_15M (if N reaches 50)
- 2026-05-15: major review both strategies

## Section 4: Strategy C (Compound Weather)

### Decision log

<!-- Add new entries with newest at top -->

---

## Section 5: Strategy C2 (EMOS + Exit Fusion)

### Decision log

<!-- Add new entries with newest at top -->

---

## Section 6: Strategy B2 (Crypto Price Edge)

### Decision log

<!-- Add new entries with newest at top -->

### 2026-04-13 B2 — Attempted spread inversion fix REVERTED (hypothesis failed live)

**Observation**: Dashboard shows B2 at +$1,694 paper PnL, 83% WR over 663 trades. Inspection of closed trades showed systematic +$0.01-0.05 exit-above-entry delta matching typical crypto spreads. Hypothesis: paper engine "earns the spread" — entry at BID (low) instead of ASK (high). Open positions visible on dashboard showed real -$200+ unrealized, confirming paper PnL was inflated vs reality.

**Data at decision time**:
  - Live N: **0** (B2_EXECUTION_MODE=paper throughout)
  - Paper N: 663 (639 sold, 14 lost, 10 open) since 2026-03-28
  - Paper PnL +$2,436 on closed "sold" trades, -$742 on "lost" resolved, net +$1,694
  - ~$1,900 of paper "sold" PnL appeared consistent with spread-width inflation

**Hypothesis**: `strategy_b2.py:262` used `clob_price = min(raw_bid, raw_ask)` = BID side (low) for paper entry; a real taker BUY pays ASK (high). Proposed fix: `min` → `max`.

**Framework gates checked**:
  - ✓ Step 0 read learnings — Section 6 empty, no prior B2 decisions
  - ✗ Step 2 min N — B2 live N=0, so any "data-driven" change impossible; reasoned as a code bug fix, not param change
  - ✗ Step 8 bootstrap P(better) — not applicable (code fix, not threshold change)
  - ✓ Step 9 revert criteria — planned to revert if live behavior contradicted hypothesis

**Decision**: Deployed fix (commit `3653692`), restarted arbo.service on arbo-dublin.

**Evidence basis**: Code trace + orderbook_provider.py comment (`best_bid=sell_price`, `best_ask=buy_price` for NegRisk /price endpoint path). Assumed same semantics for crypto markets (which use same /price endpoint via `neg_risk=True` flag).

**Revert triggers armed**: Monitor `b2_entry_summary` skip_reasons for first hour post-deploy. Expected: `clob_edge_low` should INCREASE (edge shrinks by spread width when using ASK).

**Outcome**: **REVERTED** within 3 minutes.
  - Pre-fix (old code): `qualified=7, clob_edge_low=5, reached_sizing=2` (29% pass rate)
  - Post-fix (new code): `qualified=26, clob_edge_low=1, reached_sizing=18` (69% pass rate)
  - Live `/price` probe for actual crypto token: `?side=BUY → 0.05`, `?side=SELL → 0.07` — **SELL > BUY**, OPPOSITE of normal convention
  - `max(bid,ask)` resulted in LOWER rejection rate, not higher — contradicts bug hypothesis
  - Reverted via `git revert` (commit `065a274`). Revert caused production crash (~90s downtime) due to VPS stash/pop conflict with uncommitted D-variant changes that duplicated a merged commit; resolved by `git checkout HEAD --` on conflicted files.

**Lesson**:
  1. **The `/price` endpoint semantics differ between NegRisk and standard markets.** For crypto (non-NegRisk) markets forced through `/price` via `neg_risk=True`, the BUY/SELL side return values appear inverted vs. the comment in `orderbook_provider.py:166` which states "BUY > SELL for normal markets". Actual data: SELL > BUY. Root cause of paper PnL pattern is NOT yet understood — further investigation required with real CLOB orderbook (`/book` endpoint) comparison.
  2. **Never `git stash` + `git pull` on VPS with uncommitted duplicate-of-merged-commit changes** — stash pop creates phantom conflict markers and crashes the service. Either commit the VPS local changes first, or use `git checkout HEAD --` on conflicting files after pop to discard the stashed (already-in-mainline) changes.
  3. **Small observation windows (1-2 scan cycles) can still be decisive when the effect is binary** — here, 29% vs 69% pass rate across different signal sets is a large signed divergence in the wrong direction. Trust it.
  4. **Do not "fix" a hypothesized bug without first reproducing the observed symptom offline** — I should have dumped live orderbook + /price side-by-side BEFORE touching production. Next attempt: write a standalone probe script.

**Verdict**: `REVERTED`. Reopened as "Re-investigate B2 paper pricing" task. B2 remains on paper with the ORIGINAL (min-based) pricing; known to inflate PnL but direction/magnitude of inflation now uncertain.

**Next actions**:
  - Write probe: `research/probe_b2_pricing.py` that for N=20 crypto tokens dumps `{/book best_bid, /book best_ask, /price?side=BUY, /price?side=SELL, Gamma price_yes}` so we can determine true semantics.
  - Until probe results understood: do NOT touch `strategy_b2.py`, do NOT run autoresearch on existing paper data, do NOT deploy B2 live.
  - Consider: is paper engine "over-optimistic" or "over-pessimistic" right now? Until we know, we cannot interpret the +$1,694 figure at all.

---

## Section 7: Strategy D / D_UFC / D_EPL (Sports)

### Decision log

<!-- Add new entries with newest at top -->

---

## Section 8: Recurring Anti-Patterns (things we keep doing wrong)

This section is populated when the same mistake appears 2+ times across strategies. Patterns here trigger immediate pause and framework review before next decision.

### Anti-pattern 1: Reacting to small-N live with param changes

**Observed**: 2026-04-12 (B3_15M proposed fill cap change on N=12), 2026-04-13 (similar temptation with B3 N=20)
**Fix**: Framework §10.1 minimum N table. Skill `arbo-optimize` now auto-refuses below threshold.

### Anti-pattern 2: Deploying a hypothesized bug fix without offline reproduction

**Observed**: 2026-04-13 (B2 `min→max` fix — hypothesis about /price endpoint semantics, deployed to prod, reverted within 3 min when live data contradicted hypothesis. Caused ~90s production downtime via stash/pop conflict).

**Fix**: Before touching production code to "fix a bug" derived from trace-reading:
  1. Write a standalone probe script that reproduces the observed symptom offline using real API calls.
  2. Verify the probe's output matches the symptom (same direction, same magnitude).
  3. Only then modify production code.
  4. Deploy during a quiet period; monitor for at least one FULL scan cycle before marking done.

### Anti-pattern 3: VPS `git stash` + `git pull` when uncommitted changes duplicate merged commits

**Observed**: 2026-04-13 (arbo-dublin VPS had uncommitted D-variant changes identical to commit `02362c6` already in origin/main. `git stash` + `git pull` + `git stash pop` created phantom conflict markers in `main_rdh.py` and `strategy_d_core.py` → Python SyntaxError → service crash loop).

**Fix**: On the VPS, ALWAYS check `git status` before pulling. If uncommitted changes exist:
  - If they match a merged commit: `git checkout -- <files>` to discard (safe — already in mainline)
  - If they are genuinely new work: commit them to a branch first, then pull, then rebase
  - NEVER rely on `stash pop` to cleanly replay duplicate changes

---

## Section 9: Successful Patterns (replicate these)

### Pattern 1: Shadow scanner as low-risk data collection

**Used for**: B3_15M (b3_15m_shadow.py collected 144 real-market signals without trading)
**Result**: Enabled autoresearch on real fill prices + spreads + resolutions before any live capital.
**Generalization**: Before deploying any new strategy, run shadow scanner for ≥ 2 weeks to collect real market data for autoresearch.

---

## Section 10: Framework Amendments

When the framework itself needs updating based on empirical evidence, log the amendment here.

| Date | Amendment | Rationale |
|---|---|---|
| 2026-04-12 | v1.0 created | Initial authoritative doc |
| 2026-04-13 | v2.0 — added §11 Rapid Mode + VARIANT_LEADERBOARD_SPEC.md | Serial mode alone = 10-14 days per iteration; too slow for 5+ strategies. Research into hedge fund techniques (RAPID_MODEL_DISCOVERY.md) identified champion-challenger, BO, MAB, drift detectors as 5-10× speedup. Integration preserves all serial-mode gates (DSR, PBO, revert triggers) but runs 8× more hypotheses in parallel. Dashboard leaderboard card made MANDATORY — variants without visibility = invisible failures. |
| 2026-04-13 | Phase 1 implementation COMPLETE (commits f64b409 + 43253ed) | Variant pool infrastructure live: `arbo/core/variant_pool.py` loads YAML configs, 2 champion YAMLs (b3 + b3_15m) + 3 b3_15m challenger YAMLs, `trade_details.variant_id` written on every new B3/B3_15M trade, `/api/variants` endpoint live, Variant Leaderboard cards rendered on B3 tab, `ShadowOrchestrator` skeleton class. Zero regression. Champion-only display per Phase A spec. Phase 2 next: wire orchestrator, add Optuna BO, MABWiser allocator. |

### v2.0 upgrade details (2026-04-13)

**What was added to framework:**
- §11 Rapid Mode — 16 subsections covering champion-challenger, Bayesian optimization, multi-armed bandit (Thompson Sampling), Page-Hinkley drift detection, composite mid-trade reward, block bootstrap, HRP ensemble, hypothesis factory FSM, tooling, what NOT to use (GANs, RL, stacking), dashboard requirement, decision addendum to §10.4, 4-phase implementation plan.
- §11.11 introduced MANDATORY Variant Leaderboard card requirement.

**What was explicitly rejected from rapid discovery study:**
- GAN / diffusion synthetic market data (TimeGAN, QuantGAN) — fail to capture stylized facts, would cause false DSR.
- Full reinforcement learning — wrong-sized for $175 live capital, bandit + BO gets 80% benefit at 10% complexity.
- Complex ensemble (stacking, meta-learners) — for 5-10 variant pool, HRP + quorum rule outperforms empirically.
- Real-time BO during trading — BO belongs in weekly autoresearch, not live loop.

**Skill upgraded**: `/optimize` skill now has Step 1.5 (serial vs rapid routing) and enforces Variant Leaderboard card before allowing multi-variant deployment.

**Not yet implemented (todo when user requests)**:
- `arbo/core/strategy_orchestrator.py` (generic multi-variant orchestrator)
- `arbo/core/bandit_allocator.py` (MABWiser wrapper)
- `arbo/core/drift_monitor.py` (river Page-Hinkley/ADWIN)
- `arbo/core/variant_pool.py` (declarative YAML config per variant)
- `arbo/dashboard/variant_leaderboard.py` (dashboard card backend)
- Optuna replacement for grid sweeps in `research/innovations/sweep_*.py`

**Pipeline from here**: user will invoke `/optimize` to apply framework on specific strategy; rapid mode will be chosen where appropriate (multi-variant exploration); serial mode where singular decision.

### 2026-04-13 — Project PARALLEL Phase 1 COMPLETE

**Goal**: deploy infrastructure for multi-variant champion-challenger exploration (declarative configs, per-trade attribution, dashboard visibility) without disrupting existing live B3 5-min and B3_15M strategies.

**Delivered (3 commits over batches A/B/C)**:

| Component | File | Status |
|---|---|---|
| VariantConfig dataclass + YAML loader | `arbo/core/variant_pool.py` | live |
| YAML pool root | `arbo/config/variants/` | populated |
| B3 5-min champion YAML (V6.0 mirror) | `b3/champion_v1.yaml` | live |
| B3_15M champion YAML (shadow rank #1 mirror) | `b3_15m/champion_v1.yaml` | live |
| B3_15M challenger YAMLs (3 ideas, inert) | `ch_edge_tight`, `ch_fill_cap_075`, `ch_gap_tight` | files only — no orchestrator wired yet |
| `trade_details.variant_id` injection | `strategy_b3.py` L622, `strategy_b3_15m.py` L620 | live, ✓ verified on trade 4248 |
| `/api/variants?strategy=<s>` endpoint | `web.py` L2291 | live, returns 4 B3_15M + 1 B3 variants |
| Variant Leaderboard card (B3 5-min) | `dashboard.html` near L1697 | rendered |
| Variant Leaderboard card (B3 15-min) | `dashboard.html` next sibling | rendered with 4 challenger rows |
| JS fetcher `fetchVariantLeaderboard` | `dashboard.html` L5275-ish | active on B3 tab refresh |
| ShadowOrchestrator skeleton class | `arbo/strategies/shadow_orchestrator.py` | class only — NOT in task loop |

**Verified end-to-end**:
- `python -c "from arbo.core.variant_pool import get_champion; print(get_champion('B3_15M').params['SIGMA_SCALE'])"` → `0.526` (matches quality_gate.py)
- After deploy + restart: trade 4248 (B3 5-min, placed 2026-04-13 13:36 UTC) has `trade_details->>'variant_id' = 'champion_v1'`
- `curl /api/variants?strategy=B3_15M` returns 4 variants with correct status, params_summary diffs, and capital_pct (champion 100%, challengers 0%)
- Dashboard B3 tab shows two Variant Leaderboard cards
- `ShadowOrchestrator('B3_15M')` instantiates, loads 4 variants, no errors
- Existing B3 + B3_15M live trading unchanged (verified live trades continue, no exceptions in journalctl)

**Did NOT do (intentionally — Phase 1 scope)**:
- ❌ Wire ShadowOrchestrator into task loops (Phase 2)
- ❌ Optuna BO replacement of grid sweeps (Phase 2)
- ❌ MABWiser Thompson Sampling for capital allocation (Phase 3)
- ❌ Page-Hinkley drift detectors (Phase 4)
- ❌ Composite reward (mid-trade) logging (Phase 3)
- ❌ HRP cross-strategy ensemble (Month 2)

**Anti-pattern protection in place**: skill `optimize` Step 1.5 routes any future multi-variant proposal through the framework gates. Dashboard requirement enforced — multi-variant deployment blocked without leaderboard card visible.

**Next step (when user requests Phase 2)**: refactor `strategy_b3_15m.py` scan loop to be variant-aware (accepts params dict instead of importing module constants), wire ShadowOrchestrator into `_run_b3_15m_shadow` task loop to evaluate all 4 variants on every signal, write paired-sample data to new table `shadow_variant_signals`. Estimated: 2-3 days work, no live capital at risk.

| _(next)_ | | |

---

## Maintenance

- **Append-only** below decision log headers. Old entries are historical record, never delete.
- **Outcome updates** to existing entries are allowed — add `**Updated YYYY-MM-DD:**` line with new finding.
- **Review cadence**: monthly scan for anti-patterns; quarterly sync of learnings into framework if needed.
