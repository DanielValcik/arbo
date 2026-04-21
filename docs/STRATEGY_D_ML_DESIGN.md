# Strategy D — ML Design (Exit-Timing Survival Model)

> **Scope:** Replace fixed `GREEN_BOOK_DELTA` + `MAX_HOLD_FRACTION` exit rules with a learned adaptive exit policy. Entry rule (Elo+Pinnacle edge gate) stays unchanged.
> **Date:** 2026-04-20
> **Status:** DESIGN — review before implementation
> **Related:** `docs/STRATEGY_D_SPEC.md` (original rule-based design), `LEARNINGS.md` entries D1, D2, D3.

---

## 1. Problem Statement

### Current exit mechanism (champion v1, `strategy_d_nba.py`)

```python
# 3 exit triggers, first-fired wins:
if price >= entry_price + GREEN_BOOK_DELTA:   # 0.17 for NBA
    exit("green_book")
elif price <= entry_price - STOP_LOSS_DELTA:  # 0.15
    exit("stop_loss")
elif time_elapsed >= 0.50 * game_duration:    # MAX_HOLD_FRACTION
    exit("time_exit")
else:
    # Hold to game resolution (rare; most trades close earlier)
    exit_at_resolution
```

### Inefficiencies we want to eliminate

| Problem | Current behavior | Ideal behavior |
|---|---|---|
| Fixed 17¢ target ignores volatility | Same delta for tight NBA game vs blowout | Target scales with realized σ |
| Time exit is blind to path | Exits at t=0.5 regardless of trend | Holds if still trending up, exits if flatlining |
| Stop loss is symmetric | Same 15¢ stop for all entry prices | Wider stop near p=0.5, tighter near p=0.2/0.8 (Dalen volatility surface) |
| No hazard awareness | Binary trigger each tick | Probabilistic: "P(target hit by t+10min) = 15% → not worth waiting" |
| Static parameters | One tuple across all games | Adaptive to game state (score, time remaining implicitly via price) |

### What "better" means

**Primary:** higher total PnL on held-out (temporal-OOS) trade set.
**Secondary:** lower variance (higher Sharpe), shorter average hold time (more capital turnover), fewer "exit missed" cases where price touched entry + 0.25 but we only captured 0.17.

---

## 2. Core Theoretical Foundation

We treat exit-timing as a **first-passage-time** prediction problem:

> Given market state `s(t)` at time t after entry, what is the probability that price touches level `L` (= entry + δ) at some time `t' ∈ [t, T_end]`?

This is natural for an Ornstein-Uhlenbeck / logit-jump-diffusion model of Polymarket prices (see STRATEGY_D_SPEC §2.2–2.3). The **hazard function** `h(t|s) = lim[Δt→0] P(event in [t, t+Δt] | survived to t) / Δt` directly informs the optimal stopping rule:

**Optimal exit policy (simplified):**
```
Expected remaining upside = ∫_t^T_end  h(τ|s) · (L - price(t)) · exp(−∫_t^τ h(u|s(u)) du) dτ
If current_price > some threshold AND expected_remaining_upside < opportunity_cost:
    EXIT
else:
    HOLD
```

In practice we approximate this with a **learned hazard model** that outputs `h(t|s)` given features, and a **threshold rule** derived from cross-validation.

### References

- Moskowitz (2021) — mean reversion in sports odds
- Dalen (2025) — Black-Scholes for prediction markets (implied volatility surface)
- Hashimoto et al. (2025) — OU process for betting odds
- López de Prado (2018) — triple-barrier method
- Ishwaran & Kogalur (2008) — `RandomSurvivalForest`
- Katzman et al. (2018) — DeepSurv (neural hazard model — **not used** due to small-N)

---

## 3. Data Pipeline

### Inputs

| Source | What | Freq | Coverage |
|---|---|---|---|
| `sports_backtest.sqlite` (VPS) | Per-token price trajectories | 1-min (Pass 2) or 10-min (Pass 1) | 86,947 Pass 2 moneyline markets (10,927 NBA ML) |
| `sports_backtest.sqlite` | Pinnacle no-vig + Elo ratings | Pre-game | 5,718 Pinnacle + 9,552 Elo |
| `research_d/prepare.py` | Backtest harness → TradeResult | One per trade | ~15K NBA trades expected |

### Pipeline stages

```
1. Backtest (existing prepare.py)
   ├── Runs champion_v1 params, logs each trade.
   └── Modified to optionally emit per-tick trajectory + features.
        (Opt-in via return_trajectories=True; default False.)

2. Trajectory → per-timestep expansion (NEW build_exit_timing_set.py)
   For each trade with trajectory [(t_0, p_0), (t_1, p_1), ..., (t_N, p_N)]:
     For each timestep i in [0, k_exit]:
       emit row = (trade_id, t_i, features(i), event_at_i, time_to_event, censored)

3. Feature matrix construction (in build_exit_timing_set.py)
   X[row] = rolling vol, momentum, distance-to-GB, time-elapsed, static entry features
   y_survival = (event_indicator, time_to_event)
   → Save as parquet (research_d/data/exit_timing_set_v1.parquet)

4. Train survival model (NEW train_exit_model.py)
   XGBoost with objective='survival:aft' or 'survival:cox'
   Temporal CV by game_date
   Output: model + C-index + calibration plot

5. Backtest learned policy (NEW eval_exit_policy.py)
   For each OOS trade, simulate entry with champion_v1 params.
   At each tick, query model: h(t|s). Apply threshold rule.
   Compare PnL vs fixed-rule baseline.
```

### Data scale (expected)

- ~15,000 NBA trades × avg ~40 ticks per trade (post-entry until exit or game end) = **~600,000 rows** in expanded per-timestep set. This is enough for XGBoost to learn patterns without overfitting, given ~20 features.

---

## 4. Feature Engineering

### Features per timestep `t` (relative to entry `t_0`)

All features **MUST** be computable from price trajectory + static entry context. **No lookahead** — features at time `t` use only `prices[0..t]`.

#### Price-state features

| Feature | Formula | Rationale |
|---|---|---|
| `price_now` | `prices[t]` | State — model needs current price |
| `price_return_since_entry` | `(prices[t] - entry_price)` | Unrealized PnL trajectory |
| `price_return_pct` | `(prices[t] - entry_price) / entry_price` | Normalized return |
| `max_since_entry` | `max(prices[0..t])` | Peak we've seen — informs regret |
| `min_since_entry` | `min(prices[0..t])` | Trough — informs drawdown |
| `max_minus_now` | `max_since_entry - prices[t]` | Current drawdown from peak |
| `drawdown_from_entry` | `entry_price - min_since_entry` | Worst case so far |
| `touched_target_frac` | `max_since_entry / (entry + delta)` | Did we almost touch GB? |

#### Volatility features (rolling)

| Feature | Window | Formula |
|---|---|---|
| `vol_5min` | 5 ticks | `std(prices[t-4..t])` |
| `vol_15min` | 15 ticks | `std(prices[t-14..t])` |
| `vol_60min` | 60 ticks | `std(prices[t-59..t])` |
| `vol_ratio_5_60` | — | `vol_5min / vol_60min` (regime shift signal) |
| `realized_range_15min` | 15 ticks | `max - min` over window |

#### Momentum features

| Feature | Window | Formula |
|---|---|---|
| `slope_5min` | 5 ticks | OLS slope of prices vs time |
| `slope_15min` | 15 ticks | OLS slope |
| `returns_autocorr_5` | 5 ticks | Autocorrelation of 1-tick returns |
| `pct_up_last_10` | 10 ticks | Fraction of ticks with `price[i] > price[i-1]` |

#### Time features

| Feature | Formula |
|---|---|
| `elapsed_frac` | `t / T_expected` where `T_expected = game_duration_hours * 60` |
| `elapsed_ticks` | `t` (integer tick count) |
| `remaining_frac` | `1 - elapsed_frac` |
| `at_pregame` | 1 if `t < tipoff_estimated`, else 0 (TBD — may not have tipoff timestamp) |

#### Distance-to-barrier features

| Feature | Formula |
|---|---|
| `gb_distance` | `(entry_price + delta) - prices[t]` — how far to GB target |
| `gb_distance_norm` | `gb_distance / vol_15min` — in sigmas |
| `sl_distance` | `prices[t] - (entry_price - sl_delta)` — distance to stop |
| `sl_distance_norm` | `sl_distance / vol_15min` |
| `gb_already_touched` | 1 if `max_since_entry >= entry + delta`, else 0 |

#### Static entry features (constant across `t` for a trade)

| Feature | Source |
|---|---|
| `model_prob_entry` | Elo+Pinnacle ensemble at entry |
| `edge_at_entry` | `model_prob - entry_price` |
| `entry_price_level` | `entry_price` |
| `pinnacle_available` | 1 if Pinnacle line matched for game |
| `elo_diff_entry` | Rating gap (team_a − team_b) |
| `ensemble_disagreement_entry` | `|elo_prob − pinnacle_prob|` |
| `n_prices_available` | Total trajectory length (liquidity proxy) |
| `game_day_of_week` | 0-6 |
| `game_month` | 1-12 |
| `season_phase` | categorical: early/mid/late_regular, playoffs |

**Total: ~30 features.** Deliberately conservative — more features risk overfitting on our 600K row set.

### Features DELIBERATELY excluded (leakage or low-value)

| Excluded | Reason |
|---|---|
| `price[t+1..]` | Lookahead leakage |
| `max_future`, `min_future` | Lookahead |
| `time_of_green_book` | Label, not feature |
| `realized_pnl` | Lookahead |
| `bid`, `ask` | Not populated in current data (D3 finding) |
| `volume_1m` | Not populated |
| Pinnacle mid-game moves | Not collected in DB |
| News / injury | Not collected |

**If Cesta A (bid/ask pipeline) completes**, we add: `spread_bps`, `depth_imbalance`, `volume_1m_momentum`. This is a planned upgrade path — features fit cleanly into current architecture.

---

## 5. Label Definition

### Survival label: (event, time_to_event)

For each trade, GB-target is `entry_price + delta`.

For each timestep `t` in trajectory (t ∈ [0, exit_index]):

```python
if max(prices[t+1..exit_index]) >= entry_price + delta:
    event = 1
    time_to_event = (first time j > t where prices[j] >= target) - t
else:
    event = 0  # censored
    time_to_event = exit_index - t  # censored at exit
```

### Alternative labels (for ablation)

| Label | Definition | Why consider |
|---|---|---|
| **GB_hit** (primary) | Green-book target crossed | Replicates current behavior, pure first-passage |
| **Optimal_exit** | argmax over t of (prices[t] - entry_price) | "Oracle" — hardest to learn but most informative |
| **Profitable_exit** | 1 if exit PnL > 0 | Binary — coarser signal |

For this iteration: **GB_hit** (simplest, matches current rule).

---

## 6. Model Family & Training

### Choice: XGBoost survival

**Why:**
- Already in our env (`xgboost 3.2.0`)
- Native `survival:aft` (accelerated failure time) and `survival:cox` objectives
- Handles censoring correctly
- Fast training on 600K rows
- SHAP-interpretable
- Proven in sports betting literature (Hubáček et al. 2021 — GBDT > DL on betting)

### Objective: `survival:aft`

Lifetime `T` modeled as:
```
log(T) = feature_score + σ · error
```
where `error ~ N(0, 1)` (normal AFT) or `logistic` etc.

Per xgboost docs:
- `aft_loss_distribution`: `normal`, `logistic`, `extreme` (Weibull)
- `aft_loss_distribution_scale`: σ (learned)

Preferred: **Weibull AFT** — matches hazard-rate semantics for "first passage" physics.

### Hyperparameters (initial, to tune)

```python
params = {
    "objective": "survival:aft",
    "aft_loss_distribution": "extreme",      # Weibull
    "aft_loss_distribution_scale": 1.0,
    "max_depth": 5,                          # shallow — avoid overfit
    "eta": 0.03,                             # conservative
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "min_child_weight": 20,                  # regularize
    "tree_method": "hist",
    "eval_metric": "aft-nloglik",
}
```

### Monotonic constraints

Per the hazard-rate physics:
- `gb_distance` ↑ → hazard ↓ (longer to event, more likely censored) — constraint `-1`
- `vol_15min` ↑ → hazard ↑ (higher volatility, faster first passage) — `+1`
- `elapsed_frac` ↑ → hazard ↑ (less time remaining, censoring more likely) — `+1` for the AFT target log(T), this means **longer log(T) predicted** for larger elapsed. Inverted sign.

**We WILL apply monotonic constraints** to prevent the model from learning spurious relationships (e.g., "long elapsed → hazard low" because training data has survivor bias).

### Training split

Temporal, not random:
- Train: first 60% of games by `game_date`
- Val:   next 20%
- Test:  last 20%

Strict: no trade from test's game leaks into train/val (grouping by `game_id`).

### Validation metrics

- **C-index** (concordance): standard survival AUC. Target ≥ 0.60 for meaningful signal.
- **Integrated Brier Score**: calibration across horizons.
- **Brier(t)**: per-horizon Brier (e.g., at t=30min, 60min, 90min).
- **Calibration plot**: predicted hazard vs observed event rate.

### Baseline to beat

**Fixed rule hazard baseline:** at every timestep, hazard = (GB hit in future? / remaining time). This is the "current champion" policy expressed as implicit hazard.

Our model must beat this by C-index > 0.60 on test set.

---

## 7. Policy → Backtest Integration

### Learned exit policy

```python
def should_exit(state_t, model, threshold=0.XX):
    """Return True if model says 'exit now'."""
    features = extract_features(state_t)
    hazard_t = model.predict_hazard(features, horizon=tau)  # tau ~ next 10 min
    
    # Exit conditions:
    #   (a) already profitable AND forecast hazard < threshold → take profit
    #   (b) elapsed_frac > 0.9 → time exit
    #   (c) price > entry + delta → GB hit (original rule as safety net)
    
    return hazard_t < threshold  # simplified
```

Threshold tuned on val set — choose to maximize backtest PnL on val.

### A/B evaluation

On test set only:
- **Arm A (baseline):** fixed champion_v1 rule (GB=0.17, SL=0.15, MHF=0.50)
- **Arm B (learned):** learned policy

Metrics to compare:
- Total PnL
- Sharpe
- GB-hit rate
- Avg hold time
- Max drawdown
- "Missed upside" (frac of trades where max > GB target but we didn't exit optimally)

**Promotion criterion:** B must beat A on test PnL **AND** not worsen Sharpe by > 10% **AND** not increase max DD by > 20% (relative). Otherwise we don't ship it.

---

## 8. Integration With Variant System (Project PARALLEL)

Once trained and validated, the learned policy ships as a **challenger variant**:

```yaml
# arbo/config/variants/d/ch_ml_exit_v1.yaml
variant_id: ch_ml_exit_v1
strategy: D
status: incubate
parent_variant: champion_v1
notes: "ML-driven adaptive exit (replaces fixed GB_DELTA + MHF). Trained 2026-XX-XX."

params:
  MIN_EDGE: 0.16
  MAX_EDGE: 0.25
  MIN_PRICE: 0.20
  MAX_PRICE: 0.65
  # Exit rule — NEW: ML model reference
  EXIT_POLICY: "ml_hazard_v1"
  EXIT_MODEL_PATH: "arbo/data/models/strategy_d_exit_hazard_v1.ubj"
  EXIT_HAZARD_THRESHOLD: 0.05   # tuned
  # Safety nets (preserved from champion):
  GREEN_BOOK_DELTA: 0.17        # fallback if model disabled
  STOP_LOSS_DELTA: 0.15
  MAX_HOLD_FRACTION: 0.50
```

And `strategy_d_core.py::check_exits` reads `EXIT_POLICY` and dispatches:

```python
exit_policy = self._p("EXIT_POLICY") or "fixed"
if exit_policy == "ml_hazard_v1":
    exit_reason = self._ml_exit_check(pos, price, model)
else:
    exit_reason = self._fixed_exit_check(pos, price)
```

**Crucially:** this is a **variant** — it runs only for positions tagged with `variant_id=ch_ml_exit_v1`. Champion positions use the fixed rule. Zero risk to existing live behavior.

---

## 9. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Lookahead leakage in features | Medium | Unit-test each feature; assert `features[t]` only uses `prices[0..t]` |
| Survivor bias in training | High (censored trades) | Use proper survival objective; don't filter out censored |
| Distribution shift (NBA season) | High | Temporal CV; Page-Hinkley drift monitor (already in Watchdog) |
| Model too confident → over-exit | Medium | Conformal prediction intervals on hazard; wider → smaller position |
| Overfitting on 600K rows / 30 features | Low | Shallow trees (depth 5), min_child_weight=20, regularized |
| Regime ≠ validation sample | Medium | Test on last 20% — truly OOS; accept only if both win and reject rate improve |
| Silent signal hacking | Medium | Log predictions + features for every OOS trade; reconcile to actuals |

---

## 10. Success Criteria

### Must-have before shadow deploy
1. **Pipeline correctness:** unit tests for every feature; no lookahead detected
2. **C-index ≥ 0.60** on test set (15% better than random 0.50)
3. **Test PnL ≥ baseline PnL** on the SAME held-out trades, under the same entry rule

### Must-have before live deploy (canary)
1. **4 weeks shadow evaluation** on real-time Polymarket feed
2. **P(better) ≥ 0.75** via block bootstrap (Project PARALLEL promotion engine)
3. **CEO approval via Slack** button
4. **Auto-revert triggers armed:**
   - Rolling 50-trade WR < 45% → pause
   - Rolling Sharpe < 1.0 (annualized) → pause
   - Max DD > 20% → pause
   - Page-Hinkley drift detected → pause

### Nice-to-have
1. SHAP feature importance dashboard
2. Compare 3 candidates: Weibull AFT, Cox, RandomSurvivalForest
3. Ablation: which feature groups contribute most?

---

## 11. Open Questions (before implementation)

1. **Tick interval:** NBA Pass 2 gives ~1-min ticks. For ~90-minute game that's ~90 timesteps per trade. Is the per-tick grid what we want, or should we downsample to 5-min to reduce training set size?
   - **Decision:** start with 1-min (~600K rows is manageable), downsample only if training is slow.

2. **Trajectory truncation:** trajectory goes until exit_idx. For censored trades (never hit GB), exit_idx might be game end — very long. Truncate at MAX_HOLD_FRACTION × game_duration?
   - **Decision:** Truncate at same threshold as champion rule (0.5 × game_duration). Downstream policy won't hold past that anyway.

3. **Horizon (τ) for hazard prediction:** predict `h(t|s)` for what τ? 1 min? 10 min? Total remaining?
   - **Decision:** predict total remaining hazard (time-to-event). For policy, derive short-horizon from AFT distribution assumption.

4. **Multiple trades per game:** same market may have multiple trajectories (YES + NO sides if BOTH_SIDES=True). Independent observations or grouped?
   - **Decision:** independent rows but tag with `trade_id` so we can group if needed for CV.

5. **What about stop-loss learning?** Current design focuses on profitable exits (green book). Should model also learn STOP decisions?
   - **Decision:** v1 learns only profit-take. SL stays fixed (safety net). v2 could learn SL.

6. **Calibration:** hazard outputs need to be calibrated for threshold decisions.
   - **Decision:** after XGBoost survival:aft training, fit isotonic regression on hazard → event indicator to calibrate.

---

## 12. Implementation Timeline

| Day | Task | Owner |
|---|---|---|
| 1 | Design doc review ← **YOU ARE HERE** | CEO |
| 1 | Extend prepare.py to emit trajectories (opt-in) | AI |
| 1-2 | build_exit_timing_set.py + unit tests | AI |
| 2 | train_exit_model.py (XGBoost AFT + Cox) | AI |
| 2 | Run on VPS with full NBA data | AI |
| 3 | eval_exit_policy.py — backtest compare | AI |
| 3 | Results analysis + learning log | AI |
| 4 | Decision: ship as variant / iterate / kill | CEO |

---

## 13. What We're NOT Doing (yet)

Per research findings and framework constraints:

1. **No neural networks.** 600K rows may be enough for LSTM, but XGBoost is strictly better with current sample size and more interpretable. Revisit at >5M rows.
2. **No RL / policy gradient.** Yang & Malik 2024 showed RL needs tens of thousands of trades for convergence; we have ~15K. XGBoost hazard → simple threshold policy gets 80% of RL value at 10% complexity.
3. **No entry meta-labeler right now.** D2 finding: current entry features are too thin (Test AUC 0.51). Exit features are richer because trajectory is rich. Entry stays rule-based.
4. **No Pinnacle lag model right now.** Needs Pinnacle time-series not yet collected. Parallel Cesta A/C work.
5. **No D2 (overreaction fade) / D3 (cascade).** Spec §3.3–3.4. Out of scope for this iteration.

---

## 14. Glossary

| Term | Definition |
|---|---|
| **GB / green book** | Selling a position for guaranteed profit before game resolution |
| **Hazard rate** | P(event occurs in [t, t+dt] \| no event before t) / dt |
| **AFT** | Accelerated Failure Time — survival model variant where features scale log(T) |
| **C-index** | Concordance index — rank-based metric for survival models (0.5 = random, 1.0 = perfect) |
| **First passage time** | Stopping time τ = inf{t : X(t) ≥ L} for process X and level L |
| **Triple barrier** | López de Prado labeling: upper/lower/time — first hit wins |
| **Censored** | Observation where event didn't occur within observation window |
| **Survival curve** | S(t) = P(T > t) — probability of surviving past t |

---

## Next Step

**Review this design.** Flag any concerns.
If accepted → I proceed with Section 12 Day 1-2 (prepare.py + build_exit_timing_set.py + unit tests).

---

## 15. Shadow-Exit Logger (shipped 2026-04-21)

Added after initial design — shipping model v2 as `status: shadow`
variant would have produced zero live evidence, because shadow status
skips the variant's decision path entirely. The logger closes that gap
without changing trading behavior.

### 15.1 Purpose

For every open NBA Strategy D position, run the ML model (v1 = v2 in
our current training pipeline; this section uses "v1" to match the
shipped model filename) in parallel with the fixed champion rule and
log paired decisions. After 4+ weeks of data, run paired-sample
P(better) bootstrap on live positions to validate or reject v2 exit
ahead of any canary promotion.

### 15.2 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ StrategyDNba.check_exits() — runs every 30-60s on each position  │
└──────────┬───────────────────────────────────────────────────────┘
           ↓
    ┌──────────────────────┐
    │ fixed rule evaluates │ ← actual exit decision (unchanged)
    │ gb_hit / sl_hit /    │
    │ time_exit            │
    └──────────┬───────────┘
               ↓
    ┌────────────────────────────────┐
    │ IF SHADOW_EXIT_LOG_ENABLED:    │
    │   query shadow model in        │ ← PARALLEL, passive
    │   parallel (not affecting exit)│
    │   → (should_exit, pred_log_t)  │
    └──────────┬─────────────────────┘
               ↓
    ┌─────────────────────────────────┐
    │ IF ml_should_exit AND first time│
    │   for this (token_id, side):    │ → event_type='ml_first_exit'
    │   async INSERT to               │   (at most 1 per trade)
    │   shadow_exit_decisions         │
    └─────────────────────────────────┘
               ↓
    ┌──────────────────────────────────┐
    │ IF real rule fires exit:         │
    │   async INSERT real_exit row     │ → event_type='real_exit'
    │   (pairs with any ml_first_exit) │   (exactly 1 per trade)
    └──────────────────────────────────┘
```

### 15.3 Key invariants (enforced by tests)

1. **Logger never changes exit decision.** Real rule rules.
2. **Logger survives model load failure silently.** Telemetry cannot
   affect trading.
3. **Dedup per (token_id, side).** At most 2 rows per trade
   (ml_first_exit + real_exit).
4. **Async insert.** DB stall never blocks check_exits.

### 15.4 Enabling / disabling

```python
# In strategy subclass (e.g. strategy_d_nba.py):
SHADOW_EXIT_LOG_ENABLED = True
SHADOW_EXIT_MODEL_PATH = "arbo/data/models/strategy_d_exit_v1.ubj"
SHADOW_EXIT_THRESHOLD = 6658.3
```

Defaults on `StrategyDCore` are `False`/`None`/`6658.3` — zero cost
when off. Currently enabled ONLY on NBA (v1 model trained on NBA
only). Do NOT enable on UFC/EPL until we have sport-specific trained
models.

### 15.5 Schema (alembic 016)

See `alembic/versions/016_shadow_exit_decisions.py`. Key columns:

| Column | Type | Note |
|---|---|---|
| strategy | varchar(32) | D / D_UFC / D_EPL |
| token_id, side | varchar(80) + varchar(8) | position identity |
| tick_ts | timestamptz | when decision was computed |
| event_type | varchar(20) | ml_first_exit \| real_exit |
| ml_should_exit | boolean | model's decision |
| ml_pred_log_t | numeric | AFT output |
| ml_threshold | numeric | policy threshold (for reproducibility) |
| real_exit_reason | varchar(32) | green_book / stop_loss / time_exit |
| real_exit_price | numeric(6,4) | actual exit price (when event_type=real_exit) |

### 15.6 Analysis query (P(better) proxy)

See `LEARNINGS.md` D8 for full SQL. Core idea: for each (token_id,
side) pair, group ml_first_exit + real_exit rows, compute hypothetical
ML PnL vs real PnL, then paired-bootstrap.

### 15.7 Promotion criteria (tentative, review when N≥50)

- Paired P(learned > real) ≥ 0.75 on ≥ 50 paired trades
- Rolling 30-day PnL(ML) > PnL(real) × 1.10
- Max drawdown under ML ≤ 1.15 × max DD under real
- Verified no systematic gaming of the threshold

If all pass → propose canary promotion (status: shadow → incubate →
challenger) per Project PARALLEL Phase 4 framework.

### 15.8 Files

| File | Purpose |
|---|---|
| `alembic/versions/016_shadow_exit_decisions.py` | DB schema |
| `arbo/strategies/strategy_d_core.py` | `_get_shadow_exit_policy`, `_log_shadow_exit_decision`, `_shadow_insert_async` + hooks in `check_exits` |
| `arbo/strategies/strategy_d_nba.py` | Enables shadow logging for NBA |
| `arbo/tests/test_shadow_exit_logger.py` | 7 invariant tests |
| `LEARNINGS.md` D8 | Narrative + SQL analysis query |

