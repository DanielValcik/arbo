---
name: optimize
description: MANDATORY framework for ANY Arbo strategy optimization decision — parameter changes, autoresearch validation, live performance analysis, regime questions, kill decisions. Enforces rigorous methodology (López de Prado, Bailey, Harvey) before any live param modification. Auto-triggers on questions like "should we change X", "autoresearch says X but live says Y", "why is PnL flat", "should we deploy this", "is strategy working". Applies to ALL strategies (B3 5-min, B3_15M, B2, C, C2, D, D_UFC, D_EPL, future). Manual invocation: /optimize [strategy].
argument-hint: "[strategy_name] (optional: B3, B3_15M, B2, C, C2, D)"
allowed-tools: Read, Bash, Grep, Glob, Write, Edit
---

# Arbo Strategy Optimization — Mandatory Methodology

**Manual invocation**: `/optimize` or `/optimize B3_15M` — user types this to explicitly run framework check. Without argument, ask user which strategy to evaluate.

**Auto-invocation**: Claude MUST invoke this skill when user asks about param changes, deploying, autoresearch validation, strategy performance, regime issues, or kill decisions for any Arbo strategy. Ad-hoc decisions destroy strategies; rigorous methodology preserves them.

## Three-Document System

**Theory + decision rules (v2.0, includes Rapid Mode)**:
```
/Users/dnl.vlck/Arbo/docs/STRATEGY_OPTIMIZATION_FRAMEWORK.md
```
Methodology (López de Prado, Bailey, Harvey) + Rapid Mode §11 (champion-challenger, BO, MAB, drift). Update quarterly, re-read before major decisions.

**Rapid Mode deep dive**:
```
/Users/dnl.vlck/Arbo/docs/RAPID_MODEL_DISCOVERY.md
```
538-line study: how hedge funds accelerate 5-10× via parallel variants, Thompson Sampling, Optuna BO, Page-Hinkley drift. Consult when proposing rapid exploration.

**Empirical record (append-only log)**:
```
/Users/dnl.vlck/Arbo/docs/STRATEGY_OPTIMIZATION_LEARNINGS.md
```
Every decision logged: date, data, hypothesis, framework gates checked, evidence, outcome. **Read BEFORE any new decision** (check if similar decision was made before). **Append AFTER every decision** (template in the doc).

## Step 0: Check Prior Learnings (do this FIRST)

Before any analysis, read the relevant section of `STRATEGY_OPTIMIZATION_LEARNINGS.md`:
- Has this exact decision been made before? → what was the outcome?
- Is this pattern in the "Recurring Anti-Patterns" section? → pause and review
- Is this strategy's baseline + past decisions visible in its section?

```bash
grep -A 30 '## Section [0-9]*: Strategy <NAME>' \
  /Users/dnl.vlck/Arbo/docs/STRATEGY_OPTIMIZATION_LEARNINGS.md
```

Skipping Step 0 = repeating mistakes. History is evidence, use it.

## Step 1: Identify the Decision Type

Which of these is the user asking about?

| Situation | Key action |
|---|---|
| "Should we change parameter X?" | → Step 2 (Minimum N check) |
| "Autoresearch says X, live says Y" | → Step 3 (Bayesian shadow+live weighting) |
| "Why is PnL flat / stagnating?" | → Step 4 (regime detection + payout asymmetry check) |
| "Should we deploy this autoresearch result?" | → Step 5 (validation gates: PBO, DSR, plateau) |
| "Should we kill this strategy?" | → Step 6 (kill criteria) |
| "Live is different from paper" | → Step 7 (execution reality gap) |
| "Autoresearch found a better config" | → Step 5 + §10.4 decision tree |
| **"How do we speed up discovery / test multiple ideas?"** | → **Step 1.5 (Rapid Mode routing)** |
| "Can we run multiple variants in parallel?" | → Step 1.5 |
| "Want to do BO / bandit / champion-challenger" | → Step 1.5 |

## Step 1.5: Serial vs Rapid Mode Routing

Every decision goes through EITHER serial framework (§1-10) OR rapid mode (§11). Use this table:

| Situation | Mode | Why |
|---|---|---|
| Tuning ONE param on LIVE strategy | **Serial** | Safety, disciplined gates |
| Exploring multiple NEW hypotheses | **Rapid** | Parallel = 5-10× faster via pairing |
| Regime change suspected on production | **Serial first** | Don't compound errors |
| Building new strategy from scratch | **Rapid** | Many ideas die fast/cheap |
| Single-variant capital sizing | **Serial** | High-impact, full lifecycle |
| Capital allocation across variants of SAME strategy | **Rapid (MAB)** | Bandits dominate fixed alloc |
| Replacing grid sweep in autoresearch | **Rapid (Optuna BO)** | 30-100× compute speedup |

**Both modes share end-gates**: DSR > 0.95 for promotion, PBO < 0.3 for CPCV configs, revert triggers armed, log to LEARNINGS.md.

**Rapid Mode techniques available** (detail in framework §11):
1. **Champion-Challenger**: 2-8 parallel variants, paired-sample testing
2. **Shadow Orchestrator**: fork signal to all variants, no capital risk
3. **Bayesian Optimization (Optuna)**: replace grid sweeps, 50-100 trials
4. **Multi-Armed Bandit (Thompson Sampling)**: dynamic capital allocation
5. **Page-Hinkley Drift Detector**: fires 3-5× faster than drawdown trigger
6. **Composite Reward**: 0.4×dir_60s + 0.6×norm_pnl → 3-5× faster learning
7. **Block Bootstrap (Politis-Romano)**: correct statistical tests
8. **HRP Ensemble**: combine top-3 variants, quorum rule

**Dashboard requirement (MANDATORY if using Rapid Mode)**: every strategy with ≥ 2 active variants MUST have a "Variant Leaderboard" card. Fields:
- Per-variant rolling 50-trade PnL
- Per-variant WR + 95% block-bootstrap CI
- Per-variant DSR + posterior-median Sharpe
- Current capital allocation (from MAB if live, equal if shadow)
- Drift detector status (green/yellow/red)
- Last-promoted / last-retired variant + date
- Composite reward trend

**File**: `arbo/dashboard/variant_leaderboard.py` component, per-strategy tab integration.

**Enforcement**: if user asks to run multiple variants WITHOUT the leaderboard card being in place → REFUSE and require implementation first. Variants without visibility = invisible failures.

**If you're not sure which applies**, read the full framework:
```
/Users/dnl.vlck/Arbo/docs/STRATEGY_OPTIMIZATION_FRAMEWORK.md
```

## Step 2: Minimum N Check (before ANY param change)

Consult the table in §10.1 of the framework:

| Change type | Min live N required |
|---|---|
| Risk reduction (smaller size, tighter cap) | 0 (always allowed) |
| Filter loosening (lower min_edge, etc.) | 100 |
| Filter tightening | 50 |
| Sizing parameters (kelly_cap, etc.) | 200 |
| Core model (sigma_scale, threshold) | 300 |
| Strategy-level (entry signal, label rule) | 500 |

**If live N is below threshold: REFUSE the change. Explain why.**
Standard error of WR at N=12 is ±14.4pp. You literally cannot distinguish skill from luck.

```bash
# Query live N for a strategy (example for B3_15M):
ssh arbo-dublin "sudo -u arbo psql -d arbo -c \"
SELECT COUNT(*) FROM paper_trades
WHERE strategy='B3_15M'
  AND (trade_details->>'live_fill_status') IN ('filled','partial')
  AND trade_details->>'live_exit_status' IN ('resolution','filled','partial')
  AND placed_at >= '<deploy_date>';\""
```

## Watchdog Integration (B3 + B3_15M have autonomous watchdogs)

Two trading strategies have an autonomous **Watchdog daemon** (`arbo/core/b3_watchdog.py`) running 24/7:
- `self._b3_watchdog` — for B3 5-min, separate AdaptiveConfig
- `self._b3_15m_watchdog` — for B3_15M, separate AdaptiveConfig

### What the Watchdog does autonomously (within framework limits)

Every 6 hours (or 30 trades threshold, `_MIN_TRADES_FOR_EVAL=30`) it:
1. Queries `fetch_b3_metrics(strategy=<name>)` for rolling live stats
2. Runs `_detect_anomalies` (T1–T20 triggers: WR drop, ECE drift, consecutive losses, regime breaks)
3. On anomaly: calls **Gemini 2.5 Flash** with context → APPLY | REVERT | ESCALATE | MONITOR
4. Writes parameter changes to `AdaptiveConfig` (strategy reads via `.get(param, default)`)
5. Auto-reverts after 50 trades if WR drops > 5pp

### 3-Tier Bounds (defined in `arbo/core/adaptive_config.py`)

| Tier | Autonomy | Parameters |
|---|---|---|
| Tier 1 | Autonomous | LIVE_MAX_VELOCITY, LIVE_MAX_DIR_DELTA, LIVE_MIN_EDGE, LIVE_MIN_BTC_MOVE, POSITION_PCT, EDGE_SCALING, LIVE_MAX_BTC_MOVE (15m), LIVE_MAX_MARKET_GAP (15m) |
| Tier 2 | Autonomous + CEO flag | SIGMA_SCALE, ENTRY_THRESHOLD, MIN/MAX_ENTRY_MIN |
| Tier 3 | Never autonomous | MAX_BET_SIZE, DAILY_LOSS_LIMIT, LIVE_MAX_FILL_PRICE, REQUIRE_CHAINLINK, execution_mode, capital allocation |

### How Watchdog relates to this framework

| Aspect | Watchdog | This skill (framework) |
|---|---|---|
| Scope | Tier 1 autonomous, Tier 2 escalated | Tier 3 + architectural decisions |
| Frequency | 24/7, every 6h / 30 trades | On-demand (human decisions) |
| Evidence | Gemini analysis of real-time metrics | Bootstrap P(better), CPCV, DSR, PBO |
| Min N | 30 (baseline) | 50-300 depending on change type |
| Decision authority | Autonomous within bounds | Human, with framework enforcement |
| Logs to | AdaptiveConfig audit log + Slack | `STRATEGY_OPTIMIZATION_LEARNINGS.md` |

### Watchdog ≠ skip the framework

**Watchdog changes are ALSO subject to framework tracking**. Every autonomous Watchdog change should:
1. Appear in `adaptive_config.get_change_log()` — AdaptiveConfig audit
2. Be summarized in monthly learnings log review (append Watchdog summary to LEARNINGS.md)
3. If Watchdog consistently fails (3+ reverts in a row → observation mode), that's a signal the strategy is broken — escalate to framework-level kill criteria (§8.4)

### Framework-Watchdog divergence (known)

Watchdog's `_MIN_TRADES_FOR_EVAL=30` is lower than framework's min N=50-100 for filter changes. Reasoning:
- Watchdog only moves params within Tier 1 bounds (small, safe range)
- Framework gates apply to bigger changes (outside Tier 1 bounds, or Tier 2/3)
- **Rule**: trust Watchdog for Tier 1. For anything Tier 2+, use the framework.

### When user asks "what did the Watchdog decide?"

```bash
# Recent Watchdog decisions
ssh arbo-dublin "sudo journalctl -u arbo.service --since '24h ago' 2>&1 | \
  grep -E 'watchdog_verdict|adaptive_config_set|adaptive_config_reverted' | tail -20"

# Current AdaptiveConfig overrides (both strategies)
ssh arbo-dublin "sudo journalctl -u arbo.service --since '30 min ago' 2>&1 | \
  grep -E 'adaptive_config.*status' | head -5"
```

## Step 3: Bayesian Shadow + Live Weighting

When live N is small and shadow N is large:

```
weight_shadow = N_shadow / (N_shadow + N_live)
weight_live   = N_live   / (N_shadow + N_live)
```

**Canonical example**: shadow=144, live=12 → shadow 92.3%, live 7.7%.
Live N=12 contributes <8% of evidence. It cannot override shadow statistically.

**Exception**: If regime change detected (see Step 4), downweight shadow toward zero.

## Step 4: Regime Detection (before blaming params)

Before concluding "strategy is broken" or "params need changing", check for regime change:

1. **Compute realized BTC vol** on shadow period vs live period
   - Ratio > 1.5× → significant regime difference
2. **Check PSI on key features** (sigma, velocity, dir_delta):
   - PSI < 0.10 — no significant shift
   - 0.10–0.25 — moderate, investigate
   - > 0.25 — major shift, consider regime-aware re-validation
3. **Check payout asymmetry** for flat PnL:
   - For binary markets, breakeven WR = entry_price / 1.0
   - Fill at 0.73 → breakeven WR 73%
   - If current WR < breakeven → strategy has NEGATIVE expected value

**If regime changed**: do NOT re-optimize on small post-regime sample. Wait for ≥ 200 trades in new regime. Premature re-optimization compounds losses.

## Step 5: Autoresearch Validation Gates

Before deploying ANY autoresearch winner, all must pass:

- [ ] **CPCV** (Combinatorial Purged k-fold CV) used, not standard k-fold
  (Reference: López de Prado, *AFML* Ch. 12)
- [ ] **PBO** (Probability of Backtest Overfitting) < 0.5 (target < 0.3)
- [ ] **DSR** (Deflated Sharpe Ratio) > 0.95 with N_trials documented
- [ ] **Plateau check**: Sharpe at ±10% and ±20% param perturbation ≥ 80% of peak
- [ ] **MinTRL** validation for observed Sharpe
- [ ] **Economic thesis** still holds (why does edge exist? who's on other side?)

**If ANY check fails → reject the result. Even a "99% OOS" win is noise if not validated.**

Math for DSR:
```
DSR = Φ( (SR_observed − SR_0) · √(T−1) /
        √(1 − γ̂₃·SR_0 + ((γ̂₄−1)/4)·SR_0²) )
```
where SR_0 = expected max SR under null given N_trials (False Strategy Theorem).

## Step 6: Kill Criteria

Kill (don't pause, don't re-optimize — KILL) a strategy when:

- Live DD > 2× backtest 95th-percentile DD
- 6 consecutive months underperforming paper baseline
- Regime change is structural (exchange/oracle policy change, protocol update)
- Economic thesis invalidated (why it worked is no longer true)
- DSR drops below 0.50 on rolling re-evaluation

**Killing is a feature, not a failure.** Asness/AQR: only ~5–10 robust factors survive decades of search. Most strategies should die.

## Step 7: Execution Reality Gap Diagnosis

If live Sharpe < 50% of paper Sharpe AND N ≥ 100:

1. **Spread cost** — does avg spread × 2 (entry+exit) explain gap? → fix paper engine
2. **Fill rate** — how often do PostOnly orders fail? → paper assumes 100%, reality 70-80%
3. **Adverse selection** — maker fills happen when trend reverses (winner's curse)
4. **Slippage** — your order moves the market

**Arbo-specific**: paper engine has the spread inversion bug (`paper_vs_live_pricing_bug.md`). Apply spread haircut to all paper Sharpes until fixed.

## Step 8: Evidence Threshold for Change

If Min N check passes, compute:

**Bootstrap test**:
1. Take last 200 live trades
2. Resample with replacement 1000 times
3. For each resample, compute hypothetical PnL under (a) current params, (b) proposed params
4. P(better) = fraction where (b) > (a)

**Threshold**: change only if P(better) ≥ threshold from §10.1 table (e.g., 0.75 for filter loosening).

## Step 9: Revert Armor (set BEFORE deploying)

Before any live param change, define explicit triggers:
- 50-trade rolling Sharpe drops > 50% from baseline → REVERT
- 50-trade rolling WR drops > 5pp from baseline → REVERT
- Cumulative PnL since change negative AND below 1σ band → REVERT

## Step 9b: MANDATORY — Log to Learnings Doc

**Every decision (change, hold, revert, kill) must be appended to**:
```
/Users/dnl.vlck/Arbo/docs/STRATEGY_OPTIMIZATION_LEARNINGS.md
```

Use the template in the doc header. Required fields:
- Date, strategy
- Observation (what triggered the decision)
- Data at decision time (live N, shadow N, Bayesian weights, current PnL/WR)
- Hypothesis
- Framework gates checked (✓/✗ each)
- Decision (what we did)
- Evidence basis (P(better), DSR, PBO, regime detection)
- Revert triggers armed
- Outcome (updated after N trades)

**After outcome is known** (typically after 50-100 trades post-decision):
- Update the same entry with measured effect
- Tag: `KEPT` / `REVERTED` / `INCONCLUSIVE` / `ONGOING`
- Note the lesson for future

**If a pattern of failed decisions accumulates on a strategy → add to "Recurring Anti-Patterns" section.** This triggers immediate pause on the next similar decision.

Also append to `parameter_change_log.md` (one-liner per change) if existing.

## Step 10: Process Discipline (non-negotiable)

1. **No parameter changes on weekends or after losses** — 24h cool-off
2. **Every change logged** (see Step 9)
3. **Every autoresearch run logged** with N_trials for DSR deflation
4. **One change at a time** — never bundle, can't attribute
5. **Reset is sometimes correct** — revert to last validated config if accumulated tweaks made strategy unreviewable

## One-Page Decision Checklist

Before responding to ANY request that involves changing strategy params:

```
PRE-CHANGE CHECKLIST (refuse if any "no")
──────────────────────────────────────────
[ ] Step 0: Read STRATEGY_OPTIMIZATION_LEARNINGS.md for this strategy
[ ] Is this a risk-reduction change? → skip to deploy
[ ] Is live N ≥ minimum required? (Step 2 table)
[ ] Have I computed Bayesian P(better)? (Step 3)
[ ] Have I bootstrapped last 200 trades? Improvement > 1 SE?
[ ] Is this NOT a regime issue? (Step 4)
[ ] If autoresearch-derived: all Step 5 gates passed?
[ ] Revert criteria armed? (Step 9)
[ ] Single change (not bundled)?
[ ] 24h since last loss? (cool-off)

DEPLOY ONLY IF ALL CHECKED.

POST-DECISION:
[ ] Append entry to STRATEGY_OPTIMIZATION_LEARNINGS.md (Step 9b)
[ ] Schedule outcome update after 50-100 trades
```

## How to Respond to Common Scenarios

### "Live says X, backtest says Y, what to do?"
1. Compute Bayesian weighting (Step 3)
2. Check regime (Step 4)
3. If live N < min required → refuse change, explain threshold
4. If live N sufficient + P(better) > threshold → propose change with revert armed

### "Should we deploy this autoresearch winner?"
Run Step 5 validation gates. Any failure = reject.

### "PnL is flat, what's wrong?"
1. Check payout asymmetry (breakeven WR vs actual WR)
2. Check for regime change (Step 4)
3. DO NOT immediately propose param changes
4. If under min N → explain we need more data

### "Autoresearch found edge≥0.40 gives +$3"
Ask: PBO? DSR? N_trials? Plateau check? Without these numbers, reject.

## Authoritative Document

Full methodology with math, citations, and decision trees:
```
/Users/dnl.vlck/Arbo/docs/STRATEGY_OPTIMIZATION_FRAMEWORK.md
```

603 lines, 10 sections, full reference list. Re-read before any significant strategy decision. It's 15 min reading. Saves months of losses.

## Enforcement Rule

**If the user asks you to change a parameter and you haven't gone through this skill — you are doing it wrong. Refuse and apply the framework first.**

Bias toward:
- MORE data before decisions
- SMALLER capital while uncertain
- STRUCTURED process over gut feel
- REVERTING early when evidence is unclear
- KILLING strategies that lose repeatedly
