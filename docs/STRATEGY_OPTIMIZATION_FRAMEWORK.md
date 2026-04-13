# Arbo Strategy Optimization Framework — Rigorous Methodology for Systematic Trading Decisions

> Authoritative reference for every optimization, parameter change, and live deployment decision in Arbo.
> Version 1.0 — 2026-04-12
> Synthesizes López de Prado, Bailey, Harvey, Asness, Chan, and academic best-practice into Arbo-specific rules.

---

## Executive Summary

Most quantitative trading strategies that "work in backtest" fail in live trading. The dominant reason is not bad ideas — it is **statistical malpractice during the research and deployment process**. Bailey, Borwein, López de Prado and Zhu (2014) showed that with as few as 7 independent trials on a strategy with truly zero edge, one expects to find a Sharpe ratio of ~1.0 by chance alone; with 1,000 trials the expected maximum Sharpe is ~3.26 (Bailey & López de Prado, "The Deflated Sharpe Ratio", 2014).

This document provides a **process** — not a set of indicators — that any Arbo strategy (B3, C2, D, future) must pass through before risking real capital, and a clear decision framework for when (and how) to change live parameters.

The core thesis: **discipline beats cleverness**. A mediocre strategy run with strict process will outperform a clever strategy adapted on whim.

---

## Section 1: The Core Problem — Why Ad-Hoc Parameter Changes Destroy Strategies

### 1.1 The Multiple-Testing Crisis

Every parameter sweep is implicit multiple hypothesis testing. If you test N parameter combinations and report the best one, the expected best Sharpe ratio is far above zero **even if every parameter set has zero true edge**.

The False Strategy Theorem (Bailey & López de Prado, 2014) gives the expected maximum Sharpe under the null hypothesis of zero true edge across N independent trials:

```
E[max SR_N] ≈ √V[SR] · [(1 - γ) · Φ⁻¹(1 - 1/N) + γ · Φ⁻¹(1 - 1/(N·e))]
```

where γ ≈ 0.5772 (Euler-Mascheroni constant), Φ⁻¹ is the inverse standard Normal CDF, and V[SR] is cross-sectional variance of Sharpe ratios across trials.

**Concrete numbers**: For a sweep of 1,000 parameter combos with cross-sectional SR variance of 1.0, the expected best Sharpe under the null is ~3.26. **A Sharpe of 3 means nothing if you ran 1,000 trials.**

### 1.2 Most Discoveries Are False

Harvey, Liu & Zhu (2016) cataloged 316 academic "factors" and concluded that the conventional t-stat threshold of 2.0 is wrong; for new factor discovery the correct threshold is **t > 3.0** to control for multiple testing. They estimate that the **majority of claimed findings in financial economics are likely false**.

López de Prado (2018, "10 Reasons Most ML Funds Fail", JPM 44(6)): the #1 failure mode is "Sisyphus paradigm" — repeatedly tweaking a single backtest until it looks good, with no accounting for the implicit trials.

### 1.3 The Adaptive-Tweak Trap

When Arbo observes "live N=12 says cap fill at 0.75, shadow N=144 says no cap" and changes the parameter, it is implicitly running another trial. After 20 such ad-hoc tweaks across a strategy's lifetime, the strategy has been "overfit" to its own live history — exactly the same problem as backtest overfitting, but invisible because it accumulates over time. This is sometimes called **"parameter creep"** or **"adaptive overfitting"**.

**Rule**: every parameter change must be logged with its evidence basis and counted as a trial in the deflation calculation.

---

## Section 2: The Research Lifecycle

A strategy must traverse six gates. Skipping a gate is the single most common cause of capital destruction.

| Gate | Phase | Minimum N | Decision Criterion |
|------|-------|-----------|---------------------|
| 1 | Hypothesis Formation | n/a | Economic story before data. Must answer: *why does this edge exist, who is on the other side, why won't it disappear?* |
| 2 | In-Sample Backtest | ≥ 200 trades or 2 years | Sharpe > 1.0 *after* deflation, max DD < 25% |
| 3 | Out-of-Sample (CPCV) | ≥ 30% of total data, never seen during search | Deflated Sharpe p < 0.05, PBO < 0.5 |
| 4 | Walk-Forward | Last 6+ months, rolling | Sharpe degradation < 50% vs IS |
| 5 | Paper Trading | ≥ 4 weeks AND ≥ 200 trades | Live-vs-paper price match within 5%; PnL within 1σ of expected |
| 6 | Live (Small Capital) | ≥ 200 trades at small size | Live Sharpe within 1σ of paper Sharpe; no regime-break |
| 7 | Scale-Up | Doubling rule (see §7) | Each doubling requires 100+ trades of confirming evidence |

**Why N ≥ 200 trades?** Standard error of the mean scales as 1/√N. For a strategy with true win rate 55%, sample WR has standard error √(0.55·0.45/N). At N=12, SE = 14.4%; observed WR could legitimately be anywhere from 40% to 70%. At N=200, SE = 3.5% — observed WR within 51.5%–58.5%. **Below N=200, you literally cannot distinguish skill from luck for the win rates Arbo strategies operate at.**

This is also the threshold López de Prado implicitly uses in CPCV — typically 6 paths × 50+ test labels each.

---

## Section 3: Validation Techniques

### 3.1 Why Standard k-Fold Cross-Validation Fails on Time Series

Three failure modes:
1. **Look-ahead leakage** — test labels overlap in time with training labels (relevant when labels span multiple bars, e.g. "did BTC move 1% in next 5 min" labels overlap if computed every minute)
2. **Serial correlation** — adjacent observations are not independent
3. **Single backtest path** — k-fold gives one out-of-sample sequence; can't estimate variance of strategy performance

### 3.2 Combinatorial Purged Cross-Validation (CPCV)

Developed by López de Prado (2018, *Advances in Financial Machine Learning*, Ch. 12). Procedure:
1. Split data into N sequential groups
2. Choose all C(N,k) combinations of k test groups, with N-k as training
3. **Purge** training observations whose labels overlap test-group labels
4. **Embargo** training observations immediately following test groups (default ~1% of total)
5. Compute backtest path for each combination

**Why it dominates standard k-fold**: CPCV produces multiple backtest paths (e.g., C(10,2)=45 paths from N=10, k=2) instead of one. You can compute distribution of Sharpes, drawdowns, hit rates — and feed these into the Probability of Backtest Overfitting calculation. Empirical comparisons (e.g., García-Feijóo & Jensen 2024 in *Knowledge-Based Systems*) show CPCV has lower PBO and higher Deflated Sharpe than walk-forward or hold-out.

**Arbo application**: every B3, C2, D autoresearch sweep MUST use CPCV when N_observations > 1000. For smaller datasets (B3 5-min sometimes), use walk-forward with embargo as second-best.

### 3.3 Walk-Forward Analysis

Two flavors:
- **Anchored / expanding window** — train on [0, t], test on [t, t+Δ]. Best when regime is stable.
- **Rolling window** — train on [t-W, t], test on [t, t+Δ]. Best when regime drifts.

For Arbo's crypto strategies (B2, B3): **rolling window** (e.g., W = 90 days, Δ = 30 days) because BTC regime changes quickly. For weather (C, C2): **anchored** because seasonality benefits from full history.

### 3.4 Probability of Backtest Overfitting (PBO)

Bailey, Borwein, López de Prado & Zhu (2014), *Combinatorially Symmetric Cross-Validation*. Procedure:
1. Split data into S equal slices (S even, e.g., 16)
2. Generate all C(S, S/2) train/test partitions
3. For each: pick the parameter set that maximizes IS Sharpe; record its OOS rank
4. PBO = probability that OOS rank is below median = **fraction of partitions where best-IS strategy underperforms median OOS**

**Threshold**: PBO < 0.5 is the bare minimum (better than coin-flip). Production strategies should target PBO < 0.3.

**For Arbo**: this is the single most important number to compute on every autoresearch run. If PBO > 0.5, the autoresearch is producing noise, not signal.

### 3.5 Deflated Sharpe Ratio (DSR)

Adjusts observed Sharpe for: number of trials, return non-Normality, sample length. Formula:

```
DSR = Φ( (SR_observed − SR_0) · √(T−1) / √(1 − γ̂₃·SR_0 + ((γ̂₄−1)/4)·SR_0²) )
```

where SR_0 is the expected maximum SR under the null (False Strategy Theorem, §1.1), T is sample length in trades, γ̂₃ skewness, γ̂₄ kurtosis.

**Interpretation**: DSR > 0.95 means there's < 5% probability the observed SR arose from selection bias on N trials.

**Arbo rule**: every reported autoresearch winner must include both raw Sharpe AND deflated Sharpe with N_trials documented.

### 3.6 Monte Carlo of Strategy Returns

Bootstrap resampling of trade returns gives a distribution of paths. Useful for:
- Estimating max drawdown distribution (the realized one is one sample)
- Estimating "what if I had stopped at trade X" risk
- Stress-testing position sizing rules

**Arbo rule**: before any strategy goes live, run 10,000 bootstraps of the OOS trade sequence. The 95th-percentile drawdown is the number you must be willing to lose. If it exceeds your kill-switch, reduce size.

### 3.7 Minimum Track Record Length (MinTRL)

Bailey & López de Prado (2012, "The Sharpe Ratio Efficient Frontier"): given an observed Sharpe, how many observations are needed to reject the null hypothesis SR ≤ SR* at confidence level 1-α?

```
MinTRL = 1 + (1 − γ̂₃·SR_obs + ((γ̂₄−1)/4)·SR_obs²) · (Z_α / (SR_obs − SR*))²
```

**Arbo example**: for B3 live (observed Sharpe ≈ 1.5 from N=12 trades), to reject "SR ≤ 0.5" at 95% confidence requires ~80–120 trades depending on skew/kurt. **N=12 cannot reject anything**.

---

## Section 4: Parameter Sensitivity & Robustness

### 4.1 Plateau vs Peak — Always Pick the Plateau

A robust optimum has neighbors that perform almost as well. A fragile optimum is a sharp peak surrounded by ravines.

**Diagnostic**: for the chosen parameter, perturb each parameter by ±10% and ±20%. Compute Sharpe at each perturbation.

- If the average Sharpe across 9 perturbations (3³ for 3 params) is > 80% of the peak → robust plateau
- If it drops below 50% → fragile peak, REJECT

**Why**: real markets perturb the world, not the strategy. A plateau has a margin of safety against parameter mis-estimation.

### 4.2 Coarse-to-Fine Sweep Methodology

Wrong: dense grid over wide range (high N_trials → severe deflation).

Right:
1. **Coarse pass** — 5 points per dimension over wide range
2. Identify ~3 promising regions (peaks)
3. **Refined pass** — 10 points per dimension over each region
4. Final selection done on OOS of coarse-pass winner, not refined-pass winner (refined sweep is part of the same trial budget but tighter)

This keeps log(N_trials) low for DSR purposes while still finding the optimum.

### 4.3 Parameter Ensemble vs Single Optimum

Rather than picking one parameter set, take the top-K by OOS Sharpe and **average their signals** (or trade them as K parallel sub-strategies with 1/K capital each).

Benefits:
- Reduces variance from picking a single noisy winner
- More robust to regime change (different params win in different regimes)
- Effectively a Bayesian model average

**Arbo application**: B3 watchdog can maintain a small "council" of 3–5 parameter sets and weight them by recent OOS performance. This is structurally similar to the EMOS ensemble already deployed for C1f.

---

## Section 5: Regime Detection & Adaptation

### 5.1 The Regime Problem

Strategies are conditional bets — they assume a market state. When the state changes, the strategy can break. B3's current dilemma — "shadow period was BTC volatile, now flat" — is exactly this.

### 5.2 Regime Detection Methods

**Hidden Markov Model (HMM)** — the standard. Fit a Gaussian HMM with 2–4 hidden states on rolling realized volatility (or returns + volume). The Viterbi algorithm assigns each bar a state. Use Bayesian Information Criterion (BIC) to pick the number of states.

**Bayesian Online Change-Point Detection (BOCPD)** — Adams & MacKay (2007). Computes the posterior probability distribution over the "run length" since the last change point. Useful for **near-real-time** detection (HMM is more retrospective).

**Volatility regime via realized-vol percentile** — simplest. Tag each bar as "low/mid/high" by its place in trailing 90-day distribution. Crude but works.

### 5.3 Re-Optimize vs Tolerate Drawdown

Decision matrix:

| Regime change detected? | Live performance | Action |
|---|---|---|
| No | Within expected band | Hold |
| No | Outside band (drawdown) | Tolerate (it's noise) — see kill criteria §8 |
| Yes | Within band | Investigate but hold |
| Yes | Outside band | Re-validate on new regime data; do NOT immediately re-optimize |

**Critical**: do not re-optimize on a small post-regime sample. Wait for ≥ 200 trades in the new regime. Premature re-optimization is the fastest way to compound losses.

### 5.4 Meta-Labeling (López de Prado, AFML Ch. 3)

A **secondary** ML model that takes the primary signal as input and outputs a probability that the trade will be profitable. It does not change *direction* — it sizes (or vetoes) trades.

For Arbo: a meta-labeler for B3 could take features (ADX, RSI, BTC velocity, time of day, recent fill quality) and output P(this trade is profitable). Trades with P < 0.55 → skip. Trades with P > 0.70 → full size. Trades 0.55–0.70 → half size.

This separates the *direction* decision (B3 entry filter) from the *quality* decision (will this specific trade work in the current micro-regime). Empirically, this raises precision (fewer false positives) at the cost of recall — Sharpe typically improves even if total returns dip.

---

## Section 6: Execution Reality Gap

### 6.1 Why Backtest ≠ Live

The gap has named components:
1. **Slippage** — fill price worse than quoted
2. **Adverse selection** — your maker order fills when the market moves against you (the "winner's curse" of market making)
3. **Market impact** — your own order moves the price
4. **Selection bias on fills** — PostOnly orders only fill *some* of the time, and disproportionately when the market is moving against you
5. **Latency** — by the time you observe a signal and place an order, the price has moved
6. **Spread cost** — paying full spread on entry+exit can wipe out edge

For **Polymarket binary contracts**, the dominant gaps are #4 (selection bias on fills) and #6 (17–47% spreads on weather tokens — already documented in Arbo's `paper_vs_live_pricing_bug.md`).

### 6.2 Adverse Selection on PostOnly Orders

Empirical research (e.g. Stoikov & Saglam, "Optimal Quoting under Adverse Selection") shows that on most exchanges, **the majority of maker fills are adverse**: the market moves against the maker immediately after fill. This is structural — informed taker flow is what fills makers.

**Implication for B3**: a backtest that assumes "I get filled at the bid" overstates returns. Correct backtest assumes:
- Fill probability = function of (distance from mid, queue position, recent flow)
- Conditional on fill, price moves against you with probability > 50% in next bar

### 6.3 Bridging the Gap

Three layers of validation:
1. **Realistic-friction backtest** — model spread, fill probability, latency in the simulator
2. **Shadow paper trading** — same logic running on live market data, comparing simulated fills to what would have been observed
3. **Live with small capital** — the only true test

**Arbo currently has #2 partially (paper engine) but the paper engine has the inversion bug** (see `paper_vs_live_pricing_bug.md`). Until that's fixed, paper Sharpe is an unreliable estimate of live Sharpe — paper *earns* the spread, live *pays* it.

**Action**: until paper bug is fixed, apply a **spread haircut** to all paper Sharpes. Estimate average spread × 2 (entry + exit) and subtract from per-trade return. Re-run autoresearch with corrected returns.

---

## Section 7: Risk Management & Kelly Sizing

### 7.1 Why Half / Quarter Kelly

Full Kelly maximizes long-term log-wealth growth but has brutal short-term properties:
- 50% probability of 50% drawdown
- 20% probability of 80% drawdown
- Extremely sensitive to mis-estimation of edge (overestimating edge by 2x = blow up)

Half-Kelly: ~75% of full Kelly's growth, ~half the drawdown.
Quarter-Kelly: ~44% of full Kelly's growth, ~quarter the drawdown.

**Arbo already uses KELLY_FRACTION=0.25** for Strategy C, which is appropriate given:
- Edge estimates are noisy (especially in small live samples)
- Drawdowns trigger psychological and operational issues (Slack alerts, kill switches)
- Capital is finite and irreplaceable

### 7.2 Drawdown-Based Scaling

Mechanism: as drawdown grows, *reduce* size. As profits accumulate, *cautiously increase*.

Concrete rule (used by many CTAs):
- At 0% DD: 1.0× base size
- At 5% DD: 0.75× base size
- At 10% DD: 0.5× base size
- At 15% DD: pause; re-validate
- At 20% DD: kill switch

This is **anti-martingale** — opposite of "double down on losers". Mathematically optimal under Kelly when edge is uncertain.

### 7.3 Per-Strategy Capital Allocation

Three approaches:
- **Equal dollar** — simplest, robust, used by Arbo currently (A=$400, B=$400, C=$1000)
- **Risk parity** — equalize contribution to portfolio variance
- **Correlation-adjusted Kelly** — allocate by edge/variance accounting for correlations

For 2–4 strategies (Arbo's case), equal dollar is fine. Move to risk parity only when N_strategies > 5.

---

## Section 8: The Decision Framework — When Should You Change Anything?

This is the section the user asked for most directly. Concrete rules.

### 8.1 The Bayesian Combination Rule (Shadow + Live)

When live N is small and shadow N is large, weight by inverse variance.

Let:
- μ_S = shadow mean return per trade, σ_S² = its variance, N_S = shadow trade count
- μ_L = live mean return per trade, σ_L² = its variance, N_L = live trade count

Posterior mean (assuming Normal-Normal conjugate):
```
μ_post = (μ_S · N_S/σ_S² + μ_L · N_L/σ_L²) / (N_S/σ_S² + N_L/σ_L²)
```

**Arbo concrete case**: B3 shadow N=144, live N=12. If we assume similar variance:
```
weight_shadow = 144 / (144 + 12) = 92.3%
weight_live = 12 / (144 + 12) = 7.7%
```

**Live N=12 contributes < 8% of evidence.** It cannot override shadow unless live shows extreme deviation (>3σ).

But: this only applies **if regime is stable**. If a regime change is detected, downweight shadow toward zero.

### 8.2 Evidence Threshold for a Parameter Change

A parameter change is itself a hypothesis. To change a parameter, you should require:

- **P(new params better than current | data) > 0.75**, AND
- **Expected improvement > 1 standard error of current performance**, AND
- **Change has been validated on held-out data** (not just the data that motivated the change)

Practical implementation: **bootstrap test**.
1. Take last 200 live trades
2. Resample with replacement 1000 times
3. For each resample, compute hypothetical PnL under (a) current params, (b) proposed params
4. Count fraction where (b) > (a). This is P(better).

**Threshold**: change params only if P(better) > 0.75. (3-to-1 evidence ratio.)

### 8.3 Revert Criteria

After making a change, set explicit revert triggers BEFORE deploying:
- Revert if 50-trade rolling Sharpe drops > 50% from pre-change baseline
- Revert if 50-trade rolling WR drops > 5pp from pre-change baseline
- Revert if cumulative PnL since change is negative AND below 1σ band of expected

The B3 Watchdog spec already encodes this principle (50-trade auto-revert) — extend it to all strategies.

### 8.4 Strategy Kill Criteria

Kill (not pause, not re-optimize — kill) a strategy when:
- Live drawdown exceeds 2× backtest 95th-percentile drawdown
- 6 consecutive months of underperformance vs paper-equivalent baseline
- Regime change is structural (e.g., exchange policy change, oracle change) and re-validation shows no edge in new regime
- Strategy's economic thesis is invalidated (the "why does this work" answer is no longer true)

Killing is a feature, not a failure. AQR (Asness): only ~5–10 robust factors exist after decades of academic search. Most strategies should die.

---

## Section 9: Continuous Monitoring

### 9.1 Metrics Dashboard

Track on a daily / weekly cadence:

| Metric | Frequency | Alert threshold |
|---|---|---|
| Rolling Sharpe (50-trade) | Daily | < 50% of historical |
| Rolling drawdown (peak-to-trough) | Daily | > 1.5× backtest 95th pct |
| Rolling hit rate (50-trade) | Daily | > 1σ deviation |
| Average trade PnL (50-trade) | Daily | > 1σ deviation |
| Fill rate (orders filled / placed) | Daily | < 50% of historical |
| Average spread paid | Daily | > 1.5× historical |
| PSI on key features | Weekly | > 0.2 |
| Expected Calibration Error (ECE) | Weekly | > 0.10 |
| DSR vs deployed config | Monthly | < 0.95 |

### 9.2 PSI (Population Stability Index)

For each model feature, compute distribution shift between training period and live period:
```
PSI = Σᵢ (p_live,i − p_train,i) · ln(p_live,i / p_train,i)
```
where bins i partition the feature range.

Industry-standard thresholds (used at credit risk shops, FICO):
- PSI < 0.10 — no significant shift
- 0.10 ≤ PSI < 0.25 — moderate shift, investigate
- PSI ≥ 0.25 — major shift, likely re-train

### 9.3 ECE (Expected Calibration Error)

For probabilistic models (B2 daily probability, C2 forecast prob): does P(predicted)=0.7 actually correspond to ~70% realized rate?

```
ECE = Σ_b (n_b/N) · |accuracy(b) − confidence(b)|
```

binned over predicted-probability buckets.

**Threshold**: ECE > 0.10 indicates miscalibration. If model says "70%" but actual is "55%", sizing via Kelly is wrong even if direction is right.

### 9.4 Re-Evaluation Cadence

| Cadence | What to do |
|---|---|
| Daily | Read dashboard, react only to alert thresholds |
| Weekly | Compute rolling Sharpe, PSI; review any alerts |
| Monthly | Full performance review; check vs paper baseline; update Bayesian posterior |
| Quarterly | Re-validate model on most recent quarter as held-out OOS; check DSR |
| Annually | Full strategy re-research (CPCV, PBO, all sweeps) — consider sunset |

**Avoid sub-daily reactions to live PnL**. Most decisions made on hourly noise are wrong.

---

## Section 10: The Arbo-Specific Playbook

Concrete rules tailored to Arbo's situation.

### 10.1 Minimum Live N Before Major Param Change

| Change type | Min live N | Required evidence |
|---|---|---|
| Cosmetic (logging, format) | 0 | Any |
| Risk reduction (smaller size, tighter cap) | 0 | Always allowed |
| Filter loosening (lower min_edge etc) | 100 | Bayesian P(better) > 0.75 |
| Filter tightening | 50 | Bayesian P(better) > 0.65 |
| Sizing parameters (kelly_cap etc) | 200 | Bayesian P(better) > 0.80 + plateau check |
| Core model parameters (sigma_scale etc) | 300 | Full re-autoresearch + CPCV + DSR > 0.95 |
| Strategy-level (entry signal, label rule) | 500 | Treated as new strategy, full Section 2 lifecycle |

**Risk reduction is always allowed without evidence.** Bias toward smaller, safer.

### 10.2 The B3 Live N=12 Decision

Apply §8.1 directly. Shadow N=144 dominates live N=12 by ~12:1 weight. Live observation is **not statistically significant**. Bootstrap test will confirm: with N=12 and typical variance, almost no parameter change passes the P(better) > 0.75 threshold.

**Action**: do NOT change params based on N=12. Continue current params. Wait for N ≥ 100 live trades. If at N=100 the live performance is still divergent from shadow, then:
1. Compute Bayesian posterior (now ~58% shadow weight, 42% live weight)
2. Check for regime change (HMM on BTC vol since shadow period)
3. If regime change confirmed → downweight shadow → re-validate on recent BTC-flat period only
4. If no regime change → accept that shadow was lucky, use posterior estimate

### 10.3 The Shadow-Was-Volatile, Live-Is-Flat Dilemma

This is a regime question, not a parameter question.

Test:
1. Compute realized BTC vol on shadow period vs live period
2. If ratio > 1.5× → significant regime difference
3. Re-run autoresearch on shadow data **conditioned on flat-vol periods only**
4. Compare resulting parameters to current
5. If they converge → current params are robust, just being unlucky in flat period
6. If they diverge → deploy regime-switching version: detect vol regime in real time, switch param set

Document this analysis in `b3_regime_analysis.md` before any param change.

### 10.4 Decision Tree for Common Situations

```
Live PnL down 1 day → ignore (noise)

Live PnL down 1 week + within DD band → continue monitoring

Live PnL down + DD > 50% of backtest 95-pct →
  ├─ Regime change detected? → §10.3 path
  └─ No regime change → §8.1 Bayesian update; if posterior still positive → continue, reduce size
                                                if posterior negative → pause, re-validate

Live WR > shadow WR by 5pp on N<50 → ignore (noise; Bayesian weight is tiny)

Live WR > shadow WR by 5pp on N>200 → update sizing UP by Bayesian posterior; do NOT lift filters

Autoresearch produced a "better" config →
  ├─ Was it CPCV-validated? No → reject
  ├─ Did you compute DSR with N_trials? No → reject
  ├─ DSR > 0.95? No → reject
  ├─ PBO < 0.5? No → reject
  ├─ Plateau check passed? No → reject
  ├─ Better than current on bootstrap test? P > 0.75? No → reject
  └─ All yes → deploy with revert criteria armed (§8.3)

Live Sharpe within 1σ of paper Sharpe → no action needed

Live Sharpe < 50% of paper Sharpe AND N > 100 →
  ├─ Compute spread cost — does it explain gap? → fix paper model
  ├─ Compute fill-rate gap — does it explain gap? → fix paper model
  └─ Neither → strategy is degraded, follow §8.4 kill criteria
```

### 10.5 Arbo Process Discipline Rules

1. **No parameter changes on weekends or after losses.** Decisions made under emotional stress are biased toward action. Mandatory 24-hour cool-off.
2. **Every parameter change is logged** in `parameter_change_log.md` with: date, param, old, new, evidence (P(better) bootstrap), reverts armed.
3. **Every autoresearch run is logged** with: N_trials, parameter ranges, CPCV PBO, DSR. This is your trial budget for DSR deflation across the strategy's lifetime.
4. **Reset is sometimes correct.** If accumulated tweaks have made the strategy unreviewable, revert to last validated config and start fresh.
5. **One change at a time.** Never change two parameters simultaneously. You will not be able to attribute the result.

---

## One-Page Decision Checklist (Print and Pin Above Desk)

> Before changing ANY parameter on a live strategy, answer ALL questions. If any answer is "no" or "I don't know" — STOP.

```
PRE-CHANGE CHECKLIST
─────────────────────────────────────────────
□ 1. Is this a risk-reduction change? → if yes, skip to deployment
□ 2. Have I logged the current config and PnL baseline?
□ 3. Is live N ≥ minimum required for this change type? (§10.1 table)
□ 4. Have I computed Bayesian P(new better than current)?
   → P(better) ≥ required threshold (§10.1)?
□ 5. Have I bootstrapped the last 200 trades?
   → Is improvement > 1 SE of current performance?
□ 6. Have I verified this isn't a regime issue (§5.3 matrix)?
□ 7. If autoresearch-derived: PBO < 0.5? DSR > 0.95? Plateau check passed?
□ 8. Have I armed revert criteria? (§8.3 — 50-trade rolling triggers)
□ 9. Is this a single change (not bundled with others)?
□ 10. Has 24h passed since the last loss? (cool-off)

DEPLOY ONLY IF ALL CHECKED.

POST-DEPLOY MONITORING
─────────────────────────────────────────────
□ Daily for 50 trades: rolling Sharpe vs baseline
□ Weekly: PSI on key features
□ At trade 50: revert check (rolling Sharpe < 50% of baseline → REVERT)
□ At trade 100: full evaluation, decide hold/extend/revert
□ At trade 200: integrate into Bayesian posterior, update strategy memory

KILL CRITERIA (any one triggers full pause)
─────────────────────────────────────────────
× Live DD > 2× backtest 95th-percentile DD
× 6 consecutive months underperforming paper baseline
× Economic thesis invalidated
× DSR drops below 0.50 on rolling re-evaluation
```

---

## Section 11: Rapid Mode — Parallel Exploration (v2.0 extension)

> Added 2026-04-13. Complements the serial framework above with techniques to **accelerate discovery 5-10×** via parallelism. Does NOT replace statistical rigor (DSR, PBO, min N still apply) — just runs more hypotheses in parallel with paired-sample testing.
>
> Detailed study: `docs/RAPID_MODEL_DISCOVERY.md` (538 lines). This section condenses the usable techniques.

### 11.1 When to Use Rapid vs Serial Mode

| Situation | Mode | Why |
|---|---|---|
| Tuning ONE param on a LIVE strategy | **Serial** (§10) | Safety, Bayesian shadow+live weighting requires disciplined gates |
| Exploring multiple NEW hypotheses | **Rapid** (this §) | Parallel variants in shadow give paired samples, 5-10× faster |
| Regime change suspected on production | **Serial** first (§5.3) | Don't compound errors with rapid iteration on contaminated data |
| Building new strategy from scratch | **Rapid** | Many ideas die fast, cheap |
| Single-variant capital sizing | **Serial** | Sizing is high-impact, needs full lifecycle gates |
| Capital allocation across variants of SAME strategy | **Rapid** (MAB) | Bandits dominate fixed allocation |

**Both modes share the same end-gate**: promotion to live-with-real-capital requires DSR > 0.95, PBO < 0.3, plateau check passed, revert triggers armed.

### 11.2 Champion-Challenger Pattern

Every strategy has ONE **champion** (current production, main capital) and 2-8 **challengers** (alternate configs running in parallel).

**Rules**:
- Challengers see same signals as champion (paired-sample statistics)
- Each challenger runs in **shadow** (no capital) OR with 10-25% of strategy capital
- Promotion rule: `challenger_DSR > champion_DSR + τ` across ≥ N paired observations (N per §10.1 table for change type)
- Retirement rule: challenger underperforms champion by > 2σ over 50 paired obs → retire
- Max concurrent challengers per strategy: 8 (diminishing marginal info gain, multiple-testing inflation)

**Source**: FICO decision management, Two Sigma head-to-head, AWS ML Lens MLREL-11.

### 11.3 Shadow Orchestrator (paired exploration)

Every signal from market is forked to ALL active variants. Each variant's decision is logged + simulated via paper_engine; no capital at risk.

**Why this is 5-10× faster than serial**:
- Serial: 1 config × 100 trades = 100 trades of data → 10-14 days at 8/day
- Parallel: 8 configs × same 100 signals = 800 trades of data → 10-14 days, but **with pairing** (same market conditions, same signal flow, lower variance of difference statistic)
- After 150 paired obs per variant (≈ 1 week at B3_15M rates), DSR-adjusted pairwise tests become decisive

**Implementation**: extend existing `arbo/strategies/b3_15m_shadow.py` into generic `ShadowOrchestrator` — already has scaffolding (signal generation, paper fill simulation, DB logging).

### 11.4 Bayesian Optimization for Parameter Search

**Replace grid sweeps with Optuna TPE** (Tree-structured Parzen Estimator) or scikit-optimize GP-EI.

- Grid 235k configs → BO 50-100 evaluations, same or better optimum
- **30-100× speedup** per autoresearch cycle
- BO explicitly models objective uncertainty → explores intelligently
- Use **Optuna** (discontinuity-robust, parallel trials) over vanilla GP for trading objectives with cliffs

**Library**: `optuna` (primary), `scikit-optimize` (alternative).

**Rule**: every new autoresearch sweep must be BO by default. Grid only if user explicitly requests or if BO fails convergence.

### 11.5 Multi-Armed Bandit for Capital Allocation

Once you have 3+ live variants, allocate capital via **Thompson Sampling** (TS) instead of fixed split.

**Why TS over fixed allocation**:
- Logarithmic regret bound (Agrawal & Goyal 2012) — optimal up to constants
- **Natural explore/exploit**: auto-samples posterior, no hyperparameter tuning
- Reward = per-trade realized PnL or **composite reward** (see §11.7)
- Non-stationary version: **discounted TS** (decay older rewards, γ=0.98/day) for regime adaptation
- +20% Sharpe vs uniform in CADTS paper (arXiv 2410.04217)

**Library**: `mabwiser` (Fidelity, scikit-learn style, production-grade).

**Safety rules** (prevent over-exploitation):
- Floor each active arm at ≥ 10% capital until ≥ 30 observations per arm
- Re-balance daily, not per-trade (avoid jitter)
- Always maintain at least 2 arms (prevents lock-in)

### 11.6 Drift Detection (regime change real-time)

Replace "wait for drawdown trigger" with **online drift detectors** running per-variant:

| Detector | Watch | Fires when |
|---|---|---|
| **Page-Hinkley** (CUSUM-based) | rolling 50-trade WR | Cumulative deviation exceeds λ threshold |
| **ADWIN** (adaptive windowing) | realized_pnl / paper_pnl ratio | Hoeffding bound violated on sub-window split |

**When detector fires on variant X**:
1. Watchdog pauses live capital on variant X
2. Re-run BO on recent 30-day data only (discard pre-drift)
3. Promote new champion only after 20+ post-drift trades confirm

**Benefit**: Page-Hinkley typically fires 3-5× faster than drawdown-based triggers → faster recovery from regime change.

**Library**: `river` (online ML, has both detectors built-in).

### 11.7 Composite Reward (mid-trade evaluation)

Polymarket settlement is slow ($1/$0 after 5-15 min). Use **intermediate signals** to get faster learning:

```
reward = 0.4 * directional_correct_60s + 0.6 * normalize(realized_pnl)
```

- `directional_correct_60s`: did mid-price move in our direction 60s after entry? Fast (1-min resolution), low variance.
- `realized_pnl`: final $1/$0 settlement, definitive but slow.
- Composite converges **3-5× faster** than pnl-only reward.

**This is free infrastructure** — just add `mid_at_30s`, `mid_at_60s` columns to `trade_details` on every trade.

### 11.8 Block Bootstrap for Statistical Tests

Every DSR / "is config A better than config B?" test must use **stationary block bootstrap** (Politis-Romano 1994), NOT i.i.d. bootstrap — financial returns have autocorrelation.

```python
from arch.bootstrap import StationaryBootstrap
bs = StationaryBootstrap(block_length=auto, returns)
# auto-block length per Politis & White 2004
ci = bs.conf_int(lambda x: x.mean(), reps=10000)
```

**Rule**: every promotion / revert decision must include block-bootstrap CI on the difference statistic.

### 11.9 HRP Ensemble (combine top variants, don't just pick one)

Instead of killing N-1 losers and keeping 1 winner, **ensemble top-3 variants** via Hierarchical Risk Parity:

- HRP clusters variants by correlation
- Down-weights highly-correlated variants, up-weights diversifying
- Lower OOS variance than mean-variance, works with few samples
- **Quorum rule**: enter trade only if ≥ 2 of 3 variants agree → kills 40% of trades but lifts WR 5-10pp

**Library**: `PyPortfolioOpt` (HRP in 3 lines).

**When to ensemble**: after champion-challenger phase (≥ 1 month post-launch), keep top-3 surviving variants and use HRP weights.

### 11.10 Hypothesis Factory Mindset (lifecycle FSM)

Shift from "we have one B3 strategy" to "we have a pool of B3-family hypotheses":

```
idea → shadow (no money) → incubate ($5-$25) → small live ($25-$100) → scaled ($100+) → retire
         kill if no edge        kill if DSR low      scale via MAB         alpha decay
                                after 50 trades
```

Each stage has pre-registered kill / promote rules. 60% kill rate in incubation is a **feature** — it's how the system avoids overfit. Survivors compound.

**Impact**: instead of 6-8 iterations per quarter on single strategy, we test 20-40 hypotheses per quarter across the pool.

### 11.11 Rapid Mode Dashboard Requirements (MANDATORY)

**Every strategy with ≥ 2 active variants MUST have a "Variant Leaderboard" card on the dashboard:**

Required fields:
- Per-variant rolling 50-trade PnL
- Per-variant WR + 95% block-bootstrap CI
- Per-variant DSR + posterior-median Sharpe
- Current capital allocation (from MAB if live, equal if shadow)
- Drift detector status (Page-Hinkley / ADWIN) — green/yellow/red
- Last-promoted / last-retired variant + date
- Composite reward trend (if mid-trade evaluation logged)

**Why**: without this card, variants become invisible noise in DB — user cannot oversee the pool. Watchdog automation hides decisions; dashboard must surface them.

Implementation: new `arbo/dashboard/variant_leaderboard.py` component, rendered per strategy on its tab.

### 11.12 Practical Toolchain (Arbo)

| Tool | Purpose |
|---|---|
| `optuna` | BO replaces grid sweeps |
| `mabwiser` | Thompson Sampling, UCB1, ε-greedy |
| `river` | Online learning + drift detectors (Page-Hinkley, ADWIN) |
| `arch` | Block bootstrap (Politis-Romano stationary, Politis-White auto block-length) |
| `pyportfolioopt` | HRP ensemble weighting |
| `hypothesis` | Property-based testing of orchestrator logic |

Install once: `pip install optuna mabwiser river arch pyportfolioopt hypothesis`

### 11.13 Rapid Mode — What NOT to Use (and why)

Honest exclusions:
- ❌ **GAN / diffusion synthetic market data**: TimeGAN/QuantGAN fail to capture stylized facts (fat tails, volatility clustering). Cannot use for DSR/PBO tests. Block bootstrap only.
- ❌ **Full reinforcement learning**: training RL policies on $175 capital is wrong-sized. Bandit + BO gets 80% of the benefit at 10% the complexity.
- ❌ **Over-complex ensemble (stacking, meta-learners)**: for a 5-10 variant pool, simple HRP weights + quorum rule outperforms stacking empirically.
- ❌ **Real-time BO during trading**: BO belongs in weekly autoresearch cycle, not live decision loop. Live decision = bandit (fast, simple, proven).

### 11.14 Rapid Mode Integration with Serial Framework

**Rapid ≠ skip framework gates.** Every RAPID decision still passes through:
- Min N per change type (§10.1) — BUT N is per-variant-paired, so accrues 8× faster with 8 variants
- DSR > 0.95 for promotion
- PBO < 0.3 for CPCV-validated configs
- Revert triggers armed before deploy
- Log to LEARNINGS.md

**The gates stay. The path to the gates gets 8× wider.**

### 11.15 Rapid Mode Decision Addendum (§10.4 extension)

```
New hypothesis / config to test?
  → Deploy in shadow-orchestrator variant pool first
  → Min 100 paired observations before any capital
  → If passes DSR/PBO → promote to challenger at 10-25% capital
  → If MAB daily rebalance gives challenger > 30% weight consistently for 2 weeks → promote to champion

Autoresearch on grid would be too slow?
  → Use Optuna TPE with 50-100 trials, same or better optimum
  → CPCV objective, DSR reporting, plateau check mandatory

Multiple variants in live?
  → MAB (Thompson Sampling via mabwiser) for capital allocation
  → Discounted TS (γ=0.98/day) for non-stationarity
  → Floor arms at 10% until ≥ 30 obs per arm

Regime change suspected?
  → Page-Hinkley detector fires → pause affected variants
  → Re-BO on post-drift data only
  → No promotion until 20+ post-drift confirmations
```

### 11.16 Implementation Phases (Project PARALLEL)

**Phase 1 (Week 1-2)**: `ShadowOrchestrator` + `VariantPool` + variant_id in trade_details + per-variant dashboard card.

**Phase 2 (Week 2-3)**: Optuna BO replace grid sweeps in `research/innovations/*_sweep*.py`.

**Phase 3 (Week 3-4)**: MABWiser allocator + Page-Hinkley drift + composite reward logging.

**Phase 4 (Month 2)**: Cross-strategy HRP for $1000 pool re-balance.

Full file plan: see `RAPID_MODEL_DISCOVERY.md` §12.

---

## References

### Primary Sources (López de Prado et al.)
- Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality." *Journal of Portfolio Management*, 40(5). SSRN: 2460551.
- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014). "The Probability of Backtest Overfitting." SSRN: 2326253. Published in *Journal of Computational Finance*.
- Bailey, D. H., & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier." *Journal of Risk*, 15(2). SSRN: 1821643.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. (Chapters 3 — Meta-Labeling; 7 — Cross-Validation in Finance; 11 — The Dangers of Backtesting; 12 — Backtesting through Cross-Validation; 14 — Backtest Statistics.)
- López de Prado, M. (2018). "The 10 Reasons Most Machine Learning Funds Fail." *Journal of Portfolio Management*, 44(6), 120–133. SSRN: 3104816.
- López de Prado, M., Lipton, A., & Zoonekynd, V. (2025). "Sharpe Ratio Inference: A New Standard for Decision-Making and Reporting." SSRN: 5520741.

### Multiple Testing in Finance
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). "…and the Cross-Section of Expected Returns." *Review of Financial Studies*, 29(1), 5–68. NBER w20592.
- Harvey, C. R., & Liu, Y. (2021). "Lucky Factors." *Journal of Financial Economics*.

### Validation Methodology
- López de Prado, M. (2018). "A Practical Solution to the Multiple Testing and Non-Normality Problems in Backtesting." SSRN: 3257497.
- Wikipedia: "Purged cross-validation" (CPCV procedure summary).
- García-Feijóo et al. (2024). "Backtest overfitting in the machine learning era: A comparison of out-of-sample testing methods in a synthetic controlled environment." *Knowledge-Based Systems*.

### Robustness & Factor Investing (AQR)
- Asness, C. S., et al. (multiple). "Lies, Damned Lies, and Data Mining." AQR Insights.
- Asness, C. S., et al. "It's Not Data Mining — Not Even Close." AQR Insights.
- AQR (2018). "Fact, Fiction, and the Size Effect." *Journal of Portfolio Management*, 45(1).

### Algorithmic Trading
- Chan, E. P. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*. Wiley.
- Chan, E. P. (2017). "Optimizing Trading Strategies Without Overfitting." Quantitative Trading blog.
- Chan, E. P. (2010). "How do you limit drawdown using Kelly formula?" Quantitative Trading blog.

### Market Microstructure / Adverse Selection
- Stoikov, S., & Saglam, M. (2025). "Optimal Quoting under Adverse Selection and Price Reading." arXiv:2508.20225.
- Various. "Limit Order Strategic Placement with Adverse Selection Risk." arXiv:1610.00261.

### Regime Detection
- Adams, R. P., & MacKay, D. J. C. (2007). "Bayesian Online Changepoint Detection." arXiv:0710.3742.
- Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2). [Foundational HMM regime-switching paper.]

### Kelly Sizing
- Kelly, J. L. (1956). "A New Interpretation of Information Rate." *Bell System Technical Journal*.
- Thorp, E. O. (2006). "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market."
- MacLean, L. C., Thorp, E. O., & Ziemba, W. T. (2010). *The Kelly Capital Growth Investment Criterion*. World Scientific.

### Model Monitoring
- Karakoulas, G. (2004). "Empirical Validation of Retail Credit-Scoring Models." *RMA Journal*. [PSI introduction in credit risk.]
- Various — current industry practice (Fiddler AI, Arize AI, NannyML) on PSI thresholds.

### Probabilistic Sharpe / MinTRL
- Bailey, D. H., & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier." *Journal of Risk*, 15(2).
- López de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out of Sample." *Journal of Portfolio Management*, 42(4).

---

### Rapid Mode (§11 — v2.0)
- [`docs/RAPID_MODEL_DISCOVERY.md`](RAPID_MODEL_DISCOVERY.md) — full 538-line study
- [Frazier — *Tutorial on Bayesian Optimization* (arXiv 1807.02811)](https://arxiv.org/abs/1807.02811)
- [Russo & Van Roy — *TS Tutorial* (Stanford)](https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf)
- [Agrawal & Goyal — *TS for MAB* (2012)](http://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf)
- [CADTS (arXiv 2410.04217)](https://arxiv.org/html/2410.04217v2)
- [Politis & Romano — *Stationary Bootstrap*](https://www.ssc.wisc.edu/~bhansen/718/Politis%20Romano.pdf)
- [López de Prado — *HRP* (SSRN 2708678)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678)
- [Two Sigma — *Head-to-Head Development*](https://www.twosigma.com/articles/introduction-to-head-to-head-development-part-1/)
- Tools: Optuna, MABWiser, river, arch, PyPortfolioOpt

## Document Maintenance

This document is the authoritative reference. Updates require:
- New academic source with verified citation, OR
- Empirical Arbo data demonstrating need for amended rule

Log all amendments in section header. Review quarterly.

**Next review: 2026-07-12**

---

## Changelog

- **v1.0** (2026-04-12): Initial serial-mode methodology — 10 sections, López de Prado / Bailey / Harvey grounding
- **v2.0** (2026-04-13): Added §11 Rapid Mode — champion-challenger, BO, MAB/Thompson Sampling, Page-Hinkley drift, composite reward, block bootstrap, HRP ensemble, hypothesis factory FSM. Based on `RAPID_MODEL_DISCOVERY.md` research. Dashboard requirement: variant leaderboard card per multi-variant strategy.
