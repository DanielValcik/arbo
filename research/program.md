# Arbo Autoresearch — Strategy C Weather Optimization

This is an experiment to have an AI agent autonomously optimize Strategy C
(Compound Weather) parameters and logic for Polymarket weather markets.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar10`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD.
3. **Read the in-scope files**: The research module is small. Read these files for full context:
   - `research/README.md` — this file, repository context.
   - `research/backtest_harness.py` — fixed evaluation harness. Do not modify.
   - `research/strategy_experiment.py` — the file you modify. Parameters, probability model, quality gate, sizing.
4. **Verify data exists**: Check that `research/data/weather_history.json` exists. If not, run `python3 research/backtest_harness.py` once to download it.
5. **Initialize results.tsv**: Create `research/results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs a walk-forward backtest across 5 time windows using 2 years
of historical weather data (2024-2025) for 5 cities (NYC, Chicago, London, Seoul,
Buenos Aires). You launch it as: `python3 research/backtest_harness.py`.

**What you CAN do:**
- Modify `research/strategy_experiment.py` — this is the only file you edit. Everything is fair game:
  - **Parameters**: FORECAST_SIGMA, STUDENT_T_DF, MIN_EDGE, KELLY_FRACTION, etc.
  - **Probability model**: estimate_probability() — try normal CDF, Student-t, KDE, ensemble, etc.
  - **Quality gate**: should_trade() — add/remove/reorder filters, change thresholds.
  - **Position sizing**: position_size() — try Kelly variants, fixed fraction, adaptive sizing.
  - **Days out selection**: DAYS_OUT_TO_TRADE — which lead times to evaluate.

**What you CANNOT do:**
- Modify `research/backtest_harness.py`. It is read-only. It contains the fixed evaluation,
  market simulation, walk-forward windows, and data loading.
- Install new packages or add dependencies. You can only use Python standard library.
- Modify the evaluation metric. The composite_score in backtest_harness.py is the ground truth.

**The goal: maximize composite_score.** This metric combines:
- **Sharpe ratio** (risk-adjusted return) — higher is better
- **Number of trades** (statistical significance) — sqrt(n/100) factor
- **Max drawdown** (capital preservation) — penalizes drawdown > 50%

Formula: `composite_score = sharpe * sqrt(n_trades/100) * max(0, 1 - max_drawdown/50)`

Secondary goals (in priority order):
1. High avg_sharpe (average Sharpe across 5 walk-forward windows)
2. Low sharpe_std (consistency across windows)
3. High profitable_windows (ideally 5/5)
4. High win_rate_pct (> 55%)
5. Low max_drawdown_pct (< 15%)

## Domain Knowledge

### Weather Markets on Polymarket
- Binary outcomes: "Will high temperature in NYC on March 15 be between 70-74°F?" → YES/NO
- Each city/date has ~8 temperature buckets covering the full range
- Buckets are ~5°F (2.5°C) wide, centered on climatological average
- Markets resolve based on official weather station readings
- Most weather markets are fee-free (no taker fees)

### Where Edge Comes From
1. **Better forecasts**: Our weather data sources (NOAA, Met Office, Open-Meteo) are more
   accurate than what casual Polymarket traders use.
2. **Better probability model**: Converting forecast → bucket probability using appropriate
   CDF with calibrated uncertainty (sigma) per days_out.
3. **Better quality gates**: Only trading when confidence is high, avoiding traps.
4. **Better sizing**: Kelly criterion with appropriate fraction to balance growth vs risk.

### Key Relationships
- **days_out vs accuracy**: Forecasts degrade with lead time. Day 0-1 are very accurate,
  day 3+ rapidly loses skill. The sigma model should reflect this.
- **edge vs win rate**: Higher min_edge = fewer trades but higher win rate.
  Lower min_edge = more trades but lower quality. Find the sweet spot.
- **Kelly fraction vs drawdown**: Higher Kelly = faster growth but larger drawdowns.
  Quarter-Kelly (0.25) is conservative. Try 0.15-0.40 range.
- **Conviction ratio**: forecast_prob/market_price threshold. Higher = fewer but better trades.

### Pitfalls to Avoid
- **Overfitting**: Walk-forward validation protects against this. If composite_score
  improves but sharpe_std increases dramatically, you may be overfitting to specific windows.
- **Too aggressive**: High Sharpe but huge drawdown → composite_score penalizes this.
- **Too conservative**: High win rate but 5 trades total → composite_score near 0 (n<10 → 0).
- **Tail risk**: Weather has fat tails (extreme events). Student-t is usually better than normal.

## Output format

The backtest prints a greppable summary:

```
---
composite_score:    24.090813
avg_sharpe:         6.3519
sharpe_std:         0.2320
avg_pnl_pct:        21973.80
max_drawdown_pct:   18.90
avg_win_rate:       34.9
num_trades:         3718
avg_profit_factor:  4.0543
avg_return_pct:     178.54
profitable_windows: 5/5
backtest_seconds:   0.1
```

Extract the key metric: `grep "^composite_score:" run.log`

## Logging results

When an experiment is done, log it to `research/results.tsv` (tab-separated).

The TSV has a header row and 8 columns:

```
commit	composite_score	avg_sharpe	avg_pnl_pct	max_dd	win_rate	n_trades	status	description
```

1. git commit hash (short, 7 chars)
2. composite_score — the primary optimization target
3. avg_sharpe — average Sharpe across 5 windows
4. avg_pnl_pct — average PnL % per window
5. max_drawdown_pct — worst drawdown across windows
6. avg_win_rate — average win rate %
7. num_trades — total trades across all windows
8. status: `keep`, `discard`, or `crash`
9. short description of what this experiment tried

Example:

```
commit	composite_score	avg_sharpe	avg_pnl_pct	max_dd	win_rate	n_trades	status	description
a1b2c3d	24.090813	6.35	21973.80	18.90	34.9	3718	keep	baseline
b2c3d4e	25.123456	6.52	22500.00	16.20	36.1	3650	keep	lower MIN_EDGE to 0.08
c3d4e5f	20.345678	5.80	18000.00	25.00	32.1	4200	discard	MIN_EDGE=0.03 too aggressive
d4e5f6g	0.000000	0.00	0.00	0.00	0.0	0	crash	syntax error in estimate_probability
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar10`).

LOOP FOREVER:

1. Look at the current state: git log, results.tsv, what has been tried.
2. Formulate a hypothesis about what change might improve composite_score.
3. Edit `research/strategy_experiment.py` with the change.
4. git commit with a descriptive message.
5. Run the experiment: `python3 research/backtest_harness.py > run.log 2>&1`
6. Read results: `grep "^composite_score:\|^sharpe_ratio:\|^num_trades:\|^max_drawdown_pct:\|^win_rate_pct:\|^profitable_windows:" run.log`
7. If grep output is empty, the run crashed. Run `tail -n 30 run.log` to debug.
8. Record results in `research/results.tsv` (do NOT git commit the tsv, leave untracked).
9. If composite_score improved: KEEP — advance the branch.
10. If composite_score is equal or worse: DISCARD — `git reset --hard HEAD~1`.

**Timeout**: Each backtest should take < 60 seconds. If it exceeds 120 seconds, kill it.

**Crashes**: Fix typos/bugs and re-run. If the approach is fundamentally broken, discard and move on.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. Run autonomously
until manually interrupted. If you run out of ideas, think harder — try:
- Combining parameters that individually helped
- More radical changes (different CDF, adaptive thresholds)
- Analyzing which windows perform worst and targeting those
- Reading the domain knowledge section above for new angles

## Strategy ideas to try (starting points)

1. **Lower MIN_EDGE** from 0.10 to 0.08 or 0.06 — may unlock more trades
2. **Increase KELLY_FRACTION** from 0.25 to 0.30-0.35 — more aggressive sizing
3. **Tune FORECAST_SIGMA** — the uncertainty model is critical. Try tighter sigmas for day 0-1
4. **Remove CONVICTION_RATIO** — it may be filtering good trades
5. **Widen price range** — MAX_PRICE from 0.85 to 0.90
6. **Try normal distribution** instead of Student-t — lighter tails may be better calibrated
7. **Adaptive sigma** — sigma = base + days_out * slope (linear model)
8. **Edge-weighted Kelly** — multiply Kelly fraction by confidence factor
9. **Days out selection** — try [1, 2] only (skip day 3+), or [0, 1] only
10. **Ensemble probability** — average normal and Student-t estimates

### Per-city optimization (KEY DIFFERENTIATOR)

Each city has different forecast accuracy and weather patterns. Use `CITY_SIGMA` and
`CITY_OVERRIDES` to customize per city:

- **NYC & Chicago**: NOAA data (most accurate). Tighter sigma → more confident → more trades.
- **London**: Met Office data (good accuracy). Moderate sigma.
- **Seoul & Buenos Aires**: Open-Meteo only (less accurate). Wider sigma → more conservative.

Cities also differ in temperature volatility:
- **Buenos Aires**: Subtropical, low daily variance → tighter buckets hit more often
- **Chicago**: Continental, extreme swings → wider sigma needed
- **London**: Maritime, moderate → middle ground
- **Seoul**: Continental with monsoon season → seasonal sigma adjustments could help

Ideas:
11. **CITY_SIGMA** for NOAA cities — tighter sigma for nyc/chicago (better data = more confidence)
12. **CITY_OVERRIDES min_edge** — lower min_edge for cities with better data quality
13. **City-specific Kelly** — higher Kelly for cities where our forecast is most accurate
14. **Exclude weak cities** — skip buenos_aires/seoul if they drag down Sharpe
15. **Seasonal city weights** — different parameters for summer vs winter per city
