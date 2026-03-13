# autoresearch — Strategy A (Theta Decay)

This is an experiment to have the LLM optimize Strategy A parameters autonomously.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar13`). The branch `autoresearch-a/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch-a/<tag>` from current HEAD.
3. **Read the in-scope files**: The setup is small. Read these files for full context:
   - This `program.md` — your instructions.
   - `research_a/backtest_a_harness.py` — fixed backtest engine. Simulates longshot prediction markets with walk-forward validation. Do not modify.
   - `research_a/strategy_a_experiment.py` — the file you modify. Contains all tunable parameters, signal logic, quality gate, sizing, and exit rules.
4. **Verify harness works**: Run `python3 research_a/backtest_a_harness.py` and confirm it outputs metrics.
5. **Initialize results.tsv**: Create `research_a/results_a.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Context: Strategy A (Theta Decay)

Strategy A trades longshot prediction markets on Polymarket:
- **Universe**: YES contracts priced < $0.15 (longshots, overpriced due to optimism bias)
- **Entry signal**: 3σ taker flow spike (peak retail hype) → buy NO side
- **Edge model**: YES actual prob ≈ price_yes × discount_factor (market overprices longshots by ~2x)
- **Exit**: Hold to resolution (most longshots resolve NO → profit), with partial exit and stop loss
- **Sizing**: Kelly criterion with conservative fraction

The backtest harness simulates:
- ~360 longshot markets per 90-day window (Beta(1.5,20) true probability + optimism bias)
- Taker flow with periodic 3σ+ spikes (0.8% per 4-hour tick)
- Price dynamics: mean reversion + spike impact + random walk
- 5 walk-forward windows, composite scoring

Academic reference: Snowberg & Wolfers 2010, Moskowitz 2021 (favorite-longshot bias is 2-8pp).

## Experimentation

Each experiment modifies `strategy_a_experiment.py` and runs the fixed harness. The harness runs in ~1 second — extremely fast iteration.

**What you CAN do:**
- Modify `research_a/strategy_a_experiment.py` — this is the only file you edit. Everything is fair game: parameters, signal detection logic, quality gate logic, sizing formulas, exit logic, adding new filters, structural changes to the model.

**What you CANNOT do:**
- Modify `research_a/backtest_a_harness.py`. It is read-only. It contains the fixed simulation model, market generation, walk-forward engine, and composite score computation.
- Install new packages or add dependencies.

**The goal is simple: get the highest composite_score.** The composite score formula is:

```
composite_score = avg_sharpe × sqrt(total_trades/40) × (1 - max_dd/30) × consistency
```

Where:
- `avg_sharpe` = mean Sharpe ratio across 5 walk-forward windows
- `total_trades` = sum of trades across all windows (minimum 15 for non-zero score)
- `max_dd` = worst drawdown across all windows (%)
- `consistency` = profitable_windows / total_windows (5)

Higher is better. Optimize for this single metric.

**Key parameters to explore** (starting points, not exhaustive):

| Parameter | Default | Range to explore | Impact |
|-----------|---------|-----------------|--------|
| LONGSHOT_PRICE_MAX | 0.085 | 0.06-0.15 | Market universe size |
| ZSCORE_THRESHOLD | 2.9 | 2.0-5.0 | Entry selectivity |
| SPIKE_LOOKBACK_TICKS | 20 | 10-40 | Spike detection window |
| MIN_HISTORY_TICKS | 11 | 6-20 | Min data for z-score |
| DISCOUNT_FACTOR | 0.30 | 0.10-0.60 | Edge model strength |
| MIN_EDGE | 0.02 | 0.01-0.10 | Trade quality filter |
| DISCOUNT_ZSCORE_BONUS | 0.0 | 0.0-0.10 | Zscore-dependent discount |
| EDGE_ZSCORE_BONUS | 0.0 | 0.0-0.05 | Zscore-dependent edge boost |
| KELLY_FRACTION | 0.10 | 0.01-0.50 | Position sizing conservatism |
| KELLY_MULTIPLIER | 1.0 | 0.1-3.0 | Size scaling |
| POSITION_PCT_MIN | 0.02 | 0.01-0.05 | Min % of capital per trade |
| POSITION_PCT_MAX | 0.10 | 0.03-0.20 | Max % of capital per trade |
| MAX_CONCURRENT | 25 | 5-40 | Portfolio concentration |
| STOP_LOSS_PCT | 0.30 | 0.05-0.50 | Downside protection |
| TRAILING_STOP_ENABLED | False | True/False | Lock in gains |
| TRAILING_STOP_PCT | 0.15 | 0.03-0.30 | Trailing stop drawdown |
| TRAILING_STOP_ACTIVATION | 0.05 | 0.02-0.20 | Min profit to activate |
| PARTIAL_EXIT_PROFIT_PCT | 0.50 | 0.10-1.00 | Profit taking threshold |
| TIME_EXIT_ENABLED | False | True/False | Pre-resolution exit |
| ZSCORE_DAYS_PIVOT | 10 | 5-20 | Days threshold for penalty |
| ZSCORE_PENALTY_POWER | 2 | 1-3 | Linear/quadratic/cubic |
| ZSCORE_PENALTY_COEFF | 0.002 | 0.001-0.01 | Penalty strength |
| VOLUME_EDGE_ENABLED | False | True/False | Volume-based edge adj |
| RESOLUTION_SIZE_ENABLED | False | True/False | Days-aware sizing |

**IMPORTANT v2 rules:**
- Position sizing is %-based (POSITION_PCT_MIN/MAX as fractions of total_capital)
- Harness enforces 1% of initial capital ($4) as absolute floor — can't game below this
- Do NOT try to minimize position size to reduce drawdown — that exploit is blocked

**Structural changes to try:**
- DISCOUNT_ZSCORE_BONUS: stronger spike → lower discount → more edge (already parameterized)
- EDGE_ZSCORE_BONUS: direct edge boost from zscore magnitude (already parameterized)
- Volume-dependent edge: high volume → more efficient → less bias (VOLUME_EDGE_ENABLED)
- Resolution-time sizing: shorter time → higher confidence (RESOLUTION_SIZE_ENABLED)
- Trailing stop with activation: lock in gains after min profit threshold
- Multi-threshold entry: different behavior for 3σ vs 5σ spikes via ZSCORE_PENALTY params
- Cooldown period: SPIKE_COOLDOWN_TICKS prevents re-entry

**The first run**: Your very first run should always be to establish the baseline. Run the harness as-is.

## Output format

The harness prints a summary like:

```
composite_score:    13.401922
avg_sharpe:         7.5972
avg_pnl_pct:        17.95
max_drawdown_pct:   9.20
avg_win_rate:       90.1
num_trades:         259
```

Extract the key metric:
```
grep "^composite_score:" research_a/run.log
```

## Logging results

When an experiment is done, log it to `research_a/results_a.tsv` (tab-separated).

The TSV has a header row and 8 columns:

```
commit	composite_score	avg_sharpe	avg_pnl_pct	max_drawdown_pct	avg_win_rate	num_trades	status	description
```

Status: `keep`, `discard`, or `crash`.

Example:
```
commit	composite_score	avg_sharpe	avg_pnl_pct	max_drawdown_pct	avg_win_rate	num_trades	status	description
a1b2c3d	13.401922	7.5972	17.95	9.20	90.1	259	keep	baseline
b2c3d4e	18.234567	9.1234	22.30	6.50	92.3	312	keep	lower zscore threshold to 2.5
c3d4e5f	11.000000	5.0000	12.00	12.40	88.0	180	discard	raise min_edge to 0.10
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch-a/mar13`).

LOOP FOREVER:

1. Look at the git state and recent results to plan the next experiment
2. Edit `research_a/strategy_a_experiment.py` with an experimental idea
3. `git add research_a/strategy_a_experiment.py && git commit -m "experiment: <description>" --no-gpg-sign`
4. Run: `python3 research_a/backtest_a_harness.py > research_a/run.log 2>&1`
5. Read out: `grep "^composite_score:\|^avg_sharpe:\|^num_trades:\|^max_drawdown_pct:\|^avg_win_rate:\|^avg_pnl_pct:" research_a/run.log`
6. If grep output is empty, the run crashed. Run `tail -n 30 research_a/run.log` to read the error and fix it.
7. Record the results in the TSV (do NOT commit results_a.tsv — leave it untracked by git)
8. If composite_score improved (higher): KEEP the commit, advance the branch
9. If composite_score is equal or worse: `git reset --hard HEAD~1` to discard

**Crashes**: If a run crashes, fix it and re-run. If the idea is fundamentally broken, skip it, log "crash", and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human anything. The human is likely away from the computer. You are autonomous. If you run out of ideas, think harder — try combinations of previous near-misses, try radical structural changes, re-read the harness code for new angles. The loop runs until the human interrupts you, period.

Each experiment takes ~1 second. You should be able to run 100+ experiments per hour. The human expects to return to a full log of experiments and a significantly improved composite_score.

## Strategy tips from previous autoresearch runs

From Strategy C (Weather) optimization (43 → 168, +289%):
- **Aggressive filtering** was the #1 lever (MIN_FORECAST_PROB = 0.62, price range 0.30-0.43)
- **Ultra-conservative Kelly** (fraction=0.01, multiplier=0.35) dramatically reduced drawdown
- **Per-instrument calibration** added steady gains
- **Probability sharpening** (power=1.05) improved signal quality

From Strategy B (Reflexivity Surfer) optimization (19 → 416, +2100%):
- **Phase cooldown reduction** was the single biggest improvement
- **Trailing stop 4%** locked in gains effectively
- **Stop loss tightening** compounded over time
- **Many small bets > few large bets** for this strategy type
- **Slashing volume/liquidity filters** opened 4x more markets

General patterns:
- Changes that increase trade count while maintaining quality usually improve composite_score
- Ultra-conservative sizing reduces drawdown dramatically (often worth the trade-off)
- Features that reduce trade count (confidence scaling, etc.) consistently degrade score
- The sqrt(trades/N) factor rewards more trades, but quality matters through Sharpe
