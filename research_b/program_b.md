# Arbo Autoresearch — Strategy B Reflexivity Surfer Optimization

This is an experiment to have an AI agent autonomously optimize Strategy B
(Reflexivity Surfer) parameters and logic for Polymarket crypto binary markets.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar12`). The branch `autoresearch-b/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch-b/<tag>` from current HEAD.
3. **Read the in-scope files**: The research module is small. Read these files for full context:
   - `research_b/program_b.md` — this file, repository context.
   - `research_b/backtest_b_harness.py` — fixed evaluation harness. Do not modify.
   - `research_b/strategy_b_experiment.py` — the file you modify. Parameters, signals, gate, sizing.
4. **Verify data exists**: Check that `research_b/data/crypto_history.json` exists. If not, run `python3 research_b/fetch_data.py`.
5. **Initialize results_b.tsv**: Create `research_b/results_b.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs a walk-forward backtest across 5 time windows using 2 years
of historical crypto data (2024-2025) for 20 coins. You launch it as:
`python3 research_b/backtest_b_harness.py`.

**What you CAN do:**
- Modify `research_b/strategy_b_experiment.py` — this is the only file you edit. Everything is fair game:
  - **Signal parameters**: W_VOLUME, W_DAA_PROXY, MOMENTUM_LOOKBACK, DIVERGENCE_HISTORY, SIGMA_THRESHOLD, scales.
  - **Phase state machine**: BOOM_DIVERGENCE, PEAK_DIVERGENCE, BUST thresholds, PHASE_COOLDOWN.
  - **Holding period**: HOLDING_PERIOD_DAYS — try 1, 2, 3, 5, 7 days.
  - **Quality gate**: MIN_EDGE, MAX_EDGE, MIN_VOLUME_24H, MIN_LIQUIDITY, price range, confidence.
  - **Position sizing**: KELLY_FRACTION, KELLY_MULTIPLIER, KELLY_RAW_CAP, MAX_POSITION_PCT, phase caps.
  - **Exit rules**: stop losses, partial exit gain/pct.
  - **Signal computation**: compute_signals() — try different normalization, weighting, indicators.
  - **Probability model**: estimate_probability() — try different mappings from divergence to probability.
  - **Phase logic**: get_phase_transition() — add hysteresis, decay, multi-signal confirmation.

**What you CANNOT do:**
- Modify `research_b/backtest_b_harness.py`. It is read-only. It contains the fixed evaluation,
  market simulation, walk-forward windows, and data loading.
- Install new packages or add dependencies. You can only use Python standard library.
- Modify the evaluation metric. The composite_score in backtest_b_harness.py is the ground truth.

**The goal: maximize composite_score.** This metric combines:
- **Sharpe ratio** (risk-adjusted return) — higher is better
- **Number of trades** (statistical significance) — sqrt(n/30) factor
- **Max drawdown** (capital preservation) — penalizes drawdown > 40%
- **Consistency** (profitable windows) — rewards 5/5 profitability

Formula: `composite_score = avg_sharpe * sqrt(n_trades/30) * max(0, 1 - max_drawdown/40) * consistency`

Secondary goals (in priority order):
1. High avg_sharpe (average Sharpe across 5 walk-forward windows)
2. Low sharpe_std (consistency across windows)
3. High profitable_windows (ideally 5/5)
4. High win_rate_pct (> 55%)
5. Low max_drawdown_pct (< 20%)

## Domain Knowledge

### Reflexivity Theory (George Soros)
Markets are reflexive: participant perceptions influence fundamentals, which influence
perceptions, creating self-reinforcing boom-bust cycles. In crypto:
- **Boom**: Increased on-chain activity (DAA, transactions) → attracts more capital →
  price rises → attracts more activity → boom accelerates
- **Peak**: Price rises faster than fundamentals → unsustainable → exhaustion
- **Bust**: Price drops → activity drops → further price drops → cycle completes

### The Edge Mechanism
The market maker prices crypto binary markets based on recent **price momentum only**.
Our strategy additionally uses **activity momentum** (volume, DAA proxy) to detect
divergence. When activity leads price (or vice versa), we have an information advantage.

- **Positive divergence** (activity > price): Activity is rising faster than price.
  Early signal of incoming price increase → buy YES.
- **Negative divergence** (price > activity): Price is rising without commensurate
  activity growth. Unsustainable → expect reversion → buy NO.

### Data Sources (in backtest)
- **Price**: CoinGecko daily close prices for 20 coins
- **Volume**: CoinGecko 24h trading volume
- **DAA proxy**: volume/price ratio (approximates active addresses)
- **Real DAA** (optional): Santiment daily active addresses (if fetched)

In production, we additionally have Santiment DAA (real on-chain data). The backtest
uses volume/price as a proxy. Optimized parameters transfer because the mathematical
structure is identical.

### Polymarket Crypto Markets
- Binary outcomes: "Will BTC be above $X on date Y?" → YES/NO
- Typical holding: 1-7 days
- Fee model: fee = p * (1-p) * fee_rate (most crypto = fee-enabled)
- Volume varies widely: BTC/ETH markets $50K-$500K, altcoins $1K-$20K
- Liquidity: generally lower than sports/politics markets

### Key Relationships
- **Momentum lookback vs sensitivity**: Shorter = more signals but more noise.
  Longer = fewer but more reliable signals. Sweet spot likely 5-14 days.
- **Divergence threshold vs trade count**: Lower thresholds = more trades but lower
  quality. Higher = fewer but more confident. Must balance for composite_score.
- **Holding period vs win rate**: Shorter = less time for mean reversion but lower
  drawdown risk. Longer = more time for thesis to play out but more noise.
- **Sigma threshold vs selectivity**: Higher = only extreme z-scores trigger.
  Lower = more trades. The z-score distribution depends on DIVERGENCE_HISTORY.
- **Kelly fraction vs drawdown**: Higher = faster growth but larger drawdowns.
  Crypto is volatile — likely need more conservative sizing than weather.

### Pitfalls to Avoid
- **Overfitting**: Walk-forward validation protects against this. If composite_score
  improves but sharpe_std increases dramatically, you may be overfitting.
- **Too aggressive**: High Sharpe but huge drawdown → score = 0 if max_dd > 40%.
- **Too conservative**: High win rate but 5 trades total → score ≈ 0 (n < 10 → 0).
- **Phase machine stuck**: If thresholds are too extreme, phases never trigger.
  Monitor num_trades — if it drops to near zero, thresholds are too tight.
- **Regime sensitivity**: Crypto has distinct regimes (bull market 2024 Q4,
  consolidation 2024 Q3, etc.). Parameters should work across regimes.
- **Volume noise**: Crypto volume can spike 10x on news. Momentum scales should
  handle this (tanh normalization helps).

## Output format

The backtest prints a greppable summary:

```
---
composite_score:    XX.XXXXXX
avg_sharpe:         XX.XXXX
sharpe_std:         XX.XXXX
avg_pnl_pct:        XX.XX
max_drawdown_pct:   XX.XX
avg_win_rate:       XX.X
num_trades:         XXX
avg_profit_factor:  XX.XXXX
avg_return_pct:     XX.XX
profitable_windows: X/5
backtest_seconds:   X.X
```

Extract the key metric: `grep "^composite_score:" research_b/run.log`

## Logging results

When an experiment is done, log it to `research_b/results_b.tsv` (tab-separated).

The TSV has a header row and 9 columns:

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
a1b2c3d	5.123456	2.35	15.00	12.30	58.1	87	keep	baseline
b2c3d4e	6.789012	2.81	18.50	10.20	61.3	95	keep	BOOM_DIVERGENCE -0.15
c3d4e5f	3.456789	1.80	8.00	22.00	52.1	120	discard	SIGMA_THRESHOLD 0.5 too loose
d4e5f6g	0.000000	0.00	0.00	0.00	0.0	0	crash	syntax error in compute_signals
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch-b/mar12`).

LOOP FOREVER:

1. Look at the current state: git log, results_b.tsv, what has been tried.
2. Formulate a hypothesis about what change might improve composite_score.
3. Edit `research_b/strategy_b_experiment.py` with the change.
4. git commit with a descriptive message.
5. Run the experiment: `python3 research_b/backtest_b_harness.py > research_b/run.log 2>&1`
6. Read results: `grep "^composite_score:\|^avg_sharpe:\|^num_trades:\|^max_drawdown_pct:\|^avg_win_rate:\|^profitable_windows:" research_b/run.log`
7. If grep output is empty, the run crashed. Run `tail -n 30 research_b/run.log` to debug.
8. Record results in `research_b/results_b.tsv` (do NOT git commit the tsv, leave untracked).
9. If composite_score improved: KEEP — advance the branch.
10. If composite_score is equal or worse: DISCARD — `git reset --hard HEAD~1`.

**Timeout**: Each backtest should take < 60 seconds. If it exceeds 120 seconds, kill it.

**Crashes**: Fix typos/bugs and re-run. If the approach is fundamentally broken, discard and move on.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. Run autonomously
until manually interrupted. If you run out of ideas, think harder — try:
- Combining parameters that individually helped
- More radical changes (different signal computation, adaptive thresholds)
- Analyzing which windows perform worst and targeting those
- Reading the domain knowledge section above for new angles

## Strategy ideas to try (starting points)

### Signal Computation
1. **MOMENTUM_LOOKBACK**: Try 3, 5, 7, 10, 14 days
2. **DIVERGENCE_HISTORY**: Try 14, 20, 30, 50 days
3. **Weight rebalancing**: Try W_VOLUME from 0.3 to 0.7 (W_DAA = 1 - W_VOLUME)
4. **Different normalization**: Instead of tanh, try sigmoid, log, or linear clamped
5. **EMA instead of simple delta**: Exponential moving average for momentum
6. **Multi-timeframe**: Combine 3-day and 14-day divergence signals
7. **Volume-price correlation**: Track rolling correlation as additional signal
8. **Rate of change of divergence**: Second derivative as confirmation

### Phase Machine
9. **BOOM_DIVERGENCE**: Try -0.05, -0.10, -0.15, -0.20
10. **PEAK_DIVERGENCE**: Try 0.10, 0.15, 0.20, 0.30
11. **Asymmetric thresholds**: Boom and peak may need different sensitivity
12. **SIGMA_THRESHOLD**: Try 0.5, 0.75, 1.0, 1.5, 2.0
13. **Add hysteresis**: Require divergence to stay beyond threshold for N days
14. **Phase 4 (BUST) tuning**: Adjust transition thresholds
15. **Remove Phase 4**: Simplify to just BOOM and PEAK

### Quality Gate
16. **MIN_EDGE**: Try 0.03, 0.05, 0.08, 0.10
17. **Price range**: Try MIN/MAX_MARKET_PRICE combinations
18. **MIN_CONFIDENCE**: Try 0.2, 0.3, 0.4, 0.5
19. **MIN_DIVERGENCE_ABS**: Try 0.03, 0.05, 0.08, 0.10
20. **Volume/liquidity thresholds**: Scale based on coin tier

### Position Sizing
21. **KELLY_FRACTION**: Try 0.10, 0.15, 0.25, 0.35
22. **KELLY_MULTIPLIER**: Try 0.25, 0.35, 0.50, 0.75
23. **Phase caps**: Adjust PHASE2/PHASE3_MAX_POSITION
24. **Remove phase-specific caps**: Use only global MAX_POSITION_PCT
25. **Edge-weighted sizing**: Scale Kelly by confidence or divergence magnitude

### Holding Period
26. **HOLDING_PERIOD_DAYS**: Try 1, 2, 3, 5, 7 days
27. **Phase-specific holding**: Shorter for BOOM (quick reversal), longer for PEAK

### Exit Rules
28. **Stop losses**: Try 0.10, 0.15, 0.20, 0.30 for each phase
29. **Partial exit**: Try different gain thresholds (0.20, 0.30, 0.50)
30. **No partial exit**: Remove partial exit completely
31. **Trailing stop**: Replace fixed stop with trailing stop logic

### Probability Model
32. **Linear probability**: prob = 0.50 + k * divergence * confidence
33. **Power mapping**: prob = 0.50 + sign * |div|^alpha * confidence
34. **Logistic mapping**: prob = sigmoid(k * divergence * z_score)
35. **Calibrated mapping**: Different sensitivity for different z-score ranges

### Per-Coin Optimization
36. **Coin-specific weights**: BTC/ETH may need different signal weights than altcoins
37. **Volatility-adjusted thresholds**: Higher vol coins need wider divergence thresholds
38. **Exclude low-signal coins**: Skip coins that consistently produce losing trades
39. **Coin groups**: Large-cap vs mid-cap vs small-cap parameter sets
