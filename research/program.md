# Arbo Autoresearch — Strategy C Weather Markets

Autonomous AI-driven parameter & logic optimization for Polymarket weather markets.
Uses the chronological portfolio simulator (experiment_framework.py) with 571K real
price points from Goldsky + CLOB data across 420 days.

## Setup

1. **Agree on a run tag**: e.g. `mar14`. Branch: `autoresearch-b/<tag>`.
2. **Create the branch**: `git checkout -b autoresearch-b/<tag>` from current HEAD.
3. **Read the in-scope files**:
   - `research/program.md` — this file.
   - `research/experiment_framework.py` — chronological portfolio simulator. **READ-ONLY.**
   - `research/strategy_experiment.py` — probability model, city sigma/bias constants. **READ-ONLY reference.**
   - `research/run_experiment.py` — single experiment runner. Takes JSON params, outputs score.
4. **Verify data**: `research/data/price_history.sqlite` must exist (571K price points).
5. **Initialize results.tsv**: header row only.
6. **Establish baseline**: run with current production params.

## What You Optimize

The experiment_framework simulates a portfolio hour-by-hour with:
- Concurrent positions, compound sizing (MAX_POSITION_USD=$200 cap)
- Entry via quality gate (min_edge, max_price, min_price, min_volume, etc.)
- Exit logic (edge-based, profit-take, prob floor)
- Capital utilization tracking

**The goal: maximize experiment score (0-200).** Components:
- ROI (25%) — capped at 500% = max 2.0
- Sharpe (15%) — capped at Sharpe 10 = max 2.0
- Max DD penalty (10%) — 0% DD = 1.0, 50% DD = 0.0
- Capital utilization (15%) — % of hours with open positions, 20% = max 2.0
- PnL per hour in trade (15%) — capped at $5/h = max 2.0
- Trade count (10%) — 100+ trades = max 2.0
- OOS validation (10%) — walk-forward OOS PnL

**Mandatory filters** (score = 0 if any fail):
- trades < 10
- max_drawdown > 50%
- Sharpe < 0.5

**Equal optimization goals**: profitability AND capital turnover. Capital must work, not sit idle.

## Parameter Space

```python
# ── Quality Gate ──
"min_edge":        [0.005 ... 0.15]     # min edge to enter (model_prob - market_price)
"max_edge":        [0.30 ... 0.90]      # max edge (too high = suspicious)
"max_price":       [0.25 ... 0.90]      # max market price to buy
"min_price":       [0.05 ... 0.30]      # min market price (floor against pennies)
"min_prob":        [0.01 ... 0.50]      # min model probability to enter
"min_volume":      [0 ... 2000]         # min market volume ($)
"kelly_raw_cap":   [0.10 ... 0.60]      # cap on raw Kelly fraction
"prob_sharpening": [0.70 ... 1.50]      # prob = prob^sharpening (>1 = sharpen, <1 = smooth)
"shrinkage":       [0.0 ... 0.25]       # Bayesian shrinkage toward 0.125

# ── Per-City ──
"excluded_cities": subset of 20 cities  # skip cities that hurt
"city_overrides":  {city: {min_edge, max_price, min_price}}  # per-city gates

# ── Exit ──
"exit_enabled":             bool
"min_hold_edge":            [0.0 ... 0.20]   # exit if updated edge falls below
"prob_exit_floor":          [0.0 ... 0.50]   # exit if prob drops below
"profit_take_enabled":      bool
"profit_take_threshold":    [0.30 ... 3.00]  # exit at this profit multiple
"profit_take_min_hours":    [2 ... 12]       # min hours before profit-take
"reentry_enabled":          bool
"reentry_cooldown_h":       [1 ... 6]        # hours before re-entering same city
```

## Running an Experiment

```bash
python3 research/run_experiment.py '{
  "min_edge": 0.08, "max_edge": 0.42, "max_price": 0.43,
  "min_price": 0.15, "min_prob": 0.10, "min_volume": 500,
  "kelly_raw_cap": 0.40, "prob_sharpening": 1.05, "shrinkage": 0.03,
  "exit_enabled": true, "min_hold_edge": 0.05
}'
```

Output (greppable):
```
---
score:              159.50
trades:             229
win_rate:           43.2
total_pnl:          21863.71
roi_pct:            2186.4
max_drawdown_pct:   11.9
sharpe:             10.25
capital_utilization: 14.6
avg_pnl_per_hour:   13.55
total_exits:        199
exit_saves:         144
exit_regrets:       55
---
```

Extract: `grep "^score:" run.log`

## Logging Results

Log to `research/results.tsv` (tab-separated, NOT comma-separated):

```
commit	score	trades	win_rate	pnl	dd	sharpe	util	status	description
```

- status: `keep`, `discard`, or `crash`
- Do NOT commit results.tsv (leave untracked)

## The Experiment Loop

LOOP FOREVER:

1. Read current state: results.tsv, git log, what's been tried.
2. **THINK**: Formulate a hypothesis. WHY will this change improve the score?
3. Run the experiment with new params.
4. Read results.
5. If score improved → KEEP (log as keep, this is now the new baseline).
6. If score equal or worse → DISCARD (log as discard).
7. **NEVER STOP.** Do not ask the human. Run until interrupted.

**Timeout**: Each experiment takes < 5 seconds. If > 30 seconds, kill it.

## Domain Knowledge

### Weather Markets on Polymarket
- Binary: "Will NYC high temp on Mar 15 be 70-74F?" → YES/NO
- ~8 buckets per city/date covering full temperature range
- 20 cities: NYC, Chicago, London, Seoul, Buenos Aires, Atlanta, Toronto, Ankara,
  Sao Paulo, Miami, Paris, Dallas, Seattle, Wellington, Tokyo, Munich, LA, DC, Tel Aviv, Lucknow
- Resolution via METAR airport data
- Most weather markets: 0% fee

### Where Edge Comes From
1. Better forecasts (Open-Meteo archive vs casual traders)
2. Calibrated uncertainty per city (CITY_SIGMA in strategy_experiment.py)
3. Quality gate filtering (only trade when model is confident)
4. Kelly sizing with caps

### Key Relationships
- **min_price vs returns**: Lower min_price allows cheap contracts with huge upside (10-100x)
  but lower win rate. Higher min_price = safer but smaller returns.
- **min_edge vs trade count**: Lower = more trades but lower quality. Higher = fewer but better.
- **utilization vs idle**: More concurrent positions = higher utilization. Achieved by:
  wider gates (more cities, wider price range, lower min_edge)
- **exit logic**: Exits free capital for re-entry. Good exits improve utilization AND protect capital.
- **prob_sharpening**: >1.0 makes model more decisive (pushes probs toward 0/1). <1.0 smooths.
- **shrinkage**: Bayesian pull toward 12.5% (prior). Higher = more conservative.

### Current Best (baseline to beat)
Score ~167, 272 trades, 41% WR, $28K PnL, 12.7% DD, Sharpe 9.9, 20% utilization.
Key params: min_edge=0.10, min_price=0.05, max_price=0.50, prob_sharpening=0.90,
exit_enabled=True, min_hold_edge=0, prob_exit_floor=0.25, profit_take=True @1.5x after 4h.

### Ideas to Explore
1. Utilization is only 20% — 80% of time no positions open. Can we increase?
2. Try 36h or 48h entry timing (not just 24h) — more time to enter
3. Different probability sharpening per city (sharpening_overrides?)
4. Profit-take tuning — threshold too high? Lower to 0.50-1.00?
5. Re-entry enabled with short cooldown — re-enter cities after exit
6. More aggressive gates for high-confidence cities (NYC, London, Chicago)
7. Relaxed gates for cities we currently exclude
8. Edge-adaptive Kelly — higher Kelly when edge is larger
9. Seasonal adjustments — different params for summer vs winter
10. Walk-forward stability — maximize OOS consistency across folds
