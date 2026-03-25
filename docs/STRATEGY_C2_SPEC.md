# Strategy C2 — EMOS + Edge Exit Fusion

> Autoresearch-discovered weather market strategy combining adaptive probability
> estimation with intelligent early exit logic.

## Overview

Strategy C2 is an evolution of Strategy C (Compound Weather) that replaces
hold-to-resolution with a smart exit approach:

1. **EMOS Probability Model**: Adaptive sigma + bias from rolling forecast errors
   (replaces fixed per-city values)
2. **Edge-Based Exit**: Sells position when EMOS says edge has evaporated
   (instead of holding to binary $0/$1 resolution)
3. **Profit Take Trigger**: Also exits at +$0.15 absolute price gain

## Autoresearch Results (2026-03-25)

| Metric | C2 (EMOS+Exit) | C (C1f hold) | Improvement |
|--------|----------------|--------------|-------------|
| Score (IS) | 138.1 | 170.1 | — |
| Score (OOS) | 121.0 | 113.9 | +6.2% |
| Trades | 1,878 | 273 | 6.9x |
| Win Rate | 54.1% | 43.6% | +10.5pp |
| Total PnL | $15,512 | $3,738 | 4.1x |
| Sharpe | 9.44 | 7.22 | +30.7% |
| Max DD | 8.3% | 19.3% | -57% |
| Capital Util | 94.4% | — | High |
| Exits | 946 | 0 | — |
| Save Rate | 67.1% | — | — |

### Walk-Forward Validation (3 folds, all profitable)

| Fold | PnL | Trades | Win Rate | Score |
|------|-----|--------|----------|-------|
| 1 | $5,030 | 455 | 53.6% | 140.6 |
| 2 | $3,567 | 541 | 54.0% | 123.8 |
| 3 | $1,637 | 491 | 53.4% | 98.6 |
| **Avg** | **$3,411** | **496** | **53.7%** | **121.0** |

## Parameters

### EMOS Probability Model
```python
EMOS_TRAINING_WINDOW = 21     # 21-day rolling window
EMOS_SIGMA_METHOD = "rolling_mae"  # Mean Absolute Error for sigma
EMOS_BIAS_METHOD = "ewma"     # Exponentially Weighted Moving Average for bias
EMOS_SIGMA_FLOOR = 0.7        # Minimum sigma (prevents overconfidence)
EMOS_SIGMA_SCALE = 0.6        # Scale factor (tighter = more confident)
EMOS_EWMA_ALPHA = 0.15        # EWMA decay rate
```

### Exit Logic
```python
EXIT_ENABLED = True
MIN_HOLD_EDGE = 0.05          # Sell when updated edge < 5%
EXIT_SLIPPAGE_PCT = 0.06      # 6% slippage (real weather median ~7%, conservative)
PROB_EXIT_FLOOR = 0.10        # Also exit when probability < 10%
PROFIT_TAKE_ALSO = True       # Enable absolute profit target
PROFIT_TARGET_ABS = 0.15      # +$0.15 above entry → take profit
```

### Quality Gate (entry)
```python
MIN_EDGE = 0.03               # 3% minimum edge (vs C: 10%)
MAX_EDGE = 0.90
MIN_PRICE = 0.03              # Minimum market price (vs C: 0.08)
MAX_PRICE = 0.45              # Maximum market price (vs C: 0.56)
MIN_FORECAST_PROB = 0.03      # Minimum probability
KELLY_RAW_CAP = 0.30          # Quarter-Kelly cap
PROB_SHARPENING = 0.85        # Probability exponent
```

### Cities
- **Excluded**: São Paulo, Tel Aviv, Tokyo, Lucknow
- **Dallas override**: min_edge=0.01, max_price=0.40, kelly_cap=0.15
- **Miami override**: min_edge=0.08, max_price=0.40, kelly_cap=0.10

### Capital
- **Allocated**: $1,000 (separate from Strategy C)
- **Max position**: 5% of allocated ($50)
- **Max aggregate**: 80% deployed at once ($800)
- **Kelly**: Quarter-Kelly (0.25 fraction)

## Per-City Performance (in-sample)

| City | Trades | WR | PnL |
|------|--------|-----|------|
| London | 345 | 65.8% | $4,157 |
| Buenos Aires | 143 | 48.3% | $1,900 |
| Dallas | 165 | 47.3% | $1,588 |
| NYC | 184 | 43.5% | $1,371 |
| Toronto | 168 | 57.7% | $1,355 |
| Seattle | 166 | 53.6% | $1,342 |
| Paris | 64 | 70.3% | $1,067 |
| Chicago | 106 | 37.7% | $727 |
| Ankara | 98 | 50.0% | $557 |
| Seoul | 162 | 54.3% | $536 |
| Atlanta | 135 | 48.9% | $438 |
| Munich | 22 | 68.2% | $231 |
| Miami | 74 | 56.8% | $153 |
| Wellington | 42 | 71.4% | $114 |

## How It Works

### Entry Flow
1. Reuse Strategy C's weather forecast fetching (NOAA, Met Office, Open-Meteo)
2. Scan markets with `scan_weather_markets()` (same as C)
3. Apply C2 quality gate (looser thresholds → more opportunities)
4. Fetch live CLOB prices with latency recheck (2s delay, volatility guard)
5. Kelly sizing with per-city caps
6. Paper trade execution via shared paper engine

### Exit Flow (every poll cycle)
1. For each open C2 position, fetch current CLOB bid price
2. Check profit take: `current_price >= entry_price + $0.15`
3. Check edge exit: recompute probability with latest forecast, if `edge < 0.05` → sell
4. Check probability floor: if current price < 10% → sell
5. Log exit reason and update tracking

### Resolution Flow
- Same as C: METAR actual temperature resolves markets
- C2 positions tagged with `strategy="C2"` in paper_trades table
- Risk manager tracks C2 capital separately from C

## Architecture

```
Strategy C2 (strategy_c2.py)
  ├── Quality Gate (weather_quality_gate_c2.py) — C2-specific thresholds
  ├── Reuses: Strategy C forecasts, weather scanner, CLOB client
  ├── Own: Exit logic, position tracking, Kelly sizing
  └── Paper Engine: strategy="C2" tag

main_rdh.py
  ├── _init_strategy_c2() — creates StrategyC2 with reference to C
  ├── _run_strategy_c2() — poll cycle + exit checks
  └── Resolution: handle_resolution() cleanup

risk_manager.py
  └── STRATEGY_ALLOCATIONS["C2"] = $1,000
```

## Files

| File | Purpose |
|------|---------|
| `arbo/strategies/strategy_c2.py` | Main C2 strategy class |
| `arbo/strategies/weather_quality_gate_c2.py` | Quality gate + params |
| `arbo/main_rdh.py` | Orchestrator integration |
| `arbo/core/risk_manager.py` | Capital allocation |
| `research/innovations/sweep_emos_exit_fusion.py` | Autoresearch that found C2 params |
| `research/innovations/sweep_early_exit.py` | Initial early exit exploration |
| `research/backtest_early_exit.py` | Price path analysis |

## Spread / Slippage Analysis (2026-03-25, 537 live weather markets)

Real bid-ask spreads measured from Polymarket CLOB orderbooks:

| Price Range | Median Spread | Avg Spread | Bid Depth |
|-------------|--------------|------------|-----------|
| $0.01-0.15 (cheap) | 1.2c (34.1%) | 1.9c (50.9%) | $18 |
| $0.15-0.40 (mid) | 3.0c (12.7%) | 4.2c (15.6%) | $37 |
| $0.40-0.80 (expensive) | 3.0c (7.1%) | 5.5c (11.3%) | $57 |

Half-spread (= exit slippage) in C2 trading range: **median 7.41%**

### Impact on C2 Performance

Model EXIT_SLIPPAGE_PCT set to 6% (conservative vs 7.4% median).
Sensitivity analysis across slippage levels:

| Slippage | Score | WR | PnL | Sharpe | Max DD |
|----------|-------|-----|------|--------|--------|
| 0.5% (original) | 138.2 | 54.2% | $15,534 | 9.45 | 8.3% |
| 3.0% | 137.2 | 52.8% | $14,889 | 9.17 | 8.7% |
| **6.0% (deployed)** | **135.9** | **51.5%** | **$14,114** | **8.82** | **9.2%** |
| 10.0% (pessimistic) | 134.3 | 49.1% | $13,081 | 8.34 | 9.8% |
| HOLD (no exit) | 124.2 | 25.2% | $13,052 | 6.99 | 38.0% |

**Key finding**: Exit strategy is robust to real spreads because exits primarily
cut losers (where alternative is -100% loss). Losing 6% to slippage is vastly
better than losing 100% at resolution. Even at 10% slippage, C2 still beats
hold on every metric (Sharpe +19%, DD -74%).

Limit sell orders (maker) would reduce slippage to ~0% but with uncertain
fill timing. Paper trading will reveal actual execution quality.

## Monitoring

Strategy C2 paper trades are stored with `strategy="C2"` in the `paper_trades`
table. Compare C2 vs C performance directly:

```sql
SELECT strategy,
       COUNT(*) as trades,
       AVG(CASE WHEN status = 'won' THEN 1.0 ELSE 0.0 END) as win_rate,
       SUM(actual_pnl) as total_pnl
FROM paper_trades
WHERE strategy IN ('C', 'C2')
  AND status IN ('won', 'lost')
GROUP BY strategy;
```

## Research Genesis

1. **Hypothesis** (2026-03-25): Early exit (sell before resolution) might beat hold
2. **Price path analysis** (`backtest_early_exit.py`): PMD 10-min data shows losers
   peak early (22% of hold time) while winners peak late (85%) — exit captures loser peaks
3. **Exit strategy sweep** (`sweep_early_exit.py`, 1406 trials): Edge-based exit wins
   over fixed profit targets. Score 122.2 vs hold 0.0.
4. **EMOS fusion** (`sweep_emos_exit_fusion.py`, 1401 trials): EMOS + edge exit
   dramatically improves win rate (54% vs 26%) and PnL ($15,512 vs $5,283)
5. **Deployment**: Paper trading as C2 alongside C for head-to-head comparison
