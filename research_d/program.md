# Strategy D Autoresearch Protocol

## Goal
Optimize the Strategy D (Live Edge Harvester) parameters for maximum
composite score on the sports backtest dataset.

## Setup
- `prepare.py` — IMMUTABLE evaluation harness. DO NOT MODIFY.
- `strategy_params.py` — The ONLY file you may modify.
- `data/sports_backtest.sqlite` — Historical price + game data.

## Run
```bash
PYTHONPATH=. python3 research_d/prepare.py
```

## Constraints
1. Minimum 50 trades in backtest (statistical significance)
2. Win rate > 45% (green booking should help)
3. Maximum drawdown < 25%
4. Capital turnover ratio > 3x (high turnover is a key goal)
5. Both NBA and EPL must contribute trades (if both have data)
6. Green book rate > 40% (green booking IS the edge)

## Workflow
1. Read current `strategy_params.py`
2. Hypothesize a change based on previous results
3. Edit `strategy_params.py`
4. Run: `PYTHONPATH=. python3 research_d/prepare.py`
5. Record results in `results.tsv`
6. If score improved: keep
7. If score worsened: revert
8. Repeat

## Search Priorities
1. **First**: find the right `min_edge` and `green_book_delta` per sport
   - Lower min_edge = more trades but lower quality
   - Lower delta = more green books but smaller profit per trade
2. **Second**: optimize `min_price` and `max_price` ranges
   - Too narrow = too few trades
   - Too wide = bad trades (extreme favorites/longshots)
3. **Third**: test `elo_weight` vs `pinnacle_weight` balance
4. **Fourth**: Kelly sizing (`kelly_fraction`, `kelly_raw_cap`)
5. **Fifth**: sport-specific parameter overrides

## Tips
- Green book rate > 60% is ideal (means most positions exit profitably mid-game)
- If win_rate < 45%, min_edge is probably too low
- If n_trades < 30, min_edge or min_volume is too restrictive
- Watch avg_hold_fraction: lower = faster capital turnover = better
- elo_weight vs pinnacle_weight: Pinnacle is sharper but available later
- The `could_gb_rate` metric shows what % COULD have been green booked
  (even if delta was set higher than optimal). Use it to tune delta.
- If could_gb_rate >> green_book_rate, your delta is too high
- Score = 0 means pnl_factor, trade_factor, or dd_factor is zero
