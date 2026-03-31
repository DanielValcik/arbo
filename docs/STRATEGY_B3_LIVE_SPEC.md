# Strategy B3 Live Execution Spec

> Dual-mode (paper + live) for continuous comparison. Start small, validate fills.

## 1. Architecture: Dual Mode

B3 runs **paper AND live simultaneously** on every signal:

```
Signal → Quality Gate → Risk Check
          ↓                    ↓
    Paper Fill (instant)   Live PostOnly (10s timeout)
          ↓                    ↓
    Paper Position        Live Position (small $10-15)
          ↓                    ↓
    Paper Exit (instant)   Live Taker Sell (immediate)
          ↓                    ↓
        Compare: fill rate, price diff, slippage, PnL
```

`B3_EXECUTION_MODE`:
- `paper` (default) — paper only, as today
- `dual` — paper + live side by side, live at reduced size
- `live` — live only (future, after dual validation)

## 2. Entry Flow

### Paper (unchanged)
- `paper_engine.place_trade()` at model FV price
- Instant fill, no latency

### Live (new)
- `live_executor.buy(token_id, price, size_usdc, neg_risk=False, tick_size="0.01")`
- PostOnly maker at BUY price (bid level)
- **Timeout: 10s** (not 30s — B3 holds 1-3 min, can't wait half the trade)
- If no fill in 10s → cancel, log as `live_no_fill`
- Size: `B3_LIVE_SIZE_USD` env var (default $10)
- **Entry happens AFTER paper trade succeeds** — paper is the benchmark

### Comparison logged on entry:
- Paper fill price vs live fill price
- Fill latency (ms)
- Fill rate (filled/skipped)
- Spread at time of entry

## 3. Exit Flow

### Paper (unchanged)
- `paper_engine.sell_position()` at model FV price
- Instant, no latency

### Live (new)
- **Taker sell immediately** at BUY price — no 30s maker wait
- B3 scalper can't afford 30s exit delay on 1-3 min holds
- `live_executor.sell(token_id, price, neg_risk=False, tick_size="0.01", maker_timeout_s=5)`
- 5s maker attempt, then instant taker fallback
- If resolution: no sell needed, token redeems at $1 or $0 automatically

### Comparison logged on exit:
- Paper exit price vs live exit price
- Exit slippage (difference)
- Exit reason (same for both)
- PnL comparison: paper PnL vs live PnL

## 4. Risk Limits

| Parameter | Value | Notes |
|-----------|-------|-------|
| `B3_LIVE_SIZE_USD` | $10 | Fixed per trade, env configurable |
| Max concurrent live | 1 | Only 1 live position at a time initially |
| Daily live loss limit | $50 | Kill live if daily live losses > $50 |
| Paper continues | Always | Paper never stops, even if live is killed |

## 5. Position Tracking

`B3Position` gets new field:
```python
live_shares: int = 0        # Shares bought via live executor
live_entry_price: float = 0  # Actual fill price
live_fill_status: str = ""   # "filled", "partial", "failed", "skipped"
```

This allows `check_exits()` to know which positions have live counterparts.

## 6. Comparison Tracking

### Slack notification per trade:
```
B3 Entry: BTC UP | Edge 35% | BTC $67,500
Paper: $67.00 @ 0.595 (instant)
Live:  $10.00 @ 0.590 (maker, 2.3s) ← $0.005 better
```

### Slack notification per exit:
```
B3 Exit: profit | hold 1.2 min
Paper: +$13.85 (0.595 → 0.802)
Live:  +$2.03  (0.590 → 0.795) ← $0.007 slippage
```

### Aggregate metrics (daily Slack summary):
- Fill rate: X/Y signals filled live (Z%)
- Avg slippage: entry and exit
- Paper PnL vs Live PnL (normalized to same size)
- Recommendation: go/no-go for size increase

## 7. Configuration

```bash
# .env on VPS
B3_EXECUTION_MODE=dual          # paper|dual|live
B3_LIVE_CAPITAL=300             # Live capital base for % sizing (paper uses $1000)
                                 # Same formula: min(capital * raw_pct, MAX_BET_SIZE)
                                 # $300 × 6.7% = ~$20/trade (vs paper $1000 × 6.7% = ~$67)
B3_LIVE_DAILY_LOSS_LIMIT=50     # Kill live if daily loss > this
B3_LIVE_ENTRY_TIMEOUT_S=10      # PostOnly fill timeout
B3_LIVE_EXIT_MAKER_TIMEOUT_S=5  # Maker exit attempt before taker
```

## 8. Implementation Changes

### `live_executor.py`
- Add `maker_timeout_s` param to `buy()` and `sell()` (default=existing constants)
- B3 passes shorter timeouts

### `strategy_b3.py`
- Add `live_executor` param to `__init__`
- In `poll_cycle()`: after paper trade, call `live_executor.buy()` if dual/live
- Store live fill info on B3Position
- In `check_exits()`: no change (returns triggers, doesn't execute sells)

### `main_rdh.py`
- `_init_strategy_b3()`: read `B3_EXECUTION_MODE`, create LiveExecutor if needed
- `_run_b3_exit_check()`: after paper sell, also call `live_executor.sell()` for live positions
- Track daily live PnL, kill live if limit exceeded

## 9. Deployment

1. Commit + push to GitHub
2. Deploy to Dublin VPS
3. Set `B3_EXECUTION_MODE=dual` and `B3_LIVE_SIZE_USD=10` in .env
4. Restart service
5. Monitor Slack for first live fills
6. After 50+ dual trades → evaluate fill rate + slippage
7. If paper ≈ live → increase size or switch to live-only
