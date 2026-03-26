# C2 Live Trading Spec — EMOS Exit Fusion

> Deployment plan for transitioning Strategy C2 from paper to live trading.
> Initial phase: $100 capital, full monitoring, kill switch at 25% weekly DD.

## 1. Overview

Strategy C2 trades weather temperature markets on Polymarket using:
- EMOS adaptive probability model for entry signals
- Edge-based early exit (sell when edge < 5%)
- Profit take at +$0.15 price move
- Probability floor exit at price < $0.10

Paper results (first day, 2026-03-26):
- 31 trades, 31W/0L, +$160 P&L
- Avg hold time: 5-30 minutes (fast turnover)
- NYC is top performer (+$103), Dallas marginal (+$3.34)

## 2. Capital & Sizing — Paper vs Live

| Parameter | Paper C2 (current) | Live C2 (initial) | Notes |
|-----------|-------------------|-------------------|-------|
| Capital | $1,000 | $100 | Live starts small for validation |
| Max position (5%) | $50 | $5 | Same percentage |
| Max aggregate (80%) | $800 | $80 | Same percentage |
| Kelly fraction | 0.25 | 0.25 | Quarter-Kelly |
| Kelly raw cap | 0.30 | 0.30 | Same |
| Weekly kill switch (25%) | $250 | $25 | Same percentage |
| Max trades/scan | 3 | 3 | Same |
| Min hours to resolution | 6h | 6h | Same |
| Entry scan interval | 5 min | 5 min | Same |
| Exit monitor interval | 60s | 60s | Same |
| Excluded cities | SP, TA, TK, LK | SP, TA, TK, LK | Same |
| Quality gate (min_edge) | 0.03 | 0.03 | Same |
| Price range | [0.03, 0.45] | [0.03, 0.45] | Same |
| Prob sharpening | 0.85 | 0.85 | Same |
| Min hold edge (exit) | 0.05 | 0.05 | Same |
| Profit target (exit) | +$0.15 | +$0.15 | Same |
| Prob floor (exit) | 0.10 | 0.10 | Same |

All parameters identical except capital. Live uses $100 to limit max loss
during validation phase. Once validated, scale to $500-1000.

## 3. Execution Flow (Live)

```
C2 Entry Scan (every 5 min)
  1. Fetch forecasts (NOAA + Open-Meteo)
  2. Scan weather markets (Gamma API)
  3. Quality gate (C2 thresholds)
  4. Fetch CLOB orderbook → estimate fill price
  5. Kelly sizing
  6. Submit GTC order via py-clob-client
     → Record: order_id, submitted_price, timestamp
  7. Poll for fill confirmation (or timeout 30s)
     → Record: actual_fill_price, gas_used, fill_time

C2 Exit Monitor (every 60s)
  1. Fetch CLOB bid prices for open positions
  2. Check exit triggers (edge_lost, profit_take, prob_floor)
  3. If triggered: submit SELL order (GTC or FOK)
     → Record: sell_order_id, submitted_price, actual_fill, gas, slippage
  4. Update position status → closed
```

## 4. Data Collection (trade_details JSONB)

Every live trade MUST capture these fields for analysis:

### Entry Data
```json
{
  "mode": "live",
  "city": "nyc",
  "direction": "BUY_YES",
  "target_date": "2026-03-27",
  "forecast_prob": 0.12,
  "forecast_temp_c": 15.2,
  "edge_at_scan": 0.065,
  "model": "emos_exit_fusion",

  "entry_gamma_price": 0.035,
  "entry_clob_ask": 0.038,
  "entry_submitted_price": 0.038,
  "entry_actual_fill": 0.037,
  "entry_fill_slippage": -0.001,
  "entry_gas_wei": 150000000000000,
  "entry_gas_usd": 0.004,
  "entry_orderbook_depth_usd": 25.50,
  "entry_order_id": "0xabc...",
  "entry_fill_time_ms": 1200,
  "entry_timestamp": "2026-03-27T08:33:45Z"
}
```

### Exit Data (appended on exit)
```json
{
  "exit_reason": "edge_lost",
  "exit_clob_bid": 0.045,
  "exit_submitted_price": 0.044,
  "exit_actual_fill": 0.044,
  "exit_fill_slippage": -0.001,
  "exit_gas_wei": 145000000000000,
  "exit_gas_usd": 0.004,
  "exit_orderbook_depth_usd": 18.00,
  "exit_order_id": "0xdef...",
  "exit_fill_time_ms": 800,
  "exit_timestamp": "2026-03-27T08:45:32Z",
  "hold_time_seconds": 707,

  "paper_entry_fill": 0.038,
  "paper_exit_fill": 0.045,
  "paper_pnl": 5.56,
  "live_pnl": 4.80,
  "pnl_difference": -0.76,
  "friction_total": 0.76,
  "friction_breakdown": {
    "entry_slippage": 0.15,
    "exit_slippage": 0.30,
    "entry_gas": 0.004,
    "exit_gas": 0.004
  }
}
```

### Metrics to Compute Per Trade
- **Slippage entry**: `actual_fill - clob_ask` (should be ~0)
- **Slippage exit**: `clob_bid - actual_fill` (should be ~0)
- **Total gas**: entry + exit in USD
- **Total friction**: slippage + gas
- **Paper vs live P&L gap**: how much worse is live?
- **Fill rate**: % of orders that fill within 30s
- **Hold time**: seconds between entry and exit

## 5. Risk Management

### Kill Switches
| Trigger | Action |
|---------|--------|
| Weekly P&L < -$25 (25% of $100) | Halt C2 live, alert via Slack |
| Single trade loss > $10 | Alert, continue |
| 3 consecutive losses | Alert, review |
| Fill rate < 50% | Alert — orderbook too thin |
| Avg slippage > 5% | Alert — model assumptions wrong |
| Gas > $0.05 per tx | Alert — Polygon congestion |

### Position Limits
- Max 1 position per market (same as paper)
- Max 3 trades per scan (MAX_TRADES_PER_SCAN)
- Min 6 hours to resolution (MIN_HOURS_TO_RESOLUTION)
- No BUY_NO with effective price > $0.45 (quality gate)

### Dallas Exclusion Consideration
Dallas avg move is $0.001 (1 tick). In live with real spread this is likely unprofitable. Options:
1. Exclude Dallas from live (keep in paper for monitoring)
2. Raise Dallas min_edge to 0.10 (only trade strong signals)
3. Let it run and monitor — $5 max loss per trade

## 6. Dashboard — Live Tab

### New Tab: "Live Trading"
Accessible via tab switch (Paper | Live). Shows ONLY C2 live data.

### Cards Required

**1. Live Overview**
- Capital: $100
- Deployed: $X
- Available: $Y
- Total P&L (live): +$Z
- Trades today: N
- Win rate: X%

**2. Live Trades Table**
| Time | City | Direction | Size | Entry Fill | Exit Fill | Gas | Slippage | Net P&L | Reason | Hold |
|------|------|-----------|------|-----------|----------|-----|----------|---------|--------|------|

**3. Paper vs Live Comparison**
| Metric | Paper | Live | Gap |
|--------|-------|------|-----|
| Avg P&L/trade | $5.18 | ? | ? |
| Win rate | 100% | ? | ? |
| Avg slippage | $0.005 | ? | ? |
| Avg gas | $0.007 | ? | ? |
| Avg hold time | 15 min | ? | ? |
| Fill rate | 100% | ? | ? |

**4. Friction Analysis Chart**
- Stacked bar: slippage + gas per trade
- Line: cumulative friction cost

**5. Per-City Live Performance**
| City | Trades | WR | P&L | Avg Slippage | Avg Gas | Viable? |
|------|--------|-----|-----|-------------|---------|---------|

**6. Orderbook Quality**
- Bid depth at exit moment
- Ask depth at entry moment
- Spread at entry/exit
- Fill time histogram

## 7. Implementation Steps

### Phase 1: Instrumentation (before live)
1. Add `mode` field to trade_details ("paper" or "live")
2. Extend paper engine to support live execution path
3. Add gas tracking (read tx receipt after fill)
4. Add fill price comparison (submitted vs actual)
5. Create DB migration for any new columns if needed

### Phase 2: Live Execution
1. Create `LiveExecutor` class wrapping py-clob-client
2. C2 strategy gets `execution_mode` parameter ("paper" or "live")
3. In live mode: submit real CLOB orders instead of paper_engine.place_trade()
4. Same exit logic, but sell via real CLOB sell order
5. Record all execution data in trade_details

### Phase 3: Dashboard
1. Add "Live" tab to dashboard
2. Create API endpoints: `/api/live/trades`, `/api/live/overview`, `/api/live/friction`
3. Build comparison view (paper vs live side-by-side)
4. Add real-time P&L tracking with friction breakdown

### Phase 4: Monitoring & Alerts
1. Slack alerts for: trade placed, trade exited, loss > threshold, gas spike
2. Automated daily summary: trades, P&L, friction, fill rate
3. Weekly comparison report: paper C2 vs live C2

## 8. Go-Live Checklist

- [ ] py-clob-client configured with L1+L2 auth keys
- [ ] USDC.e funded on Polygon wallet ($100 + $5 gas reserve)
- [ ] CTF Exchange approval (one-time approve tx)
- [ ] NegRisk adapter approval (for weather markets)
- [ ] Live executor tested with $1 test trade
- [ ] Kill switch verified (manual /kill command)
- [ ] Slack alerts configured
- [ ] Dashboard live tab working
- [ ] Gas monitoring in place
- [ ] Paper C2 continues running alongside for comparison

## 9. Success Criteria (First Week)

| Metric | Target | Abort If |
|--------|--------|----------|
| Fill rate | > 80% | < 50% |
| Live vs paper P&L gap | < 30% | > 50% |
| Avg slippage per trade | < $0.50 | > $2.00 |
| Gas per round-trip | < $0.02 | > $0.10 |
| Win rate | > 60% | < 30% |
| Weekly P&L | > $0 | < -$25 |

## 10. Rollback Plan

If any abort criterion is hit:
1. Stop C2 live execution (set `execution_mode = "paper"`)
2. Close all open live positions at market
3. Paper C2 continues unaffected
4. Analyze: what went wrong? Slippage? Gas? Fill rate? Model?
5. Fix and re-deploy when resolved

Live C2 is designed to be **additive** — paper C2 runs in parallel at all times.
If live fails, paper data is unaffected and we lose max $25 (kill switch).
