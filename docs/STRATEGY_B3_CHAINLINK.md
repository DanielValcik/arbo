# Strategy B3 — Chainlink Oracle Calibration

> Updated: 2026-04-01 (Chainlink pivot)

## Problem Statement

B3 BTC 5-min Up/Down scalper was profitable in backtest but flat in live.

**Root cause**: Backtest used Binance 1-min klines for resolution (`closes[i+5] >= closes[i]`), but Polymarket resolves using **Chainlink Data Streams** oracle. Binance and Chainlink disagree on **24.8% of 5-min BTC moves** — enough to destroy any edge.

## Resolution Architecture

```
                    ENTRY                           RESOLUTION
                    ─────                           ──────────
Signal source:      Binance WebSocket               N/A
Reference price:    Binance 1-min kline (S_start)   N/A
FV computation:     P(Up) = Φ(log(S/S₀) / σ·κ·√t)  N/A
Win/Loss oracle:    N/A                             Chainlink RTDS via Gamma API
```

Polymarket resolution source: `https://data.chain.link/streams/btc-usd`

### Resolution Flow

1. `check_exits()` detects `now >= event_end_ts`
2. For live positions: `b3_scanner.fetch_resolution(event_start_ts)`
   - Constructs slug: `btc-updown-5m-{start_ts}`
   - Queries: `GET https://gamma-api.polymarket.com/events?slug={slug}`
   - Parses `outcomePrices` → `["1","0"]` = UP won
3. Waits up to 2 min for oracle confirmation
4. Fallback to Binance price (paper-only or timeout)

### Key Implementation

- **Scanner**: `arbo/strategies/b3_scanner.py` → `fetch_resolution()`
- **Strategy**: `arbo/strategies/strategy_b3.py` → `check_exits()` lines 618-644
- **Parameters**: `arbo/strategies/b3_quality_gate.py`
- **Never-sell**: All live positions held to Chainlink resolution (no early exit)

## Autoresearch (2026-04-01)

### Data
- 89,419 Binance 1-min klines (89 days: 2025-12-28 → 2026-03-27)
- 20,300 Chainlink resolutions from Gamma API (79% coverage of 5-min windows)
- Stored in `research/data/crypto_price_pmd.sqlite` table `chainlink_resolutions`

### Sweep
- 2,400 trials: Gen 0 (1,200 random) → Gen 1 (800 mutations) → Gen 2 (400 fine-tune)
- Walk-forward 70/30 split, fixed $300 capital, never-sell mode
- Script: `research/innovations/sweep_b3_chainlink.py`
- Results: `research/data/experiments/b3_chainlink_sweep.json`

### Deployed Parameters (WF #5, best OOS)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SIGMA_WINDOW` | 1440 | 24h realized volatility window |
| `SIGMA_SCALE` | 0.408 | Signal amplification factor |
| `ENTRY_THRESHOLD` | 0.121 | Min \|signal - 0.50\| for entry |
| `POSITION_PCT` | 0.029 | 2.9% of capital per trade |
| `EDGE_SCALING` | 9.497 | Size × edge multiplier |
| `SPREAD` | 0.005 | Assumed bid-ask spread |
| `MIN_ENTRY_MIN` | 2 | Earliest entry: minute 2 |
| `MAX_ENTRY_MIN` | 3 | Latest entry: minute 3 |
| `REENTRY_COOLDOWN` | 0 | No cooldown between trades |

### Performance

| Metric | IS (62 days) | OOS (27 days) |
|--------|-------------|---------------|
| Trades | 9,943 | 4,423 |
| Win Rate | 74.4% | 70.7% |
| PnL ($300 fixed) | $10,970 | $2,690 |
| Daily PnL | $174 | $96 |
| Sharpe | 26.3 | 18.0 |
| Max DD | $88 (29%) | $99 (33%) |

### Realistic Live Estimates ($300 capital)

| | 100% fill | 60% fill (realistic) |
|---|---|---|
| Trades/day | 158 | ~95 |
| Daily PnL | $96 | $58 |
| Monthly PnL | $2,882 | $1,729 |
| Max DD | $99 | ~$60 |

## Trade Data Collection

Every B3 trade stores comprehensive data in `trade_details` JSONB for future optimization:

### Entry
- `btc_at_start` — Binance kline at event start
- `btc_now` — Binance WS at entry
- `btc_chainlink` — Chainlink RTDS at entry
- `btc_binance_chainlink_delta` — oracle price gap
- `sigma_per_min`, `signal_fv_up`, `market_fv_up`, `edge`
- `event_start_ts`, `event_end_ts`, `entry_minutes_elapsed`
- `orderbook_spread`, `orderbook_best_bid`, `orderbook_best_ask`
- `liq_available_usd`, `liq_slippage`
- `live_entry_price`, `live_entry_shares`, `live_fill_status`, `live_entry_latency_ms`
- `live_size_usd`, `live_capital`

### Resolution
- `live_exit_price` — $1 (won) or $0 (lost), from Chainlink oracle
- `live_exit_shares`, `live_exit_status`
- `btc_binance_at_resolution` — Binance BTC at resolution
- `btc_chainlink_at_resolution` — Chainlink BTC at resolution

## 15-Minute Markets

15-min BTC Up/Down markets (`btc-updown-15m-{ts}`) have:
- 7x more volume ($133K vs $18K avg)
- 1.5x more orderbook depth
- 1.5x wider spread (0.010 vs 0.007)

Autoresearch in progress: `research/innovations/sweep_b3_15min.py`
Resolution data: `chainlink_resolutions_15m` table in SQLite

## Compounding

| Capital | Trade Size | Fill Rate | Est. $/day |
|---------|-----------|-----------|------------|
| $300 | $9 | ~100% | $96 |
| $500 | $14 | ~100% | $160 |
| $800 | $23 | ~80% | $205 |
| $1,000+ | $29+ | ~60% | $256 |

$100/trade hardcap in code. Sweet spot: $500-800.

## Bugs Fixed (2026-04-01)

1. `_live_holding` never resolved — `check_exits()` early return skipped held positions
2. Fake resolution data — wrote `live_exit_price=0` for holding positions
3. Binance ≠ Chainlink — 24.8% wrong resolutions, 14 trades repaired
4. Slack notifications — crashed on `pos=None` for live-only resolutions
5. Dashboard table — used paper exit price fallback for resolution losses
