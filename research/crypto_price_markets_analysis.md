# Polymarket Crypto Price Markets — Research Analysis

> Date: 2026-03-26
> Purpose: Inform Strategy D crypto price sub-strategy design
> Data source: Gamma API + Polymarket website, live data

---

## 1. Market Types Overview

Polymarket offers **4 distinct crypto price market types**:

| Type | Format | Resolution | Frequency | Fee | Example |
|------|--------|------------|-----------|-----|---------|
| **Daily "Above"** | "Bitcoin above $X on [date], 12PM ET?" | Binance 1m candle close | New event every day, 11 buckets | Crypto fees (see below) | `bitcoin-above-on-march-27` |
| **Monthly "Hit"** | "What price will Bitcoin hit in [month]?" | Any Binance 1m candle High/Low during month | Monthly, 14-20 buckets | **No fees** | `what-price-will-bitcoin-hit-in-march` |
| **5-minute Up/Down** | "Bitcoin Up or Down — [date] [time]-[time+5min] ET" | Binance 1m candle | Every 5 minutes during trading hours | Crypto fees | (listed on site, couldn't fetch via API) |
| **15-minute Up/Down** | "Ethereum Up or Down — [date] [time]-[time+15min] ET" | Binance 1m candle | Every 15 minutes | Crypto fees | (listed on site, couldn't fetch via API) |

**Assets covered**: Bitcoin (BTC), Ethereum (ETH), Solana (SOL), XRP, Dogecoin (DOGE), BNB

---

## 2. Daily "Above" Markets — Detailed Structure

### Bitcoin Daily (March 27, 2026 — live data)

**Event**: "Bitcoin above ___ on March 27, 12PM ET?"
- **Resolution**: Binance BTC/USDT 1-minute candle **Close** at 12:00 noon ET
- **NegRisk**: false (NOT multi-outcome)
- **Tick size**: 0.01 for near-the-money, 0.001 for far OTM
- **Fees**: crypto_fees enabled (see Section 5)
- **Total volume**: $1,186,463
- **Total liquidity**: $469,264
- **Buckets**: 11 price thresholds

| Threshold | Probability | Bid | Ask | Spread | Volume |
|-----------|-------------|-----|-----|--------|--------|
| $60,000 | 100% | 99.9c | 0.2c | ~0 | $376K |
| $62,000 | 100% | 99.9c | 0.2c | ~0 | $66K |
| $64,000 | 99% | 99.2c | 0.9c | ~0.8c | $73K |
| $66,000 | 96% | 95.8c | 4.4c | ~0.8c | $85K |
| **$68,000** | **75%** | **75c** | **26c** | **~1c** | **$54K** |
| **$70,000** | **31%** | **31c** | **70c** | **~1c** | **$46K** |
| $72,000 | 5% | 5c | 96c | ~1c | $53K |
| $74,000 | 1% | 0.7c | 99.4c | ~0.7c | $67K |
| $76,000 | <1% | 0.2c | 99.9c | ~0.3c | $63K |
| $78,000 | <1% | 0.1c | - | ~0.1c | $234K |
| $80,000 | <1% | 0.1c | - | ~0.1c | $69K |

**Key observation**: Spreads are tightest (1c) near the money (where probability is 20-80%). Far ITM/OTM markets are illiquid. Volume concentrates at extreme thresholds ($60K and $78K) due to degenerate tail bets.

### Ethereum Daily (March 27, 2026)

**Event**: "Ethereum above ___ on March 27, 12PM ET?"
- **Resolution**: Binance ETH/USDT 1-minute candle Close at 12:00 noon ET
- **NegRisk**: false
- **Total volume**: $517,526
- **Buckets**: 11 thresholds

| Threshold | Probability | Bid | Ask | Spread | Volume |
|-----------|-------------|-----|-----|--------|--------|
| $1,600 | 100% | 99.8c | 0.3c | - | $60K |
| $1,700 | 100% | 99.8c | 0.3c | - | $23K |
| $1,800 | 100% | 99.8c | 0.5c | - | $11K |
| $1,900 | 98% | 98.3c | 1.9c | ~1.7c | $24K |
| **$2,000** | **84%** | **84.9c** | **17.8c** | **~2.7c** | **$130K** |
| **$2,100** | **31%** | **32c** | **70c** | **~2c** | **$21K** |
| $2,200 | 3% | 2.8c | 97.3c | ~2.7c | $24K |
| $2,300 | 1% | 0.6c | 99.5c | ~1c | $143K |
| $2,400 | <1% | 0.2c | 99.9c | ~0.3c | $21K |
| $2,500 | <1% | 0.2c | - | - | $31K |
| $2,600 | <1% | 0.2c | - | - | $29K |

**Tick size**: 0.001 (ETH $2,000 market confirmed)

### Solana Daily (March 27, 2026)

**Event**: "Solana above ___ on March 27, 12PM ET?"
- **Resolution**: Binance SOL/USDT 1-minute candle Close at 12:00 noon ET
- **NegRisk**: false
- **Total volume**: $32,533 (much lower than BTC/ETH)
- **Total liquidity**: $220,637
- **Buckets**: 11 thresholds ($40-$140 in $10 increments)

| Threshold | Probability | Volume |
|-----------|-------------|--------|
| $80 | 98% | $13K |
| **$90** | **23%** | **$939** |
| $100 | 1% | $10K |

**Liquidity is very thin** — only $939 volume on the near-the-money bucket.

### XRP Daily (March 27, 2026)

**Event**: "XRP above ___ on March 27, 12PM ET?"
- **Resolution**: Binance XRP/USDT 1-minute candle Close at 12:00 noon ET
- **NegRisk**: false
- **Total volume**: $35,852
- **Total liquidity**: $183,154
- **Buckets**: 11 thresholds ($0.90-$1.90 in $0.10 increments)

| Threshold | Probability | Volume |
|-----------|-------------|--------|
| $1.30 | 90% | $5.6K |
| **$1.40** | **23%** | **$285** |
| $1.50 | 1% | $1K |

---

## 3. Monthly "Hit" Markets — Detailed Structure

### Bitcoin Monthly (March 2026 — live)

**Event**: "What price will Bitcoin hit in March?"
- **Resolution**: Any Binance BTC/USDT 1-minute candle during the month; High for upper thresholds, Low for lower thresholds
- **NegRisk**: false
- **Fees**: **DISABLED** (feesEnabled: false)
- **Tick size**: 0.001
- **Total volume**: $81.5M (massive!)
- **Total liquidity**: ~$5.75M
- **Buckets**: 20 outcomes (both up-hits and down-hits)

| Direction | Threshold | Probability | Volume |
|-----------|-----------|-------------|--------|
| Up | $150,000 | <1% | $24.0M |
| Up | $110,000 | <1% | $3.2M |
| Up | $105,000 | <1% | $1.7M |
| Up | $100,000 | <1% | $3.0M |
| Up | $95,000 | <1% | $2.9M |
| Up | $90,000 | <1% | $4.4M |
| Up | $85,000 | 1% | $3.5M |
| Up | $80,000 | 2% | $5.9M |
| **Down** | **$65,000** | **30%** | **$8.0M** |
| Down | $60,000 | 6% | $4.4M |
| Down | $55,000 | 2% | $5.4M |
| Down | $50,000 | 1% | $3.4M |
| Down | $45,000 | <1% | $2.2M |
| Down | $40,000 | <1% | $1.7M |
| Down | $35,000 | <1% | $1.4M |
| Down | $30,000 | <1% | $542K |
| Down | $25,000 | <1% | $1.0M |
| Down | $20,000 | <1% | $3.1M |

**Critical difference from daily**: Monthly markets ask "will price TOUCH this level at any point during the month" — not close price at a specific time. Resolution uses High/Low (not Close) from any 1m candle during the entire month.

### Historical Resolution Example (April 2025)

**Event**: "What price will Bitcoin hit in April?" (RESOLVED)
- Volume: $26.7M
- Bitcoin touched $95K and $90K during April 2025 but NOT $100K
- Resolved YES: $95K, $90K, $85K, $80K, $75K
- Resolved NO: $200K, $150K, $120K, $110K, $100K, $70K, $60K, $50K, $40K

---

## 4. Resolution Mechanics

### Daily "Above" Markets
- **Resolution source**: Binance BTC/USDT (or ETH/USDT, SOL/USDT) 1-minute candle
- **Resolution time**: 12:00 PM ET (noon) — uses the **Close** price of the 1m candle
- **URL**: https://www.binance.com/en/trade/BTC_USDT with "1m" candles selected
- **Oracle**: UMA optimistic oracle with 500 USDC bond, $2-5 reward
- **Status flow**: Active → "In Review" (proposed) → Resolved
- **Dispute window**: After proposed resolution, there's a liveness period for UMA disputes
- **New event cadence**: New daily event created ~7 days before resolution

### Resolution Pattern Example (March 26, 2026 — just resolved)
- BTC price at 12:00 PM ET: between $68,000 and $70,000
- $60K through $68K: resolved YES
- $70K through $80K: resolved NO
- Total volume for this one day: $5.4M

### Monthly "Hit" Markets
- **Resolution**: "Yes" if ANY 1-minute candle High (for up) or Low (for down) reaches the threshold during the entire month
- **Multiple outcomes can resolve YES** (unlike daily where it's sequential)
- **Duration**: Full calendar month

---

## 5. Fee Structure — CRITICAL FOR STRATEGY

### Crypto Fees (Daily "Above" Markets)
```
Fee model: fee = price * (1 - price) * fee_rate
```
Where:
- `exponent`: 2
- `rate`: 0.25
- `taker_only`: true (makers pay NO fees)
- `rebate_rate`: 0.2 (20% of taker fees redistributed to makers)
- `maker_base_fee`: 1000 bps (informational, not charged due to taker_only)
- `taker_base_fee`: 1000 bps

**Effective fee calculation**:
```
taker_fee = p * (1 - p) * 0.25
```

| Price (p) | Fee per share |
|-----------|---------------|
| 0.05 | 0.0119 (1.19c) |
| 0.10 | 0.0225 (2.25c) |
| 0.20 | 0.0400 (4.00c) |
| 0.30 | 0.0525 (5.25c) |
| 0.50 | 0.0625 (6.25c) — maximum |
| 0.70 | 0.0525 (5.25c) |
| 0.90 | 0.0225 (2.25c) |
| 0.95 | 0.0119 (1.19c) |

**Maker rebate** = 20% of taker fee. With PostOnly orders, you PAY ZERO and RECEIVE rebate.

### Monthly "Hit" Markets
- **Fees DISABLED** (feesEnabled: false)
- No maker rebates either (no fee pool to redistribute)

### Implication for Strategy
- Daily markets: Must use **PostOnly maker orders** to avoid 1-6c fee per share
- Monthly markets: Can use any order type, no fee impact
- Maker rebate on daily markets provides additional edge for limit orders

---

## 6. NegRisk Analysis

**All crypto price markets are NOT NegRisk** (enableNegRisk: false).

This means:
- Each price threshold is an **independent binary market** (Yes/No)
- Probabilities across thresholds do NOT need to sum to 100%
- Cannot use neg_risk arbitrage between buckets
- Each market has its own separate orderbook and liquidity pool
- **No cross-bucket collateral netting** — must fund each position independently

This is a critical architectural difference from weather markets (which use NegRisk for temperature buckets).

---

## 7. Liquidity Analysis

### Volume Comparison (daily markets, single day)

| Asset | Daily Volume | Liquidity | Near-Money Spread | Buckets |
|-------|-------------|-----------|-------------------|---------|
| **Bitcoin** | $1.2M | $469K | 1-2c | 11 |
| **Ethereum** | $518K | ~$200K | 2-3c | 11 |
| **XRP** | $36K | $183K | ~5c | 11 |
| **Solana** | $33K | $221K | ~5c | 11 |

### Volume Comparison (monthly markets)

| Asset | Monthly Volume | Liquidity |
|-------|---------------|-----------|
| **Bitcoin** | $81.5M | $5.75M |

### Liquidity Concentration
- 80%+ of volume is in **Bitcoin daily** and **Bitcoin monthly**
- ETH has moderate liquidity
- SOL and XRP are **too thin for systematic trading** (<$1K volume per near-money bucket)
- Volume heavily concentrates at extreme strikes (lottery ticket behavior at <1% and >99%)

---

## 8. Spread Analysis — Near-the-Money Buckets

| Asset | Bucket | Prob | Bid | Ask | Spread | Spread % |
|-------|--------|------|-----|-----|--------|----------|
| BTC | $68K | 75% | 75c | 26c (No) | ~1c | 1.3% |
| BTC | $70K | 31% | 31c | 70c (No) | ~1c | 2.0% |
| ETH | $2,000 | 84% | 82.2c | 84.9c | 2.7c | 3.3% |
| ETH | $2,100 | 31% | 32c | 70c (No) | ~2c | 4.0% |
| SOL | $90 | 23% | 25c | 80c | 55c | HUGE |
| XRP | $1.40 | 23% | 24c | 78c | 54c | HUGE |

**Key finding**: BTC has tight 1-2c spreads near the money. ETH has 2-3c spreads. SOL and XRP have massive 50c+ spreads (essentially no real market making happening).

---

## 9. Market Structure Patterns

### Daily Lifecycle
1. **T-7 days**: New daily event created with 11 price buckets
2. **T-7 to T-0**: Trading active, liquidity builds
3. **T-0 12:00 ET**: Resolution snapshot (Binance 1m candle Close)
4. **T-0 to T+1**: "In Review" status (UMA oracle proposed resolution)
5. **T+1 to T+2**: Resolution finalized after dispute window

### Bucket Spacing
- **BTC**: $2,000 increments ($60K-$80K range)
- **ETH**: $100 increments ($1,600-$2,600 range)
- **SOL**: $10 increments ($40-$140 range)
- **XRP**: $0.10 increments ($0.90-$1.90 range)

### Implied Probability Distribution
The bucket probabilities form a **cumulative distribution function** — "above $X" is monotonically decreasing in X. The implied PDF (derivative) gives the market's view of the price distribution.

Example from March 27 BTC data:
- P(BTC > 66K) = 96%, P(BTC > 68K) = 75% → ~21% probability BTC is between $66K-$68K
- P(BTC > 68K) = 75%, P(BTC > 70K) = 31% → ~44% probability BTC is between $68K-$70K
- P(BTC > 70K) = 31%, P(BTC > 72K) = 5% → ~26% probability BTC is between $70K-$72K

The market implies a median BTC price of ~$69,000 for March 27 noon ET.

---

## 10. Strategy Opportunities

### Opportunity A: Daily "Above" — Probability Model Edge
- **Approach**: Build a BTC/ETH price distribution model (GARCH, options-implied, realized vol)
- **Edge source**: If our model says P(BTC > 70K) = 40% but market says 31%, buy YES at 31c
- **Execution**: PostOnly maker orders (zero fees + rebate)
- **Size**: BTC markets have enough liquidity ($50K+ per bucket)
- **Risk**: Single-day binary — either win or lose the full amount
- **Resolution**: Deterministic (Binance candle), no ambiguity

### Opportunity B: Monthly "Hit" — Touch Probability
- **Approach**: Model P(BTC touches $X at any point during month) using barrier option pricing
- **Edge source**: Monthly markets are less efficient and have ZERO fees
- **Volume**: $81.5M — extremely liquid
- **Risk**: 30-day duration, price can move significantly
- **Key insight**: Touch probability is always higher than close-above probability (path-dependent)

### Opportunity C: Cross-Market Arbitrage
- **Daily vs. Monthly**: If daily market implies BTC median of $69K, but monthly says only 2% chance of hitting $80K, there may be mispricing
- **Cross-asset**: BTC and ETH correlate ~0.85; if BTC daily markets imply high vol but ETH doesn't, trade the discrepancy
- **Limitation**: Markets are NOT NegRisk, so no direct neg-risk arb between buckets

### Opportunity D: Spread Capture (Market Making)
- **BTC daily**: 1-2c spread near the money, PostOnly = zero fees + rebate
- **ETH daily**: 2-3c spread near the money
- **Risk**: Inventory risk is extreme on daily binaries (full loss if wrong side)
- **NOT recommended for SOL/XRP**: spreads too wide, liquidity too thin

---

## 11. Key Findings Summary

1. **BTC daily "above" markets are the primary opportunity** — $1.2M daily volume, 1-2c spreads, deterministic resolution via Binance
2. **Monthly "hit" markets are fee-free** with $81.5M volume — massive but different (touch probability, not close probability)
3. **All crypto markets are NOT NegRisk** — each bucket is independent, no cross-bucket arbitrage
4. **Fees**: Daily markets have crypto_fees = p*(1-p)*0.25, max 6.25c at p=0.50. Makers pay ZERO + get 20% rebate. **Always use PostOnly.**
5. **Resolution**: Binance 1m candle, deterministic, no ambiguity (unlike weather which can have sensor disputes)
6. **ETH is secondary** — decent liquidity but wider spreads
7. **SOL/XRP are NOT viable** — sub-$1K volume per near-money bucket, 50c+ spreads
8. **5-minute and 15-minute markets** exist per Polymarket categories but couldn't be fetched via Gamma API; likely very high frequency, very short-lived events
9. **Bucket spacing**: BTC $2K increments, ETH $100, forming an implied CDF
10. **New events daily**: Created ~7 days ahead, resolve at noon ET

---

## 12. Comparison with Weather Markets (Strategy C)

| Dimension | Weather (Strategy C) | Crypto Daily | Crypto Monthly |
|-----------|---------------------|-------------|----------------|
| NegRisk | YES | NO | NO |
| Fee | 0% | p*(1-p)*0.25 | 0% |
| Resolution | METAR station | Binance 1m candle | Binance 1m candle |
| Frequency | Daily | Daily | Monthly |
| Buckets | 10-15 temp buckets | 11 price thresholds | 14-20 touch levels |
| Liquidity | $500-$2K/bucket | $50K-$500K/bucket | $500K-$24M/bucket |
| Edge source | Forecast accuracy | Price distribution model | Barrier option pricing |
| Model type | Normal CDF + EMOS | GARCH / options-implied vol | Touch probability model |
| Deterministic? | Mostly (METAR) | Yes (Binance candle) | Yes (Binance candle) |

**Crypto markets have 10-100x more liquidity than weather**, but also more competition from sophisticated quant firms. The fee structure on daily markets is punitive for takers but free for makers.

---

## 13. Data Sources for Strategy Development

- **Real-time BTC price**: Binance BTC/USDT WebSocket or REST API
- **Historical volatility**: Binance kline API (1m, 5m, 1h candles)
- **Options-implied vol**: Deribit BTC options (for calibrating distribution model)
- **Polymarket prices**: CLOB API `get_price` / `get_orderbook` per bucket
- **Polymarket market discovery**: Gamma API `events?title=Bitcoin%20above` or direct slug
- **Resolution verification**: Binance 1m candle at noon ET

---

## 14. Open Questions for CEO

1. **Which market type to prioritize?** Daily "above" (higher frequency, fees) vs. Monthly "hit" (no fees, longer duration)
2. **Capital allocation**: How much of the $2,000 total capital to allocate to crypto strategy?
3. **BTC-only or BTC+ETH?** ETH has less liquidity but may have more mispricing
4. **5-min/15-min markets**: Worth investigating further? High frequency but very short duration
5. **Competition concern**: Crypto markets likely have more sophisticated participants than weather — is our edge realistic?

---

## Appendix: API Query Reference

```
# Daily "above" events (open)
GET https://polymarket.com/event/bitcoin-above-on-march-27

# Monthly "hit" events
GET https://polymarket.com/event/what-price-will-bitcoin-hit-in-march-2026

# Gamma API for crypto-tagged markets
GET https://gamma-api.polymarket.com/markets?closed=false&limit=50&tag=crypto-prices

# Specific market with fee details
GET https://polymarket.com/event/bitcoin-above-on-march-27?tid=bitcoin-above-68000-on-march-27-2026-12pm-et
```
