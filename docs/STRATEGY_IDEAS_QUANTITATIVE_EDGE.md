# Strategy Ideas — Quantitative Edge on Polymarket

> Captured: 2026-02-27. Return to this after 2-4 weeks of Strategy C weather data.

## Why Weather Works

Strategy C exploits a specific pattern:
- **Quantifiable outcome** (temperature = number, not opinion)
- **Quality public data** (NOAA, Met Office, Open-Meteo) that most bettors don't use
- **High frequency** (daily resolution, hundreds of markets across cities)
- **Unsophisticated competition** (retail bettors guess, we model)

What other Polymarket categories share this pattern?

---

## 1. Economic Data Releases (Highest Potential)

**Markets:** "Will CPI be above 3.2%?", "Will unemployment drop below 4%?", "Will GDP growth exceed 2%?"

**Edge source:**
- Cleveland Fed Inflation Nowcast — free, updates daily, very accurate
- Bloomberg/Fed Survey of Professional Forecasters consensus
- FRED API — free, comprehensive economic data
- Nowcasting models (NY Fed GDP Nowcast) publicly available

**Why it could work:**
- Retail Polymarket bettors guess based on vibes/news headlines
- Professional forecasters have well-calibrated probability distributions
- CPI/jobs numbers resolve to exact values — same CDF approach as weather
- Lower competition than crypto/sports categories

**Concerns:**
- Lower frequency (monthly CPI, quarterly GDP)
- Fewer markets per event
- Potential for sudden regime changes (policy shifts)

**Data sources (free):**
- FRED API: `https://api.stlouisfed.org/fred/` (free API key)
- Cleveland Fed Inflation Nowcast: public CSV
- Philadelphia Fed Survey of Professional Forecasters
- BLS release calendar

**Similarity to weather: HIGH** — same model structure (forecast distribution vs market price via CDF)

---

## 2. Crypto Price Bands

**Markets:** "Will BTC be above $95K on Friday?", daily/weekly price targets

**Edge source:**
- Options-implied probability from Deribit/CME (Black-Scholes → CDF)
- Historical volatility models (GARCH, realized vol)
- Funding rates as directional signal

**Why it could work:**
- Hundreds of markets daily on Polymarket
- Options market is efficient → gives calibrated probabilities
- Polymarket crypto bettors may not use options data

**Concerns:**
- Crypto traders ARE sophisticated — edge may be thin
- 15min/5min crypto markets have fees (not 0%)
- High correlation between positions (BTC moves = all positions move)
- Black swan risk

**Data sources:**
- Deribit API (free): options chain → implied vol → probability
- CoinGecko (already have): price + volume
- Binance/Bybit: funding rates, open interest

**Similarity to weather: MEDIUM** — quantitative model exists but competition is stronger

---

## 3. Sports Player Props

**Markets:** "Will Haaland score 2+ goals?", "Will LeBron get a triple-double?"

**Edge source:**
- Expected goals (xG) models for soccer
- Poisson distribution for scoring events
- Historical player stats + matchup data
- Pinnacle lines (already have via The Odds API)

**Why it could work:**
- Rich statistical models exist (FBref, Understat for xG)
- Casual bettors overbet favorites and stars
- High frequency (daily games across leagues)

**Concerns:**
- Polymarket sports liquidity varies
- Need sport-specific models (not generic)
- Injury news can invalidate models instantly

**Data sources:**
- The Odds API (already have, player props available on higher tier)
- FBref / Understat (free xG data)
- NBA API (free, detailed player stats)

**Similarity to weather: MEDIUM** — quantifiable but more noise, faster-moving information

---

## Priority Ranking

| Strategy | Edge Potential | Data Cost | Implementation Effort | Frequency |
|----------|---------------|-----------|----------------------|-----------|
| Economic Data | HIGH | $0 | Medium (new connectors) | Monthly |
| Crypto Bands | MEDIUM | $0 | Low (have CoinGecko) | Daily |
| Sports Props | MEDIUM | $10-30/mo | High (sport models) | Daily |

**Recommendation:** Start with Economic Data after weather validation. Same architecture (forecast CDF vs market price), free data, less competition. Crypto bands as quick add-on if we confirm the options-implied approach works.

---

## Next Steps (After Weather Validation)

1. Analyze 2-4 weeks of Strategy C results — is the CDF approach profitable?
2. If yes: prototype Economic Data scanner with Cleveland Fed Nowcast
3. If marginal: try crypto bands (lower effort, can reuse infrastructure)
4. Build generic "quantitative CDF vs market" framework that works across categories
