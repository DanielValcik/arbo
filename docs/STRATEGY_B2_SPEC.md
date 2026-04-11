# Strategy B2 — Crypto Price Edge + Early Exit

> Specifikace nové strategie aplikující C2 model (EMOS + edge-based exit)
> na crypto price prediction markets s masivní likviditou.
> Referenční dokument pro implementaci.
> **Future**: TA Integration plánována — viz `TA_INTEGRATION_SPEC.md` sekce 6 (probability adjustment)

## 1. Proč Crypto Price Markets

### Problém C2 Weather
C2 model je backtestem ověřený (score 138.1, Sharpe 9.44) ale weather markets mají:
- $100-500 likvidita per bucket → partial fills (8-53%)
- Taker spread 60% nad paper price
- ExitManager musí opakovaně zkoušet sell
- Reálný profit je ~50% paper profitu kvůli illikviditě

### Crypto Price Markets — Řešení
Polymarket crypto price prediction markets ("Will Bitcoin hit $X by date?") nabízí:
- **$1-5M likvidita per bucket** (1000-10000x víc než weather)
- **1 tick spread** ($0.01) → minimální slippage
- **Denní/týdenní expirace** → rychlý kapitálový obrat
- **NegRisk multi-outcome** → stejná architektura jako weather
- **Fee: 0% na většině** (ověřit per market)

## 2. Informační Výhoda (Edge Source)

### Weather C2: Forecast vs Market
```
NOAA říká: zítra 15°C v NYC
Polymarket pricuje bucket "14-15°C" na $0.08
Náš model říká: pravděpodobnost 15% → edge = 7%
→ BUY, pak EXIT když forecast se zpřesní
```

### Crypto C3: Exchange Price vs Market
```
Binance real-time: BTC = $87,500
Polymarket pricuje "BTC above $88K by March 31?" na $0.45
Náš model říká: pravděpodobnost 52% (z volatility modelu) → edge = 7%
→ BUY, pak EXIT když edge zmizí (cena se pohne)
```

### Klíčový Rozdíl
- Weather: forecast se aktualizuje 1-4x denně → edge trvá hodiny
- Crypto: exchange price se mění každou sekundu → edge trvá minuty
- **B2 musí být rychlejší** — entry a exit v řádu minut, ne hodin

## 3. Probability Model

### Weather C2: EMOS
- Vstup: forecast temperature
- Sigma: adaptive z rolling forecast errors
- Výstup: P(temp falls in bucket)

### Crypto C3: Volatility-Adjusted CDF
- Vstup: aktuální exchange cena (Binance/Coinbase)
- Sigma: realized volatility (rolling 24h, 7d) + time decay
- Výstup: P(price above/below/in range X by expiry)

```python
# Crypto probability model
def estimate_crypto_prob(current_price, target_price, expiry_hours, sigma_daily):
    """Estimate probability price reaches target by expiry."""
    sigma = sigma_daily * sqrt(expiry_hours / 24)
    z = (log(target_price / current_price)) / sigma
    return 1 - norm_cdf(z)  # P(price >= target)
```

### Time Decay
- 7 dní do expirace: sigma vysoká → nejistota → mírný edge
- 1 den do expirace: sigma nízká → jistota → velký edge (nebo žádný)
- Hodiny do expirace: téměř deterministické (cena je co je)

**Toto je analogie k weather EMOS** — blíž k resolution, forecast je přesnější.

## 4. Exit Logic (Stejná jako C2)

```python
# Edge-based exit (identický princip)
MIN_HOLD_EDGE = 0.03  # Crypto markets efektivnější → nižší threshold
PROFIT_TARGET_ABS = 0.10  # Tighter target (rychlejší obrat)

# Každou minutu:
current_exchange_price = get_binance_price("BTC")
updated_prob = estimate_crypto_prob(current_exchange_price, target, expiry_hours, sigma)
updated_edge = updated_prob - current_polymarket_price

if updated_edge < MIN_HOLD_EDGE:
    SELL  # Edge lost — crypto price moved against us
if current_polymarket_price >= entry_price + PROFIT_TARGET_ABS:
    SELL  # Profit take
```

### ExitManager (z C2 live)
- Fáze 1: taker sell (okamžitý) — **na crypto markets bude fill 100%**
- Fáze 2: maker sell — pravděpodobně nepotřebujeme (likvidita dostatečná)
- **Žádné partial fills** díky hluboké likviditě

## 5. Data Pipeline

### Real-Time Data (Entry + Exit signály)
| Zdroj | Data | Frekvence | Máme? |
|-------|------|-----------|-------|
| Binance WebSocket | BTC, ETH, SOL price | Real-time | Částečně (Strategy B) |
| CoinGecko API | Historické ceny, volatilita | 1min | Ano (Strategy B) |
| Polymarket CLOB | Market prices, orderbook | 10s | Ano (C2 infrastructure) |
| Polymarket Gamma | Market discovery | 5min | Ano |

### Historická Data (Autoresearch)
| Zdroj | Data | Pro | Máme? |
|-------|------|-----|-------|
| PolymarketData.co | Historické crypto price markets + ceny | Backtest | Možná (PMD subscription) |
| Polymarket /prices-history | 30 dní cenových drah | Backtest | Potřebujeme stáhnout |
| Binance API | Historické OHLCV | Volatility model | Snadno dostupné |
| CoinGecko | Historické ceny | Cross-validation | Ano |

### Co Potřebujeme Stáhnout
1. **Polymarket crypto price events** (resolved) — condition_ids, outcomes, prices
2. **Polymarket price history** per token — 10min resolution pro cenové dráhy
3. **Binance historical klines** — odpovídající časové okno (minutová data)
4. Mapování: pro každý Polymarket bucket → jaká byla exchange cena v době tradu

## 6. Autoresearch Design

### Parametry k Optimalizaci
```python
# Probability model
"volatility_window": [6, 12, 24, 48, 72, 168],  # hours
"volatility_method": ["realized", "ewma", "garch"],
"sigma_scale": [0.5, 0.7, 0.8, 1.0, 1.2, 1.5],

# Quality gate
"min_edge": [0.01, 0.02, 0.03, 0.05, 0.08],
"max_price": [0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
"min_price": [0.01, 0.03, 0.05, 0.10],
"min_time_to_expiry_h": [1, 2, 4, 8, 12, 24],

# Exit
"min_hold_edge": [0.01, 0.02, 0.03, 0.05],
"profit_target_abs": [0.05, 0.08, 0.10, 0.15, 0.20],

# Sizing
"kelly_raw_cap": [0.10, 0.15, 0.20, 0.25, 0.30],
"max_position_pct": [0.03, 0.05, 0.08, 0.10],
```

### Fáze Autoresearch (Stejný Pattern jako C2)
1. **Gen 0** (800 trials): Random search přes celý prostor
2. **Gen 1** (400 trials): Fine-tune top-10
3. **Gen 2** (200 trials): Per-asset optimization (BTC vs ETH vs SOL)
4. **Walk-forward** validace na nejlepším modelu

### Scoring (Stejný jako C2)
```
SCORE = ROI(25) + Sharpe(15) + DD(10) + Utilization(15) + PnL/hour(15) + OOS(10) + Trades(10)
```

## 7. Architektura

```
Strategy B2 (crypto_price_strategy.py)
  ├── CryptoPriceScanner
  │   ├── Binance WebSocket → real-time BTC/ETH/SOL price
  │   ├── Polymarket Gamma → discover crypto price events
  │   ├── Volatility model → estimate probability per bucket
  │   └── Edge calculation → prob vs market price
  │
  ├── Quality Gate (crypto_quality_gate.py)
  │   ├── min_edge, price range, time to expiry
  │   └── Per-asset overrides (BTC vs ETH)
  │
  ├── Live Executor (reuse from C2)
  │   ├── Taker pricing (/price?side=SELL for BUY)
  │   ├── Verified fills (size_matched)
  │   └── Cancel remainder
  │
  ├── Exit Monitor (60s interval, reuse from C2)
  │   ├── Re-compute prob with latest exchange price
  │   ├── Edge lost → sell
  │   └── Profit take → sell
  │
  └── ExitManager (reuse from C2)
      ├── Persistent sell-down (taker → maker)
      └── Never hold to resolution
```

### Co Můžeme Reuse z C2
- `live_executor.py` — beze změny (NegRisk pricing funguje pro crypto i weather)
- `exit_manager.py` — beze změny
- `paper_engine.py` — sell_position, tracking
- Dashboard LIVE tab — přidat B2 data
- Slack notifications — přidat B2 channel
- Autoresearch framework — `experiment_framework.py`

### Co Je Nové
- `crypto_price_scanner.py` — nahrazuje weather_scanner
- `crypto_quality_gate.py` — nové thresholds
- `volatility_model.py` — nahrazuje EMOS (sigma z price data místo forecast errors)
- Binance WebSocket connector — real-time ceny
- Data downloader pro historické crypto price markets

## 8. Porovnání C2 vs C3

| Aspekt | C2 Weather | B2 Crypto Price |
|--------|-----------|-----------------|
| Likvidita | $100-500/bucket | $1-5M/bucket |
| Fill rate | 8-53% | ~100% |
| Spread (taker) | 60% nad paper | ~2% nad paper |
| Edge source | Weather forecast | Exchange price |
| Edge duration | Hodiny | Minuty |
| Entry scan | 5 min | 1 min |
| Exit monitor | 60s | 30s |
| Assets | 20 měst | BTC, ETH, SOL |
| Expirace | 24-48h | Denní/týdenní |
| Informační výhoda | NOAA/Met Office/Open-Meteo | Binance/Coinbase real-time |
| Model | EMOS (forecast errors) | Volatility CDF (price moves) |
| Gas | $0 (gasless) | $0 (gasless) |
| Geo-block | Dublin VPS (eu-west-1) | Dublin VPS (eu-west-1) |

## 9. Rizika

### Model Rizika
- Crypto markets jsou **efektivnější** než weather → menší edge, víc kompetice
- Volatilita se mění rychle (VIX-like spikes) → model musí být adaptivní
- Black swan events (exchange hack, regulace) → sudden price moves

### Execution Rizika
- **Minimální** díky likviditě — full fills, tight spreads
- NegRisk pricing: CONFIRMED funguje (z C2 live) → reuse
- GTC orders + cancel remainder → CONFIRMED (z C2 live)

### Mitigace
- Start s malým kapitálem ($100) jako C2
- Backtest na historických datech první
- Walk-forward validace
- Kill switch 25% weekly DD

## 10. Implementační Plán

### Fáze 1: Data (1-2 dny)
1. Stáhnout historické Polymarket crypto price events (resolved)
2. Stáhnout odpovídající Binance klines (minutová data)
3. Mapovat: event → buckets → resolved outcomes
4. Uložit do SQLite (stejný pattern jako weather_pmd.sqlite)

### Fáze 2: Autoresearch (1 den)
1. Adaptovat experiment_framework.py pro crypto price model
2. Volatility model (realized vol, EWMA, time decay)
3. Sweep 1400 trials (3 generace)
4. Walk-forward validace
5. Najít optimální parametry

### Fáze 3: Paper Trading (2-3 dny)
1. Implementovat crypto_price_scanner.py
2. Binance WebSocket pro real-time ceny
3. Paper trading jako Strategy B2
4. Dashboard karty + Expected vs Reality

### Fáze 4: Live Trading (1 den)
1. Reuse live_executor.py (beze změny)
2. Reuse exit_manager.py (beze změny)
3. Test $5 BUY→SELL cyklus
4. Enable live s $100-200

### Celkem: ~5-7 dní od startu k live

## 11. Poznatky z C2 Live (Aplikovatelné na C3)

### NegRisk Pricing (POTVRZENO)
- BUY: submit at `/price?side=SELL` → instant match
- SELL: submit at `/price?side=BUY` → instant match
- Funguje pro VŠECHNY NegRisk markets (weather i crypto)

### Execution Pattern (POTVRZENO)
- GTC order → wait 3s → check size_matched → cancel remainder
- Position sync z Data API (přežívá restarty)
- Lambda closure bug v cancel → vždy použít lokální proměnnou
- L2 creds: vždy derive, nikdy z configu

### Co NEFUNGUJE (Vyhnout se)
- FOK orders na illiquid markets → fail
- prob_floor na cheap tokens → instant exit loop
- Buying + selling same token → check pending_exits
- `get_price` může vrátit empty string → handle gracefully

### Fill Expectations
| Aspekt | Weather (C2) | Crypto (B2 předpoklad) |
|--------|-------------|----------------------|
| Full fill | Vzácný | Běžný (deep book) |
| Partial fill | 8-53% | Neočekáváme |
| Zero fill | Častý | Vzácný |
| ExitManager needed | Kritický | Záložní (pojistka) |

## 12. Soubory k Vytvoření

```
arbo/
├── strategies/
│   ├── crypto_price_scanner.py    # Discover + scan crypto price markets
│   ├── crypto_quality_gate.py     # Quality gate thresholds
│   └── strategy_b2.py             # Main B2 strategy class
├── models/
│   └── volatility_model.py        # Realized vol, EWMA, time decay CDF
├── connectors/
│   └── binance_ws.py              # Binance WebSocket real-time prices
research/
├── data/
│   └── crypto_price_pmd.sqlite    # Historical market data
├── innovations/
│   ├── crypto_price_loader.py     # Load historical data for backtest
│   └── sweep_crypto_price.py      # Autoresearch sweep
└── download_crypto_price_history.py # Data downloader
```

## 13. Detailní Analýza Crypto Price Markets (Live Research 2026-03-26)

Kompletní analýza: `research/crypto_price_markets_analysis.md`

### Typy Trhů

| Typ | Formát | Resolution | Fee | Frekvence |
|-----|--------|-----------|-----|-----------|
| **Daily "Above"** | "BTC above $68K on March 27, 12PM ET?" | Binance 1m candle Close | Crypto (maker 0%) | Denní, 11 bucketů |
| **Monthly "Hit"** | "What price will BTC hit in March?" | ANY Binance 1m candle touch | **0% fee** | Měsíční, 14-20 bucketů |
| **5min Up/Down** | "BTC Up or Down 10:00-10:05 ET?" | Binance 1m candle | Crypto fees | Každých 5 min |
| **15min Up/Down** | "ETH Up or Down 10:00-10:15 ET?" | Binance 1m candle | Crypto fees | Každých 15 min |

### KRITICKÉ: NEJSOU NegRisk!
Crypto price markets jsou **nezávislé binární** markety, NE multi-outcome NegRisk.
- Každý bucket ("BTC above $68K?") je samostatný YES/NO market
- Žádný NegRisk adapter, standardní CTF Exchange
- **Pricing je přímočarý** — bid/ask přímo z orderbooku (ne /price endpoint)
- Odpadá NegRisk confusion z C2 live (BUY@SELL price atd.)

### Likvidita per Asset (Live Data)

| Asset | Daily Volume | Liquidity | Spread (ATM) | Monthly Vol | Viable? |
|-------|-------------|-----------|--------------|-------------|---------|
| **BTC** | **$1.19M** | **$469K** | **1-2c** | **$81.5M** | **Hlavní target** |
| **ETH** | **$518K** | ~$200K | 2-3c | $16.2M | Sekundární |
| SOL | $36K | ~$10K | 50c+ | — | Excluded |
| XRP | $33K | ~$8K | 50c+ | — | Excluded |

### Fee Model
```
fee = price × (1 - price) × 0.25
```
- Max: 6.25c při price=0.50
- **Maker (PostOnly): 0% fee + 20% rebate z taker fees!**
- Taker: platí fee
- **Strategie: PostOnly orders pro entry, taker jen pro urgentní exit**

### BTC Daily Bucket Struktura (March 27)

| Threshold | Prob | Spread | Volume | Sweet Spot? |
|-----------|------|--------|--------|-------------|
| $60,000 | ~100% | ~0 | $376K | Ne (no edge) |
| $68,000 | 75% | ~1c | $54K | **Ano** |
| $70,000 | 31% | ~1c | $46K | **Ano** |
| $72,000 | 5% | ~1c | $53K | **Ano** |
| $80,000 | <1% | ~0 | $69K | Ne (longshot) |

**Sweet spot: buckety s prob 20-80%** — tight spread + dostatečná likvidita.

### Monthly "Hit" — Zero Fee Opportunity
- "What price will BTC hit in March?" — 14-20 thresholds
- **$0 fee** na vše (entry i exit)
- Volume: $81.5M (masivní)
- Model: barrier option pricing — P(max(BTC_t) >= K)
- **Potenciálně nejprofitabilnější** díky zero fee a obrovské likviditě

## 14. Otevřené Otázky (Answered + New)

1. ~~Které kryptoměny?~~ → **BTC primary, ETH secondary, SOL/XRP excluded**
2. ~~Denní vs týdenní?~~ → **Daily "Above" (hlavní) + Monthly "Hit" (zero fee bonus)**
3. **PostOnly vs taker**: PostOnly = 0% fee + rebate. Entry PostOnly (čekat na fill), exit taker (urgentní)?
4. **Non-NegRisk execution**: Standardní orderbook — BUY at ask, SELL at bid. Jednodušší než C2.
5. **Barrier option model** pro monthly "hit" — jiný přístup než simple CDF
6. **5min/15min markets**: Potenciální HFT příležitost — vyžaduje sub-second execution

---

*Referenční specifikace pro implementaci Strategy B2.
Poznatky z C2 live (fill handling, exit logic) přímo aplikovatelné.
Crypto markets NEJSOU NegRisk — jednodušší execution model.
V další session: `docs/STRATEGY_B2_SPEC.md` + `research/crypto_price_markets_analysis.md`*
