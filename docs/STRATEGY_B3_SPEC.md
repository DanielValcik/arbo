# Strategy B3 — Binance Oracle Scalper

> **Status**: BACKTEST COMPLETE, OOS VALIDATED
> **Date**: 2026-03-27
> **Capital**: $1,000
> **Market type**: BTC 5-min Up/Down (Polymarket)
>
> **Related Documents:**
> - `B3_WATCHDOG_SPEC.md` — Plně autonomní monitoring & self-optimization
> - `TA_INTEGRATION_SPEC.md` — TradingView TA features (RSI, ADX, MACD, BB)
> - `STRATEGY_B3_CHAINLINK.md` — Chainlink oracle resolution
> - `STRATEGY_B3_LIVE_SPEC.md` — Live deployment notes
> - `TRADINGVIEW_MCP_ANALYSIS.md` — TA výzkum

---

## 1. Popis (CZ)

B3 je high-frequency scalping strategie na BTC 5-minutove "Up or Down" markety
na Polymarket. Vyuziva informacni asymetrii: Binance BTC cena se meni DRIV nez
Polymarket market cena. Model pocita fair value z Binance ceny a vstupuje kdy
market jeste nestacil zareagovat.

Klicovy princip: **nikdy nevsazime na vysledek** — vsazime na POHYB ceny behem
5-minutoveho okna. Kdyz BTC jde nahoru, kupujeme "Up" token, prodavame ho o
1-3 minuty pozdeji s profitem. PostOnly ordery = 0% fee + rebate.

## 2. Jak to funguje

### Market struktura

Polymarket kazdy den vytvari **~54 BTC 5-minutovych Up/Down marketu** ve 3
sessions (kazda ~4 hodiny). Kazdy market se pta: "Bude BTC cena na konci
5-minutoveho okna >= cena na zacatku?"

| Parametr | Hodnota |
|---|---|
| Typ | Binary (Up/Down) |
| Trvani | 5 minut |
| Resolution | Chainlink BTC/USD (automaticka) |
| Fee | Maker: 0% + 20% rebate, Taker: `0.10 * min(p,1-p)^2` |
| Likvidita | ~$40K depth, ~$150K volume/event |
| Tick | $0.01 |
| NegRisk | NE (standardni binary market) |
| Spread | ~1-2 centy |

### Fair Value model

```
P(Up) = Phi( log(S_now / S_start) / (sigma_per_min * sqrt(t_remaining)) )
```

- `S_start` = BTC cena na Binance v moment startu eventu
- `S_now` = aktualni BTC cena (Binance WebSocket, real-time)
- `sigma_per_min` = realized volatility z poslednich 720 minut 1-min klines
- `t_remaining` = minuty do konce eventu
- `Phi` = standardni normalni CDF

### Signal (entry trigger)

Model pouziva `sigma_scale=0.644` pro signal (ne pro cenu). To zpusobuje ze
signal reaguje AGRESIVNEJI na male BTC pohyby nez market. Kdyz signal rika
ze BTC jde nahoru, ale market jeste nezareagoval, vstupujeme.

Entry trigger: `|signal_fv - 0.50| > 0.095`

### Pricing

Vstup a vystup VZDY za market fair value (sigma_scale=1.0), nikdy za model
cenu. Toto oddeleni signalu od ceny zabranne nerealistickym backtestum.

## 3. Optimalni parametry (z autoresearch)

### Autoresearch: 1,900 trialu, 89 dni dat, 17,883 5-min oken

#### Config #1 (nejlepsi OOS score)

| Parametr | Hodnota | Popis |
|---|---|---|
| `sigma_window` | 720 min | 12h Binance dat pro sigma |
| `sigma_method` | realized | std(log_returns) |
| `sigma_scale` | 0.644 | Signal citlivost (agresivnejsi nez market) |
| `entry_threshold` | 0.095 | Min signal odchylka od 0.50 |
| `min_entry_min` | 2 | Vstup od minuty 2 |
| `max_entry_min` | 2 | Vstup POUZE v minute 2 |
| `contrarian` | False | MOMENTUM (nasleduj BTC smer) |
| `profit_target` | 0.207 | Take profit $0.207/share |
| `stop_loss` | 0.038 | Stop loss $0.038/share |
| `max_hold_min` | 3 | Max 3 minuty drzet pozici |
| `edge_exit` | 0.076 | Exit kdyz edge klesne pod 7.6% |
| `exit_before_end` | 0 | Nedrzet pred koncem |
| `allow_resolution` | True | Povolit drzet do rozhodnuti |
| `spread` | 0.01 | Predpokladany spread ($0.01) |
| `maker` | True | PostOnly ordery (0% fee) |
| `position_pct` | 0.067 | 6.7% kapitalu per trade |
| `edge_scaling` | 4.838 | Velikost pozice roste s edge |
| `window_min` | 5 | 5-minutove okna |
| `reentry_cooldown` | 3 | 3 min pred re-entry |

#### Config #5 (EARLY EXIT ONLY — bez resolution rizika)

| Parametr | Hodnota |
|---|---|
| `sigma_window` | 1440 min (24h) |
| `sigma_scale` | 1.05 |
| `entry_threshold` | 0.02 |
| `profit_target` | 0.238 |
| `stop_loss` | 0.122 |
| `allow_resolution` | **False** |
| `max_hold_min` | 3 |

## 4. Walk-Forward vysledky

### In-Sample (70% dat, dny 1-62) → Out-of-Sample (30% dat, dny 63-89)

| # | IS Score | OOS Score | IS Trades | OOS Trades | IS WR | OOS WR | OOS PnL | OOS Sharpe | OOS DD |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 21,116 | **6,140** | 7,324 | 3,329 | 57.4% | 52.1% | $20,285 | **22.0** | 33.1% |
| 2 | 20,169 | 4,275 | 9,916 | 4,427 | 64.9% | 61.8% | $16,602 | 16.9 | 44.6% |
| 3 | 18,278 | 5,856 | 10,538 | 4,666 | 57.5% | 53.7% | $16,615 | 20.3 | **24.5%** |
| 4 | 15,199 | 3,069 | 7,753 | 3,454 | 72.4% | **68.9%** | $10,838 | 17.2 | 37.6% |
| 5* | 14,403 | 2,620 | 10,654 | 4,718 | 59.9% | 54.5% | $12,136 | 13.2 | 47.4% |

*Config #5 = early exit only (allow_resolution=False)

### Konzistentni vzory (vsech 5 OOS profitabilnich):

- **Momentum** (NE contrarian) — vsech 5
- **5-min okna** (NE 15-min) — vsech 5
- **Maker** (PostOnly, 0% fee) — vsech 5
- **Realized volatility** — vsech 5
- **Vstup v minute 2** — vsech 5

## 5. Realisticke odhady

Backtest simuluje VSECHNA mozna 5-min okna (201/den). Polymarket ma ~54 BTC
5-min marketu/den. Korekcni faktor: 54/201 = 0.27.

### Config #1 (resolution allowed)

| Metrika | Backtest OOS | Realisticky odhad |
|---|---|---|
| Trades/den | 123 | **33** |
| PnL/den | $751 | **$202** |
| PnL/mesic | — | **$6,055** |
| Win rate | 52.1% | 52.1% |
| Sharpe | 22.0 | ~10-15 (realisticky) |

### Config #5 (early exit only)

| Metrika | Backtest OOS | Realisticky odhad |
|---|---|---|
| Trades/den | 175 | **47** |
| PnL/den | $449 | **$121** |
| PnL/mesic | — | **$3,630** |
| Win rate | 54.5% | 54.5% |

## 6. Fee model

```python
# Maker (PostOnly) — nase primarne pouziti
entry_fee = 0.0
exit_fee = 0.0
rebate = 0.20 * 0.10 * min(p, 1-p)**2  # POZITIVNI! Vydelavame na fee

# Taker (market order)
fee = 0.10 * min(p, 1-p)**2
# Pri p=0.50: fee = $0.025/share (2.5%)
# Pri p=0.30: fee = $0.009/share (0.9%)
```

## 7. Rizika a omezeni

### Modelova rizika
- **1-min data resolution**: Backtest pouziva 1-min Binance klines. Realne
  intra-minutove pohyby mohou byt jine.
- **Market efficiency assumption**: Model predpoklada ze market_fv = true fair
  value. Pokud market makers maji lepsi model, edge mizeni.
- **Compounding efekt**: P&L cisla zahrnuji compounding. Flat sizing by dal
  konzervativnejsi vysledky.
- **Fill probability**: PostOnly ordery nemusi byt vzdy naplneny. Realne fill
  rate bude nizsi nez 100%.

### Provozni rizika
- **Latence**: Strategie vyzaduje <1s latency na Binance i Polymarket
- **Session coverage**: Markety bezi jen ~12h/den ve 3 sessions. Automatizace
  musi detekovat session starty.
- **Geo-restriction**: `restricted: true` na Up/Down marketech. Potreba VPS
  mimo restricted zemi (Dublin OK).

### Limity
- `MAX_BET_SIZE`: $100 per trade (liquidity constraint)
- `MAX_SHARES`: 500 shares per trade
- `MIN_ENTRY_MKT_FV`: 0.15 (zadne deep OTM pozice)
- `MAX_ENTRY_MKT_FV`: 0.85 (zadne deep ITM pozice)

## 8. Proc to funguje (edge explanace)

### Momentum + konvexita

Po BTC pohybu +X% za prvni 2 minuty 5-min okna:
1. **Momentum**: BTC ma tendenci pokracovat krátkodobe ve stejnem smeru
   (podmineno ze uz se pohnulo dost na trigger signal)
2. **Konvexita**: Fair value funkce je konvexni — ukazatel ze dalsi pohyb
   stejnym smerem zvysi fair_value VIC nez stejny pohyb opacnym smerem.
3. **Maker edge**: 0% fee + rebate znaci ze i maly edge je profitabilni.

### Priklad

```
Minute 0: BTC = $87,000 (event start)
Minute 2: BTC = $87,070 (+0.08%)
  sigma_rem = 0.08% * sqrt(3) = 0.139%
  z = 0.08/0.139 = 0.576
  market_fv_up = Phi(0.576) = 0.718 (71.8% Up)
  → signal triggers (> 0.50 + 0.095)
  → BUY Up at 0.718 + 0.005 spread = 0.723

Minute 4: BTC = $87,100 (+0.115%)
  sigma_rem = 0.08% * sqrt(1) = 0.08%
  z = 0.115/0.08 = 1.44
  market_fv_up = Phi(1.44) = 0.925
  → profit = 0.925 - 0.723 - 0.005 spread = 0.197/share
  → PROFIT TARGET HIT ($0.207)

Minute 5: Resolution → Up wins (BTC stayed up)
  But we already exited at minute 4 with $0.197 profit/share!
```

## 9. Implementace

### Nove soubory

```
arbo/strategies/strategy_b3.py       # Orchestrator
arbo/strategies/b3_scanner.py        # Market discovery (Gamma API tag=up-or-down)
arbo/strategies/b3_quality_gate.py   # Entry filters
arbo/connectors/binance_ws.py        # EXISTUJE (z B2)
arbo/models/volatility_model.py      # EXISTUJE (z B2)
```

### Klicove rozdily od B2

| Aspekt | B2 (daily above) | B3 (5min up/down) |
|---|---|---|
| Market type | "Above X on date" | "Up or Down in 5min" |
| Time horizon | hodiny/dny | 5 minut |
| Entry signal | Vol CDF vs market | Momentum signal od Binance |
| Hold time | hodiny | 1-3 minuty |
| Exit | Edge-based + resolution | Profit target / stop loss |
| Fee | 0% maker na monthly | 0% maker (crypto_fees) |
| Trades/den | 3-5 | **33** |
| NegRisk | Ano | NE |

### Execution flow

```
1. Gamma API: najdi otevrene btc-updown-5m eventy (tag=up-or-down)
2. Pro kazdy event:
   a. Pockat na event start (eventStartTime)
   b. V minute 2: spocitat fair_value z Binance
   c. Pokud |signal_fv - 0.50| > 0.095:
      - PostOnly BUY Up nebo Down na market_fv
      - Start exit monitoring
   d. Exit na: profit_target ($0.207), stop_loss ($0.038),
      max_hold (3min), edge_gone (<7.6%)
3. Dalsi event za 5 minut
```

## 10. Dalsi kroky

1. **Paper trading**: Nasadit na VPS (Dublin) s Binance WS + Polymarket WS
2. **Tick-level data collection**: Sbírat real-time orderbook data pro validaci
   fair value modelu
3. **Multi-asset**: Rozsirit na ETH, SOL (nizsi likvidita, ale stejny princip)
4. **15-min markets**: Testovat 15-min okna (sweep ukazal nizsi performance,
   ale stoji za dalsi vyzkum)

---

*Autoresearch: 1,900 trialu, 89 dni Binance dat (89,419 1-min klines),
17,883 simulovanych 5-min oken. Sweep: `research/innovations/sweep_b3_scalper.py`*
