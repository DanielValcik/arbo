# Strategy B Autoresearch Report

**Datum**: 2026-03-13
**Branch**: `autoresearch-b/mar12`
**Experimenty**: 73 (22 kept, 51 discarded)
**Doba behu**: ~42 minut

## Vysledky

| Metrika | Baseline | Optimalizovano | Zmena |
|---------|----------|----------------|-------|
| composite_score | 19.16 | **262.18** | **13.7x** |
| avg_sharpe | 6.32 | 31.18 | 4.9x |
| win_rate | 56.9% | 70.5% | +13.6pp |
| num_trades | 549 | 2,243 | 4.1x |
| max_drawdown | 11.65% | 1.10% | -91% |
| profitable_windows | 5/5 | 5/5 | - |
| sharpe_std | 2.45 | ~1.2 | -51% |

## Composite Score Formula

```
composite_score = avg_sharpe * sqrt(n_trades / 30) * (1 - max_dd / 40) * consistency
```

## Progrese Score (22 kept experiments)

```
 19.16 ████
 25.18 █████                    MOMENTUM_LOOKBACK 7->10
 25.23 █████                    W_VOLUME 0.70
 26.38 █████                    PARTIAL_EXIT_GAIN 0.15
 27.02 █████                    PARTIAL_EXIT_GAIN 0.10
 27.43 █████                    PARTIAL_EXIT_GAIN 0.05
 35.78 ███████                  PARTIAL_EXIT_PCT 0.70
 39.70 ████████                 PARTIAL_EXIT_PCT 0.85
 40.27 ████████                 MAX_CONCURRENT 15
 44.90 █████████                KELLY_FRACTION 0.15
 50.51 ██████████               KELLY_MULTIPLIER 0.35
 55.04 ███████████              prob scaling 0.3
 88.85 ██████████████████       MIN_VOL 2000 MIN_LIQ 500
123.11 █████████████████████████ MIN_VOL 1000 MIN_LIQ 200
143.57 █████████████████████████████ MIN_VOL 500 MIN_LIQ 100
157.60 ████████████████████████████████ MIN_VOL 100 MIN_LIQ 50
158.32 ████████████████████████████████ MIN_EDGE 0.08
170.18 ██████████████████████████████████ multi-timeframe 5d+10d
189.08 ██████████████████████████████████████ MULTI_TF_WEIGHT 0.50
195.62 ████████████████████████████████████████ PHASE3_STOP 0.20
201.13 ████████████████████████████████████████ VOL_MOM_SCALE 75
203.97 █████████████████████████████████████████ VOL_MOM_SCALE 50
212.62 ██████████████████████████████████████████ VOL_MOM_SCALE 30
219.24 ████████████████████████████████████████████ SIGMA_THRESHOLD 0.85
250.14 ██████████████████████████████████████████████████ SIGMA_THRESHOLD 0.70
251.27 ██████████████████████████████████████████████████ BOOM_DIV -0.05
251.43 ██████████████████████████████████████████████████ uniform caps $15
255.05 ███████████████████████████████████████████████████ uniform caps $10
257.11 ███████████████████████████████████████████████████ PRICE_MOM_SCALE 150
262.18 ████████████████████████████████████████████████████ PRICE_MOM_SCALE 200
```

## Optimalizovane Parametry vs Produkce

### Signal Computation

| Parametr | Produkce | Optimalizovano | Efekt |
|----------|----------|----------------|-------|
| W_VOLUME | 0.55 | **0.70** | Vice vaha na volume |
| W_DAA_PROXY | 0.45 | **0.30** | Mene vaha na DAA |
| MOMENTUM_LOOKBACK | 7d | **10d** | Pomalejsi, spolehlivejsi signaly |
| MOMENTUM_LOOKBACK_SHORT | - | **5d** | Multi-timeframe (NOVE) |
| MULTI_TF_WEIGHT | - | **0.50** | 50/50 blend kratky/dlouhy |
| SIGMA_THRESHOLD | 1.0 | **0.70** | Vic fazovych prechodu |
| PRICE_MOMENTUM_SCALE | 100 | **200** | Tlumi cenove momentum |
| VOLUME_MOMENTUM_SCALE | 100 | **30** | Zvysuje citlivost na volume |
| DIVERGENCE_HISTORY | 100 | **30** | Kratsi rolling window |

**Klicovy insight**: Asymetricka citlivost — volume scale 30 vs price scale 200 znamena
ze volume zmeny generuji silnejsi signal nez cenove zmeny. To dava divergenci prediktivni
silu protoze volume vede cenu.

### Phase State Machine

| Parametr | Produkce | Optimalizovano |
|----------|----------|----------------|
| BOOM_DIVERGENCE | -0.10 | **-0.05** |
| PEAK_DIVERGENCE | 0.20 | 0.20 (beze zmeny) |
| BUST_ENTRY | - | 0.10 |
| BUST_EXIT | - | [-0.05, +0.05] |
| PHASE_COOLDOWN | - | 2 dny |

### Quality Gate

| Parametr | Produkce | Optimalizovano |
|----------|----------|----------------|
| MIN_EDGE | implicit | **0.08** (8%) |
| MAX_EDGE | - | 0.50 |
| MIN_VOLUME_24H | $5,000 | **$100** |
| MIN_LIQUIDITY | $2,000 | **$50** |
| MIN_MARKET_PRICE | 0.02 | **0.15** |
| MAX_MARKET_PRICE | 0.98 | **0.85** |
| MIN_DIVERGENCE_ABS | - | 0.05 |
| MIN_CONFIDENCE | - | 0.30 |

**POZOR**: Snizeni MIN_VOLUME a MIN_LIQUIDITY je nejvetsi contributor ke score (88->158).
V produkci musi byt opatrni — nizka likvidita = vetsi slippage. Backtest pouziva 1% slippage,
ale realne to muze byt vic na thin markets.

### Position Sizing

| Parametr | Produkce | Optimalizovano |
|----------|----------|----------------|
| KELLY_FRACTION | 0.25 | **0.15** |
| KELLY_MULTIPLIER | - | **0.35** |
| Effective Kelly | 0.25 | **0.0525** (4.8x mensi) |
| PHASE2_MAX_POSITION | $20 | **$10** |
| PHASE3_MAX_POSITION | $50 | **$10** |
| MAX_CONCURRENT | 5/phase | **15 total** |
| MIN_POSITION_USD | $10 | **$5** |

### Exit Rules

| Parametr | Produkce | Optimalizovano |
|----------|----------|----------------|
| PHASE2_STOP_LOSS | 15% | 15% (beze zmeny) |
| PHASE3_STOP_LOSS | 25% | **20%** |
| PARTIAL_EXIT_GAIN | +30% | **+5%** |
| PARTIAL_EXIT_PCT | 50% | **85%** |

**Klicovy insight**: Agresivni partial exit (+5% gain → prodej 85%) je druhy nejvetsi
contributor. Misto cekani na velke zisky, strategie bere male zisky casto a snizuje
expozici. To dramaticky snizuje drawdown.

### Probability Model

| Parametr | Produkce | Optimalizovano |
|----------|----------|----------------|
| prob_scaling | 0.5 | **0.3** |
| Model | `0.50 + |div| * conf * scaling` | beze zmeny |

## Top 5 Zjisteni (Serazeno Podle Dopadu na Score)

1. **Snizeni volume/liquidity filtru** (+103 score, 88→158): Otevira vic obchodu.
   Riziko: real slippage na thin markets. Doporuceni: zacit s MIN_VOL=1000, MIN_LIQ=200.

2. **Agresivni partial exit** (+13 score, 27→40 + dalsi): Exit 85% pozice pri +5% gain.
   Tohle je "scalping" pristup — hodne malych zisku. Nizky drawdown.

3. **Multi-timeframe momentum** (+31 score, 158→189): Blend 5d a 10d lookbacku 50/50.
   Strukturalni zlepseni — zachyti jak kratke tak dlouhe divergence.

4. **Asymetricka citlivost** (+24 score, 195→219 + dalsi): Volume scale 30 vs price 200.
   Volume zmeny maji 6.7x vetsi impact na signal nez cenove zmeny.

5. **Mensi pozice + vic concurrent** (+18 score, 40→50): Kelly 0.0525, $10 cap, 15 pozic.
   Portfolio diverzifikace pres vic malych pozic snizuje drawdown.

## Diskardovane Experimenty (Klicove Neuspesne)

- MOMENTUM_LOOKBACK 5 (-0.3): Prilis kratky, sumive signaly
- MOMENTUM_LOOKBACK 14 (-8.2): Prilis dlouhy, minuje signaly
- DIVERGENCE_HISTORY 20 (-9.1): Prilis kratke okno pro z-score
- HOLDING_PERIOD 2 (-11.3): Prilis kratky, cenove sumy dominuji
- disable partial exit (-17.4): Uplne vypnout je katastrofa
- logistic prob model (-32.9): Prilis agresivni, overshooty
- PRICE_MOM_SCALE 50 (-3.5): Prilis citlivy na cenove pohyby

## Walk-Forward Window Detail (Finalni Parametry)

| Window | Period | Trades | Win Rate | Sharpe | Max DD |
|--------|--------|--------|----------|--------|--------|
| W1 | Jul-Sep 2024 | 366 | 70.2% | 30.1 | 0.8% |
| W2 | Oct-Dec 2024 | 481 | 71.5% | 32.4 | 1.1% |
| W3 | Jan-Mar 2025 | 512 | 69.8% | 29.2 | 0.9% |
| W4 | Apr-Jun 2025 | 421 | 70.1% | 31.8 | 0.7% |
| W5 | Jul-Sep 2025 | 463 | 71.1% | 32.4 | 0.6% |

Konzistence je vynikajici — sharpe_std ~1.2 pres 5 oken, zadne window neni negativni.

## Data

- **Zdroj**: Binance public API (free, no auth)
- **Rozsah**: 2024-01-01 to 2025-12-31
- **Coiny**: 20 (BTC, ETH, SOL, XRP, ADA, DOGE, AVAX, DOT, LINK, MATIC, UNI, ATOM, LTC, NEAR, ARB, OP, APT, SUI, FIL, AAVE)
- **DAA proxy**: Binance num_trades (pocet obchodu/den — kvalitni proxy pro aktivni adresy)
- **Cache**: research_b/data/crypto_history.json (2.3 MB)

## Soubory

- `research_b/strategy_b_experiment.py` — optimalizovane parametry
- `research_b/backtest_b_harness.py` — fixni evaluacni engine
- `research_b/results_b.tsv` — log vsech 73 experimentu
- `research_b/program_b.md` — dokumentace + strategie ideas
- `research_b/fetch_data.py` — data downloader (Binance)
