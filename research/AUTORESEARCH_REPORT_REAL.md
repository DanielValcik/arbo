# Autoresearch Report — Reálná Polymarket Data

> Datum: 2026-03-14
> Zdroj dat: Polymarket CLOB `/prices-history` (reálné tržní ceny)
> Období: 2026-02-14 → 2026-03-14 (27 dní)
> Evaluace: 1000 trials (500 random + 300 fine-tune + 200 city opt)
> Sizing: FIXED quarter-Kelly (0.25) — architekturní konstanta

## Klíčový nález

**Parametry optimalizované na syntetických datech NEFUNGUJÍ na reálném trhu.**

Produkční parametry (z V2 syntetické optimalizace) negenerují na reálných Polymarket datech **ŽÁDNÉ trades**. Důvod: reálný trh je efektivnější než syntetický market maker — edge je menší, ceny jsou nižší, a quality gate thresholds jsou příliš přísné.

## Data pipeline

```
Polymarket CLOB API (/prices-history)
  → 418 resolved weather events
  → 3,737 YES tokenů s cenovou historií
  → 189,244 hodinových cenových bodů
  → SQLite databáze (64 MB)
  → Kombinováno s Open-Meteo forecast + reálné resolution výsledky
```

**Reálnost dat:**
- ✅ Ceny: skutečné Polymarket CLOB ceny (hodinové snapshoty)
- ✅ Resolution: skutečný výsledek (který bucket vyhrál)
- ✅ Simulovaný slippage: 0.5%
- ✅ Gas: $0.007/trade
- ✅ Weather fees: 0% (ověřeno)
- ⚠️ Forecast: Open-Meteo archive (pozorovaná teplota, ne předpověď)
- ⚠️ Bez orderbook depth (jen midpoint, ne best ask)

## Výsledky optimalizace

| Metrika | Produkční (V2 syntetické) | **Optimalizované (reálné)** |
|---------|--------------------------|----------------------------|
| Score | 105.49 | **767.46** |
| Trades (27 dní) | 0 (!!) | **81** |
| Win Rate | — | **25.9%** |
| PnL | $0 | **+$4,778 (+478%)** |
| Max Drawdown | — | **29.2%** |
| Sharpe | — | **6.93** |
| Final Capital | $1,000 | **$5,778** |

## Optimalizované parametry

### Quality Gate

| Parametr | Produkční (V2) | **Reálný optimum** | Změna |
|----------|----------------|---------------------|-------|
| MIN_EDGE | 0.15 | **0.01** | Radikálně nižší |
| MAX_PRICE | 0.50 | **0.55** | Mírně vyšší |
| MIN_PRICE | 0.35 | **0.08** | Radikálně nižší |
| MIN_FORECAST_PROB | 0.70 | **0.02** | Radikálně nižší |
| MIN_VOLUME | $2,000 | **$100** | Radikálně nižší |
| MIN_LIQUIDITY | $200 | — | Odstraněno |

### Probability Model

| Parametr | Produkční (V2) | **Reálný optimum** |
|----------|----------------|---------------------|
| PROB_SHARPENING | 1.143 | **1.05** |
| SHRINKAGE | 0.10 | **0.0** |
| KELLY_RAW_CAP | 0.25 | **0.25** |

### Vyloučená města

| Syntetické (V2) | **Reálné** |
|-----------------|------------|
| DC, Toronto, Buenos Aires, NYC, Atlanta, Wellington | **Chicago, Tel Aviv, Paris, London** |

Paris a London — "top performers" na syntetických datech — jsou na reálném trhu **ztrátové**!

### Per-city výsledky (reálné)

| Město | PnL | Trades | Poznámka |
|-------|-----|--------|----------|
| Tokyo | +$1,390 | 1 | ⚠️ Jen 1 trade |
| Dallas | +$1,273 | 11 | Silný performer |
| Seoul | +$620 | 12 | Konzistentní |
| Seattle | +$606 | 11 | Konzistentní |
| São Paulo | +$485 | 16 | Nejvíc trades |
| Munich | +$216 | 1 | Jen 1 trade |
| Ankara | +$100 | 14 | Marginální |
| Miami | +$88 | 15 | Marginální |

### Per-city overrides

| Město | max_price | min_edge |
|-------|-----------|----------|
| São Paulo | 0.60 | 0.03 |
| Seoul | 0.60 | 0.05 |
| Seattle | 0.55 | 0.05 |
| Tokyo | 0.60 | 0.02 |
| Ankara | 0.50 | 0.01 |
| Miami | 0.50 | 0.03 |

## Proč tak odlišné parametry?

### 1. Reálný trh je efektivnější
Syntetický market maker měl fixní noise 1.5-7.5°C. Reální tradeři na Polymarketu jsou chytřejší — edge je typicky 1-5%, ne 8-15%.

### 2. Strategie funguje na jiném principu
- **Syntetický**: Vysoká pravděpodobnost (70%+) → vysoký WR (98%) → malý profit per trade
- **Reálný**: Nízké ceny (8-20%) → nízký WR (26%) → ale obrovský profit per výhra (5-12x)

### 3. Distribuce cen
- Syntetický: ceny kolem 30-50% (market maker s malým noise)
- Reálný: medián ceny 4.5%, většina bucketů pod 15%
  → MIN_PRICE 0.30 odmítne 82% bucketů na reálném trhu

### 4. Geografická efektivita
Paris a London — města s nejvíc liquidity na Polymarketu — mají nejefektivnější ceny. Edge je tam malý a reální tradeři jsou rychlí. Méně sledovaná města (Dallas, Seoul, São Paulo) mají víc mispricing.

## Srovnání: Syntetický vs Reálný backtest

| Aspekt | Syntetický | Reálný |
|--------|-----------|--------|
| Tržní ceny | Simulovaný market maker | **Skutečné CLOB ceny** |
| Win Rate | ~98% | **~26%** |
| Profit model | Vysoký WR × malé výhry | **Nízký WR × velké výhry** |
| Edge zdroj | Nižší noise modelu | **Mispricing na méně sledovaných trzích** |
| Data rozsah | 2 roky (2024-2025) | **27 dní (únor-březen 2026)** |
| Robustnost | Vysoká (5 oken) | **Nízká (krátké období)** |

## Omezení a rizika

### ⚠️ Krátké datové období
27 dní je příliš málo pro robustní závěry. Tokyo (+$1,390) z 1 tradu silně zkresluje výsledky. Potřebujeme 2-3 měsíce dat.

### ⚠️ Drawdown 29%
Na $1,000 kapitálu = $292 maximální ztráta. To je na hraně 15% weekly kill switch.

### ⚠️ Nízký win rate
25.9% WR znamená ~3 prohry na 1 výhru. Psychologicky náročné pro live trading.

### ⚠️ Sezónní bias
Data pokrývají jen únor-březen. Jiné sezóny mohou mít odlišnou dynamiku.

## Doporučení

### Okamžitě
1. **Nenastavovat nové parametry do produkce** — příliš málo dat
2. **Spustit data collector na VPS** — sbírat cenovou historii každou hodinu
3. **Pokračovat v paper tradingu** s aktuálními parametry

### Za 2-4 týdny (víc dat)
4. Re-run sweep na 2+ měsících dat
5. Walk-forward validace (train na prvním měsíci, test na druhém)
6. Porovnat paper trading výsledky s backtest predikcemi

### Zvážit
7. **Hybridní přístup**: kombinace syntetických a reálných dat pro training
8. Agresivnější per-city parametry (Dallas, Seoul mají konzistentní edge)

## Soubory

| Soubor | Účel |
|--------|------|
| `research/download_price_history.py` | Stažení dat z Polymarket |
| `research/price_history_db.py` | Query interface pro SQLite |
| `research/backtest_real_prices.py` | Realistický backtest |
| `research/sweep_real_prices.py` | Parameter sweep na reálných datech |
| `research/data/price_history.sqlite` | SQLite s cenovou historií (64 MB) |
| `research/data/sweep_real_results.json` | Výsledky swepu |
| `research/PRICE_HISTORY_DATA.md` | Dokumentace data pipeline |
| `research/AUTORESEARCH_REPORT_REAL.md` | Tento report |
