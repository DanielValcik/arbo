# BTC/ETH Statistical Arbitrage – Backtest Specification

## Cíl

Ověřit, zda pairs trading strategie na BTC/ETH páru generuje konzistentní výnosy po poplatcích na historických datech od vzniku ETH (2015) do současnosti. Výsledek rozhodne, zda má smysl buildovat produkční trading bot.

---

## Strategie – jak to funguje

BTC a ETH jsou kointegrovatelné páry – pohybují se spolu dlouhodobě, ale krátkodobě se jejich poměr odchyluje od normálu. Strategie vsází na návrat tohoto poměru zpátky.

**Logika:**
1. Spočítej historický "spread" mezi log(BTC) a log(ETH)
2. Sleduj Z-score spreadu (kolik standardních odchylek je aktuální hodnota od průměru)
3. Když Z-score překročí práh → otevři pozici
4. Když se Z-score vrátí k nule → zavři pozici a zaúčtuj profit/loss

---

## Data

- **Zdroj:** CoinGecko API (free, bez API klíče pro základní historii) nebo Binance public API
- **Páry:** BTC/USDT a ETH/USDT
- **Timeframe:** Denní OHLCV data, od 2015-08-01 (vznik ETH na burzách) do dnes
- **Záložní zdroj:** `yfinance` (BTC-USD, ETH-USD) – jednodušší implementace

---

## Backtest parametry

### Pair formation
- Metoda: OLS regrese `log(ETH) = β * log(BTC) + α + ε`
- Rolling window pro výpočet spreadu: **60 dní** (re-kalibruje se každý den)
- Kointegrace test: Augmented Dickey-Fuller (ADF) na reziduálech, p-value < 0.05

### Vstupní/výstupní signály
| Podmínka | Akce |
|----------|------|
| Z-score > +2.0 | Short BTC, Long ETH |
| Z-score < -2.0 | Long BTC, Short ETH |
| \|Z-score\| < 0.5 | Zavři pozici (take profit) |
| Z-score > +3.5 nebo < -3.5 | Zavři pozici (stop-loss – spread se nevrací) |

### Transakční náklady
- **0.10% per strana** (maker fee na Binance/Bybit pro futures)
- Tedy **0.20% celkem při vstupu** (long + short leg) + **0.20% při výstupu**
- Celkem **0.40% per round-trip obchod**

### Pozice sizing
- Každý obchod alokuje **10% portfolia** (konzervativní, bez leverage)
- Maximálně **1 otevřená pozice** najednou v tomto základním backttestu

---

## Výstupy které chceme vidět

### Výkonnostní metriky
- Celkový výnos (%) za celé období
- Annualized return (%)
- Sharpe ratio
- Max drawdown (%)
- Win rate (% ziskových obchodů)
- Průměrný zisk / průměrná ztráta per obchod
- Počet obchodů celkem

### Grafy
1. **Equity curve** – vývoj portfolia v čase vs. buy & hold BTC
2. **Z-score v čase** – s vyznačenými vstupními/výstupními body
3. **Spread v čase** – log(ETH) − β*log(BTC)
4. **Roční výnosy** – bar chart po jednotlivých letech

---

## Technický stack

```python
# Závislosti
yfinance          # stažení dat
pandas            # manipulace dat
numpy             # výpočty
statsmodels       # OLS regrese, ADF test
matplotlib        # grafy
```

Žádné placené API klíče nejsou potřeba.

---

## Struktura kódu

```
btc_eth_statarb/
├── data_loader.py      # stažení a cachování dat
├── strategy.py         # výpočet spreadu, Z-score, signálů
├── backtest.py         # simulace obchodů, tracking P&L
├── metrics.py          # výpočet výkonnostních metrik
├── plots.py            # generování grafů
└── main.py             # spustí vše a vypíše výsledky
```

---

## Očekávané výsledky (benchmark z literatury)

Pokud backtest odpovídá akademickým studiím, měli bychom vidět:
- Sharpe ratio: 1.0–1.5
- Roční výnos po poplatcích: 15–30%
- Max drawdown: pod 25%

Pokud výsledky jsou výrazně horší → strategie na BTC/ETH nefunguje a přejdeme na jiné páry nebo timeframe.

---

## Následující kroky po backttestu

| Výsledek | Akce |
|----------|------|
| Sharpe > 1.0, výnos > 15% | Buildovat live trading bot |
| Sharpe 0.5–1.0 | Optimalizovat parametry, testovat více párů |
| Sharpe < 0.5 | Zahodit nebo změnit strategii |

---

## Poznámky pro vývojáře

- Backtest musí zahrnovat **transakční náklady** – bez nich jsou výsledky nerealisticky dobré
- Používat **out-of-sample testování**: formace na 2015–2020, trading na 2021–2025
- Pozor na **look-ahead bias** – Z-score musí být počítán pouze z dat dostupných v daný moment
- Rolling window je důležitá – statický spread by ignoroval změny vztahu BTC/ETH v čase
