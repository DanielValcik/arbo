# TradingView MCP & Technick Analysis Integration — Analza a Specifikace

> **Status**: RESEARCH COMPLETE — implementační plán viz `TA_INTEGRATION_SPEC.md`
> **Datum**: 2026-04-11
> **Autor**: CTO
> **Scope**: B3, B2, D + celkov ekosystm
> **Related**: `TA_INTEGRATION_SPEC.md` (implementace), `B3_WATCHDOG_SPEC.md` (autonomní watchdog), `TECHNICAL_DECISIONS.md` (TD-014 až TD-017)

---

## 1. Executive Summary

TradingView MCP (Model Context Protocol) je ekosystm komunitnch server, kter
zpstupuj technickou analzu (RSI, MACD, Bollinger, ADX, 50+ indiktr) pro
AI agenty. Neexistuje oficil TradingView API  ve jsou neoficin wrappery
nad internmi endpointy TradingView.

**Klov zjistn pro Arbo:**

1. **B3 (5-min BTC scalper)**: TA indiktory mohou zlepit regime detection
   a filtraci obchod o odhadovanch +15-25% PnL. ALE mus bt background
   feature (60s cache), NIKDY v hot path (latence MCP = sekundy).

2. **Architektura je DUAL**: MCP pro interaktivn research s Claude Code,
   Python knihovna (`tradingview-ta`) pro produkn kd.

3. **B2 (crypto daily)**: VYSOK hodnota  denn RSI/MACD/BB pm mapuj
   na denn "above" trhy.

4. **D (sports)**: NULOV hodnota  finann TA nem relevantn pro NBA.

5. **C/C2 (weather)**: NULOV hodnota.

6. **Rizika**: Overfitting (vce indiktr  vce), scraping-based (me
   se zlomit), korelace s existujcmi features.

---

## 2. Ekosystm TradingView MCP  Kompletn Mapovn

### 2.1 Dva Fundamentln Typy

**Typ A  Data API wrappery (bez TradingView tu)**
- Pouv Python knihovny (`tradingview-ta`, `tradingview-screener`)
- Vol internch TradingView scanner endpoint pes HTTP
- dn API kle, dn pedplatn
- Crypto data z Binance jsou near-real-time (na rozdl od akci, kter jsou 15-20 min zpodn)

**Typ B  Desktop bridge (vyaduje TradingView Desktop)**
- Pipojuje se k TradingView Electron app pes Chrome DevTools Protocol (localhost:9222)
- Vyaduje placen pedplatn TradingView ($15-60/ms)
- Pstup k emukoli, co je na vaem chartu vcetn Pine Script

### 2.2 Top Repozite (Ranking dle relevance pro Arbo)

| # | Repo | Stars | Typ | Klov Schopnosti | Pro Arbo |
|---|------|-------|-----|-----------------|----------|
| 1 | `atilaahmettaner/tradingview-mcp` | 1,638 | A (Python) | 30+ tool: TA, backtesting (6 strategi), candlestick patterns (15), multi-timeframe, Reddit sentiment, Yahoo Finance | **NEJRELEVANTNJ** |
| 2 | `tradesdontlie/tradingview-mcp` | 1,619 | B (JS) | 78 tool: Pine Script, replay, chart screenshot, alert create/delete | Research only |
| 3 | `fiale-plus/tradingview-mcp-server` | 21 | A (TypeScript) | 12 tool: 100+ screener polí, 14 presets, custom timeframes, fundamental data | Screener monosti |
| 4 | `bidouilles/mcp-tradingview-server` | 18 | A (Python) | 3 tool: indicators, OHLCV streaming, specific symbol | Nejjednodu setup |
| 5 | `crypto-indicators-mcp` | 122 | Standalone | 50+ indiktr s BUY/SELL/HOLD signly, CCXT-powered (Binance) | **B3 feature source** |

### 2.3 Dostupn Indiktory (pes tradingview-ta)

**Osciltory / Momentum:**
- RSI (14), Stochastic %K/%D, CCI (20), Momentum, MACD (signal + histogram)
- Williams %R, Ultimate Oscillator, Bull/Bear Power, Awesome Oscillator, Stochastic RSI

**Trend:**
- ADX (14) + DI+/DI-, Ichimoku (Base Line)
- EMA: 10, 20, 50, 100, 200
- SMA: 10, 20, 50, 100, 200
- Hull MA (9), VWMA

**Volatilita:**
- ATR (Average True Range)
- Bollinger Bands (20, 2)
- Bollinger %B (pozice v rámci pásem)

**Volume:**
- VWMA (Volume Weighted Moving Average)

**Sumrn signly:**
- `Recommend.All`: -1 (Strong Sell) a +1 (Strong Buy)
- `Recommend.MA`: summary moving averages
- `Recommend.Other`: summary oscillators

**asov rmce**: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 1d, 1W, 1M

### 2.4 Instalace a Nklady

```bash
# Varianta 1: MCP server pro Claude Code (interaktivn)
pip install tradingview-mcp-server

# Varianta 2: Pímá Python knihovna (produkční)
pip install tradingview_ta
```

**Nklady: $0** (dn API kle, dn pedplatn)

**Rate limity**: Nedokumentovan (TradingView me throttlovat pi nadmrnm scrapingu).
Proxy podpora k dispozici.

---

## 3. Analza: TradingView pro B3 (5-min BTC Scalper)

### 3.1 Pochopen B3 Edge

B3 je **temporln arbitr**, ne direkcionln predikce. Edge pochz z:

```
Binance BTC price (lag <100ms) → model FV computation → entry signal
    vs.
Polymarket market maker repricing (lag 2-60s) → stale orderbook
    vs.
Chainlink oracle resolution (lag 1-2s) → final truth
```

B3 vydlv, kdy:
1. BTC se pohne (nap. +$50 za 2 minuty)
2. Polymarket market maker jet nestihl repriceovat
3. B3 koup za model FV (lepší ne trn cena)
4. Drí do Chainlink resoluce (never-sell V6.0)

### 3.2 Co TA Indiktory Mohou a NEMOHOU pro B3

#### CO MOHOU (Additivní kontext):

| Use Case | Indikátor | Mechanismus | Oekvan Impact |
|----------|-----------|-------------|----------------|
| **Regime Detection** | ADX(14) na 5min | ADX > 30 = siln trend  momentum funguje lp | Filtruje CALM regime (29% WR) |
| **Mean-Reversion Risk** | RSI(14) na 5min | RSI > 80 = overbought  UP signal riskantn | Filtruje 30%+ edge trades (36.8% WR) |
| **Squeeze Detection** | BB width na 5min | zk psma = ekn vbuch  bt pipraven | Identifikuje pechod CALM  VOLATILE |
| **Volume Confirmation** | OBV/VWAP | Vysok volume na pohybu = udriteln momentum | Odli spike od bnfide trendu |
| **Multi-TF Alignment** | RSI/MACD na 1h+4h | V TF shodné = vt dvra | Sizing multiplier |
| **Trend Strength** | MACD histogram 5min | Histogram sm  potvrd momentum | Filtruje fading momentum |

#### CO NEMOHOU:

1. **Nahradit core model** — B3 edge je matematick (Normal CDF na oracle lag), ne pattern-based
2. **Bt v hot path** — TA call trvá 200ms-2s, B3 potebuje <100ms latenci
3. **Zlepit timing** — B3 vstupuje v minut 2 (deterministick), TA by pidala um
4. **Predikovat Chainlink** — TA indiktory pracuj s Binance daty, ne s Chainlink orákulem

### 3.3 Kvantifikace Potenciln Improvement

**Zklady z dat (278 live + 142 paper):**

| Metrika | Bez TA | S TA (odhad) | Mechanismus |
|---------|--------|-------------|-------------|
| Live WR (cel) | 87% | 90-92% | Vyfiltruje 2-3 bad trades z 15 |
| CALM regime WR | 29% paper | 40-50% | ADX filter skip nebo men sizing |
| 30%+ edge WR | 36.8% | 50-55% | RSI overbought filter |
| Avg PnL/trade | +$0.80 | +$0.95-1.10 | Lep sizing v dobrch reimech |
| Denn PnL | +$5.4 | +$6.5-7.0 | Menší ztráty + vt pozice v dobrch podmínkách |
| Msn PnL (odhad) | ~$81 | ~$100-110 | +25-35% zlepšení |

**Konzervatívn odhad: +15-25% PnL improvement.**

**Caveat**: Tyto odhady jsou extrapolace z 278 live tradů. Skutečný impact závisí na:
- Korelaci TA features s existujícími (sigma_norm, velocity, dir_delta)
- Kvalitě implementace (background cache, ne blocking)
- Stabilitě tradingview-ta endpoint

### 3.4 Konkrétní Integration Design pro B3

```
PRODUKČNÍ ARCHITEKTURA:

┌─────────────────────────────────────────────────────────┐
│                    VPS arbo-dublin                        │
│                                                          │
│  ┌──────────────────┐     ┌─────────────────────────┐   │
│  │ Background TA     │     │    strategy_b3.py        │   │
│  │ Feature Updater   │     │                          │   │
│  │                   │     │  poll_cycle() [10-15s]:   │   │
│  │ Každých 60s:      │     │    1. Binance WS price    │   │
│  │  tradingview_ta   │     │    2. CL buffer price     │   │
│  │  → BTC 5min:      │     │    3. Vol estimator       │   │
│  │    RSI, ADX,      │────▶│    4. Scanner → signals   │   │
│  │    MACD, BB width │     │    5. TA features (cache) │   │
│  │  → BTC 1h:        │     │    6. Entry decision      │   │
│  │    RSI, trend     │     │    7. Live execution      │   │
│  │  → BTC 4h:        │     │                          │   │
│  │    RSI, ADX       │     │  TA features NIKDY       │   │
│  │                   │     │  neblokují trade loop!    │   │
│  └──────────────────┘     └─────────────────────────┘   │
│                                                          │
│  ┌──────────────────┐                                    │
│  │ B3 Watchdog       │                                    │
│  │                   │                                    │
│  │ Každých 6h:       │                                    │
│  │  - Regime metrics │                                    │
│  │  - TA alignment   │                                    │
│  │  - Anomaly check  │                                    │
│  │  → Gemini Flash   │                                    │
│  │  → Slack report   │                                    │
│  └──────────────────┘                                    │
└─────────────────────────────────────────────────────────┘
```

### 3.5 TA Feature Schema pro B3

```python
@dataclass
class TAFeatures:
    """Cached technical analysis features, updated every 60s."""
    timestamp: float

    # 5-minute timeframe (primary for B3)
    rsi_5m: float          # RSI(14) on 5min candles
    adx_5m: float          # ADX(14) on 5min — trend strength
    macd_hist_5m: float    # MACD histogram on 5min — momentum direction
    bb_width_5m: float     # Bollinger Band width (upper-lower)/middle
    bb_pctb_5m: float      # %B — position within bands (0-1)
    recommend_5m: float    # TradingView summary (-1 to +1)

    # 1-hour timeframe (context)
    rsi_1h: float
    adx_1h: float
    recommend_1h: float

    # 4-hour timeframe (macro)
    rsi_4h: float
    adx_4h: float
    recommend_4h: float

    # Derived
    multi_tf_aligned: bool    # All timeframes agree on direction
    regime: str               # "CALM" | "TRENDING" | "SQUEEZE" | "VOLATILE"
    ta_confidence: float      # 0.0 to 1.0 composite score

    @property
    def is_stale(self) -> bool:
        """Feature cache older than 90s is stale."""
        return time.time() - self.timestamp > 90
```

### 3.6 Pouití v Entry Decision

```python
# V poll_cycle() — NEPRODLUŽUJE latenci (čte z cache):

ta = self._ta_cache  # Updated every 60s in background task

# 1. REGIME FILTER (strongest impact)
if ta and not ta.is_stale:
    if ta.adx_5m < 15 and ta.bb_width_5m < bb_width_20th_pct:
        # CALM + SQUEEZE: low edge, wait for breakout
        # Evidence: CALM regime = 29% paper WR, σn<1.5
        logger.info("b3_ta_calm_regime", adx=ta.adx_5m, bb_w=ta.bb_width_5m)
        continue  # Skip this signal

# 2. MEAN-REVERSION RISK FILTER
    if sig.direction == 1 and ta.rsi_5m > 80:
        # Overbought + UP signal = mean reversion risk
        # Evidence: 30%+ edge trades = 36.8% WR (unprofitable)
        logger.info("b3_ta_overbought_skip", rsi=ta.rsi_5m)
        continue
    if sig.direction == -1 and ta.rsi_5m < 20:
        # Oversold + DOWN signal = bounce risk
        logger.info("b3_ta_oversold_skip", rsi=ta.rsi_5m)
        continue

# 3. MOMENTUM CONFIRMATION
    if ta.macd_hist_5m > 0 and sig.direction == -1:
        # MACD says UP, we want DOWN — reduce size
        size_mult *= 0.7
    elif ta.macd_hist_5m < 0 and sig.direction == 1:
        # MACD says DOWN, we want UP — reduce size
        size_mult *= 0.7

# 4. MULTI-TF SIZING BOOST
    if ta.multi_tf_aligned:
        # All timeframes agree — higher confidence
        size_mult *= 1.3  # +30% sizing
```

### 3.7 Watchdog Enrichment

B3 Watchdog (ji navren, neimplementovn) by zísmal bohatší kontext:

```python
# V Watchdog metrics engine:
watchdog_context = {
    "regime_ta": ta.regime,                    # TA-based regime
    "regime_sigma": sigma_regime,              # Existing σ-based regime
    "regime_agreement": ta.regime == sigma_regime,  # Shoda?
    "rsi_at_entry": ta.rsi_5m,                # RSI pí vstupu
    "adx_at_entry": ta.adx_5m,                # ADX trend strength
    "multi_tf_aligned": ta.multi_tf_aligned,    # MTF confluence
}

# Gemini Flash dostane bohatší prompt:
# "Trade happened during VOLATILE regime (σ=2.3) with TRENDING TA regime
#  (ADX=34, RSI=62). Multi-TF aligned=True. Result: WIN. Is this pattern
#  consistent with our profitable trades?"
```

---

## 4. Analýza: TradingView pro B2 (Crypto Price Edge — Daily)

### 4.1 Proč B2 Profituje z TA Více Než B3

B2 obchoduje **denní "above"** trhy (nap. "BTC above $87,000 on April 12"). Drží hodiny až dny.
Na tomto časovém horizontu jsou TA indikátory **přímo relevantní** pro predikci:

| B3 (5 minut) | B2 (hodiny/dny) |
|---------------|-----------------|
| TA je background kontext | TA je potenciálně **primary signal booster** |
| RSI(5min) je noisy | RSI(daily) má prediktivní sílu |
| MACD(5min) lagging | MACD(daily) trend confirmation |
| BB(5min) = noise | BB(daily) = probability bounds |
| Volume(5min) spike | Volume(daily) = conviction indicator |

### 4.2 Konkrétní Vyuití pro B2

**Probability Adjustment:**
```python
# B2 currently uses: P(S_T >= K) = 1 - Φ(ln(K/S) / (σ√T))
# With TA: adjust model probability based on trend context

base_prob = estimate_daily_prob(price, strike, hours, sigma)

# RSI adjustment: overbought → less likely to go higher
if rsi_daily > 70:
    adjustment = -0.05 * (rsi_daily - 70) / 30  # -5% at RSI 100
elif rsi_daily < 30:
    adjustment = +0.05 * (30 - rsi_daily) / 30  # +5% at RSI 0
else:
    adjustment = 0

# Bollinger context: price above upper band → overbought territory
if bb_pctb_daily > 1.0:
    adjustment -= 0.03  # Above upper band, mean reversion likely

# MACD momentum: strong trend continuation
if macd_hist_daily > 0 and direction == "above":
    adjustment += 0.02  # Momentum confirms
elif macd_hist_daily < 0 and direction == "above":
    adjustment -= 0.02  # Momentum contradicts

adjusted_prob = max(0.01, min(0.99, base_prob + adjustment))
```

**Oekvan Impact pro B2**: +10-20% improvement (TA alignment mapuje na cenov predikci).

### 4.3 Volume CDF Model Enhancement

B2 pouívá volatility CDF model. TA me dodat:
- **ATR(14) daily** jako alternativní sigma estimate (cross-validace s realized vol)
- **Volume confirmation**: vysok volume na pohybu = udriteln, nzk = revert
- **Trend quality**: ADX > 25 = trend, ADX < 20 = range → adjust sigma_scale

---

## 5. Analza: TradingView pro Strategy D (Sports NBA)

### 5.1 Verdikt: IRELEVANTNÍ

Strategy D obchoduje NBA moneyline trhy na základ:
- Pinnacle odds (vig-free probability)
- Elo/Glicko-2 ratings (40% vha)
- Green book delta (price movement mid-game)

Finann technick analza (RSI, MACD, Bollinger) je zcela irelevantní pro sportovní výsledky.

**Vyjímka**: Pokud by D sledoval Polymarket market maker patterny (nap. jak MM repriceuje bhem zápasu), mohlo by volume/momentum analysis na samotném Polymarket orderbooku dávat smysl. Ale to nesouvisí s TradingView.

---

## 6. irší MCP Ekosystém — Relevantní Servery

Krom TradingView existuje bohat ekosystm MCP server pro finance/crypto:

### 6.1 Pmo Relevantní pro Arbo

| Server | Stars | Relevance | Use Case |
|--------|-------|-----------|----------|
| **polymarket-mcp-server** | 362 | VYSOK | 45 tool pro Polymarket (discovery, orderbook, orders). Studium implementace, NE jako nae produkn client. |
| **polymarket-paper-trader** | 183 | STEDN | Level-by-level orderbook simulace. Inspirace pro B3 backtest accuracy. |
| **crypto-indicators-mcp** | 122 | VYSOK | 50+ TA indiktr s CCXT (Binance). Alternativa k tradingview-ta pro B3. |
| **CCXT MCP** | 79 | STEDN | OHLCV z 10+ burz. Backup data source. |
| **crypto-sentiment-mcp** | 47 | NZK | Santiment sentiment. Relevantní pro budouc B1 (attention). |
| **TradeMemory Protocol** | 538 | NZK | Decision audit trail. Zajímav pro watchdog, ale mme vlastn PostgreSQL. |

### 6.2 Pro Research s Claude Code

| Server | Use Case |
|--------|----------|
| **tradingview-mcp** (atilaahmettaner) | Ad-hoc BTC analza, multi-timeframe, backtesting ideas |
| **polymarket-mcp-server** | Przkum novch trh, volume analza, market discovery |
| **prediction-market-mcp** | Cross-platform porovnání (Polymarket vs Kalshi vs PredictIt) |

---

## 7. Doporuen Architektura

### 7.1 Dva Reim  MCP vs Library

```
┌─────────────────────────────────────────────────────────┐
│                                                          │
│   RESEARCH REIM (Claude Code + MCP)                     │
│   ─────────────────────────────────                      │
│   Kdy: Interaktivní práce s Claude Code                  │
│   Co:  tradingview-mcp server                            │
│   Pro: Ad-hoc BTC analýza, regime check, strategy ideas  │
│   Latence: 2-5s per call (OK pro výzkum)                 │
│                                                          │
│   Instalace:                                             │
│   pip install tradingview-mcp-server                     │
│   + claude_desktop_config.json entry                     │
│                                                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   PRODUKČNÍ REIM (Python library, background)           │
│   ─────────────────────────────────────                  │
│   Kdy: Automatický background task na VPS                │
│   Co:  tradingview_ta Python knihovna                     │
│   Pro: Feature cache pro B3/B2 entry decisions           │
│   Latence: 0ms (čte z in-memory cache)                   │
│   Update: Každých 60s (matches B3's per-minute logic)    │
│                                                          │
│   Instalace:                                             │
│   pip install tradingview_ta                             │
│   Žádné API klíče, žádné předplatné                      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 7.2 Produkční Feature Pipeline

```python
# arbo/models/ta_feature_provider.py (NOVÝ SOUBOR)

import asyncio
from tradingview_ta import TA_Handler, Interval

class TAFeatureProvider:
    """Background TA feature cache for B3/B2 strategies.

    Updates every 60s via asyncio task. Strategies read from cache
    with zero latency. If tradingview-ta endpoint fails, features
    gracefully degrade to None (strategies continue without TA).
    """

    def __init__(self):
        self._cache: dict[str, TAFeatures] = {}
        self._update_interval = 60  # seconds
        self._running = False

    async def start(self):
        """Start background update loop."""
        self._running = True
        while self._running:
            try:
                await self._update_all()
            except Exception as e:
                logger.warning("ta_update_error", error=str(e))
            await asyncio.sleep(self._update_interval)

    async def _update_all(self):
        """Fetch TA for all tracked symbols/timeframes."""
        # Run in executor (tradingview-ta is sync, blocking)
        loop = asyncio.get_event_loop()
        features = await loop.run_in_executor(None, self._fetch_btc_ta)
        if features:
            self._cache["BTCUSDT"] = features

    def _fetch_btc_ta(self) -> TAFeatures | None:
        """Fetch BTC technical analysis across 3 timeframes."""
        try:
            # 5-minute (primary for B3)
            h5 = TA_Handler(symbol="BTCUSDT", screener="crypto",
                           exchange="BINANCE", interval=Interval.INTERVAL_5_MINUTES)
            a5 = h5.get_analysis()

            # 1-hour (context)
            h1h = TA_Handler(symbol="BTCUSDT", screener="crypto",
                            exchange="BINANCE", interval=Interval.INTERVAL_1_HOUR)
            a1h = h1h.get_analysis()

            # 4-hour (macro)
            h4h = TA_Handler(symbol="BTCUSDT", screener="crypto",
                            exchange="BINANCE", interval=Interval.INTERVAL_4_HOURS)
            a4h = h4h.get_analysis()

            return TAFeatures(
                timestamp=time.time(),
                rsi_5m=a5.indicators["RSI"],
                adx_5m=a5.indicators["ADX"],
                macd_hist_5m=a5.indicators["MACD.macd"] - a5.indicators["MACD.signal"],
                bb_upper=a5.indicators["BB.upper"],
                bb_lower=a5.indicators["BB.lower"],
                bb_middle=a5.indicators["BB.middle"],
                recommend_5m=a5.summary["RECOMMENDATION"],
                rsi_1h=a1h.indicators["RSI"],
                adx_1h=a1h.indicators["ADX"],
                recommend_1h=a1h.summary["RECOMMENDATION"],
                rsi_4h=a4h.indicators["RSI"],
                adx_4h=a4h.indicators["ADX"],
                recommend_4h=a4h.summary["RECOMMENDATION"],
            )
        except Exception:
            return None

    def get(self, symbol: str = "BTCUSDT") -> TAFeatures | None:
        """Get cached TA features (zero latency)."""
        features = self._cache.get(symbol)
        if features and not features.is_stale:
            return features
        return None
```

### 7.3 Fallback Strategie

tradingview-ta scrapuje neoficiální TradingView endpointy. Pokud se zlomí:

**Fallback 1**: `ccxt` + `pandas-ta`
```python
# CCXT pro OHLCV data z Binance (oficiální API)
import ccxt
import pandas_ta as ta

exchange = ccxt.binance()
ohlcv = exchange.fetch_ohlcv("BTC/USDT", "5m", limit=100)
df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
df["rsi"] = ta.rsi(df["c"], length=14)
df["adx"] = ta.adx(df["h"], df["l"], df["c"], length=14)["ADX_14"]
```

**Fallback 2**: Binance API přímo (u máme připojení)
```python
# Rozšířit existující binance_ws.py o 1-min OHLC buffer
# Počítat TA indikátory z vlastních dat
# Nulová závislost na třetích stranách
```

---

## 8. Implementaní Roadmap

### Phase 0: MCP pro Research (okamit, 0 riziko)

**Co**: Nainstalovat tradingview-mcp server pro Claude Code sessions.

**Proč**: Zero risk, okamitá hodnota pro ad-hoc BTC analýzu a strategy research.

```bash
pip install tradingview-mcp-server
```

Claude Code config (`~/.claude/settings.json` nebo MCP config):
```json
{
  "mcpServers": {
    "tradingview": {
      "command": "uvx",
      "args": ["--from", "tradingview-mcp-server", "tradingview-mcp"]
    }
  }
}
```

**Risk**: Zero (nepropojeno s produkcí).
**Effort**: 5 minut.

### Phase 1: Background TA Cache (po 50+ V6.0 live trades)

**Co**: Implementovat `TAFeatureProvider` jako background asyncio task.

**Proč**: Sbírat TA features (RSI, ADX, MACD, BB) ke kadému B3 tradu do trade_details JSONB. Jet ne filtrovat  jen logovat pro budoucí analzu.

**Implementace**:
1. Nový soubor `arbo/models/ta_feature_provider.py`
2. Start v `main_rdh.py` jako `asyncio.create_task(ta_provider.start())`
3. V `strategy_b3.py`: číst `ta_provider.get("BTCUSDT")` a logovat do trade_details
4. Nová pole v trade_details: `ta_rsi_5m`, `ta_adx_5m`, `ta_macd_hist`, `ta_bb_width`, `ta_recommend`, `ta_multi_tf_aligned`

**Risk**: Nízký (jen logging, nezasahuje do trade logic).
**Effort**: 2-4 hodiny kódu.
**Dependency**: `pip install tradingview_ta` na VPS.

### Phase 2: TA-Based Regime Filter (po 100+ trades s TA daty)

**Co**: Analyzovat korelaci TA features s W/L outcomes. Implementovat filtr pokud data podporují.

**Proč**: Data-driven decision, ne hypotéza.

**Analýza**:
```sql
-- Korelace TA features s výsledkem
SELECT
    CASE WHEN (trade_details->>'ta_adx_5m')::float > 30 THEN 'trending'
         WHEN (trade_details->>'ta_adx_5m')::float < 15 THEN 'calm'
         ELSE 'normal' END as regime,
    COUNT(*) as trades,
    AVG(CASE WHEN actual_pnl > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
    AVG(actual_pnl::float) as avg_pnl
FROM paper_trades
WHERE strategy = 'B3'
  AND trade_details->>'ta_adx_5m' IS NOT NULL
GROUP BY 1;
```

**Implementovat filtry POUZE pokud**:
- Cohen's d > 0.3 (alespo weak effect)
- N > 30 per bucket (statisticky smysluplné)
- Out-of-sample validace na nových datech

**Risk**: Stední (modifikuje trade logic, ale data-driven).
**Effort**: 1-2 dny analza + implementace.

### Phase 3: B2 TA Integration (a B2 jde do paper tradingu)

**Co**: Přidat TA-adjusted probability do B2 modelu.

**Proč**: B2 obchoduje denní horizonty, kde TA má přímou prediktivní hodnotu.

**Implementace**: Modifikovat `strategy_b2.py` — `adjusted_prob = base_prob + ta_adjustment`

**Risk**: Nízký (B2 teprve jde do paper tradingu).
**Effort**: 4-8 hodin.

### Phase 4: Watchdog Enrichment (až je Watchdog implementován)

**Co**: Feed TA kontext do Gemini Flash analzy v B3 Watchdog.

**Proč**: Bohatší kontext = lepší anomly detection.

**Implementace**: Přidat TA features do watchdog metrics a Gemini prompt.

**Risk**: Nulový (Watchdog je advisory-only).
**Effort**: 2-4 hodiny.

---

## 9. Risk Assessment

### 9.1 Overfitting (NEJVYŠŠÍ RIZIKO)

**Problém**: Víc indikátorů → víc parametrů → víc overfitting.

**Mitigace**:
- Phase 1 je POUZE logging (žádné filtrování)
- Phase 2 vyžaduje N>30 per bucket + OOS validaci
- TA indikátory pouívat pro FILTROVÁNÍ (reject bad trades), ne pro GENEROVÁNÍ signálů
- Core B3 model (Normal CDF + oracle lag) zůstává nezměněn
- Maximum 2-3 TA features v produkčním filtru (ne 20)

**Historická lekce**: V5.0 scoring model (10 features, 37 trades) = OVERFIT za hodiny. Zjednodušen na V6.0 (2 pravidla). TA musí následovat stejný pattern: jednoduchost > komplexita.

### 9.2 Korelace s Existujícími Features

**Problém**: TA features mohou být korelované s tím, co u měříme:
- `sigma_per_min` ≈ ATR (obojí volatilita)
- `velocity` ≈ momentum indikátory
- `spread` ≈ Bollinger width (obojí volatility proxy)

**Mitigace**:
- Před implementací filtru: měřit Pearson/Spearman korelaci
- Pokud korelace > 0.7: indikátor nepřidává novou informaci
- Zaměřit se na features s nízkou korelací: RSI (mean-reversion, nemáme), ADX (trend strength, nemáme), multi-TF alignment (nemáme)

### 9.3 Data Source Stabilita

**Problém**: tradingview-ta scrapuje neoficiální endpoint. TradingView může:
- Změnit API formát
- Blokovat IP
- Přidat CAPTCHA

**Mitigace**:
- Fallback: ccxt + pandas-ta (officiální Binance API)
- TA features jsou OPTIONAL (strategies pokračují bez nich)
- Graceful degradation: `if ta is None: continue_without_ta()`
- Rate limiting: max 3 calls/min (1 per timeframe)

### 9.4 Latence

**Problém**: tradingview-ta HTTP call = 200ms-2s.

**Mitigace**:
- NIKDY v hot path (background task)
- 60s update interval (odpovídá B3's per-minute logic)
- In-memory cache čtení = 0ms
- Pokud update selže, cache zůstává platný 90s (is_stale check)

### 9.5 False Confidence

**Problém**: TA summary "STRONG BUY" může vytvořit falešný pocit jistoty.

**Mitigace**:
- Pouívat JEDNOTLIVÉ indikátory (RSI value, ADX value), ne summary
- B3 edge je matematický (Normal CDF), ne sentiment-based
- TA je filtr (reject) a sizing modifier, ne signal generátor

---

## 10. Srovnán Pstup

| Pístup | Pros | Cons | Verdikt |
|---------|------|------|---------|
| **A: MCP only (research)** | Zero risk, instant value | Ne automatick | Phase 0 |
| **B: Library background + data collection** | Data-driven, low risk | 2-4h effort, needs validation | Phase 1 (DOPORUEN) |
| **C: Library + active filtering** | +15-25% PnL potential | Overfitting risk, needs 100+ trades | Phase 2 (PODMÍNĚNÉ daty) |
| **D: Full MCP in production** | Max integration | Latence, complexity, fragile | NEDOPORUČENO |
| **E: Custom TA from Binance data** | Zero external dependency | More code, same result | FALLBACK only |

**Doporuen cesta: A → B → C (podmíněné daty)**

---

## 11. Praktick MCP Commands pro Research

Po instalaci tradingview-mcp v Claude Code:

```
# Rychlý BTC overview
"Analyze BTCUSDT on 5-minute timeframe — give me RSI, MACD, ADX, Bollinger"

# Multi-timeframe alignment (klíčové pro B3)
"Compare BTCUSDT across 5m, 1h, and 4h — are all timeframes aligned?"

# Regime detection
"Is BTCUSDT in a trending or ranging market right now? Use ADX and BB width."

# Pre-deployment check
"Before I deploy B3 changes, analyze current BTC market conditions across
 all timeframes. Is this a good environment for momentum scalping?"

# Backtest idea
"Backtest RSI strategy on BTC-USD with 2-year daily data. What's the Sharpe?"

# Candlestick pattern check
"Check BTCUSDT 5-minute chart for candlestick patterns in the last 20 bars."
```

---

## 12. Závr

### Co Dlat Okamit (Phase 0)
1. Nainstalovat `tradingview-mcp-server` pro Claude Code research
2. Pouívat pro ad-hoc BTC analzu a strategy ideation
3. Zero risk, zero effort, okamit hodnota

### Co Dlat Po 50+ V6.0 Trades (Phase 1)
1. Implementovat `TAFeatureProvider` (background asyncio task)
2. Logovat TA features do trade_details JSONB
3. Nefiltrovat  jen sbírat data

### Co Dlat Po 100+ Trades s TA Daty (Phase 2)
1. Analyzovat korelaci TA  W/L outcomes
2. Implementovat filtr POUZE pokud data jasn ukazuj edge
3. ADX regime filter (nejpravděpodobněji první)
4. RSI mean-reversion filter (druhý)

### Co NEDLAT
- Nepouvat MCP v live trading loop (latence)
- Nepidvat >3 TA features najednou (overfitting)
- Nenahrazovat core model (Normal CDF je základ edge)
- Neimplementovat bez dat (Phase 2 vyžaduje 100+ trades)
- Nedvovat "STRONG BUY" summary (pouívat raw values)

### Odhadovan Net Impact
- **B3**: +15-25% PnL (~+$20-30/msíc pi $300 kapitálu)
- **B2**: +10-20% PnL (denní TA přímo relevantní)
- **D**: 0% (irelevantní)
- **Research productivity**: +30-50% (rychlejší ad-hoc analza s MCP)
