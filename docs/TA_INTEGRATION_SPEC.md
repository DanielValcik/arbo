# TA Integration Specification — Technical Analysis pro Arbo

> **Status**: SPEC COMPLETE — ceka na implementaci Phase 0 (MCP) a Phase 1 (TAFeatureProvider)
> **Datum**: 2026-04-11
> **Related**: `B3_WATCHDOG_SPEC.md`, `TRADINGVIEW_MCP_ANALYSIS.md`, `TECHNICAL_DECISIONS.md`

---

## 1. Executive Summary

- TradingView TA indikatory (RSI, ADX, MACD, Bollinger Bands) integrovane jako background feature cache
- Primarni pouziti: B3 (5-min BTC scalper) regime detection + autonomni Watchdog enrichment
- Sekundarni: B2 (crypto daily) probability adjustment
- Architektura: `tradingview_ta` Python knihovna (ne MCP server) v asyncio background task
- Cost: $0 (zadne API klice), ~100MB RAM, 60 HTTP calls/hr
- Predpoklad: VPS arbo-dublin (2 vCPU, 4GB RAM, MemoryMax=3.5G)

---

## 2. Architektura

### 2.1 Dva Rezimy Pouziti

**Research Rezim (Claude Code + MCP)**
- `tradingview-mcp-server` (pip install) pro interaktivni analyzu
- Ad-hoc BTC conditions check, multi-timeframe analysis, strategy ideation
- Zero connection to production
- Install: `pip install tradingview-mcp-server` + Claude Code MCP config

**Produkcni Rezim (Python library + background cache)**
- `tradingview_ta` knihovna (pip install tradingview_ta)
- Background asyncio task na VPS, update kazdych 60s
- In-memory cache, strategies ctou s 0ms latenci
- Graceful degradation: pokud tradingview-ta selze, strategies pokracuji bez TA

### 2.2 Produkcni Pipeline

```
+----------------------------------------------------------+
|                    VPS arbo-dublin                        |
|                                                          |
|  +----------------------+                                |
|  | TAFeatureProvider     |  <- asyncio.create_task()     |
|  |                      |  <- update kazdych 60s         |
|  | tradingview_ta:      |  <- run_in_executor() (sync->async) |
|  |   BTC 5m: RSI,ADX,  |                                |
|  |     MACD,BB,ATR      |                                |
|  |   BTC 1h: RSI,ADX   |-->  self._cache (in-memory dict)|
|  |   BTC 4h: RSI,ADX   |      |                         |
|  |   BTC 1d: RSI,MACD  |      |                         |
|  +----------------------+      |                         |
|                                |                         |
|                    +-----------+----------+               |
|                    |                      |               |
|              +-----v------+    +---------v--------+      |
|              |strategy_b3 |    | B3 Watchdog       |     |
|              |            |    |                   |     |
|              | poll_cycle: |    | Metrics Engine:   |     |
|              |  ta=cache.  |    |  TA regime bucket |     |
|              |   get()    |    |  TA PSI drift     |     |
|              |  log to    |    |  TA in Gemini ctx |     |
|              |  trade_    |    |                   |     |
|              |  details   |    | Autonomni:        |     |
|              +------------+    |  TA filter on/off |     |
|                                +-------------------+     |
|                                                          |
|              +------------+                              |
|              |strategy_b2 | (budouci)                    |
|              |            |                              |
|              | daily TA:  |                              |
|              |  prob adj  |                              |
|              +------------+                              |
+----------------------------------------------------------+
```

---

## 3. TAFeatures Schema

```python
@dataclass
class TAFeatures:
    """Cached technical analysis features, updated every 60s."""
    timestamp: float

    # 5-minute timeframe (primary for B3)
    rsi_5m: float | None         # RSI(14) on 5min candles
    adx_5m: float | None         # ADX(14) -- trend strength
    macd_hist_5m: float | None   # MACD histogram (macd - signal)
    bb_width_5m: float | None    # (upper - lower) / middle
    bb_pctb_5m: float | None     # %B position within bands (0-1)
    atr_5m: float | None         # ATR(14) -- volatility proxy
    recommend_5m: str | None     # "STRONG_BUY"|"BUY"|"NEUTRAL"|"SELL"|"STRONG_SELL"

    # 1-hour timeframe (context)
    rsi_1h: float | None
    adx_1h: float | None
    recommend_1h: str | None

    # 4-hour timeframe (macro)
    rsi_4h: float | None
    adx_4h: float | None
    recommend_4h: str | None

    # Daily timeframe (for B2 integration)
    rsi_1d: float | None
    macd_hist_1d: float | None
    bb_pctb_1d: float | None
    recommend_1d: str | None

    # Derived features
    @property
    def multi_tf_aligned(self) -> bool:
        """True if 5m, 1h, 4h all agree on direction."""
        recs = [self.recommend_5m, self.recommend_1h, self.recommend_4h]
        if any(r is None for r in recs):
            return False
        buy_signals = {"STRONG_BUY", "BUY"}
        sell_signals = {"STRONG_SELL", "SELL"}
        all_buy = all(r in buy_signals for r in recs)
        all_sell = all(r in sell_signals for r in recs)
        return all_buy or all_sell

    @property
    def adx_regime(self) -> str:
        """ADX-based regime classification."""
        if self.adx_5m is None:
            return "UNKNOWN"
        if self.adx_5m < 15:
            return "RANGING"
        elif self.adx_5m < 25:
            return "WEAK_TREND"
        else:
            return "STRONG_TREND"

    @property
    def rsi_zone(self) -> str:
        """RSI zone for mean-reversion risk."""
        if self.rsi_5m is None:
            return "UNKNOWN"
        if self.rsi_5m > 70:
            return "OVERBOUGHT"
        elif self.rsi_5m < 30:
            return "OVERSOLD"
        else:
            return "NEUTRAL"

    @property
    def is_stale(self) -> bool:
        """Cache older than 90s is stale."""
        return time.time() - self.timestamp > 90
```

---

## 4. TAFeatureProvider Implementace

```python
# arbo/models/ta_feature_provider.py

class TAFeatureProvider:
    """Background TA feature cache.

    Strategies ctou z cache s 0ms latenci.
    tradingview-ta je sync (blocking I/O) -> run_in_executor.
    Pokud update selze, cache zustava platny 90s.
    """

    def __init__(self):
        self._cache: dict[str, TAFeatures] = {}
        self._update_interval = 60  # seconds
        self._running = False
        self._consecutive_failures = 0
        self._max_failures = 10  # After 10 failures -> stop trying

    async def start(self):
        """Background update loop. Called as asyncio.create_task()."""
        self._running = True
        while self._running:
            try:
                loop = asyncio.get_event_loop()
                features = await loop.run_in_executor(None, self._fetch_btc_ta)
                if features:
                    self._cache["BTCUSDT"] = features
                    self._consecutive_failures = 0
            except Exception as e:
                self._consecutive_failures += 1
                logger.warning("ta_update_error",
                             error=str(e),
                             failures=self._consecutive_failures)
                if self._consecutive_failures >= self._max_failures:
                    logger.error("ta_provider_disabled",
                               msg="Too many failures, stopping TA updates")
                    break
            await asyncio.sleep(self._update_interval)

    def _fetch_btc_ta(self) -> TAFeatures | None:
        """Fetch BTC TA across timeframes (sync, runs in executor)."""
        from tradingview_ta import TA_Handler, Interval

        try:
            h5 = TA_Handler(symbol="BTCUSDT", screener="crypto",
                           exchange="BINANCE", interval=Interval.INTERVAL_5_MINUTES)
            a5 = h5.get_analysis()

            h1h = TA_Handler(symbol="BTCUSDT", screener="crypto",
                            exchange="BINANCE", interval=Interval.INTERVAL_1_HOUR)
            a1h = h1h.get_analysis()

            h4h = TA_Handler(symbol="BTCUSDT", screener="crypto",
                            exchange="BINANCE", interval=Interval.INTERVAL_4_HOURS)
            a4h = h4h.get_analysis()

            h1d = TA_Handler(symbol="BTCUSDT", screener="crypto",
                            exchange="BINANCE", interval=Interval.INTERVAL_1_DAY)
            a1d = h1d.get_analysis()

            bb_upper = a5.indicators.get("BB.upper", 0)
            bb_lower = a5.indicators.get("BB.lower", 0)
            bb_middle = a5.indicators.get("BB.middle", 1)
            bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle else 0

            macd = a5.indicators.get("MACD.macd", 0)
            signal = a5.indicators.get("MACD.signal", 0)

            return TAFeatures(
                timestamp=time.time(),
                rsi_5m=a5.indicators.get("RSI"),
                adx_5m=a5.indicators.get("ADX"),
                macd_hist_5m=(macd - signal) if macd and signal else None,
                bb_width_5m=bb_width,
                bb_pctb_5m=None,  # Computed from bands
                atr_5m=a5.indicators.get("ATR"),
                recommend_5m=a5.summary.get("RECOMMENDATION"),
                rsi_1h=a1h.indicators.get("RSI"),
                adx_1h=a1h.indicators.get("ADX"),
                recommend_1h=a1h.summary.get("RECOMMENDATION"),
                rsi_4h=a4h.indicators.get("RSI"),
                adx_4h=a4h.indicators.get("ADX"),
                recommend_4h=a4h.summary.get("RECOMMENDATION"),
                rsi_1d=a1d.indicators.get("RSI"),
                macd_hist_1d=...,
                bb_pctb_1d=...,
                recommend_1d=a1d.summary.get("RECOMMENDATION"),
            )
        except Exception:
            return None

    def get(self, symbol: str = "BTCUSDT") -> TAFeatures | None:
        """Read from cache (0ms latency)."""
        features = self._cache.get(symbol)
        if features and not features.is_stale:
            return features
        return None

    async def stop(self):
        self._running = False
```

---

## 5. Integrace s B3

### 5.1 Trade Details Logging (Phase 1 — jen data collection)

V `strategy_b3.py` `poll_cycle()` — po vypoctu signalu, pred trade execution:

```python
ta = self._ta_provider.get("BTCUSDT") if self._ta_provider else None

# Log to trade_details JSONB (no filtering!)
if ta and not ta.is_stale:
    trade_details["ta_rsi_5m"] = round(ta.rsi_5m, 1) if ta.rsi_5m else None
    trade_details["ta_adx_5m"] = round(ta.adx_5m, 1) if ta.adx_5m else None
    trade_details["ta_macd_hist_5m"] = round(ta.macd_hist_5m, 4) if ta.macd_hist_5m else None
    trade_details["ta_bb_width_5m"] = round(ta.bb_width_5m, 4) if ta.bb_width_5m else None
    trade_details["ta_recommend_5m"] = ta.recommend_5m
    trade_details["ta_rsi_1h"] = round(ta.rsi_1h, 1) if ta.rsi_1h else None
    trade_details["ta_adx_1h"] = round(ta.adx_1h, 1) if ta.adx_1h else None
    trade_details["ta_multi_tf_aligned"] = ta.multi_tf_aligned
    trade_details["ta_adx_regime"] = ta.adx_regime
    trade_details["ta_rsi_zone"] = ta.rsi_zone
```

### 5.2 Autonomni Watchdog TA Filtering (Phase 2 — po 100+ tradech s TA daty)

Watchdog automaticky aktivuje/deaktivuje TA filtry pres adaptive_config:

```python
# Watchdog rozhodne na zaklade dat:
if adx_ranging_wr < 0.40 and adx_ranging_n > 30:
    adaptive_config.set("TA_ADX_MIN", 15, reason="RANGING WR too low")
    # -> strategy_b3 zacne skipovat trades kde ADX < 15

if rsi_extreme_momentum_conflict_wr < 0.45 and n > 20:
    adaptive_config.set("TA_RSI_FILTER", True, reason="RSI extremes hurt momentum")
    # -> strategy_b3 skipuje UP+RSI>80 nebo DOWN+RSI<20
```

---

## 6. Integrace s B2 (Budouci — Phase 3)

B2 obchoduje denni "above" trhy. TA na dennim timeframe je primo prediktivni:

```python
# V strategy_b2.py:
base_prob = estimate_daily_prob(price, strike, hours, sigma)

ta = self._ta_provider.get("BTCUSDT")
if ta:
    adjustment = 0.0
    # RSI: overbought -> less likely above
    if ta.rsi_1d and ta.rsi_1d > 70:
        adjustment -= 0.03 * (ta.rsi_1d - 70) / 30
    elif ta.rsi_1d and ta.rsi_1d < 30:
        adjustment += 0.03 * (30 - ta.rsi_1d) / 30
    # MACD: trend confirmation
    if ta.macd_hist_1d and ta.macd_hist_1d > 0:
        adjustment += 0.01
    elif ta.macd_hist_1d and ta.macd_hist_1d < 0:
        adjustment -= 0.01

    adjusted_prob = max(0.01, min(0.99, base_prob + adjustment))
```

---

## 7. Integrace s main_rdh.py

```python
# V RDHOrchestrator.__init__:
self._ta_provider: TAFeatureProvider | None = None

# V _init_components() (po Binance WS init):
try:
    from arbo.models.ta_feature_provider import TAFeatureProvider
    self._ta_provider = TAFeatureProvider()
    logger.info("ta_provider_initialized")
except ImportError:
    logger.warning("ta_provider_unavailable", msg="tradingview_ta not installed")

# V _start_internal_tasks():
if self._ta_provider:
    self._internal_tasks.append(
        asyncio.create_task(self._ta_provider.start(), name="ta_features")
    )

# V _init_strategy_b3() -- pass ta_provider:
s = StrategyB3(
    risk_manager=self._risk_manager,
    paper_engine=self._paper_engine,
    binance_ws=self._binance_ws,
    rtds_feed=rtds_feed,
    execution_mode=execution_mode,
    live_executor=live_executor,
    ta_provider=self._ta_provider,  # NEW
)

# V stop():
if self._ta_provider:
    await self._ta_provider.stop()
```

---

## 8. Fallback Strategie

### 8.1 Pokud tradingview-ta selze (endpoint change, IP block):

**Fallback 1**: `ccxt` + `pandas-ta`
```python
import ccxt
import pandas_ta as ta

exchange = ccxt.binance()
ohlcv = exchange.fetch_ohlcv("BTC/USDT", "5m", limit=100)
df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
rsi = ta.rsi(df["c"], length=14).iloc[-1]
adx_df = ta.adx(df["h"], df["l"], df["c"], length=14)
adx = adx_df["ADX_14"].iloc[-1]
```

**Fallback 2**: Binance API primo (jiz mame pripojeni)
- Rozsirit `binance_ws.py` o 1-min OHLC buffer
- Pocitat TA indikatory z vlastnich dat (zero external dependency)

### 8.2 Graceful Degradation

TAFeatureProvider vraci `None` pokud cache je stale (>90s). Strategies pokracuji bez TA:

```python
ta = self._ta_provider.get() if self._ta_provider else None
if ta is None:
    # Continue without TA -- all TA fields in trade_details = None
    # Watchdog TA regime analysis skipped
    pass
```

---

## 9. VPS Capacity Assessment

| Metrika | Aktualni | Po TA | Delta | Status |
|---------|---------|-------|-------|--------|
| Asyncio tasks | ~18 | ~20 | +2 | OK |
| HTTP calls/hr | ~600 | ~660 | +10% | OK |
| RAM peak | ~3.0G | ~3.1G | +100MB | OK (limit 3.5G) |
| CPU | ~15% avg | ~16% | +1% | OK |

**tradingview_ta blocking**: Knihovna je synchronni. Musi bezet v `loop.run_in_executor(None, ...)` aby neblokovala event loop. Kazdy call trva 200ms-2s -> v executoru bezpecne.

**Network**: 4 HTTP calls per 60s update (5m + 1h + 4h + 1d). TradingView endpoint muze throttlovat. Retry s exponential backoff, max 10 consecutive failures -> disable.

---

## 10. Risks a Mitigace

| Risk | Severity | Pravdepodobnost | Mitigace |
|------|----------|-----------------|----------|
| Overfitting (vic features = vic) | VYSOKA | Stredni | Logging-first, min 100 tradu, auto-revert |
| Korelace s existujicimi features | Stredni | Vysoka | Merit Pearson pred implementaci filtru |
| tradingview-ta endpoint break | Stredni | Nizka | Fallback: ccxt + pandas-ta |
| Latence v hot path | VYSOKA | Nulova | Background cache, 0ms read |
| False confidence z TA summary | Stredni | Stredni | Pouzivat raw values, ne summary |
| VPS memory overflow | Nizka | Nizka | 100MB headroom, graceful degradation |

---

## 11. Implementacni Roadmap

| Phase | Co | Kdy | Zavislosti |
|-------|-----|-----|------------|
| **0** | MCP pro Claude Code research | Okamzite | Zadne |
| **1** | TAFeatureProvider + B3 logging | Po schvaleni | pip install tradingview_ta |
| **2** | Watchdog TA regime + autonomni filtering | Po 100+ tradech s TA | Phase 1 + Watchdog Faze 2 |
| **3** | B2 TA probability adjustment | Az B2 jde do paper | Phase 1 + B2 autoresearch |

---

## 12. Relevance pro Jednotlive Strategie

| Strategie | TA Relevance | Duvod |
|-----------|-------------|-------|
| **B3 (5-min BTC)** | VYSOKA | Regime detection, mean-reversion risk, MTF alignment |
| **B2 (crypto daily)** | VYSOKA | Denni RSI/MACD primo mapuji na cenovou predikci |
| **B3 15-min** | VYSOKA | Stejne jako B3, vetsi timeframe = TA relevantnejsi |
| **D (sports NBA)** | NULOVA | Financni TA irelevantni pro sportovni vysledky |
| **C/C2 (weather)** | NULOVA | Pocasi nekoreluje s financnimi indikatory |
| **A (theta decay)** | NIZKA | Longshot selling nezavisi na BTC TA |

---

## 13. Prenositelnost na Dalsi Modely

TAFeatureProvider je navrzen jako **genericky system**:

1. **Symbol-agnosticky**: `provider.get("ETHUSDT")`, `provider.get("SOLUSDT")`
2. **Timeframe-konfigurovatelny**: Pridani noveho TF = 1 radek kodu
3. **Strategy-nezavisly**: Cache je sdilena, kazda strategie cte co potrebuje
4. **Watchdog-kompatibilni**: TA features automaticky v Watchdog context

Pro pridani nove strategie:
```python
# 1. Strategy cte z provideru
ta = self._ta_provider.get("ETHUSDT")
# 2. Loguje do trade_details
trade_details["ta_rsi_5m"] = ta.rsi_5m
# 3. Watchdog automaticky analyzuje TA buckety pro novou strategii
```

---

> **Status**: SPEC COMPLETE — ceka na implementaci Phase 0 (MCP) a Phase 1 (TAFeatureProvider)
> **Related**: `B3_WATCHDOG_SPEC.md`, `TRADINGVIEW_MCP_ANALYSIS.md`, `TECHNICAL_DECISIONS.md`
