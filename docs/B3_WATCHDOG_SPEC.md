# B3 Watchdog Service — Design Specification

> **Status**: RESEARCH COMPLETE, READY FOR IMPLEMENTATION
> **Date**: 2026-04-10
> **Strategy**: B3 (Binance Oracle Scalper, 5-min BTC Up/Down)
> **Author**: CTO / CEO review pending

---

## 1. Executive Summary

B3 Watchdog je lokální Python daemon na VPS (arbo-dublin), který kontinuálně
vyhodnocuje výkon B3 strategie, detekuje anomálie v tržních podmínkách a
navrhuje optimalizace s důkazy. Operuje v režimu **plně autonomní optimizer** — sám detekuje anomálie, analyzuje root cause přes Gemini Flash, rozhoduje o parametrických změnách v rámci bezpečnostních bounds a implementuje je za běhu. CEO dostává do Slack report co se stalo, proč a jaký je expected impact. Bezpečnost zajišťuje 3-tier model autonomie s auto-revert mechanismem.

### Proč právě B3

B3 je ideální kandidát pro watchdog z těchto důvodů:

1. **Vysoká frekvence dat**: ~33 paper tradů/den, 278+ live tradů za 10 dní
2. **Bohatá telemetrie**: 35+ polí v `trade_details` per trade (velocity, dir_delta,
   z_score, sigma_norm, spread, combined_risk, cl_ratio, fill_to_model...)
3. **Citlivost na režimy**: Regime research prokázal 4 σ-buckety s dramaticky
   odlišným WR (CALM 29% → VOLATILE 95%)
4. **Rychlý feedback loop**: Trades se resolvují za 1-5 minut (vs dny/týdny u C/D)
5. **Známé failure modes**: Spread widening, CL-Binance delta drift, velocity spikes

### Proč NE Managed Agents

| Kritérium | Managed Agents | Lokální Watchdog |
|-----------|---------------|-----------------|
| Náklady | ~$80-100/měsíc (runtime + Claude tokens) | ~$1/měsíc (Gemini Flash při anomáliích) |
| Přístup k datům | API fetch přes internet | Přímý PostgreSQL + in-memory |
| Latence | Sekundy (network) | Milisekundy (localhost) |
| LLM flexibilita | Claude only | Gemini Flash ($0.10/MTok) + fallback |
| Vendor lock-in | Anthropic runtime | Žádný |
| Kontrola | Managed sandbox | Plná kontrola, git-versioned |

---

## 2. Architektura

```
┌────────────────────────────────────────────────────────────────┐
│                     VPS arbo-dublin                             │
│                                                                │
│  ┌────────────┐          ┌──────────────┐                      │
│  │ strategy_b3│─trades──▶│  PostgreSQL   │                     │
│  │ (live loop)│          │ paper_trades  │                     │
│  └────────────┘          │ trade_details │                     │
│                          │ daily_pnl     │                     │
│                          │ paper_snapshots│                    │
│                          └──────┬────────┘                     │
│                                 │                              │
│                    ┌────────────▼────────────┐                 │
│                    │    B3 Watchdog Daemon    │                 │
│                    │                         │                 │
│                    │  ┌───────────────────┐  │                 │
│                    │  │ 1. Data Collector  │  │ ← Každých 6h   │
│                    │  │    SQL queries     │  │   NEBO po 50    │
│                    │  └────────┬──────────┘  │   nových B3     │
│                    │           │              │   tradech        │
│                    │  ┌────────▼──────────┐  │                 │
│                    │  │ 2. Metrics Engine  │  │                 │
│                    │  │  - Rolling WR/PnL  │  │                 │
│                    │  │  - Regime breakdown│  │                 │
│                    │  │  - Calibration ECE │  │                 │
│                    │  │  - PSI drift       │  │                 │
│                    │  │  - CUSUM/BOCPD     │  │                 │
│                    │  └────────┬──────────┘  │                 │
│                    │           │              │                 │
│                    │  ┌────────▼──────────┐  │                 │
│                    │  │ 3. Anomaly Router  │  │                 │
│                    │  │  - Threshold check │  │                 │
│                    │  │  - Severity rating │  │                 │
│                    │  │  - Dedup/suppress  │  │                 │
│                    │  └────────┬──────────┘  │                 │
│                    │           │              │                 │
│                    │     anomálie?            │                 │
│                    │     ┌─────┴─────┐       │                 │
│                    │     NE         ANO      │                 │
│                    │     │           │        │                 │
│                    │   (spí)    ┌────▼─────┐  │                 │
│                    │           │ 4. Gemini │  │                 │
│                    │           │   Flash   │  │                 │
│                    │           │  Analyst  │  │                 │
│                    │           └────┬─────┘  │                 │
│                    │                │         │                 │
│                    │   ┌────────────▼──────┐  │                 │
│                    │   │ 5. Slack Reporter │  │                 │
│                    │   │  #b3-watchdog     │  │                 │
│                    │   └──────────────────┘  │                 │
│                    └────────────────────────┘                  │
│                                                                │
│                    CEO: ✅ approve / ❌ reject                  │
└────────────────────────────────────────────────────────────────┘
```

### Autonomní Decision Engine

```
Watchdog Decision Flow:
  
  Data Collection (SQL)
      │
  Metrics Engine (WR, PnL, PSI, CUSUM, BOCPD, TA)
      │
  Anomaly Detection
      │
  ┌───┴───────────────────────┐
  │ Tier Check                │
  │                           │
  │ Tier 1 (autonomous):     │
  │   APPLY → Slack report   │
  │                           │
  │ Tier 2 (autonomous+flag):│
  │   APPLY → Slack + FLAG   │
  │                           │
  │ Tier 3 (escalate only):  │
  │   Slack CRITICAL → CEO   │
  └───────────────────────────┘
      │
  Auto-Revert Check (50 trades)
      │
  Self-Learning Log (DB)
```

#### 3-Tier Bezpečnostní Model

**Tier 1: PLNĚ AUTONOMNÍ** (Watchdog mění okamžitě, reportuje do Slack)

| Parametr | Default | Autonomní Rozsah | Hard Bound |
|----------|---------|-----------------|------------|
| `LIVE_MAX_VELOCITY` | 60 $/min | 40 – 80 | [30, 100] |
| `LIVE_MAX_DIR_DELTA` | $15 | $10 – $25 | [5, 40] |
| `LIVE_MIN_EDGE` | 0.40 | 0.30 – 0.60 | [0.20, 0.80] |
| `POSITION_PCT` | 0.026 | 0.013 – 0.052 | ±50% default |
| `EDGE_SCALING` | 10.0 | 5.0 – 20.0 | [2, 25] |
| Trading pause/resume | running | pause při DD > threshold | Auto-resume po recovery |
| TA-based filters (RSI, ADX) | off | enable/disable | Min 50 trades validace |

**Tier 2: AUTONOMNÍ S ESKALACÍ** (Watchdog mění, flaguje pro CEO review)

| Parametr | Hard Bound | Eskalace Podmínka |
|----------|------------|-------------------|
| `SIGMA_SCALE` | [0.25, 0.50] | Změna > ±15% od default |
| `ENTRY_THRESHOLD` | [0.01, 0.05] | Core model parameter |
| Nový filtr (spread, BB width) | On/Off | První aktivace |
| `MIN_ENTRY_MIN` / `MAX_ENTRY_MIN` | [1, 4] | Timing change |

**Tier 3: NIKDY AUTONOMNÍ** (hardcoded, vyžaduje deploy)

| Parametr | Hodnota | Důvod |
|----------|---------|-------|
| `MAX_BET_SIZE` | $100 | Polymarket liquidity constraint |
| `MIN_ORDER_SIZE` | $5 | Polymarket minimum |
| `DAILY_LOSS_LIMIT` | $50 | Risk management floor |
| `LIVE_MAX_FILL_PRICE` | 0.75 | Data-driven cap (278 tradů) |
| `REQUIRE_CHAINLINK` | True | Oracle trust |
| Execution mode | dual | Deployment decision |
| Capital allocation | $1000 | CEO decision |

#### Runtime Adaptive Config

```python
# arbo/core/adaptive_config.py — Runtime parametrický systém
#
# Pattern:
# 1. Default hodnoty z b3_quality_gate.py (read-only)
# 2. Watchdog overrides v in-memory dict (adaptive_overrides)
# 3. strategy_b3.py čte: adaptive_config.get("LIVE_MAX_VELOCITY", default=60)
# 4. Každá změna → audit_log tabulka (timestamp, param, old, new, reason, trigger_id)
# 5. Auto-revert: pokud WR klesne o >5pp za 50 tradů po změně → revert
#
# Watchdog NIKDY nemění b3_quality_gate.py soubor — jen runtime dict.
# Restart systému → parametry se vrátí na default (safe fallback).
```

#### Auto-Revert Mechanismus

Po každé autonomní změně Watchdog:
1. Zaznamená `change_id`, `param`, `old_value`, `new_value`, `trade_count_at_change`
2. Po dalších 50 tradech vyhodnotí:
   - WR po změně vs WR před změnou (sliding window)
   - Pokud WR kleslo o > 5pp → **automatický revert** na `old_value`
   - Pokud WR stabilní nebo lepší → změna se stane novým default
3. Revert se reportuje do Slack: "Auto-reverted PARAM from X to Y — WR dropped from A% to B%"

#### Self-Optimization Cíle

Watchdog kontinuálně optimalizuje pro 4 metriky:
1. **Win Rate** — minimalizovat loss rate filtrováním špatných podmínek
2. **Drawdown** — pausovat/snížit sizing při nepříznivém regime
3. **Obratka kapitálu** — maximalizovat počet kvalitních tradů (ne filtrovat příliš agresivně)
4. **Edge stabilita** — detekovat alpha decay a adaptovat parametry dřív než CEO

### Integrace s existující infrastrukturou

Watchdog se integruje do stávajícího systému, nenahrazuje ho:

| Existující komponenta | Jak ji Watchdog využívá |
|----------------------|------------------------|
| `paper_trades` tabulka (JSONB `trade_details`) | Primární datový zdroj — queryuje per-trade metriky |
| `daily_pnl` tabulka | Denní agregáty pro trend analýzu |
| `paper_snapshots` tabulka | Equity curve pro drawdown výpočty |
| `health_check.py` (12h cycle) | Watchdog rozšíří, ne nahradí. Přidá B3-specifické kontroly |
| `risk_manager.py` (kill switches) | Watchdog NEKONTROLUJE risk. Pouze reportuje anomálie |
| `slack_bot.py` (Socket Mode) | Watchdog posílá do nového kanálu `#b3-watchdog` |
| `main_rdh.py` orchestrátor | Watchdog poběží jako další `asyncio.create_task()` |
| ta_feature_provider.py | Background TA cache (RSI, ADX, MACD, BB) — Watchdog čte pro enriched regime |

### Data Flow

```
PostgreSQL (paper_trades WHERE strategy='B3')
    │
    ▼
SQL query: SELECT trade_details, actual_pnl, status, placed_at, resolved_at
    │
    ▼
Pandas DataFrame (posledních 200 tradů + all-time baseline)
    │
    ├──▶ Rolling Metrics (WR, PnL, Sharpe na window=50)
    ├──▶ Regime Breakdown (per sigma/velocity/spread bucket)
    ├──▶ Calibration (ECE na predicted edge vs actual outcome)
    ├──▶ Distribution Drift (PSI na velocity, dir_delta, spread, fill_price)
    ├──▶ Change Point Detection (CUSUM na cumulative PnL)
    └──▶ Bayesian Regime Detection (BOCPD na daily returns)
```

---

## 3. Metriky — Co Watchdog Sleduje

### 3.1 Outcome Metriky (rolling window = 50 tradů)

| Metrika | Výpočet | Baseline (V6.0) | Zdroj |
|---------|---------|-----------------|-------|
| Win Rate (paper) | wins / resolved | 68% (38t) | paper_trades WHERE status IN ('won','lost','sold') |
| Win Rate (live) | live wins / live resolved | 87% (15t) | paper_trades WHERE trade_details->>'live_fill_status'='filled' |
| Avg PnL per trade | mean(actual_pnl) | +$0.53 paper | paper_trades.actual_pnl |
| Rolling Sharpe | mean(pnl) / std(pnl) × √(trades/day) | TBD (need 50+ trades) | Computed |
| Max consecutive losses | longest losing streak | 2 (V6.0 live) | Sequential scan |
| Daily PnL | sum(actual_pnl) per day | +$5.4/active day | daily_pnl table |

### 3.2 Regime Metriky (per-bucket WR a PnL)

B3 `trade_details` JSONB obsahuje všechna potřebná pole. Watchdog je buckétuje:

**Sigma Norm** (`trade_details->>'sigma_norm'`):

| Bucket | σ_norm range | Baseline WR (paper) | Baseline WR (live) | N (live) |
|--------|-------------|---------------------|---------------------|----------|
| CALM | < 1.5 | 29% | 75%* | 4 |
| NORMAL | 1.5 – 2.0 | 56% | 69% | 18 |
| ACTIVE | 2.0 – 2.5 | 73% | 74% | 15 |
| VOLATILE | > 2.5 | 95% | 100%* | 5 |

*Nedostatečný vzorek — sledovat, nevyhodnocovat.

**Velocity** (`trade_details->>'velocity'` nebo `'velocity_paper'`):

| Bucket | Range ($/min) | Baseline WR (paper) |
|--------|-------------|---------------------|
| SLOW | < 30 | 78% |
| MEDIUM | 30 – 60 | ~65% |
| FAST | > 60 | 6% (filtrováno V6.0) |

**Orderbook Spread** (`trade_details->>'orderbook_spread'`):

| Bucket | Range ($) | Baseline WR (live) |
|--------|----------|---------------------|
| TIGHT | < 0.02 | 80% |
| NORMAL | 0.02 – 0.05 | 67% |
| WIDE | > 0.10 | 45% |

**Combined Risk** (`trade_details->>'combined_risk'`):

| Bucket | CR range | Baseline WR (live) |
|--------|---------|---------------------|
| LOW | < 1.0 | 100% (4t)* |
| MEDIUM | 1.0 – 1.5 | 80% (5t) |
| HIGH | > 1.5 | 50% (4t) |

**Direction** (`trade_details->>'direction'`):

| Direction | Baseline WR | Notes |
|-----------|------------|-------|
| DOWN | vyšší (v5 scoring) | Gravity/panic dynamics |
| UP | nižší | Optimism bias |

**Time of Day** (UTC hour z `placed_at`):

| Window | UTC hours | Baseline WR |
|--------|----------|-------------|
| Asia session | 00 – 08 | TBD |
| Europe session | 08 – 16 | TBD |
| US session | 16 – 24 | TBD |
| Peak hours | 02 – 03 UTC | 66-72% (backtest) |

**ADX Trend Strength** (`trade_details->>'ta_adx_5m'`):

| Bucket | ADX range | Expected WR | Mechanismus |
|--------|-----------|-------------|-------------|
| RANGING | < 15 | ~30-40% | Žádný trend, momentum nespolehlivý |
| WEAK_TREND | 15 – 25 | ~55-65% | Slabý trend, opatrně |
| STRONG_TREND | > 25 | ~70-80% | Silný trend, momentum funguje |

*Baseline bude stanoven po 100+ tradech s TA daty.*

**RSI Zone** (`trade_details->>'ta_rsi_5m'`):

| Bucket | RSI range | Risk pro momentum | Mechanismus |
|--------|-----------|-------------------|-------------|
| OVERSOLD | < 30 | UP risky (bounce) | Mean reversion nahoru |
| NEUTRAL | 30 – 70 | Baseline | Normální podmínky |
| OVERBOUGHT | > 70 | DOWN risky (pullback) | Mean reversion dolů |

*Pozor: RSI extremes mohou POMÁHAT momentum trades v opačném směru.*

**Multi-TF Alignment** (`trade_details->>'ta_multi_tf_aligned'`):

| Stav | Expected Impact | Mechanismus |
|------|-----------------|-------------|
| Aligned (5m + 1h + 4h souhlasí) | WR boost, sizing +30% | Všechny timeframes potvrzují |
| Divergent (mixed signals) | Baseline sizing | Konfliktní signály |
| Counter-aligned (majority against) | WR drop, sizing -30% | Většina TF proti B3 signálu |

*Baseline bude stanoven po 200+ tradech s TA daty.*

### 3.3 Market Microstructure Metriky

| Metrika | Výpočet | Zdroj |
|---------|---------|-------|
| Avg orderbook spread | mean(trade_details->>'orderbook_spread') | JSONB |
| CL-Binance delta trend | mean(trade_details->>'btc_binance_chainlink_delta') | JSONB |
| Avg fill price | mean(trade_details->>'live_entry_price') | JSONB |
| Avg available liquidity | mean(trade_details->>'liq_available_usd') | JSONB |
| Fill price vs model | mean(trade_details->>'fill_to_model') | JSONB |
| Avg entry latency | mean(trade_details->>'live_entry_latency_ms') | JSONB |

### 3.4 Edge Calibration (klíčová metrika)

Edge calibration odpovídá na zásadní otázku: **Koreluje model-predicted edge
s actual outcomes?**

```
Příklad:
- Model říká edge=0.40 (40pp od fair value) → expectation: ~87% WR
- Model říká edge=0.20 → expectation: ~72% WR
- Pokud edge=0.40 má skutečně jen 65% WR → model je mis-kalibrovaný
```

**Implementace — Expected Calibration Error (ECE):**

```python
def compute_b3_ece(trades_df, n_bins=5):
    """ECE pro B3: predicted edge → actual win rate.

    Nižší ECE = model lépe predikuje.
    ECE < 0.05: výborná kalibrace
    ECE 0.05-0.10: akceptovatelná
    ECE > 0.15: problém — model neodpovídá realitě
    """
    # Predicted "confidence" = edge (higher = model more confident)
    edges = trades_df["edge"].values
    outcomes = (trades_df["status"] == "won").astype(float).values

    # Bin by edge quantiles (not uniform — edge distribution is skewed)
    bin_boundaries = np.quantile(edges, np.linspace(0, 1, n_bins + 1))

    ece = 0.0
    n = len(edges)
    for i in range(n_bins):
        mask = (edges >= bin_boundaries[i]) & (edges < bin_boundaries[i + 1])
        count = mask.sum()
        if count < 3:  # Minimum per bin
            continue
        actual_wr = outcomes[mask].mean()
        avg_edge = edges[mask].mean()
        # Expected WR from edge (rough mapping)
        expected_wr = 0.50 + avg_edge  # simplified — edge is deviation from 0.50
        ece += (count / n) * abs(actual_wr - expected_wr)

    return ece
```

**Reliability Diagram (týdenní Slack snapshot):**

```
Edge Bin   | Predicted WR | Actual WR | N trades | Gap
0.02-0.10  |     55%      |    52%    |    45    |  -3pp ✅
0.10-0.20  |     65%      |    61%    |    38    |  -4pp ✅
0.20-0.40  |     75%      |    73%    |    25    |  -2pp ✅
0.40-0.60  |     90%      |    87%    |    15    |  -3pp ✅
0.60+      |     95%      |    82%    |     8    | -13pp ⚠️
```

---

## 4. Anomaly Detection — Statistické Metody

### 4.1 CUSUM (Cumulative Sum Control Chart)

CUSUM detekuje malé, perzistentní posuny v průměru — perfektní pro
postupnou degradaci edge, kterou single-trade metriky nevidí.

**Jak funguje:**
```
Akumuluje odchylky od expected value. Malý negativní drift se hromadí
přes desítky tradů → alarm, i když žádný jednotlivý trade není outlier.
```

**Implementace pro B3:**

```python
def cusum_b3(pnl_series, threshold_sigma=4.0, drift_fraction=0.5):
    """CUSUM na B3 cumulative PnL.

    Parametry:
        pnl_series: array per-trade PnL (chronologicky)
        threshold_sigma: alarm threshold (vyšší = méně false alarms)
        drift_fraction: slack parameter (0.5 × std je standard)

    Returns:
        alarm_indices: kde CUSUM detekoval shift
        gp, gn: upper/lower cumulative sums (pro vizualizaci)
    """
    mu = np.mean(pnl_series)
    sigma = np.std(pnl_series)
    threshold = threshold_sigma * sigma
    drift = drift_fraction * sigma

    gp = np.zeros(len(pnl_series))  # Cumulative sum pro + shift
    gn = np.zeros(len(pnl_series))  # Cumulative sum pro - shift
    alarms = []

    for i in range(1, len(pnl_series)):
        deviation = pnl_series[i] - mu
        gp[i] = max(0, gp[i-1] + deviation - drift)
        gn[i] = max(0, gn[i-1] - deviation - drift)

        if gp[i] > threshold or gn[i] > threshold:
            alarms.append(i)
            gp[i] = 0  # Reset po alarmu
            gn[i] = 0

    return alarms, gp, gn
```

**Doporučené parametry pro B3:**
- `threshold_sigma = 4.0` — konzervativní, málo false alarms
- `drift_fraction = 0.5` — standard (half minimum detectable shift)
- Input: per-trade PnL normalizovaný historical std
- Trigger action: Slack WARNING + Gemini Flash analýza

### 4.2 Bayesian Online Changepoint Detection (BOCPD)

Adams & MacKay 2007. Na rozdíl od CUSUM, BOCPD dává **pravděpodobnost**
změny režimu, ne binární alarm. To umožňuje graduální reakci.

**Proč je lepší než CUSUM pro B3:**
- CUSUM: "Režim se změnil" (binární) → co dál?
- BOCPD: "30% šance změny režimu" → snížit sizing o 30%
- BOCPD: "80% šance změny režimu" → killswitch

**Klíčové hyperparametry pro B3:**

```python
# Hazard rate: prior expectation of changepoint frequency
# B3 dělá ~33 tradů/den. Očekáváme regime change ~ každých 6 dní.
HAZARD_RATE = 1 / 200  # = 1/(33 trades/day × 6 days)

# Burn-in: minimum trades before BOCPD starts evaluating
BURN_IN = 50  # ~1.5 dne tradů

# Kill threshold: changepoint probability triggering action
SHOCK_THRESHOLD = 0.50   # Okamžitá anomálie (flash crash, structure change)
WARNING_THRESHOLD = 0.30  # Slack warning, investigate

# Erosion detection: slow alpha decay
# When expected run length < l_min for m consecutive ticks
L_MIN = 50   # Danger zone: run length shorter than expected
M_CONSECUTIVE = 15  # Need 15 consecutive ticks in danger zone
```

**Dual-trigger systém:**
1. **Shock Detection**: changepoint probability > 0.50 → okamžitý alarm
2. **Erosion Detection**: expected run length klesá pod minimum → pozvolný decay

### 4.3 Population Stability Index (PSI) — Feature Drift

PSI měří, jak moc se distribuce featur posunula oproti referenčnímu období.

```python
def calculate_psi(reference, current, n_bins=10):
    """PSI pro detekci feature distribution drift.

    PSI < 0.10: žádný drift
    PSI 0.10-0.25: moderate drift, investigate
    PSI >= 0.25: significant drift, trigger retraining
    """
    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints[-1] = np.inf
    breakpoints[0] = -np.inf

    ref_pct = np.histogram(reference, breakpoints)[0] / len(reference)
    cur_pct = np.histogram(current, breakpoints)[0] / len(current)

    # Avoid log(0)
    ref_pct = np.clip(ref_pct, 1e-6, None)
    cur_pct = np.clip(cur_pct, 1e-6, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
```

**Featury k monitorování pomocí PSI:**

| Feature | Proč je důležitá | PSI alarm threshold |
|---------|-----------------|---------------------|
| `velocity` | Klíčový V6.0 filter, mění se s BTC volatilitou | 0.20 |
| `dir_delta` | CL-Binance lag, mění se s Chainlink upgrady | 0.20 |
| `orderbook_spread` | MM behavior, likvidita. Nejsilnější prediktor WR | 0.15 |
| `fill_price` (live) | Execution quality, slippage trends | 0.20 |
| `sigma_norm` | Volatilní režim, mění se s makro | 0.25 |
| `liq_available_usd` | Celková likvidita, degraduje s nižším zájmem | 0.20 |
| `ta_adx_5m` | Trend strength se mění s BTC charakterem | 0.25 |
| `ta_rsi_5m` | Mean-reversion pressure | 0.25 |
| `ta_bb_width_5m` | Squeeze/expansion cykly | 0.20 |

**Cadence**: PSI vypočítat po každém evaluation cycle (50 tradů).
Reference window: prvních 200 V6.0 tradů (baseline).

---

## 5. Anomaly Triggery — Kdy Reagovat

### 5.1 Třístupňový Severity Systém

Inspirováno production monitoring best practices (PagerDuty, Datadog).
Kritické pro prevenci alert fatigue.

| Severity | Slack formát | Akce | Auto-resolve |
|----------|-------------|------|-------------|
| 🔴 **CRITICAL** | Bold červený blok, @CEO mention | Okamžitá pozornost. Zvážit pozastavení live. | Nikdy |
| 🟡 **WARNING** | Žlutý blok, bez mention | Investigate do 24h. Může být šum. | Ano, pokud metrika se vrátí do normy do 100 tradů |
| ℹ️ **INFO** | Šedý blok, informativní | Žádná akce. Kontextové pozorování. | Po 7 dnech |

### 5.2 Trigger Tabulka

| # | Trigger | Podmínka | Min N | Severity | Akce |
|---|---------|----------|-------|----------|------|
| T1 | Rolling WR drop (paper) | WR < baseline − 2σ | 50 | 🟡 WARNING | Gemini analýza → autonomní rozhodnutí |
| T2 | Rolling WR drop (live) | WR < baseline − 2σ | 30 | 🔴 CRITICAL | Gemini analýza → auto-pause pokud Tier 1 |
| T3 | Consecutive losses (live) | ≥ 5 za sebou | 5 | 🔴 CRITICAL | Auto-pause live + Slack report |
| T4 | Consecutive losses (paper) | ≥ 8 za sebou | 8 | 🟡 WARNING | Gemini analýza → autonomní rozhodnutí |
| T5 | Regime WR collapse | Bucket WR < 40% | 20/bucket | 🟡 WARNING | Gemini analýza → autonomní rozhodnutí |
| T6 | Spread distribution shift | PSI(spread) > 0.15 | 50 | 🟡 WARNING | Gemini analýza → autonomní rozhodnutí |
| T7 | CL-Binance delta drift | Mean delta shift > $20 | 50 | 🟡 WARNING | Gemini analýza → autonomní rozhodnutí |
| T8 | Fill price distribution shift | PSI(fill_price) > 0.20 | 50 | 🟡 WARNING | Gemini analýza → autonomní rozhodnutí |
| T9 | Calibration breakdown | ECE > 0.15 | 50 | 🔴 CRITICAL | Gemini analýza → auto-pause pokud Tier 1 |
| T10 | CUSUM alarm | Threshold breach (4σ) | 100 | 🟡 WARNING | Gemini analýza → autonomní rozhodnutí |
| T11 | BOCPD shock | Changepoint prob > 0.50 | 50 | 🔴 CRITICAL | Auto-pause live + Slack report |
| T12 | BOCPD erosion | Run length < l_min for 15 ticks | 100 | 🟡 WARNING | Gemini analýza → autonomní rozhodnutí |
| T13 | Daily PnL crash (live) | < −$20 single day | 1 day | 🔴 CRITICAL | Auto-pause live + Slack report |
| T14 | Velocity distribution shift | PSI(velocity) > 0.20 | 50 | ℹ️ INFO | Gemini analýza → autonomní rozhodnutí |
| T15 | New high-WR cluster found | Bucket WR > 85%, N > 20 | 20 | ℹ️ INFO (opportunity) | Gemini analýza → autonomní rozhodnutí |
| T16 | TA regime change | ADX crossed 15/25 threshold | 30 | ℹ️ INFO | Log + TA regime update |
| T17 | RSI extreme + momentum conflict | RSI >80 + UP signal OR RSI <20 + DOWN | 20 | 🟡 WARNING | Auto-filter nebo sizing reduction |
| T18 | Multi-TF divergence spike | MTF alignment flipped, PSI(MTF) > 0.30 | 50 | ℹ️ INFO | Sizing adjustment |
| T19 | Auto-revert triggered | Changed param reverted (WR drop >5pp) | 1 | 🟡 WARNING | Log revert + analyze |
| T20 | Autonomous change success | Changed param validated (WR stable/improved after 50t) | 1 | ℹ️ INFO | Promote to new default |

### 5.3 Anti-Alert-Fatigue Pravidla

1. **Dedup window**: 6 hodin. Stejný trigger type neopakovat dříve než za 6h.
2. **Suppression**: WARNING auto-resolves pokud metrika se vrátí do normy do
   dalšího eval cyklu (50 tradů / 6h).
3. **Batch**: Více triggerů ve stejném eval cyklu → jeden souhrnný Slack message.
4. **Runbook link**: Každý alert obsahuje odkaz na relevantní sekci tohoto dokumentu.
5. **Týdenní audit**: Jednou za týden v INFO reportu: kolik alertů fired, kolik
   bylo false alarm, threshold tuning.
6. **Nikdy nealertovat na jednotlivý trade**: B3 má 13-32% loss rate by design.
   Alertujeme na *patterny*, ne na jeden loss.

---

## 6. LLM Analyst — Gemini Flash Integrace

### 6.1 Kdy se LLM volá

LLM se volá **jen při anomálii** (WARNING nebo CRITICAL trigger), ne rutinně.
Očekávaná frekvence: 2-5× denně v klidném období, 5-10× při market stress.

**Náklady:**
- Gemini 2.5 Flash: $0.15/MTok input, $0.60/MTok output (paid tier, EEA)
- Context per call: ~3K tokens input, ~1K output = ~$0.001/call
- 10 calls/den × 30 dní = $0.30/měsíc
- Worst case (100 calls/den sustained): $3/měsíc

### 6.2 Context Packet (vstup pro LLM)

```python
context_packet = {
    # Co se stalo
    "trigger": {
        "type": "T5_regime_wr_collapse",
        "severity": "WARNING",
        "description": "CALM regime (σn<1.5) WR dropped to 22% (baseline 29%)",
        "timestamp": "2026-04-10T14:30:00Z",
    },

    # Aktuální výkon (rolling 50 tradů)
    "current_metrics": {
        "rolling_50_wr_paper": 0.62,
        "rolling_50_wr_live": 0.78,
        "rolling_50_avg_pnl": -0.12,
        "rolling_50_sharpe": 1.8,
        "consecutive_losses_paper": 2,
        "consecutive_losses_live": 0,
        "total_trades_since_v6": 312,
        "live_trades_since_v6": 48,
    },

    # Baseline (první 200 V6.0 tradů)
    "baseline_metrics": {
        "wr_paper": 0.68,
        "wr_live": 0.87,
        "avg_pnl_paper": 0.53,
        "sharpe": 2.4,
    },

    # Regime breakdown (aktuální vs baseline)
    "regime_breakdown": {
        "sigma_norm": {
            "CALM":     {"current_wr": 0.22, "baseline_wr": 0.29, "n": 24, "pnl": -8.50},
            "NORMAL":   {"current_wr": 0.58, "baseline_wr": 0.56, "n": 32, "pnl": 2.10},
            "ACTIVE":   {"current_wr": 0.71, "baseline_wr": 0.73, "n": 28, "pnl": 5.30},
            "VOLATILE": {"current_wr": 0.91, "baseline_wr": 0.95, "n": 11, "pnl": 12.40},
        },
        "velocity": { ... },
        "spread": { ... },
        "direction": { ... },
        "time_of_day": { ... },
    },

    # Market microstructure changes
    "market_structure": {
        "avg_spread_current": 0.045,
        "avg_spread_baseline": 0.030,
        "avg_cl_delta_current": 28.5,
        "avg_cl_delta_baseline": 18.0,
        "avg_fill_price_current": 0.52,
        "avg_fill_price_baseline": 0.48,
        "avg_liquidity_current": 280,
        "avg_liquidity_baseline": 450,
    },

    # PSI drift scores
    "feature_drift": {
        "velocity_psi": 0.08,
        "dir_delta_psi": 0.12,
        "spread_psi": 0.23,    # ← Moderate drift
        "fill_price_psi": 0.05,
        "sigma_norm_psi": 0.09,
    },

    # TA features (from TAFeatureProvider background cache)
    "ta_context": {
        "btc_5m": {"rsi": 62.3, "adx": 34.1, "macd_hist": 0.12, "bb_width": 0.015, "recommend": "BUY"},
        "btc_1h": {"rsi": 58.7, "adx": 28.4, "recommend": "BUY"},
        "btc_4h": {"rsi": 55.2, "adx": 22.1, "recommend": "NEUTRAL"},
        "multi_tf_aligned": True,
        "ta_regime": "STRONG_TREND",
    },

    "ta_regime_breakdown": {
        "RANGING":      {"current_wr": 0.35, "baseline_wr": None, "n": 12},
        "WEAK_TREND":   {"current_wr": 0.58, "baseline_wr": None, "n": 18},
        "STRONG_TREND": {"current_wr": 0.78, "baseline_wr": None, "n": 22},
    },

    # Autonomous decision history (last 5 decisions)
    "recent_decisions": [
        {"timestamp": "...", "param": "LIVE_MAX_VELOCITY", "old": 60, "new": 55,
         "reason": "WR in fast-move bucket dropped 12pp", "outcome": "pending"},
    ],

    # Posledních 10 ztrátových tradů (plné trade_details)
    "recent_losses": [ ... ],

    # Posledních 10 vítězných tradů (pro kontrast)
    "recent_wins": [ ... ],

    # Aktuální parametry
    "current_params": {
        "velocity_cap": 60,
        "dir_delta_cap": 15,
        "sigma_scale": 0.348,
        "entry_threshold": 0.020,
        "fill_price_cap": 0.75,
        "min_entry_min": 1,
        "max_entry_min": 3,
        "position_pct": 0.026,
        "edge_scaling": 10.0,
    },
}
```

### 6.3 System Prompt pro LLM Analyst

```
Jsi quantitative strategy optimizer pro B3 (Binance Oracle Scalper na Polymarket).
B3 je momentum scalper na 5-min BTC Up/Down markets. Vstupuje v minutě 2-3,
drží do resolution (never-sell mode live). CDF model s Chainlink oracle.

TY ROZHODUJEŠ. Nejsi poradce — jsi autonomní decision engine. Tvůj output
se aplikuje okamžitě (v rámci bezpečnostních bounds).

Dostal jsi data o anomálii v B3 výkonu. Tvůj úkol:

1. IDENTIFIKUJ ROOT CAUSE (ne symptom). Proč WR kleslo?
   Je to spread widening? Regime shift? Model miscalibration? TA divergence?

2. ANALYZUJ REGIME-SPECIFIC DATA. Nekritizuj celkový WR.
   Hledej: který bucket degradoval a proč? Koreluje to s TA regime změnou?
   Porovnej σ-based regime s ADX-based regime — souhlasí?

3. ROZHODNÍ O KONKRÉTNÍ AKCI:
   - `APPLY`: Změnit parametr (musí být v Tier 1/2 bounds)
   - `REVERT`: Vrátit předchozí změnu (pokud zhoršila výkon)
   - `ESCALATE`: Problém mimo bounds → Slack alert pro CEO
   - `MONITOR`: Nedostatek dat, sledovat dalších 50 tradů

4. PRO APPLY/REVERT SPECIFIKUJ:
   - Parametr a novou hodnotu
   - Expected impact (WR change, trades filtered, PnL impact)
   - Evaluation window (kolik tradů před auto-revert check)

5. OPTIMALIZUJ PRO: vysoký WR, nízký DD, velkou obratku kapitálu, stabilní edge.
   Trade-off: příliš agresivní filtrování = méně tradů = menší obratka.
   Hledej sweet spot.

6. POKUD DATA NESTAČÍ (< 30 tradů v relevantním bucketu), zvol MONITOR.

7. NIKDY nenavrhuj změnu Tier 3 parametrů (MAX_BET_SIZE, daily loss limit,
   fill price cap, oracle trust, execution mode, capital allocation).

TA Features Context:
- ADX > 25 = silný trend → momentum edge větší (B3 je momentum strategie)
- ADX < 15 = ranging → momentum edge malý, CALM analogie
- RSI extremes (>80/<20) = mean-reversion risk
- Multi-TF alignment = všechny timeframes souhlasí → vyšší confidence
- Bollinger squeeze = připravit se na breakout (CALM → VOLATILE)

Odpověz strukturovaně v JSON formátu. Buď stručný a datově podložený.
```

### 6.4 LLM Output Formát

```json
{
    "verdict": "APPLY | REVERT | ESCALATE | MONITOR",
    "root_cause": "Spread widening z $0.03 na $0.045 koreluje s ...",
    "evidence": [
        "PSI(spread) = 0.23 (moderate drift)",
        "CALM regime WR: 22% (24t) vs 29% baseline",
        "Spread > $0.04: 45% WR vs 72% WR při spread < $0.03"
    ],
    "action": {
        "type": "param_change | filter_toggle | pause | resume | revert",
        "param": "LIVE_MAX_VELOCITY",
        "old_value": 60,
        "new_value": 55,
        "tier": 1,
        "expected_impact": "Eliminuje ~12% fastest trades, WR boost ~3pp",
        "evaluation_window": 50,
        "auto_revert_threshold": 5
    },
    "risks": [
        "Filtruje 18% tradů — potenciálně missed profitable trades",
        "Spread je noisy metrikon — 50 tradů nemusí stačit",
        "Spread se může vrátit na baseline → zbytečný filter"
    ],
    "confidence": "HIGH | MEDIUM | LOW"
}
```

---

## 7. Slack Integration

### 7.1 Nový Kanál: `#b3-watchdog`

Dedikovaný kanál pro watchdog outputy. Separovaný od `#daily-brief` a
`#review-queue` aby neprodukoval noise v existujících kanálech.

### 7.2 Message Formáty

**Periodický INFO Report (každých 6h nebo po 50 tradech):**

```
━━━ B3 Watchdog Report ━━━
📊 Period: posledních 50 tradů (6.2h)

Paper: 48t | 31W 17L | WR 64.6% (baseline 68%)
Live:   4t |  3W  1L | WR 75.0% (baseline 87%)

Regime Breakdown:
  σ CALM:     5t, 40% WR ⚠️ (baseline 29%)
  σ NORMAL:  18t, 61% WR ✓
  σ ACTIVE:  15t, 73% WR ✓
  σ VOLATILE: 10t, 90% WR ✓

Feature Drift (PSI):
  velocity: 0.08 ✓ | spread: 0.12 ⚠️ | dir_delta: 0.05 ✓

ECE (calibration): 0.07 ✓

Status: ✅ NO ANOMALIES
```

**AUTONOMOUS ACTION Report:**

```
━━━ ⚡ B3 Watchdog — AUTONOMOUS ACTION ━━━
Trigger: T6 — Spread distribution shift
Action: APPLIED — LIVE_MAX_VELOCITY: 60 → 55

📈 Root Cause:
  PSI(orderbook_spread) = 0.23 (threshold: 0.15)
  Spread widening: $0.030 → $0.048 (+60%)
  Fast trades (vel>55) WR: 42% (vs 72% for vel<55)

📊 Expected Impact:
  −8% trades filtered | +3pp WR boost | net +$1.5/den

🤖 Gemini Flash Confidence: HIGH
  Based on 180 trades, 45 in affected bucket

⏱ Auto-revert check: po 50 tradech
  Revert threshold: WR drop > 5pp

Previous decisions (last 3):
  ✅ POSITION_PCT 0.026→0.020 (4d ago, WR +2pp — kept)
  ✅ LIVE_MIN_EDGE 0.40→0.45 (8d ago, WR +4pp — kept)
  ↩️ EDGE_SCALING 10→15 (12d ago, WR -3pp — reverted)
```

**CRITICAL Alert:**

```
━━━ 🔴 B3 Watchdog CRITICAL ━━━
Trigger: T11 — BOCPD structural regime change

⚡ Changepoint probability: 63%
   Poslední regime shift: před 312 trades
   Current regime hold: 8 trades (danger zone)

📊 Current rolling (50t):
   Paper WR: 52% (baseline 68%, drop −16pp)
   Live WR: 60% (baseline 87%, drop −27pp)
   Rolling Sharpe: 0.4 (baseline 2.4)

🤖 Gemini Flash analýza:
   Root cause: BTC vstoupilo do low-volatility ranging period.
   σ24h kleslo na 0.00018 (vs baseline 0.00035). CDF model
   generuje slabé signály (edge 0.02-0.05), V6.0 filters
   propouští low-quality trades.

   Doporučení: PAUSE live execution do návratu σ > 0.00025.
   Alternativa: Zvýšit LIVE_MIN_EDGE z 0.40 na 0.55.
   Confidence: HIGH

@CEO — vyžaduje okamžitou pozornost
```

---

## 8. Hardcoded vs Adaptive Parametry

Inspirováno Kalshi-bot patternem: explicitní separace parametrů které se
NIKDY automaticky nemění vs parametry které jsou kandidáty na optimalizaci.

### 8.1 HARDCODED — Nikdy Automaticky Neměnit

Změna vyžaduje CEO approval + kód review + deploy.

```python
# Risk Management (risk_manager.py)
MAX_BET_SIZE = 100.0          # Polymarket liquidity constraint
MIN_ORDER_SIZE = 5.0          # Polymarket minimum
DAILY_LOSS_LIMIT = 50.0       # B3 daily kill switch
MAX_SHARES = 500              # Per-trade cap
B3_ALLOCATED_CAPITAL = 1000   # Strategy allocation

# Execution Mode
EXECUTION_MODE = "dual"       # paper/dual/live
NEVER_SELL = True             # Live holds to resolution

# Oracle Trust
REQUIRE_CHAINLINK = True      # Live MUST have CL confirmation
BINANCE_FALLBACK = False      # Never fall back to Binance for live
RESOLUTION_TIMEOUT = 1800     # 30 min before Binance fallback

# Fill Price Cap
LIVE_MAX_FILL_PRICE = 0.75    # Data-driven (278 trades), hard engineering limit
```

### 8.2 ADAPTIVE — Kandidáti na Watchdog Optimalizaci

Watchdog může navrhnout změnu, CEO schválí, paper-first validace.

| Parametr | Aktuální | Range | Citlivost | Min N pro evaluaci |
|----------|---------|-------|-----------|-------------------|
| `LIVE_MAX_VELOCITY` | 60 $/min | 40 – 80 | Vysoká (paper: <30=78% WR, >80=6% WR) | 100 |
| `LIVE_MAX_DIR_DELTA` | $15 | $10 – $25 | Střední | 100 |
| `SIGMA_SCALE` | 0.348 | 0.25 – 0.50 | Velmi vysoká (core model parameter) | 200 |
| `ENTRY_THRESHOLD` | 0.020 | 0.01 – 0.05 | Nízká (0.02 je velmi široký) | 100 |
| `POSITION_PCT` | 0.026 | 0.01 – 0.05 | Střední (sizing) | 200 |
| `EDGE_SCALING` | 10.0 | 5 – 20 | Střední (sizing) | 200 |
| `MIN_ENTRY_MIN` | 1 | 1 – 2 | Nízká | 100 |
| `MAX_ENTRY_MIN` | 3 | 2 – 4 | Vysoká (min 4+ = 0W/2L v backteste) | 50 |
| `LIVE_MIN_EDGE` | 0.40 | 0.30 – 0.60 | Vysoká (kontroluje kolik tradů projde) | 100 |
| `LIVE_MIN_BTC_MOVE` | $50 | $30 – $80 | Střední | 100 |

### 8.3 NOVÉ FILTRY — Kandidáti na Přidání

Watchdog může navrhnout přidání nového filtru na základě dat:

| Potenciální filtr | Data evidence | Status |
|-------------------|--------------|--------|
| `MAX_ORDERBOOK_SPREAD ≤ $0.05-0.08` | Live: tight=80% WR, wide=45% WR | Čeká na 100+ live tradů |
| `MIN_SIGMA_NORM ≥ 1.5` | Paper: CALM=29% WR | Čeká na 50+ live CALM tradů |
| `MAX_COMBINED_RISK ≤ 1.5` | Live: CR>1.5=50% WR | Čeká na 50+ live CR>1.5 tradů |
| `PREFERRED_HOURS` (UTC 00-08) | Backtest: 02-03=72% WR | Čeká na time-of-day live data |
| `MIN_ADX_5M ≥ 15` | Analogie: CALM σ regime = 29% WR | Čeká na 100+ tradů s TA daty |
| `RSI extreme skip (>80 UP / <20 DOWN)` | 30%+ edge trades = 36.8% WR | Čeká na 100+ tradů |
| `Multi-TF aligned → sizing boost +30%` | Hypotéza, potřebuje validaci | Čeká na 200+ tradů |

---

## 9. Anti-Overfitting Safeguards

Klíčové poučení z výzkumu: **anti-overfitting je důležitější než optimalizace.**
Nunchi (103 experimentů) zjistil, že odebírání complexity bije přidávání.
ATLAS downweightoval svého vlastního CIO. DEV.to bot měl 83% WR ale
ztrácel kvůli špatnému sizingu.

### 9.1 Minimum Sample Sizes

| Rozhodnutí | Minimum tradů | Důvod |
|-----------|--------------|-------|
| Souhrnné WR hodnocení | 50 | Statisticky smysluplné |
| Per-bucket hodnocení | 20 per bucket | Minimum pro per-regime inference |
| Parametrická změna | 100 | Přežije noise |
| Nový filtr | 200 | Musí projít walk-forward |
| Model přetrénování | 500+ | SIGMA_SCALE, ENTRY_THRESHOLD |

### 9.2 Validační Pipeline pro Každou Změnu

```
1. Watchdog detekuje anomálii
   ↓
2. Gemini Flash analyzuje, navrhuje změnu
   ↓
3. CEO review v Slack (approve/reject)
   ↓ (approve)
4. PAPER-FIRST: Změna se aplikuje POUZE na paper parametry
   Live zůstává beze změny
   ↓
5. Paper validace: minimum 100 tradů (~3 dny)
   Porovnání: nový paper WR vs starý paper WR (paired t-test, p < 0.05)
   ↓
6. Walk-forward check: funguje na posledních 30% dat (out-of-sample)?
   ↓
7. CEO final approval pro live deployment
   ↓
8. Live deploy + monitoring dalších 50 live tradů
   ↓
9. Revert trigger: pokud live WR zhorší o > 5pp za 50 tradů → auto-revert návrh
```

### 9.3 One Change at a Time

NIKDY neměnit 2+ parametry najednou. Jinak nelze izolovat efekt.
Watchdog vynucuje: pokud probíhá paper validace jedné změny, další
návrhy se řadí do fronty.

### 9.4 Autonomní Auto-Revert (Primary Safety Net)

Každá autonomní změna má vestavěný revert mechanismus:

1. **Change tracking**: `watchdog_decisions` DB tabulka
   - `decision_id`, `timestamp`, `trigger_id`, `param_name`
   - `old_value`, `new_value`, `tier`, `gemini_confidence`
   - `trade_count_at_change`, `wr_at_change`, `pnl_at_change`
   - `status`: ACTIVE | REVERTED | PROMOTED
   - `evaluation_result`: NULL | KEPT | REVERTED

2. **Evaluation window**: 50 tradů po změně
   - Porovnání: WR_after vs WR_before (same-window comparison)
   - Threshold: drop > 5pp → auto-revert
   - Neutral (±2pp): extend evaluation o dalších 50 tradů
   - Positive (> +2pp): promote to new default

3. **Cascade prevention**: Maximálně 1 aktivní Tier 1 změna najednou
   - Pokud probíhá evaluace → další návrhy jdou do fronty
   - Tier 2 změny nezávislé na Tier 1 queue

4. **Hard reset**: Pokud 3 po sobě jdoucí změny se revertovaly:
   - Watchdog přejde do MONITOR-ONLY režimu na 200 tradů
   - Slack CRITICAL: "3 consecutive reverts — entering observation mode"
   - Po 200 tradech: automatický návrat do autonomního režimu

### 9.5 Stress Gating (DEV.to Pattern)

Během drawdownu (live PnL < −$20 za týden) se thresholdy zpřísní:
- Parametrická změna vyžaduje 20% improvement proof (ne jen "better")
- Minimum tradů se zdvojnásobí (100 → 200)
- Nový filtr vyžaduje 300 tradů místo 200

---

## 10. Implementační Plán

### Fáze 0 — Pasivní Sběr (TEĎ, 0 effort)

B3 už loguje všechna potřebná data v `trade_details` JSONB. Tato fáze
nevyžaduje žádný nový kód. Jen sbírat data.

**Cíl**: 200+ live V6.0 tradů pro baseline establishment.
**Aktuální stav**: 278+ live tradů (ale část je pre-V6.0). Potřebujeme
tracking od V6.0 deployment (2026-04-06). Aktuálně ~48 V6.0 live tradů.

**Milestone**: 200 V6.0 live tradů = ~6 dní aktivního tradingu od teď.

### Fáze 1 — Metrics Only (1-2 dny implementace)

Daemon bez LLM. Pouze SQL queries + metriky + Slack report.

**Soubory k vytvoření:**
```
arbo/core/b3_watchdog.py     # Main daemon
arbo/core/b3_metrics.py      # Metrics computation (PSI, CUSUM, ECE, regime)
```

**Integrace do `main_rdh.py`:**
```python
# V RDHOrchestrator.__init__:
self._b3_watchdog = B3Watchdog(session_factory=get_session_factory())

# V _start_monitoring_tasks:
asyncio.create_task(self._b3_watchdog.run_loop())
```

**Slack output**: Periodický INFO report (každých 6h). Žádné alerty.

**Deliverable**: "Tady je co vidím" — holá čísla bez interpretace.

### Fáze 2 — Anomaly Detection (1-2 dny implementace)

Přidáme threshold checks + triggery + Gemini Flash jako **autonomní decision engine**.
Watchdog sám rozhoduje a implementuje parametrické změny v Tier 1 bounds.
Tier 2 změny aplikuje s FLAG pro CEO review.
Přidáme `adaptive_config.py` pro runtime parametrický systém
a `watchdog_decisions` DB tabulku pro audit trail.

**Nové soubory:**
```
arbo/core/b3_anomaly.py       # Trigger logic + severity routing
arbo/core/adaptive_config.py  # Runtime params + audit log
arbo/agents/watchdog_agent.py # Gemini Flash autonomous decision engine
```

**Milestone**: Po 200 V6.0 live tradech (baseline established).

### Fáze 3 — Auto-Revert & Self-Learning (2-3 dny implementace)

Auto-revert mechanismus + self-learning loop.
Watchdog evaluuje každou vlastní změnu po 50 tradech.
Self-learning: loguje rozhodnutí + outcomes, Gemini dostává historii.

**Nové soubory:**
```
arbo/core/b3_paper_ab.py      # A/B paper testing engine
```

**Milestone**: Po 500 V6.0 live tradech (dostatek dat pro validaci).

### Fáze 4 — Self-Learning Loop (budoucnost)

Watchdog si pamatuje úspěšné a neúspěšné proposals. Gemini dostává
historii minulých decisions. Meta-learning: "Co fungovalo?"

**Milestone**: Po 1000+ V6.0 live tradech + 10+ schválených proposals.

---

## 11. Příloha A — SQL Queries pro Watchdog

### Query 1: Posledních N B3 tradů s trade_details

```sql
SELECT
    id, status, actual_pnl, placed_at, resolved_at,
    trade_details->>'direction' AS direction,
    (trade_details->>'velocity')::float AS velocity,
    (trade_details->>'dir_delta')::float AS dir_delta,
    (trade_details->>'sigma_norm')::float AS sigma_norm,
    (trade_details->>'orderbook_spread')::float AS spread,
    (trade_details->>'combined_risk')::float AS combined_risk,
    (trade_details->>'edge')::float AS edge,
    (trade_details->>'z_score')::float AS z_score,
    (trade_details->>'cl_ratio')::float AS cl_ratio,
    (trade_details->>'fill_to_model')::float AS fill_to_model,
    (trade_details->>'liq_available_usd')::float AS liquidity,
    (trade_details->>'live_entry_price')::float AS live_fill_price,
    (trade_details->>'live_fill_status') AS live_status,
    (trade_details->>'live_entry_latency_ms')::int AS latency_ms,
    (trade_details->>'btc_binance_chainlink_delta')::float AS cl_delta,
    (trade_details->>'velocity_paper')::float AS velocity_paper,
    (trade_details->>'abs_dir_delta_paper')::float AS abs_dd_paper
FROM paper_trades
WHERE strategy = 'B3'
  AND status IN ('won', 'lost', 'sold')
  AND actual_pnl IS NOT NULL
ORDER BY resolved_at DESC
LIMIT :n;
```

### Query 2: Denní B3 agregáty

```sql
SELECT
    DATE(resolved_at) AS trade_date,
    COUNT(*) AS trades,
    SUM(CASE WHEN status = 'won' THEN 1 ELSE 0 END) AS wins,
    ROUND(SUM(CASE WHEN status = 'won' THEN 1 ELSE 0 END)::numeric
          / NULLIF(COUNT(*), 0), 3) AS win_rate,
    SUM(actual_pnl) AS daily_pnl,
    AVG(actual_pnl) AS avg_pnl,
    -- Live-only metrics
    SUM(CASE WHEN trade_details->>'live_fill_status' = 'filled' THEN 1 ELSE 0 END) AS live_trades,
    SUM(CASE WHEN trade_details->>'live_fill_status' = 'filled'
             AND status = 'won' THEN 1 ELSE 0 END) AS live_wins
FROM paper_trades
WHERE strategy = 'B3'
  AND status IN ('won', 'lost', 'sold')
  AND actual_pnl IS NOT NULL
GROUP BY DATE(resolved_at)
ORDER BY trade_date DESC
LIMIT 30;
```

### Query 3: Regime bucket breakdown

```sql
WITH b3_trades AS (
    SELECT
        status,
        actual_pnl,
        (trade_details->>'sigma_norm')::float AS sigma_norm,
        (trade_details->>'velocity_paper')::float AS velocity,
        (trade_details->>'orderbook_spread')::float AS spread
    FROM paper_trades
    WHERE strategy = 'B3'
      AND status IN ('won', 'lost', 'sold')
      AND actual_pnl IS NOT NULL
      AND resolved_at >= NOW() - INTERVAL ':window_hours hours'
)
SELECT
    CASE
        WHEN sigma_norm < 1.5 THEN 'CALM'
        WHEN sigma_norm < 2.0 THEN 'NORMAL'
        WHEN sigma_norm < 2.5 THEN 'ACTIVE'
        ELSE 'VOLATILE'
    END AS sigma_regime,
    COUNT(*) AS trades,
    SUM(CASE WHEN status = 'won' THEN 1 ELSE 0 END) AS wins,
    ROUND(SUM(CASE WHEN status = 'won' THEN 1 ELSE 0 END)::numeric
          / NULLIF(COUNT(*), 0), 3) AS win_rate,
    ROUND(SUM(actual_pnl)::numeric, 2) AS total_pnl,
    ROUND(AVG(actual_pnl)::numeric, 3) AS avg_pnl
FROM b3_trades
GROUP BY 1
ORDER BY 1;
```

---

## 12. Příloha B — Competitive Intelligence

### Co dělají jiní (verified z web research, duben 2026)

> **Posun od advisory k autonomii**: Náš watchdog se posunul od "analyst" modelu (Kalshi-bot
> pattern) k plně autonomnímu optimizeru. Nejbližší analogie jsou ATLAS (auto-spawn specialistů,
> Darwinovské váhy) a Apex-Alpha-Bot (auto-weight adjustment). Klíčový rozdíl: 3-tier safety
> model s auto-revert mechanismem zajišťuje bezpečnost bez nutnosti CEO approval pro každou změnu.

| Projekt | Typ | Klíčový pattern | Výsledek |
|---------|-----|----------------|----------|
| **Kalshi-bot** | Prediction market | Denní `self_improve.py`, hardcoded vs adaptive separation, category-specific hit rates | Stabilní provoz, Discord review |
| **ATLAS** (General Intelligence Capital) | Multi-agent | 25 LLM agentů, Darwinovské váhy (±5%/den), auto-spawn specialistů po 3× same error | Sám identifikoval svůj slabý článek |
| **Nunchi auto-researchtrading** | Autoresearch | Karpathy loop, 103 experimentů, score=sharpe×√N-DD, immutable backtester | Sharpe 2.7→20.6 (ODEBÍRÁNÍ complexity > přidávání) |
| **Apex-Alpha-Bot** | Crypto HFT | Signal weights (+0.02 correct, −0.01 false, EMA smoothed, cap 0.05-0.35) | Confidence threshold 0.55 |
| **DEV.to Bot** | Multi-strategy | Regime-aware triggers, stress gating (DD>0.8→20% proof), min 10 paper trades pre accept | 83% WR ale špatný sizing → redesign |

### Akademické reference

| Metoda | Zdroj | Relevance pro B3 |
|--------|-------|-------------------|
| CUSUM | Page 1954, adapted Marcos López de Prado 2018 | Gradual drift detection |
| BOCPD | Adams & MacKay 2007 | Probabilistic regime change |
| PSI | Credit scoring industry (1990s) | Feature distribution monitoring |
| ECE | Naeini et al. 2015 | Probability calibration |
| Garman-Klass vol | Garman & Klass 1980 | Instant vol estimation (8× efficient) |
| Mean reversion | Bianchi 2020, Makarov & Schoar 2020 | BTC negative autocorrelation 1-5 min |
| Alpha decay | Maven Securities research 2025 | HFT: days-weeks, momentum: 3-6 months |

### Poučení z praxe (z výzkumu)

1. **Odebírání complexity bije přidávání** (Nunchi: −filters = +Sharpe)
2. **83% WR neznamená profit** (DEV.to: sizing error stačí na ztrátu)
3. **Systém si sám najde svůj slabý článek** (ATLAS: CIO = nejhorší agent)
4. **Hardcoded risk pravidla NIKDY adaptovat** (Kalshi: invariantní safety floor)
5. **Publication accelerates decay** (Maven: 30% Sharpe variance from publication year alone)
6. **Expected live DD = 1.5-2× backtest DD** (industry consensus)

---

## 13. Příloha C — Odhadovaný ROI Watchdogu

### Konzervativní odhad

B3 V6.0 generuje ~$2.7/den (87% WR, ~5 live tradů/den). Watchdog může pomoct ve dvou dimenzích:

**1. Loss avoidance (hlavní hodnota):**
- B3 má identifikované failure modes: wide spread (45% WR), high velocity (6% WR paper)
- Pokud watchdog odhalí a nafiltruje 15% nejhorších tradů (ty se spread > $0.05):
  - Eliminace: ~0.75 ztratových tradů/den
  - Savings: ~$0.50-1.00/den (avg loss per filtered trade)

**2. Sizing optimization (sekundární):**
- Pokud watchdog identifikuje high-confidence bucket (CR < 1.0 = 100% WR):
  - Zvýšení sizing pro best trades: +$0.20-0.50/den
  - Snížení sizing pro risky trades: −$0.10-0.30/den loss avoidance

**Konzervativní net benefit: +$0.50-1.50/den = $15-45/měsíc**
**Náklady: ~$1/měsíc (Gemini Flash)**
**ROI: 15-45×**

### Hlavní hodnota není přímý PnL ale RISK REDUCTION

Watchdog primárně chrání proti:
- Nedetekovaná alpha decay (strategy continues losing before anyone notices)
- Structural market changes (Polymarket fee changes, CL oracle upgrades)
- Model miscalibration drift (parameters optimal at deployment, stale after 3 months)

Bez watchdogu: problém se detekuje po dnech/týdnech manuální analýzy.
S watchdogem: problém se detekuje do 6 hodin / 50 tradů.

---

## 14. Otevřené Otázky pro CEO

1. **Nový Slack kanál `#b3-watchdog`** — OK, nebo integrovat do `#review-queue`?
2. **Fáze 1 priority** — Začít hned (i s < 200 V6.0 tradů), nebo čekat na milestone?
3. **Auto-revert threshold** — 5pp dostatečně konzervativní? Nebo 3pp?
4. **Rozšíření na další strategie** — C2, D priority pro watchdog po B3 validaci?
5. **Budget pro Gemini Flash** — Aktuální ~$1/měsíc OK? Cap na $5/měsíc?
6. **Tier 1 bounds review** — Jsou navržené autonomní rozsahy OK? Příliš široké/úzké?
