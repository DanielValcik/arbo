# Experiment Framework — Specifikace

> Verze: 1.0 | Datum: 2026-03-14
> Status: SPEC (implementace po dokončení Goldsky downloadu)
> Přístup: Dashboard Arbo → záložka "Experiments"

## Cíl

Najít nejprofitabilnější verzi modelu Strategy C na reálných historických datech.
Dva rovnocenné cíle:

1. **Profitabilita** — maximální ROI s akceptovatelným riskem
2. **Obratka kapitálu** — kapitál musí pracovat, ne sedět na účtu

Strategie je kompletní balík: KDY vstoupit (entry parametry + quality gate)
A KDY vystoupit (exit parametry). Obojí se optimalizuje společně.

Exit není jen "ochrana před ztrátou" — je to primárně **uvolnění kapitálu**
pro další příležitost. Pozice co sedí 48h na +300% unrealized blokuje
kapitál, který by mohl mezitím udělat 3 další trades.

Validovat proti paper trading výsledkům a mít jistotu že parametry
přeneseme do live s maximální důvěrou.

---

## 1. Data Pipeline

### 1.1 Zdroje dat

| Zdroj | Typ | Pokrytí | Refresh |
|-------|-----|---------|---------|
| **Goldsky subgraph** | On-chain OrderFilled trades → hourly VWAP | Jan 2025 → present | Při spuštění |
| **CLOB /prices-history** | Polymarket CLOB hourly snapshots | Posledních ~30 dní | Denně |
| **Paper trading (PostgreSQL)** | Reálné paper trades z VPS | Od spuštění paper mode | Live |
| **Open-Meteo archive** | Historické teploty (forecast proxy) | Neomezené | Při spuštění |

### 1.2 SQLite databáze (`research/data/price_history.sqlite`)

Tabulky:
- `events` — resolved weather eventy (city, target_date, closed_time, volume)
- `buckets` — temperature buckety (token_id, low_c, high_c, won)
- `prices` — CLOB cenové body (token_id, ts, price)
- `goldsky_trades` — on-chain trade ceny (token_id, ts, price)

`PriceHistoryDB` automaticky merguje data z obou cenových tabulek.
Při dotazu na cenu hledá nejbližší záznam v obou tabulkách.

### 1.3 Paper trading export

Paper trades se exportují z PostgreSQL na VPS:
```sql
SELECT token_id, price, edge_at_exec, size, status, actual_pnl,
       placed_at, resolved_at, strategy, market_condition_id
FROM paper_trades
WHERE strategy = 'C' AND status IN ('won', 'lost')
ORDER BY placed_at;
```

Export do JSON/CSV → `research/data/paper_trades_export.json`

Script: `research/export_paper_trades.py` (nový)
- SSH na VPS, query PostgreSQL, uloží lokálně
- Nebo: API endpoint v dashboardu co vrátí JSON

---

## 2. Experiment Framework

### 2.1 Evaluace jednoho experimentu

Vstup: parameter set (dict)
Výstup: `ExperimentResult` s metrikami

```python
@dataclass
class ExperimentResult:
    # Identifikace
    experiment_id: str          # UUID nebo auto-increment
    name: str                   # Lidsky čitelný název
    params: dict                # Všechny parametry
    timestamp: datetime         # Kdy proběhl

    # Compound metriky (PRIMÁRNÍ — sizing z aktuálního balance)
    trades: int
    wins: int
    win_rate: float             # %
    total_pnl: float            # $
    final_capital: float        # $
    roi_pct: float              # % return
    max_drawdown_pct: float     # % od peaku
    sharpe: float               # Anualizovaný
    calmar: float               # PnL / max DD

    # Obratka kapitálu (KLÍČOVÉ — kapitál musí pracovat)
    capital_utilization: float  # % času kdy je kapitál deployed (0-100)
    avg_time_in_trade_h: float  # Průměrná doba držení pozice (hodiny)
    turnover_rate: float        # Kolikrát se kapitál "otočil" za období
                                # = sum(trade_sizes) / avg_capital
    idle_hours: float           # Celkový počet hodin kdy kapitál nic nedělal
    concurrent_positions: float # Průměrný počet simultánních pozic

    # Per-trade statistiky
    avg_pnl_per_trade: float    # $ průměrný PnL
    avg_edge: float             # Průměrný edge při vstupu
    avg_price: float            # Průměrná entry cena
    median_pnl: float           # Median PnL (robustnější než průměr)
    avg_pnl_per_hour: float     # $ PnL / hodina v tradu (efektivita kapitálu)

    # Per-city breakdown
    city_results: dict[str, CityResult]

    # Per-entry-timing breakdown
    timing_results: dict[str, TimingResult]  # "24h" → result

    # Walk-forward OOS
    oos_pnl: float | None
    oos_trades: int | None
    oos_win_rate: float | None

    # Paper trading srovnání
    paper_correlation: float | None   # Korelace s paper trading
    paper_overlap_score: float | None # Jak moc se shodují trade decisions
```

### 2.2 Sizing — Compound (reinvestování)

Pozice se sizuje z **aktuálního balance**, ne z počátečního kapitálu.
To je přesně jak to funguje v produkci:

```python
capital = INITIAL_CAPITAL  # $1,000

for trade in trades_chronologically:
    # Sizing z aktuálního kapitálu
    size = min(capital * kelly_adj, capital * MAX_POSITION_PCT)

    if trade.won:
        pnl = size * (1.0 / fill_price - 1.0) - gas
    else:
        pnl = -size - gas

    capital += pnl  # Reinvestujeme zisk/ztrátu
```

Pokud strategie vydělá, větší pozice → větší zisky → snowball efekt.
Pokud prohraje, menší pozice → automatická risk redukce.
To je přesně quarter-Kelly chování.

### 2.3 Tracking obratky kapitálu

Evaluátor musí simulovat **celé období chronologicky** (ne event-by-event),
protože obratka závisí na tom co se děje paralelně:

```python
# Zjednodušený flow:
capital = 1000
deployed = {}  # token_id → {size, entry_price, entry_ts}
hourly_log = []  # Pro grafy

for hour_ts in range(period_start, period_end, 3600):

    # 1. Check exit conditions na všech otevřených pozicích
    for token_id, pos in list(deployed.items()):
        current_price = db.get_price_at(token_id, hour_ts)
        updated_edge = compute_edge(pos, hour_ts)

        if should_exit(pos, current_price, updated_edge, params):
            pnl = realize_exit(pos, current_price)
            capital += pos["size"] + pnl  # Kapitál se vrátí
            del deployed[token_id]

    # 2. Check resolution na všech pozicích co expirují tuto hodinu
    for token_id, pos in list(deployed.items()):
        if pos["closes_at"] <= hour_ts:
            pnl = resolve(pos)
            capital += pos["size"] + pnl
            del deployed[token_id]

    # 3. Hledat nové entry příležitosti (z VOLNÉHO kapitálu)
    available = capital - sum(p["size"] for p in deployed.values())
    if available > MIN_TRADE_SIZE:
        opportunities = scan_all_events(hour_ts, params)
        for opp in opportunities:
            size = compute_size(opp, available, params)
            if size > 0 and available >= size:
                deployed[opp.token_id] = open_position(opp, size, hour_ts)
                available -= size

    # 4. Log stav pro metriky
    total_deployed = sum(p["size"] for p in deployed.values())
    hourly_log.append({
        "ts": hour_ts,
        "capital": capital,
        "deployed": total_deployed,
        "idle": capital - total_deployed,
        "n_positions": len(deployed),
    })

# Výpočet metrik obratky z hourly_log
capital_utilization = mean(h["deployed"] / h["capital"] for h in hourly_log) * 100
idle_hours = sum(1 for h in hourly_log if h["deployed"] == 0)
turnover = sum(all_trade_sizes) / mean(h["capital"] for h in hourly_log)
```

Klíčový rozdíl oproti starému evaluátoru: starý procházel event-by-event
a ignoroval paralelní příležitosti. Nový simuluje portfolio v čase —
vidíme kolik příležitostí jsme vynechali kvůli zamčenému kapitálu
a kolik nových jsme chytili díky exitům.

### 2.4 Multi-timing evaluace

Každý experiment se testuje na 5 entry timingech:
- 48h, 36h, 24h, 12h, 6h před market close

```python
ENTRY_TIMINGS = [48, 36, 24, 12, 6]

timing_results = {}
for hours in ENTRY_TIMINGS:
    result = evaluate(params, entry_hours=hours)
    timing_results[f"{hours}h"] = result

# Composite score = vážený průměr
# Důraz na 24h (naše typická entry) ale nesmí úplně selhat na jiných
composite = (
    timing_results["48h"].score * 0.15 +
    timing_results["36h"].score * 0.20 +
    timing_results["24h"].score * 0.30 +  # Hlavní
    timing_results["12h"].score * 0.20 +
    timing_results["6h"].score  * 0.15
)
```

### 2.4 Train/Test split

```
Data timeline:
|←— Jan 2025 ——————————— Jan 2026 ——→|←— Feb-Mar 2026 ——→|
|            TRAINING SET              |     TEST SET       |
|     (parameter optimization)         | (final validation) |
```

- **Training**: Leden 2025 → Leden 2026 (~80% dat)
- **Test**: Únor-Březen 2026 (~20% dat) — NIKDY se nepoužije k optimalizaci
- **Walk-forward**: 3 foldy v rámci training setu

Test set se použije JEN JEDNOU na finální vybranou konfiguraci.
To zabrání overfittingu.

### 2.5 Walk-forward cross-validace

V rámci training setu:
```
Fold 1: Train [Jan-Jun 2025] → Test [Jul-Sep 2025]
Fold 2: Train [Jan-Sep 2025] → Test [Oct-Dec 2025]
Fold 3: Train [Jan-Dec 2025] → Test [Jan 2026]
```

Expandující okno (ne rolling) — každý fold má víc dat.
Reportujeme **průměr OOS metrik** přes foldy.

---

## 3. Paper Trading Validace

### 3.1 Princip

Pro každý paper trade (reálný obchod z VPS) najdeme stejný event
v backtest datech. Porovnáme:

| Metrika | Paper (live) | Backtest | Shoda? |
|---------|-------------|----------|--------|
| Vzal trade? | ANO | ? | ✓/✗ |
| Entry price | 0.15 | 0.14 | ~OK |
| Edge | 0.08 | 0.09 | ~OK |
| Size | $45 | $47 | ~OK |
| Won? | YES | YES | ✓ |
| PnL | +$255 | +$270 | ~OK |

### 3.2 Metriky srovnání

```python
@dataclass
class PaperValidation:
    # Trade overlap
    paper_trades: int          # Kolik trades paper udělal
    backtest_trades: int       # Kolik trades backtest by udělal
    overlap: int               # Kolik trades oba shodně vzali
    paper_only: int            # Paper vzal, backtest ne
    backtest_only: int         # Backtest by vzal, paper ne
    overlap_pct: float         # overlap / union

    # Directional agreement
    both_won: int              # Oba shodně vyhrály
    both_lost: int             # Oba shodně prohrály
    direction_match_pct: float # Shoda won/lost

    # Price correlation
    price_correlation: float   # Pearson korelace entry cen
    edge_correlation: float    # Pearson korelace edge hodnot
    pnl_correlation: float     # Pearson korelace PnL

    # PnL srovnání
    paper_total_pnl: float
    backtest_total_pnl: float
    pnl_ratio: float           # backtest / paper (ideálně ~1.0)

    # Confidence score (0-100)
    # Jak moc věříme že backtest predikuje live výkon
    confidence: float
```

### 3.3 Confidence score

```python
def confidence_score(validation: PaperValidation) -> float:
    """0-100 skóre jak moc backtest odpovídá paper tradingu."""

    scores = []

    # Trade overlap (30% váha) — shodují se na tom JAKÉ trades vzít?
    scores.append(validation.overlap_pct * 30)

    # Direction match (20% váha) — shoduje se won/lost?
    scores.append(validation.direction_match_pct * 20)

    # PnL ratio (30% váha) — sedí absolutní PnL?
    ratio = validation.pnl_ratio
    if ratio > 1: ratio = 1 / ratio  # Penalizuj oba směry
    scores.append(ratio * 30)

    # Price correlation (20% váha) — sedí vstupní ceny?
    scores.append(max(0, validation.price_correlation) * 20)

    return sum(scores)
```

Interpretace:
- **80-100**: Backtest věrně replikuje live → parametry důvěryhodné
- **60-80**: Přijatelná shoda → parametry použitelné s opatrností
- **40-60**: Slabá shoda → backtest má systematický bias
- **<40**: Backtest nesedí → NEPOUŽÍVAT pro rozhodování

---

## 4. Scoring & Ranking experimentů

### 4.1 Composite Score

```python
def experiment_score(result: ExperimentResult) -> float:
    """Skóre pro ranking experimentů. Vyšší = lepší.

    Dva primární cíle: profitabilita + obratka kapitálu.
    Strategie co vydělá 500% ale kapitál sedí 90% času idle
    je HORŠÍ než strategie co vydělá 300% s kapitálem stále v práci.
    """

    # ── Profitabilita (50% celkové váhy) ──

    # ROI (25%) — kolik vyděláme
    roi_score = result.roi_pct / 100

    # Sharpe (15%) — risk-adjusted return
    sharpe_score = min(result.sharpe / 5.0, 2.0)

    # Drawdown penalty (10%) — nechceme velké DD
    dd_score = max(0, 1.0 - result.max_drawdown_pct / 50)

    # ── Obratka kapitálu (30% celkové váhy) ──

    # Capital utilization (15%) — kolik % času je kapitál deployed
    util_score = result.capital_utilization / 50  # 50%+ utilization = max

    # PnL per hour (15%) — efektivita: kolik vyděláme za hodinu v tradu
    # Odměňuje strategie co exitují v profitu rychle
    pph = result.avg_pnl_per_hour
    pph_score = min(max(pph, 0) / 5.0, 2.0)  # $5/h deployed = max

    # ── Validace (20% celkové váhy) ──

    # Walk-forward OOS (10%)
    if result.oos_pnl is not None and result.oos_pnl > 0:
        oos_score = min(result.oos_pnl / 1000, 2.0)
    else:
        oos_score = 0

    # Trade count (10%) — dost trades pro statistickou validitu
    trade_score = min(result.trades / 100, 2.0)

    return (
        roi_score * 25 +
        sharpe_score * 15 +
        dd_score * 10 +
        util_score * 15 +
        pph_score * 15 +
        oos_score * 10 +
        trade_score * 10
    )
```

### 4.2 Povinné filtry (experiment musí projít)

Experiment se nezobrazí v rankingu pokud:
- trades < 10 (příliš málo dat)
- max_drawdown > 50% (příliš riskantní)
- OOS PnL < 0 (out-of-sample ztrátový)
- Sharpe < 0.5 (nedostatečný risk-adjusted return)
- capital_utilization < 5% (kapitál skoro vůbec nepracuje)

---

## 5. Dashboard — Experiment Page

### 5.1 Přístup

Dashboard Arbo (`/experiments`) — nová stránka/záložka.
Implementace: rozšíření `arbo/dashboard/web.py` + nový template.

### 5.2 Layout stránky

```
┌─────────────────────────────────────────────────────────────┐
│  EXPERIMENT DASHBOARD                          [Run Sweep]  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─── Overview ──────────────────────────────────────────┐  │
│  │ Total experiments: 4,100  │ Best ROI: 478%            │  │
│  │ Data: 500K prices         │ Best Sharpe: 8.24         │  │
│  │ Events: 1,400             │ Paper confidence: 72/100  │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─── Top 10 Experiments (ranked by composite score) ────┐  │
│  │ # │ Name    │ ROI   │ WR   │ Trades│DD  │Sharpe│Score │  │
│  │ 1 │ WF-v3   │ 520%  │ 53%  │  33   │13% │ 8.2  │ 94  │  │
│  │ 2 │ AG-v2   │ 1170% │ 20%  │  151  │29% │ 3.3  │ 87  │  │
│  │ 3 │ ...     │       │      │       │    │      │     │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─── Charts ────────────────────────────────────────────┐  │
│  │                                                       │  │
│  │  [Equity Curves]  [City Heatmap]  [Parameter Space]   │  │
│  │  [Entry Timing]   [Paper Validation]  [Score Dist]    │  │
│  │                                                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─── Selected Experiment Detail ────────────────────────┐  │
│  │ Parameters: min_edge=0.03, max_price=0.75, ...        │  │
│  │ Per-city: London +$1,200, NYC +$800, Paris +$450      │  │
│  │ Per-timing: 48h→520%, 24h→478%, 12h→120%, 6h→15%     │  │
│  │ Walk-forward: Fold1 +$800, Fold2 +$1,200, Fold3 +$600│  │
│  │ Paper validation: confidence 72/100                   │  │
│  │                                                       │  │
│  │ [Apply to Production]  [Export CSV]  [Compare]        │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Grafy (Chart.js, renderované v prohlížeči)

Data pro všechny grafy servíruje endpoint `/api/experiments/charts`.
Frontend je renderuje přes Chart.js — konzistentní s existujícím dashboardem.
Hover, zoom, toggle čar fungují nativně.

#### A) Equity Curves (hlavní graf) — Line Chart
- X: čas (dny)
- Y: kapitál ($)
- Čáry: top 5 experimentů (toggle) + paper trading (čárkovaná)
- Šedé pásmo: drawdown band pro vybraný experiment
- Vertikální čára: train/test split hranice
- Hover: konkrétní trade detail (město, bucket, edge, PnL)

#### B) City Heatmap — Stacked Bar / Matrix
- X: města (20)
- Y: PnL ($) per experiment (grouped bars)
- Barva: experiment ID
- Toggle: zobrazit top 5 / top 10 / top 20
- Pomáhá vizuálně najít které město je konzistentně dobré/špatné

#### C) Parameter Sensitivity — Scatter Chart (per parametr)
- Dropdown: výběr parametru (min_edge, max_price, prob_sharpening, ...)
- X: hodnota parametru
- Y: composite score
- Bod = jeden experiment, barva = phase (random/finetune/city)
- Identifikuje: "min_edge pod 0.05 je vždy lepší"

#### D) Entry Timing — Bar Chart
- X: entry timing (48h, 36h, 24h, 12h, 6h)
- Y: ROI %
- Grouped bars: top 5 experimentů
- Ukazuje: "24-48h je vždy nejlepší"

#### E) Paper Trading Srovnání — Dual Line + Scatter
- Line: backtest PnL (plná) vs paper PnL (čárkovaná) přes čas
- Scatter: backtest edge vs paper edge (per trade, 1:1 reference čára)
- Bar: overlap / paper-only / backtest-only počty
- Badge: confidence score (0-100)

#### F) Exit Analýza — Bar + Line
- Bar: composite score s exity vs bez exitů (per experiment)
- Line: kumulativní saves - regrets přes čas
- Overlay: backtest (plná) vs shadow tracker (čárkovaná)

#### G) Capital Deployment Timeline — Area Chart
- X: čas (hodiny)
- Y: $ deployed (area) + $ idle (bílá mezera nahoře)
- Vizuálně ukazuje jak efektivně strategie využívá kapitál
- Zelené bloky = pozice, mezery = idle kapitál
- Pro vybraný experiment: vidíme jestli kapitál sedí nebo pracuje
- Overlay: cumulative PnL čárou (druhá osa)

#### H) Score Distribution — Histogram
- X: composite score (binned)
- Y: počet experimentů
- Vertikální čára: vybraný experiment
- Pomáhá pochopit jak výjimečný je náš best result

---

## 6. Exit Mechanismus v Experimentech

### 6.1 Exit = obratka + ochrana

Exit má dva účely (v tomto pořadí důležitosti):

1. **Uvolnit kapitál pro další trade** — pokud pozice má unrealized profit
   a edge se zmenšil, je lepší vybrat profit a nasadit kapitál na novou
   příležitost s lepším edge. Kapitál nesmí sedět.

2. **Ochránit kapitál** — pokud se forecast změnil a pozice ztratila edge,
   exit omezí ztrátu (ale NE přes cenový stop-loss, ten škodí).

**Příklad obratky:**
```
Hodina 0:  Koupíme bucket "Seoul 8-10°C" za $0.12, edge 0.18
Hodina 12: Cena stoupla na $0.35, edge klesl na 0.05
           → EXIT: realizujeme +$0.23 profit (+192%)
           → Kapitál ($1,230) je volný
Hodina 13: Nový trade "Dallas 15-18°C" za $0.10, edge 0.20
           → Entry: $61.50 (5% z $1,230)
           → Bez exitu bychom tuto příležitost vynechali (kapitál zamčený v Soulu)
```

Bez exitů: 1 trade za 48h, kapitál zamčený.
S exity: 2+ trades za 48h, kapitál se otáčí.

Předchozí exit sweep (`research/sweep_exits.py`, 90+ variant na syntetických
datech) ukázal jen +0.7% zlepšení. Ale to bylo proto, že syntetický backtest
neměl hodinové ceny a neměřil obratku. Na reálných Goldsky datech s hodinovými
cenami bude dopad výrazně větší.

Shadow tracker běží v produkci (`arbo/strategies/shadow_exit_tracker.py`,
`MIN_HOLD_EDGE=0.15`) a sbírá live data.

### 6.2 Exit parametry v experimentech

Každý experiment bude mít exit konfiguraci:

```python
EXIT_PARAMS = {
    "exit_enabled": True/False,

    # ── Edge-based exit (hlavní mechanismus) ──
    "min_hold_edge": float,    # 0.0 až 0.20
                               # Exit když updated_edge < tato hodnota
                               # Nízká hodnota = trpělivý (drží i s malým edge)
                               # Vysoká = agresivní (exituje rychle → lepší obratka)
                               # Syntetický backtest optimum: 0.15

    # ── Probability floor ──
    "prob_exit_floor": float,  # 0.0 (disabled) až 0.50
                               # Exit pokud updated_prob < floor

    # ── Profit-taking pro obratku kapitálu ──
    "profit_take_enabled": True/False,
    "profit_take_threshold": float,  # Unrealized gain % → trigger exit
                                     # 0.50 = exit při +50% unrealized
                                     # 1.00 = exit při +100% (price doubled)
                                     # 2.00 = exit při +200% (cena 3x entry)
    "profit_take_min_hours": int,    # Minimum hodin v pozici před profit-take
                                     # Zabrání okamžitému flippování
                                     # 6 = aspoň 6h držet

    # ── Re-entry po exitu ──
    "reentry_enabled": True/False,   # Po exitu hledat nový trade okamžitě?
    "reentry_cooldown_h": int,       # Minimum hodin čekat před re-entry
                                     # 1 = aspoň 1h pauza

    # ── Stop-loss (DISABLED — prokázáno že škodí) ──
    "stop_loss_pct": 1.0,     # 1.0 = disabled, vždy
}
```

**Klíčový sweep: `min_hold_edge` × `profit_take_threshold`**

Tyto dva parametry řídí obratku:
- Vyšší `min_hold_edge` (0.15-0.20) → častější exity → víc obratky
  ale risiko předčasného exitu (regret)
- Nižší `profit_take_threshold` (0.50) → profit-take při menším gainu
  → rychlejší obratka ale menší profit per trade
- Optimální kombinace maximalizuje `avg_pnl_per_hour` — efektivitu
  kapitálu v čase

### 6.3 Jak exit funguje v backtestu na reálných datech

Na rozdíl od syntetického backtestu (kde se ceny měnily jen mezi dny),
Goldsky/CLOB data mají **hodinové ceny** → realistická simulace:

```
Entry: 48h před close, koupíme bucket za $0.12
  → 36h: cena klesla na $0.08, updated_edge = 0.02 < min_hold_edge (0.15)
  → EXIT: prodáme za $0.08, ztráta -33%
  → Counterfactual: bucket prohrál, ztráta by byla -100%
  → SAVE: exit zachránil 67% kapitálu

Entry: 24h před close, koupíme bucket za $0.18
  → 12h: cena stoupla na $0.35, updated_edge = 0.25 > min_hold_edge
  → HOLD: držíme
  → 6h: cena na $0.55
  → Resolution: bucket vyhrál → PnL = +$0.82 per $1
```

Implementace v evaluátoru:

```
1. Entry v čase T (entry_hours před close)
2. Pro každou hodinu od T do close:
   a. Lookup aktuální cenu z Goldsky/CLOB
   b. Přepočítat updated_prob (sigma klesá bliž k resolution)
   c. Check exit conditions:
      - Edge-based: updated_edge < min_hold_edge?
      - Profit-take: unrealized_gain > profit_take_threshold?
      - Prob floor: updated_prob < prob_exit_floor?
   d. Pokud exit triggered:
      - Realizovat PnL z aktuální ceny (s taker slippage)
      - Uvolnit kapitál
      - Track counterfactual (co by se stalo kdybychom drželi)
      - Pokud reentry_enabled: po cooldown hledat nový trade
        ve VŠECH aktivních eventech (ne jen v tomto)
3. Pokud bez exitu → PnL z resolution (won/lost)
4. Metriky: track čas v každé pozici, idle čas, concurrent positions
```

Re-entry loop je klíčový pro obratku: po exitu kapitál okamžitě
hledá další příležitost. Backtest simuluje celé období chronologicky
s aktivním portfolio managementem, ne izolované single-trade evaluace.

### 6.4 Exit jako sweep dimenze

Exit parametry se sweepují **společně** s quality gate parametry:

```python
FULL_SEARCH_SPACE = {
    # Quality gate (stávající)
    "min_edge": [...],
    "max_price": [...],
    "min_price": [...],
    ...

    # Exit (nové)
    "exit_enabled": [True, False],
    "min_hold_edge": [0.0, 0.05, 0.10, 0.15, 0.20],
    "prob_exit_floor": [0.0, 0.30, 0.40, 0.50],
    "stop_loss_pct": [1.0],  # Disabled — prokázáno že škodí
}
```

To umožňuje najít nejlepší **kombinaci** entry + exit parametrů,
ne optimalizovat je odděleně.

### 6.5 Validace proti shadow trackeru

Shadow tracker na VPS sbírá reálná data:
- `saves`: kolikrát by exit zachránil kapitál
- `regrets`: kolikrát by exit přišel o zisk
- `save_rate_pct`: poměr saves / celkem

Experiment framework porovná:
1. Backtest save rate s MIN_HOLD_EDGE=0.15 na reálných datech
2. Shadow tracker save rate z live paper tradingu
3. Pokud se shodují (±10%) → exit logika je validní pro deploy

```python
@dataclass
class ExitValidation:
    backtest_save_rate: float    # Z experiment frameworku
    shadow_save_rate: float      # Z shadow trackeru na VPS
    delta: float                 # Rozdíl
    validated: bool              # |delta| < 10%

    backtest_exits: int
    shadow_exits: int
    backtest_avg_save: float     # Průměrný $ saved per exit
    shadow_avg_save: float
```

### 6.6 Grafy pro exit analýzu (Chart.js)

Na dashboard stránce `/experiments`:

**Exit Impact Chart:**
- Bar chart: composite score s exity vs bez exitů (per experiment)
- Ukazuje jestli exit přidává hodnotu

**Save/Regret Timeline:**
- X: čas, Y: kumulativní saves - regrets
- Čára by měla stoupat (saves > regrets)
- Overlay: backtest (plná) vs shadow tracker (čárkovaná)

**Exit Timing Distribution:**
- Histogram: kolik hodin před close k exitu dochází
- Pomáhá pochopit kdy je exit nejužitečnější

---

## 7. Sweep Proces (po dokončení Goldsky)

### 7.1 Kroky

```
1. Goldsky download dokončen ✓
2. Export paper trades z VPS → JSON
3. Export shadow tracker data z VPS → JSON
4. Spustit sweep na TRAINING setu:
   - 2,000 random search (quality gate + exit params společně)
   - 1,000 fine-tuning (top 20)
   - 500 city optimization
   - 500 multi-timing evaluace
   - 500 exit optimization (exit params kolem nejlepších configs)
   - 200 walk-forward validace
5. Filtrovat (min trades, max DD, OOS > 0)
6. Top 10 → paper trading validace (confidence score)
7. Top 10 → shadow tracker exit validace
8. Top 3 → evaluace na TEST setu (jednorázově!)
9. Generovat dashboard report + Chart.js grafy
10. Finální výběr → [CEO schválení] → deploy do produkce
```

### 7.2 Parameter Space

```python
SEARCH_SPACE = {
    # ── Quality Gate ──
    "min_edge":        [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04,
                        0.05, 0.06, 0.08, 0.10, 0.12, 0.15],
    "max_edge":        [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    "max_price":       [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55,
                        0.60, 0.70, 0.80, 0.90],
    "min_price":       [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08,
                        0.10, 0.12, 0.15, 0.20, 0.25],
    "min_prob":        [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15,
                        0.20, 0.25, 0.30, 0.40, 0.50],
    "min_volume":      [0, 10, 25, 50, 100, 200, 500, 1000, 2000],
    "kelly_raw_cap":   [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                        0.50, 0.60],
    "prob_sharpening": [0.70, 0.80, 0.85, 0.90, 0.95, 1.0, 1.05,
                        1.10, 1.15, 1.20, 1.30, 1.50],
    "shrinkage":       [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10,
                        0.15, 0.20, 0.25],
    "excluded_cities": [subsets of ALL_CITIES],
    "city_overrides":  {city: {min_edge, max_price, min_price}},

    # ── Exit Params ──
    "exit_enabled":             [True, False],
    "min_hold_edge":            [0.0, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
    "prob_exit_floor":          [0.0, 0.20, 0.30, 0.40, 0.50],
    "profit_take_enabled":      [True, False],
    "profit_take_threshold":    [0.30, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00],
    "profit_take_min_hours":    [2, 4, 6, 12],
    "reentry_enabled":          [True, False],
    "reentry_cooldown_h":       [1, 2, 4, 6],
    "stop_loss_pct":            [1.0],  # Disabled — prokázáno že škodí
}
```

---

## 8. Implementační Soubory

| Soubor | Účel | Status |
|--------|------|--------|
| `research/experiment_framework.py` | Evaluátor (ExperimentResult, scoring, exit logic, walk-forward) | NOVÝ |
| `research/sweep_final.py` | Finální sweep s train/test split + exit params | NOVÝ |
| `research/export_paper_trades.py` | Export paper trades z VPS PostgreSQL (HTTP→JSON) | NOVÝ |
| `research/export_shadow_data.py` | Export shadow tracker dat z VPS | NOVÝ |
| `research/paper_validation.py` | Srovnání backtest vs paper + shadow tracker | NOVÝ |
| `arbo/dashboard/web.py` | +route `/experiments`, +`/api/experiments/charts`, +`/api/paper-trades` | ROZŠÍŘENÍ |
| `arbo/dashboard/templates/experiments.html` | Dashboard stránka (Chart.js) | NOVÝ |
| `research/data/experiments/` | JSON výsledky per experiment | ADRESÁŘ |
| `research/data/paper_trades_export.json` | Exportované paper trades | GENEROVANÝ |
| `research/data/shadow_tracker_export.json` | Exportovaná shadow tracker data | GENEROVANÝ |

---

## 9. Rozhodnutí

### Q1: Paper trade export — SSH+psql skript nebo REST API?

**Rozhodnutí: Nový API endpoint na VPS (`GET /api/paper-trades`).**

Dashboard už běží na VPS jako FastAPI app s autentizací (HTTP Basic).
Přidáme endpoint `GET /api/paper-trades?strategy=C&status=won,lost`
který vrátí JSON pole paper trades. Toto je konzistentnější s architekturou
než ad-hoc SSH skripty a umožňuje export i z dashboardu přímo.

Endpoint přidáme do `arbo/dashboard/web.py` vedle existujících
(`/api/portfolio`, `/api/trades`, `/api/daily-pnl`, atd.).

Pro lokální experiment framework: `research/export_paper_trades.py`
zavolá tento endpoint přes HTTP a uloží do
`research/data/paper_trades_export.json`.

### Q2: Grafy — Matplotlib PNG nebo interaktivní?

**Rozhodnutí: Chart.js (jako existující dashboard), žádný Matplotlib.**

Důvody:
- Dashboard už používá Jinja2 HTML templates + Chart.js pro všechny grafy
- Chart.js umožňuje hover tooltips, zoom, toggle čar — důležité pro
  porovnávání experimentů
- Konzistentní UX — experiment stránka vypadá jako zbytek dashboardu
- Data pro grafy servíruje FastAPI endpoint (`/api/experiments/charts`),
  frontend renderuje Chart.js
- Žádná Python grafická knihovna, vše v prohlížeči

### Q3: Jak často re-runovat sweep?

**Rozhodnutí: Po každém novém týdnu dat, automaticky na VPS.**

- Goldsky download: cron job 1x denně (stahuje nové trades)
- CLOB refresh: cron job 1x denně (obnoví posledních 30 dní)
- Sweep: 1x týdně (neděle noc), výsledky se zapíšou do
  `research/data/experiments/sweep_YYYYMMDD.json`
- Dashboard `/experiments` zobrazí historii sweepů — trend jak se parametry
  mění s přibývajícími daty
- Manuální sweep: tlačítko `[Run Sweep]` v dashboardu (spustí na pozadí)

### Q4: Liquidity cap — omezit max pozici absolutně?

**Rozhodnutí: ANO, absolutní cap $200 per pozice.**

Data z databáze ukazují průměrné objemy per bucket:
- Seoul: $21.7K, London: $20.2K, NYC: $18K, Ankara: $15K
- Ale minimum je $46-$182 na menších bucketech

Pravidla:
- `MAX_POSITION_USD = 200` — tvrdý cap bez ohledu na balance
- Toto simuluje reálnou limitaci: na $200 pozici nás trh absorbuje bez
  significant impact (< 1% typického bucket volume)
- S $500+ pozicí bychom hýbali cenou u menších bucketů
- V backtestu: `size = min(size, MAX_POSITION_USD)` — zabrání
  nerealistickému compound snowballu
- Compound efekt stále funguje: víc kapitálu = víc simultánních pozic,
  ne větší jednotlivé pozice
- **V produkci** toto už respektuje risk_manager.py přes MAX_POSITION_PCT,
  ale absolutní cap přidáme explicitně

Důsledek pro backtest: místo $117K z jednoho snowball run dostaneme
realističtější výsledky kde compound funguje přes diverzifikaci (víc
trades současně), ne přes eskalaci velikosti.

### Q5: Test set leak — jak oddělit?

**Rozhodnutí: Striktní chronologický split PŘED spuštěním swepu.**

```python
# V sweep_final.py:
TRAIN_END = "2026-01-31"    # Training data: vše do konce ledna
TEST_START = "2026-02-01"   # Test data: únor-březen 2026

training_events = [e for e in all_events
                   if e.target_date and e.target_date <= TRAIN_END]
test_events = [e for e in all_events
               if e.target_date and e.target_date >= TEST_START]

# Sweep probíhá VÝHRADNĚ na training_events
# test_events se použijí JEN JEDNOU na finální konfiguraci
```

- Sweep nikdy nevidí test data
- Walk-forward validace probíhá UVNITŘ training setu
- Test set evaluace = poslední krok před production deploy
- Výsledek testu se zapíše do reportu jako "final OOS performance"
- Pokud test set performance je výrazně horší než training → overfitting,
  parametry se NEDEPLOYUJÍ
