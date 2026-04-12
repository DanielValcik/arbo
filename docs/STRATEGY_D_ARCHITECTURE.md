# Strategy D — Multi-Sport Architecture

> **Decided:** 2026-04-12
> **Status:** Production (NBA live, other sports planned)

## Design Principle

**Hybrid: shared core engine + sport-specific variants.** Each sport runs as a separate registered strategy with its own allocation, halt switch, parameters, and dashboard view. All share the same green book simulation engine.

## Naming Convention

| File | Purpose | Strategy ID |
|------|---------|-------------|
| `arbo/strategies/strategy_d_core.py` | Shared engine (Kelly, exits, model prob, state) | — |
| `arbo/strategies/strategy_d_nba.py` | NBA variant | `"D"` |
| `arbo/strategies/strategy_d_ufc.py` | UFC variant (planned) | `"D_UFC"` |
| `arbo/strategies/strategy_d_nfl.py` | NFL variant (planned) | `"D_NFL"` |
| `arbo/strategies/strategy_d_epl.py` | EPL variant (planned) | `"D_EPL"` |
| `arbo/strategies/strategy_d_discovery_nba.py` | NBA market discovery | — |
| `arbo/strategies/strategy_d_discovery_ufc.py` | UFC market discovery (planned) | — |

Each sport-specific file has:
- Own class inheriting from `StrategyDCore`
- Own constants (`MIN_EDGE`, `GREEN_BOOK_DELTA`, `STOP_LOSS_DELTA`, `MAX_HOLD_FRACTION`, `BOTH_SIDES`)
- Own team abbreviation map
- Own `discover_markets()` via Gamma API with sport-specific tag

## Class Hierarchy

```
StrategyDCore (strategy_d_core.py)
├── generate_signals(markets, model_prob_fn) → [Signal]
├── execute_entry(signal) → Position | None
├── check_exits(prices) → [Position]
├── compute_model_prob(team_a, team_b, elo, pinnacle)
├── kelly_size(edge, price, capital)
└── get_status() → dict

    ├── StrategyDNba (strategy_d_nba.py)
    │   ├── SPORT_NAME = "nba"
    │   ├── STRATEGY_NAME = "D"
    │   └── Params: winner #25 from sweep v4
    │
    ├── StrategyDUfc (strategy_d_ufc.py)  [planned]
    │   ├── SPORT_NAME = "ufc"
    │   ├── STRATEGY_NAME = "D_UFC"
    │   └── Params: TBD after UFC sweep
    │
    └── StrategyDNfl (strategy_d_nfl.py)  [planned]
        ├── SPORT_NAME = "nfl"
        ├── STRATEGY_NAME = "D_NFL"
        └── Params: TBD after NFL sweep
```

## Per-Sport Parameters (Current / Planned)

### NBA (LIVE — paper mode)
```python
MIN_EDGE = 0.16
GREEN_BOOK_DELTA = 0.17
STOP_LOSS_DELTA = 0.15
MAX_HOLD_FRACTION = 0.50
BOTH_SIDES = True
ALLOCATION = $300  # Small sizing, optimization phase
```
**Backtest:** +$1,665 on $1K over 20 months, Sharpe 7.03, DD 13%, ROI ~100%/yr

### UFC (Planned — highest priority after NBA)
- Year-round season (40 events/year)
- **7,817 Pinnacle odds already in DB** (best data coverage of any sport)
- Spec: "Moderate" green book fit, "Excellent" overreaction fit
- Higher volatility per event (±20-40¢ per round) → delta likely higher (~0.20)
- Needs own sweep to find optimal params

### NFL (Planned — wait for season)
- Season: Sep-Feb
- 1,137 Pinnacle odds
- "Good" green book fit per spec
- 5-15¢ per score → medium delta (~0.15)

### EPL (Planned — off-season data gaps)
- Season: Aug-May
- Free Pinnacle via football-data.co.uk
- 15-30¢ per goal → delta similar to UFC
- Draws complicate 2-way bet sizing

## Why Separate Strategy IDs (Not One "D")

| Reason | Detail |
|--------|--------|
| **Independent allocations** | Risk manager `STRATEGY_ALLOCATIONS[{"D": 300, "D_UFC": 200}]` — each sport has own capital pool |
| **Independent halt switches** | If NBA goes rogue, halt just D — UFC keeps running. Weekly drawdown tracked per-sport |
| **Sport-specific sweeps** | Autoresearch for each sport runs independently. UFC sweep won't corrupt NBA results |
| **Dashboard clarity** | Per-sport P&L, WR, Sharpe — no lumping together |
| **Consistent with existing patterns** | B3 / B3_15M split (same mechanism, different timeframe). C / C2 split (same domain, different model) |
| **Easy to disable one sport** | Remove from `STRATEGY_ALLOCATIONS`, kill switch via env var |

## Shared vs Sport-Specific

### Shared in `strategy_d_core.py`
- `MarketData` dataclass (generic fields)
- `Signal` dataclass
- `Position` dataclass
- Kelly sizing formula
- Green book / stop loss / time exit walk logic
- P&L calculation for YES/NO sides
- Model probability ensemble (Elo + Pinnacle weighted)
- Risk manager integration
- Paper engine / live executor dispatch
- Status reporting

### Sport-Specific (in each `strategy_d_<sport>.py`)
- Parameters (constants)
- Team abbreviation maps
- `discover_markets()` via Gamma API tag
- Model data loading (which Elo/Pinnacle subset)
- Sport-specific keyword filters (e.g., "1H Moneyline" skip for NBA)
- Expected game duration (for time exit — NBA ~2.5h, UFC ~1h, NFL ~3h, EPL ~2h)

## Adding a New Sport — Checklist

1. Create `arbo/strategies/strategy_d_<sport>.py`:
   - Inherit from `StrategyDCore`
   - Override `SPORT_NAME`, `STRATEGY_NAME`
   - Set params (from sport's sweep results)
   - Provide team map
2. Create `arbo/strategies/strategy_d_discovery_<sport>.py`:
   - Implement `discover_markets(gamma_client)` with sport tag
   - Implement `_parse_teams()` with sport-specific patterns
3. Add to `arbo/core/risk_manager.py` `STRATEGY_ALLOCATIONS`
4. Add to `arbo/main_rdh.py`:
   - `_strategy_d_<sport>: Any = None`
   - `_init_strategy_d_<sport>()` method
   - Task entries in `_start_strategy_tasks()`
   - `_run_strategy_d_<sport>()` + exit check
5. Add to `arbo/dashboard/web.py`:
   - `_STRATEGY_META["D_<SPORT>"] = {...}`
   - `/api/strategy-d-<sport>` endpoint (or extend generic one)
6. Add tab to `arbo/dashboard/templates/dashboard.html`
7. Update model cache:
   - Extend `scripts/build_strategy_d_model.py` to also emit `arbo/data/strategy_d_<sport>_model.json`
   - Rebuild on VPS
8. Deploy + restart service

## Dashboard

Top-level **"D"** tab in dashboard shows:
- Per-sport breakdown (NBA, UFC, NFL, EPL) in sub-tabs or stacked cards
- Combined P&L across all D sports
- Overall green book rate, win rate
- Per-sport drill-down (positions, recent trades, params)

## Version History

| Date | Event |
|------|-------|
| 2026-04-05 | NBA live (paper) — first Strategy D variant |
| 2026-04-12 | Multi-sport architecture decided + refactored |
| TBD | UFC variant added |
| TBD | NFL variant (next NFL season) |
