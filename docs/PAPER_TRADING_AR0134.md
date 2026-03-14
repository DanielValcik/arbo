# Strategy C Paper Trading — Model AR-0134

> Deployed: 2026-03-14 | Commit: `435251f` | Status: ACTIVE

## Přehled

Model AR-0134 je nasazený na VPS v paper trading režimu. Byl vybrán z 6,000+
autoresearch experimentů jako nejrobustnější kandidát pro live validaci.

### Proč AR-0134?

| Kandidát | Score | DD% | NYC override | Kelly cap | WF PnL |
|----------|-------|-----|-------------|-----------|--------|
| AR-0770 | 171.2 | 9.6 | 4 parametry | 0.15 | $2,172 |
| AR-0691 | 171.0 | 9.5 | 3 parametry | 0.15 | $2,005 |
| **AR-0134** | **170.1** | **13.0** | **žádný** | **0.15** | **$2,218** |
| AR-0037 | 169.6 | 16.5 | žádný | 0.40 | $3,220 |

- **Nejjednodušší model** — žádný NYC-specifický override = nejmenší riziko overfittingu
- Score 170.1 je statisticky nerozlišitelný od nejlepšího (171.2)
- Nejvyšší walk-forward PnL z konzervativních modelů ($2,218)
- Konzervativní sizing (kelly_raw_cap=0.15) — bezpečný pro validaci

## Parametry

### Quality Gate (`arbo/strategies/weather_quality_gate.py`)

| Parametr | Hodnota | Předchozí | Změna |
|----------|---------|-----------|-------|
| MIN_EDGE | 0.10 | 0.08 | Přísnější filtr |
| MAX_EDGE | 0.70 | 0.42 | Širší povolený rozsah |
| MIN_PRICE | 0.05 | 0.30 | Povoleny nízké ceny |
| MAX_PRICE | 0.70 | 0.43 | Povoleny vysoké ceny |
| MIN_VOLUME | $510 | $1,000 | Nižší práh |
| MIN_FORECAST_PROB | 0.06 | 0.62 | Výrazně nižší práh |

### Vyloučená města

| Aktuální (AR-0134) | Předchozí |
|---------------------|-----------|
| Chicago, Seoul | NYC, Toronto, Buenos Aires, Atlanta, Wellington |

### Per-city overrides

| Město | max_price | min_edge | min_price |
|-------|-----------|----------|-----------|
| Ankara | 0.55 | 0.005 | 0.08 |
| Atlanta | 0.55 | 0.02 | 0.05 |
| Buenos Aires | 0.70 | 0.02 | 0.05 |
| Dallas | 0.80 | 0.05 | 0.05 |
| Miami | 0.40 | 0.05 | 0.08 |
| Seattle | 0.55 | 0.005 | 0.15 |
| Toronto | 0.40 | 0.005 | 0.05 |
| Wellington | 0.50 | 0.02 | 0.05 |

### Pravděpodobnostní model (`arbo/strategies/weather_scanner.py`)

| Parametr | Hodnota | Předchozí |
|----------|---------|-----------|
| prob_sharpening | 0.90 | 1.05 |
| shrinkage | 0.03 | 0.03 |

### Sizing (`arbo/strategies/weather_ladder.py`)

| Parametr | Hodnota | Předchozí |
|----------|---------|-----------|
| KELLY_FRACTION | 0.25 | 0.25 |
| kelly_raw_cap | 0.15 | 0.40 |

## Backtest výsledky

### Train (≤ leden 2026, 1,058 událostí)

| Metrika | Hodnota |
|---------|---------|
| Score | 170.1 / 200 |
| Trades | 273 |
| Win Rate | 43.6% |
| PnL | $27,829 |
| ROI | 2,783% |
| Max Drawdown | 13.0% |
| Sharpe | 9.9 |

### Out-of-Sample (> leden 2026, 671 událostí)

| Metrika | Hodnota |
|---------|---------|
| Trades | 175 |
| Win Rate | 38.2% |
| PnL | $297 |
| Sharpe | 2.2 |

### Walk-Forward (3 foldy)

| Fold | PnL | Trades |
|------|-----|--------|
| 1 | $1,358 | 43 |
| 2 | $3,419 | 82 |
| 3 | $1,738 | 85 |
| **Celkem** | **$2,218** | **210** |

### Per-city PnL (train)

| Město | PnL | Trades | WR% |
|-------|-----|--------|-----|
| NYC | $7,364 | 51 | 56.9% |
| Atlanta | $5,257 | 36 | 41.7% |
| Buenos Aires | $3,357 | 46 | 37.0% |
| London | $3,109 | 76 | 32.9% |
| Dallas | $3,005 | 17 | 47.1% |
| Toronto | $2,720 | 25 | 36.0% |
| Seattle | $1,087 | 5 | 80.0% |
| Wellington | $970 | 8 | 62.5% |
| Ankara | $606 | 5 | 60.0% |
| Miami | $354 | 3 | 66.7% |

## Známá rizika

### 1. Citlivost prob_sharpening (VYSOKÁ)
- ±20% změna → -18 bodů score
- Inherentní problém, ensemble sharpening (median i mean) nepomáhá
- **Mitigace**: Měsíční rekalibrce na nových datech

### 2. NYC koncentrace (STŘEDNÍ)
- 26.5% celkového PnL z jednoho města
- **Mitigace**: `max_exposure` parametr implementován v backtestu, použitelný v live

### 3. Sezónnost (NÍZKÁ)
- Dec+Jan = 67% PnL (větší teplotní variance v zimě)
- Očekávané chování, ne bug

## Jak vyhodnocovat

### Co sledovat
1. **Win rate** — backtest: 43.6%, OOS: 38.2%
2. **Edge distribuce** — průměrný edge ~0.15
3. **Trades/den** — backtest: 2-3 trades denně
4. **Per-city PnL** — NYC by mělo být top performer
5. **Drawdown** — backtest max DD: 13%

### Minimální doba hodnocení
4 týdny (architekturní pravidlo `MIN_PAPER_WEEKS = 4` v `risk_manager.py`)

### Dashboard
```
https://18.135.109.36:8080/
```

### Logy
```bash
ssh ubuntu@18.135.109.36 -i ~/.ssh/lightsail-london.pem "sudo journalctl -u arbo -f"
```

## Operační příkazy

### Reset Strategy C dat (čistý start)
```bash
ssh arbo "cd /opt/arbo && .venv/bin/python3 scripts/reset_strategy_c.py"
ssh ubuntu@18.135.109.36 -i ~/.ssh/lightsail-london.pem "sudo systemctl restart arbo"
```

### Dry-run reset (jen ukáže co se smaže)
```bash
ssh arbo "cd /opt/arbo && .venv/bin/python3 scripts/reset_strategy_c.py --dry-run"
```

### Deploy nového kódu
```bash
# 1. Commit + push lokálně
git add -A && git commit -m "..." && git push origin main

# 2. Pull na VPS
ssh arbo "cd /opt/arbo && git fetch origin main && git reset --hard origin/main"

# 3. (Volitelné) Reset dat
ssh arbo "cd /opt/arbo && .venv/bin/python3 scripts/reset_strategy_c.py"

# 4. Restart
ssh ubuntu@18.135.109.36 -i ~/.ssh/lightsail-london.pem "sudo systemctl restart arbo"
```

## Soubory

| Soubor | Účel |
|--------|------|
| `arbo/strategies/weather_quality_gate.py` | Globální thresholds + per-city overrides |
| `arbo/strategies/weather_scanner.py` | Prob model (sharpening, shrinkage, sigma, bias) |
| `arbo/strategies/weather_ladder.py` | Kelly sizing (kelly_raw_cap) |
| `arbo/strategies/strategy_c.py` | Hlavní orchestrace poll cyklu |
| `arbo/core/risk_manager.py` | KELLY_FRACTION, alokace, limity |
| `scripts/reset_strategy_c.py` | Reset paper dat pro čistý start |
| `research/data/experiments/autoresearch_latest.json` | Všech 6,000+ experimentů |
| `research/AUTORESEARCH_REPORT.md` | Report V1 (293 experimentů) |

## Historie autoresearch

| Generace | Datum | Experimentů | Nejlepší score | Klíčová změna |
|----------|-------|-------------|----------------|---------------|
| V1 | 2026-03-11 | 293 | 168.7 | Quality gate + per-city sigma |
| V2 | 2026-03-14 | 865 | 105.5 | Quarter-Kelly fixed, jiný scoring |
| V3 | 2026-03-14 | 6,000+ | 171.2 | OOS-aware search, per-city capital |
