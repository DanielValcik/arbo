# Polymarket Price History — Data Pipeline

> Datum: 2026-03-14
> Status: Aktivní sběr dat
> Zdroj: Polymarket CLOB API `/prices-history`

## Přehled

Stahujeme reálnou cenovou historii z Polymarket CLOB API pro všechny resolved weather temperature markety. Data ukládáme do SQLite databáze pro použití v realistických backtestech.

**Proč to děláme:** Původní backtest harness (`backtest_harness.py`) používal syntetické tržní ceny z umělého market makera. To vedlo k nerealistickým výsledkům (98% win rate vs 67% v reálném paper tradingu). Reálná cenová data z Polymarketu umožňují testovat strategii na skutečných podmínkách.

## Architektura

```
Polymarket CLOB API                    Open-Meteo Archive
  /prices-history                        /v1/archive
       │                                      │
       ▼                                      ▼
┌─────────────┐                    ┌──────────────────┐
│  price_      │                    │  Historické       │
│  history.    │                    │  teploty (2yr)    │
│  sqlite      │                    │                   │
└──────┬───────┘                    └────────┬──────────┘
       │                                      │
       ▼                                      ▼
┌─────────────────────────────────────────────────┐
│         backtest_real_prices.py                   │
│                                                   │
│  Pro každý resolved event:                        │
│  1. Načti reálnou cenu N hodin před close         │
│  2. Načti Open-Meteo forecast pro den resolution  │
│  3. Spočítej edge = naše_prob - tržní_cena        │
│  4. Aplikuj quality gate + sizing                 │
│  5. Vyhodnoť: won/lost na základě reálného       │
│     výsledku (který bucket vyhrál)                │
└─────────────────────────────────────────────────┘
```

## Soubory

| Soubor | Účel |
|--------|------|
| `research/download_price_history.py` | Stažení dat z Polymarket API → SQLite |
| `research/price_history_db.py` | Python query interface (PriceHistoryDB) |
| `research/backtest_real_prices.py` | Realistický backtest na stažených datech |
| `research/data/price_history.sqlite` | SQLite databáze s cenovou historií |
| `research/data/price_history_meta.json` | Metadata o stažení |
| `research/data/polymarket_weather_events.json` | Raw events z Gamma API (53 MB) |

## Jak stáhnout data

```bash
# Stáhnout/aktualizovat cenovou historii (~ 10 min)
python3 research/download_price_history.py

# Obnovit i seznam eventů z Gamma API
python3 research/download_price_history.py --refresh-events

# Jemnější granularita (30 min místo 60)
python3 research/download_price_history.py --fidelity 30
```

**Důležité:** Polymarket uchovává cenovou historii jen ~30 dní po close marketu. Skript je nutné spouštět pravidelně, aby se data stihla stáhnout.

## Databázové schéma

### Tabulka `events`
| Sloupec | Typ | Popis |
|---------|-----|-------|
| event_id | TEXT PK | Polymarket event ID |
| title | TEXT | "Highest temperature in Paris on March 8?" |
| city | TEXT | Normalizované ID ("paris", "chicago", ...) |
| target_date | TEXT | Datum resolution (ISO: "2026-03-08") |
| start_date | TEXT | Kdy market začal |
| end_date | TEXT | Kdy market končí |
| closed_time | TEXT | Kdy byl market uzavřen |
| volume | REAL | Celkový objem v USD |
| neg_risk | INTEGER | 1 = NegRisk market |
| n_buckets | INTEGER | Počet teplotních bucketů |
| resolution_src | TEXT | URL zdroje resolution (Weather Underground) |

### Tabulka `buckets`
| Sloupec | Typ | Popis |
|---------|-----|-------|
| token_id | TEXT PK | CLOB YES token ID |
| token_id_no | TEXT | CLOB NO token ID |
| event_id | TEXT FK | Reference na event |
| condition_id | TEXT | Polymarket condition ID |
| question | TEXT | "Will the high be between 8-9°C?" |
| low_c | REAL | Spodní hranice bucketu (°C) |
| high_c | REAL | Horní hranice bucketu (°C) |
| bucket_type | TEXT | "range", "below", "above", "exact" |
| unit | TEXT | "F" nebo "C" |
| won | INTEGER | 1 = tento bucket vyhrál |
| volume | REAL | Objem bucketu v USD |

### Tabulka `prices`
| Sloupec | Typ | Popis |
|---------|-----|-------|
| token_id | TEXT | CLOB YES token ID |
| ts | INTEGER | Unix timestamp |
| price | REAL | YES cena (0.0 - 1.0) |
| PK | | (token_id, ts) |

## Jak používat v kódu

```python
from price_history_db import PriceHistoryDB

db = PriceHistoryDB()

# Přehled dostupných dat
print(db.stats())
print(db.get_cities())

# Eventy pro konkrétní město
events = db.get_events(city="paris", min_date="2026-03-01")

# Cenový snapshot 24h před close (klíčové pro backtest)
snap = db.get_price_snapshot(event_id="12345", hours_before_close=24)
for bucket in snap.buckets:
    price = snap.prices.get(bucket.token_id)
    print(f"  {bucket.question}: YES @ {price:.4f}, won={bucket.won}")

# Všechny snapshoty pro město (pro walk-forward)
snapshots = db.get_all_snapshots(city="paris", hours_before_close=24)

# Kompletní cenová historie jednoho tokenu
history = db.get_price_history(token_id="6574066...")
for point in history:
    print(f"  {point.dt}: {point.price:.4f}")
```

## Jak spustit realistický backtest

```bash
python3 research/backtest_real_prices.py
```

Backtest testuje strategii na 5 různých entry timingách (48h, 36h, 24h, 12h, 6h před close) a ukazuje per-city výsledky.

## Srovnání: Syntetický vs Realistický backtest

| Aspekt | Syntetický (backtest_harness) | Realistický (backtest_real_prices) |
|--------|-------------------------------|-------------------------------------|
| Tržní ceny | Simulovaný market maker | **Reálné Polymarket CLOB** |
| Teploty | Open-Meteo archive | Open-Meteo archive |
| Resolution | Syntetické (z teplot) | **Reálné (který bucket vyhrál)** |
| Edge zdroj | Nižší noise modelu | **Skutečný edge vs trh** |
| Win rate | ~98% | **~60-70%** (realistické) |
| Dostupnost dat | 2 roky (2024-2025) | ~30 dní rolling |
| Počet eventů | Neomezený (generované) | ~350+ (resolved) |

## Omezení

1. **Retence ~30 dní**: Polymarket uchovává cenovou historii jen ~30 dní. Data starší než měsíc jsou nenávratně ztracena.
2. **Hodinová granularita**: Pro resolved markety je minimální granularita ~1h (fidelity=60).
3. **Jen YES ceny**: Stahujeme jen YES token prices (NO = 1 - YES pro NegRisk).
4. **Žádný orderbook depth**: Máme jen midpoint ceny, ne full L2 book.
5. **Forecast approximace**: Používáme Open-Meteo archive (pozorovaná teplota) místo skutečné předpovědi. Pro days_out=0-1 je to blízké, pro days_out=3+ méně přesné.

## Plán rozšíření

### Krátkodobý (nyní)
- [x] Skript na stažení cenové historie
- [x] SQLite databáze se schématem
- [x] Query interface (PriceHistoryDB)
- [x] Realistický backtest skript
- [ ] Automatické pravidelné stahování (cron na VPS)

### Střednědobý (2-4 týdny)
- [ ] Přebudovat autoresearch sweep na reálných datech
- [ ] Porovnat optimální parametry: syntetický vs realistický
- [ ] Integrovat s paper trading daty z PostgreSQL

### Dlouhodobý
- [ ] PolymarketData.co pro 1-min orderbook data (pokud zdarma nestačí)
- [ ] Vlastní data collector na VPS (live snapshot každou hodinu)
- [ ] 3+ měsíce reálných dat pro robustní optimalizaci

## Automatické stahování (VPS cron)

Pro pravidelné stahování dat na VPS přidej do crontabu:

```bash
# Stáhnout nová data každý den v 6:00 UTC
0 6 * * * cd /opt/arbo && python3 research/download_price_history.py --refresh-events >> /var/log/arbo/price_history.log 2>&1
```
