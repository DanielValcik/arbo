# Data Pipeline Guide — PolymarketData.co

> Pro development team. Kde najít data, jak s nimi pracovat, jak spustit/restartovat.
> Aktualizováno: 2026-03-16

---

## Přehled

Stahujeme kompletní historická cenová data z Polymarketu přes [PolymarketData.co](https://www.polymarketdata.co) API (Ultra tier, $360/měsíc). Dvě paralelní pipeline:

| Pipeline | Co stahuje | Kde běží | DB soubor | Rozlišení |
|----------|-----------|----------|-----------|-----------|
| **Strategy D (sport)** | 150K sportovních marketů (NBA, EPL, NFL, La Liga, Serie A, UCL, UFC, MLB, NHL, F1, NCAAB, boxing) | Download VPS (18.134.241.60) | `research_d/data/sports_backtest.sqlite` | 10-min |
| **Strategy C (počasí)** | ~18.7K teplotních marketů, 20 měst | Lokální stroj | `research/data/weather_pmd.sqlite` | 1-min |

---

## API Přístup

- **API klíč**: `POLYMARKETDATA_API_KEY` v `.env`
- **Base URL**: `https://api.polymarketdata.co/v1`
- **Auth**: Header `X-API-Key: pk_live_xxx`
- **Plan**: Ultra (1-min rozlišení, neomezená historie, 2000 RPM)
- **Dokumentace**: https://www.polymarketdata.co/docs

### Klíčové endpointy

```
GET /v1/markets?tags=nba&limit=1000         — Discovery marketů
GET /v1/markets/{id}/prices?start_ts=X&end_ts=Y&resolution=1m  — Cenová historie
GET /v1/tokens/{id}/prices?...              — Ceny jednoho tokenu
GET /v1/markets/{id}/books?...              — Orderbook historie
GET /v1/tags                                — Dostupné tagy
GET /v1/usage                               — Info o plánu a kvótě
```

---

## Strategy D — Sportovní data

### Kde běží
- **Download VPS**: `18.134.241.60` (4 vCPU, 16 GB RAM, 320 GB disk, $84/mo)
- **SSH**: `ssh -i ~/.ssh/lightsail-london.pem arbo@18.134.241.60`
- **Screen**: `screen -r pmd` (hlavní session)
- **Workery**: 2-3 paralelní Python procesy

### Soubory na download VPS
```
/opt/arbo/
├── research_d/data/
│   ├── sports_backtest.sqlite       — Hlavní DB (~15+ GB, roste)
│   ├── pmd_cache/
│   │   └── markets_all.json         — Cache 150K marketů z API discovery
│   ├── pmd_chunks/
│   │   ├── chunk_0.json             — Market IDs pro Worker 0
│   │   ├── chunk_1.json             — Market IDs pro Worker 1
│   │   └── chunk_2.json             — Market IDs pro Worker 2
│   ├── pmd_worker.py                — Worker skript (generovaný)
│   ├── pmd_worker_0.log             — Log Worker 0
│   ├── pmd_worker_1.log             — Log Worker 1
│   ├── pmd_worker_2.log             — Log Worker 2
│   ├── pmd_progress.json            — Legacy progress (nespolehlivý)
│   ├── pmd_progress_log.txt         — Append-only progress (spolehlivý)
│   ├── download_status.json         — Status pro dashboard (cron každou minutu)
│   └── update_progress.sh           — Skript generující download_status.json
```

### Monitorování

```bash
# Quick status
ssh -i ~/.ssh/lightsail-london.pem arbo@18.134.241.60 \
    "cat /opt/arbo/research_d/data/download_status.json"

# Worker logy
ssh -i ~/.ssh/lightsail-london.pem arbo@18.134.241.60 \
    "tail -5 /opt/arbo/research_d/data/pmd_worker_0.log"

# Kolik workerů běží
ssh -i ~/.ssh/lightsail-london.pem arbo@18.134.241.60 \
    "ps aux | grep pmd_worker | grep -v grep | wc -l"

# Disk
ssh -i ~/.ssh/lightsail-london.pem arbo@18.134.241.60 "df -h /"

# Dashboard
# Karta "Strategy D — Stahovani sportovnich dat" na arbo.click
```

### Restart workerů (pokud spadnou)

```bash
# SSH na download VPS
ssh -i ~/.ssh/lightsail-london.pem arbo@18.134.241.60

# Restart jednoho workeru
cd /opt/arbo
nohup /opt/arbo/.venv/bin/python3 research_d/data/pmd_worker.py \
    research_d/data/pmd_chunks/chunk_0.json 0 \
    > research_d/data/pmd_worker_0.log 2>&1 &

# Restart všech tří
for i in 0 1 2; do
    nohup /opt/arbo/.venv/bin/python3 research_d/data/pmd_worker.py \
        research_d/data/pmd_chunks/chunk_$i.json $i \
        > research_d/data/pmd_worker_$i.log 2>&1 &
done
```

### SQLite schema (sports)

```sql
-- Markety
CREATE TABLE markets (
    token_id TEXT PRIMARY KEY, game_id TEXT, event_id TEXT,
    condition_id TEXT, token_id_no TEXT, question TEXT,
    outcome TEXT, volume REAL, neg_risk INTEGER, won INTEGER,
    end_date TEXT, extra_json TEXT
);

-- Ceny (hlavní tabulka — miliony řádků)
CREATE TABLE prices (
    token_id TEXT NOT NULL, ts INTEGER NOT NULL, price REAL NOT NULL,
    PRIMARY KEY (token_id, ts)
);

-- Hry
CREATE TABLE games (game_id TEXT PRIMARY KEY, sport TEXT, league TEXT,
    home_team TEXT, away_team TEXT, game_date TEXT, ...);

-- Pinnacle odds, Elo ratings, game events — viz sports_db.py
```

### Po dokončení Pass 1

1. Spustit Pass 2 (1-min game window pro moneyline + spread):
```bash
cd /opt/arbo
PYTHONPATH=. .venv/bin/python3 research_d/download_pmd_parallel.py \
    --workers 3 --resolution 1m --game-window-hours 48 --market-type moneyline
```

2. Zkopírovat SQLite zpět na hlavní VPS:
```bash
scp -i ~/.ssh/lightsail-london.pem \
    arbo@18.134.241.60:/opt/arbo/research_d/data/sports_backtest.sqlite \
    /opt/arbo/research_d/data/sports_backtest.sqlite
```

3. Smazat download VPS (úspora $84/mo):
```bash
aws lightsail delete-instance --instance-name arbo-download --region eu-west-2
```

---

## Strategy C — Data počasí

### Kde běží
- **Lokální stroj** (MacBook) — nezávislé na sportovním downloadu
- **Proces**: `python3 research/download_weather_pmd.py`

### Soubory
```
research/data/
├── weather_pmd.sqlite           — Dedikovaná DB pro počasí (~5 GB odhad)
├── weather_pmd_cache/
│   ├── nyc.json                 — Cache marketů pro NYC
│   ├── london.json              — Cache marketů pro London
│   └── ...                      — 20 souborů (jedno per město)
├── weather_pmd_progress.txt     — Append-only progress
├── weather_status.json          — Status pro dashboard
└── weather_pmd_download.log     — Log běžícího downloadu
```

### Spuštění / restart

```bash
# Spustit (resume automaticky přeskakuje hotové markety)
PYTHONPATH=. python3 research/download_weather_pmd.py --city all --resolution 1m

# Jen jedno město
PYTHONPATH=. python3 research/download_weather_pmd.py --city london --resolution 1m

# Na pozadí
PYTHONPATH=. nohup python3 research/download_weather_pmd.py --city all --resolution 1m \
    > research/data/weather_pmd_download.log 2>&1 &
```

### SQLite schema (počasí)

```sql
CREATE TABLE markets (
    market_id TEXT PRIMARY KEY, city TEXT, question TEXT,
    status TEXT, start_date TEXT, end_date TEXT, tokens_json TEXT
);

CREATE TABLE prices (
    token_id TEXT NOT NULL, ts INTEGER NOT NULL, price REAL NOT NULL,
    PRIMARY KEY (token_id, ts)
);
```

### 20 měst

NYC, Chicago, London, Seoul, Buenos Aires, Atlanta, Toronto, Ankara,
São Paulo, Miami, Paris, Dallas, Seattle, Wellington, Tokyo, Munich,
Los Angeles, Washington DC, Tel Aviv, Lucknow

---

## Dashboard monitoring

Dashboard na `arbo.click` ukazuje dvě karty:

1. **Strategy D — Stahovani sportovnich dat** (modrý progress bar)
   - Data se synchronizují z download VPS přes cron (každou minutu)

2. **Strategy C — Stahovani dat pocasi** (oranžový progress bar)
   - Data se synchronizují ručně z lokálu (`research/data/sync_weather_status.sh`)
   - Nebo: `bash research/data/update_weather_status.sh && scp -i ~/.ssh/lightsail-london.pem research/data/weather_status.json arbo@18.135.109.36:/opt/arbo/research/data/weather_status.json`

---

## Práce s daty (Python)

### Sportovní data

```python
import sqlite3

db = sqlite3.connect("research_d/data/sports_backtest.sqlite")
db.row_factory = sqlite3.Row

# Počet cen
total = db.execute("SELECT COUNT(*) FROM prices").fetchone()[0]

# Ceny pro konkrétní token
prices = db.execute(
    "SELECT ts, price FROM prices WHERE token_id = ? ORDER BY ts",
    (token_id,)
).fetchall()

# Markety pro NBA
nba = db.execute(
    "SELECT * FROM markets m JOIN games g ON m.game_id = g.game_id WHERE g.sport = 'nba'"
).fetchall()
```

### Weather data

```python
db = sqlite3.connect("research/data/weather_pmd.sqlite")
db.row_factory = sqlite3.Row

# Markety pro London
london = db.execute(
    "SELECT * FROM markets WHERE city = 'london'"
).fetchall()

# Ceny s 1-min rozlišením
prices = db.execute(
    "SELECT ts, price FROM prices WHERE token_id = ? ORDER BY ts",
    (token_id,)
).fetchall()
```

---

## Náklady

| Položka | Cena | Poznámka |
|---------|------|----------|
| PolymarketData Ultra | $360/mo | Storno po stažení (1 měsíc) |
| Download VPS (arbo-download) | $84/mo (pro-rated) | Smazat po dokončení (~$14 za 5 dní) |
| Hlavní VPS (arbo-london) | $20/mo | Běží stále |
| **Celkem jednorázově** | **~$394** | |
| **Ongoing (po stažení)** | **$20/mo** | Jen hlavní VPS |

---

## Časový plán

1. **Sports Pass 1** (10-min, všechny markety): ~1.5 dne → ~17. března
2. **Sports Pass 2** (1-min, moneyline+spread game window): ~0.5 dne
3. **Weather** (1-min, 20 měst): ~12 hodin
4. **Po stažení**: backtest, sweep, walk-forward, autoresearch
