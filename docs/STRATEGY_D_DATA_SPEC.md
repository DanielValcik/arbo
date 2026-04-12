# Strategy D — Sports Data Specification

> Prepared: 2026-03-29
> Status: DATA COMPLETE — ready for Strategy D implementation

## Location

| What | Where |
|---|---|
| **Server** | `arbo-download` (AWS Lightsail eu-west-2a) |
| **IP** | 13.41.250.48 (static) |
| **SSH** | `ssh -i ~/.ssh/lightsail-london.pem arbo@13.41.250.48` |
| **DB file** | `/mnt/arbo-data/sports_backtest.sqlite` |
| **Symlink** | `/opt/arbo/research_d/data/sports_backtest.sqlite → /mnt/arbo-data/` |
| **DB size** | ~312 GB |
| **Disk** | 500 GB block storage mounted at `/mnt/arbo-data` (176 GB free) |

## Data Source

- **PolymarketData.co** (Ultra tier, paid API)
- API key in `.env` as `POLYMARKETDATA_API_KEY`
- Client: `research_d/download_polymarketdata.py` → `PMDataClient`
- DB helper: `research_d/sports_db.py` → `SportsDB`

## Database Schema

### `games` — 96,006 rows

Individual sports matches.

| Column | Type | Description |
|---|---|---|
| `game_id` | TEXT PK | Unique ID (e.g. `nba_pmd_643418`) |
| `sport` | TEXT | Sport category: `nba`, `nfl`, `ufc`, `nhl`, `f1`, `ucl`, `soccer`, `epl`, `boxing`, `mlb`, `unknown` |
| `league` | TEXT | League name (e.g. `NBA`, `UFC`, `NFL`) |
| `home_team` | TEXT | Home team (often `HOME` — actual name in `extra_json`) |
| `away_team` | TEXT | Away team (often `AWAY` — actual name in `extra_json`) |
| `game_date` | TEXT | Date string `YYYY-MM-DD` |
| `game_time` | TEXT | Time (usually NULL) |
| `home_score` | INTEGER | Final score (NULL if not resolved) |
| `away_score` | INTEGER | Final score (NULL if not resolved) |
| `status` | TEXT | `final`, `scheduled`, etc. |
| `season` | TEXT | Season (usually NULL) |
| `venue` | TEXT | Venue (usually NULL) |
| `extra_json` | TEXT | JSON with source details: `source`, `pmd_id`, `slug` |

### `markets` — 254,504 rows

Polymarket binary contracts linked to games.

| Column | Type | Description |
|---|---|---|
| `token_id` | TEXT PK | CLOB token ID (YES outcome) |
| `token_id_no` | TEXT | CLOB token ID (NO outcome, can be NULL) |
| `game_id` | TEXT FK | Links to `games.game_id` |
| `event_id` | TEXT | Polymarket event ID |
| `condition_id` | TEXT | Polymarket condition ID |
| `question` | TEXT | Market question (e.g. "Will the Golden State Warriors win the NBA Cup?") |
| `outcome` | TEXT | Outcome label: `yes`, `no`, `over`, `under`, team names, etc. |
| `volume` | REAL | Total traded volume |
| `neg_risk` | INTEGER | NegRisk flag (0 = standard, 1 = NegRisk) |
| `won` | INTEGER | Resolution: 1 = won, 0 = lost, NULL = unresolved |
| `end_date` | TEXT | Market end date |
| `extra_json` | TEXT | Additional metadata JSON |

### `prices` — ~958 million rows

Minute-level Polymarket price history. **This is the core data.**

| Column | Type | Description |
|---|---|---|
| `token_id` | TEXT FK | Links to `markets.token_id` |
| `ts` | INTEGER | Unix timestamp |
| `price` | REAL | Mid price (0.0 - 1.0) |
| `bid` | REAL | Best bid (NULL for 10-min data) |
| `ask` | REAL | Best ask (NULL for 10-min data) |
| `volume_1m` | REAL | 1-minute volume (NULL for 10-min data) |

**Two download passes:**
- **Pass 1** (10-min resolution): All 150,348 markets. `bid`/`ask`/`volume_1m` = NULL
- **Pass 2** (1-min resolution, 48h game window): ~86,690 moneyline markets. `bid`/`ask` populated

### `game_events` — 0 rows (empty, schema ready)

Live scoring events during games.

| Column | Type | Description |
|---|---|---|
| `game_id` | TEXT FK | Links to `games.game_id` |
| `ts` | INTEGER | Unix timestamp |
| `event_type` | TEXT | Event category |
| `team` | TEXT | Team involved |
| `score_home` | INTEGER | Running score |
| `score_away` | INTEGER | Running score |
| `detail` | TEXT | Event description |

### `pinnacle_odds` — 0 rows (empty, schema ready)

Pinnacle no-vig odds history.

| Column | Type | Description |
|---|---|---|
| `game_id` | TEXT FK | Links to `games.game_id` |
| `ts` | INTEGER | Unix timestamp |
| `home_odds` | REAL | Pinnacle home decimal odds |
| `away_odds` | REAL | Pinnacle away decimal odds |
| `draw_odds` | REAL | Pinnacle draw odds (if applicable) |
| `home_prob_novig` | REAL | De-vigged home probability |
| `away_prob_novig` | REAL | De-vigged away probability |

### `ratings` — 0 rows (empty, schema ready)

Elo/Glicko-2 team strength ratings.

| Column | Type | Description |
|---|---|---|
| `team` | TEXT | Team name |
| `sport` | TEXT | Sport |
| `date` | TEXT | Rating date |
| `elo` | REAL | Elo rating |
| `glicko_rating` | REAL | Glicko-2 rating |
| `glicko_rd` | REAL | Glicko-2 rating deviation |
| `glicko_vol` | REAL | Glicko-2 volatility |

## Data Breakdown by Sport

| Sport | Games | Markets | Notes |
|---|---|---|---|
| `unknown` | 92,943 | 247,940 | Unclassified (politics, crypto, entertainment, etc.) |
| `nba` | 724 | 1,448 | NBA regular season + playoffs |
| `ufc` | 665 | 1,514 | UFC fight cards |
| `nfl` | 571 | 1,142 | NFL regular season + playoffs |
| `nhl` | 498 | 996 | NHL |
| `f1` | 265 | 530 | Formula 1 |
| `ucl` | 161 | 454 | UEFA Champions League |
| `soccer` | 90 | 258 | Other soccer |
| `boxing` | 52 | 104 | Boxing |
| `epl` | 36 | 116 | English Premier League |
| `mlb` | 1 | 2 | MLB (minimal) |

## Market Types (by question pattern)

| Type | Count | Description |
|---|---|---|
| `other` | 145,906 | Winner markets, custom props, event outcomes |
| `spread` | 50,016 | Point spread markets |
| `moneyline` | 29,218 | Will team X win/beat Y |
| `over_under` | 19,096 | Total points over/under |
| `player_prop` | 10,268 | Player-level propositions (score X points, etc.) |

## Date Range

- **Games**: 2025-01-01 to 2027-03-31 (includes scheduled future games)
- **Prices**: Oct 2025 — Mar 2026 (active trading period)

## How to Query

```python
from research_d.sports_db import SportsDB

db = SportsDB()  # uses default path: research_d/data/sports_backtest.sqlite

# Or with custom path:
db = SportsDB("/mnt/arbo-data/sports_backtest.sqlite")
```

### Direct SQLite access:

```python
import sqlite3

conn = sqlite3.connect("/mnt/arbo-data/sports_backtest.sqlite")

# All NBA moneyline markets with 1-min prices
nba_ml = conn.execute("""
    SELECT m.token_id, m.question, m.outcome, m.won, g.game_date
    FROM markets m
    JOIN games g ON m.game_id = g.game_id
    WHERE g.sport = 'nba'
    AND m.question LIKE '%win%'
""").fetchall()

# Price series for a market (1-min resolution with bid/ask)
prices = conn.execute("""
    SELECT ts, price, bid, ask, volume_1m
    FROM prices
    WHERE token_id = ?
    AND bid IS NOT NULL
    ORDER BY ts
""", (token_id,)).fetchall()

# Price series (10-min resolution, all markets)
prices_10m = conn.execute("""
    SELECT ts, price
    FROM prices
    WHERE token_id = ?
    AND bid IS NULL
    ORDER BY ts
""", (token_id,)).fetchall()
```

## Key Notes for Strategy D Implementation

1. **Team names** are often `HOME`/`AWAY` in `games` table — actual names are in `extra_json` (JSON `slug` field) and in `markets.question`/`markets.outcome`
2. **Pass 2 data** (1-min with bid/ask) covers moneyline markets only, 48h window around game time
3. **Pass 1 data** (10-min, price only) covers ALL markets from creation to resolution
4. **`pinnacle_odds`** and **`ratings`** tables are empty — need separate data collection (The Odds API for Pinnacle, custom Elo/Glicko computation)
5. **`game_events`** empty — need live scoring feed (ESPN API, etc.)
6. **`won` column** is NULL for all markets — resolution data needs backfill
7. **`unknown` sport** (93K games) includes non-sports markets that leaked into the dataset — filter by known sports for Strategy D
8. **DB is on block storage** — read performance is good but sequential scans on 958M rows take minutes. Use indexed queries (token_id, game_id)

## Related Files

| File | Description |
|---|---|
| `research_d/sports_db.py` | SportsDB class — schema + CRUD helpers |
| `research_d/download_polymarketdata.py` | PMDataClient — API wrapper |
| `research_d/data/pass2_worker_sharded.py` | Pass 2 download workers (moneyline 1-min) |
| `research_d/download_sports_prices.py` | Pass 1 download script |
| `research_d/data/pmd_cache/markets_all.json` | Cached market catalog |
| `research_d/data/pmd_pass2_progress_moneyline_*.txt` | Download progress tracking |
| `docs/STRATEGY_D_SPEC.md` | Strategy D design spec |

## TODO for Strategy D

- [ ] Backfill `won` column from Polymarket resolution data
- [ ] Populate `pinnacle_odds` from The Odds API historical data
- [ ] Compute Elo/Glicko-2 `ratings` from game results
- [ ] Populate `game_events` from ESPN/sports API
- [ ] Classify `unknown` games into proper sports categories (or filter out)
- [ ] Index optimization for backtest queries
