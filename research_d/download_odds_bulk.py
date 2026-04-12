"""Bulk Pinnacle odds download from The Odds API — all sports.

Paid tier ($10/mo, 20K credits). Each historical request = ~10 credits.
Downloads Pinnacle h2h odds for NBA, NFL, UFC, and stores in SportsDB.

EPL is NOT included — already covered by free football-data.co.uk.

Budget: 20,000 credits → ~2,000 requests.
Plan: NBA 3 seasons + NFL 1 season + UFC = ~1,250 requests = ~12,500 credits.

Usage:
    export ODDS_API_KEY=your_key
    PYTHONPATH=. python3 research_d/download_odds_bulk.py
    PYTHONPATH=. python3 research_d/download_odds_bulk.py --dry-run
    PYTHONPATH=. python3 research_d/download_odds_bulk.py --sport nba --budget 5000
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()
    SSL_CTX.check_hostname = False
    SSL_CTX.verify_mode = ssl.CERT_NONE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_d.sports_db import SportsDB

DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = DATA_DIR / "odds_api_cache"

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
CREDITS_PER_REQUEST = 10

# ── Sport configs ─────────────────────────────────────────────────────

SPORT_CONFIGS = {
    "nba": {
        "api_key": "basketball_nba",
        "sport_db": "nba",
        "league": "NBA",
        "seasons": [
            ("2025-26", "2025-10-22", "2026-04-02"),
            ("2024-25", "2024-10-22", "2025-06-22"),
            ("2023-24", "2023-10-24", "2024-06-17"),
        ],
    },
    "nfl": {
        "api_key": "americanfootball_nfl",
        "sport_db": "nfl",
        "league": "NFL",
        "seasons": [
            ("2025-26", "2025-09-04", "2026-02-09"),
        ],
    },
    "ufc": {
        "api_key": "mma_mixed_martial_arts",
        "sport_db": "ufc",
        "league": "UFC",
        "seasons": [
            ("2025-26", "2025-01-01", "2026-04-02"),
        ],
    },
}

# Team name → abbreviation maps
NBA_TEAMS = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

NFL_TEAMS = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WSH",
}

TEAM_MAPS = {"nba": NBA_TEAMS, "nfl": NFL_TEAMS, "ufc": {}}


def _team_abbr(name: str, sport: str) -> str:
    """Convert full team name to abbreviation."""
    m = TEAM_MAPS.get(sport, {})
    return m.get(name, name[:3].upper())


# ── Vig removal ───────────────────────────────────────────────────────

def _remove_vig(home: float, away: float) -> tuple[float, float]:
    if home <= 1 or away <= 1:
        return 0.5, 0.5
    ih, ia = 1 / home, 1 / away
    t = ih + ia
    return ih / t, ia / t


# ── API ───────────────────────────────────────────────────────────────

def _api_get(url: str) -> dict | list | None:
    """GET with retry."""
    for attempt in range(3):
        req = urllib.request.Request(url, headers={"User-Agent": "ArboOddsBulk/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
                # Track remaining credits from header
                remaining = resp.headers.get("x-requests-remaining", "?")
                used = resp.headers.get("x-requests-used", "?")
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5)
            elif e.code == 422:
                return None  # No data for this date
            elif e.code in (401, 403):
                print(f"  AUTH ERROR {e.code} — check ODDS_API_KEY", flush=True)
                return None
            else:
                if attempt < 2:
                    time.sleep(2)
                else:
                    return None
        except Exception:
            if attempt < 2:
                time.sleep(2)
            else:
                return None
    return None


def download_sport_historical(
    db: SportsDB,
    api_key: str,
    sport: str,
    start_date: str,
    end_date: str,
    budget_credits: int,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Download historical Pinnacle odds for one sport/season.

    Returns: (records_stored, credits_used)
    """
    cfg = SPORT_CONFIGS[sport]
    api_sport = cfg["api_key"]
    sport_db = cfg["sport_db"]

    current = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    # Don't go past today
    today = date.today()
    if end > today:
        end = today

    total_days = (end - current).days + 1
    max_requests = budget_credits // CREDITS_PER_REQUEST

    print(f"  {sport.upper()}: {start_date} → {end} ({total_days} days, max {max_requests} requests)", flush=True)

    if dry_run:
        print(f"  DRY RUN: would use ~{total_days} requests = ~{total_days * CREDITS_PER_REQUEST} credits", flush=True)
        return 0, 0

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stored = 0
    requests_made = 0

    while current <= end and requests_made < max_requests:
        date_iso = current.strftime("%Y-%m-%dT12:00:00Z")
        cache_path = CACHE_DIR / f"{sport}_{current.strftime('%Y%m%d')}.json"

        if cache_path.exists():
            with open(cache_path) as f:
                data = json.load(f)
        else:
            url = (
                f"{ODDS_API_BASE}/historical/sports/{api_sport}/odds"
                f"?apiKey={api_key}&regions=eu&markets=h2h"
                f"&bookmakers=pinnacle&date={date_iso}"
            )
            raw = _api_get(url)
            requests_made += 1

            if not raw:
                current += timedelta(days=1)
                time.sleep(0.5)
                continue

            # Cache
            with open(cache_path, "w") as f:
                json.dump(raw, f)
            data = raw
            time.sleep(1.0)  # Rate limit

        # Parse response
        events = data.get("data", []) if isinstance(data, dict) else data
        if not isinstance(events, list):
            current += timedelta(days=1)
            continue

        games_batch = []
        odds_batch = []

        for event in events:
            commence = event.get("commence_time", "")
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            if not commence or not home or not away:
                continue

            try:
                dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                gdate = dt.strftime("%Y-%m-%d")
                gtime = dt.strftime("%H:%M")
                ts = int(dt.timestamp())
            except Exception:
                continue

            h_abbr = _team_abbr(home, sport)
            a_abbr = _team_abbr(away, sport)
            game_id = f"{sport_db}_{gdate.replace('-', '')}_{a_abbr}_{h_abbr}"

            games_batch.append({
                "game_id": game_id, "sport": sport_db, "league": cfg["league"],
                "home_team": h_abbr, "away_team": a_abbr,
                "game_date": gdate, "game_time": gtime,
                "home_score": None, "away_score": None,
                "status": "scheduled", "season": None, "venue": None,
                "extra_json": json.dumps({"source": "odds_api_hist", "home_full": home, "away_full": away}),
            })

            for bm in event.get("bookmakers", []):
                if bm.get("key") != "pinnacle":
                    continue
                for mkt in bm.get("markets", []):
                    if mkt.get("key") != "h2h":
                        continue
                    outcomes = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
                    ho = outcomes.get(home)
                    ao = outcomes.get(away)
                    if ho and ao:
                        hp, ap = _remove_vig(ho, ao)
                        odds_batch.append({
                            "game_id": game_id, "ts": ts,
                            "home_odds": ho, "away_odds": ao, "draw_odds": None,
                            "home_prob_novig": round(hp, 4),
                            "away_prob_novig": round(ap, 4),
                        })

        if games_batch:
            db.upsert_games_batch(games_batch)
        if odds_batch:
            n = db.upsert_pinnacle_odds_batch(odds_batch)
            stored += n

        if requests_made % 20 == 0:
            credits_used = requests_made * CREDITS_PER_REQUEST
            print(f"  [{current}] {requests_made} req, {stored} odds, ~{credits_used} credits", flush=True)

        current += timedelta(days=1)

    credits_used = requests_made * CREDITS_PER_REQUEST
    print(f"  Done: {stored} odds from {requests_made} requests (~{credits_used} credits)", flush=True)
    return stored, credits_used


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk Pinnacle odds download")
    parser.add_argument("--sport", choices=["nba", "nfl", "ufc", "all"], default="all")
    parser.add_argument("--budget", type=int, default=20000, help="Total credit budget")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without downloading")
    parser.add_argument("--db", default=None, help="DB path")
    args = parser.parse_args()

    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        try:
            env_path = Path(__file__).resolve().parent.parent / ".env"
            for line in env_path.read_text().splitlines():
                if line.startswith("ODDS_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass

    if not api_key and not args.dry_run:
        print("ERROR: Set ODDS_API_KEY in environment or .env")
        sys.exit(1)

    db = SportsDB(args.db) if args.db else SportsDB()

    sports = list(SPORT_CONFIGS.keys()) if args.sport == "all" else [args.sport]
    total_budget = args.budget
    remaining_budget = total_budget

    print(f"=== Bulk Pinnacle Odds Download ===")
    print(f"Budget: {total_budget} credits ({total_budget // CREDITS_PER_REQUEST} requests)")
    print(f"Sports: {', '.join(sports)}")
    print()

    total_stored = 0
    total_credits = 0

    # Priority order: NBA current → NBA previous → NFL → NBA older → UFC
    download_plan = []
    for sport in sports:
        cfg = SPORT_CONFIGS[sport]
        for season_name, start, end in cfg["seasons"]:
            download_plan.append((sport, season_name, start, end))

    for sport, season, start, end in download_plan:
        if remaining_budget < CREDITS_PER_REQUEST:
            print(f"\n  Budget exhausted! Skipping remaining seasons.")
            break

        print(f"\n{'─'*50}")
        print(f"{sport.upper()} {season}")
        print(f"{'─'*50}")

        stored, used = download_sport_historical(
            db, api_key, sport, start, end,
            budget_credits=remaining_budget,
            dry_run=args.dry_run,
        )
        total_stored += stored
        total_credits += used
        remaining_budget -= used

    # Summary
    print(f"\n{'═'*50}")
    print(f"SUMMARY")
    print(f"{'═'*50}")
    print(f"Total odds stored: {total_stored}")
    print(f"Credits used: {total_credits} / {total_budget}")
    print(f"Credits remaining: {remaining_budget}")

    if not args.dry_run:
        pinnacle_count = db.conn.execute("SELECT COUNT(*) FROM pinnacle_odds").fetchone()[0]
        print(f"Total Pinnacle odds in DB: {pinnacle_count}")

    db.close()


if __name__ == "__main__":
    main()
