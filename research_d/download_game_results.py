"""Download Historical NBA and EPL Game Results.

Fetches completed game results from ESPN (NBA) and football-data.co.uk (EPL),
storing them in the SportsDB SQLite database for backtesting and Elo/Glicko-2
model training.

Data sources:
    NBA — ESPN Scoreboard API (free, no auth, JSON)
          https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard
    EPL — football-data.co.uk CSV files (free, no auth)
          https://www.football-data.co.uk/mmz4281/{season}/E0.csv
    EPL fallback — ESPN Soccer API (if CSV download fails)
          https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard

Usage:
    python3 research_d/download_game_results.py --sport all
    python3 research_d/download_game_results.py --sport nba --start-date 2025-10-22
    python3 research_d/download_game_results.py --sport epl --seasons 2024-25,2025-26
    python3 research_d/download_game_results.py --sport nba --start-date 2024-10-22

Output:
    research_d/data/sports_backtest.sqlite  — SQLite database (via SportsDB)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
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

    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()
    SSL_CONTEXT.check_hostname = False
    SSL_CONTEXT.verify_mode = ssl.CERT_NONE

# Add project root to path so we can import SportsDB
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_d.sports_db import SportsDB

# ── Constants ────────────────────────────────────────────────────────────

ESPN_NBA_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
)
ESPN_EPL_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard"
)
FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"

# Rate limiting: 0.5s between ESPN requests (conservative)
ESPN_REQUEST_DELAY = 0.5
# football-data.co.uk is a single CSV download per season — no rate limit needed
FOOTBALL_DATA_DELAY = 1.0

# NBA season start dates (regular season opening night)
NBA_SEASON_STARTS: dict[str, date] = {
    "2023-24": date(2023, 10, 24),
    "2024-25": date(2024, 10, 22),
    "2025-26": date(2025, 10, 28),
}

# NBA season end dates (approximate, regular season + play-in)
NBA_SEASON_ENDS: dict[str, date] = {
    "2023-24": date(2024, 6, 18),
    "2024-25": date(2025, 6, 22),
    "2025-26": date(2026, 6, 21),
}

# EPL season codes for football-data.co.uk URL (YY of start year + YY of end year)
EPL_SEASON_CODES: dict[str, str] = {
    "2023-24": "2324",
    "2024-25": "2425",
    "2025-26": "2526",
}

# ── EPL Team Name → Abbreviation Mapping ─────────────────────────────────
# Maps football-data.co.uk team names to standard 3-letter abbreviations.
# These abbreviations are used in game_id construction:
#   epl_{YYYYMMDD}_{away_abbr}_{home_abbr}

EPL_TEAM_ABBR: dict[str, str] = {
    # Current Premier League teams (2025-26 season)
    "Arsenal": "ARS",
    "Aston Villa": "AVL",
    "Bournemouth": "BOU",
    "Brentford": "BRE",
    "Brighton": "BHA",
    "Brighton and Hove Albion": "BHA",
    "Burnley": "BUR",
    "Chelsea": "CHE",
    "Crystal Palace": "CRY",
    "Everton": "EVE",
    "Fulham": "FUL",
    "Ipswich": "IPS",
    "Ipswich Town": "IPS",
    "Leeds": "LEE",
    "Leeds United": "LEE",
    "Leicester": "LEI",
    "Leicester City": "LEI",
    "Liverpool": "LIV",
    "Luton": "LUT",
    "Luton Town": "LUT",
    "Man City": "MCI",
    "Manchester City": "MCI",
    "Man United": "MUN",
    "Manchester United": "MUN",
    "Newcastle": "NEW",
    "Newcastle United": "NEW",
    "Nott'm Forest": "NFO",
    "Nottingham Forest": "NFO",
    "Sheffield United": "SHU",
    "Sheffield Utd": "SHU",
    "Southampton": "SOU",
    "Tottenham": "TOT",
    "Tottenham Hotspur": "TOT",
    "West Ham": "WHU",
    "West Ham United": "WHU",
    "Wolves": "WOL",
    "Wolverhampton": "WOL",
    "Wolverhampton Wanderers": "WOL",
    "Coventry": "COV",
    "Coventry City": "COV",
    "Sunderland": "SUN",
    "Middlesbrough": "MID",
}

# ── NBA Team Abbreviations (ESPN uses these already) ─────────────────────
# ESPN provides abbreviations directly in the API response. This mapping
# is a fallback for display name → abbreviation when the API field is missing.

NBA_TEAM_ABBR: dict[str, str] = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "LA Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


# ── HTTP helpers ─────────────────────────────────────────────────────────


def _http_get_json(url: str, retries: int = 3) -> dict | list | None:
    """GET JSON with retry logic and rate-limit handling.

    Args:
        url: URL to fetch.
        retries: Number of retry attempts for transient errors.

    Returns:
        Parsed JSON response, or None on failure.
    """
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={
            "User-Agent": "ArboResearch/1.0",
            "Accept": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=30, context=SSL_CONTEXT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 2 ** (attempt + 1)
                print(f"    Rate limited (429), waiting {wait}s...")
                time.sleep(wait)
            elif e.code >= 500:
                print(f"    Server error ({e.code}), retrying...")
                time.sleep(1)
            else:
                print(f"    HTTP error {e.code} for {url}")
                return None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(0.5)
            else:
                print(f"    Error fetching {url}: {e}")
    return None


def _http_get_text(url: str, retries: int = 3) -> str | None:
    """GET raw text with retry logic (for CSV downloads).

    Args:
        url: URL to fetch.
        retries: Number of retry attempts for transient errors.

    Returns:
        Response body as text, or None on failure.
    """
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={
            "User-Agent": "ArboResearch/1.0",
        })
        try:
            with urllib.request.urlopen(req, timeout=60, context=SSL_CONTEXT) as resp:
                # football-data.co.uk uses Windows-1252 encoding for CSV files
                raw = resp.read()
                # Try UTF-8 first, fall back to latin-1 (superset of Windows-1252)
                try:
                    return raw.decode("utf-8")
                except UnicodeDecodeError:
                    return raw.decode("latin-1")
        except urllib.error.HTTPError as e:
            if e.code >= 500:
                time.sleep(1)
            else:
                print(f"    HTTP error {e.code} for {url}")
                return None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(0.5)
            else:
                print(f"    Error fetching {url}: {e}")
    return None


# ── EPL: football-data.co.uk CSV parser ──────────────────────────────────


def _epl_team_abbr(team_name: str) -> str:
    """Convert an EPL team name to its 3-letter abbreviation.

    Args:
        team_name: Team name as it appears in football-data.co.uk CSVs.

    Returns:
        3-letter abbreviation string. Falls back to first 3 chars of the
        name (uppercased) if no mapping is found.
    """
    abbr = EPL_TEAM_ABBR.get(team_name)
    if abbr:
        return abbr

    # Try case-insensitive partial match
    name_lower = team_name.lower()
    for full_name, code in EPL_TEAM_ABBR.items():
        if full_name.lower() == name_lower:
            return code

    # Last resort: first 3 chars uppercased
    fallback = team_name[:3].upper()
    print(f"    WARNING: No abbreviation for EPL team '{team_name}', using '{fallback}'")
    return fallback


def _parse_epl_date(date_str: str) -> date | None:
    """Parse a date string from football-data.co.uk CSV.

    Handles both DD/MM/YYYY and DD/MM/YY formats.

    Args:
        date_str: Date string (e.g., "15/03/2026" or "15/03/26").

    Returns:
        Parsed date, or None if parsing fails.
    """
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None


def download_epl_season(season: str, db: SportsDB) -> int:
    """Download EPL game results for a single season from football-data.co.uk.

    Args:
        season: Season string (e.g., "2024-25").
        db: SportsDB instance for storing results.

    Returns:
        Number of games inserted/updated.
    """
    season_code = EPL_SEASON_CODES.get(season)
    if not season_code:
        print(f"  ERROR: Unknown EPL season '{season}'. "
              f"Valid: {list(EPL_SEASON_CODES.keys())}")
        return 0

    url = FOOTBALL_DATA_URL.format(season_code=season_code)
    print(f"  Downloading EPL {season} from football-data.co.uk...")
    print(f"    URL: {url}")

    text = _http_get_text(url)
    if not text:
        print(f"  WARN: CSV download failed for {season}, trying ESPN fallback...")
        return _download_epl_season_espn(season, db)

    # Parse CSV
    reader = csv.DictReader(io.StringIO(text))

    # Verify required columns exist
    if reader.fieldnames is None:
        print(f"  ERROR: Empty CSV for EPL {season}")
        return 0

    required_cols = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    available_cols = set(reader.fieldnames)
    missing = required_cols - available_cols
    if missing:
        print(f"  ERROR: Missing CSV columns: {missing}")
        print(f"    Available: {sorted(available_cols)}")
        return 0

    games: list[dict[str, Any]] = []
    skipped = 0

    for row in reader:
        # Skip rows with missing data (future scheduled games have no score)
        home_goals_str = row.get("FTHG", "").strip()
        away_goals_str = row.get("FTAG", "").strip()

        if not home_goals_str or not away_goals_str:
            skipped += 1
            continue

        try:
            home_score = int(home_goals_str)
            away_score = int(away_goals_str)
        except ValueError:
            skipped += 1
            continue

        game_date = _parse_epl_date(row["Date"])
        if not game_date:
            skipped += 1
            continue

        home_team = row["HomeTeam"].strip()
        away_team = row["AwayTeam"].strip()
        home_abbr = _epl_team_abbr(home_team)
        away_abbr = _epl_team_abbr(away_team)

        # Game ID format: epl_{YYYYMMDD}_{away_abbr}_{home_abbr}
        game_id = f"epl_{game_date.strftime('%Y%m%d')}_{away_abbr}_{home_abbr}"

        # Determine winner from FTR (Full-Time Result): H=Home, A=Away, D=Draw
        ftr = row.get("FTR", "").strip()

        # Extract kick-off time if available
        game_time = row.get("Time", "").strip() or None

        # Build extra_json with detailed match info
        extra: dict[str, Any] = {
            "home_team_full": home_team,
            "away_team_full": away_team,
            "ftr": ftr,
            "source": "football-data.co.uk",
        }
        # Include half-time score if available
        hthg = row.get("HTHG", "").strip()
        htag = row.get("HTAG", "").strip()
        if hthg and htag:
            extra["ht_home_goals"] = int(hthg)
            extra["ht_away_goals"] = int(htag)

        games.append({
            "game_id": game_id,
            "sport": "epl",
            "league": "Premier League",
            "home_team": home_abbr,
            "away_team": away_abbr,
            "game_date": game_date.isoformat(),
            "game_time": game_time,
            "home_score": home_score,
            "away_score": away_score,
            "status": "final",
            "season": season,
            "venue": None,
            "extra_json": json.dumps(extra),
        })

    if not games:
        print(f"  WARNING: No completed games found in EPL {season} CSV")
        return 0

    # Batch upsert into database
    count = db.upsert_games_batch(games)

    if skipped > 0:
        print(f"    Skipped {skipped} rows (missing data / unplayed)")
    print(f"    Inserted/updated {count} EPL games for {season}")

    return count


def _download_epl_season_espn(season: str, db: SportsDB) -> int:
    """Fallback: download EPL results from ESPN API day-by-day.

    Used when football-data.co.uk CSV is unavailable.

    Args:
        season: Season string (e.g., "2025-26").
        db: SportsDB instance for storing results.

    Returns:
        Number of games inserted/updated.
    """
    print(f"  Downloading EPL {season} from ESPN API (fallback)...")

    # Determine date range for the season
    # EPL runs August through May
    start_year = int(season.split("-")[0])
    start_date = date(start_year, 8, 1)
    end_date = min(date(start_year + 1, 5, 31), date.today())

    games: list[dict[str, Any]] = []
    current = start_date
    days_processed = 0
    total_days = (end_date - start_date).days + 1

    while current <= end_date:
        date_str = current.strftime("%Y%m%d")
        url = f"{ESPN_EPL_URL}?dates={date_str}"

        data = _http_get_json(url)
        if data and isinstance(data, dict):
            parsed = _parse_espn_epl_response(data, season)
            games.extend(parsed)

        days_processed += 1
        if days_processed % 30 == 0:
            pct = days_processed / total_days * 100
            print(f"    [{pct:5.1f}%] Day {days_processed}/{total_days}, "
                  f"{len(games)} games found")

        current += timedelta(days=1)
        time.sleep(ESPN_REQUEST_DELAY)

    if not games:
        print(f"  WARNING: No EPL games found via ESPN for {season}")
        return 0

    count = db.upsert_games_batch(games)
    print(f"    Inserted/updated {count} EPL games for {season} (via ESPN)")
    return count


def _parse_espn_epl_response(
    data: dict[str, Any],
    season: str,
) -> list[dict[str, Any]]:
    """Parse ESPN EPL scoreboard response into game dicts.

    Args:
        data: ESPN API JSON response.
        season: Season string for metadata.

    Returns:
        List of game dicts ready for upsert_games_batch().
    """
    games: list[dict[str, Any]] = []

    for event in data.get("events", []):
        competitions = event.get("competitions", [])
        if not competitions:
            continue
        comp = competitions[0]

        # Only include completed games
        status_obj = comp.get("status", {})
        status_type = status_obj.get("type", {})
        if not status_type.get("completed", False):
            continue

        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        # ESPN: competitors[0] is usually home, but check homeAway field
        home_data: dict[str, Any] | None = None
        away_data: dict[str, Any] | None = None
        for c in competitors:
            if c.get("homeAway") == "home":
                home_data = c
            elif c.get("homeAway") == "away":
                away_data = c

        if not home_data or not away_data:
            continue

        home_abbr = home_data.get("team", {}).get("abbreviation", "UNK")
        away_abbr = away_data.get("team", {}).get("abbreviation", "UNK")
        home_name = home_data.get("team", {}).get("displayName", home_abbr)
        away_name = away_data.get("team", {}).get("displayName", away_abbr)

        try:
            home_score = int(home_data.get("score", 0))
            away_score = int(away_data.get("score", 0))
        except (ValueError, TypeError):
            continue

        # Parse game date from the event
        event_date_str = event.get("date", "")
        try:
            event_dt = datetime.fromisoformat(event_date_str.replace("Z", "+00:00"))
            game_date = event_dt.date()
            game_time = event_dt.strftime("%H:%M")
        except (ValueError, AttributeError):
            continue

        game_id = f"epl_{game_date.strftime('%Y%m%d')}_{away_abbr}_{home_abbr}"

        extra = {
            "home_team_full": home_name,
            "away_team_full": away_name,
            "espn_event_id": event.get("id"),
            "source": "espn",
        }

        games.append({
            "game_id": game_id,
            "sport": "epl",
            "league": "Premier League",
            "home_team": home_abbr,
            "away_team": away_abbr,
            "game_date": game_date.isoformat(),
            "game_time": game_time,
            "home_score": home_score,
            "away_score": away_score,
            "status": "final",
            "season": season,
            "venue": comp.get("venue", {}).get("fullName"),
            "extra_json": json.dumps(extra),
        })

    return games


# ── NBA: ESPN Scoreboard API ─────────────────────────────────────────────


def _parse_espn_nba_response(
    data: dict[str, Any],
    season: str,
) -> list[dict[str, Any]]:
    """Parse ESPN NBA scoreboard response into game dicts.

    ESPN response structure:
        {
          "events": [
            {
              "id": "401584793",
              "date": "2025-10-22T23:30Z",
              "competitions": [
                {
                  "competitors": [
                    {"homeAway": "home", "team": {"abbreviation": "BOS"}, "score": "110"},
                    {"homeAway": "away", "team": {"abbreviation": "NYK"}, "score": "105"}
                  ],
                  "venue": {"fullName": "TD Garden"},
                  "status": {"type": {"completed": true}}
                }
              ]
            }
          ]
        }

    Args:
        data: ESPN API JSON response.
        season: Season string for metadata (e.g., "2025-26").

    Returns:
        List of game dicts ready for upsert_games_batch().
    """
    games: list[dict[str, Any]] = []

    for event in data.get("events", []):
        competitions = event.get("competitions", [])
        if not competitions:
            continue
        comp = competitions[0]

        # Only include completed games
        status_obj = comp.get("status", {})
        status_type = status_obj.get("type", {})
        if not status_type.get("completed", False):
            continue

        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        # Identify home and away teams
        home_data: dict[str, Any] | None = None
        away_data: dict[str, Any] | None = None
        for c in competitors:
            if c.get("homeAway") == "home":
                home_data = c
            elif c.get("homeAway") == "away":
                away_data = c

        if not home_data or not away_data:
            continue

        # Extract team info — ESPN provides abbreviation directly
        home_abbr = home_data.get("team", {}).get("abbreviation", "UNK")
        away_abbr = away_data.get("team", {}).get("abbreviation", "UNK")
        home_name = home_data.get("team", {}).get("displayName", home_abbr)
        away_name = away_data.get("team", {}).get("displayName", away_abbr)

        try:
            home_score = int(home_data.get("score", 0))
            away_score = int(away_data.get("score", 0))
        except (ValueError, TypeError):
            continue

        # Parse game date/time from event
        event_date_str = event.get("date", "")
        try:
            event_dt = datetime.fromisoformat(event_date_str.replace("Z", "+00:00"))
            game_date = event_dt.date()
            game_time = event_dt.strftime("%H:%M")
        except (ValueError, AttributeError):
            continue

        # Game ID format: nba_{YYYYMMDD}_{away_abbr}_{home_abbr}
        game_id = f"nba_{game_date.strftime('%Y%m%d')}_{away_abbr}_{home_abbr}"

        venue = comp.get("venue", {}).get("fullName")

        extra = {
            "home_team_full": home_name,
            "away_team_full": away_name,
            "espn_event_id": event.get("id"),
            "source": "espn",
        }

        games.append({
            "game_id": game_id,
            "sport": "nba",
            "league": "NBA",
            "home_team": home_abbr,
            "away_team": away_abbr,
            "game_date": game_date.isoformat(),
            "game_time": game_time,
            "home_score": home_score,
            "away_score": away_score,
            "status": "final",
            "season": season,
            "venue": venue,
            "extra_json": json.dumps(extra),
        })

    return games


def download_nba_season(
    season: str,
    db: SportsDB,
    start_override: date | None = None,
) -> int:
    """Download NBA game results for a season from ESPN API.

    Iterates day-by-day through the NBA season, fetching completed
    game results from the ESPN scoreboard endpoint.

    Args:
        season: Season string (e.g., "2025-26").
        db: SportsDB instance for storing results.
        start_override: If provided, start from this date instead of
            the season start date.

    Returns:
        Number of games inserted/updated.
    """
    season_start = NBA_SEASON_STARTS.get(season)
    season_end = NBA_SEASON_ENDS.get(season)

    if not season_start or not season_end:
        print(f"  ERROR: Unknown NBA season '{season}'. "
              f"Valid: {list(NBA_SEASON_STARTS.keys())}")
        return 0

    # Apply start_override if provided (for incremental updates)
    start = start_override if start_override and start_override > season_start else season_start
    # Don't fetch future dates
    end = min(season_end, date.today())

    if start > end:
        print(f"  NBA {season}: No dates to fetch (start={start} > end={end})")
        return 0

    total_days = (end - start).days + 1
    print(f"  Downloading NBA {season} from ESPN API...")
    print(f"    Date range: {start.isoformat()} to {end.isoformat()} ({total_days} days)")

    all_games: list[dict[str, Any]] = []
    current = start
    days_processed = 0
    days_with_games = 0
    errors = 0
    t_start = time.time()

    while current <= end:
        date_str = current.strftime("%Y%m%d")
        url = f"{ESPN_NBA_URL}?dates={date_str}"

        data = _http_get_json(url)
        if data and isinstance(data, dict):
            day_games = _parse_espn_nba_response(data, season)
            if day_games:
                all_games.extend(day_games)
                days_with_games += 1
        elif data is None:
            errors += 1

        days_processed += 1

        # Progress update every 30 days
        if days_processed % 30 == 0:
            pct = days_processed / total_days * 100
            elapsed = time.time() - t_start
            eta = elapsed / days_processed * (total_days - days_processed)
            print(f"    [{pct:5.1f}%] Day {days_processed}/{total_days} | "
                  f"{len(all_games)} games | "
                  f"ETA: {eta / 60:.1f}min")

        current += timedelta(days=1)
        time.sleep(ESPN_REQUEST_DELAY)

    if not all_games:
        print(f"  WARNING: No NBA games found for {season}")
        if errors > 0:
            print(f"    ({errors} API errors encountered)")
        return 0

    # Batch upsert into database
    count = db.upsert_games_batch(all_games)

    print(f"    Inserted/updated {count} NBA games for {season}")
    print(f"    Days with games: {days_with_games}/{total_days}")
    if errors > 0:
        print(f"    API errors: {errors}")

    return count


# ── CLI and main pipeline ────────────────────────────────────────────────


def _parse_seasons(seasons_str: str) -> list[str]:
    """Parse comma-separated season list.

    Args:
        seasons_str: Comma-separated seasons (e.g., "2024-25,2025-26").

    Returns:
        List of season strings.
    """
    return [s.strip() for s in seasons_str.split(",") if s.strip()]


def _determine_nba_season(d: date) -> str | None:
    """Determine which NBA season a date falls in.

    Args:
        d: Date to check.

    Returns:
        Season string (e.g., "2025-26") or None if outside known seasons.
    """
    for season, start in NBA_SEASON_STARTS.items():
        end = NBA_SEASON_ENDS.get(season)
        if end and start <= d <= end:
            return season
    return None


def _print_summary(db: SportsDB, t_start: float) -> None:
    """Print database summary statistics after download.

    Args:
        db: SportsDB instance to query.
        t_start: Start time of the download (for elapsed time).
    """
    stats = db.stats()
    elapsed = time.time() - t_start

    print(f"\n{'=' * 60}")
    print("DATABASE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total games:    {stats['games']:,}")
    print(f"    NBA:          {stats['games_nba']:,}")
    print(f"    EPL:          {stats['games_epl']:,}")
    print(f"  Markets:        {stats['markets']:,}")
    print(f"  Price records:  {stats['prices']:,}")
    print(f"  Elapsed time:   {elapsed:.1f}s")
    print(f"  Database:       {db.db_path}")

    # Show date ranges per sport
    for sport, label in [("nba", "NBA"), ("epl", "EPL")]:
        games = db.get_games(sport=sport)
        if games:
            dates = [g["game_date"] for g in games]
            min_date = min(dates)
            max_date = max(dates)
            # Count unique seasons
            seasons = {g["season"] for g in games if g["season"]}
            print(f"\n  {label} details:")
            print(f"    Date range:   {min_date} to {max_date}")
            print(f"    Seasons:      {', '.join(sorted(seasons))}")
            print(f"    Total games:  {len(games)}")

            # Show score distribution for EPL (wins/draws/losses)
            if sport == "epl":
                home_wins = sum(
                    1 for g in games
                    if g["home_score"] is not None and g["away_score"] is not None
                    and g["home_score"] > g["away_score"]
                )
                draws = sum(
                    1 for g in games
                    if g["home_score"] is not None and g["away_score"] is not None
                    and g["home_score"] == g["away_score"]
                )
                away_wins = sum(
                    1 for g in games
                    if g["home_score"] is not None and g["away_score"] is not None
                    and g["home_score"] < g["away_score"]
                )
                total = home_wins + draws + away_wins
                if total > 0:
                    print(f"    Home wins:    {home_wins} ({home_wins / total * 100:.1f}%)")
                    print(f"    Draws:        {draws} ({draws / total * 100:.1f}%)")
                    print(f"    Away wins:    {away_wins} ({away_wins / total * 100:.1f}%)")

            # Show top teams by appearances for NBA
            if sport == "nba":
                from collections import Counter
                team_counts: Counter[str] = Counter()
                for g in games:
                    team_counts[g["home_team"]] += 1
                    team_counts[g["away_team"]] += 1
                top5 = team_counts.most_common(5)
                if top5:
                    print(f"    Top 5 teams by games:")
                    for team, count in top5:
                        print(f"      {team}: {count}")

    print(f"\n{'=' * 60}")


def main() -> None:
    """Main entry point: parse CLI args and download game results."""
    parser = argparse.ArgumentParser(
        description="Download historical NBA and EPL game results into SportsDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 research_d/download_game_results.py --sport all
  python3 research_d/download_game_results.py --sport nba --seasons 2025-26
  python3 research_d/download_game_results.py --sport epl --seasons 2023-24,2024-25,2025-26
  python3 research_d/download_game_results.py --sport nba --start-date 2025-12-01
        """,
    )
    parser.add_argument(
        "--sport",
        choices=["nba", "epl", "all"],
        default="all",
        help="Which sport(s) to download (default: all)",
    )
    parser.add_argument(
        "--seasons",
        type=str,
        default="2023-24,2024-25,2025-26",
        help="Comma-separated season list (default: 2023-24,2024-25,2025-26)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Override start date (YYYY-MM-DD). For NBA, overrides season start.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Custom SQLite database path (default: research_d/data/sports_backtest.sqlite)",
    )
    args = parser.parse_args()

    # Parse arguments
    seasons = _parse_seasons(args.seasons)
    start_date: date | None = None
    if args.start_date:
        try:
            start_date = date.fromisoformat(args.start_date)
        except ValueError:
            print(f"ERROR: Invalid date format '{args.start_date}'. Use YYYY-MM-DD.")
            sys.exit(1)

    download_nba = args.sport in ("nba", "all")
    download_epl = args.sport in ("epl", "all")

    print(f"{'=' * 60}")
    print("Download Game Results")
    print(f"{'=' * 60}")
    print(f"  Sport(s):   {args.sport}")
    print(f"  Seasons:    {', '.join(seasons)}")
    if start_date:
        print(f"  Start date: {start_date.isoformat()}")
    print()

    t_start = time.time()

    # Open database
    db = SportsDB(args.db_path)
    existing_stats = db.stats()
    print(f"Database: {db.db_path}")
    print(f"  Existing games: {existing_stats['games']:,} "
          f"(NBA: {existing_stats['games_nba']:,}, EPL: {existing_stats['games_epl']:,})")
    print()

    total_games = 0

    # ── Download EPL ──
    if download_epl:
        print(f"{'─' * 60}")
        print("EPL — Premier League")
        print(f"{'─' * 60}")

        for season in seasons:
            if season not in EPL_SEASON_CODES:
                print(f"  Skipping EPL season {season} (no season code mapping)")
                continue
            count = download_epl_season(season, db)
            total_games += count
            time.sleep(FOOTBALL_DATA_DELAY)

        print()

    # ── Download NBA ──
    if download_nba:
        print(f"{'─' * 60}")
        print("NBA — National Basketball Association")
        print(f"{'─' * 60}")

        for season in seasons:
            if season not in NBA_SEASON_STARTS:
                print(f"  Skipping NBA season {season} (no season date mapping)")
                continue

            # If --start-date is provided and falls within this season, use it
            season_start_override = start_date
            if start_date:
                season_obj = _determine_nba_season(start_date)
                if season_obj != season:
                    # start_date doesn't fall in this season, ignore override
                    season_start_override = None

            count = download_nba_season(season, db, start_override=season_start_override)
            total_games += count

        print()

    # ── Summary ──
    _print_summary(db, t_start)

    if total_games == 0:
        print("\nNo new games were downloaded. The database may already be up to date.")

    db.close()


if __name__ == "__main__":
    main()
