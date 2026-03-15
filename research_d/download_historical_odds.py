"""Download Historical Pinnacle/Sharp Odds for EPL and NBA.

Sources:
    EPL — football-data.co.uk CSV files (columns PSH/PSD/PSA = Pinnacle opening,
          PSCH/PSCD/PSCA = Pinnacle closing). FREE, all seasons 2005+.

    NBA — SportsbookReviewsOnline.com (SBRO) Excel files with consensus
          sharp closing lines (moneyline, spreads, totals). FREE, 2007-2024.
          For 2024-25+: The Odds API historical endpoint (costs credits).

These are REAL historical odds that were available at the time of each game.
Essential for realistic backtesting — without real odds, our probability
model benchmark (60% Pinnacle weight) would be meaningless.

Usage:
    python3 research_d/download_historical_odds.py --sport all
    python3 research_d/download_historical_odds.py --sport epl --seasons 2023-24,2024-25,2025-26
    python3 research_d/download_historical_odds.py --sport nba

Output:
    research_d/data/sports_backtest.sqlite — pinnacle_odds table populated
    research_d/data/odds_raw/              — cached raw downloads
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import ssl
import sys
import time
import urllib.request
from datetime import datetime, timezone
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
ODDS_CACHE_DIR = DATA_DIR / "odds_raw"

# ── Vig Removal ──────────────────────────────────────────────────────

def remove_vig_2way(home_odds: float, away_odds: float) -> tuple[float, float]:
    """Remove vig from 2-way odds (NBA/NFL).

    Args:
        home_odds: Decimal odds for home win.
        away_odds: Decimal odds for away win.

    Returns:
        (home_prob_novig, away_prob_novig) summing to ~1.0.
    """
    if home_odds <= 1 or away_odds <= 1:
        return 0.5, 0.5
    imp_h = 1.0 / home_odds
    imp_a = 1.0 / away_odds
    total = imp_h + imp_a
    return imp_h / total, imp_a / total


def remove_vig_3way(
    home_odds: float, draw_odds: float, away_odds: float
) -> tuple[float, float, float]:
    """Remove vig from 3-way odds (soccer).

    Returns:
        (home_prob, draw_prob, away_prob) summing to ~1.0.
    """
    if home_odds <= 1 or draw_odds <= 1 or away_odds <= 1:
        return 0.33, 0.34, 0.33
    imp_h = 1.0 / home_odds
    imp_d = 1.0 / draw_odds
    imp_a = 1.0 / away_odds
    total = imp_h + imp_d + imp_a
    return imp_h / total, imp_d / total, imp_a / total


# ── HTTP Helper ──────────────────────────────────────────────────────

def _http_get(url: str, binary: bool = False) -> Any:
    """Download URL with retry. Returns text or bytes."""
    for attempt in range(3):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 ArboResearch/2.0"})
        try:
            with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
                data = resp.read()
                return data if binary else data.decode("utf-8", errors="ignore")
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  ERROR downloading {url}: {e}")
                return None


# ══════════════════════════════════════════════════════════════════════
# EPL: football-data.co.uk Pinnacle odds
# ══════════════════════════════════════════════════════════════════════

# EPL team name → abbreviation mapping (from download_game_results.py)
EPL_TEAM_MAP = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU",
    "Brentford": "BRE", "Brighton": "BHA", "Burnley": "BUR",
    "Chelsea": "CHE", "Crystal Palace": "CRY", "Everton": "EVE",
    "Fulham": "FUL", "Ipswich": "IPS", "Ipswich Town": "IPS",
    "Leeds": "LEE", "Leeds United": "LEE",
    "Leicester": "LEI", "Leicester City": "LEI",
    "Liverpool": "LIV", "Luton": "LUT", "Luton Town": "LUT",
    "Man City": "MCI", "Manchester City": "MCI",
    "Man United": "MUN", "Manchester United": "MUN",
    "Newcastle": "NEW", "Newcastle United": "NEW",
    "Nott'm Forest": "NFO", "Nottingham Forest": "NFO",
    "Sheffield United": "SHU", "Sheffield Utd": "SHU",
    "Southampton": "SOU", "Tottenham": "TOT",
    "West Ham": "WHU", "West Ham United": "WHU",
    "Wolves": "WOL", "Wolverhampton": "WOL",
    "Wolverhampton Wanderers": "WOL",
}

FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"

SEASON_CODES = {
    "2023-24": "2324",
    "2024-25": "2425",
    "2025-26": "2526",
    "2022-23": "2223",
    "2021-22": "2122",
}


def _parse_epl_date(date_str: str) -> str | None:
    """Parse EPL date from CSV (DD/MM/YYYY or DD/MM/YY)."""
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def download_epl_pinnacle(
    db: SportsDB,
    seasons: list[str] | None = None,
) -> int:
    """Download EPL Pinnacle odds from football-data.co.uk CSVs.

    Extracts both opening (PSH/PSD/PSA) and closing (PSCH/PSCD/PSCA)
    Pinnacle odds. Stores closing odds as the primary benchmark, opening
    odds as a secondary snapshot.

    Args:
        db: SportsDB instance.
        seasons: List of seasons like ["2023-24", "2024-25"].

    Returns:
        Number of odds records stored.
    """
    if seasons is None:
        seasons = ["2023-24", "2024-25", "2025-26"]

    total_stored = 0
    ODDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for season in seasons:
        code = SEASON_CODES.get(season)
        if not code:
            print(f"  Unknown season code for {season}, skipping")
            continue

        # Check cache
        cache_path = ODDS_CACHE_DIR / f"epl_{code}.csv"
        if cache_path.exists():
            print(f"  Loading EPL {season} from cache")
            text = cache_path.read_text(encoding="utf-8", errors="ignore")
        else:
            url = FOOTBALL_DATA_URL.format(season_code=code)
            print(f"  Downloading EPL {season}: {url}")
            text = _http_get(url)
            if not text:
                print(f"  Failed to download EPL {season}")
                continue
            cache_path.write_text(text, encoding="utf-8")

        # Parse CSV
        reader = csv.DictReader(io.StringIO(text))
        odds_batch: list[dict] = []
        skipped = 0

        for row in reader:
            home_team_raw = row.get("HomeTeam", "").strip()
            away_team_raw = row.get("AwayTeam", "").strip()
            date_str = row.get("Date", "").strip()

            if not home_team_raw or not away_team_raw or not date_str:
                continue

            game_date = _parse_epl_date(date_str)
            if not game_date:
                continue

            home_abbr = EPL_TEAM_MAP.get(home_team_raw, home_team_raw[:3].upper())
            away_abbr = EPL_TEAM_MAP.get(away_team_raw, away_team_raw[:3].upper())
            game_id = f"epl_{game_date.replace('-', '')}_{away_abbr}_{home_abbr}"

            # Extract Pinnacle closing odds (PSCH/PSCD/PSCA) — most important
            psch = _safe_float(row.get("PSCH"))
            pscd = _safe_float(row.get("PSCD"))
            psca = _safe_float(row.get("PSCA"))

            # Fallback to opening odds (PSH/PSD/PSA) if closing not available
            if not psch:
                psch = _safe_float(row.get("PSH"))
            if not pscd:
                pscd = _safe_float(row.get("PSD"))
            if not psca:
                psca = _safe_float(row.get("PSA"))

            if not psch or not pscd or not psca:
                skipped += 1
                continue

            # Compute no-vig probabilities
            h_prob, _, a_prob = remove_vig_3way(psch, pscd, psca)

            # Use game_date midnight UTC as timestamp
            ts = int(datetime.strptime(game_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            ).timestamp())

            odds_batch.append({
                "game_id": game_id,
                "ts": ts,
                "home_odds": psch,
                "away_odds": psca,
                "draw_odds": pscd,
                "home_prob_novig": round(h_prob, 4),
                "away_prob_novig": round(a_prob, 4),
            })

            # Also store opening odds as a separate timestamp (1 hour earlier)
            psh_open = _safe_float(row.get("PSH"))
            psd_open = _safe_float(row.get("PSD"))
            psa_open = _safe_float(row.get("PSA"))
            if psh_open and psd_open and psa_open:
                h_open, _, a_open = remove_vig_3way(psh_open, psd_open, psa_open)
                odds_batch.append({
                    "game_id": game_id,
                    "ts": ts - 3600,  # 1 hour before closing
                    "home_odds": psh_open,
                    "away_odds": psa_open,
                    "draw_odds": psd_open,
                    "home_prob_novig": round(h_open, 4),
                    "away_prob_novig": round(a_open, 4),
                })

        if odds_batch:
            n = db.upsert_pinnacle_odds_batch(odds_batch)
            total_stored += n
            print(f"  EPL {season}: stored {n} odds records "
                  f"({n // 2} games with open+close), skipped {skipped}")
        else:
            print(f"  EPL {season}: no Pinnacle odds found (skipped {skipped})")

    return total_stored


# ══════════════════════════════════════════════════════════════════════
# NBA: SBRO Excel files + The Odds API historical
# ══════════════════════════════════════════════════════════════════════

# SBRO URLs for NBA seasons
SBRO_BASE = "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba"
SBRO_FILES = {
    "2023-24": f"{SBRO_BASE}/nba%20odds%202023-24.xlsx",
    "2022-23": f"{SBRO_BASE}/nba%20odds%202022-23.xlsx",
    "2021-22": f"{SBRO_BASE}/nba%20odds%202021-22.xlsx",
}

# NBA team name variations → standard abbreviation
NBA_TEAM_MAP = {
    "Atlanta": "ATL", "Boston": "BOS", "Brooklyn": "BKN",
    "Charlotte": "CHA", "Chicago": "CHI", "Cleveland": "CLE",
    "Dallas": "DAL", "Denver": "DEN", "Detroit": "DET",
    "GoldenState": "GSW", "Golden State": "GSW",
    "Houston": "HOU", "Indiana": "IND", "LAClippers": "LAC",
    "LA Clippers": "LAC", "LALakers": "LAL", "LA Lakers": "LAL",
    "Memphis": "MEM", "Miami": "MIA", "Milwaukee": "MIL",
    "Minnesota": "MIN", "NewOrleans": "NOP", "New Orleans": "NOP",
    "NewYork": "NY", "New York": "NY", "NYKnicks": "NY",
    "OklahomaCity": "OKC", "Oklahoma City": "OKC",
    "Orlando": "ORL", "Philadelphia": "PHI", "Phoenix": "PHX",
    "Portland": "POR", "Sacramento": "SAC", "SanAntonio": "SA",
    "San Antonio": "SA", "Toronto": "TOR", "Utah": "UTAH",
    "Washington": "WSH", "Wash": "WSH",
}


def _safe_float(val: Any) -> float | None:
    """Safely convert to float, returning None on failure."""
    if val is None or val == "" or val == "NaN":
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds.

    Examples: -150 → 1.667, +200 → 3.0
    """
    if american > 0:
        return 1.0 + american / 100.0
    elif american < 0:
        return 1.0 + 100.0 / abs(american)
    else:
        return 2.0  # Even money


def download_nba_sbro(
    db: SportsDB,
    seasons: list[str] | None = None,
) -> int:
    """Download NBA historical odds from SBRO Excel files.

    SBRO provides consensus sharp closing lines in American odds format.
    We convert to decimal and compute no-vig probabilities.

    Note: SBRO stopped updating after 2023-24. For newer seasons,
    use The Odds API historical endpoint.

    Args:
        db: SportsDB instance.
        seasons: List of seasons to download.

    Returns:
        Number of odds records stored.
    """
    try:
        import openpyxl
    except ImportError:
        print("  WARNING: openpyxl not installed. Trying CSV fallback.")
        print("  Install with: pip install openpyxl")
        return _download_nba_sbro_csv_fallback(db, seasons)

    if seasons is None:
        seasons = list(SBRO_FILES.keys())

    total_stored = 0
    ODDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for season in seasons:
        url = SBRO_FILES.get(season)
        if not url:
            print(f"  No SBRO data available for {season}")
            continue

        cache_path = ODDS_CACHE_DIR / f"nba_sbro_{season.replace('-', '')}.xlsx"
        if cache_path.exists():
            print(f"  Loading NBA SBRO {season} from cache")
            data = cache_path.read_bytes()
        else:
            print(f"  Downloading NBA SBRO {season}...")
            data = _http_get(url, binary=True)
            if not data:
                print(f"  Failed to download SBRO {season}")
                continue
            cache_path.write_bytes(data)

        # Parse Excel
        try:
            wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True)
            ws = wb.active
            rows = list(ws.iter_rows(values_only=True))
            if not rows:
                continue

            headers = [str(h).strip() if h else "" for h in rows[0]]
            print(f"  SBRO columns: {headers[:15]}...")

            odds_batch = _parse_sbro_rows(headers, rows[1:], season)
            if odds_batch:
                n = db.upsert_pinnacle_odds_batch(odds_batch)
                total_stored += n
                print(f"  NBA SBRO {season}: stored {n} odds records")
            wb.close()

        except Exception as e:
            print(f"  Error parsing SBRO {season}: {e}")

    return total_stored


def _parse_sbro_rows(
    headers: list[str], rows: list[tuple], season: str
) -> list[dict]:
    """Parse SBRO Excel rows into odds records.

    SBRO format varies by season, but typically has:
    - Date, Rot (rotation number), VH (V=visitor, H=home), Team
    - Close (closing spread), ML (moneyline), 2H (2nd half)

    Games are in pairs of rows: visitor row then home row.
    """
    odds_batch: list[dict] = []

    # Find relevant column indices
    def find_col(names: list[str]) -> int | None:
        for name in names:
            for i, h in enumerate(headers):
                if h.lower() == name.lower():
                    return i
        return None

    date_col = find_col(["Date", "DATE"])
    vh_col = find_col(["VH", "V/H"])
    team_col = find_col(["Team", "TEAM"])
    ml_col = find_col(["ML", "Close", "Moneyline", "CLOSE ML"])
    final_col = find_col(["Final", "FINAL", "Score", "SCORE"])

    if date_col is None or team_col is None:
        print(f"  Could not find required columns in SBRO data")
        return []

    # Process in pairs (visitor, home)
    visitor_row = None
    for row in rows:
        if not row or not row[date_col]:
            continue

        vals = list(row)
        vh = str(vals[vh_col]).strip().upper() if vh_col and vh_col < len(vals) else ""
        team_raw = str(vals[team_col]).strip() if team_col < len(vals) else ""

        if vh == "V":
            visitor_row = vals
            continue
        elif vh == "H" and visitor_row is not None:
            home_row = vals

            # Extract data
            date_val = visitor_row[date_col]
            if isinstance(date_val, (int, float)):
                # Excel serial date
                try:
                    from datetime import timedelta
                    base = datetime(1899, 12, 30)
                    game_dt = base + timedelta(days=int(date_val))
                    game_date = game_dt.strftime("%Y-%m-%d")
                except Exception:
                    visitor_row = None
                    continue
            else:
                game_date = str(date_val).strip()
                # Try various date formats
                for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y", "%m%d%Y"):
                    try:
                        game_dt = datetime.strptime(game_date, fmt)
                        game_date = game_dt.strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        continue

            away_team_raw = str(visitor_row[team_col]).strip()
            home_team_raw = str(home_row[team_col]).strip()

            away_abbr = NBA_TEAM_MAP.get(away_team_raw, away_team_raw[:3].upper())
            home_abbr = NBA_TEAM_MAP.get(home_team_raw, home_team_raw[:3].upper())

            # Moneyline odds
            away_ml = _safe_float(visitor_row[ml_col]) if ml_col and ml_col < len(visitor_row) else None
            home_ml = _safe_float(home_row[ml_col]) if ml_col and ml_col < len(home_row) else None

            if away_ml and home_ml:
                away_dec = _american_to_decimal(away_ml)
                home_dec = _american_to_decimal(home_ml)

                h_prob, a_prob = remove_vig_2way(home_dec, away_dec)

                game_id = f"nba_{game_date.replace('-', '')}_{away_abbr}_{home_abbr}"
                ts = int(datetime.strptime(game_date, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                ).timestamp())

                odds_batch.append({
                    "game_id": game_id,
                    "ts": ts,
                    "home_odds": round(home_dec, 3),
                    "away_odds": round(away_dec, 3),
                    "draw_odds": None,
                    "home_prob_novig": round(h_prob, 4),
                    "away_prob_novig": round(a_prob, 4),
                })

            visitor_row = None

    return odds_batch


def _download_nba_sbro_csv_fallback(
    db: SportsDB, seasons: list[str] | None
) -> int:
    """Fallback when openpyxl is not installed: skip SBRO Excel."""
    print("  Skipping SBRO download (openpyxl not available)")
    print("  To install: pip install openpyxl")
    return 0


# ══════════════════════════════════════════════════════════════════════
# The Odds API Historical (for recent seasons without SBRO)
# ══════════════════════════════════════════════════════════════════════

ODDS_API_BASE = "https://api.the-odds-api.com/v4"


def download_nba_odds_api_historical(
    db: SportsDB,
    api_key: str,
    start_date: str = "2024-10-22",
    end_date: str | None = None,
    max_requests: int = 50,
) -> int:
    """Download NBA historical odds from The Odds API.

    Uses the /v4/historical endpoint which returns odds as they were
    at a specific point in time. Each request costs ~10 credits.

    Args:
        db: SportsDB instance.
        api_key: The Odds API key.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD), defaults to today.
        max_requests: Maximum API requests to make.

    Returns:
        Number of odds records stored.
    """
    from datetime import date, timedelta

    if not end_date:
        end_date = date.today().strftime("%Y-%m-%d")

    print(f"  The Odds API historical: {start_date} → {end_date}")
    print(f"  Max requests: {max_requests} (~{max_requests * 10} credits)")

    total_stored = 0
    request_count = 0

    current = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    # Sample every 2 days to cover the season efficiently
    while current <= end and request_count < max_requests:
        date_iso = current.strftime("%Y-%m-%dT12:00:00Z")
        url = (
            f"{ODDS_API_BASE}/historical/sports/basketball_nba/odds"
            f"?apiKey={api_key}&regions=eu&markets=h2h"
            f"&bookmakers=pinnacle&date={date_iso}"
        )

        cache_path = ODDS_CACHE_DIR / f"nba_hist_{current.strftime('%Y%m%d')}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                result = json.load(f)
        else:
            resp_text = _http_get(url)
            request_count += 1

            if not resp_text:
                current += timedelta(days=2)
                continue

            try:
                result = json.loads(resp_text)
            except json.JSONDecodeError:
                current += timedelta(days=2)
                continue

            # Cache response
            ODDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(result, f)

        # Parse odds from response
        data = result.get("data", []) if isinstance(result, dict) else result
        if isinstance(data, list):
            games_batch, odds_batch = _parse_odds_api_response(data)
            # Ensure games exist before inserting odds (FK constraint)
            if games_batch:
                db.upsert_games_batch(games_batch)
            if odds_batch:
                n = db.upsert_pinnacle_odds_batch(odds_batch)
                total_stored += n
                print(f"  {current}: {n} odds ({len(data)} events)")
        else:
            print(f"  {current}: unexpected response format")

        current += timedelta(days=2)
        time.sleep(1.0)  # Conservative rate limiting

    print(f"  API requests used: {request_count}")
    return total_stored


def _parse_odds_api_response(
    events: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Parse The Odds API response into game + odds records.

    Returns:
        Tuple of (games_batch, odds_batch) for upsert.
    """
    odds_batch: list[dict] = []
    games_batch: list[dict] = []

    for event in events:
        commence = event.get("commence_time", "")
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")

        if not commence or not home_team or not away_team:
            continue

        # Parse date
        try:
            dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
            game_date = dt.strftime("%Y-%m-%d")
            game_time = dt.strftime("%H:%M")
            ts = int(dt.timestamp())
        except Exception:
            continue

        home_abbr = NBA_TEAM_MAP.get(home_team, home_team[:3].upper())
        away_abbr = NBA_TEAM_MAP.get(away_team, away_team[:3].upper())
        game_id = f"nba_{game_date.replace('-', '')}_{away_abbr}_{home_abbr}"

        # Ensure game exists for FK constraint
        games_batch.append({
            "game_id": game_id,
            "sport": "nba",
            "league": "NBA",
            "home_team": home_abbr,
            "away_team": away_abbr,
            "game_date": game_date,
            "game_time": game_time,
            "home_score": None,
            "away_score": None,
            "status": "scheduled",
            "season": None,
            "venue": None,
            "extra_json": json.dumps({"source": "odds_api_hist",
                                      "home_full": home_team,
                                      "away_full": away_team}),
        })

        # Find Pinnacle h2h odds
        for bm in event.get("bookmakers", []):
            if bm.get("key") != "pinnacle":
                continue

            for market in bm.get("markets", []):
                if market.get("key") != "h2h":
                    continue

                outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                home_odds = outcomes.get(home_team)
                away_odds = outcomes.get(away_team)

                if home_odds and away_odds:
                    h_prob, a_prob = remove_vig_2way(home_odds, away_odds)
                    odds_batch.append({
                        "game_id": game_id,
                        "ts": ts,
                        "home_odds": home_odds,
                        "away_odds": away_odds,
                        "draw_odds": None,
                        "home_prob_novig": round(h_prob, 4),
                        "away_prob_novig": round(a_prob, 4),
                    })

    return games_batch, odds_batch


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download historical Pinnacle/sharp odds for backtesting.",
    )
    parser.add_argument(
        "--sport", choices=["nba", "epl", "all"], default="all",
        help="Sport to download (default: all).",
    )
    parser.add_argument(
        "--seasons", default=None,
        help="Comma-separated seasons (e.g., '2023-24,2024-25').",
    )
    parser.add_argument(
        "--odds-api-historical", action="store_true",
        help="Use The Odds API historical endpoint for recent NBA seasons.",
    )
    parser.add_argument(
        "--max-api-requests", type=int, default=50,
        help="Max Odds API historical requests (default: 50, ~500 credits).",
    )
    parser.add_argument(
        "--db", default=None,
        help="Path to SQLite database.",
    )
    args = parser.parse_args()

    db = SportsDB(args.db)
    seasons = args.seasons.split(",") if args.seasons else None

    print("=" * 60)
    print("  Historical Odds Download")
    print("=" * 60)

    total = 0

    # EPL: football-data.co.uk
    if args.sport in ("epl", "all"):
        print(f"\n{'─'*40}")
        print("EPL: football-data.co.uk Pinnacle odds")
        print(f"{'─'*40}")
        n = download_epl_pinnacle(db, seasons or ["2023-24", "2024-25", "2025-26"])
        total += n
        print(f"  Total EPL odds: {n}")

    # NBA: SBRO + optional Odds API
    if args.sport in ("nba", "all"):
        print(f"\n{'─'*40}")
        print("NBA: SBRO consensus sharp closing lines")
        print(f"{'─'*40}")
        n = download_nba_sbro(db, seasons)
        total += n
        print(f"  Total NBA SBRO odds: {n}")

        if args.odds_api_historical:
            api_key = os.environ.get("ODDS_API_KEY", "")
            if not api_key:
                # Try loading from .env
                env_path = Path(__file__).resolve().parent.parent / ".env"
                if env_path.exists():
                    for line in env_path.read_text().splitlines():
                        if line.startswith("ODDS_API_KEY="):
                            api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break

            if api_key:
                print(f"\n{'─'*40}")
                print("NBA: The Odds API historical (2024-25+)")
                print(f"{'─'*40}")
                n = download_nba_odds_api_historical(
                    db, api_key,
                    start_date="2024-10-22",
                    max_requests=args.max_api_requests,
                )
                total += n
                print(f"  Total Odds API historical: {n}")
            else:
                print("  No ODDS_API_KEY found — skipping Odds API historical")

    # Summary
    stats = db.stats()
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {total} odds records stored")
    print(f"  Total Pinnacle odds in DB: {stats['pinnacle_odds']}")
    print(f"  Games: {stats['games']} | Markets: {stats['markets']}")
    print(f"{'='*60}")

    db.close()


if __name__ == "__main__":
    main()
