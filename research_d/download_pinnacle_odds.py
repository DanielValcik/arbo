"""Download Pinnacle Odds via The Odds API.

Fetches sharp Pinnacle odds for NBA, EPL, and NFL games via The Odds API,
computes vig-removed (fair) probabilities, matches them to existing games
in the SportsDB SQLite database, and stores them in the pinnacle_odds table.

API quota is tracked via response headers and logged prominently after
every request. Responses are cached locally to avoid redundant API calls.

Usage:
    python3 research_d/download_pinnacle_odds.py --sport nba
    python3 research_d/download_pinnacle_odds.py --sport epl --historical
    python3 research_d/download_pinnacle_odds.py --sport all

Output:
    research_d/data/sports_backtest.sqlite   — pinnacle_odds table updated
    research_d/data/odds_cache/              — cached API responses (JSON)
"""

from __future__ import annotations

import argparse
import json
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

    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()
    SSL_CONTEXT.check_hostname = False
    SSL_CONTEXT.verify_mode = ssl.CERT_NONE

# Append project root to sys.path so we can import sports_db
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research_d.sports_db import SportsDB

# ── Constants ────────────────────────────────────────────────────────

ODDS_API_BASE = "https://api.the-odds-api.com/v4"

CACHE_DIR = Path(__file__).parent / "data" / "odds_cache"

# Sport keys for The Odds API → our internal sport/league labels
SPORT_CONFIG: dict[str, dict[str, str]] = {
    "nba": {
        "api_key": "basketball_nba",
        "sport": "nba",
        "league": "NBA",
    },
    "epl": {
        "api_key": "soccer_epl",
        "sport": "epl",
        "league": "Premier League",
    },
    "nfl": {
        "api_key": "americanfootball_nfl",
        "sport": "nfl",
        "league": "NFL",
    },
}

# Rate limiting: conservative, one request at a time
REQUEST_DELAY = 1.0

# Minimum remaining quota before we refuse to make more requests
MIN_QUOTA_REMAINING = 10


# ── Team Name → Abbreviation Mappings ────────────────────────────────

NBA_TEAMS: dict[str, str] = {
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
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "LA Clippers": "LAC",
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

EPL_TEAMS: dict[str, str] = {
    "Arsenal": "ARS",
    "Aston Villa": "AVL",
    "AFC Bournemouth": "BOU",
    "Bournemouth": "BOU",
    "Brentford": "BRE",
    "Brighton and Hove Albion": "BHA",
    "Brighton": "BHA",
    "Chelsea": "CHE",
    "Crystal Palace": "CRY",
    "Everton": "EVE",
    "Fulham": "FUL",
    "Ipswich Town": "IPS",
    "Ipswich": "IPS",
    "Leicester City": "LEI",
    "Leicester": "LEI",
    "Liverpool": "LIV",
    "Manchester City": "MCI",
    "Manchester United": "MUN",
    "Newcastle United": "NEW",
    "Newcastle": "NEW",
    "Nottingham Forest": "NFO",
    "Southampton": "SOU",
    "Tottenham Hotspur": "TOT",
    "Tottenham": "TOT",
    "West Ham United": "WHU",
    "West Ham": "WHU",
    "Wolverhampton Wanderers": "WOL",
    "Wolves": "WOL",
}

NFL_TEAMS: dict[str, str] = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "LA Chargers": "LAC",
    "LA Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

TEAM_MAPS: dict[str, dict[str, str]] = {
    "nba": NBA_TEAMS,
    "epl": EPL_TEAMS,
    "nfl": NFL_TEAMS,
}


# ── Quota Tracker ────────────────────────────────────────────────────


class QuotaTracker:
    """Tracks The Odds API quota from response headers."""

    def __init__(self) -> None:
        self.requests_used: int | None = None
        self.requests_remaining: int | None = None
        self.last_updated: datetime | None = None

    def update_from_headers(self, headers: dict[str, str]) -> None:
        """Update quota from HTTP response headers."""
        used = headers.get("x-requests-used")
        remaining = headers.get("x-requests-remaining")
        if used is not None:
            self.requests_used = int(used)
        if remaining is not None:
            self.requests_remaining = int(remaining)
        self.last_updated = datetime.now(timezone.utc)

    def is_safe(self) -> bool:
        """Check if we have enough remaining quota to make a request."""
        if self.requests_remaining is None:
            return True  # First request, no data yet
        return self.requests_remaining >= MIN_QUOTA_REMAINING

    def log(self) -> None:
        """Print current quota status."""
        if self.requests_used is not None:
            print(
                f"  [QUOTA] Used: {self.requests_used}, "
                f"Remaining: {self.requests_remaining}"
            )
        else:
            print("  [QUOTA] Not yet known (first request)")

    def __repr__(self) -> str:
        return (
            f"QuotaTracker(used={self.requests_used}, "
            f"remaining={self.requests_remaining})"
        )


# Global quota tracker
_quota = QuotaTracker()


# ── HTTP Helpers ─────────────────────────────────────────────────────


def _load_api_key() -> str:
    """Load ODDS_API_KEY from .env file at project root.

    Returns:
        The API key string.

    Raises:
        SystemExit: If the key is not found.
    """
    # Try environment first
    key = os.environ.get("ODDS_API_KEY", "")
    if key:
        return key

    # Load from .env file
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() == "ODDS_API_KEY":
                    key = v.strip().strip("'\"")
                    if key:
                        os.environ["ODDS_API_KEY"] = key
                        return key

    print("ERROR: ODDS_API_KEY not found in environment or .env file")
    sys.exit(1)


def _http_get_odds(
    url: str, retries: int = 3
) -> tuple[Any | None, dict[str, str]]:
    """GET JSON from The Odds API with retry and quota tracking.

    Args:
        url: The full URL to fetch.
        retries: Number of retry attempts.

    Returns:
        Tuple of (parsed JSON response or None, response headers dict).
    """
    headers_out: dict[str, str] = {}
    for attempt in range(retries):
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "ArboResearch/1.0",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(
                req, timeout=30, context=SSL_CONTEXT
            ) as resp:
                # Capture quota headers
                for h in ("x-requests-used", "x-requests-remaining"):
                    val = resp.headers.get(h)
                    if val is not None:
                        headers_out[h] = val
                _quota.update_from_headers(headers_out)

                body = resp.read().decode()
                return json.loads(body), headers_out

        except urllib.error.HTTPError as e:
            # Capture quota headers even on error
            for h in ("x-requests-used", "x-requests-remaining"):
                val = e.headers.get(h) if e.headers else None
                if val is not None:
                    headers_out[h] = val
            _quota.update_from_headers(headers_out)

            if e.code == 429:
                wait = 2 ** (attempt + 1)
                print(f"    Rate limited (429), waiting {wait}s...")
                time.sleep(wait)
            elif e.code == 401:
                print("ERROR: Invalid API key (401 Unauthorized)")
                return None, headers_out
            elif e.code == 422:
                # Often means the sport is out of season
                print(f"    Unprocessable (422) — sport may be out of season")
                return None, headers_out
            elif e.code >= 500:
                time.sleep(1)
            else:
                print(f"    HTTP {e.code}: {e.reason}")
                return None, headers_out
        except Exception as e:
            print(f"    Request error: {e}")
            time.sleep(0.5)

    return None, headers_out


# ── Caching ──────────────────────────────────────────────────────────


def _cache_path(sport_key: str, endpoint: str) -> Path:
    """Get the cache file path for a given sport and endpoint.

    Args:
        sport_key: Internal sport key (e.g., "nba").
        endpoint: API endpoint type ("odds" or "scores").

    Returns:
        Path to the cache JSON file.
    """
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    return CACHE_DIR / f"{sport_key}_{endpoint}_{today}.json"


def _load_cache(sport_key: str, endpoint: str) -> list[dict[str, Any]] | None:
    """Load cached API response if available and fresh (same day).

    Args:
        sport_key: Internal sport key.
        endpoint: API endpoint type.

    Returns:
        Cached data or None if cache miss.
    """
    path = _cache_path(sport_key, endpoint)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        print(f"  Loaded from cache: {path.name} ({len(data)} events)")
        return data
    return None


def _save_cache(
    sport_key: str, endpoint: str, data: list[dict[str, Any]]
) -> None:
    """Save API response to cache.

    Args:
        sport_key: Internal sport key.
        endpoint: API endpoint type.
        data: The parsed JSON response to cache.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(sport_key, endpoint)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved to cache: {path.name}")


# ── Vig Removal ──────────────────────────────────────────────────────


def remove_vig(
    home_odds: float, away_odds: float, draw_odds: float | None = None
) -> tuple[float, float]:
    """Remove bookmaker vig to compute fair probabilities.

    Uses the multiplicative method: divide each implied probability by
    the total overround to get fair (no-vig) probabilities.

    Args:
        home_odds: Decimal odds for home team.
        away_odds: Decimal odds for away team.
        draw_odds: Decimal odds for draw (soccer only).

    Returns:
        Tuple of (home_prob_novig, away_prob_novig).
    """
    implied_home = 1.0 / home_odds
    implied_away = 1.0 / away_odds
    if draw_odds:
        implied_draw = 1.0 / draw_odds
        total = implied_home + implied_away + implied_draw
    else:
        total = implied_home + implied_away
    return implied_home / total, implied_away / total


# ── Team Matching ────────────────────────────────────────────────────


def team_to_abbrev(team_name: str, sport: str) -> str | None:
    """Convert a full team name to its abbreviation.

    Args:
        team_name: Full team name from The Odds API (e.g., "Los Angeles Lakers").
        sport: Internal sport key ("nba", "epl", "nfl").

    Returns:
        Team abbreviation (e.g., "LAL") or None if no match.
    """
    team_map = TEAM_MAPS.get(sport, {})
    abbrev = team_map.get(team_name)
    if abbrev:
        return abbrev

    # Fuzzy fallback: try case-insensitive and partial matching
    name_lower = team_name.lower()
    for full_name, ab in team_map.items():
        if full_name.lower() == name_lower:
            return ab
    for full_name, ab in team_map.items():
        # Check if key is a substring of the team name or vice versa
        if full_name.lower() in name_lower or name_lower in full_name.lower():
            return ab

    return None


def build_game_id(
    sport: str, game_date: str, away_abbrev: str, home_abbrev: str
) -> str:
    """Build a game_id in our standard format.

    Args:
        sport: Internal sport key.
        game_date: Date string in YYYY-MM-DD format.
        away_abbrev: Away team abbreviation.
        home_abbrev: Home team abbreviation.

    Returns:
        Game ID like "nba_20260315_LAL_BOS".
    """
    date_compact = game_date.replace("-", "")
    return f"{sport}_{date_compact}_{away_abbrev}_{home_abbrev}"


# ── API Fetch Functions ──────────────────────────────────────────────


def fetch_odds(api_key: str, sport_key: str) -> list[dict[str, Any]] | None:
    """Fetch upcoming game odds from The Odds API.

    Args:
        api_key: The Odds API key.
        sport_key: The Odds API sport key (e.g., "basketball_nba").

    Returns:
        List of event dicts or None on failure.
    """
    url = (
        f"{ODDS_API_BASE}/sports/{sport_key}/odds/"
        f"?apiKey={api_key}&regions=eu&markets=h2h&bookmakers=pinnacle"
    )
    print(f"  Fetching odds: {sport_key}")
    data, _ = _http_get_odds(url)
    _quota.log()
    return data


def fetch_scores(
    api_key: str, sport_key: str, days_from: int = 3
) -> list[dict[str, Any]] | None:
    """Fetch recent completed game scores from The Odds API.

    Args:
        api_key: The Odds API key.
        sport_key: The Odds API sport key.
        days_from: Number of days to look back (max 3 for free tier).

    Returns:
        List of event dicts with scores, or None on failure.
    """
    url = (
        f"{ODDS_API_BASE}/sports/{sport_key}/scores/"
        f"?apiKey={api_key}&daysFrom={days_from}"
    )
    print(f"  Fetching scores: {sport_key} (last {days_from} days)")
    data, _ = _http_get_odds(url)
    _quota.log()
    return data


def fetch_sports(api_key: str) -> list[dict[str, Any]] | None:
    """Fetch list of available sports from The Odds API.

    Useful for verifying which sports are in-season.

    Args:
        api_key: The Odds API key.

    Returns:
        List of sport dicts, or None on failure.
    """
    url = f"{ODDS_API_BASE}/sports/?apiKey={api_key}"
    print("  Fetching available sports...")
    data, _ = _http_get_odds(url)
    _quota.log()
    return data


# ── Processing ───────────────────────────────────────────────────────


def extract_pinnacle_odds_from_event(
    event: dict[str, Any], sport: str
) -> dict[str, Any] | None:
    """Extract Pinnacle h2h odds from an Odds API event.

    Args:
        event: A single event dict from The Odds API response.
        sport: Internal sport key.

    Returns:
        Dict with keys matching SportsDB.upsert_pinnacle_odds_batch schema,
        or None if Pinnacle odds not found.
    """
    home_team_full = event.get("home_team", "")
    away_team_full = event.get("away_team", "")
    commence_time = event.get("commence_time", "")

    # Parse commence time to get date and unix timestamp
    try:
        dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        game_date = dt.strftime("%Y-%m-%d")
        ts = int(dt.timestamp())
    except (ValueError, AttributeError):
        print(f"    WARNING: Cannot parse commence_time: {commence_time}")
        return None

    # Convert team names to abbreviations
    home_abbrev = team_to_abbrev(home_team_full, sport)
    away_abbrev = team_to_abbrev(away_team_full, sport)

    if not home_abbrev:
        print(f"    WARNING: Unknown home team: {home_team_full!r} ({sport})")
        return None
    if not away_abbrev:
        print(f"    WARNING: Unknown away team: {away_team_full!r} ({sport})")
        return None

    # Find Pinnacle in bookmakers list
    bookmakers = event.get("bookmakers", [])
    pinnacle = None
    for bm in bookmakers:
        if bm.get("key") == "pinnacle":
            pinnacle = bm
            break

    if not pinnacle:
        return None

    # Extract h2h market
    markets = pinnacle.get("markets", [])
    h2h_market = None
    for m in markets:
        if m.get("key") == "h2h":
            h2h_market = m
            break

    if not h2h_market:
        return None

    # Parse outcomes
    outcomes = h2h_market.get("outcomes", [])
    home_odds: float | None = None
    away_odds: float | None = None
    draw_odds: float | None = None

    for outcome in outcomes:
        name = outcome.get("name", "")
        price = outcome.get("price")
        if price is None:
            continue
        price = float(price)
        if name == home_team_full:
            home_odds = price
        elif name == away_team_full:
            away_odds = price
        elif name.lower() == "draw":
            draw_odds = price

    if home_odds is None or away_odds is None:
        print(
            f"    WARNING: Incomplete odds for "
            f"{away_team_full} @ {home_team_full}"
        )
        return None

    # Compute no-vig probabilities
    home_prob, away_prob = remove_vig(home_odds, away_odds, draw_odds)

    # Use current time as the odds snapshot timestamp (when we observed them)
    snapshot_ts = int(datetime.now(timezone.utc).timestamp())

    game_id = build_game_id(sport, game_date, away_abbrev, home_abbrev)

    return {
        "game_id": game_id,
        "ts": snapshot_ts,
        "home_odds": home_odds,
        "away_odds": away_odds,
        "draw_odds": draw_odds,
        "home_prob_novig": round(home_prob, 6),
        "away_prob_novig": round(away_prob, 6),
        # Extra metadata for game upsert (not stored in pinnacle_odds)
        "_home_team": home_abbrev,
        "_away_team": away_abbrev,
        "_home_team_full": home_team_full,
        "_away_team_full": away_team_full,
        "_game_date": game_date,
        "_game_time": dt.strftime("%H:%M"),
    }


def process_sport(
    api_key: str,
    sport: str,
    historical: bool = False,
    db: SportsDB | None = None,
) -> dict[str, int]:
    """Fetch and store Pinnacle odds for a single sport.

    Args:
        api_key: The Odds API key.
        sport: Internal sport key ("nba", "epl", "nfl").
        historical: If True, also fetch scores for completed games.
        db: SportsDB instance (created if None).

    Returns:
        Stats dict with counts of games processed, matched, stored, etc.
    """
    config = SPORT_CONFIG.get(sport)
    if not config:
        print(f"ERROR: Unknown sport: {sport!r}")
        return {"error": 1}

    api_sport_key = config["api_key"]
    league = config["league"]
    own_db = db is None
    if own_db:
        db = SportsDB()

    stats: dict[str, int] = {
        "events_fetched": 0,
        "pinnacle_found": 0,
        "games_upserted": 0,
        "odds_stored": 0,
        "teams_unmatched": 0,
        "scores_fetched": 0,
    }

    # ── 1. Check quota before proceeding ──
    if not _quota.is_safe():
        print(
            f"  QUOTA WARNING: Only {_quota.requests_remaining} requests "
            f"remaining. Skipping {sport} to preserve quota."
        )
        return stats

    # ── 2. Fetch odds (upcoming games) ──
    cached_odds = _load_cache(sport, "odds")
    if cached_odds is not None:
        odds_events = cached_odds
    else:
        odds_events = fetch_odds(api_key, api_sport_key)
        if odds_events is not None:
            _save_cache(sport, "odds", odds_events)
            time.sleep(REQUEST_DELAY)
        else:
            odds_events = []
            print(f"  No odds data returned for {sport}")

    stats["events_fetched"] += len(odds_events)

    # ── 3. Fetch scores (historical completed games) ──
    scores_events: list[dict[str, Any]] = []
    if historical:
        if not _quota.is_safe():
            print(
                f"  QUOTA WARNING: Skipping scores fetch for {sport}"
            )
        else:
            cached_scores = _load_cache(sport, "scores")
            if cached_scores is not None:
                scores_events = cached_scores
            else:
                scores_data = fetch_scores(api_key, api_sport_key, days_from=3)
                if scores_data is not None:
                    scores_events = scores_data
                    _save_cache(sport, "scores", scores_events)
                    time.sleep(REQUEST_DELAY)
                else:
                    print(f"  No scores data returned for {sport}")

        stats["scores_fetched"] = len(scores_events)

    # ── 4. Process all events ──
    all_events = odds_events
    # Merge scores events (may overlap with odds events)
    seen_event_ids: set[str] = {e.get("id", "") for e in all_events}
    for se in scores_events:
        if se.get("id", "") not in seen_event_ids:
            all_events.append(se)
            seen_event_ids.add(se.get("id", ""))

    odds_records: list[dict[str, Any]] = []

    for event in all_events:
        result = extract_pinnacle_odds_from_event(event, sport)
        if result is None:
            continue

        stats["pinnacle_found"] += 1

        # Upsert the game into the games table
        game_id = result["game_id"]

        # Determine game status from scores data
        status = "scheduled"
        home_score: int | None = None
        away_score: int | None = None
        event_completed = event.get("completed", False)
        if event_completed:
            status = "final"
            # Extract scores from the event
            scores_list = event.get("scores", [])
            if scores_list:
                for s in scores_list:
                    if s.get("name") == event.get("home_team"):
                        try:
                            home_score = int(s["score"])
                        except (ValueError, KeyError, TypeError):
                            pass
                    elif s.get("name") == event.get("away_team"):
                        try:
                            away_score = int(s["score"])
                        except (ValueError, KeyError, TypeError):
                            pass

        db.upsert_game(
            game_id=game_id,
            sport=sport,
            league=league,
            home_team=result["_home_team"],
            away_team=result["_away_team"],
            game_date=result["_game_date"],
            game_time=result["_game_time"],
            home_score=home_score,
            away_score=away_score,
            status=status,
            extra_json=json.dumps({
                "home_team_full": result["_home_team_full"],
                "away_team_full": result["_away_team_full"],
                "odds_api_id": event.get("id", ""),
            }),
        )
        stats["games_upserted"] += 1

        # Build odds record for batch upsert (strip metadata keys)
        odds_records.append({
            "game_id": result["game_id"],
            "ts": result["ts"],
            "home_odds": result["home_odds"],
            "away_odds": result["away_odds"],
            "draw_odds": result["draw_odds"],
            "home_prob_novig": result["home_prob_novig"],
            "away_prob_novig": result["away_prob_novig"],
        })

    # ── 5. Batch store odds ──
    if odds_records:
        stored = db.upsert_pinnacle_odds_batch(odds_records)
        stats["odds_stored"] = stored

    if own_db:
        db.close()

    return stats


# ── CLI & Main ───────────────────────────────────────────────────────


def print_summary(all_stats: dict[str, dict[str, int]]) -> None:
    """Print a formatted summary of all downloaded data.

    Args:
        all_stats: Dict mapping sport key to stats dict.
    """
    print(f"\n{'=' * 60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'=' * 60}")

    totals: dict[str, int] = {}
    for sport, stats in all_stats.items():
        print(f"\n  {sport.upper()}:")
        for key, val in stats.items():
            if key == "error":
                continue
            label = key.replace("_", " ").title()
            print(f"    {label:<22} {val:>6}")
            totals[key] = totals.get(key, 0) + val

    if len(all_stats) > 1:
        print(f"\n  TOTAL:")
        for key, val in totals.items():
            label = key.replace("_", " ").title()
            print(f"    {label:<22} {val:>6}")

    print(f"\n  Final quota status:")
    _quota.log()


def print_db_stats(db: SportsDB) -> None:
    """Print database statistics.

    Args:
        db: SportsDB instance.
    """
    stats = db.stats()
    print(f"\n{'─' * 60}")
    print("DATABASE STATISTICS")
    print(f"{'─' * 60}")
    print(f"  Games:           {stats['games']:>6}")
    print(f"    NBA:           {stats['games_nba']:>6}")
    print(f"    EPL:           {stats['games_epl']:>6}")
    print(f"    NFL:           {stats['games_nfl']:>6}")
    print(f"  Markets:         {stats['markets']:>6}")
    print(f"  Prices:          {stats['prices']:>6}")
    print(f"  Pinnacle Odds:   {stats['pinnacle_odds']:>6}")
    print(f"  Ratings:         {stats['ratings']:>6}")


def main() -> None:
    """Main entry point for the Pinnacle odds downloader."""
    parser = argparse.ArgumentParser(
        description="Download Pinnacle odds via The Odds API"
    )
    parser.add_argument(
        "--sport",
        choices=["nba", "epl", "nfl", "all"],
        default="all",
        help="Sport to download (default: all)",
    )
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Also fetch completed game scores (uses /scores endpoint)",
    )
    parser.add_argument(
        "--list-sports",
        action="store_true",
        help="List available sports from the API and exit",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached responses and fetch fresh data",
    )
    args = parser.parse_args()

    # ── Load API key ──
    api_key = _load_api_key()
    print(f"API key loaded (ends in ...{api_key[-4:]})")

    # ── List sports mode ──
    if args.list_sports:
        sports = fetch_sports(api_key)
        if sports:
            print(f"\nAvailable sports ({len(sports)}):")
            for s in sports:
                active = "ACTIVE" if s.get("active") else "inactive"
                print(f"  {s['key']:<40} {s.get('title', ''):<30} [{active}]")
        return

    # ── Clear cache if requested ──
    if args.no_cache and CACHE_DIR.exists():
        import shutil

        shutil.rmtree(CACHE_DIR)
        print("Cache cleared.")

    # ── Determine sports to process ──
    if args.sport == "all":
        sports_to_process = list(SPORT_CONFIG.keys())
    else:
        sports_to_process = [args.sport]

    print(f"\nSports to download: {', '.join(s.upper() for s in sports_to_process)}")
    print(f"Historical scores: {'yes' if args.historical else 'no'}")
    print(f"Cache dir: {CACHE_DIR}")
    print()

    t_start = time.time()

    # ── Process each sport ──
    db = SportsDB()
    all_stats: dict[str, dict[str, int]] = {}

    for sport in sports_to_process:
        print(f"\n{'─' * 40}")
        print(f"Processing: {sport.upper()}")
        print(f"{'─' * 40}")

        if not _quota.is_safe():
            print(
                f"\nQUOTA EXHAUSTED: Only {_quota.requests_remaining} "
                f"requests remaining. Stopping to preserve quota."
            )
            break

        stats = process_sport(
            api_key=api_key,
            sport=sport,
            historical=args.historical,
            db=db,
        )
        all_stats[sport] = stats

    # ── Summary ──
    elapsed = time.time() - t_start
    print_summary(all_stats)
    print_db_stats(db)
    print(f"\n  Elapsed time: {elapsed:.1f}s")

    db.close()


if __name__ == "__main__":
    main()
