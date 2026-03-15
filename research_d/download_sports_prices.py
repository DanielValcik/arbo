"""Download Sports Market Price History from Polymarket CLOB API.

Discovers resolved sports markets (NBA, EPL, NFL, UFC, etc.) via the
Gamma API and downloads minute-level price history from the CLOB
/prices-history endpoint. Stores data in SQLite via SportsDB for
backtesting the Live Edge Harvester (LEH).

Data availability: Polymarket retains ~30 days of price history for
resolved markets. Run this script regularly to capture data before
it expires.

Usage:
    python3 research_d/download_sports_prices.py --sport nba
    python3 research_d/download_sports_prices.py --sport epl --fidelity 1
    python3 research_d/download_sports_prices.py --sport all --refresh-events

Output:
    research_d/data/sports_backtest.sqlite    -- SQLite database (via SportsDB)
    research_d/data/sports_events_cache.json  -- Gamma API event cache
    research_d/data/sports_download_meta.json -- Download metadata
"""

import argparse
import json
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

# Add parent to path so we can import sports_db
sys.path.insert(0, str(Path(__file__).parent.parent))

from research_d.sports_db import SportsDB

DATA_DIR = Path(__file__).parent / "data"
EVENTS_CACHE = DATA_DIR / "sports_events_cache.json"
META_PATH = DATA_DIR / "sports_download_meta.json"

CLOB_BASE = "https://clob.polymarket.com"
GAMMA_BASE = "https://gamma-api.polymarket.com"

# Rate limiting
GAMMA_DELAY = 0.2  # 5 req/s for Gamma (500/10s limit)
CLOB_DELAY = 0.1   # 10 req/s for CLOB (1500/10s limit)

# Stop after this many consecutive empty price responses
CONSECUTIVE_EMPTY_STOP = 200

# ── Sport tag mapping ────────────────────────────────────────────────

# Tags used by Gamma API for different sports.
# Multiple tags may refer to the same sport.
SPORT_TAGS: dict[str, list[str]] = {
    "nba": ["nba", "basketball"],
    "epl": ["epl", "premier-league", "premier league"],
    "nfl": ["nfl", "football"],
    "soccer": ["soccer", "mls", "la-liga", "serie-a", "bundesliga", "ligue-1", "champions-league"],
    "mma": ["mma", "ufc"],
    "mlb": ["mlb", "baseball"],
    "nhl": ["nhl", "hockey"],
    "ncaab": ["ncaab", "ncaa-basketball", "march-madness"],
}

# Gamma API tag_slug values to query for each sport filter
GAMMA_SLUGS: dict[str, list[str]] = {
    "nba": ["nba"],
    "epl": ["epl", "premier-league"],
    "nfl": ["nfl"],
    "soccer": ["soccer"],
    "mma": ["mma", "ufc"],
    "mlb": ["mlb"],
    "nhl": ["nhl"],
    "ncaab": ["ncaab"],
    "all": ["sports", "nba", "nfl", "epl", "premier-league", "soccer",
            "mma", "ufc", "mlb", "nhl", "ncaab"],
}

# ── NBA team name -> abbreviation (30 teams) ────────────────────────

NBA_TEAMS: dict[str, str] = {
    # Eastern Conference — Atlantic
    "boston celtics": "BOS",
    "celtics": "BOS",
    "brooklyn nets": "BKN",
    "nets": "BKN",
    "new york knicks": "NYK",
    "knicks": "NYK",
    "philadelphia 76ers": "PHI",
    "76ers": "PHI",
    "sixers": "PHI",
    "toronto raptors": "TOR",
    "raptors": "TOR",
    # Eastern Conference — Central
    "chicago bulls": "CHI",
    "bulls": "CHI",
    "cleveland cavaliers": "CLE",
    "cavaliers": "CLE",
    "cavs": "CLE",
    "detroit pistons": "DET",
    "pistons": "DET",
    "indiana pacers": "IND",
    "pacers": "IND",
    "milwaukee bucks": "MIL",
    "bucks": "MIL",
    # Eastern Conference — Southeast
    "atlanta hawks": "ATL",
    "hawks": "ATL",
    "charlotte hornets": "CHA",
    "hornets": "CHA",
    "miami heat": "MIA",
    "heat": "MIA",
    "orlando magic": "ORL",
    "magic": "ORL",
    "washington wizards": "WAS",
    "wizards": "WAS",
    # Western Conference — Northwest
    "denver nuggets": "DEN",
    "nuggets": "DEN",
    "minnesota timberwolves": "MIN",
    "timberwolves": "MIN",
    "wolves": "MIN",
    "oklahoma city thunder": "OKC",
    "thunder": "OKC",
    "portland trail blazers": "POR",
    "trail blazers": "POR",
    "blazers": "POR",
    "utah jazz": "UTA",
    "jazz": "UTA",
    # Western Conference — Pacific
    "golden state warriors": "GSW",
    "warriors": "GSW",
    "los angeles clippers": "LAC",
    "la clippers": "LAC",
    "clippers": "LAC",
    "los angeles lakers": "LAL",
    "la lakers": "LAL",
    "lakers": "LAL",
    "phoenix suns": "PHX",
    "suns": "PHX",
    "sacramento kings": "SAC",
    "kings": "SAC",
    # Western Conference — Southwest
    "dallas mavericks": "DAL",
    "mavericks": "DAL",
    "mavs": "DAL",
    "houston rockets": "HOU",
    "rockets": "HOU",
    "memphis grizzlies": "MEM",
    "grizzlies": "MEM",
    "new orleans pelicans": "NOP",
    "pelicans": "NOP",
    "san antonio spurs": "SAS",
    "spurs": "SAS",
}

# ── EPL team name -> abbreviation (20 teams, 2025-26 season) ────────

EPL_TEAMS: dict[str, str] = {
    "arsenal": "ARS",
    "aston villa": "AVL",
    "villa": "AVL",
    "bournemouth": "BOU",
    "brentford": "BRE",
    "brighton": "BHA",
    "brighton & hove albion": "BHA",
    "brighton and hove albion": "BHA",
    "chelsea": "CHE",
    "crystal palace": "CRY",
    "palace": "CRY",
    "everton": "EVE",
    "fulham": "FUL",
    "ipswich": "IPS",
    "ipswich town": "IPS",
    "leicester": "LEI",
    "leicester city": "LEI",
    "liverpool": "LIV",
    "manchester city": "MCI",
    "man city": "MCI",
    "manchester united": "MUN",
    "man united": "MUN",
    "man utd": "MUN",
    "newcastle": "NEW",
    "newcastle united": "NEW",
    "nottingham forest": "NFO",
    "forest": "NFO",
    "southampton": "SOU",
    "tottenham": "TOT",
    "tottenham hotspur": "TOT",
    "west ham": "WHU",
    "west ham united": "WHU",
    "wolverhampton": "WOL",
    "wolverhampton wanderers": "WOL",
    "wolves": "WOL",
}

# ── NFL team name -> abbreviation (32 teams) ────────────────────────

NFL_TEAMS: dict[str, str] = {
    "arizona cardinals": "ARI",
    "cardinals": "ARI",
    "atlanta falcons": "ATL",
    "falcons": "ATL",
    "baltimore ravens": "BAL",
    "ravens": "BAL",
    "buffalo bills": "BUF",
    "bills": "BUF",
    "carolina panthers": "CAR",
    "panthers": "CAR",
    "chicago bears": "CHI",
    "bears": "CHI",
    "cincinnati bengals": "CIN",
    "bengals": "CIN",
    "cleveland browns": "CLE",
    "browns": "CLE",
    "dallas cowboys": "DAL",
    "cowboys": "DAL",
    "denver broncos": "DEN",
    "broncos": "DEN",
    "detroit lions": "DET",
    "lions": "DET",
    "green bay packers": "GB",
    "packers": "GB",
    "houston texans": "HOU",
    "texans": "HOU",
    "indianapolis colts": "IND",
    "colts": "IND",
    "jacksonville jaguars": "JAX",
    "jaguars": "JAX",
    "kansas city chiefs": "KC",
    "chiefs": "KC",
    "las vegas raiders": "LV",
    "raiders": "LV",
    "los angeles chargers": "LAC",
    "la chargers": "LAC",
    "chargers": "LAC",
    "los angeles rams": "LAR",
    "la rams": "LAR",
    "rams": "LAR",
    "miami dolphins": "MIA",
    "dolphins": "MIA",
    "minnesota vikings": "MIN",
    "vikings": "MIN",
    "new england patriots": "NE",
    "patriots": "NE",
    "new orleans saints": "NO",
    "saints": "NO",
    "new york giants": "NYG",
    "giants": "NYG",
    "new york jets": "NYJ",
    "jets": "NYJ",
    "philadelphia eagles": "PHI",
    "eagles": "PHI",
    "pittsburgh steelers": "PIT",
    "steelers": "PIT",
    "san francisco 49ers": "SF",
    "49ers": "SF",
    "seattle seahawks": "SEA",
    "seahawks": "SEA",
    "tampa bay buccaneers": "TB",
    "buccaneers": "TB",
    "bucs": "TB",
    "tennessee titans": "TEN",
    "titans": "TEN",
    "washington commanders": "WSH",
    "commanders": "WSH",
}

# Consolidated lookup by sport
_TEAM_MAPS: dict[str, dict[str, str]] = {
    "nba": NBA_TEAMS,
    "epl": EPL_TEAMS,
    "nfl": NFL_TEAMS,
}


# ── Parsing helpers ──────────────────────────────────────────────────


def parse_sport_from_tags(tags: list[str]) -> str | None:
    """Extract canonical sport identifier from Gamma API tags.

    Args:
        tags: List of tag strings from a Gamma event/market.

    Returns:
        Canonical sport string (e.g., "nba", "epl") or None if no match.
    """
    if not tags:
        return None
    lower_tags = {t.lower().strip() for t in tags}
    for sport, sport_tags in SPORT_TAGS.items():
        for st in sport_tags:
            if st in lower_tags:
                return sport
    return None


def parse_teams_from_question(question: str) -> tuple[str, str] | None:
    """Extract (team_a, team_b) from a Polymarket question string.

    Handles common patterns:
        - "Will the Los Angeles Lakers beat the Boston Celtics?"
        - "Los Angeles Lakers vs Boston Celtics"
        - "Will Arsenal beat Chelsea?"
        - "Lakers vs. Celtics"
        - "Will the Lakers win against the Celtics?"

    Args:
        question: Market question text.

    Returns:
        Tuple of (team_a, team_b) as raw strings, or None if unparseable.
        team_a is typically the subject (home/favored team).
    """
    q = question.strip()

    # Pattern 1: "Will [the] X beat [the] Y?"
    m = re.search(
        r"[Ww]ill\s+(?:the\s+)?(.+?)\s+beat\s+(?:the\s+)?(.+?)[\?\.]",
        q,
    )
    if m:
        return (m.group(1).strip(), m.group(2).strip())

    # Pattern 2: "Will [the] X win against [the] Y?"
    m = re.search(
        r"[Ww]ill\s+(?:the\s+)?(.+?)\s+win\s+(?:against|over|vs\.?)\s+(?:the\s+)?(.+?)[\?\.]",
        q,
    )
    if m:
        return (m.group(1).strip(), m.group(2).strip())

    # Pattern 3: "X vs[.] Y" (anywhere in the string)
    m = re.search(
        r"(.+?)\s+vs\.?\s+(.+?)(?:\s*[\?\.\-\|]|$)",
        q,
    )
    if m:
        a = m.group(1).strip().lstrip("Will the ").lstrip("Will ")
        return (a, m.group(2).strip())

    # Pattern 4: "Will [the] X defeat [the] Y?"
    m = re.search(
        r"[Ww]ill\s+(?:the\s+)?(.+?)\s+defeat\s+(?:the\s+)?(.+?)[\?\.]",
        q,
    )
    if m:
        return (m.group(1).strip(), m.group(2).strip())

    # Pattern 5: "X to beat Y" / "X to win"
    m = re.search(
        r"(.+?)\s+to\s+(?:beat|defeat|win\s+(?:against|over|vs\.?))\s+(?:the\s+)?(.+?)[\?\.]",
        q,
    )
    if m:
        return (m.group(1).strip(), m.group(2).strip())

    return None


def team_to_abbreviation(team_name: str, sport: str) -> str:
    """Convert a team's full or partial name to its standard abbreviation.

    Tries exact match first, then substring match against known teams
    for the given sport. Falls back to the raw name (uppercased, first 3 chars)
    if no match is found.

    Args:
        team_name: Full or partial team name (e.g., "Los Angeles Lakers").
        sport: Sport identifier (e.g., "nba", "epl", "nfl").

    Returns:
        Team abbreviation (e.g., "LAL") or a generated fallback.
    """
    team_map = _TEAM_MAPS.get(sport, {})
    lower = team_name.lower().strip()

    # Exact match
    if lower in team_map:
        return team_map[lower]

    # Substring match — try longest match first to avoid false positives
    # (e.g., "LA Clippers" should not match "LA Lakers")
    candidates: list[tuple[str, str]] = []
    for pattern, abbrev in team_map.items():
        if pattern in lower or lower in pattern:
            candidates.append((pattern, abbrev))

    if candidates:
        # Prefer the longest matching pattern
        candidates.sort(key=lambda x: len(x[0]), reverse=True)
        return candidates[0][1]

    # Fallback: uppercase first 3 characters
    cleaned = re.sub(r"[^a-zA-Z\s]", "", team_name).strip()
    words = cleaned.split()
    if len(words) >= 2:
        # Use first letter of each word (up to 3)
        return "".join(w[0].upper() for w in words[:3])
    return cleaned[:3].upper() or "UNK"


def generate_game_id(sport: str, game_date: str, home_team: str, away_team: str) -> str:
    """Generate a deterministic game ID from its components.

    Format: {sport}_{YYYYMMDD}_{HOME}_{AWAY}

    Args:
        sport: Sport identifier (e.g., "nba").
        game_date: Date string in YYYY-MM-DD format.
        home_team: Home team abbreviation.
        away_team: Away team abbreviation.

    Returns:
        Game ID string (e.g., "nba_20260315_LAL_BOS").
    """
    date_compact = game_date.replace("-", "")
    return f"{sport}_{date_compact}_{home_team}_{away_team}"


def _sport_to_league(sport: str) -> str:
    """Map sport identifier to league display name.

    Args:
        sport: Canonical sport string.

    Returns:
        Human-readable league name.
    """
    return {
        "nba": "NBA",
        "epl": "Premier League",
        "nfl": "NFL",
        "soccer": "Soccer",
        "mma": "MMA/UFC",
        "mlb": "MLB",
        "nhl": "NHL",
        "ncaab": "NCAAB",
    }.get(sport, sport.upper())


def _parse_game_date_from_event(ev: dict[str, Any]) -> str | None:
    """Extract game date (YYYY-MM-DD) from Gamma event metadata.

    Tries endDate first (closest to game time), falls back to closedTime.

    Args:
        ev: Gamma API event dict.

    Returns:
        Date string in YYYY-MM-DD format, or None.
    """
    for field in ["endDate", "closedTime"]:
        val = ev.get(field)
        if val:
            try:
                # Handle ISO format with timezone
                dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
                return dt.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                pass

    # Try parsing from title (e.g., "... on March 15, 2026")
    title = ev.get("title", "")
    m = re.search(
        r"(\d{1,2})/(\d{1,2})/(\d{2,4})",
        title,
    )
    if m:
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:
            year += 2000
        try:
            return f"{year:04d}-{month:02d}-{day:02d}"
        except ValueError:
            pass

    return None


# ── API calls ────────────────────────────────────────────────────────


def _http_get(url: str, retries: int = 3) -> dict | list | None:
    """GET JSON with retry and rate limiting.

    Args:
        url: Full URL to fetch.
        retries: Number of retry attempts on failure.

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
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 2 ** (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif e.code >= 500:
                time.sleep(1)
            else:
                return None
        except Exception:
            time.sleep(0.5)
    return None


def fetch_price_history(token_id: str, fidelity: int = 1) -> list[dict[str, Any]]:
    """Fetch price history for a single token from CLOB API.

    Args:
        token_id: Polymarket CLOB token ID.
        fidelity: Price granularity in minutes (1=minutely).

    Returns:
        List of dicts with keys "t" (unix timestamp) and "p" (price string).
    """
    url = (
        f"{CLOB_BASE}/prices-history"
        f"?market={token_id}&interval=max&fidelity={fidelity}"
    )
    data = _http_get(url)
    if data and isinstance(data, dict):
        return data.get("history", [])
    return []


def fetch_sports_events_fresh(
    sport_filter: str = "all",
    max_pages: int = 200,
) -> list[dict[str, Any]]:
    """Fetch all resolved sports events from Gamma API.

    Paginates through all results for the requested sport tags.
    Deduplicates events by event ID.

    Args:
        sport_filter: Sport to filter (e.g., "nba", "epl", "all").
        max_pages: Maximum number of pages to fetch per tag.

    Returns:
        List of Gamma event dicts.
    """
    slugs = GAMMA_SLUGS.get(sport_filter, [sport_filter])
    seen_ids: set[str] = set()
    all_events: list[dict[str, Any]] = []

    for slug in slugs:
        print(f"  Fetching tag '{slug}'...")
        offset = 0
        batch = 100

        for page in range(max_pages):
            url = (
                f"{GAMMA_BASE}/events"
                f"?tag={slug}&closed=true&limit={batch}&offset={offset}"
            )
            events = _http_get(url)
            if not events or not isinstance(events, list):
                break

            for ev in events:
                eid = str(ev.get("id", ""))
                if eid and eid not in seen_ids:
                    seen_ids.add(eid)
                    all_events.append(ev)

            fetched_count = len(events)
            print(f"    Page {page}: {fetched_count} events "
                  f"({len(all_events)} unique total)")
            offset += batch

            if fetched_count < batch:
                break
            time.sleep(GAMMA_DELAY)

        time.sleep(GAMMA_DELAY)

    # Save cache
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVENTS_CACHE, "w") as f:
        json.dump(all_events, f)
    print(f"  Cached {len(all_events)} events to {EVENTS_CACHE}")
    return all_events


def load_events_cached() -> list[dict[str, Any]]:
    """Load events from local cache file.

    Returns:
        List of Gamma event dicts, or empty list if no cache.
    """
    if not EVENTS_CACHE.exists():
        return []
    with open(EVENTS_CACHE) as f:
        return json.load(f)


# ── Market processing ────────────────────────────────────────────────


def _extract_token_ids(mkt: dict[str, Any]) -> list[str]:
    """Extract token IDs from a Gamma market dict.

    CRITICAL: clobTokenIds is a JSON-encoded string, NOT a list.

    Args:
        mkt: Gamma market dict.

    Returns:
        List of token ID strings (typically [YES_id, NO_id]).
    """
    clob_raw = mkt.get("clobTokenIds", "")
    if isinstance(clob_raw, str):
        try:
            token_ids = json.loads(clob_raw)
        except (json.JSONDecodeError, TypeError):
            return []
    elif isinstance(clob_raw, list):
        token_ids = clob_raw
    else:
        return []
    return [str(t) for t in token_ids] if token_ids else []


def _determine_won(mkt: dict[str, Any]) -> int | None:
    """Determine if a market resolved YES (1), NO (0), or is unresolved (None).

    Args:
        mkt: Gamma market dict.

    Returns:
        1 if resolved YES, 0 if resolved NO, None if unresolved.
    """
    outcome_prices = mkt.get("outcomePrices", "")
    if isinstance(outcome_prices, str):
        try:
            outcome_prices = json.loads(outcome_prices)
        except (json.JSONDecodeError, TypeError):
            return None

    if not outcome_prices:
        return None

    try:
        yes_price = float(outcome_prices[0])
        if yes_price > 0.99:
            return 1
        elif yes_price < 0.01:
            return 0
    except (ValueError, TypeError, IndexError):
        pass
    return None


def _determine_outcome_type(question: str, teams: tuple[str, str] | None) -> str:
    """Infer the outcome type from the market question.

    Args:
        question: Market question text.
        teams: Parsed (team_a, team_b) tuple, or None.

    Returns:
        Outcome type string: "moneyline", "spread", "total", "prop", etc.
    """
    q = question.lower()
    if "spread" in q or "cover" in q or "point" in q and "spread" in q:
        return "spread"
    if "over" in q and "under" in q:
        return "total"
    if "total" in q and ("over" in q or "under" in q or "points" in q or "goals" in q):
        return "total"
    if any(w in q for w in ["beat", "win", "defeat", "vs", "v."]):
        return "moneyline"
    if "mvp" in q or "scorer" in q or "assist" in q or "rebound" in q:
        return "prop"
    return "moneyline"


def process_event(
    ev: dict[str, Any],
    db: SportsDB,
    sport_override: str | None = None,
) -> list[str]:
    """Process a single Gamma event: create game + markets in DB.

    Args:
        ev: Gamma API event dict.
        db: SportsDB instance.
        sport_override: Override sport detection (e.g., force "nba").

    Returns:
        List of YES token IDs that need price history download.
    """
    title = ev.get("title", "")
    markets = ev.get("markets", [])
    if not markets:
        return []

    # Determine sport from tags
    tags_raw = ev.get("tags", [])
    if isinstance(tags_raw, str):
        try:
            tags_raw = json.loads(tags_raw)
        except (json.JSONDecodeError, TypeError):
            tags_raw = []

    # Tags can be list of dicts with "label" or plain strings
    tags: list[str] = []
    for t in tags_raw:
        if isinstance(t, dict):
            label = t.get("label", t.get("slug", t.get("name", "")))
            if label:
                tags.append(str(label))
        elif isinstance(t, str):
            tags.append(t)

    sport = sport_override or parse_sport_from_tags(tags)
    if not sport:
        # Try to infer from title
        t_lower = title.lower()
        for s in ["nba", "nfl", "epl", "nhl", "mlb", "ufc", "mma", "ncaab"]:
            if s in t_lower:
                sport = s
                break
    if not sport:
        sport = "unknown"

    league = _sport_to_league(sport)

    # Parse game date
    game_date = _parse_game_date_from_event(ev)
    if not game_date:
        game_date = "1970-01-01"

    # Parse teams from the first market question (or event title)
    primary_question = markets[0].get("question", "") if markets else title
    teams = parse_teams_from_question(primary_question)
    if not teams:
        teams = parse_teams_from_question(title)

    if teams:
        home_abbr = team_to_abbreviation(teams[0], sport)
        away_abbr = team_to_abbreviation(teams[1], sport)
    else:
        # Fallback: use event ID fragments
        home_abbr = "HOME"
        away_abbr = "AWAY"

    game_id = generate_game_id(sport, game_date, home_abbr, away_abbr)

    # Upsert the game
    db.upsert_game(
        game_id=game_id,
        sport=sport,
        league=league,
        home_team=home_abbr,
        away_team=away_abbr,
        game_date=game_date,
        status="final",
        extra_json=json.dumps({
            "event_id": str(ev.get("id", "")),
            "title": title,
            "tags": tags,
            "teams_raw": list(teams) if teams else None,
        }),
    )

    # Process each market within the event
    token_ids: list[str] = []
    event_id = str(ev.get("id", ""))

    for mkt in markets:
        tids = _extract_token_ids(mkt)
        if not tids:
            continue

        yes_token = tids[0]
        no_token = tids[1] if len(tids) > 1 else None
        question = mkt.get("question", "")
        won = _determine_won(mkt)
        outcome = _determine_outcome_type(question, teams)
        neg_risk = 1 if mkt.get("enableNegRisk") or ev.get("enableNegRisk") or ev.get("negRisk") else 0

        volume_raw = mkt.get("volume", 0)
        try:
            volume = float(volume_raw) if volume_raw else 0.0
        except (ValueError, TypeError):
            volume = 0.0

        end_date = mkt.get("endDate") or ev.get("endDate")

        db.upsert_market(
            token_id=yes_token,
            game_id=game_id,
            event_id=event_id,
            condition_id=mkt.get("conditionId"),
            token_id_no=no_token,
            question=question,
            outcome=outcome,
            volume=volume,
            neg_risk=neg_risk,
            won=won,
            end_date=end_date,
            extra_json=json.dumps({
                "outcomes": _safe_json_loads(mkt.get("outcomes", "[]")),
                "outcomePrices": _safe_json_loads(mkt.get("outcomePrices", "[]")),
                "lastTradePrice": mkt.get("lastTradePrice"),
            }),
        )

        token_ids.append(yes_token)

    return token_ids


def _safe_json_loads(val: Any) -> Any:
    """Parse JSON string if needed, return value as-is if already parsed.

    Args:
        val: Value that may be a JSON-encoded string or already a native type.

    Returns:
        Parsed value.
    """
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return val
    return val


# ── Main pipeline ────────────────────────────────────────────────────


def main() -> None:
    """Main entry point: discover sports markets and download prices."""
    parser = argparse.ArgumentParser(
        description="Download Polymarket sports price history"
    )
    parser.add_argument(
        "--sport",
        type=str,
        default="all",
        choices=["nba", "epl", "nfl", "soccer", "mma", "mlb", "nhl", "ncaab", "all"],
        help="Sport to download (default: all)",
    )
    parser.add_argument(
        "--fidelity",
        type=int,
        default=1,
        help="Price granularity in minutes (default: 1 = minutely)",
    )
    parser.add_argument(
        "--refresh-events",
        action="store_true",
        help="Re-fetch events from Gamma API (ignore cache)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Limit number of events to process (0 = all)",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ── 1. Load or fetch events ──
    print(f"{'=' * 60}")
    print(f"Polymarket Sports Price Downloader")
    print(f"Sport: {args.sport} | Fidelity: {args.fidelity}min")
    print(f"{'=' * 60}")

    if args.refresh_events or not EVENTS_CACHE.exists():
        print("\nFetching events from Gamma API...")
        raw_events = fetch_sports_events_fresh(sport_filter=args.sport)
    else:
        raw_events = load_events_cached()
        if not raw_events:
            print("\nCache empty, fetching from Gamma API...")
            raw_events = fetch_sports_events_fresh(sport_filter=args.sport)
        else:
            print(f"\nLoaded {len(raw_events)} events from cache")

    if not raw_events:
        print("No events found. Try --refresh-events or a different --sport.")
        return

    # ── 2. Open database ──
    db = SportsDB()
    print(f"Database: {db.db_path}")

    # Get already-fetched token IDs for idempotency
    already_fetched: set[str] = set()
    try:
        row = db.conn.execute("SELECT DISTINCT token_id FROM prices").fetchall()
        already_fetched = {r[0] for r in row}
    except Exception:
        pass
    print(f"Already fetched: {len(already_fetched)} tokens with price data")

    # ── 3. Process events: create games + markets, collect tokens ──
    # Sort events newest-first (Polymarket retains ~30 days of prices)
    sorted_events = sorted(
        raw_events,
        key=lambda ev: ev.get("closedTime", "") or ev.get("endDate", "") or "",
        reverse=True,
    )

    if args.max_events > 0:
        sorted_events = sorted_events[:args.max_events]

    tokens_to_fetch: list[tuple[str, str]] = []  # (token_id, description)
    total_events_processed = 0
    total_markets = 0
    skipped_no_sport = 0

    print(f"\nProcessing {len(sorted_events)} events...")

    for ev in sorted_events:
        title = ev.get("title", "")
        new_tokens = process_event(ev, db)

        if new_tokens:
            total_events_processed += 1
            total_markets += len(new_tokens)
            for tid in new_tokens:
                if tid not in already_fetched:
                    tokens_to_fetch.append((tid, title[:60]))
        else:
            skipped_no_sport += 1

    print(f"\nParsed: {total_events_processed} events, {total_markets} markets")
    print(f"Skipped (no markets/tokens): {skipped_no_sport}")
    print(f"Tokens to fetch: {len(tokens_to_fetch)} "
          f"(skipping {len(already_fetched)} already fetched)")

    if not tokens_to_fetch:
        print("\nNothing new to fetch!")
        _print_stats(db, t_start)
        db.close()
        return

    # ── 4. Fetch price history ──
    est_minutes = len(tokens_to_fetch) * CLOB_DELAY / 60
    print(f"\nFetching price history (fidelity={args.fidelity}min)...")
    print(f"Estimated time: ~{est_minutes:.1f} minutes")
    print()

    fetched = 0
    empty = 0
    errors = 0
    total_points = 0
    consecutive_empty = 0
    total_requests = 0

    for i, (token_id, description) in enumerate(tokens_to_fetch):
        total_requests += 1
        try:
            history = fetch_price_history(token_id, args.fidelity)
            if history:
                # Convert to (ts, price) tuples for insert_prices_simple
                price_tuples = []
                for h in history:
                    try:
                        ts = int(h["t"])
                        price = float(h["p"])
                        price_tuples.append((ts, price))
                    except (ValueError, TypeError, KeyError):
                        continue

                if price_tuples:
                    db.insert_prices_simple(token_id, price_tuples)
                    total_points += len(price_tuples)
                    fetched += 1
                    consecutive_empty = 0
                    print(f"  Downloaded {len(price_tuples):,} prices for "
                          f"market {token_id[:16]}... ({description})")
                else:
                    empty += 1
                    consecutive_empty += 1
            else:
                empty += 1
                consecutive_empty += 1
        except Exception as e:
            errors += 1
            consecutive_empty += 1
            if errors <= 5:
                print(f"  Error fetching {token_id[:20]}...: {e}")

        # Stop early if we've hit the historical data boundary
        if consecutive_empty >= CONSECUTIVE_EMPTY_STOP:
            print(f"\n  Stopping: {CONSECUTIVE_EMPTY_STOP} consecutive "
                  f"empty responses (older events have no price data)")
            break

        # Progress update every 50 tokens
        if (i + 1) % 50 == 0:
            pct = (i + 1) / len(tokens_to_fetch) * 100
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(tokens_to_fetch) - i - 1) / rate if rate > 0 else 0
            print(f"  [{pct:5.1f}%] {i + 1}/{len(tokens_to_fetch)} tokens | "
                  f"{fetched} with data, {empty} empty | "
                  f"{total_points:,} price points | "
                  f"ETA: {eta / 60:.1f}min")

        time.sleep(CLOB_DELAY)

    print(f"\n{'=' * 60}")
    print(f"Download complete!")
    print(f"  Tokens fetched: {fetched} with data, {empty} empty, {errors} errors")
    print(f"  Total price points: {total_points:,}")
    print(f"  Total API requests: {total_requests}")

    # ── 5. Print stats and save metadata ──
    _print_stats(db, t_start)
    _save_meta(db, t_start, fetched, empty, errors, total_points, args)

    db.close()


def _print_stats(db: SportsDB, t_start: float) -> None:
    """Print database statistics summary.

    Args:
        db: SportsDB instance.
        t_start: Pipeline start time (unix timestamp).
    """
    stats = db.stats()

    print(f"\n{'─' * 60}")
    print("DATABASE STATISTICS")
    print(f"{'─' * 60}")
    print(f"  Games:                {stats['games']:,}")
    print(f"    NBA:                {stats['games_nba']:,}")
    print(f"    EPL:                {stats['games_epl']:,}")
    print(f"    NFL:                {stats['games_nfl']:,}")
    print(f"  Markets:              {stats['markets']:,}")
    print(f"  Price records:        {stats['prices']:,}")

    # Price data range
    row = db.conn.execute("SELECT MIN(ts), MAX(ts) FROM prices").fetchone()
    if row and row[0]:
        dt_min = datetime.utcfromtimestamp(row[0])
        dt_max = datetime.utcfromtimestamp(row[1])
        days = (dt_max - dt_min).days
        print(f"  Price data range:     {dt_min.strftime('%Y-%m-%d')} -> "
              f"{dt_max.strftime('%Y-%m-%d')} ({days} days)")

    # Tokens with price data
    n_tokens = db.conn.execute(
        "SELECT COUNT(DISTINCT token_id) FROM prices"
    ).fetchone()[0]
    print(f"  Tokens with prices:   {n_tokens:,}")

    # Per-sport breakdown
    rows = db.conn.execute("""
        SELECT g.sport,
               COUNT(DISTINCT g.game_id) as games,
               COUNT(DISTINCT m.token_id) as markets,
               COUNT(p.ts) as prices
        FROM games g
        LEFT JOIN markets m ON m.game_id = g.game_id
        LEFT JOIN prices p ON p.token_id = m.token_id
        GROUP BY g.sport
        ORDER BY games DESC
    """).fetchall()

    if rows:
        print(f"\n  {'Sport':<12} {'Games':>7} {'Markets':>8} {'Prices':>10}")
        for sport, games, markets, prices in rows:
            print(f"  {sport:<12} {games:>7} {markets:>8} {prices:>10,}")

    elapsed = time.time() - t_start
    db_size = db.db_path.stat().st_size / (1024 * 1024) if db.db_path.exists() else 0
    print(f"\n  Database size:        {db_size:.1f} MB")
    print(f"  Elapsed time:         {elapsed / 60:.1f} min")


def _save_meta(
    db: SportsDB,
    t_start: float,
    fetched: int,
    empty: int,
    errors: int,
    total_points: int,
    args: argparse.Namespace,
) -> None:
    """Save download metadata to JSON file.

    Args:
        db: SportsDB instance.
        t_start: Pipeline start time.
        fetched: Number of tokens with price data.
        empty: Number of tokens with no price data.
        errors: Number of fetch errors.
        total_points: Total price records downloaded.
        args: CLI arguments namespace.
    """
    stats = db.stats()

    row = db.conn.execute("SELECT MIN(ts), MAX(ts) FROM prices").fetchone()
    price_range = None
    if row and row[0]:
        price_range = {
            "start": datetime.utcfromtimestamp(row[0]).isoformat(),
            "end": datetime.utcfromtimestamp(row[1]).isoformat(),
        }

    meta = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "sport_filter": args.sport,
        "fidelity_minutes": args.fidelity,
        "games_total": stats["games"],
        "markets_total": stats["markets"],
        "tokens_fetched": fetched,
        "tokens_empty": empty,
        "tokens_errors": errors,
        "total_price_points": stats["prices"],
        "price_data_range": price_range,
        "elapsed_seconds": round(time.time() - t_start, 1),
        "source": "Polymarket CLOB /prices-history",
        "notes": (
            "Price history is only available for ~30 days after market close. "
            "Run this script regularly to capture data before it expires. "
            "Each price point has {t: unix_timestamp, p: YES_price}. "
            "Fidelity 1 = minutely data."
        ),
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {META_PATH}")


if __name__ == "__main__":
    main()
