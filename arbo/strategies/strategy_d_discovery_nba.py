"""Strategy D — NBA Market Discovery via Polymarket Gamma API.

Discovers active NBA moneyline markets, parses team names from questions,
and returns MarketData objects ready for signal generation.

Part of multi-sport Strategy D architecture (see docs/STRATEGY_D_ARCHITECTURE.md).
"""

from __future__ import annotations

import json
import re
import ssl
import time
import urllib.error
import urllib.request
from typing import Any

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()

from arbo.strategies.strategy_d_core import MarketData
from arbo.utils.logger import get_logger

logger = get_logger("strategy_d_discovery")

GAMMA_API = "https://gamma-api.polymarket.com"

# NBA team → abbreviation
NBA_TEAMS = {
    "atlanta hawks": "ATL", "hawks": "ATL",
    "boston celtics": "BOS", "celtics": "BOS",
    "brooklyn nets": "BKN", "nets": "BKN",
    "charlotte hornets": "CHA", "hornets": "CHA",
    "chicago bulls": "CHI", "bulls": "CHI",
    "cleveland cavaliers": "CLE", "cavaliers": "CLE", "cavs": "CLE",
    "dallas mavericks": "DAL", "mavericks": "DAL", "mavs": "DAL",
    "denver nuggets": "DEN", "nuggets": "DEN",
    "detroit pistons": "DET", "pistons": "DET",
    "golden state warriors": "GSW", "warriors": "GSW",
    "houston rockets": "HOU", "rockets": "HOU",
    "indiana pacers": "IND", "pacers": "IND",
    "los angeles clippers": "LAC", "la clippers": "LAC", "clippers": "LAC",
    "los angeles lakers": "LAL", "la lakers": "LAL", "lakers": "LAL",
    "memphis grizzlies": "MEM", "grizzlies": "MEM",
    "miami heat": "MIA", "heat": "MIA",
    "milwaukee bucks": "MIL", "bucks": "MIL",
    "minnesota timberwolves": "MIN", "timberwolves": "MIN", "wolves": "MIN",
    "new orleans pelicans": "NOP", "pelicans": "NOP",
    "new york knicks": "NYK", "knicks": "NYK",
    "oklahoma city thunder": "OKC", "thunder": "OKC",
    "orlando magic": "ORL", "magic": "ORL",
    "philadelphia 76ers": "PHI", "76ers": "PHI", "sixers": "PHI",
    "phoenix suns": "PHX", "suns": "PHX",
    "portland trail blazers": "POR", "trail blazers": "POR", "blazers": "POR",
    "sacramento kings": "SAC", "kings": "SAC",
    "san antonio spurs": "SAS", "spurs": "SAS",
    "toronto raptors": "TOR", "raptors": "TOR",
    "utah jazz": "UTA", "jazz": "UTA",
    "washington wizards": "WAS", "wizards": "WAS",
}

_TEAM_PATTERNS = [
    re.compile(r"NBA:?\s*[Ww]ill (?:the )?(.+?) (?:beat|defeat) (?:the )?(.+?)(?:\s+by\b.+)?[\?\s]*$"),
    re.compile(r"^(.+?)\s+vs?\.?\s+(.+?)(?:\s*[:]\s*.+)?$"),
]

_NON_GAME_KEYWORDS = [
    "mvp", "rookie of the year", "award", "championship",
    "wins the nba", "final four", "all-star", "total wins",
    "make the playoffs", "draft", "player prop",
]

# Moneyline ONLY — skip O/U, spread, halves, quarters, props
_NON_MONEYLINE_KEYWORDS = [
    "o/u", "over/under", "spread", "handicap",
    "1h ", "2h ", "1st half", "2nd half",
    " q1", " q2", " q3", " q4", "quarter",
    "first to", "race to",
    ": o/u", ": 1h", ": 2h",
    "by more than", "by less than", "by exactly",
]


def _is_moneyline(question: str) -> bool:
    """Check if market is a moneyline (head-to-head) market."""
    if not question:
        return False
    lower = question.lower()
    if any(kw in lower for kw in _NON_MONEYLINE_KEYWORDS):
        return False
    # Must be "X vs Y" format without extra qualifiers
    return True


def _parse_teams(question: str) -> tuple[str, str] | None:
    """Parse (team_a, team_b) abbreviations from market question."""
    if not question:
        return None
    lower = question.lower()
    if any(kw in lower for kw in _NON_GAME_KEYWORDS):
        return None
    q = re.sub(r"^NBA\s*:\s*", "", question.strip()).rstrip("?").strip()

    for pat in _TEAM_PATTERNS:
        m = pat.match(q)
        if not m or not m.group(2):
            continue
        a_raw, b_raw = m.group(1).strip().lower(), m.group(2).strip().lower()
        if len(a_raw) > 40 or len(b_raw) > 40:
            continue

        a_abbr = NBA_TEAMS.get(a_raw)
        b_abbr = NBA_TEAMS.get(b_raw)
        if not a_abbr:
            for name, abbr in NBA_TEAMS.items():
                if name in a_raw:
                    a_abbr = abbr
                    break
        if not b_abbr:
            for name, abbr in NBA_TEAMS.items():
                if name in b_raw:
                    b_abbr = abbr
                    break

        if a_abbr and b_abbr and a_abbr != b_abbr:
            return a_abbr, b_abbr
    return None


async def _gamma_get(path: str, params: dict | None = None) -> Any:
    """GET request to Gamma API."""
    url = f"{GAMMA_API}{path}"
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{qs}"

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ArboStrategyD/1.0"})
            with urllib.request.urlopen(req, timeout=15, context=SSL_CTX) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                logger.warning("gamma_get_failed", url=url, error=str(e))
                return None
    return None


async def discover_nba_markets(gamma_client: Any = None) -> list[MarketData]:
    """Discover active NBA moneyline markets on Polymarket.

    Filters:
      - Active (not closed/resolved)
      - Has future game date
      - Question matches NBA game format ("X vs Y" or "Will X beat Y")
      - Both teams recognized
    """
    markets: list[MarketData] = []

    # Gamma API: events with NBA tag, active, not closed
    # Tag slug for NBA is "nba"
    now_ts = int(time.time())
    params = {
        "tag_slug": "nba",
        "closed": "false",
        "active": "true",
        "limit": 200,
    }
    events = await _gamma_get("/events", params)
    if not events or not isinstance(events, list):
        logger.info("discover_nba_markets_empty")
        return markets

    for event in events:
        end_date = event.get("endDate")
        # Skip events that have already ended
        if end_date:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                if dt.timestamp() < now_ts:
                    continue
                game_date = dt.strftime("%Y-%m-%d")
                game_time = dt.strftime("%H:%M")
            except Exception:
                game_date = ""
                game_time = None
        else:
            game_date = ""
            game_time = None

        for mkt in event.get("markets", []):
            if mkt.get("closed") or not mkt.get("active", True):
                continue

            question = mkt.get("question", "")
            if not _is_moneyline(question):
                continue
            teams = _parse_teams(question)
            if not teams:
                continue

            # Parse token IDs
            tokens_raw = mkt.get("clobTokenIds", "[]")
            try:
                token_ids = json.loads(tokens_raw) if isinstance(tokens_raw, str) else tokens_raw
            except Exception:
                continue
            if len(token_ids) < 2:
                continue

            # Parse outcome prices
            prices_raw = mkt.get("outcomePrices", "[]")
            try:
                prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
                yes_price = float(prices[0])
                no_price = float(prices[1])
            except Exception:
                continue

            # Skip if prices are at extremes (probably resolved or very late)
            if yes_price <= 0.01 or yes_price >= 0.99:
                continue

            markets.append(MarketData(
                sport="nba",
                condition_id=mkt.get("conditionId", ""),
                token_id_yes=str(token_ids[0]),
                token_id_no=str(token_ids[1]),
                question=question,
                team_a=teams[0],
                team_b=teams[1],
                game_date=game_date,
                game_time=game_time,
                yes_price=yes_price,
                no_price=no_price,
                volume=float(mkt.get("volume", 0) or 0),
                neg_risk=bool(mkt.get("negRisk", False)),
            ))

    logger.info("discover_nba_markets", found=len(markets), events=len(events))
    return markets
