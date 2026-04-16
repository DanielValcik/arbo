"""Strategy D — EPL Market Discovery via Polymarket Gamma API.

Discovers active EPL + cup competition moneyline/draw markets, parses team
names from questions. Returns MarketData objects ready for signal generation.

Handles 3 question formats:
  - "Will X beat Y?" → team_a_wins
  - "Will X win on DATE?" → single-team win (need to find opponent)
  - "Will X vs Y end in a draw?" → draw market

Part of multi-sport Strategy D architecture.
"""

from __future__ import annotations

import json
import re
import ssl
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()

from arbo.strategies.strategy_d_core import MarketData
from arbo.utils.logger import get_logger

logger = get_logger("strategy_d_discovery_epl")

GAMMA_API = "https://gamma-api.polymarket.com"


_EPL_TEAM_CODES = {
    "arsenal": "ARS",
    "astonvilla": "AVL", "villa": "AVL",
    "brighton": "BHA", "brightonhovealbion": "BHA", "brightonandhovealbion": "BHA",
    "bournemouth": "BOU", "afcbournemouth": "BOU",
    "brentford": "BRE",
    "burnley": "BUR",
    "chelsea": "CHE",
    "crystalpalace": "CRY", "palace": "CRY",
    "everton": "EVE",
    "fulham": "FUL",
    "ipswich": "IPS", "ipswichtown": "IPS",
    "leeds": "LEE", "leedsunited": "LEE",
    "leicester": "LEI", "leicestercity": "LEI",
    "liverpool": "LIV",
    "luton": "LUT", "lutontown": "LUT",
    "manchestercity": "MCI", "mancity": "MCI", "mcfc": "MCI",
    "manchesterunited": "MUN", "manunited": "MUN", "manutd": "MUN", "mufc": "MUN",
    "newcastle": "NEW", "newcastleunited": "NEW",
    "nottinghamforest": "NFO", "forest": "NFO",
    "sheffieldunited": "SHU",
    "southampton": "SOU",
    "sunderland": "SUN",
    "tottenham": "TOT", "tottenhamhotspur": "TOT", "spurs": "TOT",
    "westham": "WHU", "westhamunited": "WHU",
    "wolves": "WOL", "wolverhamptonwanderers": "WOL", "wolverhampton": "WOL",
}


def _team_key(name: str) -> str:
    """Normalize team name to EPL 3-letter code matching Pinnacle cache keys.

    Pinnacle cache uses standard football abbreviations (ARS, LIV, MCI, etc).
    We map Polymarket full names via _EPL_TEAM_CODES.
    """
    if not name:
        return ""
    normalized = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode()
    normalized = re.sub(r"\b(f\.?c\.?|fc)\b", "", normalized.lower())
    flat = re.sub(r"[^a-z]", "", normalized)
    code = _EPL_TEAM_CODES.get(flat)
    if code:
        return code
    # Fallback: first 3 chars uppercase (matches research_d abbreviation for
    # single-word names not in the lookup). Logged in scan to surface misses.
    return flat[:3].upper()


_BEAT_RE = re.compile(r"[Ww]ill\s+(.+?)\s+beat\s+(.+?)[\?\s]*$")
_VS_DRAW_RE = re.compile(r"[Ww]ill\s+(.+?)\s+vs\.?\s+(.+?)\s+end\s+in\s+(?:a\s+)?draw[\?\s]*$")

_NON_GAME_KEYWORDS = [
    "mvp", "top scorer", "golden boot", "ballon d'or",
    "relegation", "title race", "make the top 4", "finish",
    "total wins", "winner of", "premier league winner",
    "champions league spot",
]


def _parse_epl_market(question: str) -> tuple[str, str, str] | None:
    """Returns (team_a_key, team_b_key, outcome_type) or None.

    Note: We SKIP 'Will X win on DATE' single-team markets in discovery
    because we can't determine the opponent — only suitable for backtest
    where we have pre-computed Pinnacle lookup.
    """
    if not question:
        return None
    lower = question.lower()
    if any(kw in lower for kw in _NON_GAME_KEYWORDS):
        return None

    q = question.strip().rstrip("?").strip()

    # Draw: "Will X vs Y end in a draw?"
    m = _VS_DRAW_RE.search(q)
    if m:
        a = _team_key(m.group(1))
        b = _team_key(m.group(2))
        if a and b and a != b and len(a) >= 3 and len(b) >= 3:
            return a, b, "draw"
        return None

    # Beat: "Will X beat Y?"
    m = _BEAT_RE.search(q)
    if m:
        a = _team_key(m.group(1))
        b = _team_key(m.group(2))
        if a and b and a != b and len(a) >= 3 and len(b) >= 3:
            return a, b, "team_a_wins"
        return None

    return None


async def _gamma_get(path: str, params: dict | None = None) -> Any:
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


async def discover_epl_markets(gamma_client: Any = None) -> list[MarketData]:
    """Discover active EPL + cup competition markets on Polymarket."""
    markets: list[MarketData] = []
    now_ts = int(time.time())

    # Try multiple tags: EPL, FA Cup, League Cup, UCL (all English football)
    all_events = []
    for tag in ["epl", "premier-league", "fa-cup", "league-cup"]:
        params = {"tag_slug": tag, "closed": "false", "active": "true", "limit": 200}
        events = await _gamma_get("/events", params)
        if events and isinstance(events, list):
            all_events.extend(events)

    # Deduplicate by event id
    seen = set()
    unique = []
    for e in all_events:
        eid = e.get("id")
        if eid and eid not in seen:
            seen.add(eid)
            unique.append(e)

    for event in unique:
        end_date = event.get("endDate")
        if end_date:
            try:
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
            parsed = _parse_epl_market(question)
            if not parsed:
                continue
            team_a, team_b, outcome_type = parsed

            tokens_raw = mkt.get("clobTokenIds", "[]")
            try:
                token_ids = json.loads(tokens_raw) if isinstance(tokens_raw, str) else tokens_raw
            except Exception:
                continue
            if len(token_ids) < 2:
                continue

            prices_raw = mkt.get("outcomePrices", "[]")
            try:
                prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
                yes_price = float(prices[0])
                no_price = float(prices[1])
            except Exception:
                continue

            if yes_price <= 0.01 or yes_price >= 0.99:
                continue

            markets.append(MarketData(
                sport="epl",
                condition_id=mkt.get("conditionId", ""),
                token_id_yes=str(token_ids[0]),
                token_id_no=str(token_ids[1]),
                question=question,
                team_a=team_a,
                team_b=team_b,
                game_date=game_date,
                game_time=game_time,
                yes_price=yes_price,
                no_price=no_price,
                volume=float(mkt.get("volume", 0) or 0),
                neg_risk=bool(mkt.get("negRisk", False)),
                outcome_type=outcome_type,
            ))

    logger.info("discover_epl_markets", found=len(markets), events=len(unique))
    return markets
