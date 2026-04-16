"""Strategy D — UFC Market Discovery via Polymarket Gamma API.

Discovers active UFC moneyline markets, parses fighter names from questions.
Returns MarketData objects ready for signal generation.

Part of multi-sport Strategy D architecture (see docs/STRATEGY_D_ARCHITECTURE.md).
"""

from __future__ import annotations

import json
import re
import ssl
import time
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

logger = get_logger("strategy_d_discovery_ufc")

GAMMA_API = "https://gamma-api.polymarket.com"


def _fighter_key(name: str) -> str:
    """Normalize fighter name to Pinnacle cache key format.

    Cache uses first-name 3-letter prefix (e.g. 'Ilia Topuria' → 'ILI',
    'Justin Gaethje' → 'JUS'). This matches research_d/export_backtest_cache.py
    `team_to_abbreviation` which returns cleaned[:3].upper() for names treated
    as single tokens in the Pinnacle source.
    """
    if not name:
        return ""
    cleaned = re.sub(r"[^a-zA-Z\s\.-]", "", name).strip()
    parts = cleaned.split()
    if not parts:
        return ""
    # First name, first 3 chars. Pad to 3 chars with spaces for short names
    # like 'Bo ' or 'Yi ' — cache preserves trailing whitespace in keys.
    first = parts[0]
    key = first[:3].upper()
    if len(key) < 3:
        key = key.ljust(3)
    return key


_FIGHT_PATTERNS = [
    re.compile(r"^(.+?)\s+vs?\.?\s+(.+?)(?:\s*[:]\s*.+)?$"),
    re.compile(r"[Ww]ill (.+?) (?:beat|defeat) (.+?)[\?\s]*$"),
]

_NON_FIGHT_KEYWORDS = [
    "method of victory", "round", "ko", "tko", "submission", "decision",
    "fight of the night", "performance of the night", "goes the distance",
    "belt", "title shot", "retirement",
]


def _is_moneyline_fight(question: str) -> bool:
    if not question:
        return False
    lower = question.lower()
    if any(kw in lower for kw in _NON_FIGHT_KEYWORDS):
        return False
    return " vs" in lower or " vs." in lower or "will " in lower


def _parse_fighters(question: str) -> tuple[str, str] | None:
    if not question or not _is_moneyline_fight(question):
        return None
    # Strip prefix like "UFC Fight Night:", "UFC XXX:", "UFC:"
    q = re.sub(r"^UFC(?:\s+(?:Fight\s+Night|\d+|\w+))?\s*:\s*", "", question.strip(), flags=re.IGNORECASE)
    # Strip parenthetical suffix like "(Welterweight, Main Card)"
    q = re.sub(r"\s*\([^)]*\)\s*$", "", q)
    q = q.rstrip("?").strip()

    for pat in _FIGHT_PATTERNS:
        m = pat.match(q)
        if not m or not m.group(2):
            continue
        a_raw = m.group(1).strip()
        b_raw = m.group(2).strip()
        if len(a_raw) > 60 or len(b_raw) > 60:
            continue
        a = _fighter_key(a_raw)
        b = _fighter_key(b_raw)
        # Reject generic keywords
        if a in {"CARD", "PRELIMS", "MAIN", "EVENT", "UFC", "FIGHT"} or b in {"CARD", "PRELIMS", "MAIN", "EVENT", "UFC", "FIGHT"}:
            continue
        if a and b and a != b:
            return a, b
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


async def discover_ufc_markets(gamma_client: Any = None) -> list[MarketData]:
    """Discover active UFC moneyline fights on Polymarket."""
    markets: list[MarketData] = []
    now_ts = int(time.time())

    params = {
        "tag_slug": "ufc",
        "closed": "false",
        "active": "true",
        "limit": 200,
    }
    events = await _gamma_get("/events", params)
    if not events or not isinstance(events, list):
        logger.info("discover_ufc_markets_empty")
        return markets

    for event in events:
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
            fighters = _parse_fighters(question)
            if not fighters:
                continue

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
                sport="ufc",
                condition_id=mkt.get("conditionId", ""),
                token_id_yes=str(token_ids[0]),
                token_id_no=str(token_ids[1]),
                question=question,
                team_a=fighters[0],
                team_b=fighters[1],
                game_date=game_date,
                game_time=game_time,
                yes_price=yes_price,
                no_price=no_price,
                volume=float(mkt.get("volume", 0) or 0),
                neg_risk=bool(mkt.get("negRisk", False)),
            ))

    logger.info("discover_ufc_markets", found=len(markets), events=len(events))
    return markets
