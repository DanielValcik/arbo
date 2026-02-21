"""The Odds API v4 client.

Fetches bookmaker odds from https://api.the-odds-api.com/v4.
Tracks remaining API quota from response headers.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003
from decimal import Decimal  # noqa: TC003

import aiohttp
from pydantic import BaseModel

from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger("odds_api")

# Maps our internal sport names (from config) to The Odds API sport keys.
SPORT_KEY_MAP: dict[str, list[str]] = {
    "football": [
        "soccer_epl",
        "soccer_uefa_champs_league",
        "soccer_spain_la_liga",
        "soccer_germany_bundesliga",
        "soccer_italy_serie_a",
    ],
    "basketball": [
        "basketball_nba",
        "basketball_euroleague",
    ],
}


# --- DTOs ---


class OddsApiOutcome(BaseModel):
    name: str
    price: Decimal


class OddsApiMarket(BaseModel):
    key: str
    outcomes: list[OddsApiOutcome]


class OddsApiBookmaker(BaseModel):
    key: str
    title: str
    markets: list[OddsApiMarket]


class OddsApiEvent(BaseModel):
    id: str
    sport_key: str
    home_team: str
    away_team: str
    commence_time: datetime
    bookmakers: list[OddsApiBookmaker]


class OddsApiClient:
    """Client for The Odds API v4."""

    def __init__(self, session: aiohttp.ClientSession | None = None) -> None:
        self._config = get_config()
        self._api_key = self._config.odds_api_key
        self._base_url = self._config.odds_api.base_url
        self._regions = self._config.odds_api.regions
        self._markets = self._config.odds_api.markets
        self._odds_format = self._config.odds_api.odds_format
        self._min_remaining_quota = self._config.odds_api.min_remaining_quota
        self._session = session
        self._owns_session = session is None
        self._remaining_quota: int | None = None

    @property
    def remaining_quota(self) -> int | None:
        """Remaining API requests (from last response header)."""
        return self._remaining_quota

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        """Close the HTTP session if we own it."""
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def get_odds(self, sport_key: str) -> list[OddsApiEvent]:
        """Fetch odds for a single Odds API sport key.

        GET /v4/sports/{sport}/odds?apiKey=...&regions=eu&markets=h2h&oddsFormat=decimal
        """
        if self._remaining_quota is not None and self._remaining_quota < self._min_remaining_quota:
            log.warning(
                "odds_api_quota_low",
                remaining=self._remaining_quota,
                min_required=self._min_remaining_quota,
            )
            return []

        session = await self._get_session()
        url = f"{self._base_url}/sports/{sport_key}/odds"
        params = {
            "apiKey": self._api_key,
            "regions": self._regions,
            "markets": self._markets,
            "oddsFormat": self._odds_format,
        }

        async with session.get(url, params=params) as resp:
            # Track quota from headers
            remaining = resp.headers.get("x-requests-remaining")
            if remaining is not None:
                self._remaining_quota = int(remaining)
                log.info(
                    "odds_api_quota",
                    remaining=self._remaining_quota,
                    used=resp.headers.get("x-requests-used"),
                )

            if resp.status == 401:
                log.error("odds_api_auth_error", status=401)
                return []
            if resp.status == 429:
                log.error("odds_api_rate_limited")
                return []
            if resp.status != 200:
                text = await resp.text()
                log.error("odds_api_error", status=resp.status, body=text[:200])
                return []

            data = await resp.json()

        events = []
        for item in data:
            try:
                event = OddsApiEvent(
                    id=item["id"],
                    sport_key=item["sport_key"],
                    home_team=item["home_team"],
                    away_team=item["away_team"],
                    commence_time=item["commence_time"],
                    bookmakers=[
                        OddsApiBookmaker(
                            key=bm["key"],
                            title=bm["title"],
                            markets=[
                                OddsApiMarket(
                                    key=mk["key"],
                                    outcomes=[
                                        OddsApiOutcome(name=oc["name"], price=oc["price"])
                                        for oc in mk["outcomes"]
                                    ],
                                )
                                for mk in bm["markets"]
                            ],
                        )
                        for bm in item.get("bookmakers", [])
                    ],
                )
                events.append(event)
            except (KeyError, ValueError) as e:
                log.warning("odds_api_parse_error", event_id=item.get("id"), error=str(e))
                continue

        log.info("odds_api_fetched", sport_key=sport_key, events=len(events))
        return events

    async def get_all_odds(self) -> list[OddsApiEvent]:
        """Fetch odds for all configured sports. Maps internal sport names to Odds API keys."""
        all_events: list[OddsApiEvent] = []

        for sport_name in self._config.sports:
            sport_keys = SPORT_KEY_MAP.get(sport_name, [])
            for sport_key in sport_keys:
                events = await self.get_odds(sport_key)
                all_events.extend(events)

        return all_events
