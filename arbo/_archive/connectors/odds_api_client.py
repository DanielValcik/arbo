"""The Odds API v4 client for Polymarket value comparison.

Fetches bookmaker odds (primarily Pinnacle) from The Odds API v4.
Tracks remaining API quota from response headers.

Adapted from Sprint 2 Matchbook implementation for Polymarket matching.
See brief PM-003 for full specification.
"""

from __future__ import annotations

import ssl
from datetime import datetime  # noqa: TC003
from decimal import Decimal

import aiohttp
import certifi
from pydantic import BaseModel

from arbo.config.settings import get_config
from arbo.utils.logger import get_logger

logger = get_logger("odds_api")

# Maps Polymarket categories to The Odds API sport keys
# Paper trading: 3 leagues to conserve quota (~400 credits/day)
# Expand after confirming quota sustainability
SPORT_KEY_MAP: dict[str, list[str]] = {
    "soccer": [
        "soccer_epl",
        "soccer_spain_la_liga",
        "soccer_germany_bundesliga",
    ],
}

# Preferred bookmakers in priority order (Pinnacle first)
PREFERRED_BOOKMAKERS = ["pinnacle", "betfair_ex_eu", "matchbook"]


# --- DTOs ---


class OddsOutcome(BaseModel):
    """Single outcome in a bookmaker market."""

    name: str
    price: Decimal
    point: Decimal | None = None  # line value: 2.5 for totals, -1.5 for spreads


class OddsMarket(BaseModel):
    """Market type from a bookmaker (h2h, spreads, totals)."""

    key: str
    outcomes: list[OddsOutcome]


class OddsBookmaker(BaseModel):
    """Bookmaker with their markets."""

    key: str
    title: str
    markets: list[OddsMarket]


class OddsEvent(BaseModel):
    """Event from The Odds API."""

    id: str
    sport_key: str
    home_team: str
    away_team: str
    commence_time: datetime
    bookmakers: list[OddsBookmaker]

    def get_pinnacle_market(self, market_key: str) -> list[OddsOutcome] | None:
        """Get Pinnacle outcomes for a market type (h2h, totals, spreads).

        Falls back through preferred bookmakers if Pinnacle not available.

        Args:
            market_key: Market type key ("h2h", "totals", "spreads").

        Returns:
            List of OddsOutcome or None if not found.
        """
        for preferred_key in PREFERRED_BOOKMAKERS:
            for bm in self.bookmakers:
                if bm.key == preferred_key:
                    for market in bm.markets:
                        if market.key == market_key:
                            return market.outcomes
        return None

    def get_pinnacle_h2h(self) -> dict[str, Decimal] | None:
        """Extract Pinnacle h2h odds as {outcome_name: decimal_odds}.

        Falls back through preferred bookmakers if Pinnacle not available.
        """
        outcomes = self.get_pinnacle_market("h2h")
        if outcomes is None:
            return None
        return {o.name: o.price for o in outcomes}

    def get_pinnacle_totals_prob(self, line: float, over: bool = True) -> Decimal | None:
        """Get Pinnacle implied prob for Over/Under at a specific line.

        Finds the totals outcomes matching point == line, returns vig-removed prob.

        Args:
            line: The totals line to match (e.g. 2.5).
            over: If True return Over prob, else Under prob.

        Returns:
            Implied probability (0-1) or None if not found.
        """
        outcomes = self.get_pinnacle_market("totals")
        if outcomes is None:
            return None

        target_name = "Over" if over else "Under"
        line_dec = Decimal(str(line))

        # Find Over and Under outcomes at the requested line
        matching: dict[str, Decimal] = {}
        for o in outcomes:
            if o.point is not None and o.point == line_dec:
                matching[o.name] = o.price

        if target_name not in matching:
            return None

        # Vig removal: need both sides
        other_name = "Under" if over else "Over"
        if other_name not in matching:
            # Single-sided: return raw implied prob
            return Decimal("1") / matching[target_name]

        raw_target = Decimal("1") / matching[target_name]
        raw_other = Decimal("1") / matching[other_name]
        total = raw_target + raw_other

        if total <= 0:
            return None

        return raw_target / total

    def get_pinnacle_spreads_prob(self, team: str, line: float) -> Decimal | None:
        """Get Pinnacle implied prob for a team at a specific spread line.

        Uses exact team name matching against outcome names.

        Args:
            team: Team name to match (e.g. "Arsenal").
            line: The spread line (e.g. -0.5).

        Returns:
            Implied probability (0-1) or None if not found.
        """
        outcomes = self.get_pinnacle_market("spreads")
        if outcomes is None:
            return None

        line_dec = Decimal(str(line))
        team_lower = team.lower()

        # Find matching outcome for the team at the given line
        target_outcome: OddsOutcome | None = None
        other_outcome: OddsOutcome | None = None

        for o in outcomes:
            if o.point is not None and o.point == line_dec and o.name.lower() == team_lower:
                target_outcome = o
            elif o.point is not None and o.point == -line_dec and o.name.lower() != team_lower:
                other_outcome = o

        if target_outcome is None:
            return None

        raw_target = Decimal("1") / target_outcome.price

        if other_outcome is None:
            return raw_target

        raw_other = Decimal("1") / other_outcome.price
        total = raw_target + raw_other

        if total <= 0:
            return None

        return raw_target / total

    def get_pinnacle_implied_prob(self, outcome_name: str) -> Decimal | None:
        """Get Pinnacle implied probability for a specific outcome.

        Removes vig using power method before returning.

        Args:
            outcome_name: e.g. "Arsenal" or "Draw".

        Returns:
            Implied probability (0-1) or None if not found.
        """
        h2h = self.get_pinnacle_h2h()
        if h2h is None or outcome_name not in h2h:
            return None

        # Calculate raw implied probs
        raw_probs = {name: Decimal("1") / odds for name, odds in h2h.items()}
        total = sum(raw_probs.values())

        if total <= 0:
            return None

        # Remove vig (simple proportional method)
        return raw_probs[outcome_name] / total


class OddsApiClient:
    """Client for The Odds API v4.

    Features:
    - Fetch odds for configured sport keys
    - Fetch outright/futures odds (season winners)
    - Quota tracking via x-requests-remaining header
    - Pinnacle odds extraction with vig removal
    - 5-minute in-memory cache
    """

    def __init__(self, session: aiohttp.ClientSession | None = None) -> None:
        config = get_config()
        self._api_key = config.odds_api_key
        self._base_url = config.odds_api.base_url
        self._regions = config.odds_api.regions
        self._markets_param = config.odds_api.markets
        self._odds_format = config.odds_api.odds_format
        self._min_remaining_quota = config.odds_api.min_remaining_quota
        self._session = session
        self._owns_session = session is None
        self._remaining_quota: int | None = None
        self._cache: dict[str, tuple[float, list[OddsEvent]]] = {}
        self._outrights_cache: dict[str, tuple[float, dict[str, Decimal]]] = {}
        self._cache_ttl = float(config.odds_api.cache_ttl)

    @property
    def remaining_quota(self) -> int | None:
        """Remaining API requests (from last response header)."""
        return self._remaining_quota

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                connector=aiohttp.TCPConnector(ssl=ssl_ctx),
            )
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        """Close the HTTP session if we own it."""
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def get_odds(self, sport_key: str) -> list[OddsEvent]:
        """Fetch odds for a single sport key.

        Args:
            sport_key: The Odds API sport key (e.g. "soccer_epl").

        Returns:
            List of events with bookmaker odds.
        """
        # Check quota
        if self._remaining_quota is not None and self._remaining_quota < self._min_remaining_quota:
            logger.warning(
                "odds_api_quota_low",
                remaining=self._remaining_quota,
                min_required=self._min_remaining_quota,
            )
            return []

        # Check cache
        import time

        now = time.monotonic()
        if sport_key in self._cache:
            cached_at, cached_events = self._cache[sport_key]
            if now - cached_at < self._cache_ttl:
                logger.debug("odds_api_cache_hit", sport_key=sport_key)
                return cached_events

        session = await self._get_session()
        url = f"{self._base_url}/sports/{sport_key}/odds"
        params = {
            "apiKey": self._api_key,
            "regions": self._regions,
            "markets": self._markets_param,
            "oddsFormat": self._odds_format,
        }

        try:
            async with session.get(url, params=params) as resp:
                remaining = resp.headers.get("x-requests-remaining")
                if remaining is not None:
                    self._remaining_quota = int(remaining)
                    logger.info(
                        "odds_api_quota",
                        remaining=self._remaining_quota,
                        used=resp.headers.get("x-requests-used"),
                    )

                if resp.status == 401:
                    logger.error("odds_api_auth_error", status=401)
                    return []
                if resp.status == 429:
                    logger.error("odds_api_rate_limited")
                    return []
                if resp.status != 200:
                    text = await resp.text()
                    logger.error("odds_api_error", status=resp.status, body=text[:200])
                    return []

                data = await resp.json()

        except Exception as e:
            logger.error("odds_api_exception", error=str(e))
            return []

        events = self._parse_events(data, sport_key)

        # Update cache
        self._cache[sport_key] = (now, events)

        logger.info("odds_api_fetched", sport_key=sport_key, events=len(events))
        return events

    def _parse_events(self, data: list[dict], sport_key: str) -> list[OddsEvent]:
        """Parse raw API response into OddsEvent objects."""
        events: list[OddsEvent] = []

        for item in data:
            try:
                event = OddsEvent(
                    id=item["id"],
                    sport_key=item["sport_key"],
                    home_team=item["home_team"],
                    away_team=item["away_team"],
                    commence_time=item["commence_time"],
                    bookmakers=[
                        OddsBookmaker(
                            key=bm["key"],
                            title=bm["title"],
                            markets=[
                                OddsMarket(
                                    key=mk["key"],
                                    outcomes=[
                                        OddsOutcome(
                                            name=oc["name"],
                                            price=oc["price"],
                                            point=oc.get("point"),
                                        )
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
                logger.warning("odds_api_parse_error", event_id=item.get("id"), error=str(e))

        return events

    async def get_historical_odds(
        self,
        sport_key: str,
        date: str,
        markets: str = "h2h",
    ) -> tuple[list[OddsEvent], str | None, str | None]:
        """Fetch historical odds snapshot for a sport at a specific point in time.

        Uses The Odds API v4 historical endpoint. Each call consumes quota.
        No caching — each historical point in time is unique.

        Args:
            sport_key: The Odds API sport key (e.g. "soccer_epl").
            date: ISO-8601 timestamp for the snapshot (e.g. "2025-10-01T12:00:00Z").
            markets: Comma-separated market types (default "h2h").

        Returns:
            Tuple of (events, previous_timestamp, next_timestamp).
            previous_timestamp/next_timestamp are None if not provided by the API.
        """
        # Check quota
        if self._remaining_quota is not None and self._remaining_quota < self._min_remaining_quota:
            logger.warning(
                "odds_api_historical_quota_low",
                remaining=self._remaining_quota,
                min_required=self._min_remaining_quota,
            )
            return [], None, None

        session = await self._get_session()
        url = f"{self._base_url}/historical/sports/{sport_key}/odds"
        params = {
            "apiKey": self._api_key,
            "regions": self._regions,
            "markets": markets,
            "date": date,
            "oddsFormat": self._odds_format,
        }

        try:
            async with session.get(url, params=params) as resp:
                remaining = resp.headers.get("x-requests-remaining")
                if remaining is not None:
                    self._remaining_quota = int(remaining)
                    logger.info(
                        "odds_api_historical_quota",
                        remaining=self._remaining_quota,
                        used=resp.headers.get("x-requests-used"),
                    )

                if resp.status == 401:
                    logger.error("odds_api_historical_auth_error", status=401)
                    return [], None, None
                if resp.status == 422:
                    logger.error("odds_api_historical_invalid_date", date=date)
                    return [], None, None
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(
                        "odds_api_historical_error",
                        status=resp.status,
                        body=text[:200],
                    )
                    return [], None, None

                response = await resp.json()

        except Exception as e:
            logger.error("odds_api_historical_exception", error=str(e))
            return [], None, None

        data = response.get("data", [])
        previous_timestamp = response.get("previous_timestamp")
        next_timestamp = response.get("next_timestamp")

        events = self._parse_events(data, sport_key)

        logger.info(
            "odds_api_historical_fetched",
            sport_key=sport_key,
            date=date,
            events=len(events),
        )
        return events, previous_timestamp, next_timestamp

    async def get_outrights(self, sport_key: str) -> dict[str, Decimal]:
        """Fetch outright/futures odds for a sport key.

        Returns Pinnacle implied probabilities (vig removed) for each
        team/outcome. Used for matching seasonal Polymarket questions
        like "Will Arsenal win the EPL?" against bookmaker prices.

        Args:
            sport_key: The Odds API sport key (e.g. "soccer_epl").

        Returns:
            Dict mapping team/outcome name → implied probability (0-1).
        """
        # Check quota
        if self._remaining_quota is not None and self._remaining_quota < self._min_remaining_quota:
            logger.warning(
                "odds_api_quota_low_outrights",
                remaining=self._remaining_quota,
            )
            return {}

        # Check cache
        import time

        now = time.monotonic()
        if sport_key in self._outrights_cache:
            cached_at, cached_odds = self._outrights_cache[sport_key]
            if now - cached_at < self._cache_ttl:
                logger.debug("odds_api_outrights_cache_hit", sport_key=sport_key)
                return cached_odds

        session = await self._get_session()
        url = f"{self._base_url}/sports/{sport_key}/odds"
        params = {
            "apiKey": self._api_key,
            "regions": self._regions,
            "markets": "outrights",
            "oddsFormat": self._odds_format,
        }

        try:
            async with session.get(url, params=params) as resp:
                remaining = resp.headers.get("x-requests-remaining")
                if remaining is not None:
                    self._remaining_quota = int(remaining)

                if resp.status != 200:
                    text = await resp.text()
                    logger.error(
                        "odds_api_outrights_error",
                        sport_key=sport_key,
                        status=resp.status,
                        body=text[:200],
                    )
                    return {}

                data = await resp.json()

        except Exception as e:
            logger.error("odds_api_outrights_exception", error=str(e))
            return {}

        result = self._parse_outrights(data)

        # Update cache
        self._outrights_cache[sport_key] = (now, result)

        logger.info(
            "odds_api_outrights_fetched",
            sport_key=sport_key,
            teams=len(result),
        )
        return result

    def _parse_outrights(self, data: list[dict]) -> dict[str, Decimal]:
        """Parse outright odds response into vig-free implied probabilities.

        Searches for Pinnacle (or fallback bookmaker) outright market,
        then converts decimal odds to implied probabilities with vig removal.
        """
        for item in data:
            bookmakers = item.get("bookmakers", [])

            # Find preferred bookmaker's outrights
            for preferred_key in PREFERRED_BOOKMAKERS:
                for bm in bookmakers:
                    if bm.get("key") != preferred_key:
                        continue

                    for market in bm.get("markets", []):
                        if market.get("key") != "outrights":
                            continue

                        outcomes = market.get("outcomes", [])
                        if not outcomes:
                            continue

                        # Convert to vig-free implied probs
                        raw_probs: dict[str, Decimal] = {}
                        for oc in outcomes:
                            name = oc.get("name", "")
                            price = Decimal(str(oc.get("price", 0)))
                            if price > 0 and name:
                                raw_probs[name] = Decimal("1") / price

                        total = sum(raw_probs.values())
                        if total > 0:
                            return {name: prob / total for name, prob in raw_probs.items()}

        return {}

    async def get_soccer_odds(self) -> list[OddsEvent]:
        """Fetch odds for all configured soccer leagues.

        Returns:
            All soccer events across configured leagues.
        """
        all_events: list[OddsEvent] = []

        for sport_key in SPORT_KEY_MAP.get("soccer", []):
            events = await self.get_odds(sport_key)
            all_events.extend(events)

        return all_events

    async def get_all_soccer_outrights(self) -> dict[str, dict[str, Decimal]]:
        """Fetch outright odds for all configured soccer leagues.

        Returns:
            Dict mapping sport_key → {team_name → implied_prob}.
        """
        all_outrights: dict[str, dict[str, Decimal]] = {}

        for sport_key in SPORT_KEY_MAP.get("soccer", []):
            odds = await self.get_outrights(sport_key)
            if odds:
                all_outrights[sport_key] = odds

        return all_outrights
