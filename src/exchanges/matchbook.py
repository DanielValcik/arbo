"""Matchbook REST API client.

Endpoints (verified 2026-02-20):
- Auth:  POST https://api.matchbook.com/bpapi/rest/security/session
- Data:  GET  https://api.matchbook.com/edge/rest/events/...
- Offers: POST/GET/DELETE https://api.matchbook.com/edge/rest/v2/offers
- Session TTL: ~6 hours
- Commission: 4% net win (CZ)
- API cost: £100 per 1M GET
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, ClassVar

import aiohttp

from src.exchanges.base import (
    BaseExchangeClient,
    BetParams,
    BetResult,
    BetStatus,
    EventStatus,
    ExchangeEvent,
    Market,
    Runner,
    RunnerPrice,
    Side,
)
from src.utils.config import MatchbookConfig, get_config
from src.utils.logger import get_logger

log = get_logger("matchbook")


# --- Exceptions ---


class MatchbookError(Exception):
    """Base Matchbook error."""


class MatchbookAuthError(MatchbookError):
    """Authentication failure."""


class MatchbookRateLimitError(MatchbookError):
    """Rate limit exceeded (10-minute block)."""


class MatchbookAPIError(MatchbookError):
    """General API error."""

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        super().__init__(f"HTTP {status}: {message}")


# --- Token Bucket Rate Limiter ---


class TokenBucket:
    """Async token bucket rate limiter."""

    def __init__(self, rate: float = 10.0, capacity: float = 10.0) -> None:
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last_refill = now

            if self._tokens < 1.0:
                wait_time = (1.0 - self._tokens) / self.rate
                await asyncio.sleep(wait_time)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


# --- Client ---


class MatchbookClient(BaseExchangeClient):
    """Matchbook exchange REST API client."""

    # Matchbook sport ID mapping (discovered via GET /lookups/sports)
    SPORT_IDS: ClassVar[dict[str, int]] = {
        "football": 15,  # Soccer
        "basketball": 10,
        "tennis": 24,
        "horse_racing": 1,
    }

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        config: MatchbookConfig | None = None,
        redis_client: Any | None = None,
    ) -> None:
        cfg = config or get_config().matchbook
        app_config = get_config()

        self._username = username or app_config.matchbook_username
        self._password = password or app_config.matchbook_password
        self._base_url = cfg.base_url.rstrip("/")
        self._auth_url = cfg.auth_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=cfg.timeout_seconds)
        self._max_retries = cfg.max_retries
        self._session_ttl = cfg.session_ttl_seconds

        self._session_token: str | None = None
        self._session: aiohttp.ClientSession | None = None
        self._auth_lock = asyncio.Lock()
        self._rate_limiter = TokenBucket(rate=10.0, capacity=10.0)
        self._redis = redis_client

        # Cost monitoring
        self._daily_get_count = 0
        self._daily_get_date: str = ""

    async def _ensure_http_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    # --- Auth ---

    async def login(self) -> None:
        """Authenticate with Matchbook. Stores session token in Redis if available."""
        session = await self._ensure_http_session()
        url = f"{self._auth_url}/security/session"

        log.info("matchbook_login_attempt")
        try:
            async with session.post(
                url,
                json={"username": self._username, "password": self._password},
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise MatchbookAuthError(f"Login failed: HTTP {resp.status} — {body}")

                data = await resp.json()
                token = data.get("session-token") or resp.headers.get("session-token")
                if not token:
                    raise MatchbookAuthError("No session-token in response")

                self._session_token = token

                # Cache in Redis if available
                if self._redis:
                    await self._redis.set(
                        "arbo:session:matchbook",
                        token,
                        ex=self._session_ttl,
                    )

                log.info("matchbook_login_success")

        except aiohttp.ClientError as e:
            raise MatchbookAuthError(f"Login connection error: {e}") from e

    async def _get_session_token(self) -> str:
        """Get current session token, refreshing from Redis or re-authing if needed."""
        # Try Redis first
        if self._redis and not self._session_token:
            cached = await self._redis.get("arbo:session:matchbook")
            if cached:
                self._session_token = cached if isinstance(cached, str) else cached.decode()

        if not self._session_token:
            await self.login()

        return self._session_token  # type: ignore[return-value]

    async def _refresh_auth(self) -> None:
        """Re-authenticate with lock to prevent concurrent re-auth."""
        async with self._auth_lock:
            # Double-check: maybe another coroutine already refreshed
            if self._redis:
                cached = await self._redis.get("arbo:session:matchbook")
                if cached:
                    self._session_token = cached if isinstance(cached, str) else cached.decode()
                    return
            await self.login()

    # --- HTTP ---

    def _track_get(self) -> None:
        """Track daily GET count for cost monitoring."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if self._daily_get_date != today:
            if self._daily_get_count > 0:
                log.info(
                    "matchbook_daily_get_count",
                    date=self._daily_get_date,
                    count=self._daily_get_count,
                )
            self._daily_get_count = 0
            self._daily_get_date = today
        self._daily_get_count += 1

    async def _request(
        self,
        method: str,
        path: str,
        base_url: str | None = None,
        json_data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        _retried: bool = False,
    ) -> dict[str, Any]:
        """Make an authenticated request to Matchbook API.

        Handles 401 re-auth with retry, rate limiting, and exponential backoff.
        """
        await self._rate_limiter.acquire()
        session = await self._ensure_http_session()
        token = await self._get_session_token()

        url = f"{base_url or self._base_url}{path}"
        headers = {"session-token": token, "Content-Type": "application/json"}

        if method.upper() == "GET":
            self._track_get()

        backoff = 1.0
        for attempt in range(self._max_retries):
            try:
                async with session.request(
                    method, url, headers=headers, json=json_data, params=params
                ) as resp:
                    if resp.status == 401 and not _retried:
                        log.warning("matchbook_session_expired", attempt=attempt)
                        self._session_token = None
                        await self._refresh_auth()
                        return await self._request(
                            method, path, base_url, json_data, params, _retried=True
                        )

                    if resp.status == 429:
                        log.error("matchbook_rate_limited")
                        raise MatchbookRateLimitError("Rate limited — 10 minute block")

                    if resp.status >= 400:
                        body = await resp.text()
                        raise MatchbookAPIError(resp.status, body)

                    return await resp.json()  # type: ignore[no-any-return]

            except (TimeoutError, aiohttp.ClientError, OSError) as e:
                if attempt == self._max_retries - 1:
                    raise MatchbookAPIError(
                        0, f"Connection failed after {self._max_retries} retries: {e}"
                    ) from e
                log.warning(
                    "matchbook_request_retry",
                    attempt=attempt + 1,
                    backoff=backoff,
                    error=str(e),
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 300)  # max 5 min

        # Should never reach here, but satisfy type checker
        raise MatchbookAPIError(0, "Max retries exceeded")

    # --- Events ---

    async def get_events(
        self,
        sport: str,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[ExchangeEvent]:
        """Fetch events for a sport."""
        sport_id = self.SPORT_IDS.get(sport)
        if sport_id is None:
            log.warning("matchbook_unknown_sport", sport=sport)
            return []

        params: dict[str, Any] = {
            "sport-ids": sport_id,
            "states": "open",
            "per-page": 100,
            "include-markets": "true",
        }
        if date_from:
            params["after"] = date_from.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        if date_to:
            params["before"] = date_to.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        data = await self._request("GET", "/events", params=params)
        events_data = data.get("events", [])

        events = []
        for ev in events_data:
            try:
                event = self._parse_event(ev, sport)
                events.append(event)
            except (KeyError, ValueError) as e:
                log.warning("matchbook_parse_event_error", event_id=ev.get("id"), error=str(e))

        log.info("matchbook_events_fetched", sport=sport, count=len(events))
        return events

    def _parse_event(self, data: dict[str, Any], sport: str) -> ExchangeEvent:
        """Parse Matchbook event JSON to ExchangeEvent DTO."""
        name = data["name"]
        # Matchbook names: "Team A vs Team B" or "Team A v Team B"
        parts = name.replace(" v ", " vs ").split(" vs ", 1)
        home = parts[0].strip() if len(parts) > 0 else name
        away = parts[1].strip() if len(parts) > 1 else ""

        start_str = data.get("start", "")
        start_time = datetime.fromisoformat(start_str.replace("Z", "+00:00"))

        # Parse markets if included
        markets = []
        for m in data.get("markets", []):
            try:
                market = self._parse_market(m)
                markets.append(market)
            except (KeyError, ValueError):
                continue

        status_map = {
            "open": EventStatus.UPCOMING,
            "live": EventStatus.LIVE,
            "closed": EventStatus.SETTLED,
            "graded": EventStatus.SETTLED,
            "cancelled": EventStatus.CANCELLED,
        }

        return ExchangeEvent(
            event_id=data["id"],
            name=name,
            sport=sport,
            league=data.get("category-name"),
            home_team=home,
            away_team=away,
            start_time=start_time,
            status=status_map.get(data.get("status", "open"), EventStatus.UPCOMING),
            markets=markets,
        )

    def _parse_market(self, data: dict[str, Any]) -> Market:
        """Parse Matchbook market JSON."""
        market_type_map = {
            "one_x_two": "h2h",
            "moneyline": "h2h",
            "match_odds": "h2h",
            "handicap": "spreads",
            "over_under": "totals",
            "total": "totals",
        }
        raw_type = data.get("market-type", "").lower().replace(" ", "_")
        market_type = market_type_map.get(raw_type, raw_type)

        runners = []
        for r in data.get("runners", []):
            runners.append(self._parse_runner(r))

        return Market(
            market_id=data["id"],
            name=data.get("name", ""),
            market_type=market_type,
            runners=runners,
        )

    @staticmethod
    def _parse_runner(data: dict[str, Any]) -> Runner:
        """Parse Matchbook runner JSON with embedded prices."""
        side_map = {"back": Side.BACK, "lay": Side.LAY}
        prices = []
        for p in data.get("prices", []):
            side = side_map.get(p.get("side", "").lower())
            if side is None:
                continue  # Skip unknown side values
            prices.append(
                RunnerPrice(
                    odds=Decimal(str(p["odds"])),
                    available_amount=Decimal(str(p.get("available-amount", 0))),
                    side=side,
                )
            )

        return Runner(
            runner_id=data["id"],
            name=data.get("name", ""),
            prices=prices,
        )

    # --- Markets ---

    async def get_markets(self, event_id: int) -> list[Market]:
        """Fetch markets for an event."""
        data = await self._request("GET", f"/events/{event_id}/markets")
        markets = []
        for m in data.get("markets", []):
            try:
                markets.append(self._parse_market(m))
            except (KeyError, ValueError) as e:
                log.warning("matchbook_parse_market_error", event_id=event_id, error=str(e))
        return markets

    # --- Prices ---

    async def get_prices(self, event_id: int, market_id: int) -> list[Runner]:
        """Fetch runner prices for a market. Uses include-prices=true to save GETs."""
        data = await self._request(
            "GET",
            f"/events/{event_id}/markets/{market_id}/runners",
            params={"include-prices": "true"},
        )
        runners = []
        for r in data.get("runners", []):
            try:
                runners.append(self._parse_runner(r))
            except (KeyError, ValueError) as e:
                log.warning("matchbook_parse_runner_error", event_id=event_id, error=str(e))
        return runners

    # --- Betting ---

    async def place_bet(self, params: BetParams) -> BetResult:
        """Place a bet on Matchbook via POST /edge/rest/v2/offers."""
        data = await self._request(
            "POST",
            "/v2/offers",
            json_data={
                "odds": float(params.odds),
                "stake": float(params.stake),
                "side": params.side.value,
                "runner-id": params.runner_id,
            },
        )

        offers = data.get("offers", [])
        if not offers:
            raise MatchbookAPIError(0, "No offers in place_bet response")
        offer = offers[0]
        return BetResult(
            offer_id=str(offer.get("id", "")),
            status=self._map_offer_status(offer.get("status", "open")),
            odds=Decimal(str(offer.get("odds", params.odds))),
            stake=Decimal(str(offer.get("stake", params.stake))),
            matched_amount=Decimal(str(offer.get("matched-amount", 0))),
            fill_pct=(
                Decimal(str(offer.get("matched-amount", 0))) / params.stake
                if params.stake > 0
                else Decimal("0")
            ),
        )

    async def cancel_bet(self, offer_id: str) -> bool:
        """Cancel an unmatched offer."""
        try:
            await self._request("DELETE", f"/v2/offers/{offer_id}")
            log.info("matchbook_offer_cancelled", offer_id=offer_id)
            return True
        except MatchbookAPIError as e:
            log.error("matchbook_cancel_failed", offer_id=offer_id, error=str(e))
            return False

    async def get_open_bets(self) -> list[BetResult]:
        """Get all open bets."""
        data = await self._request("GET", "/bets", params={"status": "open"})
        results = []
        for bet in data.get("bets", []):
            stake = Decimal(str(bet.get("stake", 0)))
            matched = Decimal(str(bet.get("matched-amount", 0)))
            results.append(
                BetResult(
                    offer_id=str(bet.get("id", "")),
                    status=self._map_offer_status(bet.get("status", "open")),
                    odds=Decimal(str(bet.get("odds", 0))),
                    stake=stake,
                    matched_amount=matched,
                    fill_pct=matched / stake if stake > 0 else Decimal("0"),
                )
            )
        return results

    async def get_balance(self) -> Decimal:
        """Get account balance."""
        data = await self._request("GET", "/account/balance")
        return Decimal(str(data.get("balance", 0)))

    async def get_sports(self) -> list[dict[str, Any]]:
        """Fetch sport ID mapping from Matchbook. Use to discover/verify SPORT_IDS."""
        data = await self._request("GET", "/lookups/sports")
        return data.get("sports", [])  # type: ignore[no-any-return]

    @staticmethod
    def _map_offer_status(status: str) -> BetStatus:
        """Map Matchbook offer status to our BetStatus enum."""
        mapping = {
            "open": BetStatus.PENDING,
            "matched": BetStatus.MATCHED,
            "cancelled": BetStatus.CANCELLED,
            "delayed": BetStatus.PENDING,  # v2: in-play delay
            "settled": BetStatus.SETTLED,
        }
        return mapping.get(status, BetStatus.PENDING)
