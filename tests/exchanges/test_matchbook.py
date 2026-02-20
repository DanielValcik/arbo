"""Tests for Matchbook REST API client."""

from __future__ import annotations

import re
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest
from aioresponses import aioresponses

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from unittest.mock import AsyncMock

from src.exchanges.base import BetParams, BetStatus, Side
from src.exchanges.matchbook import (
    MatchbookAPIError,
    MatchbookAuthError,
    MatchbookClient,
    MatchbookConfig,
    MatchbookRateLimitError,
)

AUTH_URL = "https://api.matchbook.com/bpapi/rest"
BASE_URL = "https://api.matchbook.com/edge/rest"

# Regex patterns for URL matching (aioresponses needs these when requests include query params)
EVENTS_URL = re.compile(r"^https://api\.matchbook\.com/edge/rest/events(\?.*)?$")
RUNNERS_URL = re.compile(
    r"^https://api\.matchbook\.com/edge/rest/events/\d+/markets/\d+/runners(\?.*)?$"
)
OFFERS_URL = re.compile(r"^https://api\.matchbook\.com/edge/rest/v2/offers(/\d+)?(\?.*)?$")
BETS_URL = re.compile(r"^https://api\.matchbook\.com/edge/rest/bets(\?.*)?$")
LOGIN_URL = f"{AUTH_URL}/security/session"


@pytest.fixture
def config() -> MatchbookConfig:
    return MatchbookConfig(
        base_url=BASE_URL,
        auth_url=AUTH_URL,
        session_ttl_seconds=18000,
        max_retries=2,
        timeout_seconds=5,
    )


@pytest.fixture
async def client(config: MatchbookConfig, mock_redis: AsyncMock) -> AsyncIterator[MatchbookClient]:
    c = MatchbookClient(
        username="test_user",
        password="test_pass",
        config=config,
        redis_client=mock_redis,
    )
    yield c
    await c.close()


class TestLogin:
    async def test_successful_login(
        self,
        client: MatchbookClient,
        matchbook_session_response: dict,
        mock_redis: AsyncMock,
    ) -> None:
        with aioresponses() as m:
            m.post(
                LOGIN_URL,
                payload=matchbook_session_response,
                headers={"session-token": "test-session-token-12345"},
            )

            await client.login()

            assert client._session_token == "test-session-token-12345"
            mock_redis.set.assert_called_once_with(
                "arbo:session:matchbook",
                "test-session-token-12345",
                ex=18000,
            )

    async def test_login_failure_raises(self, client: MatchbookClient) -> None:
        with aioresponses() as m:
            m.post(LOGIN_URL, status=403, body="Forbidden")

            with pytest.raises(MatchbookAuthError, match="Login failed"):
                await client.login()

    async def test_login_no_token_raises(self, client: MatchbookClient) -> None:
        with aioresponses() as m:
            m.post(LOGIN_URL, payload={})

            with pytest.raises(MatchbookAuthError, match="No session-token"):
                await client.login()


class TestSessionManagement:
    async def test_401_triggers_reauth_and_retry(
        self,
        client: MatchbookClient,
        matchbook_session_response: dict,
        matchbook_events_response: dict,
    ) -> None:
        """On 401, client should re-authenticate and retry the original request once."""
        client._session_token = "expired-token"

        with aioresponses() as m:
            # First request: 401
            m.get(EVENTS_URL, status=401)
            # Re-auth
            m.post(
                LOGIN_URL,
                payload=matchbook_session_response,
                headers={"session-token": "new-token"},
            )
            # Retry: success
            m.get(EVENTS_URL, payload=matchbook_events_response)

            events = await client.get_events("football")
            assert len(events) == 2

    async def test_token_loaded_from_redis(
        self,
        client: MatchbookClient,
        mock_redis: AsyncMock,
        matchbook_events_response: dict,
    ) -> None:
        """Should load session token from Redis if not in memory."""
        client._session_token = None
        mock_redis.get.return_value = "cached-token"

        with aioresponses() as m:
            m.get(EVENTS_URL, payload=matchbook_events_response)
            events = await client.get_events("football")
            assert len(events) == 2
            mock_redis.get.assert_called_with("arbo:session:matchbook")


class TestGetEvents:
    async def test_fetch_events(
        self,
        client: MatchbookClient,
        matchbook_events_response: dict,
    ) -> None:
        client._session_token = "valid-token"

        with aioresponses() as m:
            m.get(EVENTS_URL, payload=matchbook_events_response)

            events = await client.get_events("football")

            assert len(events) == 2
            assert events[0].home_team == "Liverpool"
            assert events[0].away_team == "Arsenal"
            assert events[0].league == "EPL"
            assert events[0].event_id == 100001

    async def test_events_with_prices(
        self,
        client: MatchbookClient,
        matchbook_events_response: dict,
    ) -> None:
        client._session_token = "valid-token"

        with aioresponses() as m:
            m.get(EVENTS_URL, payload=matchbook_events_response)

            events = await client.get_events("football")
            market = events[0].markets[0]

            assert market.market_type == "h2h"
            assert len(market.runners) == 3

            liverpool = market.runners[0]
            assert liverpool.name == "Liverpool"
            assert liverpool.best_back is not None
            assert liverpool.best_back.odds == Decimal("2.10")
            assert liverpool.best_lay is not None
            assert liverpool.best_lay.odds == Decimal("2.14")

    async def test_unknown_sport_returns_empty(self, client: MatchbookClient) -> None:
        client._session_token = "valid-token"
        events = await client.get_events("curling")
        assert events == []


class TestGetPrices:
    async def test_fetch_prices(self, client: MatchbookClient) -> None:
        client._session_token = "valid-token"

        runner_response = {
            "runners": [
                {
                    "id": 300001,
                    "name": "Liverpool",
                    "prices": [
                        {"odds": 2.10, "available-amount": 500.0, "side": "back"},
                        {"odds": 2.14, "available-amount": 300.0, "side": "lay"},
                    ],
                }
            ]
        }

        with aioresponses() as m:
            m.get(RUNNERS_URL, payload=runner_response)

            runners = await client.get_prices(100001, 200001)

            assert len(runners) == 1
            assert runners[0].runner_id == 300001
            assert runners[0].best_back is not None
            assert runners[0].best_back.odds == Decimal("2.10")


class TestPlaceBet:
    async def test_place_bet_success(
        self,
        client: MatchbookClient,
        matchbook_offer_response: dict,
    ) -> None:
        client._session_token = "valid-token"

        with aioresponses() as m:
            m.post(OFFERS_URL, payload=matchbook_offer_response)

            result = await client.place_bet(
                BetParams(
                    event_id=100001,
                    market_id=200001,
                    runner_id=300001,
                    side=Side.BACK,
                    odds=Decimal("2.10"),
                    stake=Decimal("50"),
                )
            )

            assert result.offer_id == "999001"
            assert result.status == BetStatus.MATCHED
            assert result.matched_amount == Decimal("50")


class TestCancelBet:
    async def test_cancel_success(self, client: MatchbookClient) -> None:
        client._session_token = "valid-token"

        with aioresponses() as m:
            m.delete(OFFERS_URL, payload={"status": "cancelled"})

            result = await client.cancel_bet("999001")
            assert result is True

    async def test_cancel_failure(self, client: MatchbookClient) -> None:
        client._session_token = "valid-token"

        with aioresponses() as m:
            m.delete(OFFERS_URL, status=404, body="Not found")

            result = await client.cancel_bet("999001")
            assert result is False


class TestRateLimiting:
    async def test_429_raises_rate_limit_error(self, client: MatchbookClient) -> None:
        client._session_token = "valid-token"

        with aioresponses() as m:
            m.get(EVENTS_URL, status=429)

            with pytest.raises(MatchbookRateLimitError):
                await client.get_events("football")


class TestRetry:
    async def test_connection_error_retries(
        self,
        client: MatchbookClient,
        matchbook_events_response: dict,
    ) -> None:
        """Connection errors should be retried up to max_retries."""
        client._session_token = "valid-token"

        with aioresponses() as m:
            # First attempt: connection error
            m.get(EVENTS_URL, exception=ConnectionError("timeout"))
            # Second attempt: success
            m.get(EVENTS_URL, payload=matchbook_events_response)

            events = await client.get_events("football")
            assert len(events) == 2

    async def test_max_retries_exceeded(self, client: MatchbookClient) -> None:
        """After max retries, should raise MatchbookAPIError."""
        client._session_token = "valid-token"

        with aioresponses() as m:
            for _ in range(3):
                m.get(EVENTS_URL, exception=ConnectionError("timeout"))

            with pytest.raises(MatchbookAPIError, match="retries"):
                await client.get_events("football")


class TestGetOpenBets:
    async def test_get_open_bets(self, client: MatchbookClient) -> None:
        client._session_token = "valid-token"

        with aioresponses() as m:
            m.get(
                BETS_URL,
                payload={
                    "bets": [
                        {
                            "id": 999001,
                            "status": "open",
                            "odds": 2.10,
                            "stake": 50.0,
                            "matched-amount": 50.0,
                        }
                    ]
                },
            )

            bets = await client.get_open_bets()
            assert len(bets) == 1
            assert bets[0].offer_id == "999001"
