"""Tests for The Odds API v4 client."""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest
from aioresponses import aioresponses

from src.data.odds_api import SPORT_KEY_MAP, OddsApiClient

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_URL_PATTERN = re.compile(r"https://api\.the-odds-api\.com/v4/sports/.+/odds\?.*")


@pytest.fixture
def odds_api_response() -> list[dict]:
    """Mock Odds API response for soccer_epl."""
    return [
        {
            "id": "abc123",
            "sport_key": "soccer_epl",
            "home_team": "Liverpool",
            "away_team": "Arsenal",
            "commence_time": "2026-03-01T15:00:00Z",
            "bookmakers": [
                {
                    "key": "bet365",
                    "title": "Bet365",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Liverpool", "price": 2.10},
                                {"name": "Draw", "price": 3.40},
                                {"name": "Arsenal", "price": 3.60},
                            ],
                        }
                    ],
                },
                {
                    "key": "williamhill",
                    "title": "William Hill",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Liverpool", "price": 2.15},
                                {"name": "Draw", "price": 3.30},
                                {"name": "Arsenal", "price": 3.50},
                            ],
                        }
                    ],
                },
            ],
        },
        {
            "id": "def456",
            "sport_key": "soccer_epl",
            "home_team": "Manchester City",
            "away_team": "Chelsea",
            "commence_time": "2026-03-01T17:30:00Z",
            "bookmakers": [],
        },
    ]


@pytest.fixture
def mock_config():
    """Patch get_config for Odds API tests."""
    from src.utils.config import ArboConfig, SportConfig

    config = ArboConfig(
        MATCHBOOK_USERNAME="test",
        MATCHBOOK_PASSWORD="test",
        ODDS_API_KEY="test-key-123",
        sports={
            "football": SportConfig(leagues=["EPL"]),
            "basketball": SportConfig(leagues=["NBA"]),
        },
    )
    with patch("src.data.odds_api.get_config", return_value=config):
        yield config


class TestGetOdds:
    async def test_fetch_odds(self, mock_config, odds_api_response: list[dict]) -> None:
        client = OddsApiClient()
        try:
            with aioresponses() as m:
                m.get(
                    ODDS_URL_PATTERN,
                    payload=odds_api_response,
                    headers={"x-requests-remaining": "450", "x-requests-used": "50"},
                )
                events = await client.get_odds("soccer_epl")

            assert len(events) == 2
            assert events[0].id == "abc123"
            assert events[0].home_team == "Liverpool"
            assert events[0].away_team == "Arsenal"
            assert len(events[0].bookmakers) == 2
            assert events[0].bookmakers[0].key == "bet365"
            assert len(events[0].bookmakers[0].markets) == 1
            assert len(events[0].bookmakers[0].markets[0].outcomes) == 3
        finally:
            await client.close()

    async def test_quota_tracking(self, mock_config, odds_api_response: list[dict]) -> None:
        client = OddsApiClient()
        try:
            assert client.remaining_quota is None

            with aioresponses() as m:
                m.get(
                    ODDS_URL_PATTERN,
                    payload=odds_api_response,
                    headers={"x-requests-remaining": "450", "x-requests-used": "50"},
                )
                await client.get_odds("soccer_epl")

            assert client.remaining_quota == 450
        finally:
            await client.close()

    async def test_quota_low_stops_requests(self, mock_config) -> None:
        client = OddsApiClient()
        try:
            client._remaining_quota = 30  # Below min_remaining_quota (50)

            events = await client.get_odds("soccer_epl")
            assert events == []
        finally:
            await client.close()

    async def test_empty_response(self, mock_config) -> None:
        client = OddsApiClient()
        try:
            with aioresponses() as m:
                m.get(
                    ODDS_URL_PATTERN,
                    payload=[],
                    headers={"x-requests-remaining": "499"},
                )
                events = await client.get_odds("soccer_epl")

            assert events == []
        finally:
            await client.close()

    async def test_auth_error_returns_empty(self, mock_config) -> None:
        client = OddsApiClient()
        try:
            with aioresponses() as m:
                m.get(ODDS_URL_PATTERN, status=401)
                events = await client.get_odds("soccer_epl")

            assert events == []
        finally:
            await client.close()

    async def test_rate_limit_returns_empty(self, mock_config) -> None:
        client = OddsApiClient()
        try:
            with aioresponses() as m:
                m.get(ODDS_URL_PATTERN, status=429)
                events = await client.get_odds("soccer_epl")

            assert events == []
        finally:
            await client.close()

    async def test_server_error_returns_empty(self, mock_config) -> None:
        client = OddsApiClient()
        try:
            with aioresponses() as m:
                m.get(ODDS_URL_PATTERN, status=500, body="Internal Server Error")
                events = await client.get_odds("soccer_epl")

            assert events == []
        finally:
            await client.close()

    async def test_malformed_event_skipped(self, mock_config) -> None:
        """Events with missing required fields are skipped, not crashing."""
        client = OddsApiClient()
        try:
            with aioresponses() as m:
                m.get(
                    ODDS_URL_PATTERN,
                    payload=[{"id": "bad", "sport_key": "soccer_epl"}],  # missing fields
                    headers={"x-requests-remaining": "499"},
                )
                events = await client.get_odds("soccer_epl")

            assert events == []
        finally:
            await client.close()

    async def test_event_dto_parsing(self, mock_config, odds_api_response: list[dict]) -> None:
        """Verify the DTO correctly parses decimal prices."""
        client = OddsApiClient()
        try:
            with aioresponses() as m:
                m.get(
                    ODDS_URL_PATTERN,
                    payload=odds_api_response,
                    headers={"x-requests-remaining": "449"},
                )
                events = await client.get_odds("soccer_epl")

            from decimal import Decimal

            outcome = events[0].bookmakers[0].markets[0].outcomes[0]
            assert outcome.name == "Liverpool"
            assert outcome.price == Decimal("2.10")
        finally:
            await client.close()


class TestSportKeyMap:
    def test_football_keys_exist(self) -> None:
        assert "football" in SPORT_KEY_MAP
        assert "soccer_epl" in SPORT_KEY_MAP["football"]
        assert "soccer_uefa_champs_league" in SPORT_KEY_MAP["football"]

    def test_basketball_keys_exist(self) -> None:
        assert "basketball" in SPORT_KEY_MAP
        assert "basketball_nba" in SPORT_KEY_MAP["basketball"]


class TestGetAllOdds:
    async def test_fetches_all_configured_sports(
        self, mock_config, odds_api_response: list[dict]
    ) -> None:
        client = OddsApiClient()
        try:
            with aioresponses() as m:
                # Mock all sport key URLs
                for _ in range(10):  # enough for all sport keys
                    m.get(
                        ODDS_URL_PATTERN,
                        payload=odds_api_response,
                        headers={"x-requests-remaining": "400"},
                    )

                events = await client.get_all_odds()

            # Should have events from multiple sport keys
            assert len(events) > 0
        finally:
            await client.close()
