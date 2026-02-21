"""Tests for historical odds endpoint on OddsApiClient.

Tests verify:
1. Parse historical response wrapper (data + timestamps)
2. Returns previous/next timestamps
3. Quota tracking on historical calls
4. Low quota blocks request
5. HTTP error handling
6. Empty response handling
"""

from __future__ import annotations

import re
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from aioresponses import aioresponses

from arbo.connectors.odds_api_client import OddsApiClient, OddsEvent

# ================================================================
# Test data
# ================================================================

HISTORICAL_RESPONSE = {
    "timestamp": "2025-10-01T12:00:00Z",
    "previous_timestamp": "2025-10-01T06:00:00Z",
    "next_timestamp": "2025-10-01T18:00:00Z",
    "data": [
        {
            "id": "hist_001",
            "sport_key": "soccer_epl",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "commence_time": "2025-10-02T15:00:00Z",
            "bookmakers": [
                {
                    "key": "pinnacle",
                    "title": "Pinnacle",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Arsenal", "price": 2.10},
                                {"name": "Chelsea", "price": 3.50},
                                {"name": "Draw", "price": 3.40},
                            ],
                        }
                    ],
                }
            ],
        },
        {
            "id": "hist_002",
            "sport_key": "soccer_epl",
            "home_team": "Liverpool",
            "away_team": "Man City",
            "commence_time": "2025-10-02T17:30:00Z",
            "bookmakers": [
                {
                    "key": "pinnacle",
                    "title": "Pinnacle",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Liverpool", "price": 2.40},
                                {"name": "Man City", "price": 2.80},
                                {"name": "Draw", "price": 3.60},
                            ],
                        }
                    ],
                }
            ],
        },
    ],
}

HISTORICAL_EMPTY_RESPONSE = {
    "timestamp": "2025-10-01T12:00:00Z",
    "previous_timestamp": None,
    "next_timestamp": None,
    "data": [],
}

HISTORICAL_URL_PATTERN = re.compile(
    r"https://api\.the-odds-api\.com/v4/historical/sports/.+/odds\?.*"
)


# ================================================================
# Fixtures
# ================================================================


@pytest.fixture()
def client() -> OddsApiClient:
    """Create an OddsApiClient with test config."""
    with patch("arbo.connectors.odds_api_client.get_config") as mock_config:
        cfg = MagicMock()
        cfg.odds_api_key = "test_key"
        cfg.odds_api.base_url = "https://api.the-odds-api.com/v4"
        cfg.odds_api.regions = "eu"
        cfg.odds_api.markets = "h2h"
        cfg.odds_api.odds_format = "decimal"
        cfg.odds_api.min_remaining_quota = 50
        mock_config.return_value = cfg
        return OddsApiClient()


# ================================================================
# Tests
# ================================================================


class TestHistoricalOddsParsing:
    """Parse historical response wrapper correctly."""

    @pytest.mark.asyncio()
    async def test_parse_historical_response(self, client: OddsApiClient) -> None:
        """Should parse data array into OddsEvent objects."""
        with aioresponses() as m:
            m.get(
                HISTORICAL_URL_PATTERN,
                payload=HISTORICAL_RESPONSE,
                headers={"x-requests-remaining": "450"},
            )

            events, _prev_ts, _next_ts = await client.get_historical_odds(
                "soccer_epl", "2025-10-01T12:00:00Z"
            )

        assert len(events) == 2
        assert isinstance(events[0], OddsEvent)
        assert events[0].home_team == "Arsenal"
        assert events[1].home_team == "Liverpool"

    @pytest.mark.asyncio()
    async def test_events_have_bookmakers(self, client: OddsApiClient) -> None:
        """Parsed events should have bookmaker data."""
        with aioresponses() as m:
            m.get(
                HISTORICAL_URL_PATTERN,
                payload=HISTORICAL_RESPONSE,
                headers={"x-requests-remaining": "450"},
            )

            events, _, _ = await client.get_historical_odds("soccer_epl", "2025-10-01T12:00:00Z")

        assert len(events[0].bookmakers) == 1
        h2h = events[0].get_pinnacle_h2h()
        assert h2h is not None
        assert h2h["Arsenal"] == Decimal("2.10")


class TestHistoricalTimestamps:
    """Returns previous/next timestamps from response wrapper."""

    @pytest.mark.asyncio()
    async def test_returns_timestamps(self, client: OddsApiClient) -> None:
        """Should return previous and next timestamps."""
        with aioresponses() as m:
            m.get(
                HISTORICAL_URL_PATTERN,
                payload=HISTORICAL_RESPONSE,
                headers={"x-requests-remaining": "450"},
            )

            _, prev_ts, next_ts = await client.get_historical_odds(
                "soccer_epl", "2025-10-01T12:00:00Z"
            )

        assert prev_ts == "2025-10-01T06:00:00Z"
        assert next_ts == "2025-10-01T18:00:00Z"

    @pytest.mark.asyncio()
    async def test_none_timestamps_on_empty(self, client: OddsApiClient) -> None:
        """Empty response should return None timestamps."""
        with aioresponses() as m:
            m.get(
                HISTORICAL_URL_PATTERN,
                payload=HISTORICAL_EMPTY_RESPONSE,
                headers={"x-requests-remaining": "450"},
            )

            events, prev_ts, next_ts = await client.get_historical_odds(
                "soccer_epl", "2025-10-01T12:00:00Z"
            )

        assert len(events) == 0
        assert prev_ts is None
        assert next_ts is None


class TestHistoricalQuota:
    """Quota tracking on historical calls."""

    @pytest.mark.asyncio()
    async def test_quota_tracked(self, client: OddsApiClient) -> None:
        """Should update remaining quota from response header."""
        with aioresponses() as m:
            m.get(
                HISTORICAL_URL_PATTERN,
                payload=HISTORICAL_RESPONSE,
                headers={"x-requests-remaining": "398"},
            )

            await client.get_historical_odds("soccer_epl", "2025-10-01T12:00:00Z")

        assert client.remaining_quota == 398

    @pytest.mark.asyncio()
    async def test_low_quota_blocks_request(self, client: OddsApiClient) -> None:
        """Should return empty when quota is below minimum."""
        client._remaining_quota = 10  # Below min_remaining_quota (50)

        events, prev_ts, next_ts = await client.get_historical_odds(
            "soccer_epl", "2025-10-01T12:00:00Z"
        )

        assert events == []
        assert prev_ts is None
        assert next_ts is None


class TestHistoricalErrors:
    """HTTP error handling for historical endpoint."""

    @pytest.mark.asyncio()
    async def test_http_error_returns_empty(self, client: OddsApiClient) -> None:
        """Non-200 response should return empty results."""
        with aioresponses() as m:
            m.get(
                HISTORICAL_URL_PATTERN,
                status=500,
                body="Internal Server Error",
            )

            events, prev_ts, next_ts = await client.get_historical_odds(
                "soccer_epl", "2025-10-01T12:00:00Z"
            )

        assert events == []
        assert prev_ts is None
        assert next_ts is None

    @pytest.mark.asyncio()
    async def test_empty_data_returns_empty_list(self, client: OddsApiClient) -> None:
        """Response with empty data array should return empty event list."""
        with aioresponses() as m:
            m.get(
                HISTORICAL_URL_PATTERN,
                payload=HISTORICAL_EMPTY_RESPONSE,
                headers={"x-requests-remaining": "450"},
            )

            events, _, _ = await client.get_historical_odds("soccer_epl", "2025-10-01T12:00:00Z")

        assert events == []
