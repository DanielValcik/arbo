"""Tests for LunarCrush API v4 client (B2-01R).

All tests use mock HTTP responses — no real API calls.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arbo.connectors.lunarcrush_client import (
    CoinSnapshot,
    LunarCrushAuthError,
    LunarCrushClient,
    LunarCrushError,
    LunarCrushRateLimitError,
    TimeSeriesPoint,
    TopicDetail,
)


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

MOCK_COIN_DATA = {
    "config": {
        "sort": "market_cap_rank",
        "limit": 10,
        "page": 0,
        "total_rows": 6702,
    },
    "data": [
        {
            "id": 1,
            "symbol": "BTC",
            "name": "Bitcoin",
            "price": 95000.50,
            "market_cap": 1890000000000,
            "market_cap_rank": 1,
            "volume_24h": 45000000000,
            "percent_change_24h": -2.1,
            "percent_change_7d": 5.3,
            "percent_change_30d": 12.0,
            "social_dominance": 25.5,
            "sentiment": 68,
            "galaxy_score": 72,
            "alt_rank": 5,
            "interactions_24h": 5000000,
            "social_volume_24h": 120000,
            "topic": "bitcoin",
            "categories": "layer-1,store-of-value",
        },
        {
            "id": 2,
            "symbol": "ETH",
            "name": "Ethereum",
            "price": 3200.00,
            "market_cap": 380000000000,
            "market_cap_rank": 2,
            "volume_24h": 15000000000,
            "percent_change_24h": 1.5,
            "percent_change_7d": -3.2,
            "percent_change_30d": 8.0,
            "social_dominance": 12.3,
            "sentiment": 62,
            "galaxy_score": 68,
            "alt_rank": 8,
            "interactions_24h": 2000000,
            "social_volume_24h": 80000,
            "topic": "ethereum",
            "categories": "layer-1,smart-contracts",
        },
    ],
}

MOCK_TIMESERIES_DATA = {
    "config": {
        "coin": "BTC",
        "bucket": "hour",
        "interval": "1w",
    },
    "data": [
        {
            "time": 1739000000,
            "open": 94500.0,
            "high": 95200.0,
            "low": 94100.0,
            "close": 95000.0,
            "volume_24h": 45000000000,
            "market_cap": 1890000000000,
            "social_dominance": 25.5,
            "sentiment": 68,
            "galaxy_score": 72,
            "alt_rank": 5,
            "interactions": 2000000,
            "contributors_active": 50000,
            "posts_created": 8000,
        },
        {
            "time": 1739003600,
            "open": 95000.0,
            "high": 95500.0,
            "low": 94800.0,
            "close": 95300.0,
            "volume_24h": 44000000000,
            "market_cap": 1895000000000,
            "social_dominance": 26.0,
            "sentiment": 70,
            "galaxy_score": 73,
            "alt_rank": 4,
            "interactions": 2100000,
            "contributors_active": 52000,
            "posts_created": 8500,
        },
    ],
}

MOCK_TOPIC_DATA = {
    "config": {
        "topic": "bitcoin",
    },
    "data": {
        "topic": "bitcoin",
        "title": "Bitcoin",
        "topic_rank": 1,
        "interactions_24h": 5000000,
        "num_contributors": 50000,
        "num_posts": 30000,
        "trend": "up",
        "categories": ["layer-1", "store-of-value"],
        "types_sentiment": {
            "tweet": 65,
            "reddit-post": 70,
            "youtube-video": 72,
        },
    },
}


def _mock_response(json_data: dict, status: int = 200) -> AsyncMock:
    """Create a mock aiohttp response."""
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data)
    resp.text = AsyncMock(return_value=str(json_data))
    # context manager support
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=None)
    return resp


# ---------------------------------------------------------------------------
# Tests: Coin Snapshot
# ---------------------------------------------------------------------------


class TestGetCoinsSnapshot:
    """Tests for get_coins_snapshot endpoint."""

    async def test_parses_coin_data(self) -> None:
        """Successfully parses bulk coin snapshot response."""
        client = LunarCrushClient(api_key="test-key")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(MOCK_COIN_DATA))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        coins = await client.get_coins_snapshot(limit=10)

        assert len(coins) == 2
        assert coins[0].symbol == "BTC"
        assert coins[0].name == "Bitcoin"
        assert coins[0].price == 95000.50
        assert coins[0].social_dominance == 25.5
        assert coins[0].sentiment == 68
        assert coins[0].galaxy_score == 72
        assert coins[0].interactions_24h == 5000000

    async def test_second_coin_parsed(self) -> None:
        """Second coin in response is parsed correctly."""
        client = LunarCrushClient(api_key="test-key")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(MOCK_COIN_DATA))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        coins = await client.get_coins_snapshot()

        assert coins[1].symbol == "ETH"
        assert coins[1].market_cap_rank == 2
        assert coins[1].percent_change_24h == 1.5
        assert coins[1].categories == "layer-1,smart-contracts"

    async def test_passes_query_params(self) -> None:
        """Correct query params are sent."""
        client = LunarCrushClient(api_key="test-key")
        mock_session = AsyncMock()
        mock_resp = _mock_response(MOCK_COIN_DATA)
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        await client.get_coins_snapshot(limit=50, sort="galaxy_score", page=2)

        mock_session.get.assert_called_once()
        call_kwargs = mock_session.get.call_args
        assert call_kwargs.kwargs["params"]["limit"] == 50
        assert call_kwargs.kwargs["params"]["sort"] == "galaxy_score"
        assert call_kwargs.kwargs["params"]["page"] == 2

    async def test_caching_prevents_duplicate_calls(self) -> None:
        """Cached responses prevent duplicate API calls."""
        client = LunarCrushClient(api_key="test-key")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(MOCK_COIN_DATA))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        coins1 = await client.get_coins_snapshot()
        coins2 = await client.get_coins_snapshot()

        assert mock_session.get.call_count == 1
        assert len(coins1) == len(coins2)

    async def test_to_dict_serialization(self) -> None:
        """CoinSnapshot.to_dict() returns all required fields."""
        client = LunarCrushClient(api_key="test-key")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(MOCK_COIN_DATA))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        coins = await client.get_coins_snapshot()
        d = coins[0].to_dict()

        assert d["symbol"] == "BTC"
        assert d["social_dominance"] == 25.5
        assert d["sentiment"] == 68
        assert d["galaxy_score"] == 72
        assert "price" in d
        assert "market_cap" in d


# ---------------------------------------------------------------------------
# Tests: Time Series
# ---------------------------------------------------------------------------


class TestGetCoinTimeseries:
    """Tests for get_coin_timeseries endpoint."""

    async def test_parses_timeseries(self) -> None:
        """Successfully parses hourly time-series response."""
        client = LunarCrushClient(api_key="test-key")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(MOCK_TIMESERIES_DATA))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        points = await client.get_coin_timeseries("BTC", bucket="hour", interval="1w")

        assert len(points) == 2
        assert points[0].time == 1739000000
        assert points[0].close == 95000.0
        assert points[0].social_dominance == 25.5
        assert points[0].sentiment == 68
        assert points[0].contributors_active == 50000

    async def test_timeseries_url_contains_coin(self) -> None:
        """API URL includes coin symbol in path."""
        client = LunarCrushClient(api_key="test-key")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(MOCK_TIMESERIES_DATA))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        await client.get_coin_timeseries("SOL")

        url = mock_session.get.call_args[0][0]
        assert "/coins/SOL/time-series/v2" in url

    async def test_timeseries_caching(self) -> None:
        """Time-series cached separately per coin+bucket+interval."""
        client = LunarCrushClient(api_key="test-key")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(MOCK_TIMESERIES_DATA))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        await client.get_coin_timeseries("BTC", bucket="hour", interval="1w")
        await client.get_coin_timeseries("BTC", bucket="hour", interval="1w")
        await client.get_coin_timeseries("ETH", bucket="hour", interval="1w")

        assert mock_session.get.call_count == 2  # BTC cached, ETH is new


# ---------------------------------------------------------------------------
# Tests: Topic Detail
# ---------------------------------------------------------------------------


class TestGetTopicDetail:
    """Tests for get_topic_detail endpoint."""

    async def test_parses_topic(self) -> None:
        """Successfully parses topic detail response."""
        client = LunarCrushClient(api_key="test-key")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(MOCK_TOPIC_DATA))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        detail = await client.get_topic_detail("bitcoin")

        assert detail is not None
        assert detail.topic == "bitcoin"
        assert detail.title == "Bitcoin"
        assert detail.topic_rank == 1
        assert detail.trend == "up"
        assert detail.interactions_24h == 5000000
        assert detail.types_sentiment["tweet"] == 65

    async def test_returns_none_on_empty_data(self) -> None:
        """Returns None when topic not found."""
        client = LunarCrushClient(api_key="test-key")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(
            return_value=_mock_response({"config": {}, "data": {}})
        )
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        detail = await client.get_topic_detail("nonexistent")

        assert detail is None

    async def test_returns_none_on_error(self) -> None:
        """Returns None when API returns error."""
        client = LunarCrushClient(api_key="test-key")
        mock_session = AsyncMock()
        resp = _mock_response({}, status=404)
        resp.text = AsyncMock(return_value="Not found")
        mock_session.get = MagicMock(return_value=resp)
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        detail = await client.get_topic_detail("nonexistent")

        assert detail is None


# ---------------------------------------------------------------------------
# Tests: Error Handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for HTTP error handling and retries."""

    async def test_401_raises_auth_error(self) -> None:
        """401 response raises LunarCrushAuthError immediately."""
        client = LunarCrushClient(api_key="bad-key")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response({}, status=401))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        with pytest.raises(LunarCrushAuthError):
            await client.get_coins_snapshot()

        # Should NOT retry on 401
        assert mock_session.get.call_count == 1

    async def test_429_retries_with_backoff(self) -> None:
        """429 response triggers retries with exponential backoff."""
        client = LunarCrushClient(api_key="test-key")
        mock_session = AsyncMock()
        # All retries return 429
        mock_session.get = MagicMock(return_value=_mock_response({}, status=429))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        with pytest.raises(LunarCrushRateLimitError):
            await client.get_coins_snapshot()

        assert mock_session.get.call_count == 3  # MAX_RETRIES

    async def test_5xx_retries_then_succeeds(self) -> None:
        """5xx followed by 200 succeeds."""
        client = LunarCrushClient(api_key="test-key")
        mock_session = AsyncMock()

        fail_resp = _mock_response({}, status=500)
        fail_resp.text = AsyncMock(return_value="Internal Server Error")
        ok_resp = _mock_response(MOCK_COIN_DATA)

        # First call fails, second succeeds
        mock_session.get = MagicMock(side_effect=[fail_resp, ok_resp])
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        coins = await client.get_coins_snapshot()

        assert len(coins) == 2
        assert mock_session.get.call_count == 2

    async def test_no_api_key_still_works(self) -> None:
        """Client works without API key (free tier, limited data)."""
        client = LunarCrushClient()  # No key
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(MOCK_COIN_DATA))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        coins = await client.get_coins_snapshot()

        assert len(coins) == 2
        # Verify no Authorization header sent
        call_kwargs = mock_session.get.call_args
        assert "Authorization" not in call_kwargs.kwargs.get("headers", {})


# ---------------------------------------------------------------------------
# Tests: Rate Limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Tests for client-side rate limit tracking."""

    async def test_usage_stats_incremented(self) -> None:
        """API calls increment usage counters."""
        client = LunarCrushClient(api_key="test-key")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(MOCK_COIN_DATA))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        await client.get_coins_snapshot(limit=10, sort="market_cap_rank", page=0)
        # Bust cache for second call
        await client.get_coins_snapshot(limit=10, sort="galaxy_score", page=0)

        stats = client.usage_stats
        assert stats["daily_calls"] == 2
        assert stats["monthly_calls"] == 2

    async def test_rate_limit_wait(self) -> None:
        """Client waits when approaching rate limit."""
        client = LunarCrushClient(api_key="test-key")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(MOCK_COIN_DATA))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        # Fill up the rate limit window
        import time

        now = time.monotonic()
        client._call_timestamps = [now - i for i in range(10)]  # 10 calls in last minute

        with patch("arbo.connectors.lunarcrush_client.asyncio.sleep") as mock_sleep:
            # Different cache key so it actually calls API
            await client.get_coins_snapshot(limit=5, sort="market_cap_rank", page=99)
            mock_sleep.assert_called()


# ---------------------------------------------------------------------------
# Tests: Bearer Auth
# ---------------------------------------------------------------------------


class TestAuth:
    """Tests for Bearer token authentication."""

    async def test_bearer_token_sent(self) -> None:
        """Authorization header includes Bearer token."""
        client = LunarCrushClient(api_key="my-secret-key")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(MOCK_COIN_DATA))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        await client.get_coins_snapshot()

        call_kwargs = mock_session.get.call_args
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer my-secret-key"

    async def test_close_clears_cache(self) -> None:
        """close() clears the response cache."""
        client = LunarCrushClient(api_key="test-key")
        client._cache = {"test": (0, "data")}
        client._session = None

        await client.close()

        assert len(client._cache) == 0


# ---------------------------------------------------------------------------
# Tests: Data model
# ---------------------------------------------------------------------------


class TestDataModels:
    """Tests for dataclass correctness."""

    def test_coin_snapshot_frozen(self) -> None:
        """CoinSnapshot is immutable."""
        coin = CoinSnapshot(
            id=1, symbol="BTC", name="Bitcoin", price=95000.0,
            market_cap=1.89e12, market_cap_rank=1, volume_24h=4.5e10,
            percent_change_24h=-2.1, percent_change_7d=5.3, percent_change_30d=12.0,
            social_dominance=25.5, sentiment=68, galaxy_score=72, alt_rank=5,
            interactions_24h=5000000, social_volume_24h=120000,
            topic="bitcoin", categories="layer-1",
        )
        with pytest.raises(AttributeError):
            coin.price = 100000.0  # type: ignore[misc]

    def test_timeseries_point_frozen(self) -> None:
        """TimeSeriesPoint is immutable."""
        point = TimeSeriesPoint(
            time=1739000000, open=94500.0, high=95200.0, low=94100.0,
            close=95000.0, volume_24h=4.5e10, market_cap=1.89e12,
            social_dominance=25.5, sentiment=68, galaxy_score=72, alt_rank=5,
            interactions=2000000, contributors_active=50000, posts_created=8000,
        )
        with pytest.raises(AttributeError):
            point.close = 96000.0  # type: ignore[misc]

    def test_topic_detail_frozen(self) -> None:
        """TopicDetail is immutable."""
        detail = TopicDetail(
            topic="bitcoin", title="Bitcoin", topic_rank=1,
            interactions_24h=5000000, num_contributors=50000, num_posts=30000,
            trend="up", categories=["layer-1"], types_sentiment={"tweet": 65},
        )
        with pytest.raises(AttributeError):
            detail.topic_rank = 2  # type: ignore[misc]
