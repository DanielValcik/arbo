"""Tests for Santiment GraphQL client (B2-10).

All tests use mock HTTP responses — no real API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from arbo.connectors.santiment_client import (
    SYMBOL_TO_SLUG,
    SantimentClient,
    SantimentDataPoint,
    SantimentError,
    SantimentRateLimitError,
)

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

MOCK_BATCH_RESPONSE = {
    "data": {
        "getMetric": {
            "timeseriesDataPerSlug": [
                {
                    "datetime": "2026-02-25T00:00:00Z",
                    "data": [
                        {"slug": "bitcoin", "value": 950000.0},
                        {"slug": "ethereum", "value": 520000.0},
                    ],
                },
                {
                    "datetime": "2026-02-26T00:00:00Z",
                    "data": [
                        {"slug": "bitcoin", "value": 980000.0},
                        {"slug": "ethereum", "value": 540000.0},
                    ],
                },
            ]
        }
    }
}

MOCK_EMPTY_RESPONSE = {"data": {"getMetric": {"timeseriesDataPerSlug": []}}}

MOCK_GRAPHQL_ERROR = {"errors": [{"message": "Metric not found: invalid_metric"}]}


def _mock_response(json_data: dict, status: int = 200, headers: dict | None = None) -> AsyncMock:
    """Create a mock aiohttp response."""
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data)
    resp.text = AsyncMock(return_value=str(json_data))
    resp.headers = headers or {}
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=None)
    return resp


# ---------------------------------------------------------------------------
# Tests: Batch metric query
# ---------------------------------------------------------------------------


class TestGetMetricBatch:
    """Tests for get_metric_batch endpoint."""

    async def test_parses_batch_data(self) -> None:
        """Successfully parses batch metric response."""
        client = SantimentClient()
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_mock_response(MOCK_BATCH_RESPONSE))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        result = await client.get_metric_batch(
            metric="daily_active_addresses",
            slugs=["bitcoin", "ethereum"],
        )

        assert "bitcoin" in result
        assert "ethereum" in result
        assert len(result["bitcoin"]) == 2
        assert len(result["ethereum"]) == 2
        assert result["bitcoin"][0].value == 950000.0
        assert result["bitcoin"][1].value == 980000.0
        assert result["ethereum"][0].slug == "ethereum"

    async def test_data_point_fields(self) -> None:
        """Data points have correct fields."""
        client = SantimentClient()
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_mock_response(MOCK_BATCH_RESPONSE))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        result = await client.get_metric_batch(
            metric="daily_active_addresses",
            slugs=["bitcoin"],
        )

        point = result["bitcoin"][0]
        assert isinstance(point, SantimentDataPoint)
        assert point.metric == "daily_active_addresses"
        assert point.slug == "bitcoin"
        assert point.value == 950000.0

    async def test_empty_slugs_returns_empty(self) -> None:
        """Empty slug list returns empty dict without API call."""
        client = SantimentClient()
        mock_session = AsyncMock()
        client._session = mock_session

        result = await client.get_metric_batch(
            metric="daily_active_addresses",
            slugs=[],
        )

        assert result == {}
        mock_session.post.assert_not_called()

    async def test_missing_slug_in_response(self) -> None:
        """Slugs with no data in response return empty list."""
        client = SantimentClient()
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_mock_response(MOCK_BATCH_RESPONSE))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        result = await client.get_metric_batch(
            metric="daily_active_addresses",
            slugs=["bitcoin", "ethereum", "solana"],
        )

        assert "solana" in result
        assert len(result["solana"]) == 0  # Not in response data

    async def test_graphql_query_contains_metric(self) -> None:
        """GraphQL query includes metric name."""
        client = SantimentClient()
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_mock_response(MOCK_BATCH_RESPONSE))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        await client.get_metric_batch(
            metric="dev_activity",
            slugs=["bitcoin"],
        )

        call_args = mock_session.post.call_args
        query = call_args.kwargs["json"]["query"]
        assert "dev_activity" in query
        assert "bitcoin" in query


# ---------------------------------------------------------------------------
# Tests: Convenience methods
# ---------------------------------------------------------------------------


class TestConvenienceMethods:
    """Tests for get_daily_active_addresses, get_dev_activity, etc."""

    async def test_get_daily_active_addresses(self) -> None:
        """get_daily_active_addresses calls correct metric."""
        client = SantimentClient()
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_mock_response(MOCK_BATCH_RESPONSE))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        result = await client.get_daily_active_addresses(["bitcoin"])

        call_args = mock_session.post.call_args
        query = call_args.kwargs["json"]["query"]
        assert "daily_active_addresses" in query
        assert "bitcoin" in result

    async def test_get_dev_activity_returns_empty(self) -> None:
        """get_dev_activity returns empty lists (not supported on free tier batch)."""
        client = SantimentClient()
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_mock_response(MOCK_BATCH_RESPONSE))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        result = await client.get_dev_activity(["bitcoin"])

        # No API call should be made — dev_activity is not supported via batch
        mock_session.post.assert_not_called()
        assert "bitcoin" in result
        assert result["bitcoin"] == []

    async def test_get_transaction_count(self) -> None:
        """get_transaction_count calls correct metric."""
        client = SantimentClient()
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_mock_response(MOCK_BATCH_RESPONSE))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        await client.get_transaction_count(["bitcoin"])

        call_args = mock_session.post.call_args
        query = call_args.kwargs["json"]["query"]
        assert "transaction_volume" in query

    async def test_get_all_metrics_parallel(self) -> None:
        """get_all_metrics fetches 2 metrics via API + returns empty dev_activity."""
        client = SantimentClient()
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_mock_response(MOCK_BATCH_RESPONSE))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        result = await client.get_all_metrics(["bitcoin"])

        assert "bitcoin" in result
        assert "daily_active_addresses" in result["bitcoin"]
        assert "dev_activity" in result["bitcoin"]
        assert "transaction_volume" in result["bitcoin"]
        # dev_activity returns empty (not supported), so only 2 API calls
        assert mock_session.post.call_count == 2
        # dev_activity should be empty list
        assert result["bitcoin"]["dev_activity"] == []


# ---------------------------------------------------------------------------
# Tests: Caching
# ---------------------------------------------------------------------------


class TestCaching:
    """Tests for response caching."""

    async def test_cache_prevents_duplicate_calls(self) -> None:
        """Same query returns cached result."""
        client = SantimentClient()
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_mock_response(MOCK_BATCH_RESPONSE))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        await client.get_metric_batch("daily_active_addresses", ["bitcoin"])
        await client.get_metric_batch("daily_active_addresses", ["bitcoin"])

        assert mock_session.post.call_count == 1

    async def test_different_metrics_not_cached_together(self) -> None:
        """Different metrics get separate cache entries."""
        client = SantimentClient()
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_mock_response(MOCK_BATCH_RESPONSE))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        await client.get_metric_batch("daily_active_addresses", ["bitcoin"])
        await client.get_metric_batch("dev_activity", ["bitcoin"])

        assert mock_session.post.call_count == 2

    async def test_close_clears_cache(self) -> None:
        """close() clears the response cache."""
        client = SantimentClient()
        client._cache = {"test": (0, "data")}
        client._session = None

        await client.close()

        assert len(client._cache) == 0


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for HTTP error handling and retries."""

    async def test_429_retries_with_backoff(self) -> None:
        """429 response triggers retries."""
        client = SantimentClient()
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_mock_response({}, status=429))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        with pytest.raises(SantimentRateLimitError):
            await client.get_metric_batch("daily_active_addresses", ["bitcoin"])

        assert mock_session.post.call_count == 3  # MAX_RETRIES

    async def test_graphql_error_raises(self) -> None:
        """GraphQL errors in response body raise SantimentError."""
        client = SantimentClient()
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_mock_response(MOCK_GRAPHQL_ERROR))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        with pytest.raises(SantimentError, match="Metric not found"):
            await client.get_metric_batch("invalid_metric", ["bitcoin"])

    async def test_5xx_retries_then_succeeds(self) -> None:
        """5xx followed by 200 succeeds."""
        client = SantimentClient()
        mock_session = AsyncMock()

        fail_resp = _mock_response({}, status=500)
        fail_resp.text = AsyncMock(return_value="Internal Server Error")
        ok_resp = _mock_response(MOCK_BATCH_RESPONSE)

        mock_session.post = MagicMock(side_effect=[fail_resp, ok_resp])
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        result = await client.get_metric_batch("daily_active_addresses", ["bitcoin"])

        assert "bitcoin" in result
        assert mock_session.post.call_count == 2

    async def test_empty_response_returns_empty_lists(self) -> None:
        """Empty API response returns empty lists for all slugs."""
        client = SantimentClient()
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_mock_response(MOCK_EMPTY_RESPONSE))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        result = await client.get_metric_batch("daily_active_addresses", ["bitcoin"])

        assert result["bitcoin"] == []


# ---------------------------------------------------------------------------
# Tests: Rate limit tracking
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Tests for client-side rate limit tracking."""

    async def test_usage_stats_incremented(self) -> None:
        """API calls increment usage counters."""
        client = SantimentClient()
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_mock_response(MOCK_BATCH_RESPONSE))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        await client.get_metric_batch("daily_active_addresses", ["bitcoin"])
        await client.get_metric_batch("dev_activity", ["bitcoin"])

        stats = client.usage_stats
        assert stats["monthly_calls"] == 2
        assert stats["monthly_remaining"] == 998

    async def test_rate_limit_headers_update_count(self) -> None:
        """Server rate limit headers update internal tracking."""
        client = SantimentClient()
        mock_session = AsyncMock()
        mock_session.post = MagicMock(
            return_value=_mock_response(
                MOCK_BATCH_RESPONSE,
                headers={"x-ratelimit-remaining-month": "950"},
            )
        )
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        await client.get_metric_batch("daily_active_addresses", ["bitcoin"])

        assert client._monthly_count == 50  # 1000 - 950


# ---------------------------------------------------------------------------
# Tests: Slug mapping
# ---------------------------------------------------------------------------


class TestSlugMapping:
    """Tests for symbol → slug conversion."""

    def test_slugs_for_symbols(self) -> None:
        """Convert multiple symbols to slugs."""
        slugs = SantimentClient.slugs_for_symbols(["BTC", "ETH", "SOL"])
        assert slugs == ["bitcoin", "ethereum", "solana"]

    def test_unknown_symbol_skipped(self) -> None:
        """Unknown symbols are skipped."""
        slugs = SantimentClient.slugs_for_symbols(["BTC", "FAKECOIN", "ETH"])
        assert slugs == ["bitcoin", "ethereum"]

    def test_slug_for_symbol(self) -> None:
        """Single symbol conversion."""
        assert SantimentClient.slug_for_symbol("BTC") == "bitcoin"
        assert SantimentClient.slug_for_symbol("XRP") == "xrp"
        assert SantimentClient.slug_for_symbol("FAKECOIN") is None

    def test_top_20_slugs_mapped(self) -> None:
        """All top 20 crypto symbols have slug mappings."""
        expected = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT"]
        for symbol in expected:
            assert symbol in SYMBOL_TO_SLUG, f"Missing slug for {symbol}"


# ---------------------------------------------------------------------------
# Tests: Data models
# ---------------------------------------------------------------------------


class TestDataModels:
    """Tests for dataclass correctness."""

    def test_data_point_frozen(self) -> None:
        """SantimentDataPoint is immutable."""
        from datetime import UTC, datetime

        point = SantimentDataPoint(
            slug="bitcoin",
            metric="daily_active_addresses",
            value=950000.0,
            dt=datetime.now(UTC),
        )
        with pytest.raises(AttributeError):
            point.value = 100.0  # type: ignore[misc]
