"""Tests for CoinGecko Demo API client (B2-11).

All tests use mock HTTP responses — no real API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arbo.connectors.coingecko_client import (
    SYMBOL_TO_COINGECKO_ID,
    CoinDetail,
    CoinGeckoAuthError,
    CoinGeckoClient,
    CoinGeckoRateLimitError,
    CoinMarketData,
)

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

MOCK_MARKETS_RESPONSE = [
    {
        "id": "bitcoin",
        "symbol": "btc",
        "name": "Bitcoin",
        "current_price": 95000.50,
        "market_cap": 1890000000000,
        "market_cap_rank": 1,
        "total_volume": 45000000000,
        "price_change_24h": -2100.0,
        "price_change_percentage_24h": -2.1,
        "price_change_percentage_7d_in_currency": 5.3,
        "price_change_percentage_30d_in_currency": 12.0,
    },
    {
        "id": "ethereum",
        "symbol": "eth",
        "name": "Ethereum",
        "current_price": 3200.00,
        "market_cap": 380000000000,
        "market_cap_rank": 2,
        "total_volume": 15000000000,
        "price_change_24h": 50.0,
        "price_change_percentage_24h": 1.5,
        "price_change_percentage_7d_in_currency": -3.2,
        "price_change_percentage_30d_in_currency": 8.0,
    },
]

MOCK_COIN_DETAIL = {
    "id": "bitcoin",
    "symbol": "btc",
    "name": "Bitcoin",
    "community_data": {
        "reddit_subscribers": 5200000,
        "reddit_accounts_active_48h": 12000,
    },
    "developer_data": {
        "forks": 36000,
        "stars": 78000,
        "commit_count_4_weeks": 150,
        "code_additions_deletions_4_weeks": {
            "additions": 5000,
            "deletions": 3000,
        },
    },
    "market_data": {
        "current_price": {"usd": 95000.50},
        "market_cap": {"usd": 1890000000000},
        "total_volume": {"usd": 45000000000},
    },
}

MOCK_COIN_DETAIL_MISSING_DATA = {
    "id": "newcoin",
    "symbol": "new",
    "name": "NewCoin",
    "community_data": None,
    "developer_data": None,
    "market_data": None,
}


def _mock_response(json_data: dict | list, status: int = 200) -> AsyncMock:
    """Create a mock aiohttp response."""
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data)
    resp.text = AsyncMock(return_value=str(json_data))
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=None)
    return resp


# ---------------------------------------------------------------------------
# Tests: Bulk market data
# ---------------------------------------------------------------------------


class TestGetMarketsBulk:
    """Tests for get_markets_bulk endpoint."""

    async def test_parses_market_data(self) -> None:
        """Successfully parses bulk market response."""
        client = CoinGeckoClient(api_key="test-key")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(MOCK_MARKETS_RESPONSE))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        coins = await client.get_markets_bulk()

        assert len(coins) == 2
        assert coins[0].id == "bitcoin"
        assert coins[0].symbol == "btc"
        assert coins[0].current_price == 95000.50
        assert coins[0].market_cap == 1890000000000
        assert coins[0].price_change_percentage_24h == -2.1
        assert coins[0].price_change_percentage_7d == 5.3
        assert coins[0].price_change_percentage_30d == 12.0

    async def test_second_coin_parsed(self) -> None:
        """Second coin in response is parsed correctly."""
        client = CoinGeckoClient(api_key="test-key")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(MOCK_MARKETS_RESPONSE))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        coins = await client.get_markets_bulk()

        assert coins[1].id == "ethereum"
        assert coins[1].market_cap_rank == 2
        assert coins[1].price_change_percentage_24h == 1.5

    async def test_passes_query_params(self) -> None:
        """Correct query params are sent."""
        client = CoinGeckoClient(api_key="test-key")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(MOCK_MARKETS_RESPONSE))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        await client.get_markets_bulk(per_page=50, page=2)

        call_kwargs = mock_session.get.call_args
        params = call_kwargs.kwargs["params"]
        assert params["per_page"] == 50
        assert params["page"] == 2
        assert params["vs_currency"] == "usd"

    async def test_caching_prevents_duplicate_calls(self) -> None:
        """Cached responses prevent duplicate API calls."""
        client = CoinGeckoClient(api_key="test-key")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(MOCK_MARKETS_RESPONSE))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        await client.get_markets_bulk()
        await client.get_markets_bulk()

        assert mock_session.get.call_count == 1

    async def test_different_pages_cached_separately(self) -> None:
        """Different pages get separate cache entries."""
        client = CoinGeckoClient(api_key="test-key")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(MOCK_MARKETS_RESPONSE))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        await client.get_markets_bulk(page=1)
        await client.get_markets_bulk(page=2)

        assert mock_session.get.call_count == 2


# ---------------------------------------------------------------------------
# Tests: Coin detail
# ---------------------------------------------------------------------------


class TestGetCoinDetail:
    """Tests for get_coin_detail endpoint."""

    async def test_parses_detail_with_community(self) -> None:
        """Successfully parses coin detail with community data."""
        client = CoinGeckoClient(api_key="test-key")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(MOCK_COIN_DETAIL))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        detail = await client.get_coin_detail("bitcoin")

        assert detail is not None
        assert detail.id == "bitcoin"
        assert detail.reddit_subscribers == 5200000
        assert detail.reddit_accounts_active_48h == 12000
        assert detail.forks == 36000
        assert detail.stars == 78000
        assert detail.commit_count_4_weeks == 150
        assert detail.code_additions_4_weeks == 5000
        assert detail.code_deletions_4_weeks == 3000
        assert detail.current_price == 95000.50

    async def test_handles_missing_community_data(self) -> None:
        """Handles coins with no community/developer data."""
        client = CoinGeckoClient(api_key="test-key")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(MOCK_COIN_DETAIL_MISSING_DATA))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        detail = await client.get_coin_detail("newcoin")

        assert detail is not None
        assert detail.id == "newcoin"
        assert detail.reddit_subscribers is None
        assert detail.forks is None
        assert detail.current_price is None

    async def test_returns_none_on_error(self) -> None:
        """Returns None when API returns error."""
        client = CoinGeckoClient(api_key="test-key")
        mock_session = AsyncMock()
        resp = _mock_response({"error": "coin not found"}, status=404)
        resp.text = AsyncMock(return_value='{"error": "coin not found"}')
        mock_session.get = MagicMock(return_value=resp)
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        detail = await client.get_coin_detail("nonexistent")

        assert detail is None

    async def test_detail_cached(self) -> None:
        """Detail responses are cached."""
        client = CoinGeckoClient(api_key="test-key")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(MOCK_COIN_DETAIL))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        await client.get_coin_detail("bitcoin")
        await client.get_coin_detail("bitcoin")

        assert mock_session.get.call_count == 1


# ---------------------------------------------------------------------------
# Tests: Auth
# ---------------------------------------------------------------------------


class TestAuth:
    """Tests for API key authentication."""

    async def test_api_key_header_sent(self) -> None:
        """x-cg-demo-api-key header is sent."""
        client = CoinGeckoClient(api_key="my-demo-key")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(MOCK_MARKETS_RESPONSE))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        await client.get_markets_bulk()

        call_kwargs = mock_session.get.call_args
        assert call_kwargs.kwargs["headers"]["x-cg-demo-api-key"] == "my-demo-key"

    async def test_no_key_no_header(self) -> None:
        """No API key means no auth header."""
        client = CoinGeckoClient()  # No key
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(MOCK_MARKETS_RESPONSE))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        await client.get_markets_bulk()

        call_kwargs = mock_session.get.call_args
        assert "x-cg-demo-api-key" not in call_kwargs.kwargs.get("headers", {})

    async def test_401_raises_auth_error(self) -> None:
        """401 response raises CoinGeckoAuthError immediately."""
        client = CoinGeckoClient(api_key="bad-key")
        mock_session = AsyncMock()
        resp = _mock_response({"status": {"error_code": 401}}, status=401)
        resp.text = AsyncMock(return_value='{"status": {"error_code": 401}}')
        mock_session.get = MagicMock(return_value=resp)
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        with pytest.raises(CoinGeckoAuthError):
            await client.get_markets_bulk()

        assert mock_session.get.call_count == 1


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for HTTP error handling and retries."""

    async def test_429_retries_with_backoff(self) -> None:
        """429 response triggers retries."""
        client = CoinGeckoClient(api_key="test-key")
        mock_session = AsyncMock()
        resp = _mock_response({}, status=429)
        resp.text = AsyncMock(return_value='{"status": {"error_code": 429}}')
        mock_session.get = MagicMock(return_value=resp)
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        with pytest.raises(CoinGeckoRateLimitError):
            await client.get_markets_bulk()

        assert mock_session.get.call_count == 3  # MAX_RETRIES

    async def test_5xx_retries_then_succeeds(self) -> None:
        """5xx followed by 200 succeeds."""
        client = CoinGeckoClient(api_key="test-key")
        mock_session = AsyncMock()

        fail_resp = _mock_response({}, status=500)
        fail_resp.text = AsyncMock(return_value="Internal Server Error")
        ok_resp = _mock_response(MOCK_MARKETS_RESPONSE)

        mock_session.get = MagicMock(side_effect=[fail_resp, ok_resp])
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        coins = await client.get_markets_bulk()

        assert len(coins) == 2
        assert mock_session.get.call_count == 2


# ---------------------------------------------------------------------------
# Tests: Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Tests for client-side rate limit tracking."""

    async def test_usage_stats_incremented(self) -> None:
        """API calls increment usage counters."""
        client = CoinGeckoClient(api_key="test-key")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(MOCK_MARKETS_RESPONSE))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        await client.get_markets_bulk(page=1)
        await client.get_markets_bulk(page=2)

        stats = client.usage_stats
        assert stats["monthly_calls"] == 2
        assert stats["monthly_remaining"] == 9998

    async def test_rate_limit_wait(self) -> None:
        """Client waits when approaching per-minute rate limit."""
        client = CoinGeckoClient(api_key="test-key")
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=_mock_response(MOCK_MARKETS_RESPONSE))
        mock_session.closed = False
        client._session = mock_session
        client._owns_session = False

        import time

        now = time.monotonic()
        client._call_timestamps = [now - i for i in range(30)]  # 30 calls in last minute

        with patch("arbo.connectors.coingecko_client.asyncio.sleep") as mock_sleep:
            await client.get_markets_bulk(page=99)
            mock_sleep.assert_called()

    async def test_close_clears_cache(self) -> None:
        """close() clears the response cache."""
        client = CoinGeckoClient(api_key="test-key")
        client._cache = {"test": (0, "data")}
        client._session = None

        await client.close()

        assert len(client._cache) == 0


# ---------------------------------------------------------------------------
# Tests: Symbol mapping
# ---------------------------------------------------------------------------


class TestSymbolMapping:
    """Tests for symbol → CoinGecko ID conversion."""

    def test_coingecko_id_for_symbol(self) -> None:
        """Convert ticker to CoinGecko ID."""
        assert CoinGeckoClient.coingecko_id_for_symbol("BTC") == "bitcoin"
        assert CoinGeckoClient.coingecko_id_for_symbol("XRP") == "ripple"
        assert CoinGeckoClient.coingecko_id_for_symbol("BNB") == "binancecoin"
        assert CoinGeckoClient.coingecko_id_for_symbol("FAKECOIN") is None

    def test_case_insensitive(self) -> None:
        """Symbol lookup is case-insensitive."""
        assert CoinGeckoClient.coingecko_id_for_symbol("btc") == "bitcoin"
        assert CoinGeckoClient.coingecko_id_for_symbol("Eth") == "ethereum"

    def test_top_coins_mapped(self) -> None:
        """All top crypto symbols have ID mappings."""
        expected = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "BNB"]
        for symbol in expected:
            assert symbol in SYMBOL_TO_COINGECKO_ID, f"Missing ID for {symbol}"


# ---------------------------------------------------------------------------
# Tests: Data models
# ---------------------------------------------------------------------------


class TestDataModels:
    """Tests for dataclass correctness."""

    def test_coin_market_data_frozen(self) -> None:
        """CoinMarketData is immutable."""
        coin = CoinMarketData(
            id="bitcoin",
            symbol="btc",
            name="Bitcoin",
            current_price=95000.0,
            market_cap=1.89e12,
            market_cap_rank=1,
            total_volume=4.5e10,
            price_change_24h=-2100.0,
            price_change_percentage_24h=-2.1,
            price_change_percentage_7d=5.3,
            price_change_percentage_30d=12.0,
        )
        with pytest.raises(AttributeError):
            coin.current_price = 100000.0  # type: ignore[misc]

    def test_coin_detail_frozen(self) -> None:
        """CoinDetail is immutable."""
        detail = CoinDetail(
            id="bitcoin",
            symbol="btc",
            name="Bitcoin",
            reddit_subscribers=5200000,
            reddit_accounts_active_48h=12000,
            forks=36000,
            stars=78000,
            commit_count_4_weeks=150,
            code_additions_4_weeks=5000,
            code_deletions_4_weeks=3000,
            current_price=95000.0,
            market_cap=1.89e12,
            total_volume=4.5e10,
        )
        with pytest.raises(AttributeError):
            detail.reddit_subscribers = 0  # type: ignore[misc]
