"""CoinGecko Demo API connector for market + community metrics (Strategy B2-11).

Fetches bulk market data (/coins/markets) and per-coin community/developer
metrics (/coins/{id}). Used alongside Santiment for the Social Momentum
Divergence calculator.

Interface contract:
  - get_markets_bulk(per_page, page) → list[CoinMarketData]
  - get_coin_detail(coin_id) → CoinDetail | None

Auth: x-cg-demo-api-key header (COINGECKO_API_KEY env var).
Rate limits: 30/min, 10,000/month. No rate limit response headers.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import aiohttp

from arbo.utils.logger import get_logger

logger = get_logger("coingecko")

BASE_URL = "https://api.coingecko.com/api/v3"
MAX_RETRIES = 3
BASE_DELAY_S = 1.0
DEFAULT_CACHE_TTL_MARKETS_S = 10800  # 3 hours
DEFAULT_CACHE_TTL_DETAIL_S = 21600  # 6 hours

# Rate limits (Demo tier)
RATE_LIMIT_PER_MIN = 30
RATE_LIMIT_PER_MONTH = 10000

# Coin ID mapping (CoinGecko uses full names, not ticker symbols)
SYMBOL_TO_COINGECKO_ID: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "AVAX": "avalanche-2",
    "DOT": "polkadot",
    "LINK": "chainlink",
    "MATIC": "matic-network",
    "UNI": "uniswap",
    "ATOM": "cosmos",
    "LTC": "litecoin",
    "NEAR": "near",
    "ARB": "arbitrum",
    "OP": "optimism-ethereum",
    "APT": "aptos",
    "SUI": "sui",
    "FIL": "filecoin",
    "AAVE": "aave",
    "BNB": "binancecoin",
}


# ================================================================
# Data models
# ================================================================


@dataclass(frozen=True)
class CoinMarketData:
    """Bulk market data from /coins/markets endpoint."""

    id: str  # CoinGecko ID (e.g., "bitcoin")
    symbol: str  # Ticker (e.g., "btc")
    name: str
    current_price: float
    market_cap: float
    market_cap_rank: int
    total_volume: float
    price_change_24h: float
    price_change_percentage_24h: float
    price_change_percentage_7d: float | None
    price_change_percentage_30d: float | None
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class CoinDetail:
    """Detailed coin data including community + developer metrics."""

    id: str
    symbol: str
    name: str
    # Community metrics
    reddit_subscribers: int | None
    reddit_accounts_active_48h: int | None
    # Developer metrics
    forks: int | None
    stars: int | None
    commit_count_4_weeks: int | None
    code_additions_4_weeks: int | None
    code_deletions_4_weeks: int | None
    # Market data
    current_price: float | None
    market_cap: float | None
    total_volume: float | None
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))


# ================================================================
# Error types
# ================================================================


class CoinGeckoError(Exception):
    """Base error for CoinGecko API calls."""


class CoinGeckoRateLimitError(CoinGeckoError):
    """Rate limit exceeded."""


class CoinGeckoAuthError(CoinGeckoError):
    """Invalid or missing API key."""


# ================================================================
# Client
# ================================================================


class CoinGeckoClient:
    """Async CoinGecko Demo API client.

    Args:
        api_key: CoinGecko Demo API key (x-cg-demo-api-key header).
        session: Optional shared aiohttp session.
        cache_ttl_markets_s: Cache TTL for bulk market data (default 3h).
        cache_ttl_detail_s: Cache TTL for coin detail (default 6h).
    """

    def __init__(
        self,
        api_key: str = "",
        session: aiohttp.ClientSession | None = None,
        cache_ttl_markets_s: int = DEFAULT_CACHE_TTL_MARKETS_S,
        cache_ttl_detail_s: int = DEFAULT_CACHE_TTL_DETAIL_S,
    ) -> None:
        self._api_key = api_key
        self._session = session
        self._owns_session = session is None
        self._cache_ttl_markets_s = cache_ttl_markets_s
        self._cache_ttl_detail_s = cache_ttl_detail_s
        self._cache: dict[str, tuple[float, Any]] = {}

        # Rate limit tracking (no response headers available)
        self._call_timestamps: list[float] = []
        self._monthly_count = 0

        logger.info("coingecko_client_init", has_key=bool(api_key))

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
            )
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        """Close the HTTP session if we own it."""
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()
        self._cache.clear()
        logger.info("coingecko_client_closed", monthly_calls=self._monthly_count)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _get_cached(self, key: str, ttl_s: int) -> Any | None:
        """Return cached value if still fresh."""
        if key in self._cache:
            ts, val = self._cache[key]
            if time.monotonic() - ts < ttl_s:
                return val
        return None

    def _set_cached(self, key: str, val: Any) -> None:
        """Store a value in the cache."""
        self._cache[key] = (time.monotonic(), val)

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    async def _check_rate_limit(self) -> None:
        """Wait if approaching the 30/min rate limit."""
        now = time.monotonic()
        self._call_timestamps = [t for t in self._call_timestamps if now - t < 60]

        if len(self._call_timestamps) >= RATE_LIMIT_PER_MIN - 2:
            wait = 60 - (now - self._call_timestamps[0]) + 0.5
            logger.warning("coingecko_rate_limit_wait", wait_s=round(wait, 1))
            await asyncio.sleep(wait)

        # Monthly budget warning
        if self._monthly_count >= RATE_LIMIT_PER_MONTH - 200:
            logger.error(
                "coingecko_monthly_budget_critical",
                remaining=RATE_LIMIT_PER_MONTH - self._monthly_count,
            )

    def _record_call(self) -> None:
        """Record an API call for rate tracking."""
        self._call_timestamps.append(time.monotonic())
        self._monthly_count += 1

    @property
    def usage_stats(self) -> dict[str, int]:
        """Current API usage statistics."""
        return {
            "monthly_calls": self._monthly_count,
            "monthly_remaining": max(0, RATE_LIMIT_PER_MONTH - self._monthly_count),
        }

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    async def _request(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Make an authenticated GET request with retries.

        Args:
            path: API path (e.g., "/coins/markets").
            params: Query parameters.

        Returns:
            Parsed JSON response.

        Raises:
            CoinGeckoAuthError: On 401 (bad/missing key).
            CoinGeckoRateLimitError: On 429 after retries.
            CoinGeckoError: On other failures after retries.
        """
        await self._check_rate_limit()
        session = await self._get_session()
        url = f"{BASE_URL}{path}"
        headers: dict[str, str] = {}
        if self._api_key:
            headers["x-cg-demo-api-key"] = self._api_key

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                self._record_call()
                async with session.get(url, params=params, headers=headers) as resp:
                    if resp.status == 200:
                        return await resp.json()

                    # CoinGecko has inconsistent error formats
                    body = await resp.text()

                    if resp.status == 401:
                        raise CoinGeckoAuthError(
                            f"Invalid or missing CoinGecko API key: {body[:200]}"
                        )

                    if resp.status == 429:
                        delay = BASE_DELAY_S * (2**attempt)
                        logger.warning("coingecko_429", attempt=attempt + 1, delay_s=delay)
                        await asyncio.sleep(delay)
                        last_error = CoinGeckoRateLimitError(
                            f"Rate limited (attempt {attempt + 1})"
                        )
                        continue

                    if resp.status >= 500:
                        delay = BASE_DELAY_S * (2**attempt)
                        logger.warning(
                            "coingecko_5xx",
                            status=resp.status,
                            attempt=attempt + 1,
                        )
                        await asyncio.sleep(delay)
                        last_error = CoinGeckoError(f"Server error {resp.status}")
                        continue

                    raise CoinGeckoError(f"Unexpected status {resp.status}: {body[:200]}")

            except (aiohttp.ClientError, TimeoutError) as e:
                delay = BASE_DELAY_S * (2**attempt)
                logger.warning("coingecko_network_error", error=str(e), attempt=attempt + 1)
                await asyncio.sleep(delay)
                last_error = CoinGeckoError(f"Network error: {e}")

        raise last_error or CoinGeckoError("Request failed after retries")

    # ------------------------------------------------------------------
    # Public API — bulk market data
    # ------------------------------------------------------------------

    async def get_markets_bulk(
        self,
        vs_currency: str = "usd",
        per_page: int = 100,
        page: int = 1,
        price_change_percentage: str = "24h,7d,30d",
    ) -> list[CoinMarketData]:
        """Fetch bulk market data for top coins.

        Args:
            vs_currency: Target currency (default "usd").
            per_page: Results per page (max 250).
            page: Page number (1-indexed).
            price_change_percentage: Comma-separated intervals.

        Returns:
            List of CoinMarketData.
        """
        cache_key = f"markets:{vs_currency}:{per_page}:{page}"
        cached = self._get_cached(cache_key, self._cache_ttl_markets_s)
        if cached is not None:
            return cached  # type: ignore[return-value]

        data = await self._request(
            "/coins/markets",
            params={
                "vs_currency": vs_currency,
                "per_page": per_page,
                "page": page,
                "order": "market_cap_desc",
                "sparkline": "false",
                "price_change_percentage": price_change_percentage,
            },
        )

        now = datetime.now(UTC)
        coins = [self._parse_market_data(item, now) for item in data]

        self._set_cached(cache_key, coins)
        logger.info("coingecko_markets", count=len(coins), page=page)
        return coins

    # ------------------------------------------------------------------
    # Public API — coin detail with community data
    # ------------------------------------------------------------------

    async def get_coin_detail(self, coin_id: str) -> CoinDetail | None:
        """Fetch detailed coin data including community + developer metrics.

        Args:
            coin_id: CoinGecko coin ID (e.g., "bitcoin", "ripple").

        Returns:
            CoinDetail or None if not found.
        """
        cache_key = f"detail:{coin_id}"
        cached = self._get_cached(cache_key, self._cache_ttl_detail_s)
        if cached is not None:
            return cached  # type: ignore[return-value]

        try:
            data = await self._request(
                f"/coins/{coin_id}",
                params={
                    "localization": "false",
                    "tickers": "false",
                    "community_data": "true",
                    "developer_data": "true",
                    "sparkline": "false",
                },
            )
        except CoinGeckoError as e:
            logger.warning("coingecko_detail_error", coin_id=coin_id, error=str(e))
            return None

        detail = self._parse_coin_detail(data)
        self._set_cached(cache_key, detail)
        logger.info("coingecko_detail", coin_id=coin_id)
        return detail

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_market_data(raw: dict[str, Any], fetched_at: datetime) -> CoinMarketData:
        """Parse a coin from the /coins/markets response."""
        return CoinMarketData(
            id=raw.get("id", ""),
            symbol=raw.get("symbol", ""),
            name=raw.get("name", ""),
            current_price=float(raw.get("current_price") or 0),
            market_cap=float(raw.get("market_cap") or 0),
            market_cap_rank=int(raw.get("market_cap_rank") or 0),
            total_volume=float(raw.get("total_volume") or 0),
            price_change_24h=float(raw.get("price_change_24h") or 0),
            price_change_percentage_24h=float(raw.get("price_change_percentage_24h") or 0),
            price_change_percentage_7d=_safe_float(
                raw.get("price_change_percentage_7d_in_currency")
            ),
            price_change_percentage_30d=_safe_float(
                raw.get("price_change_percentage_30d_in_currency")
            ),
            fetched_at=fetched_at,
        )

    @staticmethod
    def _parse_coin_detail(raw: dict[str, Any]) -> CoinDetail:
        """Parse a coin from the /coins/{id} response."""
        community = raw.get("community_data") or {}
        developer = raw.get("developer_data") or {}
        market = raw.get("market_data") or {}

        # Code additions/deletions are nested
        code_changes = developer.get("code_additions_deletions_4_weeks") or {}

        return CoinDetail(
            id=raw.get("id", ""),
            symbol=raw.get("symbol", ""),
            name=raw.get("name", ""),
            reddit_subscribers=_safe_int(community.get("reddit_subscribers")),
            reddit_accounts_active_48h=_safe_int(community.get("reddit_accounts_active_48h")),
            forks=_safe_int(developer.get("forks")),
            stars=_safe_int(developer.get("stars")),
            commit_count_4_weeks=_safe_int(developer.get("commit_count_4_weeks")),
            code_additions_4_weeks=_safe_int(code_changes.get("additions")),
            code_deletions_4_weeks=_safe_int(code_changes.get("deletions")),
            current_price=_safe_float((market.get("current_price") or {}).get("usd")),
            market_cap=_safe_float((market.get("market_cap") or {}).get("usd")),
            total_volume=_safe_float((market.get("total_volume") or {}).get("usd")),
        )

    @staticmethod
    def coingecko_id_for_symbol(symbol: str) -> str | None:
        """Convert a ticker symbol to CoinGecko ID."""
        return SYMBOL_TO_COINGECKO_ID.get(symbol.upper())


# ================================================================
# Helpers
# ================================================================


def _safe_float(value: Any) -> float | None:
    """Convert value to float or None."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_int(value: Any) -> int | None:
    """Convert value to int or None."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None
