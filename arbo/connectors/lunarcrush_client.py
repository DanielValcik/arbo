"""LunarCrush API v4 connector for crypto social momentum data (Strategy B2).

Provides social dominance, sentiment, galaxy score, and market metrics for
14,000+ crypto projects. Used by the Social Momentum Divergence calculator
to detect divergences between social attention and Polymarket prices.

Interface contract:
  - get_coins_snapshot(limit, sort) → list[CoinSnapshot]
  - get_coin_timeseries(coin, bucket, interval) → list[TimeSeriesPoint]
  - get_topic_detail(topic) → TopicDetail | None

Auth: Bearer token via LUNARCRUSH_API_KEY env var.
Rate limits: ~10 req/min (Individual tier). Client-side tracking enforced.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import aiohttp

from arbo.utils.logger import get_logger

logger = get_logger("lunarcrush")

BASE_URL = "https://lunarcrush.com/api4"
MAX_RETRIES = 3
BASE_DELAY_S = 1.0
DEFAULT_CACHE_TTL_SNAPSHOT_S = 10800  # 3 hours
DEFAULT_CACHE_TTL_TIMESERIES_S = 3600  # 1 hour
DEFAULT_CACHE_TTL_TOPIC_S = 3600  # 1 hour
RATE_LIMIT_PER_MIN = 10  # Individual tier


# ================================================================
# Data models
# ================================================================


@dataclass(frozen=True)
class CoinSnapshot:
    """Single coin from LunarCrush bulk snapshot."""

    id: int
    symbol: str
    name: str
    price: float
    market_cap: float
    market_cap_rank: int
    volume_24h: float
    percent_change_24h: float
    percent_change_7d: float
    percent_change_30d: float
    social_dominance: float
    sentiment: int  # 0-100 (positive post %)
    galaxy_score: float  # 0-100 (composite quality)
    alt_rank: int
    interactions_24h: int
    social_volume_24h: int
    topic: str  # social topic slug
    categories: str  # comma-separated
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Serialize for DB insertion."""
        return {
            "symbol": self.symbol,
            "name": self.name,
            "price": self.price,
            "market_cap": self.market_cap,
            "market_cap_rank": self.market_cap_rank,
            "volume_24h": self.volume_24h,
            "percent_change_24h": self.percent_change_24h,
            "percent_change_7d": self.percent_change_7d,
            "percent_change_30d": self.percent_change_30d,
            "social_dominance": self.social_dominance,
            "sentiment": self.sentiment,
            "galaxy_score": self.galaxy_score,
            "alt_rank": self.alt_rank,
            "interactions_24h": self.interactions_24h,
            "social_volume_24h": self.social_volume_24h,
            "topic": self.topic,
            "categories": self.categories,
        }


@dataclass(frozen=True)
class TimeSeriesPoint:
    """Single data point from LunarCrush coin time-series."""

    time: int  # unix timestamp (seconds)
    open: float
    high: float
    low: float
    close: float
    volume_24h: float
    market_cap: float
    social_dominance: float
    sentiment: int
    galaxy_score: float
    alt_rank: int
    interactions: int
    contributors_active: int
    posts_created: int


@dataclass(frozen=True)
class TopicDetail:
    """Aggregated social analytics for a topic."""

    topic: str
    title: str
    topic_rank: int
    interactions_24h: int
    num_contributors: int
    num_posts: int
    trend: str  # "up", "down", "flat"
    categories: list[str]
    types_sentiment: dict[str, int]  # network → sentiment score (0-100)


# ================================================================
# Error types
# ================================================================


class LunarCrushError(Exception):
    """Base error for LunarCrush API calls."""


class LunarCrushAuthError(LunarCrushError):
    """Invalid or missing API key."""


class LunarCrushRateLimitError(LunarCrushError):
    """Rate limit exceeded."""


# ================================================================
# Client
# ================================================================


class LunarCrushClient:
    """Async LunarCrush API v4 client.

    Args:
        api_key: LunarCrush API key (Bearer token).
        session: Optional shared aiohttp session.
        cache_ttl_snapshot_s: Cache TTL for coin snapshots (default 3h).
        cache_ttl_timeseries_s: Cache TTL for time-series (default 1h).
    """

    def __init__(
        self,
        api_key: str = "",
        session: aiohttp.ClientSession | None = None,
        cache_ttl_snapshot_s: int = DEFAULT_CACHE_TTL_SNAPSHOT_S,
        cache_ttl_timeseries_s: int = DEFAULT_CACHE_TTL_TIMESERIES_S,
    ) -> None:
        self._api_key = api_key
        self._session = session
        self._owns_session = session is None
        self._cache_ttl_snapshot_s = cache_ttl_snapshot_s
        self._cache_ttl_timeseries_s = cache_ttl_timeseries_s
        self._cache: dict[str, tuple[float, Any]] = {}

        # Rate limit tracking
        self._call_timestamps: list[float] = []
        self._daily_count = 0
        self._monthly_count = 0
        self._day_start = time.monotonic()

        logger.info(
            "lunarcrush_client_init",
            has_key=bool(api_key),
        )

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
        logger.info(
            "lunarcrush_client_closed",
            daily_calls=self._daily_count,
            monthly_calls=self._monthly_count,
        )

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
        """Wait if we're approaching the rate limit (10 req/min)."""
        now = time.monotonic()
        # Remove timestamps older than 60s
        self._call_timestamps = [t for t in self._call_timestamps if now - t < 60]

        if len(self._call_timestamps) >= RATE_LIMIT_PER_MIN:
            wait = 60 - (now - self._call_timestamps[0]) + 0.1
            logger.warning("lunarcrush_rate_limit_wait", wait_s=round(wait, 1))
            await asyncio.sleep(wait)

    def _record_call(self) -> None:
        """Record an API call for rate tracking."""
        now = time.monotonic()
        self._call_timestamps.append(now)
        self._daily_count += 1
        self._monthly_count += 1

        # Reset daily counter every 24h
        if now - self._day_start > 86400:
            self._day_start = now
            self._daily_count = 1

        # Warn at 80% of daily budget (~1600 of 2000)
        if self._daily_count % 100 == 0 or self._daily_count > 1600:
            logger.info(
                "lunarcrush_api_usage",
                daily=self._daily_count,
                monthly=self._monthly_count,
            )

    @property
    def usage_stats(self) -> dict[str, int]:
        """Current API usage statistics."""
        return {
            "daily_calls": self._daily_count,
            "monthly_calls": self._monthly_count,
        }

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    async def _request(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make an authenticated GET request with retries.

        Args:
            path: API path (e.g., "/public/coins/list/v2").
            params: Query parameters.

        Returns:
            Parsed JSON response.

        Raises:
            LunarCrushAuthError: On 401 (bad/missing key).
            LunarCrushRateLimitError: On 429 after retries exhausted.
            LunarCrushError: On other failures after retries.
        """
        await self._check_rate_limit()
        session = await self._get_session()
        url = f"{BASE_URL}{path}"
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                self._record_call()
                async with session.get(url, params=params, headers=headers) as resp:
                    if resp.status == 200:
                        return await resp.json()  # type: ignore[no-any-return]

                    if resp.status == 401:
                        raise LunarCrushAuthError("Invalid or missing LunarCrush API key")

                    if resp.status == 429:
                        delay = BASE_DELAY_S * (2**attempt)
                        logger.warning(
                            "lunarcrush_429",
                            attempt=attempt + 1,
                            delay_s=delay,
                        )
                        await asyncio.sleep(delay)
                        last_error = LunarCrushRateLimitError(f"Rate limited (attempt {attempt + 1})")
                        continue

                    if resp.status >= 500:
                        delay = BASE_DELAY_S * (2**attempt)
                        body = await resp.text()
                        logger.warning(
                            "lunarcrush_5xx",
                            status=resp.status,
                            body=body[:200],
                            attempt=attempt + 1,
                            delay_s=delay,
                        )
                        await asyncio.sleep(delay)
                        last_error = LunarCrushError(f"Server error {resp.status}")
                        continue

                    body = await resp.text()
                    raise LunarCrushError(f"Unexpected status {resp.status}: {body[:200]}")

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                delay = BASE_DELAY_S * (2**attempt)
                logger.warning(
                    "lunarcrush_network_error",
                    error=str(e),
                    attempt=attempt + 1,
                    delay_s=delay,
                )
                await asyncio.sleep(delay)
                last_error = LunarCrushError(f"Network error: {e}")

        raise last_error or LunarCrushError("Request failed after retries")

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    async def get_coins_snapshot(
        self,
        limit: int = 100,
        sort: str = "market_cap_rank",
        page: int = 0,
    ) -> list[CoinSnapshot]:
        """Fetch bulk coin snapshot with social + market metrics.

        Args:
            limit: Results per page (max 1000).
            sort: Sort field (e.g., "market_cap_rank", "galaxy_score").
            page: Page number (0-indexed).

        Returns:
            List of CoinSnapshot dataclasses.
        """
        cache_key = f"snapshot:{sort}:{limit}:{page}"
        cached = self._get_cached(cache_key, self._cache_ttl_snapshot_s)
        if cached is not None:
            return cached  # type: ignore[return-value]

        resp = await self._request(
            "/public/coins/list/v2",
            params={"sort": sort, "limit": limit, "page": page},
        )

        data = resp.get("data", [])
        now = datetime.now(UTC)
        coins = [self._parse_coin(item, now) for item in data]

        self._set_cached(cache_key, coins)
        logger.info(
            "lunarcrush_snapshot",
            count=len(coins),
            page=page,
            sort=sort,
        )
        return coins

    async def get_coin_timeseries(
        self,
        coin: str,
        bucket: str = "hour",
        interval: str = "1w",
    ) -> list[TimeSeriesPoint]:
        """Fetch historical time-series for a single coin.

        Args:
            coin: Coin symbol (e.g., "BTC") or LunarCrush ID.
            bucket: Aggregation interval ("hour" or "day").
            interval: Time range ("1w", "1m", "3m", "6m", "1y").

        Returns:
            List of TimeSeriesPoint dataclasses (chronological order).
        """
        cache_key = f"timeseries:{coin}:{bucket}:{interval}"
        cached = self._get_cached(cache_key, self._cache_ttl_timeseries_s)
        if cached is not None:
            return cached  # type: ignore[return-value]

        resp = await self._request(
            f"/public/coins/{coin}/time-series/v2",
            params={"bucket": bucket, "interval": interval},
        )

        data = resp.get("data", [])
        points = [self._parse_timeseries_point(item) for item in data]

        self._set_cached(cache_key, points)
        logger.info(
            "lunarcrush_timeseries",
            coin=coin,
            bucket=bucket,
            interval=interval,
            points=len(points),
        )
        return points

    async def get_topic_detail(self, topic: str) -> TopicDetail | None:
        """Fetch detailed social analytics for a topic.

        Args:
            topic: Topic slug (e.g., "bitcoin", "ethereum").

        Returns:
            TopicDetail or None if topic not found.
        """
        cache_key = f"topic:{topic}"
        cached = self._get_cached(cache_key, DEFAULT_CACHE_TTL_TOPIC_S)
        if cached is not None:
            return cached  # type: ignore[return-value]

        try:
            resp = await self._request(f"/public/topic/{topic}/v1")
        except LunarCrushError as e:
            logger.warning("lunarcrush_topic_not_found", topic=topic, error=str(e))
            return None

        data = resp.get("data", {})
        if not data:
            return None

        detail = self._parse_topic_detail(data)
        self._set_cached(cache_key, detail)
        logger.info("lunarcrush_topic", topic=topic, rank=detail.topic_rank)
        return detail

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_coin(raw: dict[str, Any], fetched_at: datetime) -> CoinSnapshot:
        """Parse a coin from the bulk snapshot response."""
        return CoinSnapshot(
            id=raw.get("id", 0),
            symbol=raw.get("symbol", ""),
            name=raw.get("name", ""),
            price=float(raw.get("price", 0)),
            market_cap=float(raw.get("market_cap", 0)),
            market_cap_rank=int(raw.get("market_cap_rank", 0)),
            volume_24h=float(raw.get("volume_24h", 0)),
            percent_change_24h=float(raw.get("percent_change_24h", 0)),
            percent_change_7d=float(raw.get("percent_change_7d", 0)),
            percent_change_30d=float(raw.get("percent_change_30d", 0)),
            social_dominance=float(raw.get("social_dominance", 0)),
            sentiment=int(raw.get("sentiment", 0)),
            galaxy_score=float(raw.get("galaxy_score", 0)),
            alt_rank=int(raw.get("alt_rank", 0)),
            interactions_24h=int(raw.get("interactions_24h", 0)),
            social_volume_24h=int(raw.get("social_volume_24h", 0)),
            topic=raw.get("topic", ""),
            categories=raw.get("categories", ""),
            fetched_at=fetched_at,
        )

    @staticmethod
    def _parse_timeseries_point(raw: dict[str, Any]) -> TimeSeriesPoint:
        """Parse a single time-series data point."""
        return TimeSeriesPoint(
            time=int(raw.get("time", 0)),
            open=float(raw.get("open", 0)),
            high=float(raw.get("high", 0)),
            low=float(raw.get("low", 0)),
            close=float(raw.get("close", 0)),
            volume_24h=float(raw.get("volume_24h", 0)),
            market_cap=float(raw.get("market_cap", 0)),
            social_dominance=float(raw.get("social_dominance", 0)),
            sentiment=int(raw.get("sentiment", 0)),
            galaxy_score=float(raw.get("galaxy_score", 0)),
            alt_rank=int(raw.get("alt_rank", 0)),
            interactions=int(raw.get("interactions", 0)),
            contributors_active=int(raw.get("contributors_active", 0)),
            posts_created=int(raw.get("posts_created", 0)),
        )

    @staticmethod
    def _parse_topic_detail(raw: dict[str, Any]) -> TopicDetail:
        """Parse a topic detail response."""
        return TopicDetail(
            topic=raw.get("topic", ""),
            title=raw.get("title", ""),
            topic_rank=int(raw.get("topic_rank", 0)),
            interactions_24h=int(raw.get("interactions_24h", 0)),
            num_contributors=int(raw.get("num_contributors", 0)),
            num_posts=int(raw.get("num_posts", 0)),
            trend=raw.get("trend", "flat"),
            categories=raw.get("categories", []),
            types_sentiment=raw.get("types_sentiment", {}),
        )
