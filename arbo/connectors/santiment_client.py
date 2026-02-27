"""Santiment Free GraphQL connector for on-chain metrics (Strategy B2-10).

Fetches daily_active_addresses and transaction_count via Santiment's free
GraphQL API. Uses batch queries (timeseriesDataPerSlug) to minimize API call
count within the 1,000/month free tier limit.

Note: dev_activity is NOT supported via timeseriesDataPerSlug on the free tier.
It has been removed from batch queries. The get_dev_activity() method returns
empty data for backward compatibility.

Interface contract:
  - get_metric_batch(metric, slugs, from_dt, to_dt) → dict[slug, list[DataPoint]]
  - get_daily_active_addresses(slugs, days) → dict[slug, list[DataPoint]]
  - get_dev_activity(slugs, days) → dict[slug, list[DataPoint]]  (returns empty — not supported)
  - get_transaction_count(slugs, days) → dict[slug, list[DataPoint]]

Auth: None (free tier).
Rate limits: 100/min, 500/hour, 1,000/month.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import aiohttp

from arbo.utils.logger import get_logger

logger = get_logger("santiment")

GRAPHQL_URL = "https://api.santiment.net/graphql"
MAX_RETRIES = 3
BASE_DELAY_S = 1.0
DEFAULT_CACHE_TTL_S = 21600  # 6 hours

# Rate limits (free tier)
RATE_LIMIT_PER_MIN = 100
RATE_LIMIT_PER_HOUR = 500
RATE_LIMIT_PER_MONTH = 1000

# Top crypto slugs for batch queries
SYMBOL_TO_SLUG: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "xrp",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "AVAX": "avalanche",
    "DOT": "polkadot-new",
    "LINK": "chainlink",
    "MATIC": "matic-network",
    "UNI": "uniswap",
    "ATOM": "cosmos",
    "LTC": "litecoin",
    "NEAR": "near-protocol",
    "ARB": "arbitrum",
    "OP": "optimism",
    "APT": "aptos",
    "SUI": "sui",
    "FIL": "filecoin",
    "AAVE": "aave",
}

# Metrics available on free tier via timeseriesDataPerSlug (recent data)
# NOTE: dev_activity is NOT supported via batch queries on free tier.
# NOTE: transaction_volume has 30-day lag on free tier (recent data unavailable).
FREE_METRICS = frozenset(
    {
        "daily_active_addresses",
        "active_addresses_24h",
        "price_usd",
    }
)

# Metrics that exist on Santiment but are NOT available for recent data (free tier)
UNSUPPORTED_BATCH_METRICS = frozenset(
    {
        "dev_activity",         # "not implemented for dev_activity"
        "transaction_volume",   # 30-day lag, recent queries rejected
    }
)


# ================================================================
# Data models
# ================================================================


@dataclass(frozen=True)
class SantimentDataPoint:
    """Single metric data point from Santiment."""

    slug: str
    metric: str
    value: float
    dt: datetime


# ================================================================
# Error types
# ================================================================


class SantimentError(Exception):
    """Base error for Santiment API calls."""


class SantimentRateLimitError(SantimentError):
    """Rate limit exceeded."""


# ================================================================
# Client
# ================================================================


class SantimentClient:
    """Async Santiment Free-tier GraphQL client.

    Args:
        session: Optional shared aiohttp session.
        cache_ttl_s: Cache TTL in seconds (default 6h).
    """

    def __init__(
        self,
        session: aiohttp.ClientSession | None = None,
        cache_ttl_s: int = DEFAULT_CACHE_TTL_S,
    ) -> None:
        self._session = session
        self._owns_session = session is None
        self._cache_ttl_s = cache_ttl_s
        self._cache: dict[str, tuple[float, Any]] = {}

        # Rate limit tracking
        self._minute_timestamps: list[float] = []
        self._hour_timestamps: list[float] = []
        self._monthly_count = 0

        logger.info("santiment_client_init")

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
        logger.info("santiment_client_closed", monthly_calls=self._monthly_count)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _get_cached(self, key: str) -> Any | None:
        """Return cached value if still fresh."""
        if key in self._cache:
            ts, val = self._cache[key]
            if time.monotonic() - ts < self._cache_ttl_s:
                return val
        return None

    def _set_cached(self, key: str, val: Any) -> None:
        """Store a value in the cache."""
        self._cache[key] = (time.monotonic(), val)

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    async def _check_rate_limit(self) -> None:
        """Wait if approaching rate limits."""
        now = time.monotonic()

        # Per-minute check (100/min)
        self._minute_timestamps = [t for t in self._minute_timestamps if now - t < 60]
        if len(self._minute_timestamps) >= RATE_LIMIT_PER_MIN - 5:
            wait = 60 - (now - self._minute_timestamps[0]) + 0.5
            logger.warning("santiment_rate_limit_minute", wait_s=round(wait, 1))
            await asyncio.sleep(wait)

        # Per-hour check (500/hour)
        self._hour_timestamps = [t for t in self._hour_timestamps if now - t < 3600]
        if len(self._hour_timestamps) >= RATE_LIMIT_PER_HOUR - 10:
            wait = 3600 - (now - self._hour_timestamps[0]) + 1.0
            logger.warning("santiment_rate_limit_hour", wait_s=round(wait, 1))
            await asyncio.sleep(min(wait, 300))  # Max 5min wait

        # Monthly budget warning
        if self._monthly_count >= RATE_LIMIT_PER_MONTH - 50:
            logger.error(
                "santiment_monthly_budget_critical",
                remaining=RATE_LIMIT_PER_MONTH - self._monthly_count,
            )

    def _record_call(self) -> None:
        """Record an API call for rate tracking."""
        now = time.monotonic()
        self._minute_timestamps.append(now)
        self._hour_timestamps.append(now)
        self._monthly_count += 1

    def _update_from_headers(self, headers: dict[str, str]) -> None:
        """Update rate limit tracking from response headers."""
        remaining_month = headers.get("x-ratelimit-remaining-month")
        if remaining_month is not None:
            try:
                server_remaining = int(remaining_month)
                self._monthly_count = RATE_LIMIT_PER_MONTH - server_remaining
            except ValueError:
                pass

    @property
    def usage_stats(self) -> dict[str, int]:
        """Current API usage statistics."""
        return {
            "monthly_calls": self._monthly_count,
            "monthly_remaining": max(0, RATE_LIMIT_PER_MONTH - self._monthly_count),
        }

    # ------------------------------------------------------------------
    # GraphQL
    # ------------------------------------------------------------------

    async def _query(self, gql: str) -> dict[str, Any]:
        """Execute a GraphQL query with retries.

        Args:
            gql: GraphQL query string.

        Returns:
            Parsed JSON response data.

        Raises:
            SantimentRateLimitError: On 429 after retries.
            SantimentError: On other failures after retries.
        """
        await self._check_rate_limit()
        session = await self._get_session()

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                self._record_call()
                async with session.post(
                    GRAPHQL_URL,
                    json={"query": gql},
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    self._update_from_headers(dict(resp.headers))

                    if resp.status == 200:
                        body = await resp.json()
                        if "errors" in body:
                            errors = body["errors"]
                            msg = errors[0].get("message", str(errors))
                            raise SantimentError(f"GraphQL error: {msg}")
                        return body.get("data", {})  # type: ignore[no-any-return]

                    if resp.status == 429:
                        delay = BASE_DELAY_S * (2**attempt)
                        logger.warning("santiment_429", attempt=attempt + 1, delay_s=delay)
                        await asyncio.sleep(delay)
                        last_error = SantimentRateLimitError(
                            f"Rate limited (attempt {attempt + 1})"
                        )
                        continue

                    if resp.status >= 500:
                        delay = BASE_DELAY_S * (2**attempt)
                        logger.warning("santiment_5xx", status=resp.status, attempt=attempt + 1)
                        await asyncio.sleep(delay)
                        last_error = SantimentError(f"Server error {resp.status}")
                        continue

                    body_text = await resp.text()
                    raise SantimentError(f"Unexpected status {resp.status}: {body_text[:200]}")

            except (aiohttp.ClientError, TimeoutError) as e:
                delay = BASE_DELAY_S * (2**attempt)
                logger.warning("santiment_network_error", error=str(e), attempt=attempt + 1)
                await asyncio.sleep(delay)
                last_error = SantimentError(f"Network error: {e}")

        raise last_error or SantimentError("Request failed after retries")

    # ------------------------------------------------------------------
    # Public API — batch metric query
    # ------------------------------------------------------------------

    async def get_metric_batch(
        self,
        metric: str,
        slugs: list[str],
        from_dt: datetime | None = None,
        to_dt: datetime | None = None,
        interval: str = "1d",
    ) -> dict[str, list[SantimentDataPoint]]:
        """Fetch a single metric for multiple slugs in one API call.

        Uses timeseriesDataPerSlug for efficient batching.

        Args:
            metric: Santiment metric name (e.g., "daily_active_addresses").
            slugs: List of Santiment slugs (e.g., ["bitcoin", "ethereum"]).
            from_dt: Start datetime (default: 2 days ago).
            to_dt: End datetime (default: now).
            interval: Aggregation interval (default "1d").

        Returns:
            Dict of {slug: [SantimentDataPoint, ...]}.
        """
        if not slugs:
            return {}

        now = datetime.now(UTC)
        if from_dt is None:
            from_dt = now - timedelta(days=2)
        if to_dt is None:
            to_dt = now

        # Check cache
        cache_key = f"{metric}:{','.join(sorted(slugs))}:{from_dt.date()}:{to_dt.date()}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        from_iso = from_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        to_iso = to_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        slugs_str = ", ".join(f'"{s}"' for s in slugs)

        gql = f"""
        {{
          getMetric(metric: "{metric}") {{
            timeseriesDataPerSlug(
              from: "{from_iso}"
              to: "{to_iso}"
              interval: "{interval}"
              selector: {{ slugs: [{slugs_str}] }}
            ) {{
              datetime
              data {{
                slug
                value
              }}
            }}
          }}
        }}
        """

        data = await self._query(gql)

        # Parse response
        result: dict[str, list[SantimentDataPoint]] = {slug: [] for slug in slugs}
        get_metric = data.get("getMetric", {})
        timeseries = get_metric.get("timeseriesDataPerSlug", [])

        for entry in timeseries:
            dt_str = entry.get("datetime", "")
            try:
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue

            for item in entry.get("data", []):
                slug = item.get("slug", "")
                value = item.get("value")
                if slug in result and value is not None:
                    result[slug].append(
                        SantimentDataPoint(
                            slug=slug,
                            metric=metric,
                            value=float(value),
                            dt=dt,
                        )
                    )

        self._set_cached(cache_key, result)
        total_points = sum(len(pts) for pts in result.values())
        logger.info(
            "santiment_batch",
            metric=metric,
            slugs=len(slugs),
            data_points=total_points,
        )
        return result

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    async def get_daily_active_addresses(
        self,
        slugs: list[str],
        days: int = 2,
    ) -> dict[str, list[SantimentDataPoint]]:
        """Fetch daily active addresses for multiple coins."""
        from_dt = datetime.now(UTC) - timedelta(days=days)
        return await self.get_metric_batch(
            metric="daily_active_addresses",
            slugs=slugs,
            from_dt=from_dt,
        )

    async def get_dev_activity(
        self,
        slugs: list[str],
        days: int = 2,
    ) -> dict[str, list[SantimentDataPoint]]:
        """Return empty data — dev_activity is NOT supported via batch queries on free tier.

        The Santiment free tier returns:
        "The timeseries_data_per_slug function is not implemented for dev_activity"

        This method is kept for backward compatibility but always returns empty lists.
        """
        logger.debug(
            "santiment_dev_activity_skipped",
            reason="not supported via timeseriesDataPerSlug on free tier",
            slugs=len(slugs),
        )
        return {slug: [] for slug in slugs}

    async def get_transaction_count(
        self,
        slugs: list[str],
        days: int = 2,
    ) -> dict[str, list[SantimentDataPoint]]:
        """Return empty data — transaction_volume has 30-day lag on Santiment Free tier.

        The free tier restricts transaction_volume to data older than 30 days:
        "Both `from` and `to` parameters are outside the allowed interval"

        This method is kept for backward compatibility but always returns empty lists.
        """
        logger.debug(
            "santiment_transaction_volume_skipped",
            reason="30-day lag on free tier, recent data unavailable",
            slugs=len(slugs),
        )
        return {slug: [] for slug in slugs}

    async def get_all_metrics(
        self,
        slugs: list[str],
        days: int = 2,
    ) -> dict[str, dict[str, list[SantimentDataPoint]]]:
        """Fetch all free-tier batch metrics for multiple coins (2 API calls).

        Note: dev_activity is not supported via timeseriesDataPerSlug on free tier,
        so it returns empty lists. The key is still present for backward compatibility.

        Returns:
            Dict of {slug: {metric: [DataPoint, ...]}}.
        """
        results = await asyncio.gather(
            self.get_daily_active_addresses(slugs, days),
            self.get_dev_activity(slugs, days),
            self.get_transaction_count(slugs, days),
            return_exceptions=True,
        )

        empty: dict[str, list[SantimentDataPoint]] = {slug: [] for slug in slugs}
        daa = results[0] if not isinstance(results[0], BaseException) else empty
        dev = results[1] if not isinstance(results[1], BaseException) else empty
        tx = results[2] if not isinstance(results[2], BaseException) else empty

        for i, name in enumerate(["daily_active_addresses", "dev_activity", "transaction_volume"]):
            if isinstance(results[i], BaseException):
                logger.warning(
                    "santiment_metric_failed",
                    metric=name,
                    error=str(results[i]),
                )

        result: dict[str, dict[str, list[SantimentDataPoint]]] = {}
        for slug in slugs:
            result[slug] = {
                "daily_active_addresses": daa.get(slug, []),
                "dev_activity": dev.get(slug, []),
                "transaction_volume": tx.get(slug, []),
            }
        return result

    @staticmethod
    def slugs_for_symbols(symbols: list[str]) -> list[str]:
        """Convert ticker symbols to Santiment slugs."""
        return [SYMBOL_TO_SLUG[s] for s in symbols if s in SYMBOL_TO_SLUG]

    @staticmethod
    def slug_for_symbol(symbol: str) -> str | None:
        """Convert a single ticker symbol to Santiment slug."""
        return SYMBOL_TO_SLUG.get(symbol)
