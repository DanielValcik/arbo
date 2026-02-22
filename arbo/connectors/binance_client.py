"""Binance public API client for crypto market data (PM-402).

REST client for Binance spot + futures public endpoints.
Complements the WebSocket feed in temporal_crypto.py:BinanceWSFeed.

Endpoints:
- GET /api/v3/klines — OHLCV candles
- GET /api/v3/ticker/24hr — 24h statistics
- GET /fapi/v1/fundingRate — perpetual funding rates

Rate limiting: weight-aware sliding window (1200 weight/min limit,
conservative cap at 100 req/min).
"""

from __future__ import annotations

import ssl
import time
from dataclasses import dataclass
from typing import Any

import aiohttp
import certifi

from arbo.utils.logger import get_logger

logger = get_logger("binance_client")


@dataclass
class OHLCVBar:
    """Single OHLCV candle from Binance klines API."""

    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades: int
    close_time: int


@dataclass
class FundingRate:
    """Perpetual funding rate snapshot."""

    symbol: str
    funding_time: int
    rate: float
    mark_price: float


@dataclass
class Ticker24h:
    """24-hour ticker statistics."""

    symbol: str
    last_price: float
    price_change_pct: float
    volume: float
    quote_volume: float


class _RateThrottle:
    """Sliding-window weight-aware rate limiter for Binance API.

    Binance allows 1200 weight/min. We cap at a configurable max_weight_per_min.
    """

    def __init__(self, max_weight_per_min: int = 600) -> None:
        self._max_weight = max_weight_per_min
        self._window: list[tuple[float, int]] = []  # (timestamp, weight)

    def _prune(self, now: float) -> None:
        cutoff = now - 60.0
        self._window = [(t, w) for t, w in self._window if t > cutoff]

    def current_weight(self) -> int:
        """Total weight used in the current 1-minute window."""
        self._prune(time.monotonic())
        return sum(w for _, w in self._window)

    async def acquire(self, weight: int = 1) -> None:
        """Wait until enough weight budget is available, then record usage."""
        import asyncio

        while True:
            now = time.monotonic()
            self._prune(now)
            used = sum(w for _, w in self._window)
            if used + weight <= self._max_weight:
                self._window.append((now, weight))
                return
            await asyncio.sleep(0.5)


def parse_kline(raw: list[Any]) -> OHLCVBar:
    """Parse a single Binance kline array into an OHLCVBar.

    Binance kline format: [open_time, open, high, low, close, volume,
    close_time, quote_volume, trades, taker_buy_vol, taker_buy_quote_vol, ignore]
    """
    return OHLCVBar(
        open_time=int(raw[0]),
        open=float(raw[1]),
        high=float(raw[2]),
        low=float(raw[3]),
        close=float(raw[4]),
        volume=float(raw[5]),
        quote_volume=float(raw[7]),
        trades=int(raw[8]),
        close_time=int(raw[6]),
    )


def parse_funding_rate(raw: dict[str, Any]) -> FundingRate:
    """Parse a single Binance funding rate JSON object."""
    return FundingRate(
        symbol=raw["symbol"],
        funding_time=int(raw["fundingTime"]),
        rate=float(raw["fundingRate"]),
        mark_price=float(raw.get("markPrice", 0)),
    )


def parse_ticker_24h(raw: dict[str, Any]) -> Ticker24h:
    """Parse a single Binance 24hr ticker JSON object."""
    return Ticker24h(
        symbol=raw.get("symbol", ""),
        last_price=float(raw.get("lastPrice", 0)),
        price_change_pct=float(raw.get("priceChangePercent", 0)),
        volume=float(raw.get("volume", 0)),
        quote_volume=float(raw.get("quoteVolume", 0)),
    )


class BinanceClient:
    """Async REST client for Binance public API.

    Provides OHLCV history, 24h tickers, and funding rates
    with weight-aware rate limiting and response caching.
    """

    def __init__(
        self,
        base_url: str = "https://api.binance.com",
        futures_url: str = "https://fapi.binance.com",
        max_requests_per_min: int = 100,
        cache_ttl: int = 300,
    ) -> None:
        self._base_url = base_url
        self._futures_url = futures_url
        self._session: aiohttp.ClientSession | None = None
        self._throttle = _RateThrottle(max_weight_per_min=max_requests_per_min * 5)
        self._cache: dict[str, tuple[float, Any]] = {}
        self._cache_ttl = cache_ttl

    async def initialize(self) -> None:
        """Create aiohttp session with certifi SSL context."""
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"Accept": "application/json"},
            connector=aiohttp.TCPConnector(ssl=ssl_ctx),
        )
        logger.info("binance_client_initialized", base_url=self._base_url)

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_get(self, key: str) -> Any | None:
        """Get a cached value if within TTL."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.monotonic() - ts > self._cache_ttl:
            del self._cache[key]
            return None
        return value

    def _cache_set(self, key: str, value: Any) -> None:
        """Store a value in cache."""
        self._cache[key] = (time.monotonic(), value)

    # ------------------------------------------------------------------
    # Raw HTTP
    # ------------------------------------------------------------------

    async def _get(self, url: str, params: dict[str, str], weight: int = 5) -> Any:
        """Make a GET request with rate limiting.

        Args:
            url: Full URL.
            params: Query parameters.
            weight: API weight for this request.

        Returns:
            Parsed JSON response.

        Raises:
            RuntimeError: If client not initialized or HTTP error.
        """
        if not self._session:
            raise RuntimeError("BinanceClient not initialized. Call initialize() first.")

        await self._throttle.acquire(weight)

        async with self._session.get(url, params=params) as resp:
            if resp.status != 200:
                body = await resp.text()
                logger.error("binance_api_error", status=resp.status, url=url, body=body[:200])
                raise RuntimeError(f"Binance API error {resp.status}: {body[:200]}")
            return await resp.json()

    # ------------------------------------------------------------------
    # Historical data
    # ------------------------------------------------------------------

    async def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 500,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[OHLCVBar]:
        """Fetch OHLCV klines for a symbol.

        Args:
            symbol: Trading pair (e.g. "BTCUSDT").
            interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d, etc.).
            limit: Number of candles (max 1000).
            start_time: Start time in milliseconds.
            end_time: End time in milliseconds.

        Returns:
            List of OHLCVBar sorted by open_time ascending.
        """
        params: dict[str, str] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": str(min(limit, 1000)),
        }
        if start_time is not None:
            params["startTime"] = str(start_time)
        if end_time is not None:
            params["endTime"] = str(end_time)

        url = f"{self._base_url}/api/v3/klines"
        weight = 5 if limit <= 100 else 10
        raw = await self._get(url, params, weight=weight)
        return [parse_kline(k) for k in raw]

    async def get_klines_range(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[OHLCVBar]:
        """Fetch klines across a time range with automatic pagination.

        Args:
            symbol: Trading pair.
            interval: Candle interval.
            start_ms: Range start in milliseconds.
            end_ms: Range end in milliseconds.

        Returns:
            All bars in the range, deduplicated by open_time.
        """
        all_bars: list[OHLCVBar] = []
        seen_times: set[int] = set()
        current_start = start_ms

        while current_start < end_ms:
            bars = await self.get_klines(
                symbol=symbol,
                interval=interval,
                limit=1000,
                start_time=current_start,
                end_time=end_ms,
            )
            if not bars:
                break

            for bar in bars:
                if bar.open_time not in seen_times:
                    seen_times.add(bar.open_time)
                    all_bars.append(bar)

            # Move past the last bar's open time
            last_time = bars[-1].open_time
            if last_time <= current_start:
                break
            current_start = last_time + 1

        all_bars.sort(key=lambda b: b.open_time)
        return all_bars

    # ------------------------------------------------------------------
    # Live data (cached)
    # ------------------------------------------------------------------

    async def get_ticker_24h(self, symbol: str) -> Ticker24h:
        """Fetch 24h ticker statistics for a symbol.

        Args:
            symbol: Trading pair (e.g. "BTCUSDT").

        Returns:
            Ticker24h with price, volume, and change data.
        """
        cache_key = f"ticker:{symbol.upper()}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        url = f"{self._base_url}/api/v3/ticker/24hr"
        params = {"symbol": symbol.upper()}
        raw = await self._get(url, params, weight=5)
        result = parse_ticker_24h(raw)
        self._cache_set(cache_key, result)
        return result

    async def get_funding_rate(
        self,
        symbol: str,
        limit: int = 1,
    ) -> list[FundingRate]:
        """Fetch recent funding rates for a perpetual contract.

        Args:
            symbol: Futures pair (e.g. "BTCUSDT").
            limit: Number of recent rates (max 1000).

        Returns:
            List of FundingRate snapshots, most recent first.
        """
        cache_key = f"funding:{symbol.upper()}:{limit}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        url = f"{self._futures_url}/fapi/v1/fundingRate"
        params = {"symbol": symbol.upper(), "limit": str(min(limit, 1000))}
        raw = await self._get(url, params, weight=1)
        result = [parse_funding_rate(r) for r in raw]
        self._cache_set(cache_key, result)
        return result

    # ------------------------------------------------------------------
    # Computed features (static)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_volatility(bars: list[OHLCVBar], window: int | None = None) -> float:
        """Compute annualized volatility from log returns.

        Args:
            bars: OHLCV bars (at least 2 required).
            window: Use only the last N bars. None = use all.

        Returns:
            Annualized volatility as a fraction (e.g. 0.45 = 45%).
        """
        if len(bars) < 2:
            return 0.0

        import math

        use_bars = bars[-window:] if window else bars
        if len(use_bars) < 2:
            return 0.0

        log_returns = [
            math.log(use_bars[i].close / use_bars[i - 1].close)
            for i in range(1, len(use_bars))
            if use_bars[i - 1].close > 0
        ]
        if not log_returns:
            return 0.0

        mean = sum(log_returns) / len(log_returns)
        variance = sum((r - mean) ** 2 for r in log_returns) / len(log_returns)
        std = math.sqrt(variance)

        # Annualize: assume hourly bars -> 8760 hours/year
        return std * math.sqrt(8760)

    @staticmethod
    def compute_rsi(bars: list[OHLCVBar], period: int = 14) -> float:
        """Compute Relative Strength Index.

        Args:
            bars: OHLCV bars (need at least period+1).
            period: RSI lookback period.

        Returns:
            RSI value [0, 100]. Returns 50 if insufficient data.
        """
        if len(bars) < period + 1:
            return 50.0

        changes = [bars[i].close - bars[i - 1].close for i in range(1, len(bars))]
        recent = changes[-(period):]

        gains = [c for c in recent if c > 0]
        losses = [-c for c in recent if c < 0]

        avg_gain = sum(gains) / period if gains else 0.0
        avg_loss = sum(losses) / period if losses else 0.0

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def compute_momentum(bars: list[OHLCVBar]) -> float:
        """Compute price momentum as percentage change over the bar range.

        Args:
            bars: OHLCV bars (at least 2 required).

        Returns:
            Percentage change from first to last close (e.g. 0.05 = +5%).
        """
        if len(bars) < 2 or bars[0].close == 0:
            return 0.0
        return (bars[-1].close - bars[0].close) / bars[0].close
