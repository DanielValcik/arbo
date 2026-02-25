"""Tests for Binance data connector (PM-402)."""

from __future__ import annotations

import asyncio
import contextlib
import re

import pytest
from aioresponses import aioresponses

from arbo.connectors.binance_client import (
    BinanceClient,
    FundingRate,
    OHLCVBar,
    Ticker24h,
    _RateThrottle,
    parse_funding_rate,
    parse_kline,
    parse_ticker_24h,
)

# ---------------------------------------------------------------------------
# Sample API responses
# ---------------------------------------------------------------------------

SAMPLE_KLINE = [
    1708963200000,  # open time
    "95000.00",  # open
    "96000.00",  # high
    "94000.00",  # low
    "95500.00",  # close
    "1234.56",  # volume
    1708966799999,  # close time
    "117432345.67",  # quote volume
    15234,  # trades
    "678.90",  # taker buy vol
    "64567890.12",  # taker buy quote vol
    "0",  # ignore
]

SAMPLE_FUNDING = {
    "symbol": "BTCUSDT",
    "fundingTime": 1708963200000,
    "fundingRate": "0.00012345",
    "markPrice": "95100.50",
}

SAMPLE_TICKER_24H = {
    "symbol": "BTCUSDT",
    "lastPrice": "95500.00",
    "priceChangePercent": "1.23",
    "volume": "45678.90",
    "quoteVolume": "4361234567.89",
}


# ---------------------------------------------------------------------------
# Unit tests: parsing
# ---------------------------------------------------------------------------


class TestParseKline:
    def test_parse_kline_response(self) -> None:
        """Correct OHLCV fields from raw Binance kline array."""
        bar = parse_kline(SAMPLE_KLINE)
        assert isinstance(bar, OHLCVBar)
        assert bar.open_time == 1708963200000
        assert bar.open == 95000.00
        assert bar.high == 96000.00
        assert bar.low == 94000.00
        assert bar.close == 95500.00
        assert bar.volume == 1234.56
        assert bar.quote_volume == 117432345.67
        assert bar.trades == 15234
        assert bar.close_time == 1708966799999


class TestParseFundingRate:
    def test_parse_funding_rate(self) -> None:
        """Correct fields from Binance funding rate JSON."""
        fr = parse_funding_rate(SAMPLE_FUNDING)
        assert isinstance(fr, FundingRate)
        assert fr.symbol == "BTCUSDT"
        assert fr.funding_time == 1708963200000
        assert abs(fr.rate - 0.00012345) < 1e-10
        assert abs(fr.mark_price - 95100.50) < 0.01


class TestParseTicker24h:
    def test_parse_ticker(self) -> None:
        """Correct fields from Binance 24h ticker JSON."""
        t = parse_ticker_24h(SAMPLE_TICKER_24H)
        assert isinstance(t, Ticker24h)
        assert t.symbol == "BTCUSDT"
        assert t.last_price == 95500.00
        assert abs(t.price_change_pct - 1.23) < 0.01
        assert t.volume == 45678.90


# ---------------------------------------------------------------------------
# Computed features
# ---------------------------------------------------------------------------


class TestComputeVolatility:
    def test_compute_volatility_known_values(self) -> None:
        """Known std dev from known prices."""
        bars = [
            OHLCVBar(0, 100, 100, 100, 100, 0, 0, 0, 0),
            OHLCVBar(1, 100, 100, 100, 110, 0, 0, 0, 0),  # +10%
            OHLCVBar(2, 100, 100, 100, 105, 0, 0, 0, 0),  # -4.5%
            OHLCVBar(3, 100, 100, 100, 115, 0, 0, 0, 0),  # +9.5%
        ]
        vol = BinanceClient.compute_volatility(bars)
        assert vol > 0
        # Should be reasonable annualized vol (not zero, not astronomical)
        assert 0.1 < vol < 50.0

    def test_compute_volatility_insufficient_data(self) -> None:
        """Returns 0 with fewer than 2 bars."""
        assert BinanceClient.compute_volatility([]) == 0.0
        bar = OHLCVBar(0, 100, 100, 100, 100, 0, 0, 0, 0)
        assert BinanceClient.compute_volatility([bar]) == 0.0

    def test_compute_volatility_flat(self) -> None:
        """Flat prices -> zero volatility."""
        bars = [OHLCVBar(i, 100, 100, 100, 100, 0, 0, 0, 0) for i in range(10)]
        assert BinanceClient.compute_volatility(bars) == 0.0


class TestComputeRSI:
    def test_rsi_flat(self) -> None:
        """RSI=50 for flat prices."""
        bars = [OHLCVBar(i, 100, 100, 100, 100, 0, 0, 0, 0) for i in range(20)]
        assert BinanceClient.compute_rsi(bars) == 50.0

    def test_rsi_uptrend(self) -> None:
        """RSI > 70 for consistent uptrend."""
        bars = [OHLCVBar(i, 100 + i, 100 + i, 100 + i, 100 + i, 0, 0, 0, 0) for i in range(20)]
        rsi = BinanceClient.compute_rsi(bars)
        assert rsi > 70

    def test_rsi_downtrend(self) -> None:
        """RSI < 30 for consistent downtrend."""
        bars = [OHLCVBar(i, 200 - i, 200 - i, 200 - i, 200 - i, 0, 0, 0, 0) for i in range(20)]
        rsi = BinanceClient.compute_rsi(bars)
        assert rsi < 30

    def test_rsi_insufficient_data(self) -> None:
        """Returns 50 with insufficient data."""
        bars = [OHLCVBar(i, 100, 100, 100, 100, 0, 0, 0, 0) for i in range(5)]
        assert BinanceClient.compute_rsi(bars) == 50.0


class TestComputeMomentum:
    def test_momentum_positive(self) -> None:
        """Correct positive % change."""
        bars = [
            OHLCVBar(0, 100, 100, 100, 100, 0, 0, 0, 0),
            OHLCVBar(1, 100, 100, 100, 110, 0, 0, 0, 0),
        ]
        assert abs(BinanceClient.compute_momentum(bars) - 0.10) < 0.001

    def test_momentum_negative(self) -> None:
        """Correct negative % change."""
        bars = [
            OHLCVBar(0, 100, 100, 100, 100, 0, 0, 0, 0),
            OHLCVBar(1, 100, 100, 100, 90, 0, 0, 0, 0),
        ]
        assert abs(BinanceClient.compute_momentum(bars) - (-0.10)) < 0.001

    def test_momentum_insufficient_data(self) -> None:
        """Returns 0 with fewer than 2 bars."""
        assert BinanceClient.compute_momentum([]) == 0.0


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateThrottle:
    @pytest.mark.asyncio
    async def test_rate_limiting_blocks_when_exhausted(self) -> None:
        """Throttle blocks when weight budget is exhausted."""
        throttle = _RateThrottle(max_weight_per_min=10)

        # Fill the budget
        await throttle.acquire(10)
        assert throttle.current_weight() == 10

        # Next acquire should block — we'll test with a timeout
        acquired = False

        async def try_acquire() -> None:
            nonlocal acquired
            await throttle.acquire(1)
            acquired = True

        task = asyncio.create_task(try_acquire())
        await asyncio.sleep(0.2)
        assert not acquired  # Should be blocked
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


# ---------------------------------------------------------------------------
# Integration-style tests with mocked HTTP
# ---------------------------------------------------------------------------


class TestBinanceClientKlines:
    @pytest.mark.asyncio
    async def test_get_klines(self) -> None:
        """Fetch klines returns parsed OHLCVBar list."""
        client = BinanceClient()
        await client.initialize()
        try:
            with aioresponses() as m:
                m.get(
                    re.compile(r"https://api\.binance\.com/api/v3/klines.*"),
                    payload=[SAMPLE_KLINE, SAMPLE_KLINE],
                )
                bars = await client.get_klines("BTCUSDT", "1h", limit=2)
                assert len(bars) == 2
                assert bars[0].close == 95500.00
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_klines_range_pagination(self) -> None:
        """Multi-page klines assembly across time range."""
        client = BinanceClient()
        await client.initialize()
        try:
            # Page 1: two bars
            page1 = [
                [1000, "100", "100", "100", "100", "1", 1999, "100", 1, "1", "1", "0"],
                [2000, "101", "101", "101", "101", "1", 2999, "101", 1, "1", "1", "0"],
            ]
            # Page 2: one bar (end of range)
            page2 = [
                [3000, "102", "102", "102", "102", "1", 3999, "102", 1, "1", "1", "0"],
            ]

            with aioresponses() as m:
                # First call returns page1
                m.get(re.compile(r"https://api\.binance\.com/api/v3/klines.*"), payload=page1)
                # Second call returns page2
                m.get(re.compile(r"https://api\.binance\.com/api/v3/klines.*"), payload=page2)
                # Third call returns empty (stop)
                m.get(re.compile(r"https://api\.binance\.com/api/v3/klines.*"), payload=[])

                bars = await client.get_klines_range("BTCUSDT", "1h", 1000, 4000)
                assert len(bars) == 3
                assert bars[0].open_time == 1000
                assert bars[2].open_time == 3000
        finally:
            await client.close()


class TestBinanceClientTicker:
    @pytest.mark.asyncio
    async def test_get_ticker_24h(self) -> None:
        """Fetch 24h ticker returns parsed Ticker24h."""
        client = BinanceClient()
        await client.initialize()
        try:
            with aioresponses() as m:
                m.get(
                    re.compile(r"https://api\.binance\.com/api/v3/ticker/24hr.*"),
                    payload=SAMPLE_TICKER_24H,
                )
                ticker = await client.get_ticker_24h("BTCUSDT")
                assert ticker.last_price == 95500.00
                assert ticker.symbol == "BTCUSDT"
        finally:
            await client.close()


class TestBinanceClientFunding:
    @pytest.mark.asyncio
    async def test_get_funding_rate(self) -> None:
        """Fetch funding rate returns parsed FundingRate list."""
        client = BinanceClient()
        await client.initialize()
        try:
            with aioresponses() as m:
                m.get(
                    re.compile(r"https://fapi\.binance\.com/fapi/v1/fundingRate.*"),
                    payload=[SAMPLE_FUNDING],
                )
                rates = await client.get_funding_rate("BTCUSDT")
                assert len(rates) == 1
                assert rates[0].symbol == "BTCUSDT"
        finally:
            await client.close()


class TestBinanceClientCache:
    @pytest.mark.asyncio
    async def test_cache_ttl(self) -> None:
        """Cached within TTL, fresh after expiry."""
        client = BinanceClient(cache_ttl=1)  # 1 second TTL
        await client.initialize()
        try:
            with aioresponses() as m:
                m.get(
                    re.compile(r"https://api\.binance\.com/api/v3/ticker/24hr.*"),
                    payload=SAMPLE_TICKER_24H,
                )
                # Second call should return different data
                modified = {**SAMPLE_TICKER_24H, "lastPrice": "99000.00"}
                m.get(
                    re.compile(r"https://api\.binance\.com/api/v3/ticker/24hr.*"),
                    payload=modified,
                )

                # First call — fetches from API
                t1 = await client.get_ticker_24h("BTCUSDT")
                assert t1.last_price == 95500.00

                # Second call immediately — should be cached
                t2 = await client.get_ticker_24h("BTCUSDT")
                assert t2.last_price == 95500.00  # Still cached

                # Wait for cache expiry
                await asyncio.sleep(1.1)

                # Third call — cache expired, fetches fresh
                t3 = await client.get_ticker_24h("BTCUSDT")
                assert t3.last_price == 99000.00
        finally:
            await client.close()
