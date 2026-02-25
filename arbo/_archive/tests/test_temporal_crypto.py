"""Tests for PM-106: Temporal Crypto Scanner (Layer 6).

Tests verify:
1. BinanceWSFeed: parse ticker, stale returns None, multi-symbol, connected
2. Symbol extraction: bitcoin, btc, eth, unknown
3. Strike price extraction: comma, k-notation, decimal
4. TemporalCryptoScanner: divergence found, below threshold, no markets, rate limit, signal format

Acceptance: identifies >=3 active 15min crypto markets. Spot feed stable 30min.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from arbo.connectors.market_discovery import GammaMarket
from arbo.core.scanner import Signal, SignalDirection
from arbo.strategies.temporal_crypto import (
    BinanceWSFeed,
    SpotPrice,
    TemporalCryptoScanner,
    extract_strike_price,
    extract_symbol,
)

# ================================================================
# Factory helpers
# ================================================================


def _make_crypto_market(
    condition_id: str = "crypto_1",
    question: str = "Bitcoin 15 min price >= $95,000?",
    price_yes: str = "0.50",
    price_no: str = "0.50",
    volume_24h: str = "5000",
) -> GammaMarket:
    """Build a 15-min crypto GammaMarket for testing."""
    raw = {
        "conditionId": condition_id,
        "question": question,
        "slug": "test-crypto",
        "outcomes": '["Yes", "No"]',
        "outcomePrices": f'["{price_yes}", "{price_no}"]',
        "clobTokenIds": '["tok_yes_1", "tok_no_1"]',
        "volume": "100000",
        "volume24hr": volume_24h,
        "liquidity": "50000",
        "active": True,
        "closed": False,
        "feesEnabled": True,
        "enableNegRisk": False,
        "tags": [{"label": "crypto"}],
    }
    return GammaMarket(raw)


# ================================================================
# BinanceWSFeed
# ================================================================


class TestBinanceWSFeed:
    """WebSocket feed message parsing and staleness."""

    def test_parse_ticker_message(self) -> None:
        """Combined stream ticker message is parsed correctly."""
        feed = BinanceWSFeed()
        msg = '{"stream": "btcusdt@ticker", "data": {"s": "BTCUSDT", "c": "95123.45"}}'
        feed._handle_message(msg)

        price = feed.get_price("btcusdt")
        assert price is not None
        assert price.symbol == "btcusdt"
        assert price.price == Decimal("95123.45")
        assert price.source == "binance"

    def test_stale_price_returns_none(self) -> None:
        """Price older than threshold returns None."""
        feed = BinanceWSFeed()
        feed._prices["btcusdt"] = SpotPrice(
            symbol="btcusdt",
            price=Decimal("95000"),
            source="binance",
            received_at=datetime.now(UTC) - timedelta(seconds=120),  # 2 min old
        )

        assert feed.get_price("btcusdt") is None

    def test_multi_symbol_tracking(self) -> None:
        """Multiple symbols tracked independently."""
        feed = BinanceWSFeed()
        feed._handle_message('{"data": {"s": "BTCUSDT", "c": "95000"}}')
        feed._handle_message('{"data": {"s": "ETHUSDT", "c": "3200"}}')

        btc = feed.get_price("btcusdt")
        eth = feed.get_price("ethusdt")

        assert btc is not None
        assert btc.price == Decimal("95000")
        assert eth is not None
        assert eth.price == Decimal("3200")

    def test_unknown_symbol_returns_none(self) -> None:
        """Querying unknown symbol returns None."""
        feed = BinanceWSFeed()
        assert feed.get_price("xyzusdt") is None


# ================================================================
# Symbol extraction
# ================================================================


class TestSymbolExtraction:
    """Extract trading symbol from market question."""

    def test_bitcoin_keyword(self) -> None:
        assert extract_symbol("Bitcoin 15 min price >= $95,000?") == "btcusdt"

    def test_btc_abbreviation(self) -> None:
        assert extract_symbol("BTC 15 minute above $95k") == "btcusdt"

    def test_ethereum(self) -> None:
        assert extract_symbol("Ethereum 15 min price >= $3,200?") == "ethusdt"

    def test_unknown_returns_none(self) -> None:
        assert extract_symbol("Apple stock 15 min price?") is None


# ================================================================
# Strike price extraction
# ================================================================


class TestStrikePriceExtraction:
    """Extract strike price from market question."""

    def test_comma_format(self) -> None:
        result = extract_strike_price("Bitcoin 15 min price >= $95,000?")
        assert result == Decimal("95000")

    def test_k_notation(self) -> None:
        result = extract_strike_price("BTC above $95k?")
        assert result == Decimal("95000")

    def test_decimal_k_notation(self) -> None:
        result = extract_strike_price("ETH >= $3.2k?")
        assert result == Decimal("3200")

    def test_plain_number(self) -> None:
        result = extract_strike_price("BTC above $95000?")
        assert result == Decimal("95000")

    def test_no_price_returns_none(self) -> None:
        assert extract_strike_price("Will Bitcoin go up?") is None


# ================================================================
# TemporalCryptoScanner
# ================================================================


class TestTemporalCryptoScanner:
    """Temporal crypto scanner with mocked spot feed."""

    @pytest.fixture
    def mock_spot_feed(self) -> BinanceWSFeed:
        feed = BinanceWSFeed()
        feed._prices["btcusdt"] = SpotPrice(
            symbol="btcusdt",
            price=Decimal("96000"),  # Above $95,000 strike
            source="binance",
        )
        feed._prices["ethusdt"] = SpotPrice(
            symbol="ethusdt",
            price=Decimal("3200"),
            source="binance",
        )
        return feed

    @pytest.fixture
    def mock_discovery(self) -> MagicMock:
        discovery = MagicMock()
        discovery.get_crypto_15min_markets.return_value = [
            _make_crypto_market(
                "c1",
                "Bitcoin 15 min price >= $95,000?",
                price_yes="0.40",  # Contract at 0.40 but spot implies higher
            ),
        ]
        return discovery

    @patch("arbo.strategies.temporal_crypto.get_config")
    async def test_divergence_found(
        self, mock_config: MagicMock, mock_discovery: MagicMock, mock_spot_feed: BinanceWSFeed
    ) -> None:
        """Spot above strike with low contract price → signal."""
        mock_config.return_value = _mock_config()
        scanner = TemporalCryptoScanner(discovery=mock_discovery, spot_feed=mock_spot_feed)

        signals = await scanner.scan()
        assert len(signals) >= 1
        assert signals[0].layer == 6
        assert signals[0].direction == SignalDirection.BUY_YES
        assert signals[0].edge > Decimal("0")

    @patch("arbo.strategies.temporal_crypto.get_config")
    async def test_below_threshold_no_signal(
        self, mock_config: MagicMock, mock_spot_feed: BinanceWSFeed
    ) -> None:
        """Spot very close to strike → no signal."""
        mock_config.return_value = _mock_config()
        discovery = MagicMock()
        discovery.get_crypto_15min_markets.return_value = [
            _make_crypto_market(
                "c1",
                "Bitcoin 15 min price >= $95,000?",
                price_yes="0.50",  # Contract at fair value
            ),
        ]
        # Spot very close to strike
        mock_spot_feed._prices["btcusdt"] = SpotPrice(
            symbol="btcusdt",
            price=Decimal("95050"),  # Only 0.05% above
            source="binance",
        )
        scanner = TemporalCryptoScanner(discovery=discovery, spot_feed=mock_spot_feed)

        signals = await scanner.scan()
        assert len(signals) == 0

    @patch("arbo.strategies.temporal_crypto.get_config")
    async def test_no_crypto_markets(self, mock_config: MagicMock) -> None:
        """Empty discovery → no signals."""
        mock_config.return_value = _mock_config()
        discovery = MagicMock()
        discovery.get_crypto_15min_markets.return_value = []

        scanner = TemporalCryptoScanner(discovery=discovery)
        signals = await scanner.scan()
        assert signals == []

    @patch("arbo.strategies.temporal_crypto.get_config")
    async def test_rate_limit(self, mock_config: MagicMock, mock_spot_feed: BinanceWSFeed) -> None:
        """Max trades per hour limits signals."""
        config = _mock_config()
        config.temporal_crypto.max_trades_per_hour = 1
        mock_config.return_value = config

        discovery = MagicMock()
        discovery.get_crypto_15min_markets.return_value = [
            _make_crypto_market("c1", "Bitcoin 15 min price >= $95,000?", price_yes="0.30"),
            _make_crypto_market("c2", "Bitcoin 15 min price >= $94,000?", price_yes="0.30"),
        ]

        scanner = TemporalCryptoScanner(discovery=discovery, spot_feed=mock_spot_feed)
        signals = await scanner.scan()

        assert len(signals) <= 1  # Rate limited to 1

    @patch("arbo.strategies.temporal_crypto.get_config")
    async def test_signal_format(
        self, mock_config: MagicMock, mock_discovery: MagicMock, mock_spot_feed: BinanceWSFeed
    ) -> None:
        """Signal contains correct details."""
        mock_config.return_value = _mock_config()
        scanner = TemporalCryptoScanner(discovery=mock_discovery, spot_feed=mock_spot_feed)

        signals = await scanner.scan()
        assert len(signals) >= 1

        sig = signals[0]
        assert isinstance(sig, Signal)
        assert sig.layer == 6
        assert "symbol" in sig.details
        assert "spot_price" in sig.details
        assert "strike_price" in sig.details
        assert "divergence" in sig.details
        assert sig.details["use_postonly"] is True


# ================================================================
# Helpers
# ================================================================


def _mock_config() -> MagicMock:
    """Create a mock ArboConfig for temporal crypto tests."""
    config = MagicMock()
    config.temporal_crypto.price_deviation_threshold = 0.005
    config.temporal_crypto.max_trades_per_hour = 20
    config.temporal_crypto.use_postonly = True
    return config
