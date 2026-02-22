"""Tests for market discovery extension (PM-403).

Tests crypto market parsing, politics filtering, and new discovery methods.
"""

from __future__ import annotations

from decimal import Decimal

from arbo.connectors.market_discovery import (
    GammaMarket,
    MarketDiscovery,
    categorize_crypto_market,
)

# ---------------------------------------------------------------------------
# categorize_crypto_market — strike price parsing
# ---------------------------------------------------------------------------


class TestParseBTCStrike:
    def test_btc_with_comma_price(self) -> None:
        """Parse BTC strike from '$100,000' format."""
        info = categorize_crypto_market("Will BTC be above $100,000 on March 1?")
        assert info is not None
        assert info.asset == "BTC"
        assert info.symbol == "BTCUSDT"
        assert info.strike == Decimal("100000")
        assert info.direction == "above"

    def test_btc_with_k_notation(self) -> None:
        """Parse BTC strike from '$95k' format."""
        info = categorize_crypto_market("Will Bitcoin be above $95k by March 15?")
        assert info is not None
        assert info.asset == "BTC"
        assert info.strike == Decimal("95000")

    def test_btc_below_direction(self) -> None:
        """Detect 'below' direction."""
        info = categorize_crypto_market("Will BTC be below $90,000 on February 28?")
        assert info is not None
        assert info.direction == "below"


class TestParseETHStrike:
    def test_eth_strike(self) -> None:
        """Parse ETH strike from '$4,000' format."""
        info = categorize_crypto_market("Will ETH be above $4,000 by February 28?")
        assert info is not None
        assert info.asset == "ETH"
        assert info.symbol == "ETHUSDT"
        assert info.strike == Decimal("4000")


class TestParseSOLStrike:
    def test_sol_strike(self) -> None:
        """Parse SOL strike from '$200' format."""
        info = categorize_crypto_market("Will Solana be above $200 on March 15?")
        assert info is not None
        assert info.asset == "SOL"
        assert info.symbol == "SOLUSDT"
        assert info.strike == Decimal("200")


class TestParseDateFormats:
    def test_march_1(self) -> None:
        """Parse 'March 1' date format."""
        info = categorize_crypto_market("Will BTC be above $100,000 on March 1?")
        assert info is not None
        assert info.expiry is not None
        assert info.expiry.month == 3
        assert info.expiry.day == 1

    def test_february_28(self) -> None:
        """Parse 'February 28' date format."""
        info = categorize_crypto_market("Will ETH be above $4,000 by February 28?")
        assert info is not None
        assert info.expiry is not None
        assert info.expiry.month == 2
        assert info.expiry.day == 28

    def test_date_with_year(self) -> None:
        """Parse date with explicit year."""
        info = categorize_crypto_market("Will BTC be above $100,000 on March 1, 2026?")
        assert info is not None
        assert info.expiry is not None
        assert info.expiry.year == 2026
        assert info.expiry.month == 3


class TestFiveMinMarket:
    def test_5min_market_detection(self) -> None:
        """Detect '5 min' / 'Up or Down' markets."""
        info = categorize_crypto_market("Bitcoin Up or Down - February 23, 1:25PM")
        assert info is not None
        assert info.is_5min is True
        assert info.asset == "BTC"


class TestNonCrypto:
    def test_non_crypto_returns_none(self) -> None:
        """Non-crypto question returns None."""
        assert categorize_crypto_market("Will Trump win the 2028 election?") is None

    def test_no_strike_returns_none(self) -> None:
        """Crypto question without strike price returns None."""
        assert categorize_crypto_market("Will Bitcoin exist in 2030?") is None


# ---------------------------------------------------------------------------
# MarketDiscovery.get_crypto_markets / get_politics_markets
# ---------------------------------------------------------------------------


def _make_market(
    condition_id: str,
    question: str,
    category: str,
    price_yes: str = "0.50",
    active: bool = True,
    closed: bool = False,
    outcomes: int = 2,
) -> GammaMarket:
    """Create a GammaMarket from minimal data."""
    raw = {
        "conditionId": condition_id,
        "question": question,
        "slug": "test",
        "outcomePrices": f'["{price_yes}", "0.50"]' if outcomes == 2 else f'["{price_yes}"]',
        "outcomes": '["Yes", "No"]' if outcomes == 2 else '["Yes"]',
        "clobTokenIds": '["tok1", "tok2"]',
        "active": active,
        "closed": closed,
    }
    m = GammaMarket(raw)
    m.category = category
    return m


class TestGetCryptoMarkets:
    def test_filters_crypto_only(self) -> None:
        """Only returns crypto category markets."""
        d = MarketDiscovery.__new__(MarketDiscovery)
        d._markets = {
            "c1": _make_market("c1", "BTC above $100k", "crypto"),
            "p1": _make_market("p1", "Trump wins", "politics"),
        }
        result = d.get_crypto_markets()
        assert len(result) == 1
        assert result[0].condition_id == "c1"

    def test_excludes_longshots(self) -> None:
        """Excludes markets with price < 0.05 or > 0.95."""
        d = MarketDiscovery.__new__(MarketDiscovery)
        d._markets = {
            "c1": _make_market("c1", "BTC moon", "crypto", price_yes="0.02"),
            "c2": _make_market("c2", "BTC above $1", "crypto", price_yes="0.98"),
            "c3": _make_market("c3", "BTC above $100k", "crypto", price_yes="0.50"),
        }
        result = d.get_crypto_markets()
        assert len(result) == 1
        assert result[0].condition_id == "c3"

    def test_excludes_closed(self) -> None:
        """Excludes closed markets."""
        d = MarketDiscovery.__new__(MarketDiscovery)
        d._markets = {
            "c1": _make_market("c1", "BTC above $100k", "crypto", closed=True),
        }
        result = d.get_crypto_markets()
        assert len(result) == 0


class TestGetPoliticsMarkets:
    def test_filters_politics_only(self) -> None:
        """Only returns politics category markets."""
        d = MarketDiscovery.__new__(MarketDiscovery)
        d._markets = {
            "c1": _make_market("c1", "BTC above $100k", "crypto"),
            "p1": _make_market("p1", "Trump wins", "politics"),
        }
        result = d.get_politics_markets()
        assert len(result) == 1
        assert result[0].condition_id == "p1"

    def test_price_range_filter(self) -> None:
        """Only markets in 5-95 cent range."""
        d = MarketDiscovery.__new__(MarketDiscovery)
        d._markets = {
            "p1": _make_market("p1", "Election 1", "politics", price_yes="0.50"),
            "p2": _make_market("p2", "Election 2", "politics", price_yes="0.03"),
        }
        result = d.get_politics_markets()
        assert len(result) == 1
        assert result[0].condition_id == "p1"
