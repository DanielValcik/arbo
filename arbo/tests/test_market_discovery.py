"""Tests for Market Discovery (PM-002).

Tests verify:
1. GammaMarket parsing from raw API response
2. Category classification (soccer, crypto, politics, etc.)
3. Price/spread calculations
4. MarketDiscovery filtering methods
5. MM candidate filtering
6. NegRisk event filtering
7. Crypto 15-min market filtering
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from arbo.connectors.market_discovery import (
    CATEGORY_KEYWORDS,
    GammaMarket,
    MarketDiscovery,
    categorize_market,
)

# ================================================================
# Sample data factory
# ================================================================


def _make_raw_market(
    condition_id: str = "0xabc123",
    question: str = "Will Bitcoin hit $100k?",
    outcomes: list[str] | None = None,
    outcome_prices: list[str] | None = None,
    clob_token_ids: list[str] | None = None,
    volume: str = "50000",
    volume_24h: str = "5000",
    liquidity: str = "10000",
    active: bool = True,
    closed: bool = False,
    fee_enabled: bool = False,
    neg_risk: bool = False,
    tags: list[dict[str, str]] | None = None,
) -> dict:
    """Build a raw market dict as returned by Gamma API."""
    return {
        "conditionId": condition_id,
        "question": question,
        "slug": question.lower().replace(" ", "-"),
        "outcomes": outcomes or ["Yes", "No"],
        "outcomePrices": outcome_prices or ["0.65", "0.35"],
        "clobTokenIds": clob_token_ids or ["token_yes_123", "token_no_456"],
        "volume": volume,
        "volume24hr": volume_24h,
        "liquidity": liquidity,
        "active": active,
        "closed": closed,
        "feesEnabled": fee_enabled,
        "enableNegRisk": neg_risk,
        "tags": tags or [],
        "endDate": "2026-12-31T00:00:00Z",
        "description": "Test market description",
    }


# ================================================================
# categorize_market tests
# ================================================================


class TestCategorizeMarket:
    """Category classification tests."""

    def test_soccer_by_keyword(self) -> None:
        assert categorize_market("Will Arsenal win the EPL?") == "soccer"

    def test_soccer_by_tag(self) -> None:
        assert categorize_market("Who will win?", tags=["Premier League"]) == "soccer"

    def test_crypto_by_keyword(self) -> None:
        assert categorize_market("Will Bitcoin hit $100k?") == "crypto"

    def test_politics_by_keyword(self) -> None:
        assert categorize_market("Will Trump win the election?") == "politics"

    def test_esports_by_keyword(self) -> None:
        assert categorize_market("CS2 major winner?") == "esports"

    def test_entertainment_by_keyword(self) -> None:
        assert categorize_market("Who wins the Oscar for best picture?") == "entertainment"

    def test_attention_markets_by_keyword(self) -> None:
        assert categorize_market("Kaito mindshare above 5%?") == "attention_markets"

    def test_other_for_unknown(self) -> None:
        assert categorize_market("Will it rain tomorrow?") == "other"

    def test_case_insensitive(self) -> None:
        assert categorize_market("BITCOIN price above $100K") == "crypto"

    def test_all_categories_have_keywords(self) -> None:
        """Every category should have at least one keyword."""
        for cat, keywords in CATEGORY_KEYWORDS.items():
            assert len(keywords) > 0, f"Category {cat} has no keywords"


# ================================================================
# GammaMarket parsing tests
# ================================================================


class TestGammaMarketParsing:
    """GammaMarket correctly parses raw API data."""

    def test_basic_parsing(self) -> None:
        raw = _make_raw_market()
        market = GammaMarket(raw)
        assert market.condition_id == "0xabc123"
        assert market.question == "Will Bitcoin hit $100k?"
        assert market.active is True
        assert market.closed is False

    def test_price_yes(self) -> None:
        raw = _make_raw_market(outcome_prices=["0.72", "0.28"])
        market = GammaMarket(raw)
        assert market.price_yes == Decimal("0.72")

    def test_price_no(self) -> None:
        raw = _make_raw_market(outcome_prices=["0.72", "0.28"])
        market = GammaMarket(raw)
        assert market.price_no == Decimal("0.28")

    def test_price_yes_missing(self) -> None:
        raw = _make_raw_market()
        raw["outcomePrices"] = []
        market = GammaMarket(raw)
        assert market.price_yes is None

    def test_price_no_missing_single_outcome(self) -> None:
        raw = _make_raw_market(outcome_prices=["0.50"])
        market = GammaMarket(raw)
        assert market.price_no is None

    def test_token_ids(self) -> None:
        raw = _make_raw_market(clob_token_ids=["tok_yes", "tok_no"])
        market = GammaMarket(raw)
        assert market.token_id_yes == "tok_yes"
        assert market.token_id_no == "tok_no"

    def test_token_ids_empty(self) -> None:
        raw = _make_raw_market()
        raw["clobTokenIds"] = []
        market = GammaMarket(raw)
        assert market.token_id_yes is None
        assert market.token_id_no is None

    def test_spread_calculation(self) -> None:
        """Spread should be |1 - yes - no|."""
        raw = _make_raw_market(outcome_prices=["0.65", "0.35"])
        market = GammaMarket(raw)
        assert market.spread == Decimal("0")  # 1 - 0.65 - 0.35 = 0

    def test_spread_with_vig(self) -> None:
        """Non-zero spread when prices don't sum to 1."""
        raw = _make_raw_market(outcome_prices=["0.55", "0.47"])
        market = GammaMarket(raw)
        assert market.spread == Decimal("0.02")  # |1 - 0.55 - 0.47| = 0.02

    def test_volume_decimal(self) -> None:
        raw = _make_raw_market(volume="123456.78", volume_24h="9876.54")
        market = GammaMarket(raw)
        assert market.volume == Decimal("123456.78")
        assert market.volume_24h == Decimal("9876.54")

    def test_volume_null_defaults_zero(self) -> None:
        raw = _make_raw_market()
        raw["volume"] = None
        raw["volume24hr"] = None
        market = GammaMarket(raw)
        assert market.volume == Decimal("0")
        assert market.volume_24h == Decimal("0")

    def test_fee_enabled(self) -> None:
        raw = _make_raw_market(fee_enabled=True)
        market = GammaMarket(raw)
        assert market.fee_enabled is True

    def test_neg_risk(self) -> None:
        raw = _make_raw_market(neg_risk=True)
        market = GammaMarket(raw)
        assert market.neg_risk is True

    def test_category_auto_assigned(self) -> None:
        raw = _make_raw_market(question="Will Bitcoin hit $100k?")
        market = GammaMarket(raw)
        assert market.category == "crypto"

    def test_tags_dict_format(self) -> None:
        """Tags can be dicts with 'label' key from Gamma API."""
        raw = _make_raw_market(
            question="Who wins?",
            tags=[{"label": "Premier League"}, {"label": "Sports"}],
        )
        market = GammaMarket(raw)
        assert market.category == "soccer"

    def test_tags_string_format(self) -> None:
        """Tags can also be plain strings."""
        raw = _make_raw_market(question="Who wins?", tags=["Premier League", "Sports"])
        # tags param in raw is already strings, but our factory wraps them
        raw["tags"] = ["Premier League", "Sports"]
        market = GammaMarket(raw)
        assert market.category == "soccer"


class TestGammaMarketDbDict:
    """to_db_dict serialization."""

    def test_contains_all_fields(self) -> None:
        raw = _make_raw_market()
        market = GammaMarket(raw)
        db = market.to_db_dict()
        required_keys = {
            "condition_id",
            "question",
            "slug",
            "category",
            "outcomes",
            "clob_token_ids",
            "fee_enabled",
            "neg_risk",
            "volume_24h",
            "liquidity",
            "end_date",
            "active",
            "last_price_yes",
            "last_price_no",
        }
        assert required_keys.issubset(db.keys())

    def test_active_and_closed_logic(self) -> None:
        """active in db dict should be True only if active=True and closed=False."""
        raw = _make_raw_market(active=True, closed=True)
        market = GammaMarket(raw)
        db = market.to_db_dict()
        assert db["active"] is False


# ================================================================
# MarketDiscovery filtering tests
# ================================================================


class TestMarketDiscoveryFiltering:
    """Tests for MarketDiscovery filter methods (in-memory, no API calls)."""

    @pytest.fixture
    def discovery_with_markets(self) -> MarketDiscovery:
        """Create a MarketDiscovery with pre-loaded test markets."""
        disc = MarketDiscovery.__new__(MarketDiscovery)
        disc._markets = {}
        disc._last_refresh = 0
        disc._refresh_interval = 900

        # Add test markets
        markets = [
            _make_raw_market(
                condition_id="c1",
                question="Will Bitcoin hit $100k?",
                volume_24h="5000",
                liquidity="10000",
                fee_enabled=True,
            ),
            _make_raw_market(
                condition_id="c2",
                question="Will Arsenal win the EPL?",
                volume_24h="2000",
                liquidity="5000",
            ),
            _make_raw_market(
                condition_id="c3",
                question="15 minute Bitcoin above $95k?",
                volume_24h="500",
                liquidity="1000",
                fee_enabled=True,
            ),
            _make_raw_market(
                condition_id="c4",
                question="Will Trump win?",
                volume_24h="50000",
                liquidity="100000",
                neg_risk=True,
            ),
            _make_raw_market(
                condition_id="c5",
                question="Will it rain tomorrow?",
                volume_24h="100",
                liquidity="200",
            ),
        ]
        for raw in markets:
            m = GammaMarket(raw)
            disc._markets[m.condition_id] = m

        return disc

    def test_get_all(self, discovery_with_markets: MarketDiscovery) -> None:
        assert len(discovery_with_markets.get_all()) == 5

    def test_get_by_category(self, discovery_with_markets: MarketDiscovery) -> None:
        crypto = discovery_with_markets.get_by_category("crypto")
        assert len(crypto) == 2  # Bitcoin $100k + 15min Bitcoin
        assert all(m.category == "crypto" for m in crypto)

    def test_get_by_condition_id(self, discovery_with_markets: MarketDiscovery) -> None:
        market = discovery_with_markets.get_by_condition_id("c1")
        assert market is not None
        assert market.condition_id == "c1"

    def test_get_by_condition_id_missing(self, discovery_with_markets: MarketDiscovery) -> None:
        assert discovery_with_markets.get_by_condition_id("nonexistent") is None

    def test_filter_by_min_volume(self, discovery_with_markets: MarketDiscovery) -> None:
        result = discovery_with_markets.filter_markets(min_volume_24h=Decimal("3000"))
        assert len(result) == 2  # Bitcoin $100k (5000) + Trump (50000)

    def test_filter_by_min_liquidity(self, discovery_with_markets: MarketDiscovery) -> None:
        result = discovery_with_markets.filter_markets(min_liquidity=Decimal("8000"))
        assert len(result) == 2  # Bitcoin (10000) + Trump (100000)

    def test_filter_by_category(self, discovery_with_markets: MarketDiscovery) -> None:
        result = discovery_with_markets.filter_markets(category="politics")
        assert len(result) == 1
        assert result[0].condition_id == "c4"

    def test_filter_by_fee_enabled(self, discovery_with_markets: MarketDiscovery) -> None:
        result = discovery_with_markets.filter_markets(fee_enabled=True)
        assert len(result) == 2  # c1 + c3 (both crypto with fees)

    def test_filter_by_neg_risk(self, discovery_with_markets: MarketDiscovery) -> None:
        result = discovery_with_markets.filter_markets(neg_risk=True)
        assert len(result) == 1
        assert result[0].condition_id == "c4"

    def test_filter_combined(self, discovery_with_markets: MarketDiscovery) -> None:
        result = discovery_with_markets.filter_markets(
            min_volume_24h=Decimal("1000"),
            category="crypto",
        )
        assert len(result) == 1
        assert result[0].condition_id == "c1"

    def test_get_mm_candidates(self, discovery_with_markets: MarketDiscovery) -> None:
        """MM candidates: volume >= $1000/day."""
        result = discovery_with_markets.get_mm_candidates()
        assert len(result) == 3  # c1 (5000), c2 (2000), c4 (50000)

    def test_get_negrisk_events(self, discovery_with_markets: MarketDiscovery) -> None:
        result = discovery_with_markets.get_negrisk_events()
        assert len(result) == 1
        assert result[0].neg_risk is True

    def test_get_crypto_15min_markets(self, discovery_with_markets: MarketDiscovery) -> None:
        result = discovery_with_markets.get_crypto_15min_markets()
        assert len(result) == 1
        assert "15" in result[0].question.lower()
