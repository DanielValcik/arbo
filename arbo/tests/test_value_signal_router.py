"""Tests for value signal multi-model router (PM-404)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from arbo.connectors.market_discovery import GammaMarket
from arbo.strategies.value_signal import ValueSignalGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gamma_market(
    condition_id: str,
    question: str,
    category: str,
    price_yes: str = "0.50",
    end_date: str | None = "2026-03-01T00:00:00Z",
) -> GammaMarket:
    """Create a GammaMarket with minimal data."""
    raw = {
        "conditionId": condition_id,
        "question": question,
        "slug": "test",
        "outcomePrices": f'["{price_yes}", "0.50"]',
        "outcomes": '["Yes", "No"]',
        "clobTokenIds": '["tok_yes", "tok_no"]',
        "active": True,
        "closed": False,
        "volume24hr": "10000",
        "liquidity": "5000",
        "volume": "100000",
        "endDate": end_date,
    }
    m = GammaMarket(raw)
    m.category = category
    return m


def _make_discovery(
    soccer: list[GammaMarket] | None = None,
    crypto: list[GammaMarket] | None = None,
    politics: list[GammaMarket] | None = None,
) -> MagicMock:
    """Create a mock MarketDiscovery."""
    d = MagicMock()
    d.get_by_category.return_value = soccer or []
    d.get_crypto_markets.return_value = crypto or []
    d.get_politics_markets.return_value = politics or []
    return d


def _make_odds_client() -> MagicMock:
    """Create a mock OddsApiClient."""
    c = MagicMock()
    c.get_soccer_odds = AsyncMock(return_value=[])
    c.get_all_soccer_outrights = AsyncMock(return_value=[])
    return c


def _make_matcher() -> MagicMock:
    """Create a mock EventMatcher."""
    m = MagicMock()
    m.match_markets.return_value = []
    m.match_seasonal_markets.return_value = []
    m.match_seasonal_via_match_odds.return_value = []
    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRouterScanAll:
    @pytest.mark.asyncio
    async def test_scan_returns_empty_when_no_markets(self) -> None:
        """Empty scan when no markets in any category."""
        vsg = ValueSignalGenerator(
            discovery=_make_discovery(),
            odds_client=_make_odds_client(),
            matcher=_make_matcher(),
        )
        signals = await vsg.scan()
        assert signals == []
        assert vsg.scan_count == 1

    @pytest.mark.asyncio
    async def test_scan_count_increments(self) -> None:
        """Scan count increments each call."""
        vsg = ValueSignalGenerator(
            discovery=_make_discovery(),
            odds_client=_make_odds_client(),
            matcher=_make_matcher(),
        )
        await vsg.scan()
        await vsg.scan()
        assert vsg.scan_count == 2


class TestRouterScanCrypto:
    @pytest.mark.asyncio
    async def test_crypto_scan_with_model(self) -> None:
        """Crypto scan produces signals when model predicts edge."""
        crypto_market = _make_gamma_market(
            "cid1", "Will BTC be above $100,000 on March 1?", "crypto", price_yes="0.50"
        )

        crypto_model = MagicMock()
        crypto_model.is_trained = True
        crypto_model.predict_single.return_value = 0.70  # 20% edge

        binance = MagicMock()
        ticker_mock = MagicMock()
        ticker_mock.last_price = 102000.0
        ticker_mock.price_change_pct = 2.5
        ticker_mock.quote_volume = 5000000000.0
        binance.get_ticker_24h = AsyncMock(return_value=ticker_mock)

        vsg = ValueSignalGenerator(
            discovery=_make_discovery(crypto=[crypto_market]),
            odds_client=_make_odds_client(),
            matcher=_make_matcher(),
            crypto_model=crypto_model,
            binance_client=binance,
            crypto_edge_threshold=0.03,
        )
        signals = await vsg.scan()

        assert len(signals) >= 1
        crypto_sigs = [s for s in signals if s.details.get("category") == "crypto"]
        assert len(crypto_sigs) == 1
        assert crypto_sigs[0].details["asset"] == "BTC"

    @pytest.mark.asyncio
    async def test_crypto_scan_skipped_without_model(self) -> None:
        """Crypto scan returns empty when no model loaded."""
        crypto_market = _make_gamma_market(
            "cid1", "Will BTC be above $100,000 on March 1?", "crypto"
        )

        vsg = ValueSignalGenerator(
            discovery=_make_discovery(crypto=[crypto_market]),
            odds_client=_make_odds_client(),
            matcher=_make_matcher(),
            crypto_model=None,
        )
        signals = await vsg.scan()
        assert signals == []

    @pytest.mark.asyncio
    async def test_crypto_scan_no_edge(self) -> None:
        """Crypto scan filters out markets below edge threshold."""
        crypto_market = _make_gamma_market(
            "cid1", "Will BTC be above $100,000 on March 1?", "crypto", price_yes="0.50"
        )

        crypto_model = MagicMock()
        crypto_model.is_trained = True
        crypto_model.predict_single.return_value = 0.51  # ~1% edge, below threshold

        binance = MagicMock()
        ticker_mock = MagicMock()
        ticker_mock.last_price = 100500.0
        ticker_mock.price_change_pct = 0.5
        ticker_mock.quote_volume = 5000000000.0
        binance.get_ticker_24h = AsyncMock(return_value=ticker_mock)

        vsg = ValueSignalGenerator(
            discovery=_make_discovery(crypto=[crypto_market]),
            odds_client=_make_odds_client(),
            matcher=_make_matcher(),
            crypto_model=crypto_model,
            binance_client=binance,
            crypto_edge_threshold=0.05,
        )
        signals = await vsg.scan()
        assert signals == []


class TestRouterScanPolitics:
    @pytest.mark.asyncio
    async def test_politics_scan_with_gemini(self) -> None:
        """Politics scan produces signals when LLM predicts edge."""
        politics_market = _make_gamma_market(
            "pid1", "Will Trump win the election?", "politics", price_yes="0.40"
        )

        gemini = MagicMock()
        prediction_mock = MagicMock()
        prediction_mock.probability = 0.65
        prediction_mock.confidence = 0.8
        prediction_mock.reasoning = "Strong polling data indicates..."
        prediction_mock.provider = "gemini"
        gemini.predict = AsyncMock(return_value=prediction_mock)

        vsg = ValueSignalGenerator(
            discovery=_make_discovery(politics=[politics_market]),
            odds_client=_make_odds_client(),
            matcher=_make_matcher(),
            gemini=gemini,
            politics_edge_threshold=0.04,
        )
        signals = await vsg.scan()

        assert len(signals) >= 1
        pol_sigs = [s for s in signals if s.details.get("category") == "politics"]
        assert len(pol_sigs) == 1
        assert "llm_reasoning" in pol_sigs[0].details

    @pytest.mark.asyncio
    async def test_politics_scan_skipped_without_gemini(self) -> None:
        """Politics scan returns empty when no Gemini agent."""
        politics_market = _make_gamma_market("pid1", "Will Trump win the election?", "politics")

        vsg = ValueSignalGenerator(
            discovery=_make_discovery(politics=[politics_market]),
            odds_client=_make_odds_client(),
            matcher=_make_matcher(),
            gemini=None,
        )
        signals = await vsg.scan()
        assert signals == []

    @pytest.mark.asyncio
    async def test_politics_max_per_scan(self) -> None:
        """Politics scan respects max_politics_per_scan cap."""
        politics_markets = [
            _make_gamma_market(f"p{i}", f"Election question {i}", "politics", price_yes="0.40")
            for i in range(20)
        ]

        gemini = MagicMock()
        prediction_mock = MagicMock()
        prediction_mock.probability = 0.65
        prediction_mock.confidence = 0.8
        prediction_mock.reasoning = "Analysis..."
        prediction_mock.provider = "gemini"
        gemini.predict = AsyncMock(return_value=prediction_mock)

        vsg = ValueSignalGenerator(
            discovery=_make_discovery(politics=politics_markets),
            odds_client=_make_odds_client(),
            matcher=_make_matcher(),
            gemini=gemini,
            max_politics_per_scan=5,
            politics_edge_threshold=0.04,
        )
        signals = await vsg.scan()

        # Should have at most 5 politics signals (capped)
        pol_sigs = [s for s in signals if s.details.get("category") == "politics"]
        assert len(pol_sigs) <= 5

    @pytest.mark.asyncio
    async def test_politics_llm_returns_none(self) -> None:
        """Politics scan handles LLM returning None gracefully."""
        politics_market = _make_gamma_market("pid1", "Will Trump win?", "politics")

        gemini = MagicMock()
        gemini.predict = AsyncMock(return_value=None)

        vsg = ValueSignalGenerator(
            discovery=_make_discovery(politics=[politics_market]),
            odds_client=_make_odds_client(),
            matcher=_make_matcher(),
            gemini=gemini,
        )
        signals = await vsg.scan()
        assert signals == []
