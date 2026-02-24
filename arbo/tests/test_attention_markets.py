"""Tests for Attention Markets Scanner (PM-207).

Tests verify:
1. Market filtering: category included, non-attention excluded, keyword inclusion
2. Sentiment estimation: estimate returned, Gemini fail, prompt content
3. Scan cycle: above threshold, below threshold
4. Signal format: layer=8, details contain sentiment
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from arbo.connectors.market_discovery import GammaMarket, MarketDiscovery
from arbo.core.scanner import SignalDirection
from arbo.strategies.attention_markets import AttentionMarketsScanner

# ================================================================
# Helpers
# ================================================================


def _make_market(
    condition_id: str = "cond_1",
    question: str = "Test market?",
    category: str = "other",
    outcome_prices: list[str] | None = None,
    clob_token_ids: list[str] | None = None,
) -> GammaMarket:
    """Build a GammaMarket with overridden category."""
    raw = {
        "conditionId": condition_id,
        "question": question,
        "slug": "test-slug",
        "outcomes": ["Yes", "No"],
        "outcomePrices": outcome_prices or ["0.50", "0.50"],
        "clobTokenIds": clob_token_ids or ["tok_yes", "tok_no"],
        "volume": "100000",
        "volume24hr": "5000",
        "liquidity": "10000",
        "active": True,
        "closed": False,
        "feesEnabled": False,
        "enableNegRisk": False,
        "tags": [],
    }
    market = GammaMarket(raw)
    market.category = category  # Override auto-categorization
    return market


class FakePrediction:
    def __init__(self, probability: float = 0.7, confidence: float = 0.8) -> None:
        self.probability = probability
        self.confidence = confidence
        self.reasoning = "Based on social media trends"
        self.provider = "mock"
        self.latency_ms = 100
        self.model = "test"


def _make_discovery(markets: list[GammaMarket]) -> MarketDiscovery:
    disc = MarketDiscovery()
    disc._markets = {m.condition_id: m for m in markets}
    return disc


# ================================================================
# TestMarketFiltering
# ================================================================


class TestMarketFiltering:
    """Market filtering for attention markets."""

    def test_category_included(self) -> None:
        """Markets with category 'attention_markets' are included."""
        market = _make_market(category="attention_markets")
        disc = _make_discovery([market])
        scanner = AttentionMarketsScanner(discovery=disc)
        filtered = scanner._filter_attention_markets([market])
        assert len(filtered) == 1

    def test_non_attention_excluded(self) -> None:
        """Non-attention markets are excluded."""
        market = _make_market(category="soccer", question="Who wins the game?")
        disc = _make_discovery([market])
        scanner = AttentionMarketsScanner(discovery=disc)
        filtered = scanner._filter_attention_markets([market])
        assert len(filtered) == 0

    def test_keyword_inclusion(self) -> None:
        """Markets with attention keywords in question are included."""
        market = _make_market(
            category="other",
            question="Will Kaito mindshare reach 50%?",
        )
        disc = _make_discovery([market])
        scanner = AttentionMarketsScanner(discovery=disc)
        filtered = scanner._filter_attention_markets([market])
        assert len(filtered) == 1


# ================================================================
# TestSentimentEstimation
# ================================================================


class TestSentimentEstimation:
    """Sentiment estimation via LLM."""

    @pytest.mark.asyncio
    async def test_estimate_returned(self) -> None:
        """Successful LLM call returns SentimentEstimate."""
        gemini = AsyncMock()
        gemini.predict.return_value = FakePrediction(probability=0.7, confidence=0.8)

        market = _make_market()
        disc = _make_discovery([market])
        scanner = AttentionMarketsScanner(discovery=disc, gemini=gemini)

        estimate = await scanner._estimate_sentiment(market)
        assert estimate is not None
        assert estimate.sentiment_prob == Decimal("0.7")
        assert estimate.confidence == Decimal("0.8")

    @pytest.mark.asyncio
    async def test_gemini_fail_returns_none(self) -> None:
        """LLM failure returns None."""
        gemini = AsyncMock()
        gemini.predict.return_value = None

        market = _make_market()
        disc = _make_discovery([market])
        scanner = AttentionMarketsScanner(discovery=disc, gemini=gemini)

        estimate = await scanner._estimate_sentiment(market)
        assert estimate is None

    @pytest.mark.asyncio
    async def test_prompt_contains_question(self) -> None:
        """LLM prompt should contain the market question."""
        gemini = AsyncMock()
        gemini.predict.return_value = FakePrediction()

        market = _make_market(question="Will Kaito reach 50% mindshare?")
        disc = _make_discovery([market])
        scanner = AttentionMarketsScanner(discovery=disc, gemini=gemini)

        await scanner._estimate_sentiment(market)
        call_args = gemini.predict.call_args
        assert "Kaito" in call_args.kwargs.get(
            "question", call_args.args[0] if call_args.args else ""
        )


# ================================================================
# TestScanCycle
# ================================================================


class TestScanCycle:
    """Full scan cycle tests."""

    @pytest.mark.asyncio
    async def test_above_threshold_signal(self) -> None:
        """Divergence > 5% generates a signal."""
        gemini = AsyncMock()
        gemini.predict.return_value = FakePrediction(probability=0.70, confidence=0.8)

        # Market price at 0.50, sentiment says 0.70 → 20% divergence
        market = _make_market(
            category="attention_markets",
            outcome_prices=["0.50", "0.50"],
        )
        disc = _make_discovery([market])
        scanner = AttentionMarketsScanner(discovery=disc, gemini=gemini)

        signals = await scanner.scan()
        assert len(signals) == 1
        assert signals[0].direction == SignalDirection.BUY_YES

    @pytest.mark.asyncio
    async def test_below_threshold_no_signal(self) -> None:
        """Divergence < 5% generates no signal."""
        gemini = AsyncMock()
        gemini.predict.return_value = FakePrediction(probability=0.52, confidence=0.6)

        # Market price at 0.50, sentiment says 0.52 → 2% divergence (below 5%)
        market = _make_market(
            category="attention_markets",
            outcome_prices=["0.50", "0.50"],
        )
        disc = _make_discovery([market])
        scanner = AttentionMarketsScanner(discovery=disc, gemini=gemini)

        signals = await scanner.scan()
        assert len(signals) == 0


# ================================================================
# TestSignalFormat
# ================================================================


class TestSignalFormat:
    """Signal output format."""

    @pytest.mark.asyncio
    async def test_signal_layer_8(self) -> None:
        """Attention market signals have layer=8."""
        gemini = AsyncMock()
        gemini.predict.return_value = FakePrediction(probability=0.70, confidence=0.9)

        market = _make_market(
            category="attention_markets",
            outcome_prices=["0.50", "0.50"],
        )
        disc = _make_discovery([market])
        scanner = AttentionMarketsScanner(discovery=disc, gemini=gemini)

        signals = await scanner.scan()
        assert signals[0].layer == 8

    @pytest.mark.asyncio
    async def test_signal_details_contain_sentiment(self) -> None:
        """Signal details contain sentiment probability and reasoning."""
        gemini = AsyncMock()
        gemini.predict.return_value = FakePrediction(probability=0.70, confidence=0.9)

        market = _make_market(
            category="attention_markets",
            outcome_prices=["0.50", "0.50"],
        )
        disc = _make_discovery([market])
        scanner = AttentionMarketsScanner(discovery=disc, gemini=gemini)

        signals = await scanner.scan()
        assert "sentiment_prob" in signals[0].details
        assert "reasoning" in signals[0].details
