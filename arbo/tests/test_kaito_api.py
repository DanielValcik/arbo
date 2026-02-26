"""Tests for Kaito AI API stub connector (RDH-301).

Tests verify:
1. MindshareData and SentimentData creation
2. KaitoClient stub mode returns data
3. KaitoClient cache works
4. Stub overrides for custom test data
5. MarketAttention combines mindshare + sentiment
6. Live mode requires API key
7. Live mode raises NotImplementedError
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from arbo.connectors.kaito_api import (
    KaitoClient,
    MarketAttention,
    MindshareData,
    SentimentData,
)


# ================================================================
# Data model tests
# ================================================================


class TestMindshareData:
    """MindshareData dataclass."""

    def test_creation(self) -> None:
        """MindshareData stores fields correctly."""
        ms = MindshareData(
            topic="bitcoin",
            score=0.15,
            trend=0.3,
            volume=1200,
        )
        assert ms.topic == "bitcoin"
        assert ms.score == 0.15
        assert ms.trend == 0.3
        assert ms.volume == 1200
        assert ms.timestamp is not None

    def test_to_dict(self) -> None:
        """to_dict() serializes all fields."""
        ms = MindshareData(topic="eth", score=0.1, trend=-0.2, volume=500)
        d = ms.to_dict()
        assert d["topic"] == "eth"
        assert d["score"] == 0.1
        assert d["trend"] == -0.2
        assert d["volume"] == 500
        assert "timestamp" in d

    def test_frozen(self) -> None:
        """MindshareData is immutable."""
        ms = MindshareData(topic="btc", score=0.1, trend=0.0, volume=100)
        with pytest.raises(AttributeError):
            ms.score = 0.5  # type: ignore[misc]


class TestSentimentData:
    """SentimentData dataclass."""

    def test_creation(self) -> None:
        """SentimentData stores fields correctly."""
        sd = SentimentData(
            topic="election",
            sentiment=0.4,
            confidence=0.75,
            sample_size=300,
            bullish_pct=0.65,
            bearish_pct=0.35,
        )
        assert sd.topic == "election"
        assert sd.sentiment == 0.4
        assert sd.confidence == 0.75
        assert sd.sample_size == 300
        assert sd.bullish_pct == 0.65
        assert sd.bearish_pct == 0.35

    def test_to_dict(self) -> None:
        """to_dict() serializes all fields."""
        sd = SentimentData(
            topic="test",
            sentiment=-0.3,
            confidence=0.5,
            sample_size=100,
            bullish_pct=0.4,
            bearish_pct=0.6,
        )
        d = sd.to_dict()
        assert d["sentiment"] == -0.3
        assert d["bullish_pct"] == 0.4
        assert d["bearish_pct"] == 0.6


class TestMarketAttention:
    """MarketAttention combined data."""

    def test_creation(self) -> None:
        """MarketAttention stores fields correctly."""
        ms = MindshareData(topic="test", score=0.2, trend=0.1, volume=500)
        sd = SentimentData(
            topic="test",
            sentiment=0.3,
            confidence=0.6,
            sample_size=200,
            bullish_pct=0.6,
            bearish_pct=0.4,
        )
        ma = MarketAttention(
            condition_id="cond_123",
            topic="test",
            mindshare=ms,
            sentiment=sd,
            kaito_probability=0.55,
        )
        assert ma.condition_id == "cond_123"
        assert ma.kaito_probability == 0.55
        assert ma.mindshare.score == 0.2
        assert ma.sentiment.sentiment == 0.3

    def test_to_dict(self) -> None:
        """to_dict() includes nested data."""
        ms = MindshareData(topic="x", score=0.1, trend=0.0, volume=100)
        sd = SentimentData(
            topic="x", sentiment=0.0, confidence=0.5,
            sample_size=50, bullish_pct=0.5, bearish_pct=0.5,
        )
        ma = MarketAttention(
            condition_id="cond_456",
            topic="x",
            mindshare=ms,
            sentiment=sd,
            kaito_probability=0.50,
        )
        d = ma.to_dict()
        assert d["condition_id"] == "cond_456"
        assert d["mindshare"]["score"] == 0.1
        assert d["sentiment"]["sentiment"] == 0.0
        assert d["kaito_probability"] == 0.50


# ================================================================
# KaitoClient tests
# ================================================================


class TestKaitoClientStub:
    """KaitoClient in stub mode."""

    async def test_is_stub(self) -> None:
        """Default client is in stub mode."""
        client = KaitoClient()
        assert client.is_stub is True

    async def test_get_mindshare_returns_data(self) -> None:
        """get_mindshare returns MindshareData in stub mode."""
        client = KaitoClient()
        ms = await client.get_mindshare("bitcoin price prediction")
        assert isinstance(ms, MindshareData)
        assert ms.topic == "bitcoin price prediction"
        assert 0.0 <= ms.score <= 1.0
        assert -1.0 <= ms.trend <= 1.0
        assert ms.volume > 0

    async def test_get_sentiment_returns_data(self) -> None:
        """get_sentiment returns SentimentData in stub mode."""
        client = KaitoClient()
        sd = await client.get_sentiment("election outcome")
        assert isinstance(sd, SentimentData)
        assert sd.topic == "election outcome"
        assert -1.0 <= sd.sentiment <= 1.0
        assert 0.0 <= sd.confidence <= 1.0
        assert sd.sample_size > 0
        assert sd.bullish_pct + sd.bearish_pct == pytest.approx(1.0, abs=0.01)

    async def test_get_market_attention(self) -> None:
        """get_market_attention combines mindshare + sentiment."""
        client = KaitoClient()
        ma = await client.get_market_attention("cond_test", "will it rain?")
        assert isinstance(ma, MarketAttention)
        assert ma.condition_id == "cond_test"
        assert ma.topic == "will it rain?"
        assert isinstance(ma.mindshare, MindshareData)
        assert isinstance(ma.sentiment, SentimentData)
        assert 0.0 < ma.kaito_probability < 1.0

    async def test_stub_deterministic(self) -> None:
        """Stub data is deterministic for same topic within same hour."""
        client = KaitoClient()
        ms1 = await client.get_mindshare("test_topic")
        # Clear cache to force re-generation
        client._cache.clear()
        ms2 = await client.get_mindshare("test_topic")
        assert ms1.score == ms2.score
        assert ms1.volume == ms2.volume

    async def test_different_topics_different_data(self) -> None:
        """Different topics return different stub data."""
        client = KaitoClient()
        ms1 = await client.get_mindshare("topic_alpha")
        ms2 = await client.get_mindshare("topic_beta")
        # Very unlikely to be exactly the same
        assert ms1.score != ms2.score or ms1.volume != ms2.volume

    async def test_cache_returns_same_object(self) -> None:
        """Cache returns same result on second call."""
        client = KaitoClient()
        ms1 = await client.get_mindshare("cached_topic")
        ms2 = await client.get_mindshare("cached_topic")
        assert ms1 is ms2  # exact same object from cache

    async def test_stub_overrides(self) -> None:
        """Custom stub overrides work for testing."""
        custom_ms = MindshareData(topic="custom", score=0.99, trend=0.8, volume=9999)
        custom_sd = SentimentData(
            topic="custom", sentiment=0.9, confidence=0.95,
            sample_size=1000, bullish_pct=0.9, bearish_pct=0.1,
        )
        client = KaitoClient(stub_overrides={"custom": (custom_ms, custom_sd)})

        ms = await client.get_mindshare("custom")
        assert ms.score == 0.99
        assert ms.volume == 9999

        sd = await client.get_sentiment("custom")
        assert sd.sentiment == 0.9
        assert sd.confidence == 0.95

    async def test_close(self) -> None:
        """close() clears cache without error."""
        client = KaitoClient()
        await client.get_mindshare("test")
        assert len(client._cache) > 0
        await client.close()
        assert len(client._cache) == 0


class TestKaitoClientLive:
    """KaitoClient live mode validation."""

    def test_live_mode_requires_api_key(self) -> None:
        """Live mode raises ValueError without api_key."""
        with pytest.raises(ValueError, match="api_key required"):
            KaitoClient(live_mode=True)

    def test_live_mode_with_key(self) -> None:
        """Live mode initializes with api_key."""
        client = KaitoClient(live_mode=True, api_key="test-key-123")
        assert client.is_stub is False

    async def test_live_mindshare_not_implemented(self) -> None:
        """Live mindshare fetch raises NotImplementedError."""
        client = KaitoClient(live_mode=True, api_key="test-key")
        with pytest.raises(NotImplementedError, match="not yet available"):
            await client.get_mindshare("test")

    async def test_live_sentiment_not_implemented(self) -> None:
        """Live sentiment fetch raises NotImplementedError."""
        client = KaitoClient(live_mode=True, api_key="test-key")
        with pytest.raises(NotImplementedError, match="not yet available"):
            await client.get_sentiment("test")


class TestProbabilityEstimation:
    """Probability estimation from social signals."""

    async def test_neutral_signals_near_half(self) -> None:
        """Neutral sentiment and flat trend → probability near 0.5."""
        ms = MindshareData(topic="neutral", score=0.15, trend=0.0, volume=500)
        sd = SentimentData(
            topic="neutral", sentiment=0.0, confidence=0.5,
            sample_size=200, bullish_pct=0.5, bearish_pct=0.5,
        )
        client = KaitoClient(stub_overrides={"neutral": (ms, sd)})
        ma = await client.get_market_attention("cond_1", "neutral")
        assert 0.45 <= ma.kaito_probability <= 0.55

    async def test_bullish_signals_above_half(self) -> None:
        """Strong bullish sentiment → probability > 0.5."""
        ms = MindshareData(topic="bull", score=0.3, trend=0.5, volume=2000)
        sd = SentimentData(
            topic="bull", sentiment=0.8, confidence=0.9,
            sample_size=500, bullish_pct=0.85, bearish_pct=0.15,
        )
        client = KaitoClient(stub_overrides={"bull": (ms, sd)})
        ma = await client.get_market_attention("cond_2", "bull")
        assert ma.kaito_probability > 0.55

    async def test_bearish_signals_below_half(self) -> None:
        """Strong bearish sentiment → probability < 0.5."""
        ms = MindshareData(topic="bear", score=0.1, trend=-0.4, volume=300)
        sd = SentimentData(
            topic="bear", sentiment=-0.7, confidence=0.85,
            sample_size=400, bullish_pct=0.2, bearish_pct=0.8,
        )
        client = KaitoClient(stub_overrides={"bear": (ms, sd)})
        ma = await client.get_market_attention("cond_3", "bear")
        assert ma.kaito_probability < 0.45

    async def test_probability_bounded(self) -> None:
        """Probability never exceeds [0.01, 0.99] bounds."""
        ms = MindshareData(topic="extreme", score=0.99, trend=1.0, volume=9999)
        sd = SentimentData(
            topic="extreme", sentiment=1.0, confidence=1.0,
            sample_size=1000, bullish_pct=1.0, bearish_pct=0.0,
        )
        client = KaitoClient(stub_overrides={"extreme": (ms, sd)})
        ma = await client.get_market_attention("cond_4", "extreme")
        assert 0.01 <= ma.kaito_probability <= 0.99
