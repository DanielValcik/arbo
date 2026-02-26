"""Kaito AI API connector for mindshare and sentiment data (Strategy B).

Stub implementation — returns configurable mock data for development and
paper trading. When Kaito API launches publicly, swap _fetch_stub() → _fetch_live()
(estimated ≤4h work). See RDH-312 for API availability research.

Interface contract:
  - get_mindshare(topic) → MindshareData
  - get_sentiment(topic) → SentimentData
  - get_market_attention(condition_id) → MarketAttention

All methods are async. Stub mode is default; set live_mode=True + api_key
when real API is available.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from arbo.utils.logger import get_logger

logger = get_logger("kaito_api")


# ================================================================
# Data models
# ================================================================


@dataclass(frozen=True)
class MindshareData:
    """Kaito mindshare score for a topic.

    Mindshare = relative attention a topic receives across crypto/prediction
    market social channels (Twitter/X, Discord, Telegram, news).
    """

    topic: str
    score: float  # 0.0 – 1.0 (normalized attention share)
    trend: float  # -1.0 to +1.0 (24h momentum)
    volume: int  # raw mention count (24h)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for DB/API use."""
        return {
            "topic": self.topic,
            "score": self.score,
            "trend": self.trend,
            "volume": self.volume,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class SentimentData:
    """Kaito sentiment analysis for a topic.

    Sentiment derived from NLP analysis of social media posts about
    the topic in prediction market context.
    """

    topic: str
    sentiment: float  # -1.0 (bearish) to +1.0 (bullish)
    confidence: float  # 0.0 – 1.0
    sample_size: int  # number of posts analyzed
    bullish_pct: float  # % of posts classified bullish
    bearish_pct: float  # % of posts classified bearish
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for DB/API use."""
        return {
            "topic": self.topic,
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "sample_size": self.sample_size,
            "bullish_pct": self.bullish_pct,
            "bearish_pct": self.bearish_pct,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class MarketAttention:
    """Attention metrics for a specific Polymarket market.

    Used by Strategy B to compute divergence between market price
    and actual social attention/sentiment.
    """

    condition_id: str
    topic: str
    mindshare: MindshareData
    sentiment: SentimentData
    # Kaito's "actual" probability estimate based on social signals
    kaito_probability: float  # 0.0 – 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def divergence(self) -> float:
        """Compute divergence from a market price (set externally)."""
        return 0.0  # computed by strategy, not here

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for DB/API use."""
        return {
            "condition_id": self.condition_id,
            "topic": self.topic,
            "mindshare": self.mindshare.to_dict(),
            "sentiment": self.sentiment.to_dict(),
            "kaito_probability": self.kaito_probability,
            "timestamp": self.timestamp.isoformat(),
        }


# ================================================================
# Stub data generation
# ================================================================


def _deterministic_float(seed: str, low: float, high: float) -> float:
    """Generate a deterministic float from a seed string.

    Used so that stub data is consistent for the same topic within
    the same hour (makes testing predictable).
    """
    hour_key = f"{seed}:{int(time.time()) // 3600}"
    h = int(hashlib.md5(hour_key.encode()).hexdigest()[:8], 16)  # noqa: S324
    return low + (h / 0xFFFFFFFF) * (high - low)


def _generate_stub_mindshare(topic: str) -> MindshareData:
    """Generate deterministic stub mindshare data for a topic."""
    return MindshareData(
        topic=topic,
        score=round(_deterministic_float(f"ms:{topic}", 0.01, 0.30), 4),
        trend=round(_deterministic_float(f"mt:{topic}", -0.5, 0.5), 4),
        volume=int(_deterministic_float(f"mv:{topic}", 50, 5000)),
    )


def _generate_stub_sentiment(topic: str) -> SentimentData:
    """Generate deterministic stub sentiment data for a topic."""
    sentiment = round(_deterministic_float(f"ss:{topic}", -0.6, 0.6), 4)
    bullish = round(0.5 + sentiment * 0.3, 4)
    bearish = round(1.0 - bullish, 4)
    return SentimentData(
        topic=topic,
        sentiment=sentiment,
        confidence=round(_deterministic_float(f"sc:{topic}", 0.3, 0.8), 4),
        sample_size=int(_deterministic_float(f"sn:{topic}", 20, 500)),
        bullish_pct=bullish,
        bearish_pct=bearish,
    )


# ================================================================
# Client
# ================================================================


class KaitoClient:
    """Async Kaito AI API client.

    In stub mode (default), returns deterministic mock data.
    In live mode, makes real API calls (when Kaito API is available).

    Args:
        live_mode: If True, make real HTTP calls (requires api_key).
        api_key: Kaito API key (required for live mode).
        base_url: Kaito API base URL.
        cache_ttl_s: Cache TTL in seconds (default 300 = 5min).
        stub_overrides: Dict of topic → (MindshareData, SentimentData) for
            custom stub data in tests.
    """

    def __init__(
        self,
        live_mode: bool = False,
        api_key: str = "",
        base_url: str = "https://api.kaito.ai/v1",
        cache_ttl_s: int = 300,
        stub_overrides: dict[str, tuple[MindshareData, SentimentData]] | None = None,
    ) -> None:
        self._live_mode = live_mode
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._cache_ttl_s = cache_ttl_s
        self._stub_overrides = stub_overrides or {}
        self._cache: dict[str, tuple[float, Any]] = {}

        if live_mode and not api_key:
            raise ValueError("api_key required for live mode")

        mode_str = "live" if live_mode else "stub"
        logger.info("kaito_client_init", mode=mode_str)

    @property
    def is_stub(self) -> bool:
        """Whether the client is running in stub mode."""
        return not self._live_mode

    def _get_cached(self, key: str) -> Any | None:
        """Return cached value if still fresh, else None."""
        if key in self._cache:
            ts, val = self._cache[key]
            if time.monotonic() - ts < self._cache_ttl_s:
                return val
        return None

    def _set_cached(self, key: str, val: Any) -> None:
        """Store a value in the cache."""
        self._cache[key] = (time.monotonic(), val)

    async def get_mindshare(self, topic: str) -> MindshareData:
        """Get mindshare score for a topic.

        Args:
            topic: Topic string (e.g., market question, keyword).

        Returns:
            MindshareData with score, trend, and volume.
        """
        cache_key = f"mindshare:{topic}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        if self._live_mode:
            result = await self._fetch_live_mindshare(topic)
        else:
            result = self._fetch_stub_mindshare(topic)

        self._set_cached(cache_key, result)
        return result

    async def get_sentiment(self, topic: str) -> SentimentData:
        """Get sentiment analysis for a topic.

        Args:
            topic: Topic string (e.g., market question, keyword).

        Returns:
            SentimentData with sentiment score and confidence.
        """
        cache_key = f"sentiment:{topic}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        if self._live_mode:
            result = await self._fetch_live_sentiment(topic)
        else:
            result = self._fetch_stub_sentiment(topic)

        self._set_cached(cache_key, result)
        return result

    async def get_market_attention(
        self,
        condition_id: str,
        topic: str,
    ) -> MarketAttention:
        """Get full attention data for a specific market.

        Combines mindshare + sentiment into a single MarketAttention
        that Strategy B uses for divergence calculation.

        Args:
            condition_id: Polymarket condition ID.
            topic: Market question or topic keywords.

        Returns:
            MarketAttention with combined metrics.
        """
        mindshare = await self.get_mindshare(topic)
        sentiment = await self.get_sentiment(topic)

        # Estimate "actual" probability from social signals
        # In stub mode: blend mindshare score + sentiment
        kaito_prob = self._estimate_probability(mindshare, sentiment)

        return MarketAttention(
            condition_id=condition_id,
            topic=topic,
            mindshare=mindshare,
            sentiment=sentiment,
            kaito_probability=kaito_prob,
        )

    def _estimate_probability(
        self,
        mindshare: MindshareData,
        sentiment: SentimentData,
    ) -> float:
        """Estimate market probability from social signals.

        Simple model: base 0.5 + sentiment weight + mindshare trend.
        In live mode, Kaito API would provide this directly.
        """
        base = 0.5
        sentiment_adj = sentiment.sentiment * 0.2 * sentiment.confidence
        trend_adj = mindshare.trend * 0.1
        prob = max(0.01, min(0.99, base + sentiment_adj + trend_adj))
        return round(prob, 4)

    # ------------------------------------------------------------------
    # Stub data (default)
    # ------------------------------------------------------------------

    def _fetch_stub_mindshare(self, topic: str) -> MindshareData:
        """Return stub mindshare data."""
        if topic in self._stub_overrides:
            return self._stub_overrides[topic][0]
        return _generate_stub_mindshare(topic)

    def _fetch_stub_sentiment(self, topic: str) -> SentimentData:
        """Return stub sentiment data."""
        if topic in self._stub_overrides:
            return self._stub_overrides[topic][1]
        return _generate_stub_sentiment(topic)

    # ------------------------------------------------------------------
    # Live API (placeholder — swap in when Kaito API launches)
    # ------------------------------------------------------------------

    async def _fetch_live_mindshare(self, topic: str) -> MindshareData:
        """Fetch real mindshare data from Kaito API.

        TODO(RDH-405): Implement when Kaito API is available.
        Estimated ≤2h work to implement this method.
        """
        raise NotImplementedError(
            "Kaito live API not yet available. "
            "See RDH-312 research report for status."
        )

    async def _fetch_live_sentiment(self, topic: str) -> SentimentData:
        """Fetch real sentiment data from Kaito API.

        TODO(RDH-405): Implement when Kaito API is available.
        Estimated ≤2h work to implement this method.
        """
        raise NotImplementedError(
            "Kaito live API not yet available. "
            "See RDH-312 research report for status."
        )

    async def close(self) -> None:
        """Close any open connections."""
        self._cache.clear()
        logger.info("kaito_client_closed")
