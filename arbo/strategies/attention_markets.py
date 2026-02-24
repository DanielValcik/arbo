"""Attention markets scanner using Gemini sentiment analysis (PM-207).

Identifies attention/mindshare markets on Polymarket, estimates probability
via LLM analysis of social media trends, and generates Layer 8 signals
when divergence exceeds threshold.

See brief Layer 8 for full specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from arbo.config.settings import get_config
from arbo.connectors.market_discovery import MarketDiscovery  # noqa: TC001
from arbo.core.scanner import Signal, SignalDirection
from arbo.utils.logger import get_logger

logger = get_logger("attention_markets")

# Keywords that identify attention/mindshare markets
ATTENTION_KEYWORDS = [
    "mindshare",
    "attention",
    "kaito",
    "trending",
    "viral",
    "hype",
    "sentiment",
    "social",
    "popularity",
    "followers",
    "views",
    "engagement",
]


@dataclass
class SentimentEstimate:
    """Sentiment-based probability estimate from LLM."""

    market_condition_id: str
    question: str
    sentiment_prob: Decimal
    confidence: Decimal
    sources_analyzed: list[str]
    reasoning: str
    estimated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class AttentionMarketsScanner:
    """Scans attention/mindshare markets for sentiment-driven opportunities.

    Filters markets by category or keywords, estimates probabilities via
    Gemini LLM analysis, and generates Layer 8 signals when divergence
    from market price exceeds min_divergence threshold.
    """

    def __init__(
        self,
        discovery: MarketDiscovery,
        gemini: Any = None,
    ) -> None:
        self._discovery = discovery
        self._gemini = gemini
        config = get_config()
        self._min_divergence = Decimal(str(config.attention_markets.min_divergence))
        self._sources = config.attention_markets.sources
        self._total_scans = 0
        self._total_signals = 0

    async def scan(self) -> list[Signal]:
        """Run attention markets scan.

        1. Filter markets to attention/mindshare category
        2. Estimate sentiment probability via LLM for each
        3. Compare against market price
        4. Generate signal if divergence > threshold

        Returns:
            List of Layer 8 signals.
        """
        self._total_scans += 1
        all_markets = self._discovery.get_all()
        attention_markets = self._filter_attention_markets(all_markets)

        if not attention_markets:
            logger.debug("attention_no_markets_found")
            return []

        # Limit to 10 per scan to stay within health monitor heartbeat (120s)
        # Sort by volume (most liquid first) for best signal quality
        attention_markets.sort(key=lambda m: m.volume_24h, reverse=True)
        attention_markets = attention_markets[:10]

        signals: list[Signal] = []

        for market in attention_markets:
            estimate = await self._estimate_sentiment(market)
            if estimate is None:
                continue

            market_price = market.price_yes
            if market_price is None:
                continue

            divergence = abs(estimate.sentiment_prob - market_price)

            if divergence < self._min_divergence:
                continue

            # LLM divergence > 25% is likely hallucination — skip
            if divergence > Decimal("0.25"):
                logger.debug(
                    "attention_skip_hallucination",
                    question=market.question[:60],
                    divergence=str(divergence),
                )
                continue

            # Cap edge at 10% — LLM edges are qualitatively different from L2
            edge = min(divergence, Decimal("0.10"))

            # Direction: if sentiment says higher prob, BUY YES
            if estimate.sentiment_prob > market_price:
                direction = SignalDirection.BUY_YES
                token_id = market.token_id_yes or ""
            else:
                direction = SignalDirection.BUY_NO
                token_id = market.token_id_no or market.token_id_yes or ""

            signals.append(
                Signal(
                    layer=8,
                    market_condition_id=market.condition_id,
                    token_id=token_id,
                    direction=direction,
                    edge=edge,
                    confidence=estimate.confidence,
                    details={
                        "sentiment_prob": str(estimate.sentiment_prob),
                        "market_price": str(market_price),
                        "divergence": str(divergence),
                        "poly_price": str(market_price),
                        "reasoning": estimate.reasoning,
                        "sources": estimate.sources_analyzed,
                        "question": market.question[:100],
                    },
                )
            )

        self._total_signals += len(signals)
        logger.info(
            "attention_scan_complete",
            markets_checked=len(attention_markets),
            signals=len(signals),
        )

        return signals

    @property
    def stats(self) -> dict[str, Any]:
        """Get scanner statistics."""
        return {
            "total_scans": self._total_scans,
            "total_signals": self._total_signals,
        }

    def _filter_attention_markets(self, markets: list[Any]) -> list[Any]:
        """Filter markets to attention/mindshare category.

        Matches by category == "attention_markets" or keywords in question.
        """
        result = []
        for m in markets:
            # Category match
            if hasattr(m, "category") and m.category == "attention_markets":
                result.append(m)
                continue

            # Keyword match
            question = m.question.lower() if hasattr(m, "question") else ""
            if any(kw in question for kw in ATTENTION_KEYWORDS):
                result.append(m)

        return result

    async def _estimate_sentiment(self, market: Any) -> SentimentEstimate | None:
        """Estimate probability from social media sentiment via LLM.

        Args:
            market: GammaMarket object.

        Returns:
            SentimentEstimate if successful, None if LLM unavailable or fails.
        """
        if not self._gemini:
            return None

        question = market.question if hasattr(market, "question") else ""
        current_price = (
            float(market.price_yes) if hasattr(market, "price_yes") and market.price_yes else 0.5
        )

        try:
            prediction = await self._gemini.predict(
                question=(
                    f"Estimate the probability of this prediction market question based on "
                    f"current social media trends, public attention, and sentiment:\n\n"
                    f"Question: {question}\n"
                    f"Current market price: {current_price:.4f}\n\n"
                    f"Consider: Twitter/X trends, Reddit discussions, news coverage, "
                    f"and general public attention to this topic."
                ),
                current_price=current_price,
                category="attention_markets",
            )

            if prediction is None:
                return None

            return SentimentEstimate(
                market_condition_id=market.condition_id if hasattr(market, "condition_id") else "",
                question=question,
                sentiment_prob=Decimal(str(prediction.probability)),
                confidence=Decimal(str(prediction.confidence)),
                sources_analyzed=list(self._sources),
                reasoning=prediction.reasoning,
            )

        except Exception as e:
            logger.debug("sentiment_estimate_error", error=str(e))
            return None
