"""Gemini LLM probability agent (PM-104).

Primary LLM: Gemini 2.5 Flash with JSON response mode.
Fallback: Claude Haiku 4.5 via anthropic SDK.
Rate limited to max_calls_per_hour (default 60).

See brief Layer 2/5/8 for LLM usage specification.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from arbo.config.settings import get_config
from arbo.utils.logger import get_logger

logger = get_logger("gemini_agent")


@dataclass
class LLMPrediction:
    """Prediction from an LLM provider."""

    probability: float
    confidence: float
    reasoning: str
    provider: str
    latency_ms: int
    model: str


class RateLimiter:
    """Sliding window rate limiter.

    Tracks call timestamps in a deque and allows calls only when
    the number of calls within the window is below max_calls.
    """

    def __init__(self, max_calls: int, window_seconds: int = 3600) -> None:
        self._max_calls = max_calls
        self._window_seconds = window_seconds
        self._timestamps: deque[float] = deque()

    def allow(self) -> bool:
        """Check if a call is allowed and record it if so."""
        now = time.monotonic()
        self._prune(now)
        if len(self._timestamps) >= self._max_calls:
            return False
        self._timestamps.append(now)
        return True

    def remaining(self) -> int:
        """Number of calls remaining in current window."""
        self._prune(time.monotonic())
        return max(0, self._max_calls - len(self._timestamps))

    def _prune(self, now: float) -> None:
        """Remove expired timestamps."""
        cutoff = now - self._window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()


def _extract_json(text: str) -> dict[str, Any] | None:
    """Extract JSON object from LLM response text.

    Handles markdown code blocks, text before/after JSON, etc.
    """
    if not text or not text.strip():
        return None

    # Try direct parse first
    try:
        return json.loads(text)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        pass

    # Strip markdown code blocks: ```json\n...\n``` or ```\n...\n```
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    # Find first { ... } JSON object in text
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    return None


class GeminiAgent:
    """LLM probability prediction agent.

    Uses Gemini 2.0 Flash as primary provider with Claude Haiku 4.5 fallback.
    Structured JSON response mode for reliable parsing.
    Rate limited per config.llm.max_calls_per_hour.
    """

    def __init__(self) -> None:
        config = get_config()
        self._gemini_model_name = config.llm.gemini_model
        self._claude_model_name = config.llm.claude_model
        self._gemini_api_key = config.gemini_api_key
        self._anthropic_api_key = config.anthropic_api_key
        self._rate_limiter = RateLimiter(max_calls=config.llm.max_calls_per_hour)
        self._raw_rate_limiter = RateLimiter(max_calls=config.llm.max_calls_per_hour)
        self._gemini_model: Any = None
        self._anthropic_client: Any = None
        self._total_calls = 0
        self._fallback_calls = 0
        self._gemini_timeout_s = 30.0  # Gemini 2.5 Flash is a thinking model
        self._claude_timeout_s = 15.0

    async def initialize(self) -> None:
        """Initialize LLM clients."""
        if self._gemini_api_key:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self._gemini_api_key)
                self._gemini_model = genai.GenerativeModel(
                    self._gemini_model_name,
                    generation_config={"response_mime_type": "application/json"},
                )
                logger.info("gemini_initialized", model=self._gemini_model_name)
            except Exception as e:
                logger.warning("gemini_init_failed", error=str(e))
        else:
            logger.warning("gemini_no_api_key", msg="GEMINI_API_KEY not set")

        if self._anthropic_api_key:
            try:
                import anthropic

                self._anthropic_client = anthropic.AsyncAnthropic(
                    api_key=self._anthropic_api_key,
                )
                logger.info("anthropic_initialized", model=self._claude_model_name)
            except Exception as e:
                logger.warning("anthropic_init_failed", error=str(e))
        else:
            logger.warning("anthropic_no_api_key", msg="ANTHROPIC_API_KEY not set")

    async def raw_query(self, prompt: str) -> dict[str, Any] | None:
        """Send a raw prompt and return parsed JSON dict.

        Unlike predict(), does NOT wrap in probability estimation template.
        Used for graph classification, logical arb, and other non-prediction tasks.
        Uses a separate rate limiter to avoid interference with predict().
        """
        if not self._raw_rate_limiter.allow():
            return None

        # Try Gemini first
        if self._gemini_model is not None:
            result = await self._raw_call_gemini(prompt)
            if result is not None:
                self._total_calls += 1
                return result

        # Fallback to Claude
        if self._anthropic_client is not None:
            result = await self._raw_call_claude(prompt)
            if result is not None:
                self._total_calls += 1
                self._fallback_calls += 1
                return result

        logger.warning("llm_raw_both_failed", prompt=prompt[:80])
        return None

    async def _raw_call_gemini(self, prompt: str) -> dict[str, Any] | None:
        """Call Gemini with raw prompt and return parsed JSON."""
        try:
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._gemini_model.generate_content(prompt),
                ),
                timeout=self._gemini_timeout_s,
            )
            return _extract_json(response.text)
        except Exception as e:
            logger.debug("gemini_raw_error", error=str(e))
            return None

    async def _raw_call_claude(self, prompt: str) -> dict[str, Any] | None:
        """Call Claude with raw prompt and return parsed JSON."""
        try:
            response = await asyncio.wait_for(
                self._anthropic_client.messages.create(
                    model=self._claude_model_name,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=self._claude_timeout_s,
            )
            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text += block.text
            return _extract_json(text) if text else None
        except Exception as e:
            logger.debug("claude_raw_error", error=str(e))
            return None

    async def predict(
        self,
        question: str,
        current_price: float,
        category: str = "",
        volume_24h: float = 0.0,
    ) -> LLMPrediction | None:
        """Get probability prediction for a market question.

        Args:
            question: The market question text.
            current_price: Current Polymarket YES price (0-1).
            category: Market category (e.g. "soccer", "crypto").
            volume_24h: 24h trading volume in USDC.

        Returns:
            LLMPrediction if successful, None if rate limited or both providers fail.
        """
        if not self._rate_limiter.allow():
            logger.debug("llm_rate_limited", remaining=self._rate_limiter.remaining())
            return None

        prompt = self._build_prompt(question, current_price, category, volume_24h)

        # Try Gemini first
        if self._gemini_model is not None:
            prediction = await self._call_gemini(prompt)
            if prediction is not None:
                self._total_calls += 1
                return prediction

        # Fallback to Claude
        if self._anthropic_client is not None:
            prediction = await self._call_claude(prompt)
            if prediction is not None:
                self._total_calls += 1
                self._fallback_calls += 1
                return prediction

        logger.warning("llm_both_failed", question=question[:80])
        return None

    def _build_prompt(
        self,
        question: str,
        current_price: float,
        category: str,
        volume_24h: float,
    ) -> str:
        """Build structured prompt for LLM probability estimation."""
        return (
            "You are a prediction market analyst. Estimate the probability of the following event.\n\n"
            f"Question: {question}\n"
            f"Current market price: {current_price:.4f}\n"
            f"Category: {category}\n"
            f"24h volume: ${volume_24h:,.0f}\n\n"
            "Respond with JSON containing:\n"
            '- "probability": float between 0 and 1\n'
            '- "confidence": float between 0 and 1 (how confident you are)\n'
            '- "reasoning": string explaining your estimate\n\n'
            'Example: {"probability": 0.65, "confidence": 0.7, '
            '"reasoning": "Based on current form and historical data..."}'
        )

    async def _call_gemini(self, prompt: str) -> LLMPrediction | None:
        """Call Gemini API with timeout."""
        start = time.monotonic()
        try:
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._gemini_model.generate_content(prompt),
                ),
                timeout=self._gemini_timeout_s,
            )
            elapsed_ms = int((time.monotonic() - start) * 1000)

            text = response.text
            data = _extract_json(text)
            if data is None:
                logger.warning("gemini_parse_error", error="no JSON found", raw=text[:200])
                return None
            return self._validate_prediction(data, "gemini", self._gemini_model_name, elapsed_ms)

        except TimeoutError:
            logger.warning("gemini_timeout", timeout_s=self._gemini_timeout_s)
            return None
        except Exception as e:
            logger.warning("gemini_error", error=str(e))
            return None

    async def _call_claude(self, prompt: str) -> LLMPrediction | None:
        """Call Claude API as fallback."""
        start = time.monotonic()
        try:
            response = await asyncio.wait_for(
                self._anthropic_client.messages.create(
                    model=self._claude_model_name,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=self._claude_timeout_s,
            )
            elapsed_ms = int((time.monotonic() - start) * 1000)

            # Extract text from response content blocks
            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text += block.text
            if not text:
                logger.warning(
                    "claude_empty_response",
                    stop_reason=response.stop_reason,
                    content_types=[b.type for b in response.content],
                )
                return None

            data = _extract_json(text)
            if data is None:
                logger.warning("claude_parse_error", error="no JSON found", raw=text[:200])
                return None
            return self._validate_prediction(data, "claude", self._claude_model_name, elapsed_ms)

        except TimeoutError:
            logger.warning("claude_timeout", timeout_s=self._claude_timeout_s)
            return None
        except Exception as e:
            logger.warning("claude_error", error=str(e))
            return None

    def _validate_prediction(
        self,
        data: dict[str, Any],
        provider: str,
        model: str,
        latency_ms: int,
    ) -> LLMPrediction | None:
        """Validate and clamp LLM response."""
        try:
            probability = float(data["probability"])
            confidence = float(data.get("confidence", 0.5))
            reasoning = str(data.get("reasoning", ""))

            if not reasoning.strip():
                logger.warning("llm_empty_reasoning", provider=provider)
                return None

            # Clamp to valid range
            probability = max(0.0, min(1.0, probability))
            confidence = max(0.0, min(1.0, confidence))

            return LLMPrediction(
                probability=probability,
                confidence=confidence,
                reasoning=reasoning,
                provider=provider,
                latency_ms=latency_ms,
                model=model,
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("llm_validation_error", provider=provider, error=str(e))
            return None

    @property
    def stats(self) -> dict[str, Any]:
        """Get agent statistics."""
        return {
            "total_calls": self._total_calls,
            "fallback_calls": self._fallback_calls,
            "remaining_hourly": self._rate_limiter.remaining(),
        }
