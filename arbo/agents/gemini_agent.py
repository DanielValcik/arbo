"""Gemini LLM agent for structured queries (RDH-307).

Primary LLM: Gemini 2.5 Flash with JSON response mode.
Fallback: Claude Haiku 4.5 via anthropic SDK.
Rate limited to max_calls_per_hour (default 60).

Provides:
- raw_query(): send arbitrary prompt, get parsed JSON dict
- query_mindshare(): LLM-based mindshare/sentiment estimation (Kaito fallback)
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import deque
from typing import Any

from arbo.config.settings import get_config
from arbo.utils.logger import get_logger

logger = get_logger("gemini_agent")


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
    """LLM agent for structured JSON queries.

    Uses Gemini 2.5 Flash as primary provider with Claude Haiku 4.5 fallback.
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
        """Send a prompt and return parsed JSON dict.

        Used for graph classification, mindshare estimation, and other tasks.
        """
        if not self._rate_limiter.allow():
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

    async def query_mindshare(
        self,
        topic: str,
        market_question: str = "",
        current_price: float | None = None,
    ) -> dict[str, Any] | None:
        """Estimate mindshare/sentiment for a topic using LLM.

        Kaito API fallback — when Kaito is unavailable, use LLM to estimate
        social attention and sentiment for reflexivity trading (Strategy B).

        Args:
            topic: The topic or token name to analyze.
            market_question: Optional Polymarket question for context.
            current_price: Optional current YES price for context.

        Returns:
            Dict with mindshare_score (0-1), sentiment (-1 to +1),
            confidence (0-1), reasoning. None if rate limited or LLM fails.
        """
        prompt = (
            "You are a crypto/prediction market attention analyst. "
            "Estimate the current social media mindshare and sentiment "
            "for the following topic based on your training data.\n\n"
            f"Topic: {topic}\n"
        )
        if market_question:
            prompt += f"Market question: {market_question}\n"
        if current_price is not None:
            prompt += f"Current market price: {current_price:.4f}\n"
        prompt += (
            "\nRespond with JSON:\n"
            '- "mindshare_score": float 0-1 (estimated share of crypto/prediction '
            "market attention)\n"
            '- "sentiment": float -1 to +1 (negative to positive)\n'
            '- "confidence": float 0-1 (how confident in this estimate)\n'
            '- "reasoning": brief explanation\n'
        )
        result = await self.raw_query(prompt)
        if result is None:
            return None

        # Clamp values to valid ranges
        try:
            if "mindshare_score" in result:
                result["mindshare_score"] = max(0.0, min(1.0, float(result["mindshare_score"])))
            if "sentiment" in result:
                result["sentiment"] = max(-1.0, min(1.0, float(result["sentiment"])))
            if "confidence" in result:
                result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))
        except (ValueError, TypeError):
            logger.warning("mindshare_clamp_error", topic=topic[:40])
            return None

        return result

    @property
    def stats(self) -> dict[str, Any]:
        """Get agent statistics."""
        return {
            "total_calls": self._total_calls,
            "fallback_calls": self._fallback_calls,
            "remaining_hourly": self._rate_limiter.remaining(),
        }

    @property
    def rate_limiter(self) -> RateLimiter:
        """Expose rate limiter for external inspection."""
        return self._rate_limiter
