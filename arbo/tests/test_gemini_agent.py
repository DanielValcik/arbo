"""Tests for PM-104: Gemini LLM Probability Agent.

Tests verify:
1. RateLimiter: allows, blocks, replenishes
2. Prompt building: includes question + context
3. Predict: valid response, Gemini fail→Claude, both fail→None,
   rate limited→None, invalid JSON→fallback, clamp probability,
   empty reasoning→fallback
4. Initialize: no key logs warning

Acceptance: valid JSON with probability [0,1], non-empty reasoning, <5s response.
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arbo.agents.gemini_agent import GeminiAgent, LLMPrediction, RateLimiter

# ================================================================
# RateLimiter
# ================================================================


class TestRateLimiter:
    """Sliding window rate limiter."""

    def test_allows_within_limit(self) -> None:
        limiter = RateLimiter(max_calls=3, window_seconds=3600)
        assert limiter.allow() is True
        assert limiter.allow() is True
        assert limiter.allow() is True
        assert limiter.remaining() == 0

    def test_blocks_over_limit(self) -> None:
        limiter = RateLimiter(max_calls=2, window_seconds=3600)
        assert limiter.allow() is True
        assert limiter.allow() is True
        assert limiter.allow() is False  # Blocked
        assert limiter.remaining() == 0

    def test_replenishes_after_window(self) -> None:
        limiter = RateLimiter(max_calls=1, window_seconds=1)
        assert limiter.allow() is True
        assert limiter.allow() is False

        # Simulate time passing beyond window
        limiter._timestamps[0] = time.monotonic() - 2  # Expired
        assert limiter.allow() is True


# ================================================================
# Prompt building
# ================================================================


class TestPrompt:
    """Prompt includes question and context."""

    @patch("arbo.agents.gemini_agent.get_config")
    def test_prompt_includes_context(self, mock_config: MagicMock) -> None:
        mock_config.return_value = _mock_config()
        agent = GeminiAgent()

        prompt = agent._build_prompt(
            question="Will Arsenal win the Premier League?",
            current_price=0.35,
            category="soccer",
            volume_24h=50000.0,
        )

        assert "Arsenal" in prompt
        assert "Premier League" in prompt
        assert "0.3500" in prompt
        assert "soccer" in prompt
        assert "$50,000" in prompt


# ================================================================
# Predict
# ================================================================


class TestPredict:
    """GeminiAgent.predict() with mocked LLM calls."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up agent with mocked config."""
        with patch("arbo.agents.gemini_agent.get_config") as mock_cfg:
            mock_cfg.return_value = _mock_config()
            self.agent = GeminiAgent()

    async def test_valid_gemini_response(self) -> None:
        """Gemini returns valid JSON → LLMPrediction."""
        mock_model = MagicMock()
        response = MagicMock()
        response.text = json.dumps(
            {"probability": 0.65, "confidence": 0.8, "reasoning": "Strong form this season."}
        )
        mock_model.generate_content.return_value = response
        self.agent._gemini_model = mock_model

        result = await self.agent.predict("Will Arsenal win?", 0.35, "soccer", 5000.0)

        assert result is not None
        assert isinstance(result, LLMPrediction)
        assert result.probability == 0.65
        assert result.confidence == 0.8
        assert result.provider == "gemini"
        assert result.reasoning == "Strong form this season."

    async def test_gemini_fails_fallback_to_claude(self) -> None:
        """Gemini error → falls back to Claude."""
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("Gemini down")
        self.agent._gemini_model = mock_model

        mock_claude = AsyncMock()
        claude_response = MagicMock()
        claude_response.content = [
            MagicMock(
                text=json.dumps(
                    {"probability": 0.70, "confidence": 0.6, "reasoning": "Fallback analysis."}
                )
            )
        ]
        mock_claude.messages.create.return_value = claude_response
        self.agent._anthropic_client = mock_claude

        result = await self.agent.predict("Will Arsenal win?", 0.35)

        assert result is not None
        assert result.provider == "claude"
        assert result.probability == 0.70
        assert self.agent._fallback_calls == 1

    async def test_both_fail_returns_none(self) -> None:
        """Both Gemini + Claude fail → None."""
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("Gemini down")
        self.agent._gemini_model = mock_model

        mock_claude = AsyncMock()
        mock_claude.messages.create.side_effect = Exception("Claude down")
        self.agent._anthropic_client = mock_claude

        result = await self.agent.predict("Will Arsenal win?", 0.35)
        assert result is None

    async def test_rate_limited_returns_none(self) -> None:
        """Rate limiter exhausted → None without calling LLM."""
        mock_model = MagicMock()
        self.agent._gemini_model = mock_model
        self.agent._rate_limiter = RateLimiter(max_calls=0)

        result = await self.agent.predict("Question?", 0.5)

        assert result is None
        mock_model.generate_content.assert_not_called()

    async def test_invalid_json_fallback(self) -> None:
        """Gemini returns non-JSON → falls back to Claude."""
        mock_model = MagicMock()
        response = MagicMock()
        response.text = "not valid json"
        mock_model.generate_content.return_value = response
        self.agent._gemini_model = mock_model

        mock_claude = AsyncMock()
        claude_response = MagicMock()
        claude_response.content = [
            MagicMock(
                text=json.dumps({"probability": 0.55, "confidence": 0.5, "reasoning": "Backup."})
            )
        ]
        mock_claude.messages.create.return_value = claude_response
        self.agent._anthropic_client = mock_claude

        result = await self.agent.predict("Question?", 0.5)
        assert result is not None
        assert result.provider == "claude"

    async def test_clamp_probability(self) -> None:
        """Probability outside [0,1] gets clamped."""
        mock_model = MagicMock()
        response = MagicMock()
        response.text = json.dumps(
            {"probability": 1.5, "confidence": -0.3, "reasoning": "Over-confident model."}
        )
        mock_model.generate_content.return_value = response
        self.agent._gemini_model = mock_model

        result = await self.agent.predict("Question?", 0.5)

        assert result is not None
        assert result.probability == 1.0  # Clamped from 1.5
        assert result.confidence == 0.0  # Clamped from -0.3

    async def test_empty_reasoning_returns_none(self) -> None:
        """Empty reasoning string → rejected, falls back."""
        mock_model = MagicMock()
        response = MagicMock()
        response.text = json.dumps({"probability": 0.5, "confidence": 0.5, "reasoning": ""})
        mock_model.generate_content.return_value = response
        self.agent._gemini_model = mock_model

        # No Claude fallback configured
        self.agent._anthropic_client = None

        result = await self.agent.predict("Question?", 0.5)
        assert result is None


# ================================================================
# Initialize
# ================================================================


class TestInitialize:
    """GeminiAgent.initialize() logging."""

    @patch("arbo.agents.gemini_agent.get_config")
    async def test_no_keys_logs_warning(self, mock_config: MagicMock) -> None:
        """Missing API keys → warnings logged, no crash."""
        config = _mock_config()
        config.gemini_api_key = ""
        config.anthropic_api_key = ""
        mock_config.return_value = config

        agent = GeminiAgent()
        await agent.initialize()

        assert agent._gemini_model is None
        assert agent._anthropic_client is None


# ================================================================
# Stats
# ================================================================


class TestStats:
    """Agent statistics tracking."""

    @patch("arbo.agents.gemini_agent.get_config")
    async def test_stats_track_calls(self, mock_config: MagicMock) -> None:
        mock_config.return_value = _mock_config()
        agent = GeminiAgent()

        mock_model = MagicMock()
        response = MagicMock()
        response.text = json.dumps({"probability": 0.5, "confidence": 0.5, "reasoning": "Test."})
        mock_model.generate_content.return_value = response
        agent._gemini_model = mock_model

        await agent.predict("Q?", 0.5)

        stats = agent.stats
        assert stats["total_calls"] == 1
        assert stats["fallback_calls"] == 0
        assert stats["remaining_hourly"] == 59  # 60 - 1


# ================================================================
# Helpers
# ================================================================


def _mock_config() -> MagicMock:
    """Create a mock ArboConfig for testing."""
    config = MagicMock()
    config.llm.gemini_model = "gemini-2.0-flash"
    config.llm.claude_model = "claude-haiku-4-5-20251001"
    config.llm.max_calls_per_hour = 60
    config.gemini_api_key = "test-gemini-key"
    config.anthropic_api_key = "test-anthropic-key"
    return config
