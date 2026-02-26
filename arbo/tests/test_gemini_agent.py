"""Tests for Gemini LLM agent (RDH-307).

Tests verify:
1. RateLimiter: allows, blocks, replenishes
2. raw_query: Gemini primary, Claude fallback, both fail, rate limited
3. query_mindshare: builds prompt, clamps values, handles missing fields
4. Initialize: no key logs warning
5. _extract_json: various LLM response formats

Acceptance: raw_query works, rate limiter works, _extract_json works,
predict() removed, query_mindshare works.
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arbo.agents.gemini_agent import GeminiAgent, RateLimiter, _extract_json

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
# raw_query
# ================================================================


class TestRawQuery:
    """GeminiAgent.raw_query() with mocked LLM calls."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        with patch("arbo.agents.gemini_agent.get_config") as mock_cfg:
            mock_cfg.return_value = _mock_config()
            self.agent = GeminiAgent()

    async def test_gemini_returns_json(self) -> None:
        """Gemini returns valid JSON → parsed dict."""
        mock_model = MagicMock()
        response = MagicMock()
        response.text = json.dumps({"result": "ok", "score": 0.85})
        mock_model.generate_content.return_value = response
        self.agent._gemini_model = mock_model

        result = await self.agent.raw_query("Analyze this topic")

        assert result is not None
        assert result["result"] == "ok"
        assert result["score"] == 0.85

    async def test_gemini_fails_fallback_to_claude(self) -> None:
        """Gemini error → falls back to Claude."""
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("Gemini down")
        self.agent._gemini_model = mock_model

        mock_claude = AsyncMock()
        claude_response = MagicMock()
        claude_response.content = [
            MagicMock(text=json.dumps({"result": "fallback", "score": 0.7}))
        ]
        mock_claude.messages.create.return_value = claude_response
        self.agent._anthropic_client = mock_claude

        result = await self.agent.raw_query("Analyze this")

        assert result is not None
        assert result["result"] == "fallback"
        assert self.agent._fallback_calls == 1

    async def test_both_fail_returns_none(self) -> None:
        """Both Gemini + Claude fail → None."""
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("Gemini down")
        self.agent._gemini_model = mock_model

        mock_claude = AsyncMock()
        mock_claude.messages.create.side_effect = Exception("Claude down")
        self.agent._anthropic_client = mock_claude

        result = await self.agent.raw_query("Analyze this")
        assert result is None

    async def test_rate_limited_returns_none(self) -> None:
        """Rate limiter exhausted → None without calling LLM."""
        mock_model = MagicMock()
        self.agent._gemini_model = mock_model
        self.agent._rate_limiter = RateLimiter(max_calls=0)

        result = await self.agent.raw_query("Question?")

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
            MagicMock(text=json.dumps({"result": "backup"}))
        ]
        mock_claude.messages.create.return_value = claude_response
        self.agent._anthropic_client = mock_claude

        result = await self.agent.raw_query("Question?")
        assert result is not None
        assert result["result"] == "backup"

    async def test_no_providers_returns_none(self) -> None:
        """No Gemini or Claude configured → None."""
        self.agent._gemini_model = None
        self.agent._anthropic_client = None

        result = await self.agent.raw_query("Question?")
        assert result is None


# ================================================================
# query_mindshare
# ================================================================


class TestQueryMindshare:
    """GeminiAgent.query_mindshare() — Kaito LLM fallback."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        with patch("arbo.agents.gemini_agent.get_config") as mock_cfg:
            mock_cfg.return_value = _mock_config()
            self.agent = GeminiAgent()

    async def test_valid_mindshare_response(self) -> None:
        """Returns clamped mindshare data from LLM."""
        mock_model = MagicMock()
        response = MagicMock()
        response.text = json.dumps({
            "mindshare_score": 0.35,
            "sentiment": 0.6,
            "confidence": 0.7,
            "reasoning": "High social buzz around this topic.",
        })
        mock_model.generate_content.return_value = response
        self.agent._gemini_model = mock_model

        result = await self.agent.query_mindshare("Bitcoin ETF approval")

        assert result is not None
        assert result["mindshare_score"] == 0.35
        assert result["sentiment"] == 0.6
        assert result["confidence"] == 0.7
        assert "buzz" in result["reasoning"]

    async def test_clamps_out_of_range_values(self) -> None:
        """Values outside valid ranges are clamped."""
        mock_model = MagicMock()
        response = MagicMock()
        response.text = json.dumps({
            "mindshare_score": 1.5,  # > 1.0
            "sentiment": -2.0,  # < -1.0
            "confidence": 3.0,  # > 1.0
            "reasoning": "Extreme values test.",
        })
        mock_model.generate_content.return_value = response
        self.agent._gemini_model = mock_model

        result = await self.agent.query_mindshare("Test topic")

        assert result is not None
        assert result["mindshare_score"] == 1.0
        assert result["sentiment"] == -1.0
        assert result["confidence"] == 1.0

    async def test_includes_market_question_in_prompt(self) -> None:
        """Market question is included in the prompt sent to LLM."""
        mock_model = MagicMock()
        response = MagicMock()
        response.text = json.dumps({
            "mindshare_score": 0.2,
            "sentiment": 0.0,
            "confidence": 0.5,
            "reasoning": "Moderate attention.",
        })
        mock_model.generate_content.return_value = response
        self.agent._gemini_model = mock_model

        await self.agent.query_mindshare(
            "Trump",
            market_question="Will Trump win?",
            current_price=0.55,
        )

        call_args = mock_model.generate_content.call_args[0][0]
        assert "Trump" in call_args
        assert "Will Trump win?" in call_args
        assert "0.5500" in call_args

    async def test_rate_limited_returns_none(self) -> None:
        """Rate limited → returns None."""
        self.agent._rate_limiter = RateLimiter(max_calls=0)
        result = await self.agent.query_mindshare("Bitcoin")
        assert result is None

    async def test_handles_missing_fields(self) -> None:
        """LLM response missing some fields still works (no clamp crash)."""
        mock_model = MagicMock()
        response = MagicMock()
        response.text = json.dumps({
            "reasoning": "Minimal response.",
        })
        mock_model.generate_content.return_value = response
        self.agent._gemini_model = mock_model

        result = await self.agent.query_mindshare("Topic")

        assert result is not None
        assert result["reasoning"] == "Minimal response."
        assert "mindshare_score" not in result

    async def test_llm_fails_returns_none(self) -> None:
        """Both providers fail → None."""
        self.agent._gemini_model = None
        self.agent._anthropic_client = None

        result = await self.agent.query_mindshare("Bitcoin")
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
        response.text = json.dumps({"result": "ok"})
        mock_model.generate_content.return_value = response
        agent._gemini_model = mock_model

        await agent.raw_query("Test query")

        stats = agent.stats
        assert stats["total_calls"] == 1
        assert stats["fallback_calls"] == 0
        assert stats["remaining_hourly"] == 59  # 60 - 1


# ================================================================
# predict() removed
# ================================================================


class TestPredictRemoved:
    """Verify predict() method no longer exists (RDH-307)."""

    @patch("arbo.agents.gemini_agent.get_config")
    def test_predict_removed(self, mock_config: MagicMock) -> None:
        """predict() method has been removed."""
        mock_config.return_value = _mock_config()
        agent = GeminiAgent()
        assert not hasattr(agent, "predict")

    def test_llm_prediction_removed(self) -> None:
        """LLMPrediction dataclass has been removed from module."""
        import arbo.agents.gemini_agent as mod

        assert not hasattr(mod, "LLMPrediction")


# ================================================================
# JSON extraction
# ================================================================


class TestExtractJson:
    """_extract_json handles various LLM response formats."""

    def test_pure_json(self) -> None:
        data = _extract_json('{"probability": 0.5, "confidence": 0.7, "reasoning": "test"}')
        assert data is not None
        assert data["probability"] == 0.5

    def test_markdown_code_block(self) -> None:
        text = '```json\n{"probability": 0.6, "confidence": 0.8, "reasoning": "analysis"}\n```'
        data = _extract_json(text)
        assert data is not None
        assert data["probability"] == 0.6

    def test_text_before_json(self) -> None:
        text = 'Here is my analysis:\n{"probability": 0.4, "confidence": 0.5, "reasoning": "ok"}'
        data = _extract_json(text)
        assert data is not None
        assert data["probability"] == 0.4

    def test_empty_string(self) -> None:
        assert _extract_json("") is None

    def test_none_like(self) -> None:
        assert _extract_json("   ") is None

    def test_no_json(self) -> None:
        assert _extract_json("This is just text with no JSON at all.") is None


# ================================================================
# Helpers
# ================================================================


def _mock_config() -> MagicMock:
    """Create a mock ArboConfig for testing."""
    config = MagicMock()
    config.llm.gemini_model = "gemini-2.5-flash"
    config.llm.claude_model = "claude-haiku-4-5-20251001"
    config.llm.max_calls_per_hour = 60
    config.gemini_api_key = "test-gemini-key"
    config.anthropic_api_key = "test-anthropic-key"
    return config
