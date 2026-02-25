"""Tests for GeminiAgent politics prediction (PM-405).

Verifies that GeminiAgent.predict() works with category="politics",
including rate limiting, API errors, and Gemini→Claude fallback.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from arbo.agents.gemini_agent import GeminiAgent, LLMPrediction, RateLimiter


class TestPoliticsPrediction:
    @pytest.mark.asyncio
    async def test_valid_prediction_returned(self) -> None:
        """Gemini returns valid prediction for politics question."""
        agent = GeminiAgent.__new__(GeminiAgent)
        agent._rate_limiter = RateLimiter(max_calls=60)
        agent._total_calls = 0
        agent._fallback_calls = 0
        agent._gemini_timeout_s = 10.0
        agent._claude_timeout_s = 10.0
        agent._anthropic_client = None

        # Mock Gemini model
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = (
            '{"probability": 0.62, "confidence": 0.75, "reasoning": "Polling data shows..."}'
        )
        mock_model.generate_content.return_value = mock_response
        agent._gemini_model = mock_model
        agent._gemini_model_name = "gemini-2.5-flash"

        prediction = await agent.predict(
            question="Will Trump win the 2028 presidential election?",
            current_price=0.55,
            category="politics",
            volume_24h=50000,
        )

        assert prediction is not None
        assert isinstance(prediction, LLMPrediction)
        assert 0.0 <= prediction.probability <= 1.0
        assert prediction.provider == "gemini"

    @pytest.mark.asyncio
    async def test_probability_in_range(self) -> None:
        """Prediction probability is clamped to [0, 1]."""
        agent = GeminiAgent.__new__(GeminiAgent)
        agent._rate_limiter = RateLimiter(max_calls=60)
        agent._total_calls = 0
        agent._fallback_calls = 0
        agent._gemini_timeout_s = 10.0
        agent._claude_timeout_s = 10.0
        agent._anthropic_client = None

        # Return extreme probability
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = (
            '{"probability": 1.5, "confidence": 0.9, "reasoning": "Very confident"}'
        )
        mock_model.generate_content.return_value = mock_response
        agent._gemini_model = mock_model
        agent._gemini_model_name = "gemini-2.5-flash"

        prediction = await agent.predict(
            question="Will Democrats win the Senate?",
            current_price=0.45,
            category="politics",
        )

        assert prediction is not None
        assert prediction.probability <= 1.0  # Clamped

    @pytest.mark.asyncio
    async def test_rate_limiting(self) -> None:
        """Rate limiter blocks calls when exhausted."""
        agent = GeminiAgent.__new__(GeminiAgent)
        agent._rate_limiter = RateLimiter(max_calls=1, window_seconds=3600)
        agent._total_calls = 0
        agent._fallback_calls = 0
        agent._gemini_timeout_s = 10.0
        agent._claude_timeout_s = 10.0
        agent._anthropic_client = None

        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"probability": 0.60, "confidence": 0.7, "reasoning": "Data..."}'
        mock_model.generate_content.return_value = mock_response
        agent._gemini_model = mock_model
        agent._gemini_model_name = "gemini-2.5-flash"

        # First call succeeds
        p1 = await agent.predict("Election question?", 0.5, category="politics")
        assert p1 is not None

        # Second call rate limited
        p2 = await agent.predict("Another election?", 0.5, category="politics")
        assert p2 is None

    @pytest.mark.asyncio
    async def test_api_error_returns_none(self) -> None:
        """API error returns None gracefully."""
        agent = GeminiAgent.__new__(GeminiAgent)
        agent._rate_limiter = RateLimiter(max_calls=60)
        agent._total_calls = 0
        agent._fallback_calls = 0
        agent._gemini_timeout_s = 10.0
        agent._claude_timeout_s = 10.0
        agent._anthropic_client = None

        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("API Error")
        agent._gemini_model = mock_model
        agent._gemini_model_name = "gemini-2.5-flash"

        prediction = await agent.predict(
            question="Will the bill pass?",
            current_price=0.60,
            category="politics",
        )
        assert prediction is None

    @pytest.mark.asyncio
    async def test_politics_category_in_prompt(self) -> None:
        """Politics category is passed through to the prompt."""
        agent = GeminiAgent.__new__(GeminiAgent)
        agent._rate_limiter = RateLimiter(max_calls=60)
        agent._total_calls = 0
        agent._fallback_calls = 0
        agent._gemini_timeout_s = 10.0
        agent._claude_timeout_s = 10.0
        agent._anthropic_client = None

        call_args: list[str] = []
        mock_model = MagicMock()

        def capture_prompt(prompt: str) -> MagicMock:
            call_args.append(prompt)
            resp = MagicMock()
            resp.text = '{"probability": 0.55, "confidence": 0.6, "reasoning": "Analysis..."}'
            return resp

        mock_model.generate_content.side_effect = capture_prompt
        agent._gemini_model = mock_model
        agent._gemini_model_name = "gemini-2.5-flash"

        await agent.predict(
            question="Will X happen?",
            current_price=0.50,
            category="politics",
        )

        assert len(call_args) == 1
        assert "politics" in call_args[0]

    @pytest.mark.asyncio
    async def test_gemini_to_claude_fallback(self) -> None:
        """Falls back to Claude when Gemini fails."""
        agent = GeminiAgent.__new__(GeminiAgent)
        agent._rate_limiter = RateLimiter(max_calls=60)
        agent._total_calls = 0
        agent._fallback_calls = 0
        agent._gemini_timeout_s = 10.0
        agent._claude_timeout_s = 10.0

        # Gemini fails
        mock_gemini = MagicMock()
        mock_gemini.generate_content.side_effect = Exception("Gemini down")
        agent._gemini_model = mock_gemini
        agent._gemini_model_name = "gemini-2.5-flash"

        # Claude succeeds
        mock_claude = AsyncMock()
        mock_block = MagicMock()
        mock_block.text = '{"probability": 0.58, "confidence": 0.65, "reasoning": "Based on..."}'
        mock_block.type = "text"
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.stop_reason = "end_turn"
        mock_claude.messages.create.return_value = mock_response
        agent._anthropic_client = mock_claude
        agent._claude_model_name = "claude-haiku-4-5-20251001"

        prediction = await agent.predict(
            question="Will Y happen?",
            current_price=0.50,
            category="politics",
        )

        assert prediction is not None
        assert prediction.provider == "claude"
        assert agent._fallback_calls == 1
