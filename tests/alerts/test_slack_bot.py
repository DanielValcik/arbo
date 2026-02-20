"""Tests for Slack bot slash commands and alerts."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.arb_scanner import ArbOpportunity
from src.alerts.slack_bot import ArboSlackBot


@pytest.fixture
def slack_bot() -> ArboSlackBot:
    """Create a bot instance with mock tokens."""
    with patch("src.alerts.slack_bot.AsyncApp") as mock_app_cls:
        mock_app = MagicMock()
        mock_app.client = AsyncMock()
        mock_app.command = MagicMock(return_value=lambda f: f)
        mock_app_cls.return_value = mock_app

        bot = ArboSlackBot(
            bot_token="xoxb-test",
            app_token="xapp-test",
            channel_id="C12345",
        )
        # Re-attach mock app since __init__ creates real app before our mock
        bot._app = mock_app
        return bot


def _arb_opp() -> ArbOpportunity:
    return ArbOpportunity(
        event_name="Liverpool vs Arsenal",
        market_type="h2h",
        selection="Liverpool",
        back_source="bet365",
        back_odds=Decimal("2.30"),
        lay_source="matchbook",
        lay_odds=Decimal("2.14"),
        edge=Decimal("0.05"),
        commission=Decimal("0.04"),
    )


class TestSlashCommands:
    async def test_status_command_structure(self, slack_bot: ArboSlackBot) -> None:
        """Status command should be registered."""
        # Verify the command decorator was called for /status
        slack_bot._app.command.assert_any_call("/status")

    async def test_pnl_command_registered(self, slack_bot: ArboSlackBot) -> None:
        slack_bot._app.command.assert_any_call("/pnl")

    async def test_kill_command_registered(self, slack_bot: ArboSlackBot) -> None:
        slack_bot._app.command.assert_any_call("/kill")

    async def test_paper_command_registered(self, slack_bot: ArboSlackBot) -> None:
        slack_bot._app.command.assert_any_call("/paper")

    async def test_live_command_registered(self, slack_bot: ArboSlackBot) -> None:
        slack_bot._app.command.assert_any_call("/live")


class TestAlerts:
    async def test_send_arb_alert(self, slack_bot: ArboSlackBot) -> None:
        """Arb alert should post to channel with Block Kit."""
        await slack_bot.send_arb_alert(_arb_opp())

        slack_bot._app.client.chat_postMessage.assert_called_once()
        call_kwargs = slack_bot._app.client.chat_postMessage.call_args[1]

        assert call_kwargs["channel"] == "C12345"
        assert "blocks" in call_kwargs
        assert len(call_kwargs["blocks"]) >= 2
        assert "text" in call_kwargs  # Fallback text

    async def test_arb_alert_contains_edge(self, slack_bot: ArboSlackBot) -> None:
        """Alert text should mention the edge percentage."""
        await slack_bot.send_arb_alert(_arb_opp())

        call_kwargs = slack_bot._app.client.chat_postMessage.call_args[1]
        assert "5.0%" in call_kwargs["text"]

    async def test_send_daily_pnl(self, slack_bot: ArboSlackBot) -> None:
        pnl = {
            "date": "2026-03-01",
            "num_bets": 5,
            "num_wins": 4,
            "net_pnl": Decimal("25.50"),
            "roi_pct": Decimal("1.27"),
            "bankroll_end": Decimal("2025.50"),
        }
        await slack_bot.send_daily_pnl(pnl)

        slack_bot._app.client.chat_postMessage.assert_called_once()
        call_kwargs = slack_bot._app.client.chat_postMessage.call_args[1]
        assert call_kwargs["channel"] == "C12345"

    async def test_send_error_alert(self, slack_bot: ArboSlackBot) -> None:
        await slack_bot.send_error_alert("Matchbook unreachable after 3 retries")

        slack_bot._app.client.chat_postMessage.assert_called_once()
        call_kwargs = slack_bot._app.client.chat_postMessage.call_args[1]
        assert "Matchbook unreachable" in call_kwargs["text"]


class TestBotState:
    def test_initial_mode_is_paper(self, slack_bot: ArboSlackBot) -> None:
        assert slack_bot._mode == "paper"

    def test_kill_switch_off_initially(self, slack_bot: ArboSlackBot) -> None:
        assert slack_bot._kill_switch is False

    def test_app_property(self, slack_bot: ArboSlackBot) -> None:
        assert slack_bot.app is not None
