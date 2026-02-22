"""Tests for Slack bot (dashboard/slack_bot.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from arbo.dashboard.slack_bot import SlackBot


@pytest.fixture
def status_fn() -> AsyncMock:
    return AsyncMock(
        return_value={
            "mode": "paper",
            "uptime_s": 3600,
            "layers_active": 8,
            "layers_total": 11,
            "balance": "1950.00",
            "open_positions": 3,
        }
    )


@pytest.fixture
def pnl_fn() -> AsyncMock:
    return AsyncMock(
        return_value={
            "total_pnl": "-50.00",
            "roi_pct": "-2.50",
            "total_trades": 10,
            "wins": 6,
            "win_rate": 0.6,
            "per_layer_pnl": {2: "30.00", 4: "-80.00"},
        }
    )


@pytest.fixture
def shutdown_fn() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def bot(status_fn: AsyncMock, pnl_fn: AsyncMock, shutdown_fn: AsyncMock) -> SlackBot:
    return SlackBot(
        bot_token="xoxb-test",
        app_token="xapp-test",
        channel_id="C123",
        get_status_fn=status_fn,
        get_pnl_fn=pnl_fn,
        shutdown_fn=shutdown_fn,
    )


class TestSlackBotCommands:
    @staticmethod
    def _setup_bot_with_commands(bot: SlackBot) -> dict[str, object]:
        """Wire up a mock app that captures command handlers."""
        mock_app = MagicMock()
        commands: dict[str, object] = {}

        def capture_command(name: str):  # type: ignore[no-untyped-def]
            def decorator(fn):  # type: ignore[no-untyped-def]
                commands[name] = fn
                return fn

            return decorator

        mock_app.command = capture_command
        mock_app.event = lambda name: lambda fn: fn  # no-op for event
        bot._app = mock_app
        bot._register_commands()
        return commands

    @pytest.mark.asyncio
    async def test_status_command_calls_callback(self, bot: SlackBot, status_fn: AsyncMock) -> None:
        """Register commands and verify /status invokes the status callback."""
        commands = self._setup_bot_with_commands(bot)

        ack = AsyncMock()
        respond = AsyncMock()
        await commands["/status"](ack=ack, respond=respond)

        ack.assert_awaited_once()
        status_fn.assert_awaited_once()
        respond.assert_awaited_once()
        call_kwargs = respond.call_args
        assert "blocks" in call_kwargs.kwargs

    @pytest.mark.asyncio
    async def test_pnl_command_calls_callback(self, bot: SlackBot, pnl_fn: AsyncMock) -> None:
        """Verify /pnl invokes the pnl callback."""
        commands = self._setup_bot_with_commands(bot)

        ack = AsyncMock()
        respond = AsyncMock()
        await commands["/pnl"](ack=ack, respond=respond)

        ack.assert_awaited_once()
        pnl_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_kill_command_triggers_shutdown(
        self, bot: SlackBot, shutdown_fn: AsyncMock
    ) -> None:
        """Verify /kill triggers the shutdown callback."""
        commands = self._setup_bot_with_commands(bot)

        ack = AsyncMock()
        respond = AsyncMock()
        await commands["/kill"](ack=ack, respond=respond)

        ack.assert_awaited_once()
        shutdown_fn.assert_awaited_once()


class TestSlackBotMentions:
    """Tests for @arbo mention handling."""

    @staticmethod
    def _setup_bot_with_events(bot: SlackBot) -> dict[str, object]:
        """Wire up a mock app that captures event and command handlers."""
        mock_app = MagicMock()
        commands: dict[str, object] = {}
        events: dict[str, object] = {}

        def capture_command(name: str):  # type: ignore[no-untyped-def]
            def decorator(fn):  # type: ignore[no-untyped-def]
                commands[name] = fn
                return fn

            return decorator

        def capture_event(name: str):  # type: ignore[no-untyped-def]
            def decorator(fn):  # type: ignore[no-untyped-def]
                events[name] = fn
                return fn

            return decorator

        mock_app.command = capture_command
        mock_app.event = capture_event
        bot._app = mock_app
        bot._register_commands()
        return events

    @pytest.mark.asyncio
    async def test_mention_status(self, bot: SlackBot, status_fn: AsyncMock) -> None:
        events = self._setup_bot_with_events(bot)
        say = AsyncMock()
        await events["app_mention"](event={"text": "<@U123> status"}, say=say)
        status_fn.assert_awaited_once()
        say.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_mention_pnl(self, bot: SlackBot, pnl_fn: AsyncMock) -> None:
        events = self._setup_bot_with_events(bot)
        say = AsyncMock()
        await events["app_mention"](event={"text": "<@U123> pnl"}, say=say)
        pnl_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_mention_kill(self, bot: SlackBot, shutdown_fn: AsyncMock) -> None:
        events = self._setup_bot_with_events(bot)
        say = AsyncMock()
        await events["app_mention"](event={"text": "<@U123> kill"}, say=say)
        shutdown_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_mention_help(self, bot: SlackBot) -> None:
        events = self._setup_bot_with_events(bot)
        say = AsyncMock()
        await events["app_mention"](event={"text": "<@U123> help"}, say=say)
        say.assert_awaited_once()
        assert "Commands" in say.call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_mention_unknown_command(self, bot: SlackBot) -> None:
        events = self._setup_bot_with_events(bot)
        say = AsyncMock()
        await events["app_mention"](event={"text": "<@U123> foobar"}, say=say)
        assert "Unknown" in say.call_args.kwargs["text"]


class TestSlackBotMessaging:
    @pytest.mark.asyncio
    async def test_send_message_posts_to_channel(self, bot: SlackBot) -> None:
        """send_message should call chat_postMessage with text."""
        mock_app = MagicMock()
        mock_app.client.chat_postMessage = AsyncMock()
        bot._app = mock_app

        await bot.send_message("test alert")

        mock_app.client.chat_postMessage.assert_awaited_once_with(
            channel="C123",
            text="test alert",
        )

    @pytest.mark.asyncio
    async def test_send_daily_report_sends_blocks(self, bot: SlackBot) -> None:
        """send_daily_report should post Block Kit blocks."""
        mock_app = MagicMock()
        mock_app.client.chat_postMessage = AsyncMock()
        bot._app = mock_app

        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "test"}}]
        await bot.send_daily_report(blocks)

        mock_app.client.chat_postMessage.assert_awaited_once()
        call_kwargs = mock_app.client.chat_postMessage.call_args.kwargs
        assert call_kwargs["blocks"] == blocks

    @pytest.mark.asyncio
    async def test_send_message_without_app_logs_warning(self, bot: SlackBot) -> None:
        """send_message before start() should log warning, not crash."""
        bot._app = None
        await bot.send_message("should not crash")  # Should not raise

    @pytest.mark.asyncio
    async def test_send_alert_prefixes_text(self, bot: SlackBot) -> None:
        """send_alert should prefix with *ALERT:*."""
        mock_app = MagicMock()
        mock_app.client.chat_postMessage = AsyncMock()
        bot._app = mock_app

        await bot.send_alert("something bad")

        call_kwargs = mock_app.client.chat_postMessage.call_args.kwargs
        assert "*ALERT:*" in call_kwargs["text"]


class TestSlackBotFormatting:
    def test_format_status_blocks(self) -> None:
        """Status blocks should contain system info."""
        blocks = SlackBot._format_status_blocks(
            {
                "mode": "paper",
                "uptime_s": 7200,
                "layers_active": 8,
                "layers_total": 11,
                "balance": "1950.00",
                "open_positions": 3,
            }
        )
        assert blocks[0]["type"] == "header"
        assert "Status" in blocks[0]["text"]["text"]
        fields_text = str(blocks[1]["fields"])
        assert "paper" in fields_text
        assert "1950.00" in fields_text

    def test_format_pnl_blocks(self) -> None:
        """P&L blocks should contain trade stats."""
        blocks = SlackBot._format_pnl_blocks(
            {
                "total_pnl": "100.00",
                "roi_pct": "5.0",
                "total_trades": 20,
                "wins": 12,
                "win_rate": 0.6,
                "per_layer_pnl": {2: "80.00", 4: "20.00"},
            }
        )
        assert blocks[0]["type"] == "header"
        assert "P&L" in blocks[0]["text"]["text"]
