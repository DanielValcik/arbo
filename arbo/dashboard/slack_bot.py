"""Slack bot for Arbo trading alerts and commands (Socket Mode).

Supports @mention commands (@arbo status/pnl/kill) and slash commands
(/status, /pnl, /kill). Methods for sending alerts, daily/weekly reports
via Block Kit messages.

Uses dependency injection — no imports of orchestrator or paper engine.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from arbo.utils.logger import get_logger

logger = get_logger("slack_bot")

# Type aliases for injected callbacks
StatusCallback = Callable[[], Awaitable[dict[str, Any]]]
PnlCallback = Callable[[], Awaitable[dict[str, Any]]]
ShutdownCallback = Callable[[], Awaitable[None]]


class SlackBot:
    """Async Slack bot using Socket Mode (no public URL needed).

    Args:
        bot_token: Slack bot token (xoxb-...).
        app_token: Slack app token (xapp-...).
        channel_id: Default channel for alerts/reports.
        get_status_fn: Async callback returning system status dict.
        get_pnl_fn: Async callback returning P&L dict.
        shutdown_fn: Async callback to trigger emergency shutdown.
    """

    def __init__(
        self,
        bot_token: str,
        app_token: str,
        channel_id: str,
        get_status_fn: StatusCallback,
        get_pnl_fn: PnlCallback,
        shutdown_fn: ShutdownCallback,
    ) -> None:
        self._bot_token = bot_token
        self._app_token = app_token
        self._channel_id = channel_id
        self._get_status_fn = get_status_fn
        self._get_pnl_fn = get_pnl_fn
        self._shutdown_fn = shutdown_fn
        self._app: Any = None
        self._handler: Any = None

    async def start(self) -> None:
        """Start the Slack bot (Socket Mode). Blocks until closed."""
        try:
            from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
            from slack_bolt.async_app import AsyncApp
        except ImportError:
            logger.error("slack_bolt_not_installed", hint="pip install slack-bolt")
            return

        self._app = AsyncApp(token=self._bot_token)
        self._register_commands()

        self._handler = AsyncSocketModeHandler(self._app, self._app_token)
        logger.info("slack_bot_starting")
        await self._handler.start_async()

    async def close(self) -> None:
        """Stop the Slack bot gracefully."""
        if self._handler is not None:
            try:
                await self._handler.close_async()
            except Exception as e:
                logger.warning("slack_bot_close_error", error=str(e))
        logger.info("slack_bot_stopped")

    def _register_commands(self) -> None:
        """Register slash commands and @mention handler on the Slack app."""
        if self._app is None:
            return

        @self._app.command("/status")
        async def handle_status(ack: Any, respond: Any) -> None:
            await ack()
            try:
                status = await self._get_status_fn()
                blocks = self._format_status_blocks(status)
                await respond(blocks=blocks)
            except Exception as e:
                logger.error("slack_status_error", error=str(e))
                await respond(text=f"Error: {e}")

        @self._app.command("/pnl")
        async def handle_pnl(ack: Any, respond: Any) -> None:
            await ack()
            try:
                pnl = await self._get_pnl_fn()
                blocks = self._format_pnl_blocks(pnl)
                await respond(blocks=blocks)
            except Exception as e:
                logger.error("slack_pnl_error", error=str(e))
                await respond(text=f"Error: {e}")

        @self._app.command("/kill")
        async def handle_kill(ack: Any, respond: Any) -> None:
            await ack()
            await respond(text="Emergency shutdown initiated...")
            logger.critical("slack_kill_command_received")
            try:
                await self._shutdown_fn()
            except Exception as e:
                logger.error("slack_kill_error", error=str(e))

        @self._app.event("app_mention")
        async def handle_mention(event: dict[str, Any], say: Any) -> None:
            """Handle @arbo mentions: '@arbo status', '@arbo pnl', '@arbo kill'."""
            import re

            raw_text = event.get("text", "")
            # Strip the mention tag (<@BOTID>) first, then lowercase
            command = re.sub(r"<@[A-Za-z0-9]+>", "", raw_text).strip().lower()
            logger.info("slack_mention_received", command=command)

            try:
                if command in ("status", "s"):
                    status = await self._get_status_fn()
                    blocks = self._format_status_blocks(status)
                    await say(blocks=blocks, text="System Status")
                elif command in ("pnl", "p"):
                    pnl = await self._get_pnl_fn()
                    blocks = self._format_pnl_blocks(pnl)
                    await say(blocks=blocks, text="P&L Summary")
                elif command == "kill":
                    await say(text="Emergency shutdown initiated...")
                    logger.critical("slack_kill_command_received")
                    await self._shutdown_fn()
                elif command in ("help", "h", ""):
                    await say(
                        text="*Arbo Commands:*\n"
                        "• `@arbo status` — system status\n"
                        "• `@arbo pnl` — P&L summary\n"
                        "• `@arbo kill` — emergency shutdown"
                    )
                else:
                    await say(text=f"Unknown command: `{command}`. Try `@arbo help`.")
            except Exception as e:
                logger.error("slack_mention_error", error=str(e), command=command)
                await say(text=f"Error: {e}")

    async def send_message(self, text: str) -> None:
        """Send a plain text message to the default channel."""
        if self._app is None:
            logger.warning("slack_not_connected", action="send_message")
            return
        try:
            await self._app.client.chat_postMessage(
                channel=self._channel_id,
                text=text,
            )
        except Exception as e:
            logger.error("slack_send_error", error=str(e))

    async def send_blocks(self, blocks: list[dict[str, Any]], text: str = "") -> None:
        """Send Block Kit blocks to the default channel."""
        if self._app is None:
            logger.warning("slack_not_connected", action="send_blocks")
            return
        try:
            await self._app.client.chat_postMessage(
                channel=self._channel_id,
                blocks=blocks,
                text=text or "Arbo Report",
            )
        except Exception as e:
            logger.error("slack_send_blocks_error", error=str(e))

    async def send_daily_report(self, blocks: list[dict[str, Any]]) -> None:
        """Send a daily report (Block Kit) to Slack."""
        await self.send_blocks(blocks, text="Daily Report")

    async def send_weekly_report(self, blocks: list[dict[str, Any]]) -> None:
        """Send a weekly report (Block Kit) to Slack."""
        await self.send_blocks(blocks, text="Weekly Report")

    async def send_alert(self, text: str) -> None:
        """Send an alert message to the default channel."""
        await self.send_message(f"*ALERT:* {text}")

    @staticmethod
    def _format_status_blocks(status: dict[str, Any]) -> list[dict[str, Any]]:
        """Format system status as Block Kit blocks."""
        mode = status.get("mode", "unknown")
        uptime = status.get("uptime_s", 0)
        layers_active = status.get("layers_active", 0)
        layers_total = status.get("layers_total", 0)
        balance = status.get("balance", "?")
        positions = status.get("open_positions", 0)

        return [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "Arbo System Status"},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Mode:* {mode}"},
                    {
                        "type": "mrkdwn",
                        "text": f"*Uptime:* {uptime // 3600}h {(uptime % 3600) // 60}m",
                    },
                    {"type": "mrkdwn", "text": f"*Layers:* {layers_active}/{layers_total}"},
                    {"type": "mrkdwn", "text": f"*Balance:* ${balance}"},
                    {"type": "mrkdwn", "text": f"*Open Positions:* {positions}"},
                ],
            },
        ]

    @staticmethod
    def _format_pnl_blocks(pnl: dict[str, Any]) -> list[dict[str, Any]]:
        """Format P&L data as Block Kit blocks."""
        total_pnl = pnl.get("total_pnl", "0")
        roi = pnl.get("roi_pct", "0")
        trades = pnl.get("total_trades", 0)
        wins = pnl.get("wins", 0)
        win_rate = pnl.get("win_rate", 0)

        layer_pnl = pnl.get("per_layer_pnl", {})
        layer_lines = []
        for layer, lp in sorted(layer_pnl.items(), key=lambda x: int(x[0])):
            sign = "+" if float(str(lp)) >= 0 else ""
            layer_lines.append(f"  L{layer}: {sign}${lp}")
        layer_text = "\n".join(layer_lines) if layer_lines else "  No data"

        return [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "P&L Summary"},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Total P&L:* ${total_pnl}"},
                    {"type": "mrkdwn", "text": f"*ROI:* {roi}%"},
                    {"type": "mrkdwn", "text": f"*Trades:* {trades} ({wins}W)"},
                    {"type": "mrkdwn", "text": f"*Win Rate:* {float(str(win_rate)) * 100:.1f}%"},
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Per-Layer P&L:*\n```\n{layer_text}\n```",
                },
            },
        ]
