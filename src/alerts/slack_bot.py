"""Slack bot with slash commands and proactive alerts.

Uses slack-bolt AsyncApp with Socket Mode (no public URL needed).
Runs alongside main asyncio loop via asyncio.create_task().
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from slack_bolt.async_app import AsyncApp

if TYPE_CHECKING:
    from slack_sdk.web.async_client import AsyncWebClient

from src.agents.arb_scanner import ArbOpportunity  # noqa: TC001
from src.utils.logger import get_logger

log = get_logger("slack_bot")


class ArboSlackBot:
    """Slack bot for Arbo system control and alerts."""

    def __init__(
        self,
        bot_token: str,
        app_token: str,
        channel_id: str,
    ) -> None:
        self._channel_id = channel_id
        self._app_token = app_token
        self._app = AsyncApp(token=bot_token)
        self._start_time = time.monotonic()

        # State references (set by main.py after init)
        self._mode: str = "paper"
        self._kill_switch: bool = False
        self._daily_get_count: int = 0
        self._odds_api_quota: int | None = None
        self._open_bets_count: int = 0
        self._last_poll_time: str = "never"

        # Callbacks for commands that need to mutate state
        self._on_kill: object | None = None
        self._on_mode_change: object | None = None

        self._register_commands()

    @property
    def app(self) -> AsyncApp:
        """Expose the slack-bolt app for testing."""
        return self._app

    def _register_commands(self) -> None:
        """Register all slash command handlers."""

        @self._app.command("/status")
        async def handle_status(ack, respond) -> None:  # type: ignore[no-untyped-def]
            await ack()
            uptime = int(time.monotonic() - self._start_time)
            hours, remainder = divmod(uptime, 3600)
            minutes, seconds = divmod(remainder, 60)

            blocks = [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "Arbo System Status"},
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Mode:* `{self._mode}`"},
                        {
                            "type": "mrkdwn",
                            "text": f"*Kill Switch:* {'ON' if self._kill_switch else 'OFF'}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Uptime:* {hours}h {minutes}m {seconds}s",
                        },
                        {"type": "mrkdwn", "text": f"*Open Bets:* {self._open_bets_count}"},
                        {"type": "mrkdwn", "text": f"*Last Poll:* {self._last_poll_time}"},
                        {
                            "type": "mrkdwn",
                            "text": f"*Matchbook GETs:* {self._daily_get_count}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Odds API Quota:* {self._odds_api_quota or 'N/A'}",
                        },
                    ],
                },
            ]
            await respond(blocks=blocks, response_type="ephemeral")

        @self._app.command("/pnl")
        async def handle_pnl(ack, respond) -> None:  # type: ignore[no-untyped-def]
            await ack()
            await respond(
                text="P&L data not yet available. Paper trading must run first.",
                response_type="ephemeral",
            )

        @self._app.command("/kill")
        async def handle_kill(ack, respond) -> None:  # type: ignore[no-untyped-def]
            await ack()
            self._kill_switch = True
            if callable(self._on_kill):
                self._on_kill()
            await respond(
                text="KILL SWITCH ACTIVATED. All operations stopped.",
                response_type="in_channel",
            )
            log.warning("kill_switch_activated_via_slack")

        @self._app.command("/paper")
        async def handle_paper(ack, respond) -> None:  # type: ignore[no-untyped-def]
            await ack()
            self._mode = "paper"
            if callable(self._on_mode_change):
                self._on_mode_change("paper")
            await respond(
                text="Switched to PAPER mode.",
                response_type="ephemeral",
            )

        @self._app.command("/live")
        async def handle_live(ack, respond) -> None:  # type: ignore[no-untyped-def]
            await ack()
            # Sprint 4: MIN_PAPER_WEEKS = 4 check
            await respond(
                text="Live mode requires 4 weeks of paper trading. Not available yet (Sprint 4).",
                response_type="ephemeral",
            )

    async def start(self) -> None:
        """Start the Socket Mode handler. Call via asyncio.create_task()."""
        from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

        handler = AsyncSocketModeHandler(self._app, self._app_token)
        log.info("slack_bot_starting")
        await handler.start_async()

    async def send_arb_alert(self, opp: ArbOpportunity) -> None:
        """Send an arb opportunity alert to the configured channel."""
        client: AsyncWebClient = self._app.client  # type: ignore[assignment]
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "Arb Opportunity Found"},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Event:* {opp.event_name}"},
                    {"type": "mrkdwn", "text": f"*Market:* {opp.market_type}"},
                    {"type": "mrkdwn", "text": f"*Selection:* {opp.selection}"},
                    {"type": "mrkdwn", "text": f"*Back:* {opp.back_odds} @ {opp.back_source}"},
                    {"type": "mrkdwn", "text": f"*Lay:* {opp.lay_odds} @ {opp.lay_source}"},
                    {"type": "mrkdwn", "text": f"*Edge:* {float(opp.edge) * 100:.2f}%"},
                ],
            },
        ]
        if opp.recommended_stake:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Suggested Stake:* {opp.recommended_stake} EUR",
                    },
                }
            )

        await client.chat_postMessage(
            channel=self._channel_id,
            blocks=blocks,
            text=f"Arb: {opp.event_name} | {opp.selection} | Edge {float(opp.edge) * 100:.1f}%",
        )
        log.info("arb_alert_sent", match=opp.event_name, edge=str(opp.edge))

    async def send_daily_pnl(self, pnl: dict) -> None:
        """Send daily P&L summary to the configured channel."""
        client: AsyncWebClient = self._app.client  # type: ignore[assignment]
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "Daily P&L Report"},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Date:* {pnl.get('date', 'N/A')}"},
                    {"type": "mrkdwn", "text": f"*Bets:* {pnl.get('num_bets', 0)}"},
                    {"type": "mrkdwn", "text": f"*Wins:* {pnl.get('num_wins', 0)}"},
                    {"type": "mrkdwn", "text": f"*Net P&L:* {pnl.get('net_pnl', 0)} EUR"},
                    {"type": "mrkdwn", "text": f"*ROI:* {pnl.get('roi_pct', 0)}%"},
                    {"type": "mrkdwn", "text": f"*Bankroll:* {pnl.get('bankroll_end', 0)} EUR"},
                ],
            },
        ]
        await client.chat_postMessage(
            channel=self._channel_id,
            blocks=blocks,
            text=f"Daily P&L: {pnl.get('net_pnl', 0)} EUR | ROI: {pnl.get('roi_pct', 0)}%",
        )

    async def send_error_alert(self, error: str) -> None:
        """Send error alert to the configured channel."""
        client: AsyncWebClient = self._app.client  # type: ignore[assignment]
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "Error Alert"},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"```{error}```"},
            },
        ]
        await client.chat_postMessage(
            channel=self._channel_id,
            blocks=blocks,
            text=f"Error: {error[:100]}",
        )
        log.error("error_alert_sent", error=error[:200])
