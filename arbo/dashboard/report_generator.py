"""Report generator for daily/weekly summaries (PM-210).

Generates structured reports from paper trading data, exports to CSV,
and formats Slack Block Kit messages for alert delivery.

See brief PM-210 for full specification.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from arbo.utils.logger import get_logger

logger = get_logger("report_generator")


@dataclass
class StrategyDailyStats:
    """Per-strategy daily statistics."""

    trades: int = 0
    wins: int = 0
    pnl: Decimal = Decimal("0")
    deployed: Decimal = Decimal("0")


@dataclass
class DailyReport:
    """Daily trading summary report."""

    date: date
    total_trades: int
    winning_trades: int
    total_pnl: Decimal
    roi_pct: Decimal
    per_layer_pnl: dict[int, Decimal]
    top_signals: list[dict[str, Any]]
    risk_events: list[str]
    per_strategy_pnl: dict[str, StrategyDailyStats] = field(default_factory=dict)

    @property
    def losing_trades(self) -> int:
        """Number of losing trades."""
        return self.total_trades - self.winning_trades

    @property
    def win_rate(self) -> Decimal:
        """Win rate as decimal (0-1)."""
        if self.total_trades == 0:
            return Decimal("0")
        return Decimal(str(self.winning_trades)) / Decimal(str(self.total_trades))


@dataclass
class WeeklyReport:
    """Weekly trading summary report."""

    week_start: date
    week_end: date
    total_trades: int
    winning_trades: int
    total_pnl: Decimal
    roi_pct: Decimal
    avg_daily_pnl: Decimal
    best_day: DailyReport
    worst_day: DailyReport
    per_layer_pnl: dict[int, Decimal]
    per_layer_trade_count: dict[int, int]
    per_layer_win_rate: dict[int, Decimal]
    max_drawdown: Decimal
    largest_loss: Decimal
    top_5_trades: list[dict[str, Any]]
    bottom_5_trades: list[dict[str, Any]]
    confluence_score_distribution: dict[int, int]
    avg_score_of_winners: Decimal
    risk_events: list[str]
    portfolio_balance: Decimal
    per_strategy_pnl: dict[str, StrategyDailyStats] = field(default_factory=dict)
    daily_reports: list[DailyReport] = field(default_factory=list)

    @property
    def losing_trades(self) -> int:
        """Number of losing trades."""
        return self.total_trades - self.winning_trades

    @property
    def win_rate(self) -> Decimal:
        """Win rate as decimal (0-1)."""
        if self.total_trades == 0:
            return Decimal("0")
        return Decimal(str(self.winning_trades)) / Decimal(str(self.total_trades))


class ReportGenerator:
    """Generates daily/weekly reports and exports.

    Features:
    - Daily report from trade history and signals
    - Weekly aggregation of daily reports with detailed analytics
    - CSV export (daily and weekly)
    - Slack Block Kit formatting (daily and weekly)
    """

    def generate_daily(
        self,
        trades: list[dict[str, Any]],
        signals: list[dict[str, Any]],
        report_date: date | None = None,
    ) -> DailyReport:
        """Generate a daily trading report.

        Args:
            trades: List of trade dicts (from PaperTrade.to_db_dict() or similar).
            signals: List of signal dicts.
            report_date: Date for the report (defaults to today).

        Returns:
            DailyReport with aggregated statistics.
        """
        if report_date is None:
            report_date = datetime.now(UTC).date()

        total_trades = len(trades)
        winning_trades = sum(
            1
            for t in trades
            if t.get("actual_pnl") is not None and Decimal(str(t["actual_pnl"])) > 0
        )
        total_pnl = sum(
            (Decimal(str(t.get("actual_pnl", 0) or 0)) for t in trades),
            Decimal("0"),
        )

        # Per-layer P&L
        per_layer: dict[int, Decimal] = {}
        for t in trades:
            layer = t.get("layer", 0)
            pnl = Decimal(str(t.get("actual_pnl", 0) or 0))
            per_layer[layer] = per_layer.get(layer, Decimal("0")) + pnl

        # Per-strategy P&L
        per_strategy: dict[str, StrategyDailyStats] = {}
        for t in trades:
            sid = t.get("strategy", "")
            if not sid:
                continue
            if sid not in per_strategy:
                per_strategy[sid] = StrategyDailyStats()
            st = per_strategy[sid]
            st.trades += 1
            pnl = Decimal(str(t.get("actual_pnl", 0) or 0))
            st.pnl += pnl
            if pnl > 0:
                st.wins += 1
            st.deployed += Decimal(str(t.get("size", 0) or 0))

        # Top signals by edge
        sorted_signals = sorted(signals, key=lambda s: float(s.get("edge", 0)), reverse=True)
        top_signals = sorted_signals[:5]

        # Risk events
        risk_events = [
            t.get("notes", "")
            for t in trades
            if t.get("notes") and "risk" in t.get("notes", "").lower()
        ]

        roi_pct = Decimal("0")
        if total_trades > 0:
            initial = sum(
                (Decimal(str(t.get("size", 0) or 0)) for t in trades),
                Decimal("0"),
            )
            if initial > 0:
                roi_pct = (total_pnl / initial * 100).quantize(Decimal("0.01"))

        report = DailyReport(
            date=report_date,
            total_trades=total_trades,
            winning_trades=winning_trades,
            total_pnl=total_pnl,
            roi_pct=roi_pct,
            per_layer_pnl=per_layer,
            top_signals=top_signals,
            risk_events=risk_events,
            per_strategy_pnl=per_strategy,
        )

        logger.info(
            "daily_report_generated",
            date=str(report_date),
            trades=total_trades,
            pnl=str(total_pnl),
        )

        return report

    def generate_weekly(
        self,
        daily_reports: list[DailyReport],
        trades: list[dict[str, Any]] | None = None,
        portfolio_balance: Decimal = Decimal("0"),
    ) -> WeeklyReport:
        """Aggregate daily reports into weekly summary with detailed analytics.

        Args:
            daily_reports: List of DailyReport objects for the week.
            trades: Optional list of trade dicts for detailed per-trade analysis.
            portfolio_balance: Current portfolio balance.

        Returns:
            WeeklyReport with aggregated statistics and analytics.
        """
        if trades is None:
            trades = []

        if not daily_reports:
            empty_day = DailyReport(
                date=datetime.now(UTC).date(),
                total_trades=0,
                winning_trades=0,
                total_pnl=Decimal("0"),
                roi_pct=Decimal("0"),
                per_layer_pnl={},
                top_signals=[],
                risk_events=[],
            )
            return WeeklyReport(
                week_start=empty_day.date,
                week_end=empty_day.date,
                total_trades=0,
                winning_trades=0,
                total_pnl=Decimal("0"),
                roi_pct=Decimal("0"),
                avg_daily_pnl=Decimal("0"),
                best_day=empty_day,
                worst_day=empty_day,
                per_layer_pnl={},
                per_layer_trade_count={},
                per_layer_win_rate={},
                max_drawdown=Decimal("0"),
                largest_loss=Decimal("0"),
                top_5_trades=[],
                bottom_5_trades=[],
                confluence_score_distribution={},
                avg_score_of_winners=Decimal("0"),
                risk_events=[],
                portfolio_balance=portfolio_balance,
                daily_reports=[],
            )

        total_trades = sum(r.total_trades for r in daily_reports)
        winning_trades = sum(r.winning_trades for r in daily_reports)
        total_pnl = sum((r.total_pnl for r in daily_reports), Decimal("0"))
        avg_daily_pnl = (total_pnl / len(daily_reports)).quantize(Decimal("0.01"))
        best_day = max(daily_reports, key=lambda r: r.total_pnl)
        worst_day = min(daily_reports, key=lambda r: r.total_pnl)

        sorted_dates = sorted(daily_reports, key=lambda r: r.date)
        week_start = sorted_dates[0].date
        week_end = sorted_dates[-1].date

        # Aggregate per-layer P&L
        per_layer_pnl: dict[int, Decimal] = {}
        for report in daily_reports:
            for layer, pnl in report.per_layer_pnl.items():
                per_layer_pnl[layer] = per_layer_pnl.get(layer, Decimal("0")) + pnl

        # Per-layer trade count and win rate from trade-level data
        per_layer_trade_count: dict[int, int] = {}
        per_layer_wins: dict[int, int] = {}
        for t in trades:
            layer = t.get("layer", 0)
            per_layer_trade_count[layer] = per_layer_trade_count.get(layer, 0) + 1
            pnl = t.get("actual_pnl")
            if pnl is not None and Decimal(str(pnl)) > 0:
                per_layer_wins[layer] = per_layer_wins.get(layer, 0) + 1

        per_layer_win_rate: dict[int, Decimal] = {}
        for layer, count in per_layer_trade_count.items():
            wins = per_layer_wins.get(layer, 0)
            per_layer_win_rate[layer] = (
                Decimal(str(wins)) / Decimal(str(count)) if count > 0 else Decimal("0")
            )

        # Max drawdown from daily P&L sequence
        max_drawdown = self._calculate_max_drawdown(sorted_dates)

        # Largest single loss from trades
        largest_loss = Decimal("0")
        for t in trades:
            pnl = t.get("actual_pnl")
            if pnl is not None:
                pnl_dec = Decimal(str(pnl))
                if pnl_dec < largest_loss:
                    largest_loss = pnl_dec

        # Top 5 / bottom 5 trades by actual_pnl
        trades_with_pnl = [t for t in trades if t.get("actual_pnl") is not None]
        sorted_by_pnl = sorted(
            trades_with_pnl, key=lambda t: Decimal(str(t["actual_pnl"])), reverse=True
        )
        top_5_trades = sorted_by_pnl[:5]
        bottom_5_trades = (
            sorted_by_pnl[-5:][::-1] if len(sorted_by_pnl) >= 5 else sorted_by_pnl[::-1]
        )

        # Confluence score distribution
        confluence_dist: dict[int, int] = {}
        for t in trades:
            score = t.get("confluence_score", 0)
            confluence_dist[score] = confluence_dist.get(score, 0) + 1

        # Average score of winning trades
        winning_scores: list[int] = []
        for t in trades:
            pnl = t.get("actual_pnl")
            if pnl is not None and Decimal(str(pnl)) > 0:
                winning_scores.append(t.get("confluence_score", 0))
        avg_score_of_winners = (
            Decimal(str(sum(winning_scores))) / Decimal(str(len(winning_scores)))
            if winning_scores
            else Decimal("0")
        )

        # Aggregate per-strategy P&L across days
        per_strategy_pnl: dict[str, StrategyDailyStats] = {}
        for t in trades:
            sid = t.get("strategy", "")
            if not sid:
                continue
            if sid not in per_strategy_pnl:
                per_strategy_pnl[sid] = StrategyDailyStats()
            st = per_strategy_pnl[sid]
            st.trades += 1
            pnl_val = Decimal(str(t.get("actual_pnl", 0) or 0))
            st.pnl += pnl_val
            if pnl_val > 0:
                st.wins += 1
            st.deployed += Decimal(str(t.get("size", 0) or 0))

        # Collect risk events from all daily reports
        all_risk_events: list[str] = []
        for report in daily_reports:
            all_risk_events.extend(report.risk_events)

        # ROI calculation
        total_invested = sum(
            (Decimal(str(t.get("size", 0) or 0)) for t in trades),
            Decimal("0"),
        )
        roi_pct = (
            (total_pnl / total_invested * 100).quantize(Decimal("0.01"))
            if total_invested > 0
            else Decimal("0")
        )

        weekly = WeeklyReport(
            week_start=week_start,
            week_end=week_end,
            total_trades=total_trades,
            winning_trades=winning_trades,
            total_pnl=total_pnl,
            roi_pct=roi_pct,
            avg_daily_pnl=avg_daily_pnl,
            best_day=best_day,
            worst_day=worst_day,
            per_layer_pnl=per_layer_pnl,
            per_layer_trade_count=per_layer_trade_count,
            per_layer_win_rate=per_layer_win_rate,
            max_drawdown=max_drawdown,
            largest_loss=largest_loss,
            top_5_trades=top_5_trades,
            bottom_5_trades=bottom_5_trades,
            confluence_score_distribution=confluence_dist,
            avg_score_of_winners=avg_score_of_winners,
            risk_events=all_risk_events,
            portfolio_balance=portfolio_balance,
            per_strategy_pnl=per_strategy_pnl,
            daily_reports=list(daily_reports),
        )

        logger.info(
            "weekly_report_generated",
            week_start=str(week_start),
            week_end=str(week_end),
            trades=total_trades,
            pnl=str(total_pnl),
            max_drawdown=str(max_drawdown),
        )

        return weekly

    def _calculate_max_drawdown(self, daily_reports: list[DailyReport]) -> Decimal:
        """Calculate maximum drawdown from a sequence of daily P&L values.

        Args:
            daily_reports: Daily reports sorted by date.

        Returns:
            Maximum drawdown as a positive Decimal (0 if no drawdown).
        """
        if not daily_reports:
            return Decimal("0")

        cumulative = Decimal("0")
        peak = Decimal("0")
        max_dd = Decimal("0")

        for report in daily_reports:
            cumulative += report.total_pnl
            if cumulative > peak:
                peak = cumulative
            drawdown = peak - cumulative
            if drawdown > max_dd:
                max_dd = drawdown

        return max_dd

    def export_csv(self, report: DailyReport, path: str) -> None:
        """Export daily report to CSV file.

        Args:
            report: DailyReport to export.
            path: File path for the CSV.
        """
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "total_trades", "winning_trades", "total_pnl", "roi_pct"])
            writer.writerow(
                [
                    str(report.date),
                    report.total_trades,
                    report.winning_trades,
                    str(report.total_pnl),
                    str(report.roi_pct),
                ]
            )

            # Per-layer breakdown
            writer.writerow([])
            writer.writerow(["layer", "pnl"])
            for layer, pnl in sorted(report.per_layer_pnl.items()):
                writer.writerow([layer, str(pnl)])

        logger.info("csv_exported", path=path, date=str(report.date))

    def export_weekly_csv(self, report: WeeklyReport, path: str) -> None:
        """Export weekly report to CSV file with daily and layer breakdown.

        Args:
            report: WeeklyReport to export.
            path: File path for the CSV.
        """
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)

            # Summary section
            writer.writerow(["Weekly Summary"])
            writer.writerow(
                [
                    "week_start",
                    "week_end",
                    "total_trades",
                    "winning_trades",
                    "total_pnl",
                    "roi_pct",
                    "max_drawdown",
                ]
            )
            writer.writerow(
                [
                    str(report.week_start),
                    str(report.week_end),
                    report.total_trades,
                    report.winning_trades,
                    str(report.total_pnl),
                    str(report.roi_pct),
                    str(report.max_drawdown),
                ]
            )

            # Daily breakdown
            writer.writerow([])
            writer.writerow(["Daily Breakdown"])
            writer.writerow(["date", "trades", "wins", "pnl", "roi_pct"])
            for day in report.daily_reports:
                writer.writerow(
                    [
                        str(day.date),
                        day.total_trades,
                        day.winning_trades,
                        str(day.total_pnl),
                        str(day.roi_pct),
                    ]
                )

            # Layer breakdown
            writer.writerow([])
            writer.writerow(["Layer Breakdown"])
            writer.writerow(["layer", "pnl", "trades", "win_rate"])
            all_layers = sorted(
                set(report.per_layer_pnl.keys()) | set(report.per_layer_trade_count.keys())
            )
            for layer in all_layers:
                writer.writerow(
                    [
                        layer,
                        str(report.per_layer_pnl.get(layer, Decimal("0"))),
                        report.per_layer_trade_count.get(layer, 0),
                        str(report.per_layer_win_rate.get(layer, Decimal("0"))),
                    ]
                )

        logger.info(
            "weekly_csv_exported",
            path=path,
            week_start=str(report.week_start),
        )

    def format_slack_report(self, report: DailyReport) -> dict[str, Any]:
        """Format daily report as Slack Block Kit message.

        Args:
            report: DailyReport to format.

        Returns:
            Block Kit compatible dict for Slack API.
        """
        pnl_emoji = "+" if report.total_pnl >= 0 else ""
        win_rate_pct = float(report.win_rate * 100) if report.total_trades > 0 else 0

        # Per-layer summary text
        layer_lines = []
        for layer, pnl in sorted(report.per_layer_pnl.items()):
            sign = "+" if pnl >= 0 else ""
            layer_lines.append(f"  L{layer}: {sign}${pnl}")

        layer_text = "\n".join(layer_lines) if layer_lines else "  No layer data"

        blocks: list[dict[str, Any]] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"Daily Report — {report.date}",
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Trades:* {report.total_trades}"},
                    {"type": "mrkdwn", "text": f"*Win Rate:* {win_rate_pct:.1f}%"},
                    {"type": "mrkdwn", "text": f"*P&L:* {pnl_emoji}${report.total_pnl}"},
                    {"type": "mrkdwn", "text": f"*ROI:* {report.roi_pct}%"},
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

        # Per-strategy breakdown (RDH)
        if report.per_strategy_pnl:
            strat_names = {"A": "Theta Decay", "B": "Reflexivity", "C": "Weather"}
            strat_lines = []
            for sid, stats in sorted(report.per_strategy_pnl.items()):
                name = strat_names.get(sid, sid)
                sign = "+" if stats.pnl >= 0 else ""
                wr = float(stats.wins / stats.trades * 100) if stats.trades > 0 else 0
                strat_lines.append(f"  {sid} ({name}): {sign}${stats.pnl} ({stats.trades}t, {wr:.0f}%)")
            strat_text = "\n".join(strat_lines)
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Per-Strategy P&L:*\n```\n{strat_text}\n```",
                    },
                }
            )

        # Add risk events if any
        if report.risk_events:
            risk_text = "\n".join(f"- {e}" for e in report.risk_events)
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Risk Events:*\n{risk_text}",
                    },
                }
            )

        return {"blocks": blocks}

    def format_slack_weekly_report(self, report: WeeklyReport) -> dict[str, Any]:
        """Format weekly report as Slack Block Kit message.

        Args:
            report: WeeklyReport to format.

        Returns:
            Block Kit compatible dict for Slack API.
        """
        pnl_sign = "+" if report.total_pnl >= 0 else ""
        win_rate_pct = float(report.win_rate * 100) if report.total_trades > 0 else 0.0

        # Per-layer summary text
        layer_lines = []
        all_layers = sorted(
            set(report.per_layer_pnl.keys()) | set(report.per_layer_trade_count.keys())
        )
        for layer in all_layers:
            pnl = report.per_layer_pnl.get(layer, Decimal("0"))
            count = report.per_layer_trade_count.get(layer, 0)
            wr = report.per_layer_win_rate.get(layer, Decimal("0"))
            sign = "+" if pnl >= 0 else ""
            layer_lines.append(f"  L{layer}: {sign}${pnl} ({count} trades, {float(wr)*100:.0f}%)")

        layer_text = "\n".join(layer_lines) if layer_lines else "  No layer data"

        # Top trades text
        top_lines = []
        for t in report.top_5_trades[:3]:
            pnl = t.get("actual_pnl", 0)
            token = t.get("token_id", "?")[:8]
            top_lines.append(f"  +${pnl} ({token})")
        top_text = "\n".join(top_lines) if top_lines else "  None"

        blocks: list[dict[str, Any]] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"Weekly Report — {report.week_start} to {report.week_end}",
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Trades:* {report.total_trades}"},
                    {"type": "mrkdwn", "text": f"*Win Rate:* {win_rate_pct:.1f}%"},
                    {"type": "mrkdwn", "text": f"*P&L:* {pnl_sign}${report.total_pnl}"},
                    {"type": "mrkdwn", "text": f"*ROI:* {report.roi_pct}%"},
                ],
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Avg Daily P&L:* ${report.avg_daily_pnl}"},
                    {"type": "mrkdwn", "text": f"*Max Drawdown:* ${report.max_drawdown}"},
                    {"type": "mrkdwn", "text": f"*Best Day:* ${report.best_day.total_pnl}"},
                    {"type": "mrkdwn", "text": f"*Worst Day:* ${report.worst_day.total_pnl}"},
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Per-Layer Breakdown:*\n```\n{layer_text}\n```",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Top Trades:*\n```\n{top_text}\n```",
                },
            },
        ]

        # Per-strategy breakdown (RDH)
        if report.per_strategy_pnl:
            strat_names = {"A": "Theta Decay", "B": "Reflexivity", "C": "Weather"}
            strat_lines = []
            for sid, stats in sorted(report.per_strategy_pnl.items()):
                name = strat_names.get(sid, sid)
                sign = "+" if stats.pnl >= 0 else ""
                wr = float(stats.wins / stats.trades * 100) if stats.trades > 0 else 0
                strat_lines.append(
                    f"  {sid} ({name}): {sign}${stats.pnl} ({stats.trades}t, {wr:.0f}%)"
                )
            strat_text = "\n".join(strat_lines)
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Per-Strategy P&L:*\n```\n{strat_text}\n```",
                    },
                }
            )

        # Risk events
        if report.risk_events:
            risk_text = "\n".join(f"- {e}" for e in report.risk_events[:5])
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Risk Events:*\n{risk_text}",
                    },
                }
            )

        return {"blocks": blocks}
