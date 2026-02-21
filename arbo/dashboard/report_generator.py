"""Report generator for daily/weekly summaries (PM-210).

Generates structured reports from paper trading data, exports to CSV,
and formats Slack Block Kit messages for alert delivery.

See brief PM-210 for full specification.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from arbo.utils.logger import get_logger

logger = get_logger("report_generator")


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
    - Weekly aggregation of daily reports
    - CSV export
    - Slack Block Kit formatting
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
        )

        logger.info(
            "daily_report_generated",
            date=str(report_date),
            trades=total_trades,
            pnl=str(total_pnl),
        )

        return report

    def generate_weekly(self, daily_reports: list[DailyReport]) -> dict[str, Any]:
        """Aggregate daily reports into weekly summary.

        Args:
            daily_reports: List of DailyReport objects for the week.

        Returns:
            Dict with aggregated weekly statistics.
        """
        if not daily_reports:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "total_pnl": Decimal("0"),
                "avg_daily_pnl": Decimal("0"),
                "best_day_pnl": Decimal("0"),
                "worst_day_pnl": Decimal("0"),
                "per_layer_pnl": {},
                "days": 0,
            }

        total_trades = sum(r.total_trades for r in daily_reports)
        winning_trades = sum(r.winning_trades for r in daily_reports)
        total_pnl = sum((r.total_pnl for r in daily_reports), Decimal("0"))
        avg_daily_pnl = total_pnl / len(daily_reports) if daily_reports else Decimal("0")
        best_day = max(daily_reports, key=lambda r: r.total_pnl)
        worst_day = min(daily_reports, key=lambda r: r.total_pnl)

        # Aggregate per-layer P&L
        per_layer: dict[int, Decimal] = {}
        for report in daily_reports:
            for layer, pnl in report.per_layer_pnl.items():
                per_layer[layer] = per_layer.get(layer, Decimal("0")) + pnl

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "total_pnl": total_pnl,
            "avg_daily_pnl": avg_daily_pnl.quantize(Decimal("0.01")),
            "best_day_pnl": best_day.total_pnl,
            "worst_day_pnl": worst_day.total_pnl,
            "per_layer_pnl": per_layer,
            "days": len(daily_reports),
        }

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
