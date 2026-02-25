"""Tests for Dashboard + Reporting (PM-210).

Tests verify:
1. CLI Dashboard: render active, render inactive, layer table, portfolio summary
2. Report Generator: daily report, weekly aggregation, CSV export, slack format
3. LayerStatus: creation, with error
"""

from __future__ import annotations

import os
import tempfile
from datetime import date
from decimal import Decimal

from arbo.dashboard.cli_dashboard import CLIDashboard, LayerStatus, StrategyStatus
from arbo.dashboard.report_generator import DailyReport, ReportGenerator

# ================================================================
# Helpers
# ================================================================


def _make_layer_status(
    layer: int = 1,
    name: str = "Market Making",
    active: bool = True,
    signals_count: int = 5,
    error: str | None = None,
) -> LayerStatus:
    return LayerStatus(
        layer=layer,
        name=name,
        active=active,
        signals_count=signals_count,
        error=error,
    )


def _make_daily_report(
    total_trades: int = 10,
    winning_trades: int = 6,
    total_pnl: Decimal = Decimal("25.50"),
    roi_pct: Decimal = Decimal("5.10"),
) -> DailyReport:
    return DailyReport(
        date=date(2026, 2, 21),
        total_trades=total_trades,
        winning_trades=winning_trades,
        total_pnl=total_pnl,
        roi_pct=roi_pct,
        per_layer_pnl={2: Decimal("15.00"), 4: Decimal("10.50")},
        top_signals=[{"edge": 0.08, "layer": 2}],
        risk_events=[],
    )


# ================================================================
# TestCLIDashboard
# ================================================================


class TestCLIDashboard:
    """CLI dashboard rendering."""

    def test_render_active_layers(self) -> None:
        """Render with active layers shows ACTIVE status."""
        dash = CLIDashboard()
        layers = [
            _make_layer_status(1, "Market Making", active=True, signals_count=3),
            _make_layer_status(2, "Value Betting", active=True, signals_count=7),
        ]
        output = dash.render(layers)
        assert "ACTIVE" in output
        assert "Market Making" in output
        assert "Value Betting" in output

    def test_render_inactive_layers(self) -> None:
        """Render with inactive layers shows OFF status."""
        dash = CLIDashboard()
        layers = [
            _make_layer_status(9, "Sports Latency", active=False, signals_count=0),
        ]
        output = dash.render(layers)
        assert "OFF" in output
        assert "Sports Latency" in output

    def test_format_layer_table(self) -> None:
        """Layer table formats correctly with header and rows."""
        dash = CLIDashboard()
        layers = [
            _make_layer_status(1, "Market Making", True, 5),
            _make_layer_status(4, "Whale Copy", True, 2),
        ]
        table = dash.format_layer_table(layers)
        assert "LAYERS" in table
        assert "Market Making" in table
        assert "Whale Copy" in table

    def test_format_portfolio_summary(self) -> None:
        """Portfolio summary shows balance, P&L, ROI."""
        dash = CLIDashboard()
        stats = {
            "current_balance": Decimal("2050.00"),
            "total_value": Decimal("2100.00"),
            "total_pnl": Decimal("100.00"),
            "roi_pct": 5.0,
            "win_rate": 0.65,
            "open_trades": 3,
            "resolved_trades": 10,
        }
        output = dash.format_portfolio_summary(stats)
        assert "PORTFOLIO" in output
        assert "2050" in output
        assert "5.0" in output


# ================================================================
# TestReportGenerator
# ================================================================


class TestReportGenerator:
    """Report generation and export."""

    def test_daily_report(self) -> None:
        """Generate daily report from trade data."""
        gen = ReportGenerator()
        trades = [
            {"layer": 2, "actual_pnl": "10.00", "size": "50.00", "notes": ""},
            {"layer": 4, "actual_pnl": "-5.00", "size": "25.00", "notes": ""},
            {"layer": 2, "actual_pnl": "8.00", "size": "40.00", "notes": ""},
        ]
        signals = [
            {"edge": 0.08, "layer": 2},
            {"edge": 0.05, "layer": 4},
        ]

        report = gen.generate_daily(trades, signals)
        assert report.total_trades == 3
        assert report.winning_trades == 2
        assert report.total_pnl == Decimal("13.00")

    def test_weekly_aggregation(self) -> None:
        """Weekly report aggregates daily reports correctly."""
        gen = ReportGenerator()
        dailies = [
            _make_daily_report(total_trades=5, total_pnl=Decimal("10.00")),
            _make_daily_report(total_trades=8, total_pnl=Decimal("-3.00")),
            _make_daily_report(total_trades=6, total_pnl=Decimal("15.00")),
        ]

        weekly = gen.generate_weekly(dailies)
        assert weekly.total_trades == 19
        assert weekly.total_pnl == Decimal("22.00")
        assert weekly.best_day.total_pnl == Decimal("15.00")
        assert weekly.worst_day.total_pnl == Decimal("-3.00")
        assert len(weekly.daily_reports) == 3

    def test_csv_export(self) -> None:
        """CSV export creates valid file."""
        gen = ReportGenerator()
        report = _make_daily_report()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name

        try:
            gen.export_csv(report, path)
            with open(path) as f:
                content = f.read()
            assert "2026-02-21" in content
            assert "25.50" in content
        finally:
            os.unlink(path)

    def test_slack_format(self) -> None:
        """Slack Block Kit format has correct structure."""
        gen = ReportGenerator()
        report = _make_daily_report()
        slack_msg = gen.format_slack_report(report)

        assert "blocks" in slack_msg
        blocks = slack_msg["blocks"]
        assert len(blocks) >= 2

        # Header block
        assert blocks[0]["type"] == "header"
        assert "2026-02-21" in blocks[0]["text"]["text"]

        # Section with fields
        assert blocks[1]["type"] == "section"
        assert "fields" in blocks[1]


# ================================================================
# TestLayerStatus
# ================================================================


class TestLayerStatus:
    """LayerStatus dataclass."""

    def test_creation(self) -> None:
        """LayerStatus stores fields correctly."""
        ls = _make_layer_status(layer=4, name="Whale Copy", active=True, signals_count=10)
        assert ls.layer == 4
        assert ls.name == "Whale Copy"
        assert ls.active is True
        assert ls.signals_count == 10

    def test_with_error(self) -> None:
        """LayerStatus with error stores error string."""
        ls = _make_layer_status(error="Connection timeout")
        assert ls.error == "Connection timeout"


# ================================================================
# Strategy Table (RDH)
# ================================================================


class TestStrategyTable:
    """CLI dashboard strategy table."""

    def test_format_strategy_table(self) -> None:
        """Strategy table formats correctly."""
        dash = CLIDashboard()
        strategies = [
            StrategyStatus("A", "Theta Decay", Decimal("400"), Decimal("50"), Decimal("350"), 2),
            StrategyStatus("B", "Reflexivity Surfer", Decimal("400"), Decimal("0"), Decimal("400"), 0),
            StrategyStatus("C", "Compound Weather", Decimal("1000"), Decimal("200"), Decimal("800"), 3),
        ]
        table = dash.format_strategy_table(strategies)
        assert "STRATEGIES" in table
        assert "Theta Decay" in table
        assert "Reflexivity Surfer" in table
        assert "Compound Weather" in table
        assert "ACTIVE" in table

    def test_halted_strategy_shown(self) -> None:
        """Halted strategy shows HALTED status."""
        dash = CLIDashboard()
        strategies = [
            StrategyStatus("A", "Theta Decay", is_halted=True),
        ]
        table = dash.format_strategy_table(strategies)
        assert "HALTED" in table

    def test_render_with_strategies(self) -> None:
        """Render includes strategy section when provided."""
        dash = CLIDashboard()
        layers = [_make_layer_status()]
        strategies = [
            StrategyStatus("A", "Theta Decay", Decimal("400"), Decimal("50"), Decimal("350"), 2),
        ]
        output = dash.render(layers, strategies=strategies)
        assert "STRATEGIES" in output
        assert "LAYERS" in output


class TestSlackBotBlocks:
    """Slack bot Block Kit formatters."""

    def test_status_blocks_with_strategies(self) -> None:
        """Status blocks include strategy section when data present."""
        from arbo.dashboard.slack_bot import SlackBot

        status = {
            "mode": "paper",
            "uptime_s": 3600,
            "layers_active": 5,
            "layers_total": 9,
            "balance": "2050",
            "open_positions": 3,
            "strategies": {
                "A": {"name": "Theta Decay", "deployed": 50, "available": 350, "positions": 2, "is_halted": False},
            },
        }
        blocks = SlackBot._format_status_blocks(status)
        block_texts = [str(b) for b in blocks]
        assert any("Strategies" in t for t in block_texts)
        assert any("Theta Decay" in t for t in block_texts)

    def test_status_blocks_without_strategies(self) -> None:
        """Status blocks work without strategy data."""
        from arbo.dashboard.slack_bot import SlackBot

        status = {"mode": "paper", "uptime_s": 60, "balance": "2000"}
        blocks = SlackBot._format_status_blocks(status)
        assert len(blocks) == 2  # header + fields only

    def test_pnl_blocks_with_strategy_breakdown(self) -> None:
        """P&L blocks include per-strategy section."""
        from arbo.dashboard.slack_bot import SlackBot

        pnl = {
            "total_pnl": "25.00",
            "roi_pct": "1.25",
            "total_trades": 5,
            "wins": 3,
            "win_rate": 0.6,
            "per_layer_pnl": {2: "15.00"},
            "per_strategy_pnl": {
                "A": {"pnl": "10.00", "trades": 3, "wins": 2},
                "C": {"pnl": "15.00", "trades": 2, "wins": 1},
            },
        }
        blocks = SlackBot._format_pnl_blocks(pnl)
        block_texts = [str(b) for b in blocks]
        assert any("Per-Strategy" in t for t in block_texts)
        assert any("Theta Decay" in t for t in block_texts)
