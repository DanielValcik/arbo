"""Tests for Weekly Report generation (PM-210 Phase B).

Tests verify:
1. Weekly aggregation from daily reports
2. Empty reports handling
3. Max drawdown calculation
4. Top/bottom trades extraction
5. Per-layer stats (trade count, win rate)
6. Confluence score distribution
7. Slack Block Kit weekly formatting
8. CSV export content
"""

from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from arbo.dashboard.report_generator import (
    DailyReport,
    ReportGenerator,
    StrategyDailyStats,
    WeeklyReport,
)

# ================================================================
# Helpers
# ================================================================


def _make_daily(
    report_date: date,
    total_trades: int = 5,
    winning_trades: int = 3,
    total_pnl: Decimal = Decimal("10.00"),
    roi_pct: Decimal = Decimal("2.00"),
    per_layer_pnl: dict[int, Decimal] | None = None,
    risk_events: list[str] | None = None,
) -> DailyReport:
    """Create a DailyReport for testing."""
    return DailyReport(
        date=report_date,
        total_trades=total_trades,
        winning_trades=winning_trades,
        total_pnl=total_pnl,
        roi_pct=roi_pct,
        per_layer_pnl=per_layer_pnl or {},
        top_signals=[],
        risk_events=risk_events or [],
    )


def _make_trade(
    actual_pnl: float,
    layer: int = 2,
    confluence_score: int = 2,
    size: float = 50.0,
    token_id: str = "tok_1",
) -> dict[str, Any]:
    """Create a trade dict for testing."""
    return {
        "actual_pnl": actual_pnl,
        "layer": layer,
        "confluence_score": confluence_score,
        "size": size,
        "token_id": token_id,
        "side": "BUY",
    }


@pytest.fixture
def gen() -> ReportGenerator:
    return ReportGenerator()


@pytest.fixture
def sample_daily_reports() -> list[DailyReport]:
    """Seven days of daily reports for a full week."""
    return [
        _make_daily(
            date(2026, 2, 16),
            total_trades=4,
            winning_trades=3,
            total_pnl=Decimal("15.00"),
            per_layer_pnl={2: Decimal("10.00"), 4: Decimal("5.00")},
        ),
        _make_daily(
            date(2026, 2, 17),
            total_trades=6,
            winning_trades=2,
            total_pnl=Decimal("-8.00"),
            per_layer_pnl={2: Decimal("-5.00"), 7: Decimal("-3.00")},
        ),
        _make_daily(
            date(2026, 2, 18),
            total_trades=5,
            winning_trades=4,
            total_pnl=Decimal("20.00"),
            per_layer_pnl={2: Decimal("12.00"), 4: Decimal("8.00")},
        ),
        _make_daily(
            date(2026, 2, 19),
            total_trades=3,
            winning_trades=1,
            total_pnl=Decimal("-12.00"),
            per_layer_pnl={5: Decimal("-12.00")},
        ),
        _make_daily(
            date(2026, 2, 20),
            total_trades=7,
            winning_trades=5,
            total_pnl=Decimal("25.00"),
            per_layer_pnl={2: Decimal("15.00"), 7: Decimal("10.00")},
            risk_events=["risk: position size warning"],
        ),
    ]


@pytest.fixture
def sample_trades() -> list[dict[str, Any]]:
    """Sample trades for detailed analysis."""
    return [
        _make_trade(15.0, layer=2, confluence_score=3, token_id="tok_a"),
        _make_trade(8.0, layer=4, confluence_score=2, token_id="tok_b"),
        _make_trade(-5.0, layer=2, confluence_score=2, token_id="tok_c"),
        _make_trade(20.0, layer=2, confluence_score=3, token_id="tok_d"),
        _make_trade(-12.0, layer=5, confluence_score=1, token_id="tok_e"),
        _make_trade(-3.0, layer=7, confluence_score=2, token_id="tok_f"),
        _make_trade(25.0, layer=7, confluence_score=3, token_id="tok_g"),
        _make_trade(-8.0, layer=2, confluence_score=2, token_id="tok_h"),
    ]


# ================================================================
# Weekly Report from Daily Reports
# ================================================================


class TestWeeklyReportBasic:
    """Basic weekly aggregation from daily reports."""

    def test_weekly_report_from_daily_reports(
        self,
        gen: ReportGenerator,
        sample_daily_reports: list[DailyReport],
        sample_trades: list[dict[str, Any]],
    ) -> None:
        """Weekly report aggregates totals from daily reports."""
        report = gen.generate_weekly(
            sample_daily_reports,
            trades=sample_trades,
            portfolio_balance=Decimal("2040"),
        )
        assert isinstance(report, WeeklyReport)
        assert report.total_trades == 25  # 4+6+5+3+7
        assert report.winning_trades == 15  # 3+2+4+1+5
        assert report.total_pnl == Decimal("40.00")  # 15-8+20-12+25
        assert report.week_start == date(2026, 2, 16)
        assert report.week_end == date(2026, 2, 20)
        assert report.portfolio_balance == Decimal("2040")
        assert len(report.daily_reports) == 5

    def test_weekly_report_empty_reports(self, gen: ReportGenerator) -> None:
        """Empty daily reports produce zeroed weekly report."""
        report = gen.generate_weekly([], trades=[], portfolio_balance=Decimal("2000"))
        assert report.total_trades == 0
        assert report.winning_trades == 0
        assert report.total_pnl == Decimal("0")
        assert report.max_drawdown == Decimal("0")
        assert report.per_layer_pnl == {}
        assert report.top_5_trades == []
        assert report.bottom_5_trades == []

    def test_weekly_report_properties(
        self,
        gen: ReportGenerator,
        sample_daily_reports: list[DailyReport],
    ) -> None:
        """WeeklyReport computed properties work correctly."""
        report = gen.generate_weekly(sample_daily_reports)
        assert report.losing_trades == report.total_trades - report.winning_trades
        assert report.win_rate == Decimal(str(report.winning_trades)) / Decimal(
            str(report.total_trades)
        )


# ================================================================
# Max Drawdown
# ================================================================


class TestMaxDrawdown:
    """Max drawdown calculation from daily P&L sequence."""

    def test_weekly_report_max_drawdown_calculation(self, gen: ReportGenerator) -> None:
        """Max drawdown tracks peak-to-trough decline."""
        # Sequence: +15, -8, +20, -12, +25
        # Cumulative: 15, 7, 27, 15, 40
        # Peak: 15, 15, 27, 27, 40
        # Drawdown: 0, 8, 0, 12, 0
        # Max drawdown = 12
        reports = [
            _make_daily(date(2026, 2, 16), total_pnl=Decimal("15.00")),
            _make_daily(date(2026, 2, 17), total_pnl=Decimal("-8.00")),
            _make_daily(date(2026, 2, 18), total_pnl=Decimal("20.00")),
            _make_daily(date(2026, 2, 19), total_pnl=Decimal("-12.00")),
            _make_daily(date(2026, 2, 20), total_pnl=Decimal("25.00")),
        ]
        report = gen.generate_weekly(reports)
        assert report.max_drawdown == Decimal("12.00")

    def test_max_drawdown_no_drawdown(self, gen: ReportGenerator) -> None:
        """If P&L is always positive, drawdown is 0."""
        reports = [
            _make_daily(date(2026, 2, 16), total_pnl=Decimal("5.00")),
            _make_daily(date(2026, 2, 17), total_pnl=Decimal("10.00")),
            _make_daily(date(2026, 2, 18), total_pnl=Decimal("3.00")),
        ]
        report = gen.generate_weekly(reports)
        assert report.max_drawdown == Decimal("0")


# ================================================================
# Top/Bottom Trades
# ================================================================


class TestTopBottomTrades:
    """Top 5 and bottom 5 trades extraction."""

    def test_weekly_report_top_bottom_trades(
        self,
        gen: ReportGenerator,
        sample_daily_reports: list[DailyReport],
        sample_trades: list[dict[str, Any]],
    ) -> None:
        """Top and bottom trades are sorted by actual_pnl."""
        report = gen.generate_weekly(
            sample_daily_reports,
            trades=sample_trades,
        )
        # Top 5: 25, 20, 15, 8, -3 (top 5 by descending pnl)
        assert len(report.top_5_trades) == 5
        assert Decimal(str(report.top_5_trades[0]["actual_pnl"])) == Decimal("25")
        assert Decimal(str(report.top_5_trades[1]["actual_pnl"])) == Decimal("20")

        # Bottom 5: sorted ascending (worst first when reversed)
        assert len(report.bottom_5_trades) == 5
        assert Decimal(str(report.bottom_5_trades[0]["actual_pnl"])) == Decimal("-12")

    def test_top_bottom_fewer_than_5(self, gen: ReportGenerator) -> None:
        """With fewer than 5 trades, returns all of them."""
        reports = [_make_daily(date(2026, 2, 16), total_trades=2, winning_trades=1)]
        trades = [_make_trade(10.0), _make_trade(-5.0)]
        report = gen.generate_weekly(reports, trades=trades)
        assert len(report.top_5_trades) == 2
        assert len(report.bottom_5_trades) == 2

    def test_largest_loss(
        self,
        gen: ReportGenerator,
        sample_daily_reports: list[DailyReport],
        sample_trades: list[dict[str, Any]],
    ) -> None:
        """Largest loss is tracked."""
        report = gen.generate_weekly(sample_daily_reports, trades=sample_trades)
        assert report.largest_loss == Decimal("-12")


# ================================================================
# Per-Layer Stats
# ================================================================


class TestPerLayerStats:
    """Per-layer trade count and win rate from trade data."""

    def test_weekly_report_per_layer_stats(
        self,
        gen: ReportGenerator,
        sample_daily_reports: list[DailyReport],
        sample_trades: list[dict[str, Any]],
    ) -> None:
        """Per-layer trade count and win rate are computed from trades."""
        report = gen.generate_weekly(sample_daily_reports, trades=sample_trades)

        # Layer 2 trades: pnl=[15, -5, 20, -8] → 4 trades, 2 wins
        assert report.per_layer_trade_count[2] == 4
        assert report.per_layer_win_rate[2] == Decimal("2") / Decimal("4")

        # Layer 4: pnl=[8] → 1 trade, 1 win
        assert report.per_layer_trade_count[4] == 1
        assert report.per_layer_win_rate[4] == Decimal("1")

        # Layer 5: pnl=[-12] → 1 trade, 0 wins
        assert report.per_layer_trade_count[5] == 1
        assert report.per_layer_win_rate[5] == Decimal("0")

        # Layer 7: pnl=[-3, 25] → 2 trades, 1 win
        assert report.per_layer_trade_count[7] == 2
        assert report.per_layer_win_rate[7] == Decimal("1") / Decimal("2")


# ================================================================
# Confluence Distribution
# ================================================================


class TestConfluenceDistribution:
    """Confluence score distribution and average score of winners."""

    def test_weekly_report_confluence_distribution(
        self,
        gen: ReportGenerator,
        sample_daily_reports: list[DailyReport],
        sample_trades: list[dict[str, Any]],
    ) -> None:
        """Confluence score distribution counts trades per score."""
        report = gen.generate_weekly(sample_daily_reports, trades=sample_trades)

        # Scores: 3,2,2,3,1,2,3,2 → {1:1, 2:4, 3:3}
        assert report.confluence_score_distribution[1] == 1
        assert report.confluence_score_distribution[2] == 4
        assert report.confluence_score_distribution[3] == 3

    def test_avg_score_of_winners(
        self,
        gen: ReportGenerator,
        sample_daily_reports: list[DailyReport],
        sample_trades: list[dict[str, Any]],
    ) -> None:
        """Average confluence score of winning trades is correct."""
        report = gen.generate_weekly(sample_daily_reports, trades=sample_trades)

        # Winners: pnl>0 → 15(score=3), 8(score=2), 20(score=3), 25(score=3) → avg = 11/4 = 2.75
        expected = Decimal("11") / Decimal("4")
        assert report.avg_score_of_winners == expected


# ================================================================
# Slack Block Kit Weekly
# ================================================================


class TestSlackWeeklyReport:
    """Slack Block Kit formatted weekly report."""

    def test_format_slack_weekly_report(
        self,
        gen: ReportGenerator,
        sample_daily_reports: list[DailyReport],
        sample_trades: list[dict[str, Any]],
    ) -> None:
        """Slack weekly report contains expected Block Kit structure."""
        report = gen.generate_weekly(
            sample_daily_reports,
            trades=sample_trades,
            portfolio_balance=Decimal("2040"),
        )
        slack_msg = gen.format_slack_weekly_report(report)

        assert "blocks" in slack_msg
        blocks = slack_msg["blocks"]

        # Header block
        assert blocks[0]["type"] == "header"
        assert "Weekly Report" in blocks[0]["text"]["text"]

        # Summary section with fields
        assert blocks[1]["type"] == "section"
        assert "fields" in blocks[1]
        field_texts = [f["text"] for f in blocks[1]["fields"]]
        assert any("Trades" in t for t in field_texts)
        assert any("Win Rate" in t for t in field_texts)
        assert any("P&L" in t for t in field_texts)

        # Drawdown / best/worst day section
        assert blocks[2]["type"] == "section"
        dd_fields = [f["text"] for f in blocks[2]["fields"]]
        assert any("Drawdown" in t for t in dd_fields)

        # Risk events block should be present (sample has risk events)
        risk_blocks = [b for b in blocks if "Risk Events" in str(b)]
        assert len(risk_blocks) >= 1


# ================================================================
# CSV Export
# ================================================================


class TestWeeklyCSVExport:
    """Weekly CSV export content verification."""

    def test_export_weekly_csv(
        self,
        gen: ReportGenerator,
        sample_daily_reports: list[DailyReport],
        sample_trades: list[dict[str, Any]],
        tmp_path: Any,
    ) -> None:
        """CSV file contains summary, daily, and layer breakdown sections."""
        report = gen.generate_weekly(
            sample_daily_reports,
            trades=sample_trades,
            portfolio_balance=Decimal("2040"),
        )
        csv_path = str(tmp_path / "weekly.csv")
        gen.export_weekly_csv(report, csv_path)

        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Check summary header
        assert rows[0] == ["Weekly Summary"]
        assert "week_start" in rows[1]

        # Find daily breakdown section
        daily_idx = next(i for i, r in enumerate(rows) if r and r[0] == "Daily Breakdown")
        assert rows[daily_idx + 1][0] == "date"
        # 5 daily rows
        daily_data_rows = []
        for r in rows[daily_idx + 2 :]:
            if not r or r[0] == "":
                break
            daily_data_rows.append(r)
        assert len(daily_data_rows) == 5

        # Find layer breakdown section
        layer_idx = next(i for i, r in enumerate(rows) if r and r[0] == "Layer Breakdown")
        assert rows[layer_idx + 1][0] == "layer"


# ================================================================
# Per-Strategy P&L (RDH)
# ================================================================


class TestPerStrategyDaily:
    """Per-strategy P&L in daily reports."""

    def test_daily_report_per_strategy_pnl(self, gen: ReportGenerator) -> None:
        """Daily report computes per-strategy P&L from trade strategy field."""
        trades = [
            {"actual_pnl": 10.0, "layer": 7, "size": 30, "strategy": "A"},
            {"actual_pnl": -5.0, "layer": 7, "size": 25, "strategy": "A"},
            {"actual_pnl": 8.0, "layer": 2, "size": 50, "strategy": "B"},
        ]
        report = gen.generate_daily(trades, signals=[])
        assert "A" in report.per_strategy_pnl
        assert "B" in report.per_strategy_pnl
        assert report.per_strategy_pnl["A"].trades == 2
        assert report.per_strategy_pnl["A"].pnl == Decimal("5")
        assert report.per_strategy_pnl["A"].wins == 1
        assert report.per_strategy_pnl["B"].trades == 1
        assert report.per_strategy_pnl["B"].pnl == Decimal("8")

    def test_daily_report_no_strategy(self, gen: ReportGenerator) -> None:
        """Trades without strategy field don't appear in per_strategy_pnl."""
        trades = [
            {"actual_pnl": 10.0, "layer": 2, "size": 50},
        ]
        report = gen.generate_daily(trades, signals=[])
        assert report.per_strategy_pnl == {}

    def test_daily_report_empty_strategy(self, gen: ReportGenerator) -> None:
        """Trades with empty strategy are excluded."""
        trades = [
            {"actual_pnl": 10.0, "layer": 2, "size": 50, "strategy": ""},
        ]
        report = gen.generate_daily(trades, signals=[])
        assert report.per_strategy_pnl == {}


class TestPerStrategyWeekly:
    """Per-strategy P&L in weekly reports."""

    def test_weekly_per_strategy_aggregation(self, gen: ReportGenerator) -> None:
        """Weekly report aggregates per-strategy P&L from trade data."""
        daily = [
            _make_daily(date(2026, 2, 16), total_pnl=Decimal("10")),
            _make_daily(date(2026, 2, 17), total_pnl=Decimal("-5")),
        ]
        trades = [
            {"actual_pnl": 15.0, "layer": 7, "size": 30, "strategy": "A"},
            {"actual_pnl": -3.0, "layer": 7, "size": 25, "strategy": "A"},
            {"actual_pnl": 8.0, "layer": 2, "size": 40, "strategy": "C"},
            {"actual_pnl": -15.0, "layer": 2, "size": 50, "strategy": "C"},
        ]
        report = gen.generate_weekly(daily, trades=trades)

        assert "A" in report.per_strategy_pnl
        assert report.per_strategy_pnl["A"].trades == 2
        assert report.per_strategy_pnl["A"].pnl == Decimal("12")
        assert report.per_strategy_pnl["A"].wins == 1

        assert "C" in report.per_strategy_pnl
        assert report.per_strategy_pnl["C"].trades == 2
        assert report.per_strategy_pnl["C"].pnl == Decimal("-7")
        assert report.per_strategy_pnl["C"].wins == 1

    def test_weekly_per_strategy_empty(self, gen: ReportGenerator) -> None:
        """No strategy trades → empty per_strategy_pnl."""
        report = gen.generate_weekly([], trades=[])
        assert report.per_strategy_pnl == {}


class TestSlackStrategyBlocks:
    """Slack Block Kit includes per-strategy section."""

    def test_daily_slack_strategy_section(self, gen: ReportGenerator) -> None:
        """Daily Slack report includes per-strategy block when data present."""
        trades = [
            {"actual_pnl": 10.0, "layer": 7, "size": 30, "strategy": "A"},
            {"actual_pnl": -5.0, "layer": 2, "size": 50, "strategy": "C"},
        ]
        report = gen.generate_daily(trades, signals=[])
        msg = gen.format_slack_report(report)
        block_texts = [str(b) for b in msg["blocks"]]
        assert any("Per-Strategy" in t for t in block_texts)
        assert any("Theta Decay" in t for t in block_texts)

    def test_weekly_slack_strategy_section(self, gen: ReportGenerator) -> None:
        """Weekly Slack report includes per-strategy block."""
        daily = [_make_daily(date(2026, 2, 16))]
        trades = [
            {"actual_pnl": 10.0, "layer": 7, "size": 30, "strategy": "A"},
        ]
        report = gen.generate_weekly(daily, trades=trades)
        msg = gen.format_slack_weekly_report(report)
        block_texts = [str(b) for b in msg["blocks"]]
        assert any("Per-Strategy" in t for t in block_texts)

    def test_no_strategy_no_block(self, gen: ReportGenerator) -> None:
        """No strategy data → no per-strategy block."""
        trades = [{"actual_pnl": 10.0, "layer": 2, "size": 50}]
        report = gen.generate_daily(trades, signals=[])
        msg = gen.format_slack_report(report)
        block_texts = [str(b) for b in msg["blocks"]]
        assert not any("Per-Strategy" in t for t in block_texts)
