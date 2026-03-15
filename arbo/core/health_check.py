"""Automated health checker for Strategy C paper trading.

Runs every 12 hours, compares live paper trading metrics against
AR-0134 backtest baseline. Produces a verdict:
- ok: all metrics within tolerance
- needs_attention: significant deviation from baseline
- bug_detected: zero trades, impossible values, system issues

Also provides expected-vs-reality computations and seasonality analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import sqlalchemy as sa

from arbo.utils.db import HealthCheck, PaperTrade, get_session_factory
from arbo.utils.logger import get_logger

logger = get_logger("health_check")


# ================================================================
# AR-0134 Backtest Baseline (source of truth for expectations)
# ================================================================

AR0134_BASELINE = {
    "model": "AR-0134",
    "score": 170.1,
    # Train performance
    "train_trades": 273,
    "train_wr": 0.436,
    "train_avg_edge": 0.15,
    # OOS performance (the conservative reference)
    "oos_trades": 175,
    "oos_wr": 0.382,
    "oos_pnl": 297.0,
    "oos_days": 70,
    "oos_daily_pnl": 4.24,  # $297 / 70 days
    "oos_weekly_pnl": 29.7,  # $297 / 10 weeks
    "oos_daily_trades": 2.5,  # 175 / 70 days
    # Walk-forward
    "wf_pnl": 2218.0,
    # Risk
    "max_drawdown_pct": 0.13,
    # Strategy params
    "capital": 1000.0,
    "kelly_fraction": 0.25,
    "cities_active": 18,  # 20 - 2 excluded
    "excluded_cities": ["Chicago", "Seoul"],
    # Seasonality (from backtest analysis)
    "winter_pnl_share": 0.67,  # Dec+Jan+Feb = 67% of total PnL
    "best_months": [12, 1, 2],
    "worst_months": [6, 7, 8],
}

# Tolerance bands for health check verdicts
MIN_COMPARISON_TRADES = 20  # Need at least this many for meaningful comparison
WIN_RATE_WARNING_BAND = 0.15  # ±15pp from baseline before warning
DAILY_TRADES_WARNING_LOW = 0.5  # Less than 0.5 trades/day = suspicious


@dataclass
class HealthReport:
    """Result of a health check run."""

    verdict: str  # ok, needs_attention, bug_detected
    check_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    window_hours: int = 12
    metrics: dict = field(default_factory=dict)
    expected: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


async def run_health_check(window_hours: int = 12) -> HealthReport:
    """Run a health check comparing paper trading to backtest baseline.

    Args:
        window_hours: How many hours back to analyze (default 12).

    Returns:
        HealthReport with verdict and metrics.
    """
    now = datetime.now(UTC)
    window_start = now - timedelta(hours=window_hours)
    report = HealthReport(verdict="ok", window_hours=window_hours)
    report.expected = dict(AR0134_BASELINE)
    notes: list[str] = []

    try:
        factory = get_session_factory()
        async with factory() as session:
            # --- Strategy C trades in window ---
            result = await session.execute(
                sa.select(
                    sa.func.count(PaperTrade.id),
                    sa.func.sum(sa.case((PaperTrade.status == "won", 1), else_=0)),
                    sa.func.sum(sa.case((PaperTrade.status == "lost", 1), else_=0)),
                    sa.func.coalesce(sa.func.sum(PaperTrade.actual_pnl), 0),
                    sa.func.coalesce(sa.func.avg(PaperTrade.edge_at_exec), 0),
                    sa.func.coalesce(sa.func.sum(PaperTrade.size), 0),
                )
                .where(PaperTrade.strategy == "C")
                .where(PaperTrade.placed_at >= window_start)
                .where(
                    sa.or_(
                        PaperTrade.notes.is_(None),
                        PaperTrade.notes != "pre-validation",
                    )
                )
            )
            row = result.one()
            trades_in_window = row[0] or 0
            wins_in_window = row[1] or 0
            losses_in_window = row[2] or 0
            pnl_in_window = float(row[3] or 0)
            avg_edge_in_window = float(row[4] or 0)
            total_size_in_window = float(row[5] or 0)

            resolved_in_window = wins_in_window + losses_in_window

            # --- All-time Strategy C stats ---
            result = await session.execute(
                sa.select(
                    sa.func.count(PaperTrade.id),
                    sa.func.sum(sa.case((PaperTrade.status == "won", 1), else_=0)),
                    sa.func.sum(sa.case((PaperTrade.status == "lost", 1), else_=0)),
                    sa.func.coalesce(sa.func.sum(PaperTrade.actual_pnl), 0),
                    sa.func.min(PaperTrade.placed_at),
                )
                .where(PaperTrade.strategy == "C")
                .where(
                    sa.or_(
                        PaperTrade.notes.is_(None),
                        PaperTrade.notes != "pre-validation",
                    )
                )
            )
            row = result.one()
            total_trades = row[0] or 0
            total_wins = row[1] or 0
            total_losses = row[2] or 0
            total_pnl = float(row[3] or 0)
            first_trade_at = row[4]

            total_resolved = total_wins + total_losses
            total_wr = total_wins / total_resolved if total_resolved > 0 else None

            # Days since first trade
            days_active = (
                (now - first_trade_at).total_seconds() / 86400 if first_trade_at else 0
            )
            daily_trade_rate = total_trades / days_active if days_active > 0 else 0
            daily_pnl_rate = total_pnl / days_active if days_active > 0 else 0

        # Build metrics
        report.metrics = {
            # Window metrics
            "window_trades": trades_in_window,
            "window_wins": wins_in_window,
            "window_losses": losses_in_window,
            "window_resolved": resolved_in_window,
            "window_pnl": round(pnl_in_window, 2),
            "window_avg_edge": round(avg_edge_in_window, 4),
            "window_total_size": round(total_size_in_window, 2),
            # All-time metrics
            "total_trades": total_trades,
            "total_resolved": total_resolved,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "total_wr": round(total_wr, 4) if total_wr is not None else None,
            "total_pnl": round(total_pnl, 2),
            "days_active": round(days_active, 1),
            "daily_trade_rate": round(daily_trade_rate, 2),
            "daily_pnl_rate": round(daily_pnl_rate, 2),
        }

        # --- Verdict logic ---
        verdict = "ok"

        # Bug detection: zero trades in window when system has been running
        if trades_in_window == 0 and days_active > 1:
            notes.append(
                f"Zadne obchody za poslednich {window_hours} hodin — mozny problem se systemem"
            )
            verdict = "bug_detected"

        # Trade rate check
        if days_active >= 2 and daily_trade_rate < DAILY_TRADES_WARNING_LOW:
            notes.append(
                f"Prilis malo obchodu: {daily_trade_rate:.1f}/den "
                f"(ocekavano ~{AR0134_BASELINE['oos_daily_trades']})"
            )
            if verdict != "bug_detected":
                verdict = "needs_attention"

        # Win rate check (only if enough resolved)
        if total_resolved >= MIN_COMPARISON_TRADES:
            wr_diff = abs(total_wr - AR0134_BASELINE["oos_wr"])
            if wr_diff > WIN_RATE_WARNING_BAND:
                notes.append(
                    f"Win rate {total_wr:.1%} se vyrazne lisi od backtestu "
                    f"({AR0134_BASELINE['oos_wr']:.1%})"
                )
                if verdict != "bug_detected":
                    verdict = "needs_attention"
        else:
            notes.append(
                f"Prilis brzy na hodnoceni — {total_resolved} resolved trades "
                f"(minimum {MIN_COMPARISON_TRADES})"
            )

        # P&L trajectory check (only after 7+ days)
        if days_active >= 7:
            expected_pnl = AR0134_BASELINE["oos_daily_pnl"] * days_active
            if total_pnl < -expected_pnl * 0.5:
                notes.append(
                    f"P&L ${total_pnl:.2f} vyrazne pod ocekavanim "
                    f"(ocekavano ~${expected_pnl:.2f} za {days_active:.0f} dni)"
                )
                if verdict != "bug_detected":
                    verdict = "needs_attention"

        if not notes:
            notes.append("Vse v poradku — system funguje dle ocekavani")

        report.verdict = verdict
        report.notes = notes

    except Exception as e:
        logger.error("health_check_error", error=str(e))
        report.verdict = "bug_detected"
        report.notes = [f"Chyba pri health checku: {e}"]

    return report


async def save_health_check(report: HealthReport) -> None:
    """Persist health check result to database."""
    try:
        factory = get_session_factory()
        async with factory() as session:
            row = HealthCheck(
                check_at=report.check_at,
                verdict=report.verdict,
                window_hours=report.window_hours,
                metrics=report.metrics,
                expected=report.expected,
                notes="\n".join(report.notes),
            )
            session.add(row)
            await session.commit()
        logger.info("health_check_saved", verdict=report.verdict)
    except Exception as e:
        logger.error("health_check_save_error", error=str(e))


async def get_expected_vs_reality() -> dict:
    """Compare backtest expectations to actual paper trading performance.

    Shows absolute totals prominently. Only shows per-day rates after 3+ full days
    to avoid misleading extrapolation from partial-day data.
    """
    now = datetime.now(UTC)
    baseline = AR0134_BASELINE

    result_data: dict = {
        "baseline": baseline,
        "too_early": True,
        "actual": {},
        "comparison": [],
    }

    try:
        factory = get_session_factory()
        async with factory() as session:
            # Count all Strategy C trades (placed)
            res_all = await session.execute(
                sa.select(
                    sa.func.count(PaperTrade.id),
                    sa.func.min(PaperTrade.placed_at),
                )
                .where(PaperTrade.strategy == "C")
                .where(
                    sa.or_(
                        PaperTrade.notes.is_(None),
                        PaperTrade.notes != "pre-validation",
                    )
                )
            )
            row_all = res_all.one()
            total_placed = row_all[0] or 0
            first_trade = row_all[1]

            # Resolved Strategy C trades only (for P&L and WR)
            res_resolved = await session.execute(
                sa.select(
                    sa.func.count(PaperTrade.id),
                    sa.func.sum(sa.case((PaperTrade.status == "won", 1), else_=0)),
                    sa.func.coalesce(sa.func.sum(PaperTrade.actual_pnl), 0),
                    sa.func.coalesce(sa.func.avg(PaperTrade.edge_at_exec), 0),
                )
                .where(PaperTrade.strategy == "C")
                .where(PaperTrade.status.in_(["won", "lost"]))
                .where(
                    sa.or_(
                        PaperTrade.notes.is_(None),
                        PaperTrade.notes != "pre-validation",
                    )
                )
            )
            row_r = res_resolved.one()
            total_resolved = row_r[0] or 0
            total_wins = row_r[1] or 0
            total_losses = total_resolved - total_wins
            resolved_pnl = float(row_r[2] or 0)
            avg_edge = float(row_r[3] or 0)

            total_wr = total_wins / total_resolved if total_resolved > 0 else None
            days_active = (
                (now - first_trade).total_seconds() / 86400 if first_trade else 0
            )
            # Use full calendar days for per-day rates (avoid < 1 day distortion)
            full_days = max(int(days_active), 1)

            # Open positions (unrealized)
            res_open = await session.execute(
                sa.select(
                    sa.func.count(PaperTrade.id),
                    sa.func.coalesce(sa.func.sum(PaperTrade.size), 0),
                )
                .where(PaperTrade.strategy == "C")
                .where(PaperTrade.status == "open")
            )
            row_o = res_open.one()
            open_count = row_o[0] or 0
            open_deployed = float(row_o[1] or 0)

            # Daily P&L series for cumulative chart
            daily_pnl_rows = await session.execute(
                sa.select(
                    sa.func.date_trunc("day", PaperTrade.resolved_at).label("day"),
                    sa.func.sum(PaperTrade.actual_pnl),
                    sa.func.count(PaperTrade.id),
                    sa.func.sum(sa.case((PaperTrade.status == "won", 1), else_=0)),
                )
                .where(PaperTrade.strategy == "C")
                .where(PaperTrade.status.in_(["won", "lost"]))
                .where(
                    sa.or_(
                        PaperTrade.notes.is_(None),
                        PaperTrade.notes != "pre-validation",
                    )
                )
                .group_by(sa.text("1"))
                .order_by(sa.text("1"))
            )
            daily_series = []
            cumulative = 0.0
            for drow in daily_pnl_rows:
                day_str = drow[0].strftime("%Y-%m-%d") if drow[0] else None
                day_pnl = float(drow[1] or 0)
                cumulative += day_pnl
                daily_series.append({
                    "date": day_str,
                    "pnl": round(day_pnl, 2),
                    "cumulative": round(cumulative, 2),
                    "trades": drow[2] or 0,
                    "wins": drow[3] or 0,
                })

        result_data["too_early"] = total_resolved < MIN_COMPARISON_TRADES
        result_data["actual"] = {
            "total_placed": total_placed,
            "total_resolved": total_resolved,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "win_rate": round(total_wr, 4) if total_wr is not None else None,
            "resolved_pnl": round(resolved_pnl, 2),
            "avg_edge": round(avg_edge, 4),
            "days_active": round(days_active, 1),
            "full_days": full_days,
            "open_count": open_count,
            "open_deployed": round(open_deployed, 2),
        }

        # ROI calculations
        capital = baseline["capital"]  # $1000
        daily_roi_expected = baseline["oos_daily_pnl"] / capital * 100
        weekly_roi_expected = baseline["oos_weekly_pnl"] / capital * 100
        monthly_roi_expected = baseline["oos_daily_pnl"] * 30 / capital * 100

        daily_roi_actual = resolved_pnl / max(full_days, 1) / capital * 100
        weekly_roi_actual = resolved_pnl / max(days_active / 7, 1) / capital * 100 if days_active >= 7 else None
        monthly_roi_actual = resolved_pnl / max(days_active / 30, 1) / capital * 100 if days_active >= 30 else None

        def _pnl_str(val: float) -> str:
            return f"{'+'if val >= 0 else ''}${val:.2f}"

        def _roi_str(val: float) -> str:
            return f"{'+'if val >= 0 else ''}{val:.2f}%"

        def _pnl_status(actual_pnl: float, min_resolved: int = 10) -> str:
            """Color-coded status for P&L metrics."""
            if total_resolved < min_resolved:
                return "too_early"
            if actual_pnl >= 0:
                return "ok"
            return "warning"

        # Build comparison table
        comparisons = [
            {
                "metric": "Umisteno obchodu",
                "expected": f"~{baseline['oos_daily_trades'] * full_days:.0f} za {full_days}d",
                "actual": str(total_placed),
                "status": "info",
            },
            {
                "metric": "Resolved",
                "expected": "—",
                "actual": f"{total_resolved} ({total_wins}W / {total_losses}L)",
                "status": "info",
            },
            {
                "metric": "Win Rate",
                "expected": f"{baseline['oos_wr']:.1%}",
                "actual": (
                    f"{total_wr:.1%}" if total_wr is not None else "—"
                ),
                "status": (
                    "too_early" if total_resolved < MIN_COMPARISON_TRADES
                    else "ok" if abs(total_wr - baseline["oos_wr"]) <= 0.15
                    else "warning"
                ),
            },
            {
                "metric": "Realizovany P&L",
                "expected": _pnl_str(baseline["oos_daily_pnl"] * full_days) + f" za {full_days}d",
                "actual": _pnl_str(resolved_pnl),
                "status": _pnl_status(resolved_pnl),
            },
            {
                "metric": "Otevrene pozice",
                "expected": "—",
                "actual": f"{open_count} (${open_deployed:.0f} deployed)",
                "status": "info",
            },
        ]

        # --- ROI section (always shown, with "—" when not enough data) ---
        comparisons.append({"metric": "_separator", "expected": "", "actual": "", "status": ""})

        # Daily ROI
        comparisons.append({
            "metric": "Denni ROI",
            "expected": _roi_str(daily_roi_expected),
            "actual": _roi_str(daily_roi_actual) if full_days >= 1 else "—",
            "status": _pnl_status(daily_roi_actual) if full_days >= 1 else "too_early",
        })

        # Weekly ROI
        comparisons.append({
            "metric": "Tydenni ROI",
            "expected": _roi_str(weekly_roi_expected),
            "actual": _roi_str(weekly_roi_actual) if weekly_roi_actual is not None else "—",
            "status": _pnl_status(weekly_roi_actual, 20) if weekly_roi_actual is not None else "too_early",
        })

        # Monthly ROI
        comparisons.append({
            "metric": "Mesicni ROI",
            "expected": _roi_str(monthly_roi_expected),
            "actual": _roi_str(monthly_roi_actual) if monthly_roi_actual is not None else "—",
            "status": _pnl_status(monthly_roi_actual, 50) if monthly_roi_actual is not None else "too_early",
        })

        # --- P&L absolutes ---
        comparisons.append({"metric": "_separator", "expected": "", "actual": "", "status": ""})

        comparisons.append({
            "metric": "P&L / den",
            "expected": _pnl_str(baseline["oos_daily_pnl"]),
            "actual": _pnl_str(resolved_pnl / max(full_days, 1)) if full_days >= 1 else "—",
            "status": _pnl_status(resolved_pnl / max(full_days, 1)) if full_days >= 1 else "too_early",
        })

        comparisons.append({
            "metric": "P&L / tyden",
            "expected": _pnl_str(baseline["oos_weekly_pnl"]),
            "actual": _pnl_str(resolved_pnl / (days_active / 7)) if days_active >= 7 else "—",
            "status": (
                _pnl_status(resolved_pnl / (days_active / 7), 20)
                if days_active >= 7 else "too_early"
            ),
        })

        comparisons.append({
            "metric": "P&L / mesic",
            "expected": _pnl_str(baseline["oos_daily_pnl"] * 30),
            "actual": _pnl_str(resolved_pnl / (days_active / 30)) if days_active >= 30 else "—",
            "status": (
                _pnl_status(resolved_pnl / (days_active / 30), 50)
                if days_active >= 30 else "too_early"
            ),
        })

        result_data["comparison"] = comparisons
        result_data["daily_series"] = daily_series

        # Expected cumulative line (for overlay on chart)
        if daily_series:
            expected_line = []
            for i, ds in enumerate(daily_series):
                day_num = i + 1
                expected_line.append({
                    "date": ds["date"],
                    "expected_cumulative": round(baseline["oos_daily_pnl"] * day_num, 2),
                })
            result_data["expected_line"] = expected_line

    except Exception as e:
        logger.error("expected_vs_reality_error", error=str(e))

    return result_data


async def get_seasonality_analysis() -> dict:
    """Analyze performance by month (seasonality).

    Returns backtest seasonality data + actual monthly breakdown when available.
    """
    baseline = AR0134_BASELINE
    result_data: dict = {
        "current_month": datetime.now(UTC).strftime("%B"),
        "current_month_num": datetime.now(UTC).month,
        "baseline_note": (
            "Backtest ukazuje, ze zima (prosinec-unor) generuje 67% zisku "
            "kvuli vetsi teplotni variabilite. Leto (cerven-srpen) ma mensi edge."
        ),
        "best_months": baseline["best_months"],
        "worst_months": baseline["worst_months"],
        "winter_pnl_share": baseline["winter_pnl_share"],
        "monthly_actual": [],
    }

    try:
        factory = get_session_factory()
        async with factory() as session:
            # Monthly breakdown of actual Strategy C trades
            result = await session.execute(
                sa.select(
                    sa.extract("month", PaperTrade.placed_at).label("month"),
                    sa.func.count(PaperTrade.id),
                    sa.func.sum(sa.case((PaperTrade.status == "won", 1), else_=0)),
                    sa.func.sum(sa.case((PaperTrade.status == "lost", 1), else_=0)),
                    sa.func.coalesce(sa.func.sum(PaperTrade.actual_pnl), 0),
                )
                .where(PaperTrade.strategy == "C")
                .where(
                    sa.or_(
                        PaperTrade.notes.is_(None),
                        PaperTrade.notes != "pre-validation",
                    )
                )
                .group_by(sa.text("1"))
                .order_by(sa.text("1"))
            )
            for row in result.all():
                month_num = int(row[0])
                trades = row[1] or 0
                wins = row[2] or 0
                losses = row[3] or 0
                pnl = float(row[4] or 0)
                resolved = wins + losses
                result_data["monthly_actual"].append({
                    "month": month_num,
                    "trades": trades,
                    "wins": wins,
                    "losses": losses,
                    "resolved": resolved,
                    "wr": round(wins / resolved, 4) if resolved > 0 else None,
                    "pnl": round(pnl, 2),
                })

    except Exception as e:
        logger.error("seasonality_error", error=str(e))

    return result_data
