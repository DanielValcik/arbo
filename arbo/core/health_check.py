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
from sqlalchemy.ext.asyncio import AsyncSession

from arbo.utils.db import HealthCheck, PaperTrade, get_session_factory
from arbo.utils.logger import get_logger

logger = get_logger("health_check")


# ================================================================
# AR-0134 Backtest Baseline (source of truth for expectations)
# ================================================================

AR0134_BASELINE = {
    "model": "C1f-ensemble",
    "score": 139.7,
    # Train performance (IS)
    "train_trades": 219,
    "train_wr": 0.365,
    "train_avg_edge": 0.15,
    # OOS performance (walk-forward 3-fold)
    "oos_trades": 153,
    "oos_wr": 0.376,
    "oos_pnl": 1877.0,
    "oos_days": 70,
    "oos_daily_pnl": 26.8,  # $1877 / 70 days
    "oos_weekly_pnl": 187.7,  # $1877 / 10 weeks
    "oos_daily_trades": 3.0,  # ~3-5 per day
    # Walk-forward
    "wf_pnl": 1877.0,
    # Risk
    "max_drawdown_pct": 0.156,
    # Strategy params
    "capital": 1000.0,
    "kelly_fraction": 0.25,
    "cities_active": 19,  # all except lucknow, wellington (ensemble covers 19)
    "excluded_cities": ["Lucknow", "Wellington"],
    # Seasonality (from backtest analysis)
    "winter_pnl_share": 0.67,  # Dec+Jan+Feb = 67% of total PnL
    "best_months": [12, 1, 2],
    "worst_months": [6, 7, 8],
}

C2_BASELINE = {
    "model": "EMOS-Exit-Fusion",
    "score": 138.1,
    # Train performance (IS)
    "train_trades": 1878,
    "train_wr": 0.541,
    "train_avg_edge": 0.06,
    # OOS performance (walk-forward 3-fold)
    "oos_trades": 496,
    "oos_wr": 0.537,
    "oos_pnl": 3411.0,
    "oos_days": 70,
    "oos_daily_pnl": 48.7,  # $3411 / 70 days
    "oos_weekly_pnl": 341.1,  # $3411 / 10 weeks
    "oos_daily_trades": 7.1,  # ~496 / 70 days
    # Walk-forward
    "wf_pnl": 3411.0,
    # Risk
    "max_drawdown_pct": 0.083,
    # Strategy params
    "capital": 1000.0,
    "kelly_fraction": 0.25,
    "cities_active": 15,
    "excluded_cities": ["São Paulo", "Tel Aviv", "Tokyo", "Lucknow"],
    # Exit model
    "exit_type": "edge-based",
    "min_hold_edge": 0.05,
    "profit_target_abs": 0.15,
    "exit_slippage_pct": 0.06,
    "emos_window": 21,
    "emos_method": "rolling_mae",
}

# Tolerance bands for health check verdicts
MIN_COMPARISON_TRADES = 20  # Need at least this many for meaningful comparison
WIN_RATE_WARNING_BAND = 0.15  # ±15pp from baseline before warning
DAILY_TRADES_WARNING_LOW = 0.5  # Less than 0.5 trades/day = suspicious

# Per-strategy tracking thresholds. Rationale in LEARNINGS.md (Global §G-17).
# If a strategy hasn't traded in RETIRED_AFTER_DAYS, it's treated as intentionally
# disabled — no verdict escalation, informational note only.
# DORMANT_AFTER_DAYS covers the gap where a strategy paused but may return.
ACTIVITY_DISCOVERY_DAYS = 14  # strategies with trades in last N days are auto-tracked
DORMANT_AFTER_DAYS = 3        # no trade >3d → dormant (informational, no bug)
RETIRED_AFTER_DAYS = 14       # no trade >14d → retired (excluded from overall verdict)

# Strategies that have a backtest baseline — only these get WR / P&L-trajectory checks.
# All other strategies get activity-only checks (is it still producing trades?).
STRATEGY_BASELINES: dict[str, dict] = {
    "C": AR0134_BASELINE,
    "C2": C2_BASELINE,
}

# Verdict severity order (worse → higher)
_VERDICT_RANK = {"ok": 0, "needs_attention": 1, "bug_detected": 2}


@dataclass
class HealthReport:
    """Result of a health check run."""

    verdict: str  # ok, needs_attention, bug_detected
    check_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    window_hours: int = 12
    metrics: dict = field(default_factory=dict)
    expected: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _escalate(current: str, new: str) -> str:
    """Return the more severe of two verdicts."""
    return new if _VERDICT_RANK[new] > _VERDICT_RANK[current] else current


async def _collect_strategy_stats(
    session: AsyncSession,
    strategy: str,
    window_start: datetime,
    now: datetime,
) -> dict:
    """Gather window + all-time stats for one strategy."""
    _won = PaperTrade.status == "won"
    _lost = PaperTrade.status == "lost"
    _not_preval = sa.or_(
        PaperTrade.notes.is_(None), PaperTrade.notes != "pre-validation"
    )

    # Window-level (placed in window)
    res = await session.execute(
        sa.select(
            sa.func.count(PaperTrade.id),
            sa.func.sum(sa.case((_won, 1), else_=0)),
            sa.func.sum(sa.case((_lost, 1), else_=0)),
            sa.func.coalesce(sa.func.sum(PaperTrade.actual_pnl), 0),
        )
        .where(PaperTrade.strategy == strategy)
        .where(PaperTrade.placed_at >= window_start)
        .where(_not_preval)
    )
    w_row = res.one()
    window_trades = w_row[0] or 0
    window_wins = w_row[1] or 0
    window_losses = w_row[2] or 0
    window_pnl = float(w_row[3] or 0)

    # Also count resolved-in-window (placed earlier, resolved now)
    res = await session.execute(
        sa.select(sa.func.count(PaperTrade.id))
        .where(PaperTrade.strategy == strategy)
        .where(PaperTrade.resolved_at >= window_start)
        .where(PaperTrade.status.in_(["won", "lost", "sold"]))
    )
    resolved_in_window = max(window_wins + window_losses, res.scalar() or 0)

    # All-time
    res = await session.execute(
        sa.select(
            sa.func.count(PaperTrade.id),
            sa.func.sum(sa.case((_won, 1), else_=0)),
            sa.func.sum(sa.case((_lost, 1), else_=0)),
            sa.func.coalesce(sa.func.sum(PaperTrade.actual_pnl), 0),
            sa.func.min(PaperTrade.placed_at),
            sa.func.max(PaperTrade.placed_at),
            sa.func.sum(
                sa.case((PaperTrade.status == "open", 1), else_=0)
            ),
        )
        .where(PaperTrade.strategy == strategy)
        .where(_not_preval)
    )
    t_row = res.one()
    total_trades = t_row[0] or 0
    total_wins = t_row[1] or 0
    total_losses = t_row[2] or 0
    total_pnl = float(t_row[3] or 0)
    first_trade_at = t_row[4]
    last_trade_at = t_row[5]
    open_positions = t_row[6] or 0

    total_resolved = total_wins + total_losses
    total_wr = total_wins / total_resolved if total_resolved > 0 else None
    days_active = (
        (now - first_trade_at).total_seconds() / 86400 if first_trade_at else 0
    )
    days_since_last_trade = (
        (now - last_trade_at).total_seconds() / 86400 if last_trade_at else None
    )
    daily_trade_rate = total_trades / days_active if days_active > 0 else 0

    # Activity classification — drives verdict escalation logic
    if days_since_last_trade is None:
        status = "never_traded"
    elif days_since_last_trade > RETIRED_AFTER_DAYS and open_positions == 0:
        status = "retired"
    elif days_since_last_trade > DORMANT_AFTER_DAYS:
        status = "dormant"
    else:
        status = "active"

    return {
        "strategy": strategy,
        "status": status,
        "window_trades": window_trades,
        "window_wins": window_wins,
        "window_losses": window_losses,
        "window_resolved": resolved_in_window,
        "window_pnl": round(window_pnl, 2),
        "total_trades": total_trades,
        "total_resolved": total_resolved,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "total_wr": round(total_wr, 4) if total_wr is not None else None,
        "total_pnl": round(total_pnl, 2),
        "open_positions": open_positions,
        "days_active": round(days_active, 1),
        "days_since_last_trade": (
            round(days_since_last_trade, 2) if days_since_last_trade is not None else None
        ),
        "daily_trade_rate": round(daily_trade_rate, 2),
        "first_trade_at": first_trade_at.isoformat() if first_trade_at else None,
        "last_trade_at": last_trade_at.isoformat() if last_trade_at else None,
    }


def _evaluate_strategy(stats: dict, window_hours: int) -> tuple[str, list[str]]:
    """Produce per-strategy verdict + human notes.

    retired / never_traded → ok verdict, no notes (filtered out by caller).
    dormant → informational note, no verdict escalation (strategy paused).
    active → full checks (window activity, WR vs baseline, P&L trajectory).
    """
    strategy = stats["strategy"]
    status = stats["status"]

    if status in ("retired", "never_traded"):
        return "ok", []

    if status == "dormant":
        days = stats["days_since_last_trade"]
        return "ok", [
            f"strategie neaktivní {days:.1f}d (posledni trade). "
            f"Za >{RETIRED_AFTER_DAYS}d bude oznacena jako retired."
        ]

    # status == "active"
    verdict = "ok"
    notes: list[str] = []
    baseline = STRATEGY_BASELINES.get(strategy)

    # Bug detection: zero activity (placed AND resolved both 0) in window
    if (
        stats["window_trades"] == 0
        and stats["window_resolved"] == 0
        and stats["days_active"] > 1
    ):
        notes.append(
            f"zadna aktivita za poslednich {window_hours}h — system moznym problemem"
        )
        verdict = _escalate(verdict, "bug_detected")

    # Trade-rate check (only strategies with a baseline — we know what to expect)
    if (
        baseline
        and stats["days_active"] >= 2
        and stats["daily_trade_rate"] < DAILY_TRADES_WARNING_LOW
    ):
        notes.append(
            f"prilis malo obchodu: {stats['daily_trade_rate']:.1f}/den "
            f"(backtest ~{baseline['oos_daily_trades']})"
        )
        verdict = _escalate(verdict, "needs_attention")

    # Win rate check
    if baseline and stats["total_resolved"] >= MIN_COMPARISON_TRADES:
        wr_diff = abs((stats["total_wr"] or 0) - baseline["oos_wr"])
        if wr_diff > WIN_RATE_WARNING_BAND:
            notes.append(
                f"win rate {stats['total_wr']:.1%} se vyrazne lisi "
                f"od backtestu ({baseline['oos_wr']:.1%})"
            )
            verdict = _escalate(verdict, "needs_attention")

    # P&L trajectory (baseline + 7d+ of history)
    if baseline and stats["days_active"] >= 7:
        expected_pnl = baseline["oos_daily_pnl"] * stats["days_active"]
        if stats["total_pnl"] < -expected_pnl * 0.5:
            notes.append(
                f"P&L ${stats['total_pnl']:.2f} vyrazne pod ocekavanim "
                f"(~${expected_pnl:.2f} za {stats['days_active']:.0f}d)"
            )
            verdict = _escalate(verdict, "needs_attention")

    return verdict, notes


async def run_health_check(window_hours: int = 12) -> HealthReport:
    """Per-strategy health check. Auto-discovers strategies with recent activity.

    - Strategies with activity in the last ACTIVITY_DISCOVERY_DAYS are evaluated.
    - Strategies dormant (>3d) are noted but don't escalate verdict.
    - Strategies retired (>14d, no open positions) are silent.
    - C and C2 additionally get WR/P&L-vs-baseline checks.

    Args:
        window_hours: Short-term window for zero-activity detection (default 12h).

    Returns:
        HealthReport with aggregated verdict, notes, and per_strategy metrics.
        Top-level metrics keys (total_trades, total_wr, total_pnl, ...) are
        kept for Strategy C for dashboard back-compat.
    """
    now = datetime.now(UTC)
    window_start = now - timedelta(hours=window_hours)
    report = HealthReport(verdict="ok", window_hours=window_hours)
    report.expected = dict(AR0134_BASELINE)  # back-compat field
    notes: list[str] = []

    try:
        factory = get_session_factory()
        async with factory() as session:
            # Auto-discover strategies with activity in discovery window
            discovery_cutoff = now - timedelta(days=ACTIVITY_DISCOVERY_DAYS)
            res = await session.execute(
                sa.select(sa.distinct(PaperTrade.strategy))
                .where(PaperTrade.placed_at >= discovery_cutoff)
                .where(
                    sa.or_(
                        PaperTrade.notes.is_(None),
                        PaperTrade.notes != "pre-validation",
                    )
                )
            )
            active_strategies: set[str] = {r[0] for r in res.all() if r[0]}
            # Always include baseline strategies (so retirement is visible)
            tracked = sorted(active_strategies | set(STRATEGY_BASELINES))

            per_strategy: dict[str, dict] = {}
            for strat in tracked:
                per_strategy[strat] = await _collect_strategy_stats(
                    session, strat, window_start, now
                )

        # Per-strategy evaluation + aggregate rollup
        overall_verdict = "ok"
        any_active = False
        for strat, stats in per_strategy.items():
            v, strat_notes = _evaluate_strategy(stats, window_hours)
            stats["verdict"] = v
            stats["notes"] = strat_notes
            if stats["status"] == "active":
                any_active = True
            if stats["status"] in ("retired", "never_traded"):
                continue  # silent — operator intentionally disabled it
            if strat_notes:
                notes.extend(f"[{strat}] {n}" for n in strat_notes)
            overall_verdict = _escalate(overall_verdict, v)

        if not any_active:
            notes.append(
                "zadne aktivne obchodovane strategie — "
                "zkontrolovat zda je to ocekavany stav"
            )
            # Not a bug — could be legitimate system pause
            overall_verdict = _escalate(overall_verdict, "needs_attention")

        if not notes:
            notes.append("vse v poradku — vsechny aktivni strategie bezi dle ocekavani")

        # Build flat metrics. Dashboard timeline reads top-level C keys
        # (total_trades, total_wr, total_pnl); preserve them.
        c_stats = per_strategy.get("C", {})
        report.metrics = {
            # Back-compat: Strategy C top-level (dashboard.html:3174)
            "window_trades": c_stats.get("window_trades", 0),
            "window_wins": c_stats.get("window_wins", 0),
            "window_losses": c_stats.get("window_losses", 0),
            "window_resolved": c_stats.get("window_resolved", 0),
            "window_pnl": c_stats.get("window_pnl", 0),
            "total_trades": c_stats.get("total_trades", 0),
            "total_resolved": c_stats.get("total_resolved", 0),
            "total_wins": c_stats.get("total_wins", 0),
            "total_losses": c_stats.get("total_losses", 0),
            "total_wr": c_stats.get("total_wr"),
            "total_pnl": c_stats.get("total_pnl", 0),
            "days_active": c_stats.get("days_active", 0),
            "daily_trade_rate": c_stats.get("daily_trade_rate", 0),
            # New: per-strategy breakdown + tracking summary
            "per_strategy": per_strategy,
            "tracked": tracked,
            "active_strategies": sorted(
                s for s, v in per_strategy.items() if v["status"] == "active"
            ),
            "dormant_strategies": sorted(
                s for s, v in per_strategy.items() if v["status"] == "dormant"
            ),
            "retired_strategies": sorted(
                s for s, v in per_strategy.items() if v["status"] == "retired"
            ),
        }
        report.verdict = overall_verdict
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
                    sa.func.sum(
                        sa.case((PaperTrade.status == "won", 1), (sa.and_(PaperTrade.status == "sold", PaperTrade.actual_pnl >= 0), 1), else_=0)
                    ),
                    sa.func.coalesce(sa.func.sum(PaperTrade.actual_pnl), 0),
                    sa.func.coalesce(sa.func.avg(PaperTrade.edge_at_exec), 0),
                )
                .where(PaperTrade.strategy == "C")
                .where(PaperTrade.status.in_(["won", "lost", "sold"]))
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
                    sa.func.sum(
                        sa.case((PaperTrade.status == "won", 1), (sa.and_(PaperTrade.status == "sold", PaperTrade.actual_pnl >= 0), 1), else_=0)
                    ),
                )
                .where(PaperTrade.strategy == "C")
                .where(PaperTrade.status.in_(["won", "lost", "sold"]))
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


async def get_expected_vs_reality_c2() -> dict:
    """Compare C2 model expectations to actual paper trading performance."""
    now = datetime.now(UTC)
    baseline = C2_BASELINE

    result_data: dict = {
        "baseline": baseline,
        "too_early": True,
        "actual": {},
        "comparison": [],
    }

    try:
        factory = get_session_factory()
        async with factory() as session:
            # Count all C2 trades
            res_all = await session.execute(
                sa.select(
                    sa.func.count(PaperTrade.id),
                    sa.func.min(PaperTrade.placed_at),
                )
                .where(PaperTrade.strategy == "C2")
                .where(sa.or_(PaperTrade.notes.is_(None), PaperTrade.notes != "pre-validation"))
            )
            row_all = res_all.one()
            total_placed = row_all[0] or 0
            first_trade = row_all[1]

            # Resolved C2 trades
            res_resolved = await session.execute(
                sa.select(
                    sa.func.count(PaperTrade.id),
                    sa.func.sum(sa.case((PaperTrade.status == "won", 1), (sa.and_(PaperTrade.status == "sold", PaperTrade.actual_pnl >= 0), 1), else_=0)),
                    sa.func.coalesce(sa.func.sum(PaperTrade.actual_pnl), 0),
                    sa.func.coalesce(sa.func.avg(PaperTrade.edge_at_exec), 0),
                )
                .where(PaperTrade.strategy == "C2")
                .where(PaperTrade.status.in_(["won", "lost", "sold"]))
                .where(sa.or_(PaperTrade.notes.is_(None), PaperTrade.notes != "pre-validation"))
            )
            row_r = res_resolved.one()
            total_resolved = row_r[0] or 0
            total_wins = row_r[1] or 0
            total_losses = total_resolved - total_wins
            resolved_pnl = float(row_r[2] or 0)

            total_wr = total_wins / total_resolved if total_resolved > 0 else None
            days_active = (now - first_trade).total_seconds() / 86400 if first_trade else 0
            full_days = max(int(days_active), 1)

            # Open positions
            res_open = await session.execute(
                sa.select(
                    sa.func.count(PaperTrade.id),
                    sa.func.coalesce(sa.func.sum(PaperTrade.size), 0),
                )
                .where(PaperTrade.strategy == "C2")
                .where(PaperTrade.status == "open")
            )
            row_o = res_open.one()
            open_count = row_o[0] or 0
            open_deployed = float(row_o[1] or 0)

            # Daily P&L series for chart
            daily_pnl_rows = await session.execute(
                sa.select(
                    sa.func.date_trunc("day", PaperTrade.resolved_at).label("day"),
                    sa.func.sum(PaperTrade.actual_pnl),
                    sa.func.count(PaperTrade.id),
                    sa.func.sum(sa.case((PaperTrade.status == "won", 1), (sa.and_(PaperTrade.status == "sold", PaperTrade.actual_pnl >= 0), 1), else_=0)),
                )
                .where(PaperTrade.strategy == "C2")
                .where(PaperTrade.status.in_(["won", "lost", "sold"]))
                .where(sa.or_(PaperTrade.notes.is_(None), PaperTrade.notes != "pre-validation"))
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
            "days_active": round(days_active, 1),
            "full_days": full_days,
            "open_count": open_count,
            "open_deployed": round(open_deployed, 2),
        }

        capital = baseline["capital"]
        daily_roi_expected = baseline["oos_daily_pnl"] / capital * 100
        weekly_roi_expected = baseline["oos_weekly_pnl"] / capital * 100
        monthly_roi_expected = baseline["oos_daily_pnl"] * 30 / capital * 100
        daily_roi_actual = resolved_pnl / max(full_days, 1) / capital * 100

        def _pnl_str(val: float) -> str:
            return f"{'+'if val >= 0 else ''}${val:.2f}"

        def _roi_str(val: float) -> str:
            return f"{'+'if val >= 0 else ''}{val:.2f}%"

        def _pnl_status(actual_pnl: float, min_resolved: int = 10) -> str:
            if total_resolved < min_resolved:
                return "too_early"
            return "ok" if actual_pnl >= 0 else "warning"

        comparisons = [
            {"metric": "Umisteno obchodu", "expected": f"~{baseline['oos_daily_trades'] * full_days:.0f} za {full_days}d", "actual": str(total_placed), "status": "info"},
            {"metric": "Resolved", "expected": "—", "actual": f"{total_resolved} ({total_wins}W / {total_losses}L)", "status": "info"},
            {"metric": "Win Rate", "expected": f"{baseline['oos_wr']:.1%}", "actual": f"{total_wr:.1%}" if total_wr is not None else "—", "status": "too_early" if total_resolved < MIN_COMPARISON_TRADES else "ok" if total_wr and abs(total_wr - baseline["oos_wr"]) <= 0.15 else "warning"},
            {"metric": "Realizovany P&L", "expected": _pnl_str(baseline["oos_daily_pnl"] * full_days) + f" za {full_days}d", "actual": _pnl_str(resolved_pnl), "status": _pnl_status(resolved_pnl)},
            {"metric": "Otevrene pozice", "expected": "—", "actual": f"{open_count} (${open_deployed:.0f} deployed)", "status": "info"},
            {"metric": "_separator", "expected": "", "actual": "", "status": ""},
            {"metric": "Denni ROI", "expected": _roi_str(daily_roi_expected), "actual": _roi_str(daily_roi_actual), "status": _pnl_status(resolved_pnl)},
            {"metric": "Tydenni ROI", "expected": _roi_str(weekly_roi_expected), "actual": "—" if days_active < 7 else _roi_str(resolved_pnl / (days_active / 7) / capital * 100), "status": "too_early" if days_active < 7 else _pnl_status(resolved_pnl)},
            {"metric": "P&L / den", "expected": _pnl_str(baseline["oos_daily_pnl"]), "actual": _pnl_str(resolved_pnl / max(full_days, 1)), "status": _pnl_status(resolved_pnl)},
        ]

        result_data["comparison"] = comparisons
        result_data["daily_series"] = daily_series

    except Exception as e:
        logger.error("expected_vs_reality_c2_error", error=str(e))

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
                    sa.func.sum(sa.case((PaperTrade.status == "won", 1), (sa.and_(PaperTrade.status == "sold", PaperTrade.actual_pnl >= 0), 1), else_=0)),
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
