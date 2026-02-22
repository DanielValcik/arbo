"""Web dashboard for Arbo trading system monitoring.

FastAPI app serving a single-page dashboard with real-time API endpoints.
Integrated into the orchestrator as an internal task (same pattern as Slack bot).
"""

from __future__ import annotations

import contextlib
import os
import secrets
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import psutil
import sqlalchemy as sa
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

from arbo.utils.logger import get_logger

logger = get_logger("web_dashboard")


# ---------------------------------------------------------------------------
# Shared state — injected by orchestrator at init
# ---------------------------------------------------------------------------


class DashboardState:
    """Shared state injected by orchestrator."""

    orchestrator: Any = None


state = DashboardState()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Arbo Dashboard", docs_url=None, redoc_url=None)
security = HTTPBasic()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _verify_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """Verify HTTP Basic Auth credentials."""
    expected_user = os.getenv("DASHBOARD_USER", "arbo")
    expected_pass = os.getenv("DASHBOARD_PASSWORD", "")

    if not expected_pass:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dashboard password not configured",
        )

    user_ok = secrets.compare_digest(credentials.username.encode(), expected_user.encode())
    pass_ok = secrets.compare_digest(credentials.password.encode(), expected_pass.encode())

    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# ---------------------------------------------------------------------------
# Helper: decimal-safe JSON serialization
# ---------------------------------------------------------------------------


def _dec(val: Any) -> float | None:
    """Convert Decimal/numeric to float for JSON serialization."""
    if val is None:
        return None
    if isinstance(val, Decimal):
        return float(val)
    return float(val)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def dashboard_page(
    request: Request, _user: str = Depends(_verify_credentials)
) -> HTMLResponse:
    """Serve the main dashboard HTML page."""
    orch = state.orchestrator
    mode = orch._mode if orch else "unknown"
    return templates.TemplateResponse("dashboard.html", {"request": request, "mode": mode})


@app.get("/api/portfolio")
async def api_portfolio(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Portfolio summary: balance, P&L, snapshot history for chart."""
    orch = state.orchestrator
    if orch is None:
        return {"error": "Orchestrator not available"}

    engine = orch._paper_engine
    balance = _dec(engine.balance) if engine else 0.0
    total_value = _dec(engine.total_value) if engine else 0.0
    initial = _dec(orch._capital)

    # Query DB for daily/weekly P&L and snapshots
    daily_pnl = 0.0
    weekly_pnl = 0.0
    snapshots: list[dict[str, Any]] = []

    try:
        from arbo.utils.db import PaperSnapshot, PaperTrade, get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            # Daily P&L
            today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
            result = await session.execute(
                sa.select(sa.func.coalesce(sa.func.sum(PaperTrade.actual_pnl), 0)).where(
                    PaperTrade.placed_at >= today,
                    PaperTrade.status.in_(["won", "lost"]),
                )
            )
            daily_pnl = _dec(result.scalar()) or 0.0

            # Weekly P&L
            week_start = today - timedelta(days=today.weekday())
            result = await session.execute(
                sa.select(sa.func.coalesce(sa.func.sum(PaperTrade.actual_pnl), 0)).where(
                    PaperTrade.placed_at >= week_start,
                    PaperTrade.status.in_(["won", "lost"]),
                )
            )
            weekly_pnl = _dec(result.scalar()) or 0.0

            # Snapshots for chart (last 7 days)
            result = await session.execute(
                sa.select(
                    PaperSnapshot.balance,
                    PaperSnapshot.total_value,
                    PaperSnapshot.unrealized_pnl,
                    PaperSnapshot.snapshot_at,
                )
                .order_by(PaperSnapshot.snapshot_at.desc())
                .limit(168)
            )
            for row in result.all():
                snapshots.append(
                    {
                        "balance": _dec(row.balance),
                        "total_value": _dec(row.total_value),
                        "unrealized_pnl": _dec(row.unrealized_pnl),
                        "timestamp": row.snapshot_at.isoformat() if row.snapshot_at else None,
                    }
                )
            snapshots.reverse()
    except Exception as e:
        logger.warning("portfolio_query_error", error=str(e))

    total_pnl = (total_value or 0.0) - (initial or 0.0)
    roi_pct = (total_pnl / initial * 100) if initial else 0.0

    return {
        "balance": balance,
        "total_value": total_value,
        "initial_capital": initial,
        "daily_pnl": daily_pnl,
        "weekly_pnl": weekly_pnl,
        "total_pnl": round(total_pnl, 2),
        "roi_pct": round(roi_pct, 2),
        "snapshots": snapshots,
    }


@app.get("/api/positions")
async def api_positions(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Open positions with market names."""
    orch = state.orchestrator
    if orch is None:
        return {"positions": []}

    positions: list[dict[str, Any]] = []
    try:
        from arbo.utils.db import Market, PaperPosition, get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                sa.select(PaperPosition, Market.question)
                .outerjoin(Market, PaperPosition.market_condition_id == Market.condition_id)
                .order_by(PaperPosition.opened_at.desc())
            )
            for row in result.all():
                pos = row[0]
                question = row[1] or pos.market_condition_id[:20]
                positions.append(
                    {
                        "market": question,
                        "condition_id": pos.market_condition_id,
                        "token_id": pos.token_id,
                        "side": pos.side,
                        "size": _dec(pos.size),
                        "avg_price": _dec(pos.avg_price),
                        "current_price": _dec(pos.current_price),
                        "unrealized_pnl": _dec(pos.unrealized_pnl),
                        "layer": pos.layer,
                        "opened_at": pos.opened_at.isoformat() if pos.opened_at else None,
                    }
                )
    except Exception as e:
        logger.warning("positions_query_error", error=str(e))

    return {"positions": positions}


@app.get("/api/signals")
async def api_signals(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Last 100 signals with market names."""
    signals: list[dict[str, Any]] = []
    try:
        from arbo.utils.db import Market, get_session_factory
        from arbo.utils.db import Signal as DBSignal

        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                sa.select(DBSignal, Market.question)
                .outerjoin(Market, DBSignal.market_condition_id == Market.condition_id)
                .order_by(DBSignal.detected_at.desc())
                .limit(100)
            )
            for row in result.all():
                sig = row[0]
                question = row[1] or sig.market_condition_id[:20]
                details = sig.details or {}
                reason = details.get("reason", details.get("match_type", ""))
                signals.append(
                    {
                        "timestamp": sig.detected_at.isoformat() if sig.detected_at else None,
                        "layer": sig.layer,
                        "market": question,
                        "direction": sig.direction,
                        "edge": _dec(sig.edge),
                        "confidence": _dec(sig.confidence),
                        "confluence_score": sig.confluence_score,
                        "reason": str(reason)[:100] if reason else "",
                    }
                )
    except Exception as e:
        logger.warning("signals_query_error", error=str(e))

    return {"signals": signals}


@app.get("/api/layers")
async def api_layers(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Per-layer status: green/yellow/red, heartbeat, restart count, 24h signal count."""
    orch = state.orchestrator
    if orch is None:
        return {"layers": []}

    now = time.monotonic()
    layers: list[dict[str, Any]] = []

    # Query signal counts per layer (last 24h)
    signal_counts: dict[int, int] = {}
    try:
        from arbo.utils.db import Signal as DBSignal
        from arbo.utils.db import get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            cutoff = datetime.now(UTC) - timedelta(hours=24)
            result = await session.execute(
                sa.select(DBSignal.layer, sa.func.count())
                .where(DBSignal.detected_at > cutoff)
                .group_by(DBSignal.layer)
            )
            for row in result.all():
                signal_counts[row[0]] = row[1]
    except Exception as e:
        logger.warning("layer_signal_count_error", error=str(e))

    for name, ls in orch._layers.items():
        heartbeat_ago = now - ls.last_heartbeat
        task_running = ls.task is not None and not ls.task.done()

        # Determine status color
        if ls.permanent_stop or (ls.task and ls.task.done()):
            status_color = "red"
        elif heartbeat_ago > 300 or ls.restart_count > 3:
            status_color = "yellow"
            if heartbeat_ago > 300:
                status_color = "red"
        elif ls.restart_count > 0 and ls.restart_count <= 3:
            status_color = "yellow"
        else:
            status_color = "green"

        # Extract layer number from name
        layer_num = 0
        if name.startswith("L") and "_" in name:
            with contextlib.suppress(ValueError, IndexError):
                layer_num = int(name.split("_")[0][1:])

        layers.append(
            {
                "name": name,
                "status": status_color,
                "running": task_running,
                "enabled": ls.enabled,
                "permanent_stop": ls.permanent_stop,
                "heartbeat_ago_s": int(heartbeat_ago),
                "restart_count": ls.restart_count,
                "signals_24h": signal_counts.get(layer_num, 0),
            }
        )

    return {"layers": layers}


@app.get("/api/risk")
async def api_risk(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Risk metrics: daily/weekly loss utilization, category exposure."""
    orch = state.orchestrator
    if orch is None:
        return {"error": "Orchestrator not available"}

    rm = orch._risk_manager
    if rm is None:
        return {"error": "Risk manager not available"}

    capital = _dec(orch._capital) or 1.0

    # Get exposure state from risk manager
    daily_pnl = _dec(getattr(rm, "_daily_pnl", Decimal("0"))) or 0.0
    weekly_pnl = _dec(getattr(rm, "_weekly_pnl", Decimal("0"))) or 0.0
    category_exposure: dict[str, float] = {}
    raw_cat = getattr(rm, "_category_exposure", {})
    for cat, val in raw_cat.items():
        category_exposure[cat] = _dec(val) or 0.0

    # Risk limits (hardcoded constants)
    daily_limit = capital * 0.10
    weekly_limit = capital * 0.20

    daily_utilization = abs(daily_pnl) / daily_limit * 100 if daily_limit else 0.0
    weekly_utilization = abs(weekly_pnl) / weekly_limit * 100 if weekly_limit else 0.0

    return {
        "daily_pnl": daily_pnl,
        "daily_limit": daily_limit,
        "daily_utilization_pct": round(min(daily_utilization, 100), 1),
        "weekly_pnl": weekly_pnl,
        "weekly_limit": weekly_limit,
        "weekly_utilization_pct": round(min(weekly_utilization, 100), 1),
        "category_exposure": category_exposure,
        "max_category_pct": 30.0,
        "shutdown_active": getattr(rm, "_shutdown", False),
    }


@app.get("/api/infra")
async def api_infra(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Infrastructure: uptime, system resources, Odds API quota."""
    orch = state.orchestrator
    uptime_s = int(time.monotonic() - orch._start_time) if orch and orch._start_time else 0

    # System resources
    process = psutil.Process()
    mem_info = process.memory_info()
    cpu_pct = process.cpu_percent(interval=0)

    # Odds API quota
    odds_quota: dict[str, Any] = {"remaining": "N/A", "total": "N/A"}
    if orch and orch._odds_client:
        remaining = getattr(orch._odds_client, "_remaining_quota", None)
        if remaining is not None:
            odds_quota = {"remaining": remaining, "total": 20000}

    # Confluence threshold
    confluence_threshold = 2  # default
    if orch and hasattr(orch, "_config") and hasattr(orch._config, "confluence"):
        confluence_threshold = orch._config.confluence.min_score

    return {
        "uptime_s": uptime_s,
        "uptime_human": _format_uptime(uptime_s),
        "memory_mb": round(mem_info.rss / 1024 / 1024, 1),
        "cpu_pct": round(cpu_pct, 1),
        "system_memory_pct": round(psutil.virtual_memory().percent, 1),
        "odds_api_quota": odds_quota,
        "llm_degraded": orch._llm_degraded if orch else False,
        "mode": orch._mode if orch else "unknown",
        "confluence_threshold": confluence_threshold,
    }


@app.get("/api/trades")
async def api_trades(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Last 50 trades with market names."""
    trades: list[dict[str, Any]] = []
    try:
        from arbo.utils.db import Market, PaperTrade, get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                sa.select(PaperTrade, Market.question)
                .outerjoin(Market, PaperTrade.market_condition_id == Market.condition_id)
                .order_by(PaperTrade.placed_at.desc())
                .limit(50)
            )
            for row in result.all():
                trade = row[0]
                question = row[1] or trade.market_condition_id[:20]
                trades.append(
                    {
                        "id": trade.id,
                        "market": question,
                        "layer": trade.layer,
                        "side": trade.side,
                        "price": _dec(trade.price),
                        "size": _dec(trade.size),
                        "edge": _dec(trade.edge_at_exec),
                        "confluence_score": trade.confluence_score,
                        "status": trade.status,
                        "actual_pnl": _dec(trade.actual_pnl),
                        "fee_paid": _dec(trade.fee_paid),
                        "placed_at": trade.placed_at.isoformat() if trade.placed_at else None,
                        "resolved_at": (
                            trade.resolved_at.isoformat() if trade.resolved_at else None
                        ),
                    }
                )
    except Exception as e:
        logger.warning("trades_query_error", error=str(e))

    return {"trades": trades}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_uptime(seconds: int) -> str:
    """Format uptime seconds into human-readable string."""
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, _ = divmod(remainder, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return " ".join(parts)
