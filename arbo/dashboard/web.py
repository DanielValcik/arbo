"""Web dashboard for Arbo trading system monitoring.

FastAPI app serving a single-page dashboard with real-time API endpoints.
Integrated into the orchestrator as an internal task (same pattern as Slack bot).
"""

from __future__ import annotations

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


# Strategy metadata (RDH architecture)
_STRATEGY_META: dict[str, dict[str, str]] = {
    "A": {"name": "Theta Decay", "category": "Longshots", "description": "Sell optimism premium on longshot YES contracts"},
    "B": {"name": "Reflexivity Surfer", "category": "Trending", "description": "Ride reflexive momentum in trending markets"},
    "C": {"name": "Compound Weather", "category": "Weather", "description": "Weather temperature ladder trades"},
}


def _infer_category(question: str | None, details: dict[str, Any] | None = None) -> str:
    """Infer market category from signal details or market question."""
    if details:
        cat = details.get("category", "")
        if cat:
            return cat.capitalize()
    if not question:
        return "Other"
    q = question.lower()
    if any(kw in q for kw in ("btc", "bitcoin", "eth", "ethereum", "sol", "solana", "crypto")):
        return "Crypto"
    if any(kw in q for kw in ("election", "trump", "biden", "senate", "congress", "president")):
        return "Politics"
    if any(
        kw in q
        for kw in (
            "win on",
            "beat",
            "draw",
            "premier league",
            "la liga",
            "bundesliga",
            "serie a",
            "soccer",
            "football",
        )
    ):
        return "Soccer"
    return "Other"


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
    """Portfolio summary: balance, P&L, snapshot history, per-category breakdown."""
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
    category_pnl: dict[str, dict[str, Any]] = {}

    try:
        from arbo.utils.db import (
            Market,
            PaperPosition,
            PaperSnapshot,
            PaperTrade,
            get_session_factory,
        )

        factory = get_session_factory()
        async with factory() as session:
            # Daily P&L = change in total portfolio value since midnight UTC
            today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
            result = await session.execute(
                sa.select(PaperSnapshot.total_value)
                .where(PaperSnapshot.snapshot_at <= today)
                .order_by(PaperSnapshot.snapshot_at.desc())
                .limit(1)
            )
            day_start_value = _dec(result.scalar())
            if day_start_value is not None:
                daily_pnl = round((total_value or 0.0) - day_start_value, 2)
            else:
                # No snapshot before today — use initial capital
                daily_pnl = round((total_value or 0.0) - (initial or 0.0), 2)

            # Weekly P&L = change in total portfolio value since Monday 00:00 UTC
            week_start = today - timedelta(days=today.weekday())
            result = await session.execute(
                sa.select(PaperSnapshot.total_value)
                .where(PaperSnapshot.snapshot_at <= week_start)
                .order_by(PaperSnapshot.snapshot_at.desc())
                .limit(1)
            )
            week_start_value = _dec(result.scalar())
            if week_start_value is not None:
                weekly_pnl = round((total_value or 0.0) - week_start_value, 2)
            else:
                weekly_pnl = round((total_value or 0.0) - (initial or 0.0), 2)

            # Per-category P&L: join trades with markets to get category
            # Exclude pre-validation trades from dashboard metrics
            result = await session.execute(
                sa.select(
                    Market.category,
                    sa.func.count(PaperTrade.id),
                    sa.func.coalesce(sa.func.sum(PaperTrade.actual_pnl), 0),
                    sa.func.coalesce(
                        sa.func.sum(
                            sa.case(
                                (PaperTrade.status == "open", PaperTrade.size),
                                else_=sa.literal(0),
                            )
                        ),
                        0,
                    ),
                )
                .outerjoin(Market, PaperTrade.market_condition_id == Market.condition_id)
                .where(
                    sa.or_(
                        PaperTrade.notes.is_(None),
                        PaperTrade.notes != "pre-validation",
                    )
                )
                .group_by(Market.category)
            )
            for row in result.all():
                cat = (row[0] or "other").capitalize()
                category_pnl[cat] = {
                    "trades": row[1],
                    "realized_pnl": _dec(row[2]) or 0.0,
                    "invested": _dec(row[3]) or 0.0,
                }

            # Open positions per category
            result = await session.execute(
                sa.select(
                    Market.category,
                    sa.func.count(PaperPosition.id),
                    sa.func.coalesce(sa.func.sum(PaperPosition.size), 0),
                    sa.func.coalesce(sa.func.sum(PaperPosition.unrealized_pnl), 0),
                )
                .outerjoin(Market, PaperPosition.market_condition_id == Market.condition_id)
                .group_by(Market.category)
            )
            for row in result.all():
                cat = (row[0] or "other").capitalize()
                if cat not in category_pnl:
                    category_pnl[cat] = {"trades": 0, "realized_pnl": 0.0, "invested": 0.0}
                category_pnl[cat]["open_positions"] = row[1]
                category_pnl[cat]["invested_open"] = _dec(row[2]) or 0.0
                category_pnl[cat]["unrealized_pnl"] = _dec(row[3]) or 0.0

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

    # Per-strategy P&L from risk manager
    strategy_pnl: dict[str, dict[str, Any]] = {}
    if orch._risk_manager:
        rm = orch._risk_manager
        for sid, meta in _STRATEGY_META.items():
            ss = rm.get_strategy_state(sid)
            if ss:
                strategy_pnl[sid] = {
                    "name": meta["name"],
                    "allocated": _dec(ss.allocated),
                    "deployed": _dec(ss.deployed),
                    "available": _dec(ss.available),
                    "weekly_pnl": _dec(ss.weekly_pnl),
                    "total_pnl": _dec(ss.total_pnl),
                    "positions": ss.position_count,
                    "is_halted": ss.is_halted,
                }

    return {
        "balance": balance,
        "total_value": total_value,
        "initial_capital": initial,
        "daily_pnl": daily_pnl,
        "weekly_pnl": weekly_pnl,
        "total_pnl": round(total_pnl, 2),
        "roi_pct": round(roi_pct, 2),
        "snapshots": snapshots,
        "category_pnl": category_pnl,
        "strategy_pnl": strategy_pnl,
    }


@app.get("/api/positions")
async def api_positions(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Open positions with market names and category."""
    orch = state.orchestrator
    if orch is None:
        return {"positions": []}

    positions: list[dict[str, Any]] = []
    try:
        from arbo.utils.db import Market, PaperPosition, get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                sa.select(PaperPosition, Market.question, Market.category)
                .outerjoin(Market, PaperPosition.market_condition_id == Market.condition_id)
                .order_by(PaperPosition.opened_at.desc())
            )
            for row in result.all():
                pos = row[0]
                question = row[1] or pos.market_condition_id[:20]
                category = (row[2] or "other").capitalize()
                if category == "Other":
                    category = _infer_category(question)
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
                        "category": category,
                        "opened_at": pos.opened_at.isoformat() if pos.opened_at else None,
                    }
                )
    except Exception as e:
        logger.warning("positions_query_error", error=str(e))

    return {"positions": positions}


@app.get("/api/signals")
async def api_signals(
    _user: str = Depends(_verify_credentials),
    strategy: str | None = None,
) -> dict[str, Any]:
    """Last 100 tradeable signals (edge ≤ 12%) with market names and category.

    Signals with edge > 12% are excluded (model error / line mismatch).
    By default only RDH signals are shown (layer=0). Pass strategy=A/B/C to
    filter further, or strategy=all to include legacy signals.
    """
    signals: list[dict[str, Any]] = []
    max_edge = Decimal("0.12")
    try:
        from arbo.utils.db import Market, get_session_factory
        from arbo.utils.db import Signal as DBSignal

        factory = get_session_factory()
        async with factory() as session:
            query = (
                sa.select(DBSignal, Market.question, Market.category)
                .outerjoin(Market, DBSignal.market_condition_id == Market.condition_id)
                .where(DBSignal.edge <= max_edge)
            )

            # Strategy filter: default = RDH only (layer 0)
            if strategy and strategy.upper() in ("A", "B", "C"):
                # Filter to specific strategy via JSONB details->>'strategy'
                query = query.where(
                    DBSignal.details["strategy"].astext == strategy.upper()
                )
            elif strategy and strategy.lower() == "all":
                pass  # Show everything including legacy
            else:
                # Default: only RDH signals (layer=0)
                query = query.where(DBSignal.layer == 0)

            query = query.order_by(DBSignal.detected_at.desc()).limit(100)
            result = await session.execute(query)
            for row in result.all():
                sig = row[0]
                question = row[1] or sig.market_condition_id[:20]
                details = sig.details or {}
                reason = details.get("reason", details.get("match_type", ""))
                # Category from signal details (most accurate) or market table
                category = details.get("category", row[2] or "other")
                if isinstance(category, str):
                    category = category.capitalize()
                sig_strategy = details.get("strategy", "")
                signals.append(
                    {
                        "timestamp": sig.detected_at.isoformat() if sig.detected_at else None,
                        "layer": sig.layer,
                        "strategy": sig_strategy,
                        "market": question,
                        "direction": sig.direction,
                        "edge": _dec(sig.edge),
                        "confidence": _dec(sig.confidence),
                        "confluence_score": sig.confluence_score,
                        "category": category,
                        "reason": str(reason)[:100] if reason else "",
                    }
                )
    except Exception as e:
        logger.warning("signals_query_error", error=str(e))

    return {"signals": signals}


@app.get("/api/layers")
async def api_layers(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Per-task status (RDH: _tasks, legacy: _layers)."""
    orch = state.orchestrator
    if orch is None:
        return {"layers": []}

    now = time.monotonic()
    layers: list[dict[str, Any]] = []

    # Use _tasks (RDH) or _layers (legacy)
    task_dict: dict[str, Any] = getattr(orch, "_tasks", None) or getattr(
        orch, "_layers", {}
    )

    # Strategy name mapping for RDH tasks
    task_labels: dict[str, str] = {
        "discovery": "Discovery",
        "strategy_A": "Strategy A — Theta Decay",
        "strategy_B": "Strategy B — Reflexivity Surfer",
        "strategy_C": "Strategy C — Compound Weather",
    }

    for name, ts in task_dict.items():
        heartbeat_ago = now - ts.last_heartbeat
        task_running = ts.task is not None and not ts.task.done()

        # Determine status color
        if ts.permanent_stop or (ts.task and ts.task.done()) or heartbeat_ago > 300:
            status_color = "red"
        elif ts.restart_count > 0:
            status_color = "yellow"
        else:
            status_color = "green"

        layers.append(
            {
                "name": name,
                "label": task_labels.get(name, name),
                "status": status_color,
                "running": task_running,
                "enabled": ts.enabled,
                "permanent_stop": ts.permanent_stop,
                "heartbeat_ago_s": int(heartbeat_ago),
                "restart_count": ts.restart_count,
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
    """Infrastructure: uptime, system resources, strategy task count."""
    orch = state.orchestrator
    uptime_s = int(time.monotonic() - orch._start_time) if orch and orch._start_time else 0

    # System resources
    process = psutil.Process()
    mem_info = process.memory_info()
    cpu_pct = process.cpu_percent(interval=0)

    # Odds API quota (available on legacy orchestrator, optional on RDH)
    odds_quota: dict[str, Any] = {"remaining": "N/A", "total": "N/A"}
    odds_client = getattr(orch, "_odds_client", None) if orch else None
    if odds_client is not None:
        remaining = getattr(odds_client, "_remaining_quota", None)
        if remaining is not None:
            odds_quota = {"remaining": remaining, "total": 20000}

    # Active strategy/task count
    task_dict = getattr(orch, "_tasks", {}) if orch else {}
    active_tasks = sum(1 for ts in task_dict.values() if ts.enabled and not ts.permanent_stop)

    return {
        "uptime_s": uptime_s,
        "uptime_human": _format_uptime(uptime_s),
        "memory_mb": round(mem_info.rss / 1024 / 1024, 1),
        "cpu_pct": round(cpu_pct, 1),
        "system_memory_pct": round(psutil.virtual_memory().percent, 1),
        "odds_api_quota": odds_quota,
        "active_tasks": active_tasks,
        "mode": getattr(orch, "_mode", "unknown") if orch else "unknown",
    }


@app.get("/api/trades")
async def api_trades(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Last 50 trades with market names and category."""
    trades: list[dict[str, Any]] = []
    try:
        from arbo.utils.db import Market, PaperTrade, get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            # Exclude pre-validation trades from trade log
            result = await session.execute(
                sa.select(PaperTrade, Market.question, Market.category)
                .outerjoin(Market, PaperTrade.market_condition_id == Market.condition_id)
                .where(
                    sa.or_(
                        PaperTrade.notes.is_(None),
                        PaperTrade.notes != "pre-validation",
                    )
                )
                .order_by(PaperTrade.placed_at.desc())
                .limit(50)
            )
            for row in result.all():
                trade = row[0]
                question = row[1] or trade.market_condition_id[:20]
                category = (row[2] or "other").capitalize()
                if category == "Other":
                    category = _infer_category(question)
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
                        "category": category,
                        "placed_at": trade.placed_at.isoformat() if trade.placed_at else None,
                        "resolved_at": (
                            trade.resolved_at.isoformat() if trade.resolved_at else None
                        ),
                    }
                )
    except Exception as e:
        logger.warning("trades_query_error", error=str(e))

    return {"trades": trades}


@app.get("/api/strategies")
async def api_strategies(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Per-strategy status: allocation, P&L, positions, halt state."""
    orch = state.orchestrator
    if orch is None:
        return {"strategies": []}

    rm = orch._risk_manager
    strategies: list[dict[str, Any]] = []

    for sid, meta in _STRATEGY_META.items():
        ss = rm.get_strategy_state(sid) if rm else None
        entry: dict[str, Any] = {
            "id": sid,
            "name": meta["name"],
            "category": meta["category"],
            "description": meta["description"],
        }
        if ss:
            entry.update(
                {
                    "allocated": _dec(ss.allocated),
                    "deployed": _dec(ss.deployed),
                    "available": _dec(ss.available),
                    "weekly_pnl": _dec(ss.weekly_pnl),
                    "total_pnl": _dec(ss.total_pnl),
                    "positions": ss.position_count,
                    "is_halted": ss.is_halted,
                }
            )
        else:
            entry.update(
                {
                    "allocated": 0.0,
                    "deployed": 0.0,
                    "available": 0.0,
                    "weekly_pnl": 0.0,
                    "total_pnl": 0.0,
                    "positions": 0,
                    "is_halted": False,
                }
            )
        strategies.append(entry)

    return {"strategies": strategies}


@app.get("/api/capital")
async def api_capital(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Capital allocation overview across strategies + reserve."""
    orch = state.orchestrator
    if orch is None:
        return {"error": "Orchestrator not available"}

    rm = orch._risk_manager
    total_capital = _dec(orch._capital) or 0.0

    allocations: list[dict[str, Any]] = []
    total_deployed = 0.0
    total_available = 0.0

    if rm:
        for sid, meta in _STRATEGY_META.items():
            ss = rm.get_strategy_state(sid)
            if ss:
                deployed = _dec(ss.deployed) or 0.0
                available = _dec(ss.available) or 0.0
                allocations.append(
                    {
                        "strategy": sid,
                        "name": meta["name"],
                        "allocated": _dec(ss.allocated),
                        "deployed": deployed,
                        "available": available,
                        "utilization_pct": round(
                            deployed / float(ss.allocated) * 100, 1
                        )
                        if ss.allocated
                        else 0.0,
                        "is_halted": ss.is_halted,
                    }
                )
                total_deployed += deployed
                total_available += available

    # Reserve (10% = $200)
    from arbo.core.risk_manager import RESERVE_CAPITAL

    reserve = _dec(RESERVE_CAPITAL) or 0.0

    return {
        "total_capital": total_capital,
        "total_deployed": round(total_deployed, 2),
        "total_available": round(total_available, 2),
        "reserve": reserve,
        "allocations": allocations,
    }


@app.get("/api/taker-flow")
async def api_taker_flow(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Recent taker flow snapshots from DB (last 24h)."""
    snapshots: list[dict[str, Any]] = []
    try:
        from arbo.utils.db import TakerFlowSnapshot, get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            cutoff = datetime.now(UTC) - timedelta(hours=24)
            result = await session.execute(
                sa.select(TakerFlowSnapshot)
                .where(TakerFlowSnapshot.captured_at > cutoff)
                .order_by(TakerFlowSnapshot.captured_at.desc())
                .limit(200)
            )
            for row in result.scalars().all():
                snapshots.append(
                    {
                        "market_condition_id": row.market_condition_id,
                        "yes_volume": _dec(row.yes_taker_volume),
                        "no_volume": _dec(row.no_taker_volume),
                        "yes_ratio": _dec(row.yes_no_ratio),
                        "z_score": _dec(row.z_score),
                        "is_peak": row.is_peak_optimism,
                        "captured_at": row.captured_at.isoformat() if row.captured_at else None,
                    }
                )
    except Exception as e:
        logger.warning("taker_flow_query_error", error=str(e))

    return {"snapshots": snapshots}


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


def create_app(orchestrator: Any) -> FastAPI:
    """Factory function for RDH orchestrator integration.

    Called by main_rdh.py to inject the orchestrator into dashboard state.
    """
    state.orchestrator = orchestrator
    return app
