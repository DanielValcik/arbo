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
# CryptoArb state reader (file-based bridge)
# ---------------------------------------------------------------------------

CRYPTOARB_STATE_PATH = os.getenv(
    "CRYPTOARB_STATE_PATH", "/opt/cryptoarb/production_data/state.json"
)


class CryptoArbReader:
    """Reads CryptoArb state.json with caching and stale detection."""

    CACHE_TTL = 10  # seconds
    STALE_THRESHOLD = 7200  # 2 hours

    def __init__(self, path: str = CRYPTOARB_STATE_PATH):
        self.path = path
        self._cache: dict[str, Any] | None = None
        self._cache_time: float = 0.0

    def read(self) -> dict[str, Any] | None:
        """Read state.json with 10s cache. Returns None if file missing."""
        now = time.time()
        if self._cache is not None and (now - self._cache_time) < self.CACHE_TTL:
            return self._cache
        try:
            import json as _json

            with open(self.path) as f:
                data = _json.load(f)
            self._cache = data
            self._cache_time = now
            return data
        except FileNotFoundError:
            return None
        except Exception:
            # JSON parse error — return cached value if available
            return self._cache

    def is_stale(self) -> bool:
        """Check if CryptoArb data is stale.

        Uses last_daily_update from state: stale if it's more than 1 day old.
        Falls back to file mtime (24h threshold) if field is missing.
        """
        data = self.read()
        if data is None:
            return True
        last_daily = data.get("last_daily_update", "")
        if last_daily:
            try:
                last_date = datetime.strptime(last_daily, "%Y-%m-%d").date()
                today = datetime.now(UTC).date()
                return (today - last_date).days > 1
            except ValueError:
                pass
        # Fallback: file mtime > 24h
        try:
            mtime = os.path.getmtime(self.path)
            return (time.time() - mtime) > 86400
        except OSError:
            return True


_cryptoarb = CryptoArbReader()


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
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "mode": mode,
        "nightcap_api_url": os.environ.get("NIGHTCAP_API_URL", ""),
        "nightcap_api_key": os.environ.get("NIGHTCAP_API_KEY", ""),
    })


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

    # Query DB for snapshots and per-category breakdown
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

            # Reconstruct corrected chart from cumulative realized P&L at each snapshot time.
            # Paper engine balance drifts after restarts; use trade-level P&L instead.
            if snapshots:
                import bisect

                result = await session.execute(
                    sa.select(PaperTrade.resolved_at, PaperTrade.actual_pnl)
                    .where(PaperTrade.status.in_(["won", "lost"]))
                    .where(PaperTrade.resolved_at.isnot(None))
                    .where(
                        sa.or_(
                            PaperTrade.notes.is_(None),
                            PaperTrade.notes != "pre-validation",
                        )
                    )
                    .order_by(PaperTrade.resolved_at)
                )
                trades_pnl = [
                    (row.resolved_at.isoformat(), _dec(row.actual_pnl) or 0.0)
                    for row in result.all()
                ]

                # Build cumulative P&L series from trades
                trade_times = [t[0] for t in trades_pnl]
                cumulative = []
                running = 0.0
                for _, pnl in trades_pnl:
                    running += pnl
                    cumulative.append(running)

                # For each snapshot, find cumulative realized P&L at that time
                init = initial or 0.0
                for s in snapshots:
                    ts = s["timestamp"] or ""
                    idx = bisect.bisect_right(trade_times, ts) - 1
                    realized_at_time = cumulative[idx] if idx >= 0 else 0.0
                    s["total_value"] = round(
                        init + realized_at_time + (s["unrealized_pnl"] or 0.0), 2
                    )
    except Exception as e:
        logger.warning("portfolio_query_error", error=str(e))

    # Per-strategy P&L from risk manager (reliable source of truth)
    strategy_pnl: dict[str, dict[str, Any]] = {}
    realized_total_pnl = 0.0
    realized_weekly_pnl = 0.0
    if orch._risk_manager:
        rm = orch._risk_manager
        for sid, meta in _STRATEGY_META.items():
            ss = rm.get_strategy_state(sid)
            if ss:
                s_total = _dec(ss.total_pnl) or 0.0
                s_weekly = _dec(ss.weekly_pnl) or 0.0
                realized_total_pnl += s_total
                realized_weekly_pnl += s_weekly
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

    # Unrealized P&L from open positions
    unrealized_pnl = 0.0
    if engine:
        pos_dict = getattr(engine, "_positions", None) or getattr(engine, "positions", {})
        for pos in pos_dict.values():
            unrealized_pnl += _dec(getattr(pos, "unrealized_pnl", 0)) or 0.0

    # Use strategy-level P&L (reliable) instead of snapshot-based (drifts after restarts)
    total_pnl = realized_total_pnl + unrealized_pnl
    corrected_total_value = (initial or 0.0) + total_pnl
    roi_pct = (total_pnl / initial * 100) if initial else 0.0

    # Per-strategy daily + monthly P&L from resolved trades
    monthly_pnl = 0.0
    realized_daily_pnl = 0.0
    try:
        from arbo.utils.db import PaperTrade, get_session_factory as _gsf

        factory = _gsf()
        async with factory() as session:
            now_utc = datetime.now(UTC)
            today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
            month_start = now_utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

            # Per-strategy daily + monthly in one query
            result = await session.execute(
                sa.select(
                    PaperTrade.strategy,
                    sa.func.coalesce(
                        sa.func.sum(
                            sa.case(
                                (PaperTrade.resolved_at >= today_start, PaperTrade.actual_pnl),
                                else_=sa.literal(0),
                            )
                        ),
                        0,
                    ).label("daily"),
                    sa.func.coalesce(
                        sa.func.sum(
                            sa.case(
                                (PaperTrade.resolved_at >= month_start, PaperTrade.actual_pnl),
                                else_=sa.literal(0),
                            )
                        ),
                        0,
                    ).label("monthly"),
                )
                .where(PaperTrade.status.in_(["won", "lost"]))
                .where(
                    sa.or_(
                        PaperTrade.notes.is_(None),
                        PaperTrade.notes != "pre-validation",
                    )
                )
                .group_by(PaperTrade.strategy)
            )
            for row in result.all():
                sid = row[0] or ""
                s_daily = _dec(row[1]) or 0.0
                s_monthly = _dec(row[2]) or 0.0
                realized_daily_pnl += s_daily
                monthly_pnl += s_monthly
                if sid in strategy_pnl:
                    strategy_pnl[sid]["daily_pnl"] = round(s_daily, 2)
                    strategy_pnl[sid]["monthly_pnl"] = round(s_monthly, 2)

            # Ensure all strategies have daily/monthly keys
            for sid in strategy_pnl:
                strategy_pnl[sid].setdefault("daily_pnl", 0.0)
                strategy_pnl[sid].setdefault("monthly_pnl", 0.0)
                # ROI per strategy
                alloc = strategy_pnl[sid].get("allocated", 0) or 1
                strategy_pnl[sid]["roi_pct"] = round(
                    (strategy_pnl[sid].get("total_pnl", 0) or 0) / alloc * 100, 2
                )

    except Exception as e:
        logger.warning("strategy_pnl_query_error", error=str(e))

    return {
        "balance": round(corrected_total_value, 2),
        "total_value": round(corrected_total_value, 2),
        "initial_capital": initial,
        "daily_pnl": round(realized_daily_pnl, 2),
        "weekly_pnl": round(realized_weekly_pnl, 2),
        "monthly_pnl": round(monthly_pnl, 2),
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
                sa.select(
                    PaperPosition, Market.question, Market.category, Market.end_date
                )
                .outerjoin(Market, PaperPosition.market_condition_id == Market.condition_id)
                .order_by(PaperPosition.opened_at.desc())
            )
            now = datetime.now(UTC)
            for row in result.all():
                pos = row[0]
                question = row[1] or pos.market_condition_id[:20]
                category = (row[2] or "other").capitalize()
                end_date = row[3]
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
                        "strategy": getattr(pos, "strategy", "") or "",
                        "category": category,
                        "opened_at": pos.opened_at.isoformat() if pos.opened_at else None,
                        "end_date": end_date.isoformat() if end_date else None,
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
                .where(sa.or_(DBSignal.edge <= max_edge, DBSignal.edge.is_(None)))
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
    rs = rm._state if hasattr(rm, "_state") else None
    daily_pnl = _dec(rs.daily_pnl if rs else Decimal("0")) or 0.0
    weekly_pnl = _dec(rs.weekly_pnl if rs else Decimal("0")) or 0.0
    category_exposure: dict[str, float] = {}
    raw_cat = rs.category_exposure if rs else {}
    for cat, val in raw_cat.items():
        category_exposure[cat] = _dec(val) or 0.0

    # Risk limits (hardcoded constants)
    daily_limit = capital * 0.10
    weekly_limit = capital * 0.20

    # Only show utilization on LOSSES (positive P&L = no risk concern)
    daily_loss = max(0.0, -daily_pnl)
    weekly_loss = max(0.0, -weekly_pnl)
    daily_utilization = daily_loss / daily_limit * 100 if daily_limit else 0.0
    weekly_utilization = weekly_loss / weekly_limit * 100 if weekly_limit else 0.0

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
                        "strategy": getattr(trade, "strategy", "") or "",
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


@app.get("/api/closed-positions")
async def api_closed_positions(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Closed (resolved) positions — won and lost trades."""
    trades: list[dict[str, Any]] = []
    try:
        from arbo.utils.db import Market, PaperTrade, get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                sa.select(PaperTrade, Market.question, Market.category)
                .outerjoin(Market, PaperTrade.market_condition_id == Market.condition_id)
                .where(PaperTrade.status.in_(["won", "lost"]))
                .where(
                    sa.or_(
                        PaperTrade.notes.is_(None),
                        PaperTrade.notes != "pre-validation",
                    )
                )
                .order_by(PaperTrade.resolved_at.desc())
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
                        "strategy": getattr(trade, "strategy", "") or "",
                        "side": trade.side,
                        "price": _dec(trade.price),
                        "size": _dec(trade.size),
                        "status": trade.status,
                        "actual_pnl": _dec(trade.actual_pnl),
                        "category": category,
                        "placed_at": trade.placed_at.isoformat() if trade.placed_at else None,
                        "resolved_at": (
                            trade.resolved_at.isoformat() if trade.resolved_at else None
                        ),
                    }
                )
    except Exception as e:
        logger.warning("closed_positions_query_error", error=str(e))

    return {"trades": trades}


@app.get("/api/daily-pnl")
async def api_daily_pnl(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Daily realized P&L aggregated by date, plus per-strategy daily averages."""
    days: list[dict[str, Any]] = []
    avg_by_strategy: dict[str, float] = {}
    try:
        from arbo.utils.db import PaperTrade, get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            # Daily totals
            result = await session.execute(
                sa.select(
                    sa.func.date(PaperTrade.resolved_at).label("day"),
                    sa.func.sum(PaperTrade.actual_pnl).label("pnl"),
                    sa.func.count().label("trades"),
                )
                .where(PaperTrade.status.in_(["won", "lost"]))
                .where(PaperTrade.resolved_at.isnot(None))
                .group_by(sa.func.date(PaperTrade.resolved_at))
                .order_by(sa.func.date(PaperTrade.resolved_at))
            )
            for row in result.all():
                days.append(
                    {
                        "date": row.day.isoformat() if row.day else None,
                        "pnl": round(float(row.pnl or 0), 2),
                        "trades": row.trades,
                    }
                )

            # Per-strategy: total P&L and number of distinct days
            strat_result = await session.execute(
                sa.select(
                    PaperTrade.strategy,
                    sa.func.sum(PaperTrade.actual_pnl).label("total_pnl"),
                    sa.func.count(sa.func.distinct(sa.func.date(PaperTrade.resolved_at))).label(
                        "num_days"
                    ),
                )
                .where(PaperTrade.status.in_(["won", "lost"]))
                .where(PaperTrade.resolved_at.isnot(None))
                .where(PaperTrade.strategy.isnot(None))
                .group_by(PaperTrade.strategy)
            )
            for row in strat_result.all():
                strat = row.strategy or "?"
                total = float(row.total_pnl or 0)
                num_days = row.num_days or 1
                avg_by_strategy[strat] = round(total / num_days, 2)

    except Exception as e:
        logger.warning("daily_pnl_query_error", error=str(e))

    # Overall daily average
    num_total_days = len(days) or 1
    total_pnl = sum(d["pnl"] for d in days)
    avg_total = round(total_pnl / num_total_days, 2)

    return {
        "days": days,
        "avg_daily": avg_total,
        "avg_by_strategy": avg_by_strategy,
    }


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
        # Add model metadata for Strategy C
        if sid == "C":
            entry["model"] = {
                "name": "AR-0134",
                "score": 170.1,
                "train_trades": 273,
                "train_wr": 43.6,
                "oos_pnl": 297,
                "oos_wr": 38.2,
                "walkforward_pnl": 2218,
                "max_drawdown_pct": 13.0,
                "cities_active": 18,
                "expected_daily_trades": "2-3",
                "min_edge": 0.10,
                "kelly_fraction": 0.25,
                "prob_sharpening": 0.90,
                "excluded_cities": ["Chicago", "Seoul"],
            }

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


@app.get("/api/drift")
async def api_drift(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Price drift analytics from Strategy C latency re-check."""
    orch = state.orchestrator
    if orch is None:
        return {"samples": 0, "avg_drift": 0, "avg_abs_drift": 0, "max_abs_drift": 0,
                "skipped_volatility": 0, "recent": []}

    sc = getattr(orch, "_strategy_c", None)
    if sc is None:
        return {"samples": 0, "avg_drift": 0, "avg_abs_drift": 0, "max_abs_drift": 0,
                "skipped_volatility": 0, "recent": []}

    log = sc.get_drift_log()
    stats = sc.stats
    # Last 50 entries for the chart
    recent = log[-50:] if log else []
    return {
        "samples": stats.get("drift_samples", 0),
        "avg_drift": stats.get("avg_drift", 0),
        "avg_abs_drift": stats.get("avg_abs_drift", 0),
        "max_abs_drift": stats.get("max_abs_drift", 0),
        "skipped_volatility": stats.get("trades_skipped_volatility", 0),
        "recent": recent,
    }


# ---------------------------------------------------------------------------
# CryptoArb API endpoints
# ---------------------------------------------------------------------------


@app.get("/api/cryptoarb/overview")
async def api_cryptoarb_overview(
    _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Portfolio overview + all pair summaries."""
    data = _cryptoarb.read()
    if data is None:
        return {"error": "CryptoArb state not available"}

    # Compute win rate from trade returns
    trade_returns = data.get("trade_returns", [])
    wins = sum(1 for r in trade_returns if r > 0)
    win_rate = (wins / len(trade_returns) * 100) if trade_returns else 0.0

    # Compute drawdown from equity history
    equity_hist = data.get("equity_history", [])
    drawdown = 0.0
    if equity_hist:
        peak = equity_hist[0]
        for v in equity_hist:
            if v > peak:
                peak = v
            dd = (v - peak) / peak if peak > 0 else 0.0
            if dd < drawdown:
                drawdown = dd

    # Compute Sharpe estimate from trade returns
    import math

    sharpe = 0.0
    if len(trade_returns) > 1:
        mean_r = sum(trade_returns) / len(trade_returns)
        var_r = sum((r - mean_r) ** 2 for r in trade_returns) / (len(trade_returns) - 1)
        if var_r > 0:
            sharpe = mean_r / math.sqrt(var_r) * math.sqrt(365)

    pairs = {}
    for name, ps in data.get("pairs", {}).items():
        z_hist = ps.get("z_history", [])
        current_z = z_hist[-1] if z_hist else 0.0
        dz = (z_hist[-1] - z_hist[-4]) if len(z_hist) >= 4 else 0.0
        pos = ps.get("position", 0)
        pairs[name] = {
            "position": "LONG" if pos == 1 else ("SHORT" if pos == -1 else "FLAT"),
            "z": round(current_z, 4),
            "dz_3d": round(dz, 4),
            "allocation": ps.get("allocation", 0.0),
            "trade_pnl": round(ps.get("trade_pnl", 0.0), 6),
            "acf": round(ps.get("acf", 0.0), 3),
            "beta_stability": round(ps.get("beta_stability", 1.0), 3),
            "entry_z": ps.get("entry_z", 0.0),
            "entry_time": ps.get("entry_time", ""),
            "hold_count": ps.get("hold_count", 0),
        }

    return {
        "equity": data.get("portfolio_equity", 1.0),
        "drawdown_pct": round(drawdown * 100, 2),
        "total_trades": data.get("total_trades", 0),
        "win_rate": round(win_rate, 1),
        "sharpe": round(sharpe, 2),
        "last_daily_update": data.get("last_daily_update", ""),
        "last_signal_eval": data.get("last_signal_eval", ""),
        "stale": _cryptoarb.is_stale(),
        "mode": "paper",
        "pairs": pairs,
    }


@app.get("/api/cryptoarb/charts")
async def api_cryptoarb_charts(
    _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Timestamped equity + Z-score series for Chart.js."""
    data = _cryptoarb.read()
    if data is None:
        return {"error": "CryptoArb state not available"}

    # Equity series
    equity_snapshots = data.get("equity_snapshots", [])
    if not equity_snapshots:
        # Fallback: index-based from equity_history
        equity_hist = data.get("equity_history", [])
        equity_snapshots = [{"t": str(i), "v": v} for i, v in enumerate(equity_hist[-365:])]

    # Z-score series per pair
    z_series: dict[str, list[dict]] = {}
    for name, ps in data.get("pairs", {}).items():
        snaps = ps.get("z_snapshots", [])
        if snaps:
            z_series[name] = snaps
        else:
            # Fallback: index-based from z_history
            z_hist = ps.get("z_history", [])
            z_series[name] = [{"t": str(i), "z": v} for i, v in enumerate(z_hist)]

    return {
        "equity": equity_snapshots,
        "z_scores": z_series,
    }


@app.get("/api/cryptoarb/trades")
async def api_cryptoarb_trades(
    _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Trade log (most recent first)."""
    data = _cryptoarb.read()
    if data is None:
        return {"error": "CryptoArb state not available"}

    trade_log = data.get("trade_log", [])
    # Return most recent first
    return {"trades": list(reversed(trade_log))}


@app.get("/api/cryptoarb/alerts")
async def api_cryptoarb_alerts(
    _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Alert history (most recent first)."""
    data = _cryptoarb.read()
    if data is None:
        return {"error": "CryptoArb state not available"}

    alerts = data.get("alerts", [])
    return {"alerts": list(reversed(alerts))}


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


# ---------------------------------------------------------------------------
# Experiments Dashboard
# ---------------------------------------------------------------------------

_EXPERIMENTS_DIR = Path(__file__).parent.parent.parent / "research" / "data" / "experiments"


def _load_latest_sweep() -> dict[str, Any] | None:
    """Load the most recent sweep or autoresearch results JSON."""
    if not _EXPERIMENTS_DIR.exists():
        return None

    # Try autoresearch first (newer), then sweep
    ar_file = _EXPERIMENTS_DIR / "autoresearch_latest.json"
    if ar_file.exists():
        try:
            return _convert_autoresearch(ar_file)
        except Exception:
            pass

    sweep_files = sorted(_EXPERIMENTS_DIR.glob("sweep_*.json"), reverse=True)
    if not sweep_files:
        return None
    try:
        import json as _json
        with open(sweep_files[0]) as f:
            return _json.load(f)
    except Exception:
        return None


def _convert_autoresearch(path: Path) -> dict[str, Any]:
    """Convert autoresearch JSON to sweep-compatible format for the dashboard."""
    import json as _json
    with open(path) as f:
        ar = _json.load(f)

    meta = ar.get("meta", {})

    # Convert history to all_results format (compact — no params/city_results)
    all_results = []
    for h in ar.get("history", []):
        all_results.append({
            "experiment_id": h.get("id", ""),
            "score": h.get("score", 0),
            "trades": h.get("trades", 0),
            "win_rate": h.get("win_rate", 0),
            "total_pnl": h.get("pnl", 0),
            "roi_pct": h.get("roi_pct", 0),
            "max_drawdown_pct": h.get("dd", 0),
            "sharpe": h.get("sharpe", 0),
            "capital_utilization": h.get("util", 0),
            "avg_pnl_per_hour": h.get("pph", 0),
            "total_exits": h.get("exits", 0),
            "exit_saves": h.get("saves", 0),
            "exit_regrets": h.get("regrets", 0),
            "status": h.get("status", ""),
            "description": h.get("description", ""),
        })

    # Top results from enriched data (includes params, city_results, equity_curve, OOS)
    top_enriched = ar.get("top_enriched", [])
    if top_enriched:
        top_results = []
        for h in top_enriched:
            top_results.append({
                "experiment_id": h.get("id", ""),
                "score": h.get("score_with_oos", h.get("score", 0)),
                "score_train": h.get("score", 0),
                "trades": h.get("trades", 0),
                "win_rate": h.get("win_rate", 0),
                "total_pnl": h.get("pnl", 0),
                "roi_pct": h.get("roi_pct", 0),
                "max_drawdown_pct": h.get("dd", 0),
                "sharpe": h.get("sharpe", 0),
                "capital_utilization": h.get("util", 0),
                "avg_pnl_per_hour": h.get("pph", 0),
                "total_exits": h.get("exits", 0),
                "exit_saves": h.get("saves", 0),
                "exit_regrets": h.get("regrets", 0),
                "status": h.get("status", ""),
                "description": h.get("description", ""),
                "params": h.get("params"),
                "city_results": h.get("city_results"),
                "equity_curve": h.get("equity_curve"),
                "oos_pnl": h.get("oos_pnl"),
                "oos_trades": h.get("oos_trades"),
                "oos_win_rate": h.get("oos_win_rate"),
                "oos_roi_pct": h.get("oos_roi_pct"),
                "wf_oos_pnl": h.get("wf_oos_pnl"),
                "wf_score": h.get("wf_score"),
            })
    else:
        # Fallback: derive from all_results (no enrichment)
        sorted_results = sorted(all_results, key=lambda r: -r["score"])
        seen_scores: set[float] = set()
        top_results = []
        for r in sorted_results:
            key = round(r["score"], 1)
            if key not in seen_scores:
                seen_scores.add(key)
                top_results.append(r)
            if len(top_results) >= 20:
                break

    return {
        "meta": {
            "sweep_id": "autoresearch",
            "timestamp": meta.get("timestamp", ""),
            "total_trials": meta.get("total_experiments", len(all_results)),
            "phases": f"keeps={meta.get('keeps', 0)}, discards={meta.get('discards', 0)}",
            "quick_mode": False,
        },
        "top_results": top_results,
        "all_results": all_results,
        "best_params": ar.get("best_params"),
        "city_solo": ar.get("city_solo"),
    }


@app.get("/experiments", response_class=HTMLResponse)
async def experiments_page(
    request: Request, _user: str = Depends(_verify_credentials)
) -> HTMLResponse:
    """Serve the experiments dashboard page."""
    return templates.TemplateResponse("experiments.html", {"request": request})


@app.get("/api/experiments/results")
async def api_experiments_results(
    _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Return latest sweep results for the experiments dashboard."""
    data = _load_latest_sweep()
    if data is None:
        return {"error": "No sweep results found. Run: python3 research/sweep_final.py"}
    return data


@app.get("/api/experiments/sweeps")
async def api_experiments_sweeps(
    _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """List all available sweep result files."""
    if not _EXPERIMENTS_DIR.exists():
        return {"sweeps": []}

    sweeps = []

    # Autoresearch (show first)
    ar_file = _EXPERIMENTS_DIR / "autoresearch_latest.json"
    if ar_file.exists():
        try:
            import json as _json
            with open(ar_file) as fh:
                meta = _json.load(fh).get("meta", {})
            sweeps.append({
                "filename": ar_file.name,
                "sweep_id": "autoresearch",
                "timestamp": meta.get("timestamp", ""),
                "total_trials": meta.get("total_experiments", 0),
                "quick_mode": False,
            })
        except Exception:
            sweeps.append({"filename": ar_file.name, "sweep_id": "autoresearch"})

    # Sweep files
    sweep_files = sorted(_EXPERIMENTS_DIR.glob("sweep_*.json"), reverse=True)
    for f in sweep_files:
        try:
            import json as _json
            with open(f) as fh:
                meta = _json.load(fh).get("meta", {})
            sweeps.append({
                "filename": f.name,
                "sweep_id": meta.get("sweep_id", f.stem),
                "timestamp": meta.get("timestamp", ""),
                "total_trials": meta.get("total_trials", 0),
                "quick_mode": meta.get("quick_mode", False),
            })
        except Exception:
            sweeps.append({"filename": f.name, "sweep_id": f.stem})
    return {"sweeps": sweeps}


@app.get("/api/experiments/sweep/{sweep_id}")
async def api_experiments_sweep_by_id(
    sweep_id: str, _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Load a specific sweep by ID."""
    if not _EXPERIMENTS_DIR.exists():
        return {"error": "No experiments directory"}

    # Autoresearch
    if sweep_id == "autoresearch":
        ar_file = _EXPERIMENTS_DIR / "autoresearch_latest.json"
        if ar_file.exists():
            try:
                return _convert_autoresearch(ar_file)
            except Exception as e:
                return {"error": f"Failed to load autoresearch: {e}"}
        return {"error": "Autoresearch results not found"}

    # Sweep files
    for f in _EXPERIMENTS_DIR.glob("sweep_*.json"):
        if sweep_id in f.stem:
            try:
                import json as _json
                with open(f) as fh:
                    return _json.load(fh)
            except Exception:
                return {"error": f"Failed to load {f.name}"}
    return {"error": f"Sweep {sweep_id} not found"}


@app.get("/api/paper-trades")
async def api_paper_trades(
    strategy: str = "C",
    status: str = "won,lost",
    _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Export paper trades for experiment validation."""
    orch = state.orchestrator
    if orch is None:
        return {"error": "Orchestrator not available"}

    statuses = [s.strip() for s in status.split(",")]

    try:
        async with orch._db_engine.begin() as conn:
            rows = await conn.execute(
                sa.text(
                    "SELECT id, market_condition_id, token_id, side, price, "
                    "size, edge_at_exec, confluence_score, kelly_fraction, "
                    "status, actual_pnl, fee_paid, placed_at, resolved_at, "
                    "strategy, notes "
                    "FROM paper_trades "
                    "WHERE strategy = :strategy AND status = ANY(:statuses) "
                    "ORDER BY placed_at"
                ),
                {"strategy": strategy, "statuses": statuses},
            )
            trades = []
            for row in rows:
                trades.append({
                    "id": row.id,
                    "token_id": row.token_id,
                    "side": row.side,
                    "price": float(row.price) if row.price else None,
                    "size": float(row.size) if row.size else None,
                    "edge": float(row.edge_at_exec) if row.edge_at_exec else None,
                    "status": row.status,
                    "pnl": float(row.actual_pnl) if row.actual_pnl else None,
                    "placed_at": row.placed_at.isoformat() if row.placed_at else None,
                    "resolved_at": row.resolved_at.isoformat() if row.resolved_at else None,
                    "strategy": row.strategy,
                })
            return {"trades": trades, "count": len(trades)}
    except Exception as e:
        return {"error": str(e), "trades": [], "count": 0}


# ---------------------------------------------------------------------------
# Health Check, Expected vs Reality, Seasonality, City Volumes
# ---------------------------------------------------------------------------


@app.get("/api/health-checks")
async def api_health_checks(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Last 14 health check results for timeline display."""
    try:
        from arbo.utils.db import HealthCheck, get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                sa.select(
                    HealthCheck.id,
                    HealthCheck.check_at,
                    HealthCheck.verdict,
                    HealthCheck.window_hours,
                    HealthCheck.metrics,
                    HealthCheck.notes,
                )
                .order_by(HealthCheck.check_at.desc())
                .limit(14)
            )
            checks = []
            for row in result.all():
                checks.append({
                    "id": row[0],
                    "check_at": row[1].isoformat() if row[1] else None,
                    "verdict": row[2],
                    "window_hours": row[3],
                    "metrics": row[4],
                    "notes": row[5],
                })
            return {"checks": list(reversed(checks))}
    except Exception as e:
        return {"checks": [], "error": str(e)}


@app.get("/api/expected-vs-reality")
async def api_expected_vs_reality(
    _user: str = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Compare backtest expectations to actual paper trading performance."""
    try:
        from arbo.core.health_check import get_expected_vs_reality

        return await get_expected_vs_reality()
    except Exception as e:
        return {"error": str(e), "too_early": True, "comparison": [], "actual": {}}


@app.get("/api/seasonality")
async def api_seasonality(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Seasonality analysis for Strategy C."""
    try:
        from arbo.core.health_check import get_seasonality_analysis

        return await get_seasonality_analysis()
    except Exception as e:
        return {"error": str(e), "monthly_actual": []}


@app.get("/api/city-volumes")
async def api_city_volumes(_user: str = Depends(_verify_credentials)) -> dict[str, Any]:
    """Per-city volume stats: current, 7d average, max position size."""
    try:
        from arbo.utils.db import CityVolumeDaily, get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            # Last 7 days of volume data per city
            seven_days_ago = datetime.now(UTC) - timedelta(days=7)
            result = await session.execute(
                sa.select(
                    CityVolumeDaily.city,
                    CityVolumeDaily.date,
                    CityVolumeDaily.volume_24h,
                    CityVolumeDaily.liquidity,
                    CityVolumeDaily.num_markets,
                    CityVolumeDaily.avg_price,
                )
                .where(CityVolumeDaily.date >= seven_days_ago.date())
                .order_by(CityVolumeDaily.city, CityVolumeDaily.date)
            )

            from collections import defaultdict

            city_data: dict[str, list[dict]] = defaultdict(list)
            for row in result.all():
                city_data[row[0]].append({
                    "date": row[1].isoformat() if row[1] else None,
                    "volume_24h": _dec(row[2]),
                    "liquidity": _dec(row[3]),
                    "num_markets": row[4],
                    "avg_price": row[5],
                })

            # Compute summary per city
            cities: list[dict] = []
            for city_name, days in sorted(city_data.items()):
                volumes = [d["volume_24h"] for d in days if d["volume_24h"]]
                avg_vol = sum(volumes) / len(volumes) if volumes else 0
                latest = days[-1] if days else {}
                today_vol = latest.get("volume_24h", 0) or 0

                # Max position = 5% of avg 24h volume
                max_position = round(avg_vol * 0.05, 2) if avg_vol > 0 else 0

                # Trend: compare today vs 7d average
                trend = "stable"
                if avg_vol > 0 and today_vol > 0:
                    ratio = today_vol / avg_vol
                    if ratio > 1.3:
                        trend = "up"
                    elif ratio < 0.7:
                        trend = "down"

                cities.append({
                    "city": city_name,
                    "today_volume": round(today_vol, 2),
                    "avg_7d_volume": round(avg_vol, 2),
                    "today_liquidity": latest.get("liquidity", 0),
                    "num_markets": latest.get("num_markets", 0),
                    "max_position": max_position,
                    "trend": trend,
                    "days": days,
                })

            return {"cities": cities}
    except Exception as e:
        return {"cities": [], "error": str(e)}


def create_app(orchestrator: Any) -> FastAPI:
    """Factory function for RDH orchestrator integration.

    Called by main_rdh.py to inject the orchestrator into dashboard state.
    """
    state.orchestrator = orchestrator
    return app
